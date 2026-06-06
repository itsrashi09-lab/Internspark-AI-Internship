# %% [markdown]
# # Task 2: Deep Learning — Text Classification (Sentiment Analysis)
#
# **Objective:** Build a deep learning model for sentiment analysis on movie reviews
# using transfer learning with a pretrained transformer model.
#
# ## Overview
# - **Dataset:** IMDB Movie Review Dataset (50,000 reviews, binary sentiment)
# - **Model:** DistilBERT (a smaller, faster variant of BERT)
# - **Approach:** Transfer Learning — fine-tune a pretrained language model
# - **Framework:** PyTorch + HuggingFace Transformers
#
# ## Why Transfer Learning?
# Training a language model from scratch requires massive datasets and compute.
# Transfer learning lets us leverage a model that has already learned rich language
# representations from billions of words. We only need to fine-tune the final
# classification layer on our specific task, achieving high accuracy with minimal
# training.
#
# ## Pipeline
# 1. Load and subset the IMDB dataset
# 2. Tokenize text using DistilBERT's tokenizer
# 3. Create PyTorch DataLoaders
# 4. Load pretrained DistilBERT and add a classification head
# 5. Fine-tune for 2-3 epochs
# 6. Evaluate with accuracy, precision, recall, F1
# 7. Save the model and run inference on custom text

# %%
# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
import os
import sys
import time
import warnings
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Configuration & Reproducibility
#
# We set all random seeds to ensure reproducible results. This is critical
# for scientific experiments and internship submissions.

# %%
# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Hyperparameters ---
TRAIN_SIZE = 5000       # Number of training samples (subset for speed)
TEST_SIZE = 1000        # Number of test samples
MAX_LENGTH = 256        # Maximum token length for DistilBERT (max is 512)
BATCH_SIZE = 16         # Batch size for training and evaluation
NUM_EPOCHS = 3          # Number of fine-tuning epochs
LEARNING_RATE = 2e-5    # Learning rate (standard for transformer fine-tuning)
WEIGHT_DECAY = 0.01     # L2 regularization to prevent overfitting
WARMUP_RATIO = 0.1      # Fraction of steps for learning rate warmup

# --- Model ---
MODEL_NAME = "distilbert-base-uncased"  # Pretrained model to fine-tune
NUM_LABELS = 2                          # Binary classification (pos/neg)

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "sentiment_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Reproducibility ---
SEED = 42

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# --- Device Detection ---
# CUDA = NVIDIA GPU, MPS = Apple Silicon GPU, else CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  Using CPU (training will be slower)")

print(f"   PyTorch version: {torch.__version__}")
print(f"   Device: {DEVICE}")

# %% [markdown]
# ## 2. Load the IMDB Dataset
#
# The IMDB dataset contains 50,000 movie reviews labeled as positive or negative.
# We use HuggingFace's `datasets` library for easy downloading and processing.
#
# To keep training fast (especially on CPU), we use a random subset:
# - **5,000 training** samples (from 25,000)
# - **1,000 test** samples (from 25,000)

# %%
# ============================================================================
# DATA LOADING
# ============================================================================
print("\n" + "=" * 60)
print("📂 LOADING IMDB DATASET")
print("=" * 60)

# Load the full IMDB dataset from HuggingFace Hub
dataset = load_dataset("stanfordnlp/imdb")

print(f"\nFull dataset structure:")
print(f"  Train: {len(dataset['train']):,} samples")
print(f"  Test:  {len(dataset['test']):,} samples")

# Shuffle and select subsets for faster training
train_dataset = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE))
test_dataset = dataset["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

print(f"\nSubset sizes:")
print(f"  Train: {len(train_dataset):,} samples")
print(f"  Test:  {len(test_dataset):,} samples")

# Display a sample review
sample = train_dataset[0]
print(f"\n--- Sample Review ---")
print(f"Label: {'Positive ✅' if sample['label'] == 1 else 'Negative ❌'}")
print(f"Text (first 300 chars): {sample['text'][:300]}...")

# Check class distribution
train_labels = train_dataset["label"]
print(f"\nClass distribution (train):")
print(f"  Positive: {sum(train_labels):,} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
print(f"  Negative: {len(train_labels) - sum(train_labels):,} ({(len(train_labels) - sum(train_labels))/len(train_labels)*100:.1f}%)")

# %% [markdown]
# ## 3. Tokenization
#
# Transformers like DistilBERT don't work with raw text. They require:
# 1. **Tokenization** — splitting text into subword tokens
# 2. **Encoding** — converting tokens to numerical IDs
# 3. **Attention masks** — indicating which tokens are real vs padding
#
# The `DistilBertTokenizerFast` handles all of this efficiently. We use:
# - `max_length=256` to balance speed and context coverage
# - `truncation=True` to handle long reviews
# - `padding="max_length"` to ensure uniform tensor sizes

# %%
# ============================================================================
# TOKENIZATION
# ============================================================================
print("\n" + "=" * 60)
print("🔤 TOKENIZING TEXT")
print("=" * 60)

# Load the tokenizer for DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

print(f"Tokenizer: {MODEL_NAME}")
print(f"Vocabulary size: {tokenizer.vocab_size:,}")
print(f"Max sequence length: {MAX_LENGTH}")

# Tokenize a sample to see what happens
sample_text = "This movie was absolutely fantastic! I loved every minute."
sample_tokens = tokenizer(sample_text, padding=False, truncation=False)
print(f"\nTokenization example:")
print(f"  Input text:  \"{sample_text}\"")
print(f"  Token IDs:   {sample_tokens['input_ids']}")
print(f"  Decoded:     {tokenizer.convert_ids_to_tokens(sample_tokens['input_ids'])}")


def tokenize_data(examples):
    """
    Tokenize a batch of text examples.

    This function is designed to be used with HuggingFace's dataset.map() method.
    It tokenizes the text, truncates to MAX_LENGTH, and pads shorter sequences.
    """
    return tokenizer(
        examples["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )


# Apply tokenization to both splits (batched for efficiency)
print("\nTokenizing training data...")
train_tokenized = train_dataset.map(tokenize_data, batched=True, batch_size=256)

print("Tokenizing test data...")
test_tokenized = test_dataset.map(tokenize_data, batched=True, batch_size=256)

# Set format to PyTorch tensors — this is crucial for DataLoader compatibility
train_tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)
test_tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)

print("✅ Tokenization complete!")
print(f"   Each sample has: input_ids ({MAX_LENGTH}), attention_mask ({MAX_LENGTH}), label (1)")

# %% [markdown]
# ## 4. PyTorch DataLoaders
#
# DataLoaders handle batching, shuffling, and efficient data loading.
# - **Training**: shuffle=True for stochastic gradient descent
# - **Evaluation**: shuffle=False for consistent evaluation order

# %%
# ============================================================================
# DATALOADERS
# ============================================================================
print("\n" + "=" * 60)
print("📦 CREATING DATALOADERS")
print("=" * 60)

train_loader = DataLoader(
    train_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=True,        # Shuffle training data each epoch
    num_workers=0,       # Use 0 for Windows compatibility
    pin_memory=True if DEVICE.type == "cuda" else False,
)

test_loader = DataLoader(
    test_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=False,       # Don't shuffle test data
    num_workers=0,
    pin_memory=True if DEVICE.type == "cuda" else False,
)

print(f"Training batches: {len(train_loader)} (batch size={BATCH_SIZE})")
print(f"Test batches:     {len(test_loader)} (batch size={BATCH_SIZE})")

# Verify a batch
batch = next(iter(train_loader))
print(f"\nSample batch shapes:")
print(f"  input_ids:      {batch['input_ids'].shape}")
print(f"  attention_mask:  {batch['attention_mask'].shape}")
print(f"  labels:          {batch['label'].shape}")

# %% [markdown]
# ## 5. Model Setup
#
# ### Architecture: DistilBERT for Sequence Classification
#
# **DistilBERT** is a distilled (compressed) version of BERT:
# - 40% smaller than BERT, 60% faster
# - Retains 97% of BERT's language understanding
# - 6 transformer layers (vs 12 in BERT-base)
# - 66M parameters (vs 110M in BERT-base)
#
# `DistilBertForSequenceClassification` adds a classification head on top:
# ```
# [CLS] token embedding → Linear(768 → 768) → ReLU → Dropout → Linear(768 → 2)
# ```
#
# ### Transfer Learning Strategy
# We fine-tune ALL layers of the model. For larger datasets, you could freeze
# earlier layers and only train the classification head.

# %%
# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("\n" + "=" * 60)
print("🤖 LOADING PRETRAINED MODEL")
print("=" * 60)

# Load DistilBERT with a classification head for 2 classes (pos/neg)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

# Move model to device (GPU if available)
model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: {MODEL_NAME}")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1e6:.0f} MB (FP32)")

# Print model architecture summary
print(f"\nModel architecture:")
print(f"  Embedding: WordPiece (vocab={tokenizer.vocab_size:,})")
print(f"  Transformer: 6 layers, 12 heads, hidden=768")
print(f"  Classifier: Linear(768→768) → ReLU → Dropout(0.2) → Linear(768→{NUM_LABELS})")

# %% [markdown]
# ## 6. Optimizer & Learning Rate Scheduler
#
# ### AdamW Optimizer
# AdamW is the standard optimizer for transformer fine-tuning. It decouples
# weight decay from the gradient update, leading to better generalization.
#
# ### Linear Warmup Schedule
# The learning rate starts from 0, linearly increases during warmup, then
# linearly decays. This prevents the model from making large, destructive
# updates in the early steps when gradients are noisy.

# %%
# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================
print("\n" + "=" * 60)
print("⚙️  SETTING UP OPTIMIZER & SCHEDULER")
print("=" * 60)

# Total training steps
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

# AdamW optimizer with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,  # Numerical stability
)

# Linear warmup + decay scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"Scheduler: Linear warmup + decay")
print(f"  Total steps:  {total_steps}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Epochs:       {NUM_EPOCHS}")

# %% [markdown]
# ## 7. Training Loop
#
# The training loop performs the following for each epoch:
# 1. Set model to training mode (`model.train()`)
# 2. Iterate over batches
# 3. Forward pass: compute predictions and loss
# 4. Backward pass: compute gradients
# 5. Gradient clipping: prevent exploding gradients
# 6. Optimizer step: update weights
# 7. Scheduler step: update learning rate
#
# We track loss at each step for visualization.

# %%
# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

# Storage for training metrics
train_losses = []           # Loss at each step
epoch_avg_losses = []       # Average loss per epoch
epoch_accuracies = []       # Training accuracy per epoch

training_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()  # Set model to training mode

    epoch_loss = 0.0
    correct = 0
    total = 0

    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

    for step, batch in enumerate(train_loader):
        # Move batch tensors to device
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # Forward pass
        # The model computes CrossEntropyLoss internally when labels are provided
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss       # Cross-entropy loss
        logits = outputs.logits   # Raw predictions (before softmax)

        # Track metrics
        train_losses.append(loss.item())
        epoch_loss += loss.item()

        # Calculate training accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Backward pass — compute gradients
        loss.backward()

        # Gradient clipping — prevent exploding gradients
        # This is especially important for transformer models
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step — update model weights
        optimizer.step()

        # Scheduler step — update learning rate
        scheduler.step()

        # Zero gradients for next step
        optimizer.zero_grad()

        # Print progress every 50 steps
        if (step + 1) % 50 == 0 or step == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Step {step + 1:>4}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Acc: {correct / total:.4f}"
            )

    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    avg_loss = epoch_loss / len(train_loader)
    epoch_accuracy = correct / total

    epoch_avg_losses.append(avg_loss)
    epoch_accuracies.append(epoch_accuracy)

    print(f"\n  Epoch {epoch + 1} Summary:")
    print(f"    Average Loss: {avg_loss:.4f}")
    print(f"    Accuracy:     {epoch_accuracy:.4f} ({correct}/{total})")
    print(f"    Time:         {epoch_time:.1f}s")

total_time = time.time() - training_start_time
print(f"\n✅ Training complete! Total time: {total_time / 60:.1f} minutes")

# %% [markdown]
# ## 8. Training Curves
#
# Visualizing the training loss helps us understand:
# - **Convergence**: Is the model learning?
# - **Overfitting**: Does the loss plateau or increase?
# - **Learning rate**: Was the schedule appropriate?

# %%
# ============================================================================
# PLOT TRAINING CURVES
# ============================================================================
print("\n" + "=" * 60)
print("📊 PLOTTING TRAINING CURVES")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Step-level Training Loss ---
axes[0].plot(train_losses, color="#2196F3", alpha=0.3, linewidth=0.5, label="Step loss")

# Add smoothed trend line (moving average)
window = min(50, len(train_losses) // 5)
if window > 1:
    smoothed = np.convolve(train_losses, np.ones(window) / window, mode="valid")
    axes[0].plot(
        range(window - 1, len(train_losses)),
        smoothed,
        color="#F44336",
        linewidth=2,
        label=f"Smoothed (window={window})",
    )

axes[0].set_xlabel("Training Step", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Training Loss (Step-level)", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Add epoch boundary markers
steps_per_epoch = len(train_loader)
for e in range(1, NUM_EPOCHS):
    axes[0].axvline(x=e * steps_per_epoch, color="gray", linestyle="--", alpha=0.5)
    axes[0].text(
        e * steps_per_epoch, max(train_losses) * 0.95,
        f"Epoch {e + 1}", fontsize=9, ha="center", color="gray"
    )

# --- Plot 2: Epoch-level Average Loss ---
epochs_range = range(1, NUM_EPOCHS + 1)
axes[1].plot(
    epochs_range, epoch_avg_losses,
    "o-", color="#4CAF50", linewidth=2, markersize=8, label="Avg Loss"
)
for i, (ep, loss_val) in enumerate(zip(epochs_range, epoch_avg_losses)):
    axes[1].annotate(
        f"{loss_val:.4f}",
        (ep, loss_val),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Average Loss", fontsize=12)
axes[1].set_title("Average Training Loss per Epoch", fontsize=14, fontweight="bold")
axes[1].set_xticks(list(epochs_range))
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# --- Plot 3: Training Accuracy per Epoch ---
axes[2].plot(
    epochs_range, epoch_accuracies,
    "s-", color="#FF9800", linewidth=2, markersize=8, label="Accuracy"
)
for i, (ep, acc_val) in enumerate(zip(epochs_range, epoch_accuracies)):
    axes[2].annotate(
        f"{acc_val:.4f}",
        (ep, acc_val),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
axes[2].set_xlabel("Epoch", fontsize=12)
axes[2].set_ylabel("Accuracy", fontsize=12)
axes[2].set_title("Training Accuracy per Epoch", fontsize=14, fontweight="bold")
axes[2].set_xticks(list(epochs_range))
axes[2].set_ylim(0, 1.05)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Training curves saved to: {plot_path}")

# %% [markdown]
# ## 9. Evaluation on Test Set
#
# We evaluate the fine-tuned model on the held-out test set using:
# - **Accuracy**: Fraction of correct predictions
# - **Precision**: Of predicted positives, how many are truly positive?
# - **Recall**: Of actual positives, how many did we find?
# - **F1 Score**: Harmonic mean of precision and recall
#
# We also generate a full classification report and confusion matrix.

# %%
# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("📈 EVALUATING ON TEST SET")
print("=" * 60)

model.eval()  # Set model to evaluation mode (disables dropout)

all_predictions = []
all_labels = []
all_probs = []

eval_start_time = time.time()

# No gradient computation needed during evaluation
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

eval_time = time.time() - eval_start_time

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# --- Compute Metrics ---
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average="binary")
recall = recall_score(all_labels, all_predictions, average="binary")
f1 = f1_score(all_labels, all_predictions, average="binary")

print(f"\n{'=' * 40}")
print(f"  TEST SET RESULTS")
print(f"{'=' * 40}")
print(f"  Accuracy:  {accuracy:.4f}  ({int(accuracy * TEST_SIZE)}/{TEST_SIZE})")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  Eval Time: {eval_time:.1f}s")
print(f"{'=' * 40}")

# --- Full Classification Report ---
print("\nDetailed Classification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=["Negative", "Positive"],
    digits=4,
))

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(f"                  Predicted")
print(f"              Neg      Pos")
print(f"  Actual Neg  {cm[0][0]:<8} {cm[0][1]:<8}")
print(f"  Actual Pos  {cm[1][0]:<8} {cm[1][1]:<8}")

# %% [markdown]
# ## 10. Save Confusion Matrix Plot

# %%
# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 6))

# Create heatmap manually
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax, shrink=0.8)

classes = ["Negative", "Positive"]
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    title="Confusion Matrix",
    ylabel="True Label",
    xlabel="Predicted Label",
)
ax.title.set_fontsize(14)
ax.title.set_fontweight("bold")

# Add text annotations to cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=16, fontweight="bold",
        )

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Confusion matrix saved to: {cm_path}")

# %% [markdown]
# ## 11. Save the Fine-Tuned Model
#
# We save both the model weights and the tokenizer so that the model
# can be loaded later for inference without re-training.

# %%
# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 60)
print("💾 SAVING MODEL")
print("=" * 60)

# Save model and tokenizer using HuggingFace's built-in methods
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# Also save training metadata
import json

metadata = {
    "model_name": MODEL_NAME,
    "dataset": "imdb",
    "train_size": TRAIN_SIZE,
    "test_size": TEST_SIZE,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "training_time_seconds": total_time,
    "device": str(DEVICE),
    "seed": SEED,
}

metadata_path = os.path.join(MODEL_DIR, "training_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved to: {MODEL_DIR}")
print(f"✅ Metadata saved to: {metadata_path}")

# List saved files
saved_files = os.listdir(MODEL_DIR)
print(f"\nSaved files:")
for fname in sorted(saved_files):
    fpath = os.path.join(MODEL_DIR, fname)
    size = os.path.getsize(fpath)
    if size > 1e6:
        print(f"  {fname:.<40} {size / 1e6:.1f} MB")
    else:
        print(f"  {fname:.<40} {size / 1e3:.1f} KB")

# %% [markdown]
# ## 12. Inference on Custom Text
#
# Now let's use the fine-tuned model to predict sentiment on new, unseen text.
# This demonstrates the practical value of our trained model.
#
# The inference pipeline:
# 1. Tokenize the input text
# 2. Pass through the model
# 3. Apply softmax to get probabilities
# 4. Return the predicted label and confidence

# %%
# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_sentiment(
    text: str,
    model=None,
    tokenizer=None,
    device=DEVICE,
    max_length: int = MAX_LENGTH,
):
    """
    Predict sentiment of a given text using the fine-tuned model.

    Args:
        text: The input text to classify.
        model: The fine-tuned DistilBERT model.
        tokenizer: The DistilBERT tokenizer.
        device: The device to run inference on.
        max_length: Maximum token length.

    Returns:
        dict with 'label', 'confidence', 'positive_prob', 'negative_prob'.
    """
    model.eval()  # Ensure evaluation mode

    # Tokenize the input
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Forward pass (no gradient needed)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # Extract results
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()

    return {
        "label": "Positive" if predicted_class == 1 else "Negative",
        "confidence": confidence,
        "positive_prob": probs[0][1].item(),
        "negative_prob": probs[0][0].item(),
    }


# %%
# ============================================================================
# INFERENCE EXAMPLES
# ============================================================================
print("\n" + "=" * 60)
print("🔮 INFERENCE ON CUSTOM TEXT")
print("=" * 60)

# Test with various example reviews
test_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.",
    "Terrible film. Waste of time and money. The script was awful and the acting was wooden.",
    "It was okay, nothing special. Some parts were good but overall pretty average.",
    "A masterpiece of modern cinema. Every frame is beautifully crafted. I was moved to tears.",
    "I fell asleep halfway through. Boring, predictable, and poorly directed.",
    "The visual effects were stunning but the story lacked depth. Mixed feelings overall.",
]

print("\nPredicting sentiment for sample reviews:\n")

for i, review in enumerate(test_reviews, 1):
    result = predict_sentiment(review, model=model, tokenizer=tokenizer)

    # Color-coded emoji based on sentiment
    emoji = "✅" if result["label"] == "Positive" else "❌"
    bar_len = int(result["confidence"] * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)

    print(f"Review {i}: \"{review[:80]}{'...' if len(review) > 80 else ''}\"")
    print(f"  {emoji} {result['label']} (confidence: {result['confidence']:.4f})")
    print(f"  [{bar}] Pos: {result['positive_prob']:.4f} | Neg: {result['negative_prob']:.4f}")
    print()

# %% [markdown]
# ## 13. Loading the Saved Model (For Future Use)
#
# This section shows how to load the saved model for inference later,
# without needing to retrain.

# %%
# ============================================================================
# LOADING SAVED MODEL (DEMONSTRATION)
# ============================================================================
print("\n" + "=" * 60)
print("📂 LOADING SAVED MODEL (DEMONSTRATION)")
print("=" * 60)

# Load the saved model and tokenizer
loaded_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
loaded_tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
loaded_model = loaded_model.to(DEVICE)

# Verify with a test prediction
test_text = "An incredible journey through human emotion. This film is a triumph."
result = predict_sentiment(test_text, model=loaded_model, tokenizer=loaded_tokenizer)

print(f"\nTest prediction with loaded model:")
print(f"  Text: \"{test_text}\"")
print(f"  Prediction: {result['label']} (confidence: {result['confidence']:.4f})")
print(f"\n✅ Model loaded and verified successfully!")

# %% [markdown]
# ## Summary
#
# ### What We Built
# - A **sentiment analysis classifier** using transfer learning
# - Fine-tuned **DistilBERT** on the **IMDB** dataset
# - Achieved strong performance with only **5,000 training samples** and **3 epochs**
#
# ### Key Takeaways
# 1. **Transfer learning** dramatically reduces training data and time requirements
# 2. **DistilBERT** provides an excellent speed/accuracy tradeoff
# 3. **HuggingFace ecosystem** makes transformer-based NLP accessible
# 4. **Proper evaluation** with multiple metrics gives a complete picture of model performance
#
# ### Potential Improvements
# - Train on the full IMDB dataset (25,000 samples) for higher accuracy
# - Use a larger model (BERT-base, RoBERTa) for better performance
# - Implement validation set and early stopping to prevent overfitting
# - Add data augmentation (back-translation, synonym replacement)
# - Experiment with different learning rates and schedulers
# - Use mixed precision training (FP16) for faster GPU training

# %%
print("\n" + "=" * 60)
print("🎉 TASK 2 COMPLETE!")
print("=" * 60)
print(f"\nOutputs saved:")
print(f"  📊 Training curves: {os.path.join(OUTPUT_DIR, 'training_curves.png')}")
print(f"  📊 Confusion matrix: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
print(f"  🤖 Model:           {MODEL_DIR}")
print(f"  📋 Metadata:        {os.path.join(MODEL_DIR, 'training_metadata.json')}")
