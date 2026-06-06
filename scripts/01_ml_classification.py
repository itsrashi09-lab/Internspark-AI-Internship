# %% [markdown]
# # Task 1: Supervised Machine Learning Classification
# ## Breast Cancer Detection using Logistic Regression & Random Forest
#
# **Objective:** Build, evaluate, and compare supervised classification models
# to predict whether a breast tumor is **malignant** or **benign** using the
# Wisconsin Breast Cancer dataset.
#
# **Author:** AI Internship Submission
# **Date:** June 2026
#
# ---
#
# ### Workflow Overview
# 1. Load and explore the dataset
# 2. Preprocess the data (scaling, splitting)
# 3. Train two classifiers: Logistic Regression & Random Forest
# 4. Evaluate using accuracy, precision, recall, F1-score, ROC-AUC
# 5. Perform 5-fold cross-validation
# 6. Visualize results (confusion matrices, ROC curves, feature importance)
# 7. Save the best model

# %% [markdown]
# ## 1. Import Libraries
# We import all necessary libraries upfront for clarity.

# %%
# ── Standard Libraries ──────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd

# ── Scikit-learn: Dataset & Preprocessing ────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Scikit-learn: Models ─────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ── Scikit-learn: Metrics ────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ── Visualization ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# ── Model Persistence ────────────────────────────────────────────────────────────
import joblib

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plot style for professional-looking visualizations
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

print("✅ All libraries imported successfully.")

# %% [markdown]
# ## 2. Setup Output Directories
# We create directories for saving models and output plots.

# %%
# ── Directory Setup ──────────────────────────────────────────────────────────────
# All paths are relative to the script location for portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print(f"📁 Models directory : {os.path.abspath(MODELS_DIR)}")
print(f"📁 Outputs directory: {os.path.abspath(OUTPUTS_DIR)}")

# %% [markdown]
# ## 3. Load and Explore the Dataset
#
# The **Wisconsin Breast Cancer** dataset is a classic benchmark for binary
# classification. It contains **569 samples** with **30 numeric features**
# computed from digitized images of fine-needle aspirates (FNA) of breast masses.
#
# - **Target classes:** 0 = Malignant (212 samples), 1 = Benign (357 samples)
# - **Features:** mean, standard error, and "worst" (largest) values for
#   10 cell-nucleus measurements (radius, texture, perimeter, area, etc.)

# %%
# ── Load Dataset ─────────────────────────────────────────────────────────────────
data = load_breast_cancer()

# Convert to a pandas DataFrame for easier exploration
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"\n📊 Shape: {df.shape[0]} samples × {df.shape[1] - 2} features")
print(f"📋 Target classes: {dict(zip(data.target_names, np.bincount(data.target)))}")
print(f"\n🔢 Feature names ({len(data.feature_names)}):")
for i, name in enumerate(data.feature_names, 1):
    print(f"   {i:2d}. {name}")

# %%
# ── Basic Statistics ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS (first 5 features)")
print("=" * 70)
print(df[data.feature_names[:5]].describe().round(3).to_string())

# %%
# ── Check for Missing Values ─────────────────────────────────────────────────────
missing = df.isnull().sum().sum()
print(f"\n🔍 Missing values in dataset: {missing}")
if missing == 0:
    print("   ✅ No missing values — no imputation needed.")

# %%
# ── Class Distribution ───────────────────────────────────────────────────────────
print("\n📊 Class Distribution:")
class_counts = df["diagnosis"].value_counts()
for cls, count in class_counts.items():
    pct = count / len(df) * 100
    print(f"   {cls:>10s}: {count:4d} ({pct:.1f}%)")

# %% [markdown]
# ## 4. Data Preprocessing
#
# ### Why Preprocessing Matters
# - **Feature Scaling:** Many ML algorithms (especially Logistic Regression)
#   are sensitive to the scale of input features. `StandardScaler` transforms
#   each feature to have zero mean and unit variance.
# - **Train/Test Split:** We hold out 20% of data for unbiased evaluation.
#   Using `stratify` ensures both sets maintain the same class proportions.

# %%
# ── Separate Features and Target ─────────────────────────────────────────────────
X = data.data   # Feature matrix (569 × 30)
y = data.target  # Target vector  (569,)

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape : {y.shape}")

# %%
# ── Train/Test Split (80/20) ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,  # Maintain class proportions in both sets
)

print(f"\n🔀 Train/Test Split (80/20, stratified):")
print(f"   Training set  : {X_train.shape[0]} samples")
print(f"   Test set      : {X_test.shape[0]} samples")
print(f"   Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"   Test  class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# %%
# ── Feature Scaling (StandardScaler) ─────────────────────────────────────────────
# IMPORTANT: Fit the scaler ONLY on training data, then transform both sets.
# This prevents data leakage from the test set.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n⚖️  Feature Scaling Applied (StandardScaler)")
print(f"   Train mean ≈ {X_train_scaled.mean():.6f} (should be ≈ 0)")
print(f"   Train std  ≈ {X_train_scaled.std():.6f}  (should be ≈ 1)")

# %% [markdown]
# ## 5. Model Training
#
# We train two fundamentally different classifiers:
#
# | Model | Type | Strengths |
# |-------|------|-----------|
# | **Logistic Regression** | Linear model | Interpretable, fast, works well when features are linearly separable |
# | **Random Forest** | Ensemble of decision trees | Handles non-linear relationships, robust to outliers, provides feature importance |

# %%
# ── 5a. Logistic Regression ─────────────────────────────────────────────────────
print("=" * 70)
print("TRAINING: Logistic Regression")
print("=" * 70)

lr_model = LogisticRegression(
    max_iter=10000,         # Ensure convergence
    random_state=RANDOM_STATE,
    solver="lbfgs",         # Efficient for small-to-medium datasets
    C=1.0,                  # Regularization strength (default)
)
lr_model.fit(X_train_scaled, y_train)
print("✅ Logistic Regression trained successfully.")

# %%
# ── 5b. Random Forest Classifier ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING: Random Forest Classifier")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,       # Number of trees in the forest
    max_depth=None,         # Trees grow until pure leaves (no limit)
    min_samples_split=2,    # Minimum samples to split a node
    min_samples_leaf=1,     # Minimum samples in a leaf
    random_state=RANDOM_STATE,
    n_jobs=-1,              # Use all CPU cores for parallel training
)
rf_model.fit(X_train_scaled, y_train)
print("✅ Random Forest Classifier trained successfully.")

# %% [markdown]
# ## 6. Model Evaluation
#
# We evaluate each model using multiple metrics to get a comprehensive view:
#
# - **Accuracy:** Overall correctness (can be misleading with imbalanced data)
# - **Precision:** Of predicted positives, how many are truly positive?
# - **Recall (Sensitivity):** Of actual positives, how many were detected?
# - **F1-Score:** Harmonic mean of precision and recall (balanced metric)
# - **ROC-AUC:** Area under the ROC curve (discrimination ability)
#
# > In medical diagnosis, **recall** is especially important — we want to
# > minimize false negatives (missing a malignant tumor is dangerous).

# %%
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained classifier and return a dictionary of metrics.

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True test labels.
    model_name : str
        Human-readable name for display.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Benign)

    # Calculate metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    # Print detailed report
    print(f"\n{'=' * 70}")
    print(f"EVALUATION: {model_name}")
    print(f"{'=' * 70}")
    print(f"\n📊 Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=["Malignant (0)", "Benign (1)"],
        digits=4,
    ))
    print(f"🎯 Summary Metrics:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"   {metric:<12s}: {value:.4f}")

    return metrics, y_pred, y_prob


# ── Evaluate Both Models ─────────────────────────────────────────────────────────
lr_metrics, lr_pred, lr_prob = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
rf_metrics, rf_pred, rf_prob = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

# %% [markdown]
# ## 7. Cross-Validation (5-Fold)
#
# Cross-validation provides a more robust estimate of model performance by
# training and evaluating the model on different subsets of the data.
#
# We use **Stratified K-Fold** to maintain class proportions in each fold.

# %%
print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ── Cross-validation for Logistic Regression ─────────────────────────────────────
lr_cv_scores = cross_val_score(
    LogisticRegression(max_iter=10000, random_state=RANDOM_STATE, solver="lbfgs"),
    scaler.fit_transform(X), y,  # Use ALL data for cross-validation
    cv=cv_strategy,
    scoring="accuracy",
)

print(f"\n📈 Logistic Regression — 5-Fold CV Accuracy:")
print(f"   Fold scores : {lr_cv_scores.round(4)}")
print(f"   Mean ± Std  : {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

# ── Cross-validation for Random Forest ───────────────────────────────────────────
rf_cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    scaler.fit_transform(X), y,
    cv=cv_strategy,
    scoring="accuracy",
)

print(f"\n📈 Random Forest — 5-Fold CV Accuracy:")
print(f"   Fold scores : {rf_cv_scores.round(4)}")
print(f"   Mean ± Std  : {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

# ── Add CV scores to metrics ─────────────────────────────────────────────────────
lr_metrics["CV Accuracy (Mean)"] = lr_cv_scores.mean()
lr_metrics["CV Accuracy (Std)"] = lr_cv_scores.std()
rf_metrics["CV Accuracy (Mean)"] = rf_cv_scores.mean()
rf_metrics["CV Accuracy (Std)"] = rf_cv_scores.std()

# %% [markdown]
# ## 8. Results Comparison Table

# %%
# ── Build Comparison DataFrame ───────────────────────────────────────────────────
results_df = pd.DataFrame([lr_metrics, rf_metrics])
results_df = results_df.set_index("Model")

print("\n" + "=" * 70)
print("MODEL COMPARISON — ALL METRICS")
print("=" * 70)
print(results_df.round(4).to_string())

# %% [markdown]
# ## 9. Visualization
#
# We generate three key visualizations:
# 1. **Confusion Matrices** — Show true vs. predicted labels for each model
# 2. **ROC Curves** — Compare discrimination ability across thresholds
# 3. **Feature Importance** — Identify the most influential features (Random Forest)

# %%
# ── 9a. Confusion Matrices ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, name, color in zip(
    axes,
    [lr_pred, rf_pred],
    ["Logistic Regression", "Random Forest"],
    ["Blues", "Greens"],
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=color,
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax, cbar=False,
        annot_kws={"size": 16, "fontweight": "bold"},
        linewidths=1, linecolor="white",
    )
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

fig.suptitle("Confusion Matrix Comparison", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
plt.savefig(cm_path)
plt.close()
print(f"💾 Confusion matrices saved to: {os.path.abspath(cm_path)}")

# %%
# ── 9b. ROC Curves Comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Logistic Regression ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
ax.plot(lr_fpr, lr_tpr, color="#2563EB", lw=2.5,
        label=f"Logistic Regression (AUC = {lr_metrics['ROC-AUC']:.4f})")

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
ax.plot(rf_fpr, rf_tpr, color="#16A34A", lw=2.5,
        label=f"Random Forest (AUC = {rf_metrics['ROC-AUC']:.4f})")

# Diagonal baseline (random classifier)
ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--",
        label="Random Classifier (AUC = 0.5000)")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

roc_path = os.path.join(OUTPUTS_DIR, "roc_curves_comparison.png")
plt.savefig(roc_path)
plt.close()
print(f"💾 ROC curves saved to: {os.path.abspath(roc_path)}")

# %%
# ── 9c. Feature Importance (Random Forest) ──────────────────────────────────────
importances = rf_model.feature_importances_
feature_names = data.feature_names
indices = np.argsort(importances)[::-1]

# Show top 15 features for readability
top_n = 15
fig, ax = plt.subplots(figsize=(10, 7))

colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
bars = ax.barh(
    range(top_n),
    importances[indices[:top_n]][::-1],
    color=colors,
    edgecolor="white",
    linewidth=0.5,
)
ax.set_yticks(range(top_n))
ax.set_yticklabels(feature_names[indices[:top_n]][::-1], fontsize=10)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=14,
             fontweight="bold", pad=12)
ax.grid(axis="x", alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, importances[indices[:top_n]][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9, color="#374151")

fi_path = os.path.join(OUTPUTS_DIR, "feature_importance_rf.png")
plt.savefig(fi_path)
plt.close()
print(f"💾 Feature importance plot saved to: {os.path.abspath(fi_path)}")

# %%
# ── Print Top 10 Features ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP 10 MOST IMPORTANT FEATURES (Random Forest)")
print("=" * 70)
for rank, idx in enumerate(indices[:10], 1):
    print(f"   {rank:2d}. {feature_names[idx]:<30s}  importance = {importances[idx]:.4f}")

# %% [markdown]
# ## 10. Save the Best Model
#
# We select the model with the **highest ROC-AUC** on the test set as the best
# model, since ROC-AUC provides a threshold-independent measure of
# discriminative performance — critical in medical diagnosis.

# %%
# ── Determine and Save Best Model ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL SELECTION & SAVING")
print("=" * 70)

if lr_metrics["ROC-AUC"] >= rf_metrics["ROC-AUC"]:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_metrics = lr_metrics
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_metrics = rf_metrics

print(f"\n🏆 Best Model: {best_name}")
print(f"   ROC-AUC  : {best_metrics['ROC-AUC']:.4f}")
print(f"   Accuracy : {best_metrics['Accuracy']:.4f}")
print(f"   F1-Score : {best_metrics['F1-Score']:.4f}")

# Save model along with the scaler and metadata
model_path = os.path.join(MODELS_DIR, "best_classification_model.joblib")
model_artifact = {
    "model": best_model,
    "scaler": scaler,
    "model_name": best_name,
    "metrics": best_metrics,
    "feature_names": list(data.feature_names),
    "target_names": list(data.target_names),
    "random_state": RANDOM_STATE,
}
joblib.dump(model_artifact, model_path)
print(f"\n💾 Best model saved to: {os.path.abspath(model_path)}")
print(f"   (Includes: trained model, scaler, metadata, feature names)")

# %% [markdown]
# ## 11. Summary & Conclusion
#
# ### Key Findings
# - Both Logistic Regression and Random Forest achieved **excellent** performance
#   on the breast cancer classification task.
# - The dataset's 30 features provide strong discriminative power between
#   malignant and benign tumors.
# - Cross-validation confirmed that the models generalize well and are not
#   overfitting to the training data.
#
# ### Model Comparison Insights
# - **Logistic Regression** is a simpler, more interpretable model that works
#   exceptionally well when features are linearly separable (as in this dataset).
# - **Random Forest** is an ensemble method that can capture non-linear
#   patterns and provides built-in feature importance rankings.
#
# ### Clinical Relevance
# - In breast cancer screening, **high recall** (sensitivity) is critical to
#   minimize false negatives — we must avoid missing malignant cases.
# - Both models demonstrate strong recall, making them suitable candidates
#   for clinical decision support tools.
#
# ### Future Improvements
# 1. Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
# 2. Trying additional algorithms (SVM, Gradient Boosting, Neural Networks)
# 3. Feature selection to reduce dimensionality
# 4. Testing on external validation datasets
# 5. Implementing model explainability (SHAP values, LIME)

# %%
# ── Final Summary Print ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    TASK 1 — CLASSIFICATION RESULTS                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Dataset     : Wisconsin Breast Cancer (569 samples, 30 features)  ║
║  Split       : 80% Train / 20% Test (stratified)                   ║
║  CV Strategy : 5-Fold Stratified Cross-Validation                  ║
║                                                                    ║
║  ┌──────────────────────┬───────────┬──────────────┐               ║
║  │ Metric               │ Log. Reg. │ Random Forest│               ║
║  ├──────────────────────┼───────────┼──────────────┤               ║
║  │ Accuracy             │ {lr_metrics['Accuracy']:.4f}    │ {rf_metrics['Accuracy']:.4f}       │               ║
║  │ Precision            │ {lr_metrics['Precision']:.4f}    │ {rf_metrics['Precision']:.4f}       │               ║
║  │ Recall               │ {lr_metrics['Recall']:.4f}    │ {rf_metrics['Recall']:.4f}       │               ║
║  │ F1-Score             │ {lr_metrics['F1-Score']:.4f}    │ {rf_metrics['F1-Score']:.4f}       │               ║
║  │ ROC-AUC              │ {lr_metrics['ROC-AUC']:.4f}    │ {rf_metrics['ROC-AUC']:.4f}       │               ║
║  │ CV Accuracy (Mean)   │ {lr_metrics['CV Accuracy (Mean)']:.4f}    │ {rf_metrics['CV Accuracy (Mean)']:.4f}       │               ║
║  └──────────────────────┴───────────┴──────────────┘               ║
║                                                                    ║
║  🏆 Best Model: {best_name:<50s}║
║                                                                    ║
║  📁 Outputs:                                                       ║
║     • Model : ../models/best_classification_model.joblib           ║
║     • Plots : ../outputs/confusion_matrices.png                    ║
║              ../outputs/roc_curves_comparison.png                  ║
║              ../outputs/feature_importance_rf.png                  ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")
print("✅ Task 1 — ML Classification completed successfully!")
