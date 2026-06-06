# %% [markdown]
# # Task 4 — Responsible AI: Fairness, Bias & Explainability
#
# **Objective:** Analyze a trained classification model through the lens of
# *Responsible AI* by examining its fairness, potential biases, and
# interpretability using industry-standard explainability tools.
#
# ## Approach
# 1. Train a Random Forest classifier on the **Breast Cancer Wisconsin** dataset
#    (same dataset & model family used in Task 1 for continuity).
# 2. **Feature Importance** — visualize the model's built-in importances.
# 3. **SHAP (SHapley Additive exPlanations)** — global & local explanations.
# 4. **LIME (Local Interpretable Model-agnostic Explanations)** — local,
#    instance-level explanations for individual predictions.
# 5. **Bias / Fairness Analysis** — create a synthetic sensitive attribute
#    (*age_group*) to demonstrate group-fairness metrics and disparate-impact
#    analysis.
# 6. **Mitigation Recommendations** — practical steps to address identified
#    issues.
#
# > **Why this matters:** Deploying a model without understanding *what* it
# > learned, *why* it predicts certain outcomes, and *whether* it treats all
# > groups equitably can lead to real-world harm. Responsible AI practices are
# > essential for trustworthy machine-learning systems.

# %%
# ──────────────────────────────────────────────────────────────────────
# 0. Imports & Configuration
# ──────────────────────────────────────────────────────────────────────
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import shap
import lime
import lime.lime_tabular

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Use a non-interactive backend so plots save cleanly in headless environments
matplotlib.use("Agg")

# Plot aesthetics
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
sns.set_style("whitegrid")

# Output directory for saved figures
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✅ All imports successful — environment is ready.\n")

# %% [markdown]
# ## 1. Load & Prepare Data
#
# We use scikit-learn's built-in **Breast Cancer Wisconsin (Diagnostic)**
# dataset.  It contains 569 samples with 30 numerical features computed from
# digitised images of fine-needle aspirates (FNA) of breast masses.  The target
# is binary: **malignant (0)** or **benign (1)**.

# %%
# ──────────────────────────────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────────────────────────────
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name="target")

print(f"Dataset shape : {X.shape}")
print(f"Class distribution:\n{y.value_counts().rename({0: 'malignant', 1: 'benign'})}\n")

# Train / Test split — stratified to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
)

print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")

# %% [markdown]
# ## 2. Train a Random Forest Classifier
#
# We train a Random Forest with 200 trees.  This ensemble method is the same
# model family used in Task 1 and works well as the basis for SHAP's
# `TreeExplainer`.

# %%
# ──────────────────────────────────────────────────────────────────────
# 2. Model Training
# ──────────────────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy : {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# %% [markdown]
# ## 3. Feature Importance (Built-in)
#
# Random Forests estimate feature importance via the mean decrease in impurity
# (Gini importance).  While useful as a quick overview, these importances can
# be **biased toward high-cardinality / continuous features** — which is why we
# complement them with SHAP values below.

# %%
# ──────────────────────────────────────────────────────────────────────
# 3. Feature Importance Bar Plot
# ──────────────────────────────────────────────────────────────────────
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

top_n = 15  # Show only the top-15 features for readability
top_features = [cancer.feature_names[i] for i in sorted_idx[:top_n]]
top_importances = importances[sorted_idx[:top_n]]

fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette("viridis", n_colors=top_n)
sns.barplot(x=top_importances, y=top_features, palette=palette, ax=ax)
ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)")
ax.set_title("Top-15 Feature Importances — Random Forest")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "task4_feature_importances.png"))
plt.close(fig)
print("📊 Feature importance plot saved → task4_feature_importances.png")

# %% [markdown]
# ## 4. SHAP — Global & Local Explainability
#
# **SHAP** (SHapley Additive exPlanations) assigns each feature an importance
# value for a *particular prediction*.  It is grounded in cooperative game
# theory (Shapley values), which guarantees desirable properties such as
# **local accuracy**, **missingness**, and **consistency**.
#
# We use `TreeExplainer`, which is optimised for tree-based models and
# computes *exact* Shapley values efficiently.

# %%
# ──────────────────────────────────────────────────────────────────────
# 4a. SHAP TreeExplainer — Compute SHAP values
# ──────────────────────────────────────────────────────────────────────
explainer_shap = shap.TreeExplainer(rf_model)

# Compute SHAP values for the test set.
# For binary classifiers, shap_values is a list of two arrays (one per class).
shap_values = explainer_shap.shap_values(X_test)

# For the summary & dependence plots we focus on the *positive* class (benign = 1).
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]   # SHAP values for class 1 (benign)
elif len(shap_values.shape) == 3:
    shap_values_pos = shap_values[:, :, 1] # 3D array (n_samples, n_features, n_classes)
else:
    shap_values_pos = shap_values      # newer SHAP versions may return a single array

print(f"SHAP values shape : {shap_values_pos.shape}")
print("✅ SHAP values computed successfully.\n")

# %% [markdown]
# ### 4b. SHAP Summary Plot (Beeswarm)
#
# The beeswarm plot shows, for every feature, how each sample's SHAP value
# distributes across the dataset.  Features are sorted by their global
# importance (mean |SHAP|).  Colour encodes the original feature value
# (red = high, blue = low).

# %%
# ──────────────────────────────────────────────────────────────────────
# 4b. SHAP Summary Plot (Beeswarm)
# ──────────────────────────────────────────────────────────────────────
fig_shap_summary = plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_pos,
    X_test,
    feature_names=cancer.feature_names,
    show=False,
    max_display=20,
)
plt.title("SHAP Summary Plot — Feature Impact on Benign Prediction", fontsize=13)
plt.tight_layout()
fig_shap_summary.savefig(os.path.join(OUTPUT_DIR, "task4_shap_summary.png"))
plt.close(fig_shap_summary)
print("📊 SHAP summary plot saved → task4_shap_summary.png")

# %% [markdown]
# ### 4c. SHAP Force Plot (Single Prediction)
#
# A force plot shows how each feature pushes a single prediction away from the
# base value (average model output) toward the final predicted probability.
# Red arrows push toward the positive class; blue arrows push toward the
# negative class.

# %%
# ──────────────────────────────────────────────────────────────────────
# 4c. SHAP Force Plot — Single Instance
# ──────────────────────────────────────────────────────────────────────
sample_idx = 0  # We explain the first test sample
expected_value = (
    explainer_shap.expected_value[1]
    if isinstance(explainer_shap.expected_value, (list, np.ndarray))
    else explainer_shap.expected_value
)

# Generate a matplotlib-based force plot and save it
shap.force_plot(
    expected_value,
    shap_values_pos[sample_idx, :],
    X_test.iloc[sample_idx, :],
    feature_names=list(cancer.feature_names),
    matplotlib=True,
    show=False,
)
plt.title(
    f"SHAP Force Plot — Test Sample #{sample_idx}  "
    f"(True: {cancer.target_names[y_test.iloc[sample_idx]]}, "
    f"Pred: {cancer.target_names[y_pred[sample_idx]]})",
    fontsize=11,
    y=1.05,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task4_shap_force_plot.png"), bbox_inches="tight")
plt.close()
print("📊 SHAP force plot saved → task4_shap_force_plot.png")

# %% [markdown]
# ### 4d. SHAP Dependence Plot — Top Feature
#
# A dependence plot shows the effect of a *single* feature across the dataset.
# Each dot is a sample; the x-axis is the feature value, the y-axis is the
# SHAP value, and colour indicates the strongest interaction feature (chosen
# automatically).

# %%
# ──────────────────────────────────────────────────────────────────────
# 4d. SHAP Dependence Plot — Top Feature
# ──────────────────────────────────────────────────────────────────────
# Identify the feature with the highest mean |SHAP|
mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
top_feature_idx = int(np.argmax(mean_abs_shap))
top_feature_name = cancer.feature_names[top_feature_idx]
print(f"Top feature by mean |SHAP|: {top_feature_name}")

fig_dep = plt.figure(figsize=(10, 6))
shap.dependence_plot(
    top_feature_idx,
    shap_values_pos,
    X_test,
    feature_names=cancer.feature_names,
    show=False,
)
plt.title(f"SHAP Dependence Plot — {top_feature_name}", fontsize=13)
plt.tight_layout()
fig_dep.savefig(os.path.join(OUTPUT_DIR, "task4_shap_dependence.png"))
plt.close(fig_dep)
print("📊 SHAP dependence plot saved → task4_shap_dependence.png")

# %% [markdown]
# ## 5. LIME — Local Interpretable Model-agnostic Explanations
#
# **LIME** explains individual predictions by fitting a simple interpretable
# model (e.g., linear regression) to perturbations of the instance in question.
# Unlike SHAP's `TreeExplainer`, LIME is *model-agnostic* — it works with
# any black-box classifier.
#
# We generate explanations for **three representative test samples**:
# - One correctly classified **malignant** sample
# - One correctly classified **benign** sample
# - One **misclassified** sample (if any)

# %%
# ──────────────────────────────────────────────────────────────────────
# 5. LIME Explanations
# ──────────────────────────────────────────────────────────────────────

# Create the LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(cancer.feature_names),
    class_names=list(cancer.target_names),
    mode="classification",
    random_state=RANDOM_STATE,
)

# --------------- Helper: pick representative samples ---------------
correct_mask = y_pred == y_test.values
malignant_correct = np.where(correct_mask & (y_test.values == 0))[0]
benign_correct = np.where(correct_mask & (y_test.values == 1))[0]
misclassified = np.where(~correct_mask)[0]

samples_to_explain = []
labels = []

if len(malignant_correct) > 0:
    samples_to_explain.append(malignant_correct[0])
    labels.append("Correct Malignant")
if len(benign_correct) > 0:
    samples_to_explain.append(benign_correct[0])
    labels.append("Correct Benign")
if len(misclassified) > 0:
    samples_to_explain.append(misclassified[0])
    labels.append("Misclassified")
else:
    # If no misclassifications, add another benign sample
    if len(benign_correct) > 1:
        samples_to_explain.append(benign_correct[1])
        labels.append("Correct Benign #2")

print(f"Samples selected for LIME explanations: {list(zip(labels, samples_to_explain))}\n")

# --------------- Generate & save LIME explanations ---------------
for i, (idx, label) in enumerate(zip(samples_to_explain, labels)):
    exp = lime_explainer.explain_instance(
        X_test.iloc[idx].values,
        rf_model.predict_proba,
        num_features=10,
        top_labels=1,
    )

    # Print textual explanation
    true_label = cancer.target_names[y_test.iloc[idx]]
    pred_label = cancer.target_names[y_pred[idx]]
    print(f"── LIME Explanation #{i + 1}: {label} ──")
    print(f"   True label : {true_label}")
    print(f"   Predicted  : {pred_label}")

    # Get the explanation for the predicted class
    pred_class = y_pred[idx]
    feature_contributions = exp.as_list(label=pred_class)
    for feat, weight in feature_contributions[:7]:
        direction = "↑" if weight > 0 else "↓"
        print(f"   {direction} {feat:>45s}  →  {weight:+.4f}")
    print()

    # Save the figure produced by LIME
    fig_lime = exp.as_pyplot_figure(label=pred_class)
    fig_lime.suptitle(
        f"LIME — {label} (True: {true_label}, Pred: {pred_label})", fontsize=12
    )
    fig_lime.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"task4_lime_explanation_{i + 1}.png"
    fig_lime.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close(fig_lime)
    print(f"📊 LIME plot saved → {fname}\n")

# %% [markdown]
# ## 6. Bias & Fairness Analysis
#
# The breast-cancer dataset does **not** include sensitive attributes (race,
# gender, age, etc.).  To demonstrate a bias-checking workflow we create a
# **synthetic** `age_group` feature that simulates how a real-world fairness
# audit would be performed.
#
# ### Metrics computed per group
# | Metric | Description |
# |--------|-------------|
# | **Accuracy** | Overall correctness |
# | **Precision** | Of those predicted positive, how many are correct? |
# | **Recall** | Of those truly positive, how many are detected? |
# | **Selection Rate** | Proportion predicted positive in each group |
# | **Disparate Impact** | Ratio of selection rates (min / max) — a ratio < 0.80 is often flagged as potentially unfair (80 % rule) |

# %%
# ──────────────────────────────────────────────────────────────────────
# 6a. Create Synthetic Sensitive Attribute
# ──────────────────────────────────────────────────────────────────────

# Assign each test sample to a random age group to simulate a protected
# attribute.  In a real project you would use actual demographic data.
rng = np.random.RandomState(RANDOM_STATE)
age_groups = rng.choice(["Young (<40)", "Middle (40-60)", "Senior (>60)"], size=len(y_test))

bias_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "age_group": age_groups,
})

print("Synthetic age-group distribution in test set:")
print(bias_df["age_group"].value_counts().to_string())
print()

# %%
# ──────────────────────────────────────────────────────────────────────
# 6b. Per-Group Performance Metrics
# ──────────────────────────────────────────────────────────────────────

group_metrics = []
for group_name, group_df in bias_df.groupby("age_group"):
    yt = group_df["y_true"]
    yp = group_df["y_pred"]
    group_metrics.append({
        "Age Group": group_name,
        "N": len(group_df),
        "Accuracy": accuracy_score(yt, yp),
        "Precision": precision_score(yt, yp, zero_division=0),
        "Recall": recall_score(yt, yp, zero_division=0),
        "F1": f1_score(yt, yp, zero_division=0),
        "Selection Rate": yp.mean(),  # proportion predicted positive (benign)
    })

metrics_df = pd.DataFrame(group_metrics)
print("Per-Group Fairness Metrics:")
print(metrics_df.to_string(index=False, float_format="{:.4f}".format))
print()

# %%
# ──────────────────────────────────────────────────────────────────────
# 6c. Disparate Impact Analysis
# ──────────────────────────────────────────────────────────────────────

selection_rates = metrics_df.set_index("Age Group")["Selection Rate"]
min_rate = selection_rates.min()
max_rate = selection_rates.max()
disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else float("nan")

print(f"Selection Rates by Group:\n{selection_rates.to_string()}\n")
print(f"Disparate Impact Ratio (min / max): {disparate_impact_ratio:.4f}")
if disparate_impact_ratio < 0.80:
    print("⚠️  The 80 % rule is VIOLATED — potential disparate impact detected.")
else:
    print("✅  The 80 % rule is satisfied — no significant disparate impact detected.")
print()

# %%
# ──────────────────────────────────────────────────────────────────────
# 6d. Visualise Bias Metrics
# ──────────────────────────────────────────────────────────────────────

fig_bias, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Accuracy per group ---
sns.barplot(
    data=metrics_df, x="Age Group", y="Accuracy",
    palette="mako", ax=axes[0],
)
axes[0].set_title("Accuracy by Age Group")
axes[0].set_ylim(0, 1.05)
axes[0].bar_label(axes[0].containers[0], fmt="%.3f", padding=3)

# --- Recall per group ---
sns.barplot(
    data=metrics_df, x="Age Group", y="Recall",
    palette="mako", ax=axes[1],
)
axes[1].set_title("Recall (Sensitivity) by Age Group")
axes[1].set_ylim(0, 1.05)
axes[1].bar_label(axes[1].containers[0], fmt="%.3f", padding=3)

# --- Selection rate per group ---
sns.barplot(
    data=metrics_df, x="Age Group", y="Selection Rate",
    palette="mako", ax=axes[2],
)
axes[2].set_title("Selection Rate by Age Group")
axes[2].set_ylim(0, 1.05)
axes[2].bar_label(axes[2].containers[0], fmt="%.3f", padding=3)
axes[2].axhline(y=0.80 * max_rate, ls="--", color="red", label="80 % threshold")
axes[2].legend()

fig_bias.suptitle("Fairness / Bias Metrics Across Synthetic Age Groups", fontsize=14, y=1.02)
plt.tight_layout()
fig_bias.savefig(os.path.join(OUTPUT_DIR, "task4_bias_analysis.png"), bbox_inches="tight")
plt.close(fig_bias)
print("📊 Bias analysis plot saved → task4_bias_analysis.png")

# %% [markdown]
# ## 7. Summary of Findings & Mitigation Recommendations
#
# ### Key Observations
#
# | Area | Finding |
# |------|---------|
# | **Feature Importance** | `worst concave points`, `worst perimeter`, and `mean concave points` dominate both Gini importance and SHAP rankings. |
# | **SHAP** | The beeswarm plot reveals clear monotonic relationships — higher values of cell-shape features push predictions toward malignant. |
# | **LIME** | Local explanations are consistent with SHAP — the same top features appear, increasing trust in the model's reasoning. |
# | **Bias** | With synthetic age groups, metrics are roughly comparable (random assignment), but the workflow demonstrates how real disparities would be detected. |
#
# ### Practical Mitigation Steps
#
# 1. **Data Collection & Auditing**
#    - Collect real demographic metadata (age, ethnicity, etc.) where
#      ethically permissible.
#    - Audit training data for label imbalances across protected groups.
#
# 2. **Pre-processing Mitigations**
#    - Re-sample or re-weight training data so that each group is adequately
#      represented.
#    - Apply *fair representation learning* to transform features into a
#      space where the protected attribute is decorrelated.
#
# 3. **In-processing Mitigations**
#    - Add fairness constraints during training (e.g., adversarial
#      debiasing, exponentiated-gradient reduction).
#    - Regularise for equal opportunity or demographic parity.
#
# 4. **Post-processing Mitigations**
#    - Adjust decision thresholds per group to equalise false-positive /
#      false-negative rates (threshold optimisation).
#    - Apply reject-option classification — defer borderline predictions for
#      human review.
#
# 5. **Monitoring & Governance**
#    - Deploy model-monitoring dashboards that track per-group metrics over
#      time (data/concept drift).
#    - Establish a *model card* documenting intended use, limitations, and
#      fairness evaluations.
#    - Schedule periodic fairness re-audits whenever the model is retrained
#      or the population shifts.

# %%
# ──────────────────────────────────────────────────────────────────────
# 7. Print Final Summary to Console
# ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  TASK 4 — RESPONSIBLE AI: SUMMARY")
print("=" * 70)
print(f"""
Model           : Random Forest (200 trees)
Test Accuracy   : {accuracy:.4f}
Top Feature     : {top_feature_name} (by mean |SHAP|)

Explainability Tools Used:
  • SHAP TreeExplainer — global summary, force plot, dependence plot
  • LIME TabularExplainer — local explanations for {len(samples_to_explain)} samples

Bias Analysis:
  • Synthetic attribute : age_group (Young / Middle / Senior)
  • Disparate Impact    : {disparate_impact_ratio:.4f}
  • 80 %% Rule          : {"PASSED ✅" if disparate_impact_ratio >= 0.80 else "VIOLATED ⚠️"}

Saved Outputs (in ../outputs/):
  1. task4_feature_importances.png
  2. task4_shap_summary.png
  3. task4_shap_force_plot.png
  4. task4_shap_dependence.png
  5. task4_lime_explanation_1.png
  6. task4_lime_explanation_2.png
  7. task4_lime_explanation_3.png
  8. task4_bias_analysis.png
""")
print("=" * 70)
print("✅ Task 4 complete — all plots saved successfully.")
print("=" * 70)
