# %% [markdown]
# # 🚀 AI Internship — Beginner-Level Tasks
#
# **Author:** Poonam  
# **Date:** 2026-06-03  
#
# This notebook covers all **four beginner tasks** required for the AI internship:
#
# | # | Task | Description |
# |---|------|-------------|
# | 1 | **Python Setup Verification** | Print Python version, key packages, and system info |
# | 2 | **Data Cleaning** | Load a dataset, introduce NaNs, handle missing values & type conversions |
# | 3 | **Exploratory Data Analysis (EDA)** | Summary statistics + 4 publication-quality plots |
# | 4 | **Linear Regression** | Train a model with scikit-learn, evaluate with MAE & RMSE |
#
# All datasets come from **scikit-learn built-in datasets** so no downloads are needed.

# %% [markdown]
# ---
# ## Task 1 — Python Setup Verification
#
# Before doing any data science work we confirm that the environment is
# correctly set up by printing the Python version and the versions of every
# library we will use.

# %%
# ─── Standard-library imports ───────────────────────────────────────────
import sys
import platform
import os
from datetime import datetime

# ─── Third-party imports ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# scikit-learn specific imports we'll need later
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
# ─── Print environment information ──────────────────────────────────────
print("=" * 65)
print("  🐍  PYTHON ENVIRONMENT VERIFICATION")
print("=" * 65)
print(f"  Python version  : {sys.version}")
print(f"  Platform        : {platform.platform()}")
print(f"  Machine         : {platform.machine()}")
print(f"  Processor       : {platform.processor()}")
print(f"  Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 65)

# Dictionary of packages to check
packages = {
    "NumPy": np.__version__,
    "Pandas": pd.__version__,
    "Matplotlib": matplotlib.__version__,
    "Seaborn": sns.__version__,
    "Scikit-learn": sklearn.__version__,
}

print("  📦  Installed Package Versions:")
for name, version in packages.items():
    print(f"       {name:15s} → {version}")
print("=" * 65)
print("\n✅ All packages imported successfully. Environment is ready!\n")

# %% [markdown]
# ---
# ## Task 2 — Data Loading & Cleaning
#
# For this task we:
# 1. Load the **California Housing** dataset from scikit-learn.
# 2. Convert it to a Pandas DataFrame.
# 3. **Intentionally introduce missing values** (NaN) to simulate a "dirty" dataset.
# 4. Demonstrate cleaning steps: detection, imputation, type conversion, and
#    duplicate removal.

# %%
# ─── 2.1  Load the California Housing dataset ───────────────────────────
housing_bunch = fetch_california_housing(as_frame=True)

# The `as_frame=True` option returns a Bunch with a `.frame` attribute that
# is already a nicely formatted DataFrame with both features and target.
df_raw = housing_bunch.frame.copy()

print("📋  Raw dataset shape:", df_raw.shape)
print("\nFirst 5 rows:")
print(df_raw.head())

# %%
# ─── 2.2  Inspect data types and basic info ─────────────────────────────
print("\n📊  Data types:\n")
print(df_raw.dtypes)
print(f"\nMemory usage: {df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")

# %%
# ─── 2.3  Introduce artificial missing values ───────────────────────────
# We set a random seed for reproducibility, then randomly replace ~5 % of
# values in selected columns with NaN.

np.random.seed(42)
df_dirty = df_raw.copy()

# Columns where we'll inject NaNs and the fraction to remove
nan_config = {
    "MedInc": 0.05,       # 5 % missing
    "HouseAge": 0.03,     # 3 % missing
    "AveRooms": 0.04,     # 4 % missing
    "Population": 0.02,   # 2 % missing
}

for col, frac in nan_config.items():
    mask = np.random.rand(len(df_dirty)) < frac
    df_dirty.loc[mask, col] = np.nan

# Also inject a few duplicate rows to demonstrate duplicate removal
duplicates = df_dirty.sample(n=50, random_state=42)
df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)

print("🔧  Dirty dataset shape (after adding NaNs + duplicates):", df_dirty.shape)
print("\nMissing values per column:")
print(df_dirty.isnull().sum())
print(f"\nDuplicate rows: {df_dirty.duplicated().sum()}")

# %%
# ─── 2.4  Data Cleaning Steps ───────────────────────────────────────────

# STEP A — Remove duplicate rows
df_clean = df_dirty.drop_duplicates()
print(f"✂️  After removing duplicates: {df_clean.shape[0]} rows "
      f"(removed {df_dirty.shape[0] - df_clean.shape[0]})")

# STEP B — Handle missing values using MEDIAN imputation
#   Median is more robust to outliers than mean for numerical features.
for col in nan_config.keys():
    median_val = df_clean[col].median()
    n_missing = df_clean[col].isnull().sum()
    df_clean[col] = df_clean[col].fillna(median_val)
    print(f"   🩹 Filled {n_missing:>4d} NaNs in '{col}' with median = {median_val:.4f}")

# STEP C — Verify no missing values remain
assert df_clean.isnull().sum().sum() == 0, "There are still missing values!"
print(f"\n✅ Missing values after cleaning: {df_clean.isnull().sum().sum()}")

# STEP D — Type conversions (demonstration)
#   HouseAge is logically an integer (years), so let's convert it.
df_clean["HouseAge"] = df_clean["HouseAge"].astype(int)
print(f"\n🔄 Converted 'HouseAge' to {df_clean['HouseAge'].dtype}")

# STEP E — Reset the index after cleaning
df_clean = df_clean.reset_index(drop=True)
print(f"\n📋  Clean dataset shape: {df_clean.shape}")
print(df_clean.head())

# %% [markdown]
# ---
# ## Task 3 — Exploratory Data Analysis (EDA)
#
# We will produce:
# 1. **Summary statistics** for both datasets (Iris & California Housing).
# 2. **Four plots**, each saved as a PNG to `../outputs/`:
#    - **Histogram** — distribution of a numerical feature
#    - **Box plot** — spread & outliers across categories
#    - **Scatter plot** — relationship between two variables
#    - **Bar chart** — mean values per category

# %%
# ─── 3.0  Setup: output directory & plot style ──────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📂  Plots will be saved to: {os.path.abspath(OUTPUT_DIR)}")

# Use a clean, modern plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# %%
# ─── 3.1  Load the Iris dataset (for categorical EDA) ───────────────────
iris_bunch = load_iris(as_frame=True)
df_iris = iris_bunch.frame.copy()

# Rename the target column for clarity
df_iris.rename(columns={"target": "species_id"}, inplace=True)

# Map integer labels to species names
species_map = dict(enumerate(iris_bunch.target_names))
df_iris["species"] = df_iris["species_id"].map(species_map)

print("🌸  Iris dataset shape:", df_iris.shape)
print(df_iris.head())

# %%
# ─── 3.2  Summary statistics ────────────────────────────────────────────
print("\n" + "=" * 65)
print("  📈  SUMMARY STATISTICS — Iris Dataset")
print("=" * 65)
print(df_iris.describe().round(2))

print("\n" + "=" * 65)
print("  📈  SUMMARY STATISTICS — California Housing (Cleaned)")
print("=" * 65)
print(df_clean.describe().round(2))

# %%
# ─── 3.3  Plot 1: Histogram ─────────────────────────────────────────────
# Distribution of sepal length coloured by species.

fig, ax = plt.subplots(figsize=(8, 5))
for species in df_iris["species"].unique():
    subset = df_iris[df_iris["species"] == species]
    ax.hist(subset["sepal length (cm)"], bins=15, alpha=0.6, label=species, edgecolor="white")

ax.set_title("Distribution of Sepal Length by Species", fontsize=14, fontweight="bold")
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Frequency")
ax.legend(title="Species")
sns.despine()

plot_path = os.path.join(OUTPUT_DIR, "plot1_histogram.png")
fig.savefig(plot_path)
plt.show()
print(f"💾  Saved → {plot_path}")

# %%
# ─── 3.4  Plot 2: Box Plot ──────────────────────────────────────────────
# Petal width distribution across species — great for spotting outliers.

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_iris,
    x="species",
    y="petal width (cm)",
    palette="Set2",
    linewidth=1.2,
    ax=ax,
)
ax.set_title("Petal Width Distribution by Species", fontsize=14, fontweight="bold")
ax.set_xlabel("Species")
ax.set_ylabel("Petal Width (cm)")
sns.despine()

plot_path = os.path.join(OUTPUT_DIR, "plot2_boxplot.png")
fig.savefig(plot_path)
plt.show()
print(f"💾  Saved → {plot_path}")

# %%
# ─── 3.5  Plot 3: Scatter Plot ──────────────────────────────────────────
# Sepal length vs. petal length — reveals clear species clustering.

fig, ax = plt.subplots(figsize=(8, 5))
scatter = sns.scatterplot(
    data=df_iris,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species",
    style="species",
    s=80,
    alpha=0.8,
    ax=ax,
)
ax.set_title("Sepal Length vs. Petal Length", fontsize=14, fontweight="bold")
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Petal Length (cm)")
ax.legend(title="Species", loc="upper left")
sns.despine()

plot_path = os.path.join(OUTPUT_DIR, "plot3_scatter.png")
fig.savefig(plot_path)
plt.show()
print(f"💾  Saved → {plot_path}")

# %%
# ─── 3.6  Plot 4: Bar Chart ─────────────────────────────────────────────
# Mean petal length per species — a simple but effective summary visual.

mean_petal = df_iris.groupby("species")["petal length (cm)"].mean().sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    mean_petal.index,
    mean_petal.values,
    color=sns.color_palette("Set2", n_colors=3),
    edgecolor="white",
    linewidth=1.5,
)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.05,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax.set_title("Mean Petal Length per Species", fontsize=14, fontweight="bold")
ax.set_xlabel("Species")
ax.set_ylabel("Mean Petal Length (cm)")
ax.set_ylim(0, mean_petal.max() * 1.2)
sns.despine()

plot_path = os.path.join(OUTPUT_DIR, "plot4_barchart.png")
fig.savefig(plot_path)
plt.show()
print(f"💾  Saved → {plot_path}")

# %% [markdown]
# ---
# ## Task 4 — Simple Linear Regression
#
# **Objective:** Predict California house prices (`MedHouseVal`) from the
# available features using a simple Linear Regression model.
#
# **Workflow:**
# 1. Prepare features (`X`) and target (`y`).
# 2. Split into training (80 %) and testing (20 %) sets.
# 3. Fit a `LinearRegression` model.
# 4. Predict on the test set.
# 5. Evaluate with **MAE**, **RMSE**, and **R²**.

# %%
# ─── 4.1  Prepare features and target ───────────────────────────────────
# We use the cleaned California Housing DataFrame from Task 2.

feature_cols = [c for c in df_clean.columns if c != "MedHouseVal"]
X = df_clean[feature_cols]
y = df_clean["MedHouseVal"]

print("Feature matrix shape :", X.shape)
print("Target vector shape  :", y.shape)
print(f"\nFeatures used ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i}. {col}")

# %%
# ─── 4.2  Train / Test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 20 % for testing
    random_state=42,       # reproducibility
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

# %%
# ─── 4.3  Train the Linear Regression model ─────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print("🎓  Model trained successfully!")
print(f"\n   Intercept : {model.intercept_:.4f}")
print("\n   Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"      {name:15s} → {coef:+.6f}")

# %%
# ─── 4.4  Make predictions on the test set ───────────────────────────────
y_pred = model.predict(X_test)

# %%
# ─── 4.5  Evaluate the model ────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 65)
print("  📊  LINEAR REGRESSION — EVALUATION METRICS")
print("=" * 65)
print(f"   Mean Absolute Error  (MAE)  : {mae:.4f}")
print(f"   Mean Squared Error   (MSE)  : {mse:.4f}")
print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"   R² Score                     : {r2:.4f}")
print("=" * 65)

# Interpretation
print("\n📝  Interpretation:")
print(f"   • On average, our predictions are off by ~${mae * 100_000:,.0f} "
      f"(MAE × $100k scale).")
print(f"   • The model explains {r2 * 100:.1f}% of the variance in house prices.")
if r2 > 0.5:
    print("   • R² > 0.5 indicates a reasonably good fit for a simple linear model.")
else:
    print("   • R² ≤ 0.5 suggests that a more complex model may be needed.")

# %%
# ─── 4.6  Bonus: Actual vs. Predicted scatter plot ──────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="steelblue", label="Predictions")

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

ax.set_title("Actual vs. Predicted House Prices", fontsize=14, fontweight="bold")
ax.set_xlabel("Actual Median House Value ($100k)")
ax.set_ylabel("Predicted Median House Value ($100k)")
ax.legend()
sns.despine()

# Add metrics annotation
textstr = f"MAE  = {mae:.4f}\nRMSE = {rmse:.4f}\nR²   = {r2:.4f}"
props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props, fontfamily="monospace")

plot_path = os.path.join(OUTPUT_DIR, "plot5_regression_results.png")
fig.savefig(plot_path)
plt.show()
print(f"💾  Saved → {plot_path}")

# %% [markdown]
# ---
# ## ✅ Summary
#
# | Task | Status |
# |------|--------|
# | 1. Python Setup Verification | ✅ Complete |
# | 2. Data Cleaning | ✅ Complete |
# | 3. EDA with 4 Plots | ✅ Complete |
# | 4. Linear Regression | ✅ Complete |
#
# All plots have been saved to the `../outputs/` directory.
# The linear regression model achieves reasonable performance on the
# California Housing dataset using only basic features and no
# feature engineering.

# %%
print("\n" + "=" * 65)
print("  🎉  ALL BEGINNER TASKS COMPLETED SUCCESSFULLY!")
print("=" * 65)
print(f"\n  📂  Plots saved in: {os.path.abspath(OUTPUT_DIR)}")
print("  📄  Plots generated:")
print("       1. plot1_histogram.png")
print("       2. plot2_boxplot.png")
print("       3. plot3_scatter.png")
print("       4. plot4_barchart.png")
print("       5. plot5_regression_results.png  (bonus)")
print("\n  👋  Thank you for reviewing!")
