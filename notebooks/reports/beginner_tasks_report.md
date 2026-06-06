# 📋 Beginner Tasks — Summary Report

**AI Internship · Beginner Level**  
**Date:** 2026-06-03  

---

## 1. Python Setup Verification

The development environment was verified with all required packages installed:

| Package | Purpose |
|---------|---------|
| **Python 3.x** | Core language |
| **NumPy** | Numerical computation |
| **Pandas** | Data manipulation |
| **Matplotlib** | Plotting |
| **Seaborn** | Statistical visualisation |
| **Scikit-learn** | Machine learning |

> ✅ All packages imported successfully. Environment is ready.

---

## 2. Data Cleaning

**Dataset:** California Housing (scikit-learn built-in, 20,640 samples × 9 columns)

### Steps Performed

1. **Loaded** the dataset as a Pandas DataFrame.
2. **Introduced artificial NaN values** (~2–5 % in `MedInc`, `HouseAge`, `AveRooms`, `Population`) to simulate real-world dirty data.
3. **Added 50 duplicate rows** for demonstration.
4. **Cleaning pipeline:**
   - Removed **50 duplicate** rows with `drop_duplicates()`.
   - Imputed missing values using **median imputation** (robust to outliers).
   - Converted `HouseAge` from float to **int** (type conversion demo).
   - Reset the index.
5. **Post-cleaning:** 0 missing values, 0 duplicates. Dataset ready for analysis.

---

## 3. Exploratory Data Analysis (EDA)

### Summary Statistics

Summary statistics (count, mean, std, min, quartiles, max) were printed for both the **Iris** and **California Housing** datasets.

### Plots

All plots are saved to `../outputs/` relative to the scripts folder.

| # | Plot Type | File | Description |
|---|-----------|------|-------------|
| 1 | **Histogram** | `plot1_histogram.png` | Distribution of sepal length by Iris species |
| 2 | **Box Plot** | `plot2_boxplot.png` | Petal width spread & outliers across Iris species |
| 3 | **Scatter Plot** | `plot3_scatter.png` | Sepal length vs. petal length (species clustering) |
| 4 | **Bar Chart** | `plot4_barchart.png` | Mean petal length per Iris species |

**Key Observations:**
- *Setosa* has distinctly shorter petals and narrower petal width.
- *Virginica* and *versicolor* show some overlap but are mostly separable.
- Sepal length vs. petal length scatter plot reveals clear species clusters.

---

## 4. Linear Regression

### Setup

| Item | Value |
|------|-------|
| **Dataset** | California Housing (cleaned) |
| **Target** | `MedHouseVal` — median house value in $100k |
| **Features** | 8 features (MedInc, HouseAge, AveRooms, etc.) |
| **Split** | 80% train / 20% test |
| **Model** | `sklearn.linear_model.LinearRegression` |

### Results

| Metric | Value |
|--------|-------|
| **MAE** (Mean Absolute Error) | ~0.53 |
| **RMSE** (Root Mean Squared Error) | ~0.74 |
| **R² Score** | ~0.60 |

> **Interpretation:** The model explains approximately 60% of the variance in
> house prices. On average, predictions deviate by ~$53,000 from actual values.
> For a simple linear model with no feature engineering, this is a reasonable
> baseline.

### Bonus Plot

| # | Plot Type | File | Description |
|---|-----------|------|-------------|
| 5 | **Actual vs. Predicted** | `plot5_regression_results.png` | Scatter of true vs. predicted values with perfect-prediction line |

---

## 📁 Output Files

```
ai-internship-tasks/
├── scripts/
│   └── 00_beginner_tasks.py       ← Main script (this notebook)
├── outputs/
│   ├── plot1_histogram.png
│   ├── plot2_boxplot.png
│   ├── plot3_scatter.png
│   ├── plot4_barchart.png
│   └── plot5_regression_results.png
└── reports/
    └── beginner_tasks_report.md   ← This report
```

---

## ✅ Conclusion

All four beginner-level tasks have been completed successfully:

1. ✅ Python environment verified
2. ✅ Data loaded, dirtied, and cleaned
3. ✅ EDA with summary statistics and 4 plots
4. ✅ Linear regression trained and evaluated (MAE, RMSE, R²)

The script is fully self-contained, uses only scikit-learn built-in datasets, and can be executed as a Jupyter notebook in VS Code using the `# %%` cell syntax.
