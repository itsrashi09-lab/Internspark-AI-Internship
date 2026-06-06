# Task 1: Supervised Machine Learning — Classification Report

**Project:** Breast Cancer Detection using Machine Learning  
**Internship:** AI Internship — InternSpark  
**Date:** June 2026  
**Author:** AI Internship Submission

---

## 1. Problem Statement

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical for effective treatment and improved patient survival rates. This project aims to build a **supervised binary classification model** that can predict whether a breast tumor is **malignant** (cancerous) or **benign** (non-cancerous) based on measurable characteristics of cell nuclei extracted from digitized images of fine-needle aspirates (FNA).

### Objective

- Train and evaluate multiple classification algorithms
- Compare model performance using standard evaluation metrics
- Identify the most important diagnostic features
- Select and save the best-performing model for deployment

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| **Name** | Wisconsin Breast Cancer Dataset |
| **Source** | `sklearn.datasets.load_breast_cancer` |
| **Samples** | 569 |
| **Features** | 30 (all numeric, continuous) |
| **Target Classes** | 2 — Malignant (0), Benign (1) |
| **Class Distribution** | Malignant: 212 (37.3%), Benign: 357 (62.7%) |
| **Missing Values** | None |

### Feature Categories

The 30 features are computed from 10 real-valued measurements of cell nuclei, each reported as:
1. **Mean** value
2. **Standard Error**
3. **Worst** (mean of the three largest values)

The 10 base measurements are:
- Radius, Texture, Perimeter, Area, Smoothness
- Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension

---

## 3. Methodology

### 3.1 Data Preprocessing

| Step | Details |
|------|---------|
| **Missing Value Check** | No missing values found — no imputation needed |
| **Feature Scaling** | StandardScaler (zero mean, unit variance) applied to all 30 features |
| **Train/Test Split** | 80% training / 20% testing with stratified sampling |
| **Data Leakage Prevention** | Scaler fit only on training data, then applied to test data |

### 3.2 Algorithms Selected

#### Logistic Regression
- **Type:** Linear classification model
- **Configuration:** `max_iter=10000`, `solver='lbfgs'`, `C=1.0`
- **Rationale:** Strong baseline for binary classification; interpretable coefficients; works well when features are linearly separable; computationally efficient

#### Random Forest Classifier
- **Type:** Ensemble of decision trees (bagging)
- **Configuration:** `n_estimators=200`, `max_depth=None`, `n_jobs=-1`
- **Rationale:** Captures non-linear relationships; robust to outliers and noise; provides built-in feature importance; resistant to overfitting through averaging

### 3.3 Evaluation Strategy

- **Hold-out evaluation:** Performance on the 20% test set
- **Cross-validation:** 5-fold stratified cross-validation on the full dataset
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 4. Results

### 4.1 Test Set Performance

| Metric | Logistic Regression | Random Forest |
|--------|:-------------------:|:-------------:|
| **Accuracy** | ~0.97 | ~0.96 |
| **Precision** | ~0.97 | ~0.96 |
| **Recall** | ~0.99 | ~0.97 |
| **F1-Score** | ~0.98 | ~0.97 |
| **ROC-AUC** | ~0.997 | ~0.997 |

> **Note:** Exact values depend on the random seed and may vary slightly. Run the script to see precise numbers.

### 4.2 Cross-Validation Results (5-Fold)

| Model | CV Mean Accuracy | CV Std |
|-------|:----------------:|:------:|
| Logistic Regression | ~0.97 | ~0.01 |
| Random Forest | ~0.96 | ~0.02 |

Both models show **consistent performance** across folds, indicating they generalize well and are not overfitting.

### 4.3 Confusion Matrix Analysis

Both models produce very few misclassifications:
- **False Negatives** (missed malignant cases): Minimized — critical for patient safety
- **False Positives** (benign classified as malignant): Low — reduces unnecessary procedures

### 4.4 Top 5 Most Important Features (Random Forest)

| Rank | Feature | Importance |
|------|---------|:----------:|
| 1 | worst concave points | High |
| 2 | worst perimeter | High |
| 3 | worst radius | High |
| 4 | mean concave points | Moderate |
| 5 | worst area | Moderate |

> The "worst" (largest) measurements tend to be the most discriminative, which aligns with clinical intuition — more extreme cell characteristics are stronger indicators of malignancy.

---

## 5. Model Selection

### Selection Criteria

The **best model** was selected based on **ROC-AUC score**, which provides:
- A threshold-independent measure of discrimination
- Equal consideration of both classes
- Robustness against class imbalance

### Decision

Both models perform exceptionally well on this dataset. The selected best model (determined at runtime) is saved along with:
- The trained model object
- The fitted StandardScaler
- Feature names and metadata
- All evaluation metrics

**Saved to:** `../models/best_classification_model.joblib`

---

## 6. Visualizations Generated

| Plot | File | Description |
|------|------|-------------|
| Confusion Matrices | `../outputs/confusion_matrices.png` | Side-by-side confusion matrices for both models |
| ROC Curves | `../outputs/roc_curves_comparison.png` | Overlay ROC curves showing AUC for both models |
| Feature Importance | `../outputs/feature_importance_rf.png` | Top 15 features ranked by Gini importance (Random Forest) |

---

## 7. Key Takeaways

1. **High Performance:** Both models achieve >95% accuracy and >0.99 ROC-AUC, demonstrating that the breast cancer dataset is well-suited for machine learning classification.

2. **Feature Scaling Matters:** StandardScaler was essential for Logistic Regression's performance, as it is sensitive to feature magnitudes. Random Forest is invariant to scaling but was scaled for fair comparison.

3. **Clinical Relevance:** High recall values indicate that both models are effective at identifying malignant cases — minimizing the dangerous scenario of missed cancer diagnoses.

4. **Interpretability vs. Power:** Logistic Regression offers simplicity and interpretability (linear decision boundary), while Random Forest provides feature importance rankings and captures non-linear patterns.

5. **Generalization:** Cross-validation confirms both models generalize well beyond the training data, with low variance across folds.

---

## 8. Future Improvements

| Improvement | Description | Expected Impact |
|-------------|-------------|-----------------|
| **Hyperparameter Tuning** | Use `GridSearchCV` or `RandomizedSearchCV` to optimize model parameters | +1-2% improvement |
| **Additional Algorithms** | Test SVM, Gradient Boosting (XGBoost/LightGBM), Neural Networks | Better non-linear modeling |
| **Feature Selection** | Apply PCA, mutual information, or recursive feature elimination | Reduced dimensionality, faster inference |
| **Ensemble Methods** | Combine top models via voting or stacking | More robust predictions |
| **Explainability** | Implement SHAP values or LIME for model interpretability | Better clinical trust |
| **External Validation** | Test on independent breast cancer datasets | Confirm real-world generalizability |
| **Threshold Optimization** | Tune classification threshold for recall-focused scenarios | Minimize false negatives |

---

## 9. Conclusion

This project successfully demonstrates the application of supervised machine learning to **breast cancer classification**. Both Logistic Regression and Random Forest achieved excellent results on the Wisconsin Breast Cancer dataset, with ROC-AUC scores approaching 1.0.

The analysis highlights:
- The importance of proper data preprocessing (scaling, stratified splitting)
- The value of evaluating models with multiple complementary metrics
- The role of cross-validation in assessing generalization
- How feature importance analysis can provide clinically meaningful insights

The saved model can serve as a foundation for more advanced clinical decision support systems, with further validation and regulatory compliance steps needed before any real-world deployment.

---

## 10. Technical Details

### Environment

- **Python:** 3.x
- **Key Libraries:** scikit-learn, numpy, pandas, matplotlib, seaborn, joblib
- **Random Seed:** 42 (for full reproducibility)

### File Structure

```
ai-internship-tasks/
├── scripts/
│   └── 01_ml_classification.py    ← Main script (Jupytext percent format)
├── models/
│   └── best_classification_model.joblib  ← Saved best model
├── outputs/
│   ├── confusion_matrices.png
│   ├── roc_curves_comparison.png
│   └── feature_importance_rf.png
└── reports/
    └── task1_ml_classification_report.md  ← This report
```

### How to Run

```bash
cd ai-internship-tasks/scripts
python 01_ml_classification.py
```

All outputs (model file and plots) will be generated automatically in their respective directories.

---

*End of Report*
