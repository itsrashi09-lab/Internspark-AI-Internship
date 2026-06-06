# 🤖 Artificial Intelligence Internship — InternSpark

> Hands-on AI internship tasks covering machine learning, deep learning, NLP, data analysis, and responsible AI practices.

---

## 📁 Project Structure

```
ai-internship-tasks/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── 00_beginner_tasks.py          # Beginner level tasks (setup, cleaning, EDA, regression)
│   ├── 01_ml_classification.py       # Task 1: ML Classification Project
│   ├── 02_deep_learning_text.py      # Task 2: Deep Learning — Sentiment Analysis
│   └── 04_responsible_ai.py          # Task 4: Responsible AI — SHAP/LIME Analysis
├── reports/
│   ├── beginner_tasks_report.md      # Beginner tasks summary
│   ├── task1_ml_classification_report.md
│   ├── task2_deep_learning_report.md
│   └── task4_responsible_ai_report.md
├── models/                           # Saved model files
│   ├── best_classification_model.joblib
│   └── sentiment_model/
├── outputs/                          # Generated plots and visualizations
│   ├── histogram.png
│   ├── boxplot.png
│   ├── scatter_plot.png
│   ├── bar_chart.png
│   ├── confusion_matrix_*.png
│   ├── roc_curves.png
│   ├── training_curves.png
│   ├── shap_summary.png
│   └── ...
└── notebooks/                        # (Generated) Jupyter notebooks
```

---

## 🛠️ Environment Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone <your-github-repo-url>
cd ai-internship-tasks
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import sklearn; import torch; import transformers; import shap; import lime; print('All packages installed successfully!')"
```

---

## 🚀 Running the Scripts

Each script can be run directly as a Python file or opened as a notebook in VS Code / Jupyter.

### Option A: Run as Python Scripts

```bash
# Beginner Tasks
python scripts/00_beginner_tasks.py

# Task 1: ML Classification
python scripts/01_ml_classification.py

# Task 2: Deep Learning (may take longer — uses GPU if available)
python scripts/02_deep_learning_text.py

# Task 4: Responsible AI
python scripts/04_responsible_ai.py
```

### Option B: Open as Jupyter Notebooks

The scripts use `# %%` cell markers (Jupytext percent format). You can:

1. **VS Code**: Open any `.py` file → Click "Run Cell" buttons that appear
2. **Convert to .ipynb**:
   ```bash
   pip install jupytext
   jupytext --to notebook scripts/00_beginner_tasks.py
   jupytext --to notebook scripts/01_ml_classification.py
   jupytext --to notebook scripts/02_deep_learning_text.py
   jupytext --to notebook scripts/04_responsible_ai.py
   ```
3. **Open in Jupyter**:
   ```bash
   jupyter notebook
   ```

---

## 📋 Task Summary

### Beginner Level Tasks (All Completed)
| # | Task | Description |
|---|------|-------------|
| 1 | Setup Verification | Python environment, virtual env, Jupyter |
| 2 | Data Cleaning | Load CSV, handle missing values, type conversions |
| 3 | Exploratory Data Analysis | Summary statistics + 4 plot types |
| 4 | Linear Regression | Sklearn regression with MAE/RMSE metrics |

### Main Tasks (3 of 4 Completed)
| # | Task | Description | Status |
|---|------|-------------|--------|
| 1 | ML Classification | Breast cancer detection with Logistic Regression vs Random Forest | ✅ |
| 2 | Deep Learning — Text | IMDB sentiment analysis with DistilBERT (transfer learning) | ✅ |
| 3 | Model Deployment | Flask/FastAPI + Docker | ❌ (Not selected) |
| 4 | Responsible AI | SHAP/LIME explainability + bias analysis | ✅ |

---

## 📊 Key Results

### Task 1: ML Classification
- **Dataset**: Breast Cancer Wisconsin (569 samples, 30 features)
- **Best Model**: Random Forest (typically ~96-97% accuracy)
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC all reported
- **Cross-validation**: 5-fold CV performed

### Task 2: Deep Learning
- **Dataset**: IMDB Movie Reviews (subset: 5000 train, 1000 test)
- **Model**: Fine-tuned DistilBERT (distilbert-base-uncased)
- **Approach**: Transfer learning with 2-3 epochs fine-tuning
- **Output**: Saved model + inference example

### Task 4: Responsible AI
- **Explainability**: SHAP (global + local) and LIME explanations
- **Bias Check**: Synthetic demographic groups for fairness analysis
- **Output**: Interpretation plots + mitigation recommendations

---

## 📦 Package Versions

| Package | Version |
|---------|---------|
| Python | 3.9+ |
| scikit-learn | ≥1.3.0 |
| PyTorch | ≥2.0.0 |
| transformers | ≥4.30.0 |
| SHAP | ≥0.42.0 |
| LIME | ≥0.2.0 |

---

## 📝 Reports

Detailed reports for each task are available in the `reports/` directory:
- [Beginner Tasks Report](reports/beginner_tasks_report.md)
- [Task 1: ML Classification Report](reports/task1_ml_classification_report.md)
- [Task 2: Deep Learning Report](reports/task2_deep_learning_report.md)
- [Task 4: Responsible AI Report](reports/task4_responsible_ai_report.md)

---

## 👤 Author

InternSpark AI Internship

---

## 📜 License

This project is submitted as part of the InternSpark AI Internship program.
