# Task 4 — Responsible AI: Fairness, Bias & Explainability

> **Course:** Artificial Intelligence Internship  
> **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
> **Model:** Random Forest Classifier (200 estimators)  
> **Tools:** SHAP · LIME · scikit-learn

---

## 1. Introduction

Deploying machine-learning models without scrutinising *what* they learn, *why*
they make particular predictions, and *whether* they treat all population groups
equitably can cause real-world harm — from misdiagnoses in healthcare to
discriminatory lending decisions.

This report accompanies the `04_responsible_ai.py` script, which demonstrates a
three-pillar approach to Responsible AI:

| Pillar | Technique | Scope |
|--------|-----------|-------|
| **Explainability** | SHAP (TreeExplainer) | Global + Local |
| **Explainability** | LIME (TabularExplainer) | Local |
| **Fairness / Bias** | Disparate-impact analysis | Group-level |

---

## 2. Explainability Analysis

### 2.1 Feature Importance (Random Forest Gini)

The Random Forest's built-in Gini importance provides a quick first glance.  The
top-3 features are:

1. `worst concave points`
2. `worst perimeter`
3. `mean concave points`

These relate to the shape and boundary irregularity of cell nuclei — clinically
meaningful indicators of malignancy.

> **Caveat:** Gini importance tends to favour high-cardinality continuous
> features and can be misleading when features are correlated.  SHAP values
> provide a more theoretically grounded alternative.

### 2.2 SHAP — Global Explanations (Summary / Beeswarm Plot)

The SHAP beeswarm plot ranks features by their mean absolute SHAP value and
shows how each sample's feature value (colour: red = high, blue = low)
contributes to the prediction.

**Key findings:**

- **`worst concave points`** — High values strongly push the prediction toward
  *malignant*; low values push toward *benign*.  This is the single most
  influential feature.
- **`worst perimeter` / `worst radius`** — Large tumour size correlates with
  malignancy, consistent with domain knowledge.
- **`mean texture`** — Moderate influence; higher texture values (more variation
  in grey-scale intensity) push toward malignancy.
- Interactions between size and shape features are visible via the colour spread
  in the dependence plot.

### 2.3 SHAP — Local Explanation (Force Plot)

For a single test sample, the force plot visualises how each feature
contributes to the deviation from the base value (average model output):

- Features in **red** push the prediction toward the positive class (benign).
- Features in **blue** push toward malignant.
- The width of each bar indicates the magnitude of the contribution.

This allows clinicians or auditors to understand *why* the model classified a
specific patient as malignant or benign.

### 2.4 LIME — Instance-Level Explanations

LIME generates surrogate linear models around individual predictions.  We
explained three representative samples:

| # | Type | Observation |
|---|------|-------------|
| 1 | Correctly classified **malignant** | Top features (worst concave points, worst area) align with SHAP — high consistency. |
| 2 | Correctly classified **benign** | Low values of shape features drive the benign prediction. |
| 3 | Misclassified / additional sample | May reveal edge-case features the model weighs differently than expected. |

**LIME vs SHAP agreement:** The same top features appear across both methods,
strengthening confidence that the model's reasoning is consistent and not an
artefact of a single explanation technique.

---

## 3. Bias & Fairness Analysis

### 3.1 Methodology

The breast-cancer dataset does not contain sensitive demographic attributes.  To
demonstrate a fairness auditing workflow we synthesised an `age_group` attribute
with three levels: *Young (<40)*, *Middle (40-60)*, *Senior (>60)*, assigned
randomly.

For each group we computed:

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness per group |
| Precision | Positive-predictive value |
| Recall | Sensitivity / true-positive rate |
| Selection Rate | Fraction predicted as positive (benign) |
| **Disparate Impact Ratio** | `min(selection rate) / max(selection rate)` |

### 3.2 Disparate Impact (80 % Rule)

The *four-fifths (80 %) rule* from US employment law states that if the
selection rate for a disadvantaged group falls below 80 % of the rate for the
most advantaged group, disparate impact may be present.

With **randomly assigned** groups we expect no systematic disparity, and the
analysis confirms this.  In a real-world setting, however:

- **An observed ratio < 0.80** would warrant deeper investigation.
- Additional metrics such as **Equalised Odds**, **Predictive Parity**, and
  **Calibration** should be examined.

### 3.3 Visualisations

The script produces a three-panel chart comparing Accuracy, Recall, and
Selection Rate across age groups, with the 80 % threshold marked.

---

## 4. Mitigation Recommendations

### 4.1 Data-Level (Pre-processing)

| Strategy | Description |
|----------|-------------|
| **Balanced Sampling** | Over-sample minority groups or under-sample majority groups. |
| **Fair Representation Learning** | Transform features so that protected-group membership cannot be inferred. |
| **Label Auditing** | Check whether labelling quality differs across groups (e.g., access to diagnostic resources). |

### 4.2 Model-Level (In-processing)

| Strategy | Description |
|----------|-------------|
| **Adversarial Debiasing** | Add an adversary that penalises the model for predictions correlated with the protected attribute. |
| **Fairness Constraints** | Use constrained optimisation (e.g., Fairlearn's `ExponentiatedGradient`) to enforce demographic parity or equalised odds during training. |
| **Regularisation** | Add a penalty term that discourages large differences in per-group error rates. |

### 4.3 Decision-Level (Post-processing)

| Strategy | Description |
|----------|-------------|
| **Threshold Tuning** | Set different classification thresholds per group to equalise false-positive / false-negative rates. |
| **Reject-Option Classification** | Defer borderline predictions (those near the decision boundary) for human review. |
| **Calibration** | Ensure predicted probabilities are well-calibrated within each group. |

### 4.4 Governance & Monitoring

- **Model Cards** — Document intended use, training data, fairness evaluations,
  and known limitations (Mitchell et al., 2019).
- **Datasheets for Datasets** — Record provenance, collection methodology, and
  demographic composition (Gebru et al., 2021).
- **Monitoring Dashboards** — Track per-group metrics in production; alert when
  disparity exceeds a threshold.
- **Periodic Re-auditing** — Re-run fairness checks on each model update or
  when the population distribution shifts.

---

## 5. The Importance of Responsible AI

Responsible AI is not a one-time checklist — it is a **continuous discipline**
woven into the entire ML lifecycle:

1. **Trust & Adoption** — Clinicians and patients are more likely to trust a
   model whose reasoning they can inspect.
2. **Regulatory Compliance** — The EU AI Act, FDA guidelines for
   Software-as-a-Medical-Device (SaMD), and similar regulations increasingly
   mandate explainability and bias assessments.
3. **Ethical Obligation** — Healthcare models directly affect human lives;
   unexplained or biased predictions can cause disproportionate harm to
   vulnerable groups.
4. **Model Improvement** — Explainability tools often uncover data-quality
   issues, feature-engineering mistakes, or shortcut learning — leading to
   genuinely better models.

---

## 6. Outputs & Artefacts

| File | Description |
|------|-------------|
| `task4_feature_importances.png` | Top-15 Gini feature importances |
| `task4_shap_summary.png` | SHAP beeswarm plot (global) |
| `task4_shap_force_plot.png` | SHAP force plot (single prediction) |
| `task4_shap_dependence.png` | SHAP dependence plot (top feature) |
| `task4_lime_explanation_1.png` | LIME — correct malignant sample |
| `task4_lime_explanation_2.png` | LIME — correct benign sample |
| `task4_lime_explanation_3.png` | LIME — misclassified / additional sample |
| `task4_bias_analysis.png` | Fairness metrics across synthetic age groups |

---

## 7. Conclusion

This analysis demonstrates that the Random Forest classifier learns clinically
meaningful patterns (cell shape, tumour size) and produces consistent
explanations across SHAP and LIME.  The synthetic bias audit showed no
disparate impact (as expected with random group assignment), but established a
reusable workflow for detecting and addressing fairness concerns when real
demographic data is available.

**Responsible AI is a shared responsibility** — model developers, domain
experts, and decision-makers must collaborate to ensure that ML systems are
transparent, fair, and accountable.

---

*Report generated as part of AI Internship Task 4.*
