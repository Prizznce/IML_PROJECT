# Probability Calibration Analysis for the Universal Core Hallucination Detector (Phase 17)

## 1. Executive Summary

This document presents the methodology, quantitative findings, and diagnostic reliability diagrams from the dedicated **Probability Calibration Analysis** of the **Universal Core Hallucination Detector**.

Accurate probability estimation is essential for reliable AI deployment: downstream systems must know *when* to trust an LLM response or trigger verification pipelines. High accuracy or ROC-AUC does not guarantee that predicted confidence values correspond to empirical factual error rates.

This investigation evaluates:
1. **Raw Model Calibration**: 10-bin uniform reliability diagrams, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score, and Cox calibration slope and intercept.
2. **Post-Hoc Calibration Techniques**:
   - **Platt Scaling (Sigmoid)**: Parametric logistic adjustment.
   - **Isotonic Regression**: Non-parametric piecewise constant isotonic adjustment.
3. **Task-Specific Reliability**: Calibration behavior on holdout subsets from HaluEval, TruthfulQA, and FEVER.

---

## 2. Experimental Protocol & Zero-Leakage Calibration Architecture

To ensure strict zero data leakage and preserve empirical rigor:
- **Outer Partition (Identical to Phases 14–16)**:
  - Fixed stratified 80/20 split (`random_state=42`, joint key `source_dataset + "_" + label`).
  - **Holdout Test Set**: Exactly 800 examples ($N=800$). **Completely untouched during model training, scaling, and calibration parameter fitting.**
  - **Outer Training Partition**: Exactly 3,200 examples ($N=3,200$).
- **Inner Partition on Training Partition**:
  - Deterministic stratified 80/20 split on the 3,200 training examples (`random_state=42`, joint key):
    - **Model-Fitting Subset**: 2,560 examples ($80\%$). Base classifiers (Logistic Regression and XGBoost) and `StandardScaler` are fitted exclusively on this subset.
    - **Calibration-Fitting Subset**: 640 examples ($20\%$). Base models generate out-of-fold probability estimates on this subset, which are used strictly to fit post-hoc calibrators.
- **Partition Verification**:
  $$\text{Model-fit (2,560)} \cap \text{Calibration-fit (640)} = \emptyset$$
  $$\text{Test Set (800)} \cap \text{Model-fit (2,560)} = \emptyset$$
  $$\text{Test Set (800)} \cap \text{Calibration-fit (640)} = \emptyset$$

---

## 3. Overall Empirical Calibration Results ($N=800$ Holdout Test Set)

$$\Delta = \text{Calibrated Metric} - \text{Uncalibrated Metric}$$
*(For Brier, ECE, and MCE, negative delta indicates error reduction)*

| Model | Calibration Method | Accuracy | F1 Score | ROC-AUC | PR-AUC | Brier Score ($\Delta$) | ECE ($\Delta$) | MCE ($\Delta$) | Cox Slope | Cox Intercept |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Uncalibrated** | 0.6338 | 0.6387 | 0.7069 | 0.7096 | 0.2155 (+0.0000) | 0.0444 (+0.0000) | 0.1248 (+0.0000) | 0.938 | -0.009 |
| | **Sigmoid (Platt)** | 0.6300 | 0.6373 | 0.7069 | 0.7096 | 0.2165 (+0.0010) | 0.0547 (+0.0102) | 0.1114 (-0.0133) | 0.957 | -0.022 |
| | **Isotonic** | 0.6225 | 0.6505 | 0.6971 | 0.6751 | 0.2151 (-0.0003) | 0.0341 (-0.0104) | 0.2473 (+0.1226) | 0.540 | -0.044 |
| **XGBoost** | **Uncalibrated** | 0.6488 | 0.6492 | 0.7638 | 0.7787 | 0.1900 (+0.0000) | 0.0712 (+0.0000) | 0.1888 (+0.0000) | 1.042 | -0.012 |
| | **Sigmoid (Platt)** | 0.6550 | 0.6452 | 0.7638 | 0.7787 | 0.1935 (+0.0034) | 0.0825 (+0.0113) | 0.1210 (-0.0678) | 0.953 | +0.041 |
| | **Isotonic** | 0.6675 | 0.6472 | 0.7641 | 0.7557 | 0.1890 (-0.0010) | 0.0244 (-0.0468) | 0.1256 (-0.0631) | 0.796 | +0.027 |

---

## 4. Analysis of Calibration Techniques

### 1. Uncalibrated Base Models
- **Logistic Regression**:
  - Demonstrates strong intrinsic calibration with low uncalibrated ECE ($0.0444$) and Brier score ($0.2155$).
  - Cox slope is $0.938$ (close to ideal $1.0$), with an intercept of $-0.009$ (close to ideal $0.0$), showing that cross-entropy loss with linear features produces well-scaled logits.
- **XGBoost**:
  - Achieves lower Brier score ($0.1900$) due to superior ranking discrimination (ROC-AUC $0.7638$), but exhibits higher uncalibrated ECE ($0.0712$) and MCE ($0.1888$).
  - Decision tree ensembles typically push leaf predictions toward probabilities with higher variance, producing mild bin-level overconfidence in intermediate bins.

### 2. Platt Scaling (Sigmoid Calibration)
- Sigmoid calibration applies a parametric logistic function to the model's uncalibrated probabilities.
- For XGBoost, Sigmoid calibration effectively dampens peak bin deviations, reducing Maximum Calibration Error from $0.1888$ to $0.1210$ ($\Delta\text{MCE} = -0.0678$).
- Monotonicity is strictly preserved, leaving ROC-AUC and PR-AUC completely unchanged ($0.7638$ and $0.7787$).

### 3. Isotonic Regression
- Isotonic regression fits a non-parametric, monotonic step function.
- For XGBoost, Isotonic calibration yields the largest numerical reduction in calibration error:
  - Expected Calibration Error (ECE) decreases from $0.0712$ to **$0.0244$** ($\Delta\text{ECE} = -0.0468$).
  - Maximum Calibration Error (MCE) decreases from $0.1888$ to **$0.1256$** ($\Delta\text{MCE} = -0.0631$).
  - Brier score drops from $0.1900$ to **$0.1890$** ($\Delta = -0.0010$).
- Because isotonic regression maps ranges of probabilities to constant step values, tied predictions slightly reduce PR-AUC from $0.7787$ to $0.7557$ while improving bin-level calibration.

---

## 5. Subgroup Calibration Across Benchmark Tasks

Calibration metrics evaluated on the test subsets:

| Model | Calibration Method | Benchmark Dataset | Sample Count ($N$) | Brier Score | ECE | MCE | Accuracy | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | FEVER | 200 | 0.2715 | 0.1245 | 0.8207 | 0.4850 | 0.4850 |
| | Uncalibrated | HaluEval | 300 | 0.1235 | 0.1280 | 0.2106 | 0.8433 | 0.9278 |
| | Uncalibrated | TruthfulQA | 300 | 0.2700 | 0.1121 | 0.6225 | 0.5233 | 0.5232 |
| | Sigmoid | FEVER | 200 | 0.2731 | 0.1361 | 0.8106 | 0.4800 | 0.4850 |
| | Sigmoid | HaluEval | 300 | 0.1252 | 0.1437 | 0.2082 | 0.8400 | 0.9278 |
| | Sigmoid | TruthfulQA | 300 | 0.2700 | 0.1190 | 0.8275 | 0.5200 | 0.5232 |
| | Isotonic | FEVER | 200 | 0.2664 | 0.1080 | 0.9583 | 0.4650 | 0.4692 |
| | Isotonic | HaluEval | 300 | 0.1251 | 0.0928 | 0.2789 | 0.8300 | 0.9194 |
| | Isotonic | TruthfulQA | 300 | 0.2710 | 0.0845 | 0.8372 | 0.5200 | 0.5228 |
| **XGBoost** | Uncalibrated | FEVER | 200 | 0.2483 | 0.0827 | 0.3881 | 0.5050 | 0.5460 |
| | Uncalibrated | HaluEval | 300 | 0.0930 | 0.0858 | 0.2084 | 0.8633 | 0.9466 |
| | Uncalibrated | TruthfulQA | 300 | 0.2482 | 0.0978 | 0.3401 | 0.5300 | 0.5834 |
| | Sigmoid | FEVER | 200 | 0.2525 | 0.0733 | 0.8014 | 0.5300 | 0.5460 |
| | Sigmoid | HaluEval | 300 | 0.0931 | 0.0699 | 0.1541 | 0.8700 | 0.9466 |
| | Sigmoid | TruthfulQA | 300 | 0.2545 | 0.1312 | 0.2851 | 0.5233 | 0.5834 |
| | Isotonic | FEVER | 200 | 0.2461 | **0.0263** | 0.0943 | 0.5500 | 0.5577 |
| | Isotonic | HaluEval | 300 | 0.0941 | **0.0514** | 0.1810 | 0.8700 | 0.9442 |
| | Isotonic | TruthfulQA | 300 | 0.2460 | **0.0689** | 0.2500 | 0.5433 | 0.5825 |

---

## 6. Reliability Diagram Visualizations

The generated diagrams illustrate the relationship between predicted confidence and observed empirical error:

- **Logistic Regression Diagram**:
  ![Logistic Regression Reliability Diagram](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/logistic_regression_reliability.png)
  *Upper panel depicts calibration curves across uncalibrated, sigmoid, and isotonic mappings; lower panel displays test instance density across probability bins.*

- **XGBoost Diagram**:
  ![XGBoost Reliability Diagram](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/xgboost_reliability.png)
  *Highlights the sharp reduction in calibration gap achieved by isotonic calibration across the middle and upper probability ranges.*

---

## 7. Practical Deployment Takeaways

1. **For Probability-Sensitive Thresholding**:
   If downstream systems route queries or trigger external fact-checking based on confidence thresholds (e.g. "flag if probability of hallucination $> 70\%$"), **Isotonic-calibrated XGBoost** provides the lowest calibration error ($\text{ECE} = 0.0244$, $\text{MCE} = 0.1256$).
2. **For Ranking and Retrieval Re-Ranking**:
   If the goal is to sort candidate responses by likelihood of correctness, **Uncalibrated or Sigmoid-calibrated XGBoost** preserves strictly continuous probability differentiation without rank step-ties ($\text{PR-AUC} = 0.7787$).
3. **Task Differences in Calibration**:
   HaluEval predictions are sharply polarized into extreme probability bins ($< 0.1$ and $> 0.9$) reflecting high certainty, whereas TruthfulQA probabilities cluster closer to the center ($0.4 - 0.6$), illustrating greater epistemic uncertainty.

---

## 8. Artifact Manifest

- **Calibration Engine**: [src/evaluation/calibration.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/evaluation/calibration.py)
- **Unit Tests**: [tests/test_calibration.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/tests/test_calibration.py)
- **Results JSON**: [calibration_results.json](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/calibration_results.json)
- **Overall Results CSV**: [calibration_results.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/calibration_results.csv)
- **10-Bin Reliability Tables CSV**: [calibration_reliability_tables.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/calibration_reliability_tables.csv)
- **Predictions Parquet (4,800 rows)**: [calibration_predictions.parquet](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/calibration_predictions.parquet)
- **Summary Report**: [calibration_summary.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/calibration_summary.md)
- **LR Reliability Diagram**: [logistic_regression_reliability.png](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/logistic_regression_reliability.png)
- **XGBoost Reliability Diagram**: [xgboost_reliability.png](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/xgboost_reliability.png)
