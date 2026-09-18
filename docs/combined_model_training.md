# Universal Combined Hallucination Detector — Training and Validation (Phase 14)

## 1. Executive Summary

This document presents the implementation, mathematical validation, empirical results, and baseline comparative analysis for the **Universal Core Hallucination Detector** trained across the 19 canonical features established in Phase 13.

The detector integrates three complementary signal families:
1. **Internal Generation Signals (11 features)**: Token probabilities, predictive entropy, perplexity, and rank dispersion statistics computed directly on the primary evaluated response.
2. **Self-Consistency Signals (5 features)**: Exact match consensus and pairwise semantic similarity distributions across 5 stochastically sampled generations ($T=0.7, \text{top\_p}=0.9$).
3. **NLI Agreement Signals (3 features)**: Cross-encoder natural language inference metrics (entailment, contradiction, and calibrated disagreement) across all $\binom{5}{2} = 10$ unordered generation pairs.

---

## 2. Crucial Methodological Distinction

> [!IMPORTANT]
> **Evaluated Response Signals vs. Unlabeled Behavior Probes:**
> 
> - **Internal Token Signals**: Evaluated strictly on the **original benchmark response** whose truthfulness is being classified. These signals quantify the model's immediate token-by-token internal confidence when uttering that specific statement.
> - **Self-Consistency & NLI Agreement Signals**: Derived from **multiple newly generated responses** elicited under the same prompt. These newly generated responses are **unlabeled behavioral probes**.
> - **Ground Truth Isolation**: The original benchmark label (0 or 1) belongs **strictly to the target benchmark response**. It is **never assigned** to any of the 5 newly generated responses. The self-consistency and NLI signals characterize the stability and semantic consensus of the model's hypothesis space around that prompt; they are features of the prompt-response pair, not labels.
> - **Predictive Associations vs. Causality**: Feature correlations with hallucination targets reflect statistical associations within the evaluated corpus. No feature or group of features is claimed to have a causal relationship with hallucination.

---

## 3. Experimental Protocol & Partitioning

To maintain strict, uncompromised comparability with the Phase 6 baseline:
- **Input Table**: `experiments/baselines/combined/universal_features.parquet` (4,000 rows $\times$ 22 columns).
- **Feature Matrix ($X$)**: Exactly 19 features; zero metadata, zero text payloads, zero sequence length (`num_tokens`), zero retrieval features.
- **Stratified Partitioning**: 80/20 train/test split with `random_state=42`, stratified by the composite joint key `source_dataset + "_" + label`.
  - **Training Set**: 3,200 samples (1,600 label 0; 1,600 label 1).
  - **Holdout Test Set**: 800 samples (400 label 0; 400 label 1).
  - **Test Subgroup Distribution**: HaluEval ($N=300$), TruthfulQA ($N=300$), FEVER ($N=200$).
- **Data Leakage Elimination**:
  - `StandardScaler` in the Logistic Regression pipeline is fit **strictly on the training partition**.
  - All test and subgroup evaluations operate exclusively on holdout test instances.

---

## 4. Evaluated Model Architectures

### Model 1: Regularized Logistic Regression (Linear Baseline)
- **Pipeline**: `StandardScaler` $\to$ `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)`
- **Loss**: Cross-entropy with $L_2$ weight regularization.

### Model 2: Gradient Boosted Trees (XGBoost)
- **Estimator**: `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=0.1, eval_metric="logloss", random_state=42, n_jobs=-1)`
- **Configuration**: Preserved identically from the Phase 6 baseline to isolate the exact empirical impact of feature integration without confounding hyperparameter shifts.

---

## 5. Overall Performance and Baseline Comparison

The table below contrasts the 19-feature **Universal Combined Model** against the 11-feature **Internal Signals Baseline** evaluated on the identical 800-sample test set.

$$\Delta = \text{Combined (19 features)} - \text{Baseline (11 features)}$$

| Model | Metric | Baseline (11 feats) | Combined (19 feats) | Delta ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | **Accuracy** | 0.6212 | **0.6288** | **+0.0075** |
| | Precision | 0.6120 | 0.6241 | +0.0121 |
| | Recall | 0.6625 | 0.6475 | -0.0150 |
| | **F1 Score** | 0.6363 | **0.6356** | **-0.0007** |
| | **ROC-AUC** | 0.6930 | **0.7079** | **+0.0149** |
| | **PR-AUC** | 0.6618 | **0.7124** | **+0.0505** |
| | **Brier Score** | 0.2177 | **0.2151** | **-0.0026** |
| | **ECE (10-bin)** | 0.0649 | **0.0525** | **-0.0124** |
| | Confusion Matrix | TP=265, FP=168<br>TN=232, FN=135 | TP=259, FP=156<br>TN=244, FN=141 | $\Delta\text{FP}=-12, \Delta\text{TN}=+12$ |
| **XGBoost** | **Accuracy** | 0.6700 | **0.6788** | **+0.0088** |
| | Precision | 0.6518 | 0.6829 | +0.0311 |
| | Recall | 0.7300 | 0.6675 | -0.0625 |
| | **F1 Score** | 0.6887 | **0.6751** | **-0.0136** |
| | **ROC-AUC** | 0.7411 | **0.7688** | **+0.0277** |
| | **PR-AUC** | 0.7322 | **0.7795** | **+0.0474** |
| | **Brier Score** | 0.1999 | **0.1891** | **-0.0107** |
| | **ECE (10-bin)** | 0.0228 | **0.0341** | **+0.0113** |
| | Confusion Matrix | TP=292, FP=156<br>TN=244, FN=108 | TP=267, FP=124<br>TN=276, FN=133 | $\Delta\text{FP}=-32, \Delta\text{TN}=+32$ |

---

## 6. Subgroup Performance by Benchmark Origin

Evaluation on the holdout test set partitioned by benchmark task family:

### 1. HaluEval ($N=300$ test samples: 150 factual, 150 hallucinated)
- **Logistic Regression**:
  - Accuracy: `0.8400` ($\Delta = -0.0067$)
  - F1 Score: `0.8345` ($\Delta = +0.0011$)
  - ROC-AUC: `0.9311` ($\Delta = -0.0156$)
  - PR-AUC: `0.9352`
  - Brier Score: `0.1224` | ECE: `0.1333`
  - Confusion: TP=121, FP=19, TN=131, FN=29
- **XGBoost**:
  - Accuracy: `0.8633` ($\Delta = 0.0000$)
  - F1 Score: `0.8620` ($\Delta = +0.0019$)
  - ROC-AUC: `0.9433` ($\Delta = +0.0076$)
  - PR-AUC: `0.9368`
  - Brier Score: `0.0961` | ECE: `0.0668`
  - Confusion: TP=128, FP=19, TN=131, FN=22

### 2. TruthfulQA ($N=300$ test samples: 150 truthful, 150 falsehoods)
- **Logistic Regression**:
  - Accuracy: `0.5133` ($\Delta = 0.0000$)
  - F1 Score: `0.5166` ($\Delta = -0.0300$)
  - ROC-AUC: `0.5210` ($\Delta = +0.0166$)
  - PR-AUC: `0.4996`
  - Brier Score: `0.2703` | ECE: `0.1209`
  - Confusion: TP=78, FP=74, TN=76, FN=72
- **XGBoost**:
  - Accuracy: `0.5933` ($\Delta = +0.0167$)
  - F1 Score: `0.5933` ($\Delta = -0.0276$)
  - ROC-AUC: `0.6066` ($\Delta = +0.0083$)
  - PR-AUC: `0.6171`
  - Brier Score: `0.2420` | ECE: `0.0527`
  - Confusion: TP=89, FP=61, TN=89, FN=61

### 3. FEVER ($N=200$ test samples: 100 supported, 100 refuted)
- **Logistic Regression**:
  - Accuracy: `0.4850` ($\Delta = +0.0400$)
  - F1 Score: `0.5381` ($\Delta = +0.0105$)
  - ROC-AUC: `0.4870` ($\Delta = +0.0214$)
  - PR-AUC: `0.5003`
  - Brier Score: `0.2713` | ECE: `0.1303`
  - Confusion: TP=60, FP=63, TN=37, FN=40
- **XGBoost**:
  - Accuracy: `0.5300` ($\Delta = +0.0100$)
  - F1 Score: `0.5155` ($\Delta = -0.0482$)
  - ROC-AUC: `0.5441` ($\Delta = +0.0587$)
  - PR-AUC: `0.5682`
  - Brier Score: `0.2495` | ECE: `0.0655`
  - Confusion: TP=50, FP=44, TN=56, FN=50

---

## 7. Key Empirical Observations

1. **Ranking & Threshold-Free Discrimination (AUC gains)**:
   - For XGBoost, ROC-AUC increased from `0.7411` to `0.7688` ($\Delta = +0.0277$), and PR-AUC increased from `0.7322` to `0.7795` ($\Delta = +0.0474$).
   - For Logistic Regression, PR-AUC improved by $+0.0505$ (from `0.6618` to `0.7124`).
2. **Precision vs. Recall Trade-Off**:
   - Both models experienced a reduction in false positives ($\Delta\text{FP} = -32$ for XGBoost, $\Delta\text{FP} = -12$ for Logistic Regression).
   - XGBoost precision rose from $65.18\%$ to $68.29\%$, while default-threshold ($0.5$) recall decreased from $73.00\%$ to $66.75\%$.
3. **Probabilistic Scoring Quality**:
   - XGBoost Brier score improved from `0.1999` to `0.1891` ($\Delta = -0.0107$).
   - Logistic Regression Brier score improved from `0.2177` to `0.2151` ($\Delta = -0.0026$), with ECE decreasing from `0.0649` to `0.0525` ($\Delta = -0.0124$).

---

## 8. Artifacts and Reproducibility

All experiment outputs have been generated and validated:
- Pipeline: [src/training/combined_models.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/training/combined_models.py)
- Test Suite: [tests/test_combined_models.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/tests/test_combined_models.py)
- Metrics Summary JSON: [combined_model_results.json](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/combined_model_results.json)
- Metrics Summary CSV: [combined_model_results.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/combined_model_results.csv)
- Test Predictions Parquet: [combined_test_predictions.parquet](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/combined_test_predictions.parquet)
- Test Predictions CSV: [combined_test_predictions.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/combined_test_predictions.csv)
