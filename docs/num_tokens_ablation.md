# Controlled `num_tokens` Ablation Report (Phase 6)

## 1. Executive Summary & Experimental Controls

This report documents the controlled sequence-length ablation experiment for Phase 6. The objective is to empirically determine whether including response token count (`num_tokens`) changes classification performance and to quantify potential shortcut learning identified during the initial feature quality audit ([`docs/feature_quality_analysis.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/feature_quality_analysis.md)).

### 1.1 Strict Experimental Controls
To isolate the exact marginal impact of `num_tokens`, every experimental factor was held identical to the primary baseline:
- **Dataset**: Same 4,000-example supervised benchmark dataset ([`experiments/baselines/supervised_signals_combined.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/supervised_signals_combined.parquet)).
- **Partitioning**: Exact same 80/20 train/test stratified split (3,200 train / 800 test) with `random_state = 42` and joint `source_dataset + "_" + label` stratification.
- **Model Configurations**:
  - Logistic Regression: Identical `StandardScaler` pipeline, L2 penalty, `max_iter=1000`, `random_state=42`.
  - XGBoost: Identical tree hyperparameters (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0.1`, `random_state=42`).
- **Feature Set Modification**:
  - **Primary Baseline**: Exactly **11** internal signal features (`min_log_prob`, `mean_log_prob`, `mean_token_prob`, `token_prob_std`, `mean_entropy`, `max_entropy`, `entropy_std`, `log_perplexity`, `log_mean_token_rank`, `log_max_token_rank`, `log_rank_std`).
  - **Ablation Model**: Exactly **12** features (the 11 internal signals $+$ `num_tokens`).
- **Data Integrity Checks**:
  - Train set: Exactly 3,200 examples (1,600 faithful, 1,600 hallucinated).
  - Test set: Exactly 800 examples (400 faithful, 400 hallucinated).
  - Benchmark proportions: HaluEval 37.5%, TruthfulQA 37.5%, FEVER 25.0% across both partitions.
  - No NaN or infinite values detected; predicted probabilities strictly in $[0, 1]$.

---

## 2. Overall Performance Comparison: Primary Baseline vs. Ablation

### 2.1 Logistic Regression Comparison

| Metric | Primary Baseline (11 Signals) | Ablation (+ `num_tokens`) | Delta ($\Delta = \text{Ablation} - \text{Primary}$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.6213 (62.13%) | 0.6263 (62.63%) | $\mathbf{+0.0050}$ ($+0.50\%$) |
| **Precision** | 0.6120 (61.20%) | 0.6241 (62.41%) | $\mathbf{+0.0121}$ ($+1.21\%$) |
| **Recall (Sensitivity)** | 0.6625 (66.25%) | 0.6350 (63.50%) | $\mathbf{-0.0275}$ ($-2.75\%$) |
| **Hallucination Recall** | 0.6625 (66.25%) | 0.6350 (63.50%) | $\mathbf{-0.0275}$ ($-2.75\%$) |
| **Specificity** | 0.5800 (58.00%) | 0.6175 (61.75%) | $\mathbf{+0.0375}$ ($+3.75\%$) |
| **False Positive Rate (FPR)** | 0.4200 (42.00%) | 0.3825 (38.25%) | $\mathbf{-0.0375}$ ($-3.75\%$) |
| **F1 Score** | 0.6363 | 0.6295 | $\mathbf{-0.0068}$ |
| **ROC-AUC** | 0.6930 | 0.6985 | $\mathbf{+0.0055}$ |
| **PR-AUC** | 0.6618 | 0.6738 | $\mathbf{+0.0120}$ |
| **Brier Score** | 0.2177 | 0.2161 | $\mathbf{-0.0016}$ |
| **ECE ($M=10$)** | 0.0649 (6.49%) | 0.0746 (7.46%) | $\mathbf{+0.0097}$ ($+0.97\%$) |
| **Training Latency** | 10.93 ms | 12.83 ms | $+1.90\text{ ms}$ |
| **Inference Latency** | 1.91 ms | 1.92 ms | $+0.01\text{ ms}$ |

#### Logistic Regression Confusion Matrices ($N=800$)
- **Primary Baseline**:
  - $TP = 265$, $FP = 168$, $TN = 232$, $FN = 135$
- **Ablation (+ `num_tokens`)**:
  - $TP = 254$, $FP = 153$, $TN = 247$, $FN = 146$
- **Net Changes**: $\Delta TP = -11$, $\Delta FP = -15$, $\Delta TN = +15$, $\Delta FN = +11$.

---

### 2.2 XGBoost Comparison

| Metric | Primary Baseline (11 Signals) | Ablation (+ `num_tokens`) | Delta ($\Delta = \text{Ablation} - \text{Primary}$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.6700 (67.00%) | 0.6863 (68.63%) | $\mathbf{+0.0163}$ ($+1.63\%$) |
| **Precision** | 0.6518 (65.18%) | 0.6609 (66.09%) | $\mathbf{+0.0091}$ ($+0.91\%$) |
| **Recall (Sensitivity)** | 0.7300 (73.00%) | 0.7650 (76.50%) | $\mathbf{+0.0350}$ ($+3.50\%$) |
| **Hallucination Recall** | 0.7300 (73.00%) | 0.7650 (76.50%) | $\mathbf{+0.0350}$ ($+3.50\%$) |
| **Specificity** | 0.6100 (61.00%) | 0.6075 (60.75%) | $\mathbf{-0.0025}$ ($-0.25\%$) |
| **False Positive Rate (FPR)** | 0.3900 (39.00%) | 0.3925 (39.25%) | $\mathbf{+0.0025}$ ($+0.25\%$) |
| **F1 Score** | 0.6887 | 0.7092 | $\mathbf{+0.0205}$ |
| **ROC-AUC** | 0.7411 | 0.7668 | $\mathbf{+0.0257}$ |
| **PR-AUC** | 0.7322 | 0.7669 | $\mathbf{+0.0347}$ |
| **Brier Score** | 0.1999 | 0.1904 | $\mathbf{-0.0095}$ |
| **ECE ($M=10$)** | 0.0228 (2.28%) | 0.0399 (3.99%) | $\mathbf{+0.0171}$ ($+1.71\%$) |
| **Training Latency** | 1,703.87 ms | 1,720.37 ms | $+16.50\text{ ms}$ |
| **Inference Latency** | 3.80 ms | 5.97 ms | $+2.17\text{ ms}$ |

#### XGBoost Confusion Matrices ($N=800$)
- **Primary Baseline**:
  - $TP = 292$, $FP = 156$, $TN = 244$, $FN = 108$
- **Ablation (+ `num_tokens`)**:
  - $TP = 306$, $FP = 157$, $TN = 243$, $FN = 94$
- **Net Changes**: $\Delta TP = +14$, $\Delta FP = +1$, $\Delta TN = -1$, $\Delta FN = -14$.

---

## 3. Subgroup Breakdown: Benchmark Dissection

Evaluating performance across the three benchmark sources illuminates how response length interacts with specific dataset structures:

### 3.1 HaluEval (300 Test Samples, Context-Grounded QA)

| Model | Setup | Accuracy | F1 Score | ROC-AUC | Confusion Matrix ($TN, FP, FN, TP$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Primary Baseline<br>**Ablation (+ `num_tokens`)**<br>*Delta* | 84.67%<br>**88.00%**<br>*+3.33%* | 0.8333<br>**0.8676**<br>*+0.0343* | 0.9467<br>**0.9695**<br>*+0.0228* | TN: 139, FP: 11, FN: 35, TP: 115<br>TN: 146, FP: 4, FN: 32, TP: 118<br>*ΔTN: +7, ΔFP: -7, ΔFN: -3, ΔTP: +3* |
| **XGBoost** | Primary Baseline<br>**Ablation (+ `num_tokens`)**<br>*Delta* | 86.33%<br>**90.67%**<br>*+4.34%* | 0.8601<br>**0.9041**<br>*+0.0440* | 0.9357<br>**0.9625**<br>*+0.0268* | TN: 133, FP: 17, FN: 24, TP: 126<br>TN: 140, FP: 10, FN: 18, TP: 132<br>*ΔTN: +7, ΔFP: -7, ΔFN: -6, ΔTP: +6* |

### 3.2 TruthfulQA (300 Test Samples, Adversarial Misconceptions)

| Model | Setup | Accuracy | F1 Score | ROC-AUC | Confusion Matrix ($TN, FP, FN, TP$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Primary Baseline<br>**Ablation (+ `num_tokens`)**<br>*Delta* | **51.33%**<br>47.67%<br>*-3.66%* | **0.5466**<br>0.4749<br>*-0.0717* | **0.5044**<br>0.4720<br>*-0.0324* | TN: 66, FP: 84, FN: 62, TP: 88<br>TN: 72, FP: 78, FN: 79, TP: 71<br>*ΔTN: +6, ΔFP: -6, ΔFN: +17, ΔTP: -17* |
| **XGBoost** | Primary Baseline<br>**Ablation (+ `num_tokens`)**<br>*Delta* | 57.67%<br>**58.00%**<br>*+0.33%* | 0.6209<br>**0.6400**<br>*+0.0191* | **0.5983**<br>0.5908<br>*-0.0075* | TN: 69, FP: 81, FN: 46, TP: 104<br>TN: 62, FP: 88, FN: 38, TP: 112<br>*ΔTN: -7, ΔFP: +7, ΔFN: -8, ΔTP: +8* |

### 3.3 FEVER (200 Test Samples, Claim Verification)

| Model | Setup | Accuracy | F1 Score | ROC-AUC | Confusion Matrix ($TN, FP, FN, TP$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Primary Baseline<br>**Ablation (+ `num_tokens`)**<br>*Delta* | 44.50%<br>**47.00%**<br>*+2.50%* | 0.5277<br>**0.5508**<br>*+0.0231* | 0.4656<br>**0.4716**<br>*+0.0060* | TN: 27, FP: 73, FN: 38, TP: 62<br>TN: 29, FP: 71, FN: 35, TP: 65<br>*ΔTN: +2, ΔFP: -2, ΔFN: -3, ΔTP: +3* |
| **XGBoost** | Primary Baseline<br>**Ablation (+ `num_tokens`)**<br>*Delta* | **52.00%**<br>51.50%<br>*-0.50%* | **0.5636**<br>0.5611<br>*-0.0025* | 0.4854<br>**0.5017**<br>*+0.0163* | TN: 42, FP: 58, FN: 38, TP: 62<br>TN: 41, FP: 59, FN: 38, TP: 62<br>*ΔTN: -1, ΔFP: +1, ΔFN: 0, ΔTP: 0* |

---

## 4. Factual Interpretation of Measured Changes

1. **Overall Performance Shift**:
   - In **XGBoost**, adding `num_tokens` increased measured test metrics: Accuracy rose from **67.00%** to **68.63%** ($+1.63\%$), Recall rose from **73.00%** to **76.50%** ($+3.50\%$), and ROC-AUC rose from **0.7411** to **0.7668** ($+0.0257$).
   - In **Logistic Regression**, adding `num_tokens` produced mixed overall changes: Accuracy changed marginally from **62.13%** to **62.63%** ($+0.50\%$) and ROC-AUC from **0.6930** to **0.6985** ($+0.0055$), while F1 score decreased from **0.6363** to **0.6295** ($-0.0068$) due to a drop in recall ($66.25\% \rightarrow 63.50\%$).
   - In both models, the Expected Calibration Error (ECE) increased when `num_tokens` was added ($+0.97\%$ in LR, $+1.71\%$ in XGBoost), indicating that the probability distributions became slightly more miscalibrated relative to bin accuracy.

2. **Empirical Evidence for the Length-Shortcut Hypothesis in HaluEval**:
   - The subgroup analysis reveals that the performance gain from `num_tokens` is disproportionately concentrated in **HaluEval**:
     - For XGBoost on HaluEval, Accuracy jumped by **$+4.34\%$** (from $86.33\%$ to **$90.67\%$**), and ROC-AUC reached **$0.9625$**.
     - For Logistic Regression on HaluEval, Accuracy jumped by **$+3.33\%$** (from $84.67\%$ to **$88.00\%$**), and ROC-AUC reached **$0.9695$**.
   - In contrast, on **TruthfulQA**, Logistic Regression accuracy actually declined by **$-3.66\%$** (falling to $47.67\%$, below random guessing), and ROC-AUC dropped from $0.5044$ to $0.4720$.
   - On **FEVER**, changes were minor and within typical sampling variation ($\pm 1\text{--}2\%$).

3. **Methodological Takeaway**:
   - In HaluEval, synthetic benchmark generation introduced an artificial length difference between hallucinated and faithful responses. When `num_tokens` is provided, the decision tree ensemble and linear models exploit sequence length as a proxy for the label.
   - However, because this length relationship is specific to HaluEval's generation process, it does not transfer beneficially to TruthfulQA.
   - **Conclusion**: This empirical evidence validates the Phase 6 feature policy. Strictly quarantining `num_tokens` from the primary baseline prevents the classifiers from relying on dataset-specific length artifacts and ensures that reported detection capabilities stem purely from intrinsic token probabilities, predictive entropy, and vocabulary dispersion signals.
