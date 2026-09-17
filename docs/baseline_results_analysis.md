# Baseline Classification Results Analysis (Phase 6)

## 1. Executive Overview & Experimental Setup

This report provides a factual, read-only analysis of the primary supervised classification baselines trained to detect LLM hallucinations using strictly internal model signals from `Qwen/Qwen3.5-0.8B` (`bfloat16`).

The evaluation was executed on the canonical combined dataset:
[`experiments/baselines/supervised_signals_combined.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/supervised_signals_combined.parquet)

The machine-readable metrics and confusion matrices are recorded in:
- Raw metrics: [`experiments/baselines/results/baseline_metrics.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/results/baseline_metrics.json)
- Structured summary: [`experiments/baselines/results/baseline_results_summary.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/results/baseline_results_summary.json)

### 1.1 Dataset Partitioning & Feature Policy
- **Total Dataset**: 4,000 examples (balanced: 2,000 faithful, 2,000 hallucinated).
- **Split Scheme**: 80/20 train/test stratified split with `random_state = 42`.
- **Training Partition**: **3,200** samples (1,600 label 0 / 1,600 label 1).
- **Test Partition**: **800** samples (400 label 0 / 400 label 1).
- **Stratification Strategy**: Joint stratification on `source_dataset + "_" + label` to guarantee matched benchmark proportions across partitions:
  - HaluEval: 1,200 train / 300 test (50% label 0, 50% label 1).
  - TruthfulQA: 1,200 train / 300 test (50% label 0, 50% label 1).
  - FEVER: 800 train / 200 test (50% label 0, 50% label 1).
- **Feature Set (11 internal signals)**:
  `min_log_prob`, `mean_log_prob`, `mean_token_prob`, `token_prob_std`, `mean_entropy`, `max_entropy`, `entropy_std`, `log_perplexity`, `log_mean_token_rank`, `log_max_token_rank`, `log_rank_std`.
- **Quarantined Columns**: `num_tokens` (strictly quarantined to prevent length-shortcut learning), `id`, `source_dataset`, `label`, `prompt`, `context`, `response`, `model_input`, `forward_time_s`.

---

## 2. Quantitative Baseline Results

### 2.1 Model Performance Summary on the 800-Example Test Split

| Metric | Logistic Regression | XGBoost | Absolute Difference ($\Delta = \text{XGB} - \text{LR}$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.6213 (62.13%) | 0.6700 (67.00%) | $+0.0488$ ($+4.88\%$) |
| **Precision** | 0.6120 (61.20%) | 0.6518 (65.18%) | $+0.0398$ ($+3.98\%$) |
| **Recall (Sensitivity)** | 0.6625 (66.25%) | 0.7300 (73.00%) | $+0.0675$ ($+6.75\%$) |
| **Hallucination Recall** ($\frac{TP}{TP+FN}$) | 0.6625 (66.25%) | 0.7300 (73.00%) | $+0.0675$ ($+6.75\%$) |
| **Specificity** ($\frac{TN}{TN+FP}$) | 0.5800 (58.00%) | 0.6100 (61.00%) | $+0.0300$ ($+3.00\%$) |
| **False Positive Rate (FPR)** ($\frac{FP}{FP+TN}$) | 0.4200 (42.00%) | 0.3900 (39.00%) | $-0.0300$ ($-3.00\%$) |
| **F1 Score** | 0.6363 | 0.6887 | $+0.0524$ |
| **ROC-AUC** | 0.6930 | 0.7411 | $+0.0481$ |
| **PR-AUC (Average Precision)** | 0.6618 | 0.7322 | $+0.0703$ |
| **Brier Score** | 0.2177 | 0.1999 | $-0.0178$ |
| **Expected Calibration Error (ECE, $M=10$)** | 0.0649 (6.49%) | 0.0228 (2.28%) | $-0.0421$ ($-4.21\%$) |
| **Training (Fit) Latency** | **10.93 ms** | 1,703.87 ms | $+1,692.94\text{ ms}$ |
| **Test Prediction Latency** | **1.91 ms** | 3.80 ms | $+1.89\text{ ms}$ |

---

### 2.2 Confusion Matrices Breakdown

#### Logistic Regression
Total test samples: $N = 800$ (Ground truth: 400 Faithful, 400 Hallucinated).
```
                       Predicted Faithful (0)    Predicted Hallucinated (1)
Actual Faithful (0)            TN = 232                  FP = 168
Actual Hallucinated (1)        FN = 135                  TP = 265
```
- **True Positives ($TP$)**: 265 correctly identified hallucinations.
- **True Negatives ($TN$)**: 232 correctly identified faithful responses.
- **False Positives ($FP$)**: 168 faithful responses incorrectly flagged as hallucinations (Type I error).
- **False Negatives ($FN$)**: 135 hallucinations that went undetected (Type II error).
- **Hallucination Recall**: $\frac{265}{265 + 135} = \frac{265}{400} = 66.25\%$
- **Specificity**: $\frac{232}{232 + 168} = \frac{232}{400} = 58.00\%$
- **False Positive Rate**: $\frac{168}{168 + 232} = \frac{168}{400} = 42.00\%$

#### XGBoost
Total test samples: $N = 800$ (Ground truth: 400 Faithful, 400 Hallucinated).
```
                       Predicted Faithful (0)    Predicted Hallucinated (1)
Actual Faithful (0)            TN = 244                  FP = 156
Actual Hallucinated (1)        FN = 108                  TP = 292
```
- **True Positives ($TP$)**: 292 correctly identified hallucinations ($+27$ compared to LR).
- **True Negatives ($TN$)**: 244 correctly identified faithful responses ($+12$ compared to LR).
- **False Positives ($FP$)**: 156 faithful responses incorrectly flagged ($-12$ compared to LR).
- **False Negatives ($FN$)**: 108 hallucinations that went undetected ($-27$ compared to LR).
- **Hallucination Recall**: $\frac{292}{292 + 108} = \frac{292}{400} = 73.00\%$
- **Specificity**: $\frac{244}{244 + 156} = \frac{244}{400} = 61.00\%$
- **False Positive Rate**: $\frac{156}{156 + 244} = \frac{156}{400} = 39.00\%$

---

## 3. Subgroup Breakdown Across Benchmark Sources

Evaluating performance per benchmark source on the hold-out test split reveals substantial variance across task formulations:

| Benchmark Subgroup | Test Samples | Model | Accuracy | F1 Score | ROC-AUC | Confusion Matrix ($TN, FP, FN, TP$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **HaluEval** (QA with Context) | 300 | Logistic Regression<br>XGBoost | 84.67%<br>**86.33%** | 0.8333<br>**0.8601** | **0.9467**<br>0.9357 | TN: 139, FP: 11, FN: 35, TP: 115<br>TN: 133, FP: 17, FN: 24, TP: 126 |
| **TruthfulQA** (Adversarial False Beliefs) | 300 | Logistic Regression<br>XGBoost | 51.33%<br>**57.67%** | 0.5466<br>**0.6209** | 0.5044<br>**0.5983** | TN: 66, FP: 84, FN: 62, TP: 88<br>TN: 69, FP: 81, FN: 46, TP: 104 |
| **FEVER** (Fact Verification / Claims) | 200 | Logistic Regression<br>XGBoost | 44.50%<br>**52.00%** | 0.5277<br>**0.5636** | 0.4656<br>**0.4854** | TN: 27, FP: 73, FN: 38, TP: 62<br>TN: 42, FP: 58, FN: 38, TP: 62 |

---

## 4. Factual Interpretation of Measured Results

### 4.1 Discrimination Capacity on This Test Split
1. **Evidence of Signal Utility**:
   - Both classifiers substantially exceed the chance baseline (Accuracy = 50.0%, ROC-AUC = 0.50) on this combined test split.
   - Logistic Regression achieves an accuracy of **62.13%** and an ROC-AUC of **0.6930**.
   - XGBoost achieves an accuracy of **67.00%** and an ROC-AUC of **0.7411**.
   - These observations confirm that raw logit signals (token likelihoods, predictive entropy, perplexity, and rank dispersion) contain measurable statistical separation between faithful and hallucinated sequences without relying on external retrievers, web search, or secondary judge models.

2. **Linear vs. Non-linear Ensembles on This Partition**:
   - On this specific test split, XGBoost demonstrated higher point estimates across discrimination metrics:
     - Accuracy: $+4.88\%$ absolute difference ($0.6700$ vs $0.6213$).
     - ROC-AUC: $+0.0481$ ($0.7411$ vs $0.6930$).
     - PR-AUC: $+0.0703$ ($0.7322$ vs $0.6618$).
     - Recall on Hallucinations: $+6.75\%$ ($73.00\%$ vs $66.25\%$).
   - This difference suggests that non-linear decision boundaries and feature interaction terms (e.g., interaction between entropy standard deviation and worst-case token rank) provide additional empirical utility beyond purely additive linear weights on standardized features.
   - However, Logistic Regression offers vastly lower training latency (10.93 ms vs 1,703.87 ms, a $\sim 156\times$ speedup) and direct interpretability via linear weights, while both models execute test inference in $<4\text{ ms}$.
   - **Boundary condition**: We do not claim that XGBoost is universally superior across arbitrary out-of-domain distributions; rather, on this fixed stratified test split, tree-based interactions yielded higher measured discrimination.

3. **Calibration Profile**:
   - Both models demonstrated respectable probability calibration on this test set.
   - Logistic Regression yielded a Brier Score of **0.2177** and an ECE ($M=10$) of **0.0649** (6.49%).
   - XGBoost yielded a Brier Score of **0.1999** and an ECE ($M=10$) of **0.0228** (2.28%).
   - While XGBoost raw probabilities aligned closer to observed empirical accuracy in uniform confidence bins on this split, post-hoc calibration has not yet been applied.

---

## 5. Methodological Analysis & Design Rationale

### 5.1 Why `num_tokens` Was Quarantined from the Primary Baseline
As documented in the feature quality audit ([`docs/feature_quality_analysis.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/feature_quality_analysis.md)), response length correlates with labels in specific benchmark datasets due to synthetic generation artifacts:
- In HaluEval, synthetic hallucination perturbations altered average string lengths relative to human ground-truth responses.
- If sequence length is accessible to the classifier during training, decision trees and linear models assign substantial importance to token count. The classifier learns a **length-shortcut heuristic** rather than evaluating the intrinsic linguistic uncertainty of the model's forward pass.
- By strictly enforcing the exclusion of `num_tokens`, the primary baseline measures true semantic and probabilistic signals, preserving scientific validity.

### 5.2 Why the Next Experiment Must Be a Controlled `num_tokens` Ablation
While length was quarantined to prevent shortcut learning, evaluating the exact magnitude of length bias is an essential scientific question.
- In the next planned experiment, models will be trained on the identical 80/20 split with the addition of `num_tokens` (`--include-num-tokens`).
- Comparing the resulting metrics ($\Delta \text{Accuracy}$, $\Delta \text{ROC-AUC}$, feature importance ranks) will empirically quantify:
  1. How much predictive gain (if any) is attributable to length artifacts.
  2. Which internal signals are most vulnerable to collinearity with sequence length.
  3. Whether tree ensembles exploit length more aggressively than regularized linear models.

### 5.3 Why Probability Calibration Must Be Evaluated Separately
Although raw probabilities yielded an ECE of $2.28\%$ (XGBoost) and $6.49\%$ (Logistic Regression):
- Standard classifier training objectives (logistic loss, binary cross-entropy) optimize discrimination (ranking loss), not probabilistic calibration.
- In mission-critical hallucination detection, downstream systems require trustworthy risk estimates $P(\text{Hallucination} \mid \mathbf{x})$.
- Post-hoc calibration methods (Platt scaling / temperature scaling, Isotonic regression) must be fit on a dedicated validation split to avoid overfitting confidence bins.
- Evaluating calibration separately from discrimination ensures that calibration improvements do not distort rank-order metrics (ROC-AUC is rank-invariant under monotonic probability transforms, but Brier score and ECE are sensitive).

### 5.4 Why Cross-Dataset Evaluation Is Necessary
The subgroup breakdown in Section 3 reveals critical task divergence:
- On **HaluEval** (where responses are conditioned on a reference context), the internal signals achieve an ROC-AUC of **0.9467** (LR) and **0.9357** (XGBoost). The model's internal probability drops sharply when an answer departs from the given context.
- On **TruthfulQA** (adversarial questions probing widespread human misconceptions), ROC-AUC drops to **0.5044** (LR) and **0.5983** (XGBoost). When the LLM generates a common misconception learned during pre-training, it often does so with high confidence (low entropy, low token rank), attenuating pure logit signals.
- On **FEVER** (isolated factual claims without retrieval context), ROC-AUC is **0.4656** (LR) and **0.4854** (XGBoost).
- **Conclusion**: A model trained on a pooled dataset may memorize dataset-specific signal distributions. Cross-dataset evaluation (e.g., train on HaluEval + FEVER, test on TruthfulQA) is essential to evaluate true out-of-domain transferability and establish the operational boundary of internal signals.

---

## 6. Current Findings (Report Summary)

> **Key Findings for Final Project Report**:
>
> 1. **Primary Baseline Benchmark**: On the 800-example hold-out test partition of the 4,000-example combined benchmark, internal generation signals from `Qwen3.5-0.8B` demonstrated clear predictive signal for hallucination detection:
>    - **Logistic Regression**: Accuracy = **62.13%**, F1 = **0.6363**, ROC-AUC = **0.6930**, PR-AUC = **0.6618**, ECE = **6.49%**.
>    - **XGBoost**: Accuracy = **67.00%**, F1 = **0.6887**, ROC-AUC = **0.7411**, PR-AUC = **0.7322**, ECE = **2.28%**.
> 2. **Error Characteristics**:
>    - XGBoost demonstrated a higher hallucination detection recall ($73.00\%$ vs $66.25\%$, detecting 27 additional hallucinations) and higher specificity ($61.00\%$ vs $58.00\%$, producing 12 fewer false alarms) relative to Logistic Regression on this test split.
> 3. **Task-Dependent Efficacy**:
>    - Internal signals showed high discriminative power on context-grounded tasks (HaluEval ROC-AUC $> 0.93$), but struggled on adversarial misconceptions (TruthfulQA ROC-AUC $\sim 0.50\text{--}0.60$) and unconditioned claim verification (FEVER ROC-AUC $< 0.50$), indicating that internal confidence alone cannot overcome pre-training memorization of common false beliefs.
> 4. **Quarantine Success**:
>    - These baselines were established without reliance on sequence length (`num_tokens`), guaranteeing that performance reflects genuine probabilistic uncertainty rather than length artifacts.
