# Universal Cross-Dataset Generalization Evaluation (Phase 16)

## 1. Executive Summary

This document presents the methodology, quantitative findings, and diagnostic hypotheses for the **Leave-One-Dataset-Out (LODO) Cross-Dataset Generalization Study** of the Universal Core Hallucination Detector.

The objective is to evaluate how effectively supervised hallucination detection signals transfer to completely unseen task distributions without domain-specific fine-tuning or target-domain adaptation.

The evaluation investigates 3 Leave-One-Dataset-Out experiments across 4 nested feature configurations and 2 model architectures, yielding 24 full training and evaluation runs.

---

## 2. Experimental Protocol & Partition Invariance

### Zero-Leakage Leave-One-Dataset-Out (LODO) Design
1. **Experiment 1 (Held-out FEVER)**:
   - **Training Set**: HaluEval + TruthfulQA ($N = 1,500 + 1,500 = 3,000$).
   - **Test Set**: FEVER ($N = 1,000$).
2. **Experiment 2 (Held-out TruthfulQA)**:
   - **Training Set**: HaluEval + FEVER ($N = 1,500 + 1,000 = 2,500$).
   - **Test Set**: TruthfulQA ($N = 1,500$).
3. **Experiment 3 (Held-out HaluEval)**:
   - **Training Set**: TruthfulQA + FEVER ($N = 1,500 + 1,000 = 2,500$).
   - **Test Set**: HaluEval ($N = 1,500$).

### Partition Verification
- Zero random splitting: training partitions contain 100% of rows from the two training datasets, and test partitions contain 100% of rows from the held-out dataset.
- Exact sample counts: Exp 1 (Train=3,000, Test=1,000); Exp 2 (Train=2,500, Test=1,500); Exp 3 (Train=2,500, Test=1,500).
- Zero ID overlap: $\text{len}(\text{train\_ids} \cap \text{test\_ids}) = 0$ across all 3 experiments.
- Preprocessing isolation: `StandardScaler` in the Logistic Regression pipeline is fit **exclusively on the training partition**. The held-out test dataset is never seen during scaling or model fitting.

---

## 3. Four Feature Configurations

To isolate which signal families facilitate or hinder cross-domain transfer, every experiment evaluates:

| Config ID | Key | Features Included | Feat Count | Role |
| :---: | :--- | :--- | :---: | :--- |
| **A** | `internal_only` | 11 internal token probability, entropy, rank signals | 11 | **Reference Baseline** |
| **B** | `internal_plus_self_consistency` | 11 internal + 5 self-consistency similarity signals | 16 | Hybrid token + sampling consensus |
| **C** | `internal_plus_nli` | 11 internal + 3 NLI agreement signals | 14 | Hybrid token + logical entailment |
| **D** | `all_three` | 11 internal + 5 self-consistency + 3 NLI signals | 19 | Universal Core Detector |

All deltas are computed relative to the internal reference baseline:
$$\Delta = \text{Configuration Metric} - \text{Internal ONLY Metric}$$

---

## 4. Complete Cross-Dataset Empirical Results (24 Runs)

| Test Dataset | Configuration | Feats | Model | Accuracy ($\Delta$) | F1 ($\Delta$) | ROC-AUC ($\Delta$) | PR-AUC ($\Delta$) | Brier ($\Delta$) | ECE ($\Delta$) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FEVER** ($N=1,000$) | `internal_only` | 11 | Logistic Regression | 0.5060 (+0.0000) | 0.6141 (+0.0000) | 0.4868 (+0.0000) | 0.4818 (+0.0000) | 0.2952 (+0.0000) | 0.1716 (+0.0000) |
| | `internal_only` | 11 | XGBoost | 0.4830 (+0.0000) | 0.4772 (+0.0000) | 0.4903 (+0.0000) | 0.4870 (+0.0000) | 0.2681 (+0.0000) | 0.1186 (+0.0000) |
| | `internal_plus_sc` | 16 | Logistic Regression | 0.5060 (+0.0000) | 0.6246 (+0.0106) | 0.4882 (+0.0014) | 0.4825 (+0.0007) | 0.2993 (+0.0041) | 0.1825 (+0.0109) |
| | `internal_plus_sc` | 16 | XGBoost | 0.5090 (+0.0260) | 0.5475 (+0.0702) | 0.4999 (+0.0096) | 0.4891 (+0.0022) | 0.2636 (-0.0044) | 0.0824 (-0.0362) |
| | `internal_plus_nli` | 14 | Logistic Regression | 0.4970 (-0.0090) | 0.6209 (+0.0069) | 0.4875 (+0.0007) | 0.4847 (+0.0029) | 0.3072 (+0.0120) | 0.2096 (+0.0379) |
| | `internal_plus_nli` | 14 | XGBoost | 0.5070 (+0.0240) | 0.5397 (+0.0624) | 0.5143 (+0.0241) | 0.5260 (+0.0390) | 0.2593 (-0.0088) | 0.0876 (-0.0310) |
| | `all_three` | 19 | Logistic Regression | 0.4960 (-0.0100) | 0.6216 (+0.0076) | 0.4875 (+0.0007) | 0.4840 (+0.0021) | 0.3072 (+0.0120) | 0.2064 (+0.0348) |
| | `all_three` | 19 | XGBoost | 0.5060 (+0.0230) | 0.5581 (+0.0809) | 0.5167 (+0.0264) | 0.5078 (+0.0208) | 0.2599 (-0.0082) | 0.0912 (-0.0275) |
| **TruthfulQA** ($N=1,500$) | `internal_only` | 11 | Logistic Regression | 0.4660 (+0.0000) | 0.5206 (+0.0000) | 0.4439 (+0.0000) | 0.4544 (+0.0000) | 0.3248 (+0.0000) | 0.2299 (+0.0000) |
| | `internal_only` | 11 | XGBoost | 0.4793 (+0.0000) | 0.5951 (+0.0000) | 0.4539 (+0.0000) | 0.4584 (+0.0000) | 0.3651 (+0.0000) | 0.2949 (+0.0000) |
| | `internal_plus_sc` | 16 | Logistic Regression | 0.4787 (+0.0127) | 0.4723 (-0.0483) | 0.4554 (+0.0115) | 0.4593 (+0.0049) | 0.3179 (-0.0069) | 0.2052 (-0.0247) |
| | `internal_plus_sc` | 16 | XGBoost | 0.4813 (+0.0020) | 0.5948 (-0.0003) | 0.4582 (+0.0043) | 0.4678 (+0.0094) | 0.3434 (-0.0217) | 0.2621 (-0.0327) |
| | `internal_plus_nli` | 14 | Logistic Regression | 0.4747 (+0.0087) | 0.4580 (-0.0626) | 0.4456 (+0.0016) | 0.4541 (-0.0002) | 0.3216 (-0.0032) | 0.2115 (-0.0184) |
| | `internal_plus_nli` | 14 | XGBoost | 0.4833 (+0.0040) | 0.5862 (-0.0089) | 0.4581 (+0.0043) | 0.4657 (+0.0072) | 0.3431 (-0.0219) | 0.2555 (-0.0394) |
| | `all_three` | 19 | Logistic Regression | 0.4753 (+0.0093) | 0.4334 (-0.0872) | 0.4575 (+0.0136) | 0.4604 (+0.0060) | 0.3201 (-0.0048) | 0.2169 (-0.0130) |
| | `all_three` | 19 | XGBoost | 0.4853 (+0.0060) | 0.5971 (+0.0020) | 0.4608 (+0.0069) | 0.4748 (+0.0164) | 0.3339 (-0.0311) | 0.2437 (-0.0512) |
| **HaluEval** ($N=1,500$) | `internal_only` | 11 | Logistic Regression | 0.2140 (+0.0000) | 0.2135 (+0.0000) | 0.1601 (+0.0000) | 0.3366 (+0.0000) | 0.3563 (+0.0000) | 0.4194 (+0.0000) |
| | `internal_only` | 11 | XGBoost | 0.3980 (+0.0000) | 0.3984 (+0.0000) | 0.3599 (+0.0000) | 0.4215 (+0.0000) | 0.3083 (+0.0000) | 0.2272 (+0.0000) |
| | `internal_plus_sc` | 16 | Logistic Regression | 0.4187 (+0.2047) | 0.5468 (+0.3333) | 0.2706 (+0.1105) | 0.3630 (+0.0265) | 0.3605 (+0.0042) | 0.3213 (-0.0980) |
| | `internal_plus_sc` | 16 | XGBoost | 0.4160 (+0.0180) | 0.5335 (+0.1351) | 0.3994 (+0.0395) | 0.4508 (+0.0294) | 0.3049 (-0.0034) | 0.2277 (+0.0005) |
| | `internal_plus_nli` | 14 | Logistic Regression | 0.3880 (+0.1740) | 0.5086 (+0.2951) | 0.1988 (+0.0388) | 0.3447 (+0.0081) | 0.3601 (+0.0038) | 0.3839 (-0.0355) |
| | `internal_plus_nli` | 14 | XGBoost | 0.4433 (+0.0453) | 0.5764 (+0.1780) | 0.4445 (+0.0847) | 0.4944 (+0.0729) | 0.3025 (-0.0058) | 0.2273 (+0.0001) |
| | `all_three` | 19 | Logistic Regression | 0.4493 (+0.2353) | 0.5820 (+0.3685) | 0.2583 (+0.0982) | 0.3593 (+0.0228) | 0.3631 (+0.0068) | 0.3229 (-0.0964) |
| | `all_three` | 19 | XGBoost | 0.4320 (+0.0340) | 0.5644 (+0.1660) | 0.4368 (+0.0769) | 0.4664 (+0.0449) | 0.3048 (-0.0035) | 0.2302 (+0.0030) |

---

## 5. Dataset-Specific Transfer Observations & Working Hypotheses

### 1. FEVER ($N=1,000$) Transfer Analysis
- **Observed Behavior**:
  - `internal_only` yields near-chance discrimination (XGBoost ROC-AUC $0.4903$, Accuracy $0.4830$).
  - Incorporating NLI agreement (`internal_plus_nli`) improves XGBoost ROC-AUC by $+0.0241$ (to $0.5143$) and PR-AUC by $+0.0390$ (to $0.5260$).
  - In `all_three`, XGBoost ROC-AUC reaches $0.5167$ ($\Delta = +0.0264$), and Brier score improves from $0.2681$ to $0.2599$.
- **Hypothesis**:
  - *Fact Verification Alignment*: FEVER requires verifying whether a claim is supported or refuted by world knowledge. The pairwise NLI agreement signals evaluate logical contradiction and entailment between independently generated claims, which maps closely to FEVER's task structure, providing positive transfer signal that internal token probabilities alone lack.

### 2. TruthfulQA ($N=1,500$) Transfer Analysis
- **Observed Behavior**:
  - Across all four feature configurations, out-of-domain transfer to TruthfulQA remains below chance discrimination (ROC-AUC ranges from $0.4439$ to $0.4608$).
  - Adding consistency and NLI signals yields small positive increments in ROC-AUC (XGBoost: $+0.0043$ for SC, $+0.0043$ for NLI, $+0.0069$ for All Three) and decreases Brier error (from $0.3651$ to $0.3339$, $\Delta = -0.0311$).
- **Hypothesis**:
  - *Misconception Confidence Inversion*: TruthfulQA targets widespread human misconceptions and conspiracy theories. Models frequently generate these misconceptions with high internal confidence (low perplexity, high token probabilities) and repeat them consistently across stochastic samples. Consequently, models trained on HaluEval and FEVER (where hallucinations correlate with high uncertainty or contradiction) encounter inverted calibration when applied out-of-domain to TruthfulQA.

### 3. HaluEval ($N=1,500$) Transfer Analysis
- **Observed Behavior**:
  - When trained strictly on TruthfulQA + FEVER ($N=2,500$), the `internal_only` model suffers severe negative transfer on HaluEval (Logistic Regression Accuracy $0.2140$, ROC-AUC $0.1601$; XGBoost Accuracy $0.3980$, ROC-AUC $0.3599$).
  - Adding Self-Consistency (`internal_plus_sc`) substantially mitigates this failure: Logistic Regression Accuracy rises by $+0.2047$ (to $0.4187$) and F1 by $+0.3333$ (to $0.5468$).
  - Adding NLI (`internal_plus_nli`) yields the strongest recovery: XGBoost ROC-AUC increases by $+0.0847$ (from $0.3599$ to $0.4445$), and PR-AUC increases by $+0.0729$ (from $0.4215$ to $0.4944$).
  - In `all_three`, Logistic Regression accuracy improves by $+0.2353$ (to $0.4493$).
- **Hypothesis**:
  - *Signal Orthogonality and Inversion Damping*: In HaluEval, dialogue and QA hallucination patterns differ markedly in sequence structure and prompt length from short FEVER claims and TruthfulQA questions. Internal token statistics become miscalibrated across this distribution gap. Adding behavioral consensus and cross-encoder logic provides scale-invariant semantic signals that counteract internal feature distortion.

---

## 6. Methodological Integrity Checks

- **Zero Overlap Confirmed**: All training partitions strictly excluded the held-out dataset ($0$ rows in train).
- **Preprocessing Isolation**: `StandardScaler` was fit only on the two training datasets; no test set data was accessible during feature scaling.
- **No Forbidden Features**: Neither `source_dataset`, `id`, `prompt`, `response`, `context`, nor `num_tokens` entered the model feature matrix.
- **Numerical Validity**: All probabilities are strictly within $[0.0, 1.0]$ across all 32,000 predictions, and all computed metrics are finite.

---

## 7. Artifact Manifest

- **Evaluation Pipeline**: [src/evaluation/cross_dataset.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/evaluation/cross_dataset.py)
- **Module Interface**: [src/evaluation/\_\_init\_\_.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/evaluation/__init__.py)
- **Unit & Integration Tests**: [tests/test_cross_dataset.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/tests/test_cross_dataset.py)
- **Results JSON**: [cross_dataset_results.json](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/cross_dataset/cross_dataset_results.json)
- **Results CSV (24 runs)**: [cross_dataset_results.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/cross_dataset/cross_dataset_results.csv)
- **Per-Dataset CSV (24 runs)**: [cross_dataset_per_dataset.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/cross_dataset/cross_dataset_per_dataset.csv)
- **Predictions Parquet (32,000 rows)**: [cross_dataset_predictions.parquet](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/cross_dataset/cross_dataset_predictions.parquet)
- **Summary Report**: [cross_dataset_summary.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/cross_dataset/cross_dataset_summary.md)
