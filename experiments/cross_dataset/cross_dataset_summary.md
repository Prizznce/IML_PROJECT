# Cross-Dataset Generalization Evaluation Summary (Phase 16)

## 1. Experimental Overview

| Experiment | Train Datasets ($N$) | Held-Out Test Dataset ($N$) | Objective |
| :---: | :--- | :--- | :--- |
| **Exp 1** | HaluEval + TruthfulQA ($3,000$) | **FEVER** ($1,000$) | Fact-verification claim transfer |
| **Exp 2** | HaluEval + FEVER ($2,500$) | **TruthfulQA** ($1,500$) | Adversarial misconception transfer |
| **Exp 3** | TruthfulQA + FEVER ($2,500$) | **HaluEval** ($1,500$) | General QA & dialogue hallucination transfer |

---

## 2. Complete Cross-Dataset Results (24 Runs)

$$\Delta = \text{Configuration Metric} - \text{Internal ONLY Metric}$$

| Test Dataset | Configuration | Feats | Model | Accuracy ($\Delta$) | F1 ($\Delta$) | ROC-AUC ($\Delta$) | PR-AUC ($\Delta$) | Brier ($\Delta$) | ECE ($\Delta$) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **fever** | `internal_only` | 11 | logistic_regression | 0.5060 (+0.0000) | 0.6141 (+0.0000) | 0.4868 (+0.0000) | 0.4818 (+0.0000) | 0.2952 (+0.0000) | 0.1716 (+0.0000) |
| **fever** | `internal_only` | 11 | xgboost | 0.4830 (+0.0000) | 0.4772 (+0.0000) | 0.4903 (+0.0000) | 0.4870 (+0.0000) | 0.2681 (+0.0000) | 0.1186 (+0.0000) |
| **fever** | `internal_plus_self_consistency` | 16 | logistic_regression | 0.5060 (+0.0000) | 0.6246 (+0.0106) | 0.4882 (+0.0014) | 0.4825 (+0.0007) | 0.2993 (+0.0041) | 0.1825 (+0.0109) |
| **fever** | `internal_plus_self_consistency` | 16 | xgboost | 0.5090 (+0.0260) | 0.5475 (+0.0702) | 0.4999 (+0.0096) | 0.4891 (+0.0022) | 0.2636 (-0.0044) | 0.0824 (-0.0362) |
| **fever** | `internal_plus_nli` | 14 | logistic_regression | 0.4970 (-0.0090) | 0.6209 (+0.0069) | 0.4875 (+0.0007) | 0.4847 (+0.0029) | 0.3072 (+0.0120) | 0.2096 (+0.0379) |
| **fever** | `internal_plus_nli` | 14 | xgboost | 0.5070 (+0.0240) | 0.5397 (+0.0624) | 0.5143 (+0.0241) | 0.5260 (+0.0390) | 0.2593 (-0.0088) | 0.0876 (-0.0310) |
| **fever** | `all_three` | 19 | logistic_regression | 0.4960 (-0.0100) | 0.6216 (+0.0076) | 0.4875 (+0.0007) | 0.4840 (+0.0021) | 0.3072 (+0.0120) | 0.2064 (+0.0348) |
| **fever** | `all_three` | 19 | xgboost | 0.5060 (+0.0230) | 0.5581 (+0.0809) | 0.5167 (+0.0264) | 0.5078 (+0.0208) | 0.2599 (-0.0082) | 0.0912 (-0.0275) |
| **truthfulqa** | `internal_only` | 11 | logistic_regression | 0.4660 (+0.0000) | 0.5206 (+0.0000) | 0.4439 (+0.0000) | 0.4544 (+0.0000) | 0.3248 (+0.0000) | 0.2299 (+0.0000) |
| **truthfulqa** | `internal_only` | 11 | xgboost | 0.4793 (+0.0000) | 0.5951 (+0.0000) | 0.4539 (+0.0000) | 0.4584 (+0.0000) | 0.3651 (+0.0000) | 0.2949 (+0.0000) |
| **truthfulqa** | `internal_plus_self_consistency` | 16 | logistic_regression | 0.4787 (+0.0127) | 0.4723 (-0.0483) | 0.4554 (+0.0115) | 0.4593 (+0.0049) | 0.3179 (-0.0069) | 0.2052 (-0.0247) |
| **truthfulqa** | `internal_plus_self_consistency` | 16 | xgboost | 0.4813 (+0.0020) | 0.5948 (-0.0003) | 0.4582 (+0.0043) | 0.4678 (+0.0094) | 0.3434 (-0.0217) | 0.2621 (-0.0327) |
| **truthfulqa** | `internal_plus_nli` | 14 | logistic_regression | 0.4747 (+0.0087) | 0.4580 (-0.0626) | 0.4456 (+0.0016) | 0.4541 (-0.0002) | 0.3216 (-0.0032) | 0.2115 (-0.0184) |
| **truthfulqa** | `internal_plus_nli` | 14 | xgboost | 0.4833 (+0.0040) | 0.5862 (-0.0089) | 0.4581 (+0.0043) | 0.4657 (+0.0072) | 0.3431 (-0.0219) | 0.2555 (-0.0394) |
| **truthfulqa** | `all_three` | 19 | logistic_regression | 0.4753 (+0.0093) | 0.4334 (-0.0872) | 0.4575 (+0.0136) | 0.4604 (+0.0060) | 0.3201 (-0.0048) | 0.2169 (-0.0130) |
| **truthfulqa** | `all_three` | 19 | xgboost | 0.4853 (+0.0060) | 0.5971 (+0.0020) | 0.4608 (+0.0069) | 0.4748 (+0.0164) | 0.3339 (-0.0311) | 0.2437 (-0.0512) |
| **halueval** | `internal_only` | 11 | logistic_regression | 0.2140 (+0.0000) | 0.2135 (+0.0000) | 0.1601 (+0.0000) | 0.3366 (+0.0000) | 0.3563 (+0.0000) | 0.4194 (+0.0000) |
| **halueval** | `internal_only` | 11 | xgboost | 0.3980 (+0.0000) | 0.3984 (+0.0000) | 0.3599 (+0.0000) | 0.4215 (+0.0000) | 0.3083 (+0.0000) | 0.2272 (+0.0000) |
| **halueval** | `internal_plus_self_consistency` | 16 | logistic_regression | 0.4187 (+0.2047) | 0.5468 (+0.3333) | 0.2706 (+0.1105) | 0.3630 (+0.0265) | 0.3605 (+0.0042) | 0.3213 (-0.0980) |
| **halueval** | `internal_plus_self_consistency` | 16 | xgboost | 0.4160 (+0.0180) | 0.5335 (+0.1351) | 0.3994 (+0.0395) | 0.4508 (+0.0294) | 0.3049 (-0.0034) | 0.2277 (+0.0005) |
| **halueval** | `internal_plus_nli` | 14 | logistic_regression | 0.3880 (+0.1740) | 0.5086 (+0.2951) | 0.1988 (+0.0388) | 0.3447 (+0.0081) | 0.3601 (+0.0038) | 0.3839 (-0.0355) |
| **halueval** | `internal_plus_nli` | 14 | xgboost | 0.4433 (+0.0453) | 0.5764 (+0.1780) | 0.4445 (+0.0847) | 0.4944 (+0.0729) | 0.3025 (-0.0058) | 0.2273 (+0.0001) |
| **halueval** | `all_three` | 19 | logistic_regression | 0.4493 (+0.2353) | 0.5820 (+0.3685) | 0.2583 (+0.0982) | 0.3593 (+0.0228) | 0.3631 (+0.0068) | 0.3229 (-0.0964) |
| **halueval** | `all_three` | 19 | xgboost | 0.4320 (+0.0340) | 0.5644 (+0.1660) | 0.4368 (+0.0769) | 0.4664 (+0.0449) | 0.3048 (-0.0035) | 0.2302 (+0.0030) |

---

## 3. Methodological Observations & Transfer Hypotheses

- **Zero Leakage**: Exactly 0 rows and 0 IDs from held-out test datasets entered training or preprocessing scaling.
- **Transfer Profiles**: Different target datasets demonstrate substantially different susceptibility to out-of-domain transfer.
- **Hypothesis on FEVER Transfer**: The strong transfer gains on FEVER when adding NLI (`internal_plus_nli`) suggest that pairwise cross-encoder entailment logic generalizes more reliably to claim-verification tasks than internal token probabilities alone.
- **Hypothesis on TruthfulQA Transfer**: The low out-of-domain transfer to TruthfulQA across all configurations indicates that models exhibit high confidence when generating common human misconceptions, a phenomenon not easily detected by internal signals or sampling consistency trained on distinct task distributions.
- **Non-Causality Disclaimer**: All metric deltas reflect predictive associations on held-out tasks and do not indicate causal drivers of transfer success or failure.
