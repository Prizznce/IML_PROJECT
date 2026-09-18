# Universal Hallucination Detector — Ablation Study Summary

## 1. Experimental Conditions Overview

| Condition | Feature Group | Features | Count | Description |
| :---: | :--- | :--- | :---: | :--- |
| **A** | `internal_only` | Internal generation signals | 11 | Reference Baseline |
| **B** | `self_consistency_only` | Self-consistency signals | 5 | Semantic consensus across generations |
| **C** | `nli_only` | NLI agreement signals | 3 | Pairwise cross-generation logic |
| **D** | `internal_plus_sc` | Internal + Self-Consistency | 16 | Dual internal + sampling consistency |
| **E** | `internal_plus_nli` | Internal + NLI | 14 | Dual internal + cross-encoder logic |
| **F** | `sc_plus_nli` | Self-Consistency + NLI | 8 | Black-box behavioral probes only |
| **G** | `all_three` | Internal + SC + NLI | 19 | Universal Core Combined Detector |

---

## 2. Overall Model Performance and Deltas Relative to Reference Baseline (Condition A)

$$\Delta = \text{Ablation Metric} - \text{Internal Baseline Metric (Condition A)}$$

| Condition | Model | Feats | Accuracy ($\Delta$) | F1 ($\Delta$) | ROC-AUC ($\Delta$) | PR-AUC ($\Delta$) | Brier ($\Delta$) | ECE ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `internal_only` | logistic_regression | 11 | 0.6212 (+0.0000) | 0.6363 (+0.0000) | 0.6930 (+0.0000) | 0.6618 (+0.0000) | 0.2177 (+0.0000) | 0.0649 (+0.0000) |
| `internal_only` | xgboost | 11 | 0.6625 (+0.0000) | 0.6793 (+0.0000) | 0.7397 (+0.0000) | 0.7329 (+0.0000) | 0.1993 (+0.0000) | 0.0228 (+0.0000) |
| `self_consistency_only` | logistic_regression | 5 | 0.4900 (-0.1312) | 0.4545 (-0.1817) | 0.4845 (-0.2086) | 0.4863 (-0.1755) | 0.2515 (+0.0339) | 0.0300 (-0.0349) |
| `self_consistency_only` | xgboost | 5 | 0.5100 (-0.1525) | 0.5410 (-0.1384) | 0.4896 (-0.2501) | 0.4870 (-0.2459) | 0.2567 (+0.0574) | 0.0450 (+0.0222) |
| `nli_only` | logistic_regression | 3 | 0.5038 (-0.1175) | 0.5031 (-0.1331) | 0.5050 (-0.1881) | 0.5025 (-0.1593) | 0.2503 (+0.0326) | 0.0088 (-0.0561) |
| `nli_only` | xgboost | 3 | 0.4713 (-0.1912) | 0.4885 (-0.1908) | 0.4912 (-0.2485) | 0.5005 (-0.2324) | 0.2543 (+0.0550) | 0.0731 (+0.0504) |
| `internal_plus_self_consistency` | logistic_regression | 16 | 0.6200 (-0.0012) | 0.6311 (-0.0052) | 0.7036 (+0.0106) | 0.7043 (+0.0424) | 0.2159 (-0.0018) | 0.0613 (-0.0036) |
| `internal_plus_self_consistency` | xgboost | 16 | 0.6512 (-0.0112) | 0.6610 (-0.0183) | 0.7563 (+0.0166) | 0.7675 (+0.0346) | 0.1918 (-0.0075) | 0.0450 (+0.0222) |
| `internal_plus_nli` | logistic_regression | 14 | 0.6225 (+0.0013) | 0.6308 (-0.0054) | 0.7033 (+0.0102) | 0.6994 (+0.0376) | 0.2161 (-0.0016) | 0.0576 (-0.0072) |
| `internal_plus_nli` | xgboost | 14 | 0.6700 (+0.0075) | 0.6615 (-0.0178) | 0.7659 (+0.0261) | 0.7697 (+0.0368) | 0.1909 (-0.0084) | 0.0257 (+0.0030) |
| `self_consistency_plus_nli` | logistic_regression | 8 | 0.4750 (-0.1462) | 0.4697 (-0.1666) | 0.4874 (-0.2057) | 0.4795 (-0.1824) | 0.2516 (+0.0339) | 0.0475 (-0.0174) |
| `self_consistency_plus_nli` | xgboost | 8 | 0.5062 (-0.1562) | 0.5153 (-0.1640) | 0.5053 (-0.2344) | 0.5128 (-0.2201) | 0.2541 (+0.0548) | 0.0522 (+0.0294) |
| `all_three` | logistic_regression | 19 | 0.6288 (+0.0075) | 0.6356 (-0.0007) | 0.7079 (+0.0149) | 0.7124 (+0.0505) | 0.2151 (-0.0026) | 0.0525 (-0.0124) |
| `all_three` | xgboost | 19 | 0.6787 (+0.0162) | 0.6751 (-0.0042) | 0.7688 (+0.0291) | 0.7795 (+0.0466) | 0.1891 (-0.0102) | 0.0341 (+0.0113) |

---

## 3. Methodological Observations

- **Mathematical Redundancy**: `mean_log_prob` and `log_perplexity` maintain exact collinearity ($r = -1.0$) across all internal signal conditions without numerical instability.
- **Behavioral Probes Standalone Capacity**: Condition F (`sc_plus_nli`, 8 features) operates strictly on generated outputs without accessing model weights or internal logits, providing a benchmark for black-box detection settings.
- **Partition Integrity**: Exactly 3,200 train and 800 test instances with zero ID overlap across all 7 conditions.
- **Non-Causality Disclaimer**: Metric shifts reflect empirical predictive associations within this evaluation benchmark and do not indicate causal mechanisms of hallucination generation.
