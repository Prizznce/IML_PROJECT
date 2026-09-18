# Universal Hallucination Detector — Feature Family Ablation Study (Phase 15)

## 1. Executive Summary

This document reports the experimental methodology, quantitative findings, and diagnostic insights from a systematically controlled ablation study of the **Universal Core Hallucination Detector**.

The investigation evaluates the individual and combined predictive utility of three major signal families across 7 structured conditions:
1. **Internal Generation Signals (11 features)**: Log probabilities, entropy, perplexity, and token rank statistics computed directly on the evaluated response.
2. **Self-Consistency Signals (5 features)**: Exact match consensus and pairwise embedding cosine similarity distributions across 5 stochastically sampled generations.
3. **NLI Agreement Signals (3 features)**: Cross-encoder pairwise entailment, contradiction, and calibrated disagreement across all 10 generation pairs.

---

## 2. Experimental Protocol & Controls

To guarantee unconfounded, scientifically rigorous comparability across all conditions:
- **Exact Data Partitioning**: All 7 conditions were evaluated on the **exact same 800-sample holdout test partition** with `random_state=42` and joint stratification by `source_dataset + "_" + label`.
- **Sample Sizes**: 3,200 training examples (1,600 label 0; 1,600 label 1) and 800 test examples (400 label 0; 400 label 1).
- **Subgroup Composition**: HaluEval ($N=300$), TruthfulQA ($N=300$), FEVER ($N=200$).
- **No Test-Set Fitting**: All feature scalers in the Logistic Regression pipeline were fitted strictly on the 3,200 training instances.
- **Reference Baseline**: Condition A (**Internal Signals ONLY**, 11 features) serves as the canonical reference condition against which all deltas are calculated:
  $$\Delta = \text{Ablation Metric} - \text{Internal Baseline Metric (Condition A)}$$
- **Non-Causality Principle**: Feature-target correlations reflect predictive associations within this benchmark corpus. Standalone performance does not imply that any signal family causes hallucination.

---

## 3. The Seven Ablation Conditions

| Condition | Feature Group | Features Included | Feat Count | Description |
| :---: | :--- | :--- | :---: | :--- |
| **A** | `internal_only` | 11 internal generation signals | 11 | **Reference Baseline** (white-box token signals) |
| **B** | `self_consistency_only` | 5 similarity & agreement signals | 5 | Semantic consensus across stochastically sampled outputs |
| **C** | `nli_only` | 3 NLI agreement signals | 3 | Pairwise logical cross-encoder inference |
| **D** | `internal_plus_sc` | 11 internal + 5 self-consistency | 16 | Hybrid token-confidence and generation clustering |
| **E** | `internal_plus_nli` | 11 internal + 3 NLI agreement | 14 | Hybrid token-confidence and cross-encoder logic |
| **F** | `sc_plus_nli` | 5 self-consistency + 3 NLI agreement | 8 | **Black-Box behavioral probes only** (no internal logits) |
| **G** | `all_three` | 11 internal + 5 self-consistency + 3 NLI | 19 | **Universal Core Detector** (all signal groups) |

---

## 4. Overall Empirical Results & Baseline Comparison

Evaluation metrics on the identical 800-instance holdout test set across all 7 conditions:

| Condition | Model | Feats | Accuracy ($\Delta$) | F1 ($\Delta$) | ROC-AUC ($\Delta$) | PR-AUC ($\Delta$) | Brier ($\Delta$) | ECE ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Internal ONLY (Ref)** | Logistic Regression | 11 | 0.6212 (+0.0000) | 0.6363 (+0.0000) | 0.6930 (+0.0000) | 0.6618 (+0.0000) | 0.2177 (+0.0000) | 0.0649 (+0.0000) |
| | XGBoost | 11 | 0.6625 (+0.0000) | 0.6793 (+0.0000) | 0.7397 (+0.0000) | 0.7329 (+0.0000) | 0.1993 (+0.0000) | 0.0228 (+0.0000) |
| **B. Self-Consistency ONLY** | Logistic Regression | 5 | 0.4900 (-0.1312) | 0.4545 (-0.1817) | 0.4845 (-0.2086) | 0.4863 (-0.1755) | 0.2515 (+0.0339) | 0.0300 (-0.0349) |
| | XGBoost | 5 | 0.5100 (-0.1525) | 0.5410 (-0.1384) | 0.4896 (-0.2501) | 0.4870 (-0.2459) | 0.2567 (+0.0574) | 0.0450 (+0.0222) |
| **C. NLI ONLY** | Logistic Regression | 3 | 0.5038 (-0.1175) | 0.5031 (-0.1331) | 0.5050 (-0.1881) | 0.5025 (-0.1593) | 0.2503 (+0.0326) | 0.0088 (-0.0561) |
| | XGBoost | 3 | 0.4713 (-0.1912) | 0.4885 (-0.1908) | 0.4912 (-0.2485) | 0.5005 (-0.2324) | 0.2543 (+0.0550) | 0.0731 (+0.0504) |
| **D. Internal + SC** | Logistic Regression | 16 | 0.6200 (-0.0012) | 0.6311 (-0.0052) | 0.7036 (+0.0106) | 0.7043 (+0.0424) | 0.2159 (-0.0018) | 0.0613 (-0.0036) |
| | XGBoost | 16 | 0.6512 (-0.0112) | 0.6610 (-0.0183) | 0.7563 (+0.0166) | 0.7675 (+0.0346) | 0.1918 (-0.0075) | 0.0450 (+0.0222) |
| **E. Internal + NLI** | Logistic Regression | 14 | 0.6225 (+0.0013) | 0.6308 (-0.0054) | 0.7033 (+0.0102) | 0.6994 (+0.0376) | 0.2161 (-0.0016) | 0.0576 (-0.0072) |
| | XGBoost | 14 | 0.6700 (+0.0075) | 0.6615 (-0.0178) | 0.7659 (+0.0261) | 0.7697 (+0.0368) | 0.1909 (-0.0084) | 0.0257 (+0.0030) |
| **F. SC + NLI (Black-Box)** | Logistic Regression | 8 | 0.4750 (-0.1462) | 0.4697 (-0.1666) | 0.4874 (-0.2057) | 0.4795 (-0.1824) | 0.2516 (+0.0339) | 0.0475 (-0.0174) |
| | XGBoost | 8 | 0.5062 (-0.1562) | 0.5153 (-0.1640) | 0.5053 (-0.2344) | 0.5128 (-0.2201) | 0.2541 (+0.0548) | 0.0522 (+0.0294) |
| **G. ALL THREE (Universal)** | Logistic Regression | 19 | 0.6288 (+0.0075) | 0.6356 (-0.0007) | 0.7079 (+0.0149) | 0.7124 (+0.0505) | 0.2151 (-0.0026) | 0.0525 (-0.0124) |
| | XGBoost | 19 | 0.6787 (+0.0162) | 0.6751 (-0.0042) | 0.7688 (+0.0291) | 0.7795 (+0.0466) | 0.1891 (-0.0102) | 0.0341 (+0.0113) |

---

## 5. Key Empirical Observations

### 1. Standalone Capacity of Behavioral Sampling Probes (Conditions B, C, and F)
- When evaluated without internal model signals, neither self-consistency (Condition B: ROC-AUC $0.4896$ for XGBoost) nor NLI agreement (Condition C: ROC-AUC $0.4912$) achieved global discrimination above chance level on the heterogeneous 3-task benchmark.
- Combining both behavioral probe families in black-box Condition F yields ROC-AUC $0.5053$.
- *Insight*: Across diverse tasks, sampling consistency alone does not indicate correctness because language models can consistently hallucinate plausible falsehoods (particularly on TruthfulQA).

### 2. Synergy When Combined with Internal Signals (Conditions D, E, and G)
- Although behavioral probes exhibit low standalone discrimination globally, combining them with internal generation signals yields measurable improvements in threshold-free ranking metrics (ROC-AUC and PR-AUC) and calibration:
  - **Internal + SC (D)**: XGBoost ROC-AUC increases by $+0.0166$ (to $0.7563$), PR-AUC increases by $+0.0346$.
  - **Internal + NLI (E)**: XGBoost ROC-AUC increases by $+0.0261$ (to $0.7659$), PR-AUC increases by $+0.0368$.
  - **All Three (G)**: XGBoost ROC-AUC increases by $+0.0291$ (to $0.7688$), PR-AUC increases by $+0.0466$ (to $0.7795$), with Brier score reducing from $0.1993$ to $0.1891$ ($\Delta = -0.0102$).

### 3. Task-Level Divergence Across Benchmark Subgroups
- **HaluEval ($N=300$)**:
  - Strongly driven by internal signals: Internal-only XGBoost achieves Accuracy $0.8633$ and ROC-AUC $0.9428$.
  - All-three combination achieves Accuracy $0.8633$ and ROC-AUC $0.9433$.
- **TruthfulQA ($N=300$)**:
  - Internal signals alone achieve moderate discrimination (XGBoost Accuracy $0.5833$, ROC-AUC $0.5963$).
  - Adding consistency and NLI (Condition G) shifts Accuracy to $0.5933$ ($\Delta = +0.0100$) and ROC-AUC to $0.6066$ ($\Delta = +0.0103$).
- **FEVER ($N=200$)**:
  - Standalone NLI (Condition C) achieves ROC-AUC $0.6177$ on Logistic Regression and $0.5608$ on XGBoost, showing that logical cross-checking has domain-specific relevance for claim verification.
  - Adding NLI to Internal signals (Condition E) elevates XGBoost ROC-AUC from $0.4518$ to $0.5361$ ($\Delta = +0.0843$), and in Condition G achieves $0.5441$ ($\Delta = +0.0923$).

---

## 6. Collinearity and Redundancy Audit

As documented in Phase 13, `mean_log_prob` and `log_perplexity` exhibit exact inverse linear collinearity:
$$\text{Perplexity} = \exp(-\text{mean\_log\_prob}) \implies \log(\text{Perplexity}) = -\text{mean\_log\_prob} \implies r = -1.000000$$

Both features were retained in all internal-signal conditions (A, D, E, G) in compliance with the predefined specification. Neither Logistic Regression (stabilized by $L_2$ regularization) nor XGBoost exhibited numerical instability or optimization divergence.

---

## 7. Artifact Manifest

- **Ablation Pipeline**: [src/evaluation/ablation_study.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/evaluation/ablation_study.py)
- **Unit & Integration Tests**: [tests/test_ablation_study.py](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/tests/test_ablation_study.py)
- **Full Results JSON**: [ablation_results.json](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/ablation/ablation_results.json)
- **Overall Metrics CSV**: [ablation_results.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/ablation/ablation_results.csv)
- **Per-Dataset CSV**: [ablation_per_dataset.csv](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/ablation/ablation_per_dataset.csv)
- **Test Predictions Parquet**: [ablation_test_predictions.parquet](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/ablation/ablation_test_predictions.parquet)
- **Summary Report**: [ablation_summary.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/ablation/ablation_summary.md)
