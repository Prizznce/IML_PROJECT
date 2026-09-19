# Master Results Table: Multi-Signal Hallucination Detection

This document provides the authoritative, unified master results synthesis across all empirical experiments completed in **Phases 3–19** of the Hallucination Detection research pipeline. All reported metrics represent exact, validated outputs extracted directly from persisted experimental artifacts without modification or re-estimation.

---

## 1. Primary Benchmark

The **Phase 14 Universal Core Detector** serves as the primary scientific benchmark, evaluated on the fixed, stratified 80/20 partition ($3,200$ train / $800$ holdout test, `random_state=42`) across the full $N = 4,000$ combined benchmark. It is compared directly against the **Phase 6 Baseline** (11 internal generation token signals) evaluated on the identical holdout partition.

| Experiment / Pipeline | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Phase 6 Baseline (11 features) | Logistic Regression | 0.6212 | 0.6120 | 0.6625 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| Phase 6 Baseline (11 features) | XGBoost | 0.6700 | 0.6518 | 0.7300 | 0.6887 | 0.7411 | 0.7322 | 0.1999 | 0.0228 |
| Phase 14 Universal Core (19 features) | Logistic Regression | 0.6288 | 0.6241 | 0.6475 | 0.6356 | 0.7079 | 0.7124 | 0.2151 | 0.0525 |
| Phase 14 Universal Core (19 features) | XGBoost | 0.6788 | 0.6829 | 0.6675 | 0.6751 | 0.7688 | 0.7795 | 0.1891 | 0.0341 |

### Comparative Observations
- **XGBoost Improvements**: Incorporating behavioral self-consistency ($k=5$) and cross-encoder NLI agreement signals into the 11 baseline internal signals increases XGBoost ROC-AUC from `0.7411` to `0.7688` ($+0.0277$) and PR-AUC from `0.7322` to `0.7795` ($+0.0474$).
- **Logistic Regression Improvements**: Logistic Regression ROC-AUC increases from `0.6930` to `0.7079` ($+0.0149$) and PR-AUC increases from `0.6618` to `0.7124` ($+0.0505$).
- **Calibration Refinement**: Brier scores improve across both models (`0.2177` $\to$ `0.2151` for LR; `0.1999` $\to$ `0.1891` for XGBoost).

---

## 2. Feature-Family Ablation

Evaluated across the identical 800-example Phase 14 holdout test set to isolate the individual, pairwise, and combined contributions of the three primary signal families:
- **Group 1: Internal Signals (11)**: Token log-probabilities, entropy, perplexity, token ranks from `Qwen3.5-0.8B`.
- **Group 2: Self-Consistency (5)**: Exact match agreement, pairwise embedding cosine similarity statistics ($k=5$).
- **Group 3: NLI Agreement (3)**: Bidirectional pairwise entailment, contradiction, and disagreement fractions via `cross-encoder/nli-MiniLM2-L6-H768`.

| Feature Configuration | Features | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Internal (11) | 11 | Logistic Regression | 0.6212 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| Internal (11) | 11 | XGBoost | 0.6625 | 0.6793 | 0.7397 | 0.7329 | 0.1993 | 0.0228 |
| Self-Consistency (5) | 5 | Logistic Regression | 0.4900 | 0.4545 | 0.4845 | 0.4863 | 0.2515 | 0.0300 |
| Self-Consistency (5) | 5 | XGBoost | 0.5100 | 0.5410 | 0.4896 | 0.4870 | 0.2567 | 0.0450 |
| NLI (3) | 3 | Logistic Regression | 0.5038 | 0.5031 | 0.5050 | 0.5025 | 0.2503 | 0.0088 |
| NLI (3) | 3 | XGBoost | 0.4713 | 0.4885 | 0.4912 | 0.5005 | 0.2543 | 0.0731 |
| Internal + Self-Consistency (16) | 16 | Logistic Regression | 0.6200 | 0.6311 | 0.7036 | 0.7043 | 0.2159 | 0.0613 |
| Internal + Self-Consistency (16) | 16 | XGBoost | 0.6512 | 0.6610 | 0.7563 | 0.7675 | 0.1918 | 0.0450 |
| Internal + NLI (14) | 14 | Logistic Regression | 0.6225 | 0.6308 | 0.7033 | 0.6994 | 0.2161 | 0.0576 |
| Internal + NLI (14) | 14 | XGBoost | 0.6700 | 0.6615 | 0.7659 | 0.7697 | 0.1909 | 0.0257 |
| Self-Consistency + NLI (8) | 8 | Logistic Regression | 0.4750 | 0.4697 | 0.4874 | 0.4795 | 0.2516 | 0.0475 |
| Self-Consistency + NLI (8) | 8 | XGBoost | 0.5062 | 0.5153 | 0.5053 | 0.5128 | 0.2541 | 0.0522 |
| All Three / Universal Core (19) | 19 | Logistic Regression | 0.6288 | 0.6356 | 0.7079 | 0.7124 | 0.2151 | 0.0525 |
| All Three / Universal Core (19) | 19 | XGBoost | 0.6788 | 0.6751 | 0.7688 | 0.7795 | 0.1891 | 0.0341 |

### Key Ablation Insights
1. **Internal Signals as Backbone**: Standalone internal token signals achieve ROC-AUC `0.7397` (XGBoost) and `0.6930` (LR), providing the primary foundation of predictive power.
2. **Behavioral Probes Require Grounding**: Standalone self-consistency (`0.4896`) and NLI (`0.4912`) hover near random guessing when ungrounded by internal certainty signals.
3. **Synergistic Multi-Modal Gain**: When combined with internal signals, behavioral probes provide orthogonal discrimination, elevating XGBoost ROC-AUC to `0.7688` and PR-AUC to `0.7795`.

---

## 3. Cross-Dataset Generalization

Evaluated using a rigorous Leave-One-Dataset-Out (LODO) paradigm where models are trained exclusively on two datasets and tested on the 100% unseen third dataset, measuring out-of-domain transferability.

| Test Dataset (Held-Out) | Training Datasets | Feature Configuration | Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal (11) | Logistic Regression | 0.5060 | 0.4868 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal (11) | XGBoost | 0.4830 | 0.4903 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal + Self-Consistency (16) | Logistic Regression | 0.5060 | 0.4882 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal + Self-Consistency (16) | XGBoost | 0.5090 | 0.4999 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal + NLI (14) | Logistic Regression | 0.4970 | 0.4875 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal + NLI (14) | XGBoost | 0.5070 | 0.5143 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | All Three / Universal Core (19) | Logistic Regression | 0.4960 | 0.4875 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | All Three / Universal Core (19) | XGBoost | 0.5060 | 0.5167 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal (11) | Logistic Regression | 0.4660 | 0.4439 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal (11) | XGBoost | 0.4793 | 0.4539 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal + Self-Consistency (16) | Logistic Regression | 0.4787 | 0.4554 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal + Self-Consistency (16) | XGBoost | 0.4813 | 0.4582 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal + NLI (14) | Logistic Regression | 0.4747 | 0.4456 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal + NLI (14) | XGBoost | 0.4833 | 0.4581 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | All Three / Universal Core (19) | Logistic Regression | 0.4753 | 0.4575 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | All Three / Universal Core (19) | XGBoost | 0.4853 | 0.4608 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal (11) | Logistic Regression | 0.2140 | 0.1601 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal (11) | XGBoost | 0.3980 | 0.3599 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal + Self-Consistency (16) | Logistic Regression | 0.4187 | 0.2706 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal + Self-Consistency (16) | XGBoost | 0.4160 | 0.3994 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal + NLI (14) | Logistic Regression | 0.3880 | 0.1988 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal + NLI (14) | XGBoost | 0.4433 | 0.4445 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | All Three / Universal Core (19) | Logistic Regression | 0.4493 | 0.2583 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | All Three / Universal Core (19) | XGBoost | 0.4320 | 0.4368 |

### Methodological Observations
- **Substantial Domain Invariance Challenge**: Zero-shot transfer across distinct benchmark distributions exhibits severe performance degradation, confirming that decision thresholds learned on one task formulation (e.g., entity factoid verification vs passage summarization) do not naively generalize without multi-benchmark exposure.
- **Multi-Signal Buffering**: In the hardest transfer setting (testing on HaluEval after training on TruthfulQA + FEVER), adding behavioral signals improves XGBoost ROC-AUC from `0.3599` to `0.4368` and accuracy from `0.3980` to `0.4320`.

---

## 4. Calibration

Evaluated using leak-free nested partitioning to evaluate post-hoc calibration methods (Platt scaling and Isotonic regression) on the 800-instance holdout test set.

> [!IMPORTANT]
> **Comparability Note**: Phase 17 calibration uses an internal 3-way nested split ($2,560$ base-training instances $+$ $640$ calibration tuning instances $+$ $800$ untouched test instances) derived from the $3,200$ training partition. Consequently, the base models are trained on $2,560$ examples rather than the full $3,200$ examples used in Phase 14. Uncalibrated numbers in this table reflect this 2,560-sample model and must **NOT** be presented as directly identical to the Phase 14 primary benchmark numbers.

| Model | Calibration Method | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | Uncalibrated | 0.6338 | 0.6387 | 0.7069 | 0.7096 | 0.2155 | 0.0444 | 0.1248 |
| Logistic Regression | Sigmoid / Platt | 0.6300 | 0.6373 | 0.7069 | 0.7096 | 0.2165 | 0.0547 | 0.1114 |
| Logistic Regression | Isotonic | 0.6225 | 0.6505 | 0.6971 | 0.6751 | 0.2151 | 0.0341 | 0.2473 |
| XGBoost | Uncalibrated | 0.6488 | 0.6492 | 0.7638 | 0.7787 | 0.1900 | 0.0712 | 0.1888 |
| XGBoost | Sigmoid / Platt | 0.6550 | 0.6452 | 0.7638 | 0.7787 | 0.1935 | 0.0825 | 0.1210 |
| XGBoost | Isotonic | 0.6675 | 0.6472 | 0.7641 | 0.7557 | 0.1890 | 0.0244 | 0.1256 |

### Calibration Findings
- **Isotonic Calibration ECE Minimization**: Isotonic regression dramatically reduces Expected Calibration Error (ECE) for XGBoost from `0.0712` to `0.0244` (a $65.7\%$ reduction) and Brier score to `0.1890`.
- **Ranking Invariance**: Sigmoid (Platt) scaling strictly preserves rank ordering, maintaining identical ROC-AUC (`0.7069` for LR, `0.7638` for XGBoost) and PR-AUC (`0.7096` for LR, `0.7787` for XGBoost).

---

## 5. Retrieval-Augmented Standalone Variant

The retrieval-augmented pipeline evaluates whether adding external text-evidence features improves detection.

> [!WARNING]
> **Standalone Variant Specification**:
> - Evaluated strictly on the $N = 2,500$ usable population where local evidence retrieval is structurally supported (HaluEval: 1,500, FEVER: 1,000).
> - Evaluated on a dedicated 500-instance holdout test set ($2,000$ train / $500$ test, balanced 250/250).
> - **TruthfulQA ($N=1,500$) is excluded** because its context consists of web reference URLs rather than local text passages, and open-web queries were prohibited.
> - This is a **standalone experimental variant** and is **NOT part of the Universal Core Detector**.

| Condition | Features | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Condition A (Internal-Only) | 11 | Logistic Regression | 0.7460 | 0.7581 | 0.8463 | 0.8486 | 0.1597 | 0.0494 |
| Condition A (Internal-Only) | 11 | XGBoost | 0.7880 | 0.8000 | 0.8978 | 0.8982 | 0.1264 | 0.0323 |
| Condition B (Universal Core) | 19 | Logistic Regression | 0.7580 | 0.7660 | 0.8582 | 0.8657 | 0.1538 | 0.0738 |
| Condition B (Universal Core) | 19 | XGBoost | 0.7860 | 0.7864 | 0.9011 | 0.9035 | 0.1246 | 0.0482 |
| Condition C (Retrieval-Only) | 6 | Logistic Regression | 0.6500 | 0.6824 | 0.6990 | 0.6543 | 0.2187 | 0.0350 |
| Condition C (Retrieval-Only) | 6 | XGBoost | 0.6800 | 0.7193 | 0.7491 | 0.7293 | 0.2053 | 0.0523 |
| Condition D (Internal + Retrieval) | 17 | Logistic Regression | 0.7660 | 0.7754 | 0.8637 | 0.8675 | 0.1505 | 0.0380 |
| Condition D (Internal + Retrieval) | 17 | XGBoost | 0.8060 | 0.8159 | 0.9184 | 0.9214 | 0.1180 | 0.0593 |
| Condition E (Universal Core + Retrieval) | 25 | Logistic Regression | 0.7620 | 0.7680 | 0.8706 | 0.8823 | 0.1459 | 0.0606 |
| Condition E (Universal Core + Retrieval) | 25 | XGBoost | 0.8120 | 0.8178 | 0.9169 | 0.9201 | 0.1185 | 0.0518 |

### Retrieval Analysis
- On benchmark tasks with accessible evidence corpora, retrieval features provide a strong boost: XGBoost ROC-AUC rises from `0.8978` (Internal-Only) and `0.9011` (Universal Core) to `0.9184` (Internal + Retrieval) and `0.9169` (Universal Core + Retrieval).
- Standalone retrieval (Condition C) achieves `0.7491` ROC-AUC, confirming that external retrieval and internal token likelihoods measure complementary dimensions of factual validity.

---

## 6. Dataset-Level Primary Benchmark

Subgroup performance of the **Phase 14 Universal Core Detector** across individual benchmark sources on the 800-instance primary holdout test set ($N_{\text{HaluEval}}=300$, $N_{\text{TruthfulQA}}=300$, $N_{\text{FEVER}}=200$):

| Dataset | Test Samples (N) | Task Domain | Model | Accuracy | ROC-AUC |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **HaluEval** | 300 | Dialogue, QA, Summarization (with context) | Logistic Regression | 0.8400 | 0.9311 |
| **HaluEval** | 300 | Dialogue, QA, Summarization (with context) | XGBoost | 0.8633 | 0.9433 |
| **TruthfulQA** | 300 | Misconceptions & Falsehoods (closed-book) | Logistic Regression | 0.5133 | 0.5210 |
| **TruthfulQA** | 300 | Misconceptions & Falsehoods (closed-book) | XGBoost | 0.5933 | 0.6066 |
| **FEVER** | 200 | Wikipedia Claim Verification (factoid claims) | Logistic Regression | 0.4850 | 0.4870 |
| **FEVER** | 200 | Wikipedia Claim Verification (factoid claims) | XGBoost | 0.5300 | 0.5441 |

### Subgroup Insights
- **HaluEval Exceptional Discrimination**: Internal token likelihoods, rank dispersion, and NLI agreement achieve outstanding separation when reference context is present (XGBoost ROC-AUC `0.9433`, Accuracy `86.33%`).
- **TruthfulQA Moderate Separation**: XGBoost captures behavioral and entropy patterns to attain `0.6066` ROC-AUC and `59.33%` accuracy on adversarial misconceptions.
- **FEVER In-Context Challenge**: Closed-book forward generation signals without external Wikipedia retrieval hover near chance (`0.5441` ROC-AUC for XGBoost), validating that parametric uncertainty alone cannot detect factual fabrication when contextual support is absent.

---

## 7. Error Analysis Summary

Based on descriptive evaluation of the Phase 14 Universal Core holdout predictions ($N=800$):

### 7.1 Overall Accuracy and Error Rates
- **Logistic Regression**:
  - Accuracy: `0.6288` ($503/800$)
  - Overall Error Rate: `0.3712` ($297/800$)
  - False Positives (FP): $156$ ($19.5\%$ of test set, $39.0\%$ of faithful instances)
  - False Negatives (FN): $141$ ($17.6\%$ of test set, $35.25\%$ of hallucinated instances)
- **XGBoost**:
  - Accuracy: `0.6788` ($543/800$)
  - Overall Error Rate: `0.3212` ($257/800$)
  - False Positives (FP): $124$ ($15.5\%$ of test set, $31.0\%$ of faithful instances)
  - False Negatives (FN): $133$ ($16.6\%$ of test set, $33.25\%$ of hallucinated instances)

### 7.2 Dataset-Level Error Breakdown
| Model | Dataset Slice | Samples (N) | Error Rate | False Positive Rate | False Negative Rate | Accuracy | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | **Overall** | 800 | 0.3712 | 0.3900 | 0.3525 | 0.6288 | 0.7079 |
| Logistic Regression | FEVER | 200 | 0.5150 | 0.6300 | 0.4000 | 0.4850 | 0.4870 |
| Logistic Regression | HaluEval | 300 | 0.1600 | 0.1267 | 0.1933 | 0.8400 | 0.9311 |
| Logistic Regression | TruthfulQA | 300 | 0.4867 | 0.4933 | 0.4800 | 0.5133 | 0.5210 |
| XGBoost | **Overall** | 800 | 0.3212 | 0.3100 | 0.3325 | 0.6788 | 0.7688 |
| XGBoost | FEVER | 200 | 0.4700 | 0.4400 | 0.5000 | 0.5300 | 0.5441 |
| XGBoost | HaluEval | 300 | 0.1367 | 0.1267 | 0.1467 | 0.8633 | 0.9433 |
| XGBoost | TruthfulQA | 300 | 0.4067 | 0.4067 | 0.4067 | 0.5933 | 0.6066 |

### 7.3 Model Agreement & Disagreement Matrix ($N=800$)
| Metric Category | Count | Fraction of Holdout Test |
| :--- | :---: | :---: |
| **Total Test Instances** | 800 | 100.0% |
| **Model Agreement** ($\hat{y}_{\text{LR}} = \hat{y}_{\text{XGB}}$) | 580 | 72.50% |
| **Model Disagreement** ($\hat{y}_{\text{LR}} \neq \hat{y}_{\text{XGB}}$) | 220 | 27.50% |
| **Both Models Correct** | 413 | 51.62% |
| **Both Models Wrong** | 167 | 20.88% |
| **LR Correct & XGBoost Wrong** | 90 | 11.25% |
| **LR Wrong & XGBoost Correct** | 130 | 16.25% |

### 7.4 Important Descriptive Error Patterns
1. **Response Length Association**:
   - Correctly identified faithful responses (True Negatives) were consistently short (mean word count: `4.06` words for both models).
   - Faithful responses incorrectly classified as hallucinations (False Positives) exhibited substantially longer outputs (mean word count: `9.65` words for LR, `8.74` words for XGBoost).
   - Hallucinated responses correctly identified (True Positives) were long (mean word count: `10.75` words for LR, `10.96` words for XGBoost), whereas missed hallucinations (False Negatives) were shorter (mean word count: `6.01` words for LR, `5.76` words for XGBoost).
2. **Prediction Confidence Boundary Clustering**:
   - Model errors clustered systematically closer to the decision boundary: False Positives had mean predicted probability `0.6197` (LR) / `0.6074` (XGBoost), and False Negatives had mean probability `0.3795` (LR) / `0.3806` (XGBoost).
   - In contrast, confident correct classifications showed distinct separation: True Positives averaged `0.6842` (LR) / `0.7394` (XGBoost), while True Negatives averaged `0.3054` (LR) / `0.2078` (XGBoost).
3. **Domain Discrepancy**:
   - HaluEval accounts for only $41$ errors for XGBoost ($13.7\%$ error rate), whereas FEVER ($94$ errors, $47.0\%$) and TruthfulQA ($122$ errors, $40.7\%$) account for $84.0\%$ of all XGBoost classification errors.

*(Note: These patterns represent purely empirical, descriptive characterizations of the prediction data. Sequence length was intentionally excluded as an input feature in the 19-feature model).*

---

## 8. Final Experiment Map

| Experiment | Dataset | N | Features | Models | Purpose | Primary / Secondary / Variant |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **Phase 6: Baseline Classifiers** | Combined (HaluEval, TruthfulQA, FEVER) | 4,000 | 11 | Logistic Regression, XGBoost | Evaluate standalone white-box internal generation signals | Secondary (Baseline Reference) |
| **Phase 7: Sequence Length Ablation** | Combined (HaluEval, TruthfulQA, FEVER) | 4,000 | 11 vs 12 | Logistic Regression, XGBoost | Quantify sensitivity and potential confound from token length feature | Secondary (Methodological Ablation) |
| **Phase 14: Universal Core Detector** | Combined (HaluEval, TruthfulQA, FEVER) | 4,000 | 19 | Logistic Regression, XGBoost | Primary multi-signal benchmark combining internal, self-consistency, and NLI | **PRIMARY REFERENCE BENCHMARK** |
| **Phase 15: Feature-Family Ablation** | Combined (HaluEval, TruthfulQA, FEVER) | 4,000 | 3, 5, 8, 11, 14, 16, 19 | Logistic Regression, XGBoost | Systematically isolate individual and pairwise contributions of feature groups | Secondary (Ablation Study) |
| **Phase 16: Cross-Dataset Generalization** | LODO (FEVER, TruthfulQA, HaluEval) | 4,000 | 11, 14, 16, 19 | Logistic Regression, XGBoost | Measure out-of-domain transfer performance on completely unseen benchmarks | Secondary (Generalization Evaluation) |
| **Phase 17: Probability Calibration** | Combined (HaluEval, TruthfulQA, FEVER) | 4,000 (2,560 / 640 / 800) | 19 | Logistic Regression, XGBoost (Platt, Isotonic) | Evaluate post-hoc calibration to ensure reliable posterior probability outputs | Secondary (Calibration Analysis) |
| **Phase 18: Comprehensive Error Analysis** | Phase 14 Holdout Set | 800 | 19 | Logistic Regression, XGBoost | Post-hoc descriptive audit of confusion patterns, length profiles, and disagreements | Secondary (Error Profiling) |
| **Phase 19: Retrieval-Augmented Variant** | HaluEval + FEVER (usable subset) | 2,500 (2,000 / 500) | 6, 11, 17, 19, 25 | Logistic Regression, XGBoost | Evaluate external evidence retrieval as an augmentation to internal signals | Standalone Variant (Not Core) |
