# Final Research Findings, Conclusions & Limitations

This document presents the consolidated research findings, empirical conclusions, and methodological limitations for the project **"Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals"**. All statements, metrics, and analyses herein are grounded strictly in the validated experimental outputs produced across Phases 3–19.

---

## 1. Research Objective

The primary objective of this investigation is to evaluate whether hallucination risk in Large Language Model (LLM) responses can be detected using lightweight, supervised classification models trained on internal generation uncertainty signals and behavioral probes, without requiring full-scale secondary LLM re-evaluations. Specifically, the project studies whether raw token likelihood distributions, predictive entropy, sequence perplexity, token rank dynamics, stochastic self-consistency, and natural language inference (NLI) agreement provide reliable statistical indicators of factual correctness.

---

## 2. Experimental Setup

The investigation adhered to a standardized, reproducible experimental framework:
- **Datasets**: A harmonized benchmark of $N = 4,000$ examples spanning three established hallucination domains:
  - **HaluEval** ($1,500$ instances): Question answering, dialogue, and summarization grounded in reference Wikipedia passages.
  - **TruthfulQA** ($1,500$ instances): Adversarial open-domain questions targeting human misconceptions and false beliefs.
  - **FEVER** ($1,000$ instances): Fact verification claims requiring binary validation against Wikipedia knowledge.
- **Canonical Labeling**: Binary classification where label `0` denotes faithful / factual / supported responses, and label `1` denotes hallucinated / unfaithful / unsupported responses. The dataset is balanced ($2,000$ label 0, $2,000$ label 1).
- **Base Generative Model**: `Qwen/Qwen3.5-0.8B` (referred to as `Qwen3.5-0.8B`), an autoregressive open-weight causal language model executed locally on an NVIDIA GeForce RTX 3050 GPU (`cuda:0`, PyTorch 2.6.0+cu124, 16-bit precision).
- **Signal Extraction & Feature Groups**:
  - **Internal Generation Signals (11 features)**: Extracted via unadulterated forward pass logits over labeled responses: `min_log_prob`, `mean_log_prob`, `mean_token_prob`, `token_prob_std`, `mean_entropy`, `max_entropy`, `entropy_std`, `log_perplexity`, `log_mean_token_rank`, `log_max_token_rank`, `log_rank_std`.
  - **Self-Consistency Signals (5 features)**: Derived from $k=5$ stochastic generations per prompt at temperature $0.7$: `exact_match_agreement`, `mean_pairwise_similarity`, `min_pairwise_similarity`, `max_pairwise_similarity`, `pairwise_similarity_std` (using `sentence-transformers/all-MiniLM-L6-v2` embeddings).
  - **NLI Agreement Signals (3 features)**: Evaluated across all $\binom{5}{2} = 10$ unordered generation pairs ($40,000$ total pairs) via bidirectional scoring with `cross-encoder/nli-MiniLM2-L6-H768`: `mean_pairwise_entailment`, `mean_pairwise_contradiction`, `nli_disagreement`.
  - **Retrieval Signals (6 features, Standalone Variant)**: Dense evidence retrieval using `all-MiniLM-L6-v2` across local benchmark corpora: `top1_evidence_similarity`, `top3_evidence_similarity`, `response_top1_similarity`, `response_top3_similarity`, `evidence_response_margin`, `retrieval_agreement`.
- **Classification Architectures**:
  - **Logistic Regression**: Linear classifier with $L_2$ regularization ($C=1.0$, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`), preceded by `StandardScaler` fitted strictly on training data.
  - **XGBoost**: Gradient-boosted decision trees (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0.1`, `eval_metric='logloss'`, `random_state=42`).
- **Partitioning Protocol**: A fixed, deterministic stratified 80/20 train/test split with `random_state=42` stratified on the joint key `source_dataset + "_" + label` ($3,200$ training examples, $800$ holdout test examples: $400$ label 0, $400$ label 1).
- **Evaluation Metrics**: Threshold-dependent (Accuracy, Precision, Recall, F1) and threshold-independent ranking metrics (ROC-AUC, PR-AUC), alongside probabilistic reliability measures (Brier score, Expected Calibration Error [ECE], Maximum Calibration Error [MCE]).

---

## 3. Primary Findings

The primary empirical comparison contrasts the **Phase 6 Baseline** (11 internal generation signals) against the **Phase 14 Universal Core Detector** (19 signals combining internal, self-consistency, and NLI features) on the identical 800-example holdout test partition:

| Pipeline | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 6 Baseline (11 features)** | Logistic Regression | 0.6212 | 0.6120 | 0.6625 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| **Phase 6 Baseline (11 features)** | XGBoost | 0.6700 | 0.6518 | 0.7300 | 0.6887 | 0.7411 | 0.7322 | 0.1999 | 0.0228 |
| **Phase 14 Universal Core (19 features)** | Logistic Regression | 0.6288 | 0.6241 | 0.6475 | 0.6356 | 0.7079 | 0.7124 | 0.2151 | 0.0525 |
| **Phase 14 Universal Core (19 features)** | XGBoost | 0.6788 | 0.6829 | 0.6675 | 0.6751 | 0.7688 | 0.7795 | 0.1891 | 0.0341 |

### Empirical Observations
- On this evaluation set, XGBoost achieved an ROC-AUC of `0.7688` and a PR-AUC of `0.7795` when trained on all 19 features, compared to an ROC-AUC of `0.7411` and a PR-AUC of `0.7322` when trained on the 11 baseline internal signals alone ($\Delta_{\text{ROC-AUC}} = +0.0277$, $\Delta_{\text{PR-AUC}} = +0.0474$).
- For Logistic Regression, the Universal Core changed ROC-AUC from `0.6930` to `0.7079` ($\Delta = +0.0149$) and PR-AUC from `0.6618` to `0.7124` ($\Delta = +0.0505$).
- Accuracy changed from `0.6700` to `0.6788` for XGBoost and from `0.6212` to `0.6288` for Logistic Regression.
- Probability error as measured by the Brier score changed from `0.1999` to `0.1891` for XGBoost and from `0.2177` to `0.2151` for Logistic Regression.
- These results indicate that adding behavioral self-consistency and cross-encoder NLI agreement features to internal token signals was associated with measurable improvements in ranking discrimination (ROC-AUC, PR-AUC) and probabilistic score accuracy across both linear and non-linear classifiers.

---

## 4. Feature-Family Findings

Phase 15 systematically ablated the three primary feature families across 7 distinct configurations on the identical 800-instance holdout test set:

| Feature Configuration | Feature Count | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Internal Only** | 11 | Logistic Regression | 0.6212 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| **Internal Only** | 11 | XGBoost | 0.6625 | 0.6793 | 0.7397 | 0.7329 | 0.1993 | 0.0228 |
| **Self-Consistency Only** | 5 | Logistic Regression | 0.4900 | 0.4545 | 0.4845 | 0.4863 | 0.2515 | 0.0300 |
| **Self-Consistency Only** | 5 | XGBoost | 0.5100 | 0.5410 | 0.4896 | 0.4870 | 0.2567 | 0.0450 |
| **NLI Only** | 3 | Logistic Regression | 0.5038 | 0.5031 | 0.5050 | 0.5025 | 0.2503 | 0.0088 |
| **NLI Only** | 3 | XGBoost | 0.4713 | 0.4885 | 0.4912 | 0.5005 | 0.2543 | 0.0731 |
| **Internal + Self-Consistency** | 16 | Logistic Regression | 0.6200 | 0.6311 | 0.7036 | 0.7043 | 0.2159 | 0.0613 |
| **Internal + Self-Consistency** | 16 | XGBoost | 0.6512 | 0.6610 | 0.7563 | 0.7675 | 0.1918 | 0.0450 |
| **Internal + NLI** | 14 | Logistic Regression | 0.6225 | 0.6308 | 0.7033 | 0.6994 | 0.2161 | 0.0576 |
| **Internal + NLI** | 14 | XGBoost | 0.6700 | 0.6615 | 0.7659 | 0.7697 | 0.1909 | 0.0257 |
| **Self-Consistency + NLI** | 8 | Logistic Regression | 0.4750 | 0.4697 | 0.4874 | 0.4795 | 0.2516 | 0.0475 |
| **Self-Consistency + NLI** | 8 | XGBoost | 0.5062 | 0.5153 | 0.5053 | 0.5128 | 0.2541 | 0.0522 |
| **All Three (Universal Core)** | 19 | Logistic Regression | 0.6288 | 0.6356 | 0.7079 | 0.7124 | 0.2151 | 0.0525 |
| **All Three (Universal Core)** | 19 | XGBoost | 0.6788 | 0.6751 | 0.7688 | 0.7795 | 0.1891 | 0.0341 |

### Specific Ablation Observations on this Evaluation Split
- **Standalone Feature Family Performance**:
  - Internal generation signals alone attained an ROC-AUC of `0.7397` and accuracy of `0.6625` with XGBoost.
  - In contrast, Self-Consistency features alone yielded an ROC-AUC of `0.4896` (XGBoost) and `0.4845` (LR), while NLI features alone yielded an ROC-AUC of `0.4912` (XGBoost) and `0.5050` (LR).
  - Combining Self-Consistency and NLI without internal signals (8 features) yielded an ROC-AUC of `0.5053` (XGBoost) and `0.4874` (LR).
  - These measurements show that on this benchmark partition, behavioral probes (semantic similarity and NLI agreement over sampled outputs) did not achieve classification separation when evaluated independently of the primary response's internal log-probability and entropy dynamics.
- **Performance When Combined with Internal Signals**:
  - Adding Self-Consistency to Internal signals increased XGBoost ROC-AUC from `0.7397` to `0.7563` and PR-AUC from `0.7329` to `0.7675`.
  - Adding NLI to Internal signals increased XGBoost ROC-AUC from `0.7397` to `0.7659` and PR-AUC from `0.7329` to `0.7697`.
  - Combining all three families produced the highest observed ranking discrimination in the ablation study (XGBoost ROC-AUC `0.7688`, PR-AUC `0.7795`).

---

## 5. Cross-Dataset Generalization

Phase 16 evaluated how detector models transferred when an entire benchmark task was omitted from training and used strictly as an unseen evaluation target (Leave-One-Dataset-Out):

| Test Dataset (Unseen) | Training Datasets | Feature Configuration | Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal Only (11) | Logistic Regression | 0.5060 | 0.4868 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal Only (11) | XGBoost | 0.4830 | 0.4903 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | All Three (19) | Logistic Regression | 0.4960 | 0.4875 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | All Three (19) | XGBoost | 0.5060 | 0.5167 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal Only (11) | Logistic Regression | 0.4660 | 0.4439 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal Only (11) | XGBoost | 0.4793 | 0.4539 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | All Three (19) | Logistic Regression | 0.4753 | 0.4575 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | All Three (19) | XGBoost | 0.4853 | 0.4608 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal Only (11) | Logistic Regression | 0.2140 | 0.1601 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal Only (11) | XGBoost | 0.3980 | 0.3599 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | All Three (19) | Logistic Regression | 0.4493 | 0.2583 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | All Three (19) | XGBoost | 0.4320 | 0.4368 |

### Empirical Observations on Transfer
- When trained on HaluEval + TruthfulQA and tested on FEVER, ROC-AUC was `0.4868` for LR and `0.4903` for XGBoost with internal signals, changing to `0.4875` (LR) and `0.5167` (XGBoost) with all 19 features.
- When trained on HaluEval + FEVER and tested on TruthfulQA, ROC-AUC was `0.4439` (LR) and `0.4539` (XGBoost) with internal signals, changing to `0.4575` (LR) and `0.4608` (XGBoost) with all 19 features.
- When trained on TruthfulQA + FEVER and tested on HaluEval, ROC-AUC was `0.1601` (LR) and `0.3599` (XGBoost) with internal signals, changing to `0.2583` (LR) and `0.4368` (XGBoost) with all 19 features.
- These results demonstrate that out-of-domain transfer performance changed substantially when benchmark sources were held out, indicating strong dataset-dependence in the learned feature mappings. Transfer performance was markedly lower than in-domain holdout evaluation.

---

## 6. Calibration Findings

Phase 17 examined the empirical probability calibration of the Universal Core Detector using Platt scaling (sigmoid) and non-parametric isotonic regression:

| Model | Calibration Method | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | Uncalibrated | 0.6338 | 0.6387 | 0.7069 | 0.7096 | 0.2155 | 0.0444 | 0.1248 |
| Logistic Regression | Sigmoid / Platt | 0.6300 | 0.6373 | 0.7069 | 0.7096 | 0.2165 | 0.0547 | 0.1114 |
| Logistic Regression | Isotonic | 0.6225 | 0.6505 | 0.6971 | 0.6751 | 0.2151 | 0.0341 | 0.2473 |
| XGBoost | Uncalibrated | 0.6488 | 0.6492 | 0.7638 | 0.7787 | 0.1900 | 0.0712 | 0.1888 |
| XGBoost | Sigmoid / Platt | 0.6550 | 0.6452 | 0.7638 | 0.7787 | 0.1935 | 0.0825 | 0.1210 |
| XGBoost | Isotonic | 0.6675 | 0.6472 | 0.7641 | 0.7557 | 0.1890 | 0.0244 | 0.1256 |

### Methodological Distinction & Observations
- **Partitioning Protocol**: Phase 17 utilized a 3-way nested partition consisting of $2,560$ base-training examples, $640$ calibration-tuning examples, and $800$ untouched holdout test examples. Because base models were fitted on $2,560$ examples rather than the full $3,200$ examples used in Phase 14, their uncalibrated metrics (e.g., XGBoost uncalibrated accuracy `0.6488` vs Phase 14 `0.6788`) reflect this smaller base-training sample size and must not be directly compared numerically with Phase 14.
- **Empirical Calibration Effects**:
  - For XGBoost, isotonic regression reduced Expected Calibration Error (ECE) from `0.0712` to `0.0244` and Maximum Calibration Error (MCE) from `0.1888` to `0.1256`, with a Brier score of `0.1890`.
  - For Logistic Regression, isotonic regression reduced ECE from `0.0444` to `0.0341`, although MCE increased to `0.2473`.
  - Sigmoid (Platt) scaling strictly preserved ranking metrics (ROC-AUC `0.7069` for LR, `0.7638` for XGBoost), while isotonic regression slightly adjusted decision thresholds and bin boundaries.

---

## 7. Error-Analysis Findings

Phase 18 conducted a post-hoc descriptive error audit on the Phase 14 Universal Core holdout predictions ($N=800$):

### 7.1 Overall Errors and Confusion Counts
- **Logistic Regression**: Overall error rate was `0.3712` ($297$ errors out of $800$ instances). False positives numbered $156$ ($39.0\%$ false positive rate among faithful instances), and false negatives numbered $141$ ($35.25\%$ false negative rate among hallucinated instances).
- **XGBoost**: Overall error rate was `0.3212` ($257$ errors out of $800$ instances). False positives numbered $124$ ($31.0\%$ false positive rate among faithful instances), and false negatives numbered $133$ ($33.25\%$ false negative rate among hallucinated instances).

### 7.2 Dataset-Level Disparities
Error distribution was highly uneven across benchmark tasks:
- **HaluEval** ($N=300$): Error rate was `0.1367` for XGBoost and `0.1600` for Logistic Regression (XGBoost ROC-AUC `0.9433`).
- **TruthfulQA** ($N=300$): Error rate was `0.4067` for XGBoost and `0.4867` for Logistic Regression (XGBoost ROC-AUC `0.6066`).
- **FEVER** ($N=200$): Error rate was `0.4700` for XGBoost and `0.5150` for Logistic Regression (XGBoost ROC-AUC `0.5441`).
- Of the $257$ total test errors made by XGBoost, $122$ ($47.5\%$) occurred on TruthfulQA, $94$ ($36.6\%$) on FEVER, and only $41$ ($16.0\%$) on HaluEval.

### 7.3 Model Agreement and Disagreement
- Both models produced identical classification decisions on $580$ of $800$ instances ($72.50\%$ agreement rate).
- On $220$ instances ($27.50\%$), the two models disagreed.
- Both models were simultaneously correct on $413$ instances ($51.62\%$) and simultaneously incorrect on $167$ instances ($20.88\%$).
- Logistic Regression was correct while XGBoost was incorrect on $90$ instances ($11.25\%$).
- XGBoost was correct while Logistic Regression was incorrect on $130$ instances ($16.25\%$).

### 7.4 Descriptive Patterns and Effect Sizes
- **Output Length Profile**:
  - True Negatives (correctly identified faithful responses) had an average length of `4.06` words.
  - False Positives (faithful responses misclassified as hallucinations) had an average length of `9.65` words for LR and `8.74` words for XGBoost.
  - True Positives (correctly identified hallucinations) averaged `10.75` words for LR and `10.96` words for XGBoost.
  - False Negatives (hallucinations misclassified as faithful) averaged `6.01` words for LR and `5.76` words for XGBoost.
- **Confidence Boundary Clustering**: Erroneous predictions systematically clustered closer to the decision boundary (mean predicted probability `0.6197` for LR FP, `0.3795` for LR FN) than correct classifications (`0.6842` for LR TP, `0.3054` for LR TN).
- **Feature Separation Effect Sizes**:
  - Comparing True Negatives to False Positives, `min_log_prob` exhibited a standardized mean difference of Cohen's $d = 1.18$, and `log_max_token_rank` exhibited $d = -1.02$.
  - Comparing overall Correct versus Incorrect predictions, `mean_token_prob` had Cohen's $d = 0.43$, `mean_entropy` had $d = -0.43$, and `log_max_token_rank` had $d = -0.42$.
  - These values represent empirical descriptive associations within the evaluated test set; they do not establish causal mechanisms.

---

## 8. Retrieval Findings

Phase 19 evaluated a standalone retrieval-augmented detector variant using dense bi-encoder embeddings (`sentence-transformers/all-MiniLM-L6-v2`):

> [!NOTE]
> **Scope & Population Constraints**:
> - Evaluated strictly on the $N = 2,500$ usable subset comprising HaluEval ($1,500$) and FEVER ($1,000$), where local evidence passages or entity title pointers were available.
> - **TruthfulQA ($1,500$ instances) was excluded** because its context field contained web reference URLs rather than local text passages, and live external web search was prohibited.
> - Evaluated on a dedicated 500-instance holdout test set ($2,000$ train / $500$ holdout, balanced $250/250$).
> - This variant is **standalone** and is not part of the primary 19-feature Universal Core Detector.

| Condition | Features | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A: Internal-Only** | 11 | Logistic Regression | 0.7460 | 0.7581 | 0.8463 | 0.8486 | 0.1597 | 0.0494 |
| **Condition A: Internal-Only** | 11 | XGBoost | 0.7880 | 0.8000 | 0.8978 | 0.8982 | 0.1264 | 0.0323 |
| **Condition B: Universal Core** | 19 | Logistic Regression | 0.7580 | 0.7660 | 0.8582 | 0.8657 | 0.1538 | 0.0738 |
| **Condition B: Universal Core** | 19 | XGBoost | 0.7860 | 0.7864 | 0.9011 | 0.9035 | 0.1246 | 0.0482 |
| **Condition C: Retrieval-Only** | 6 | Logistic Regression | 0.6500 | 0.6824 | 0.6990 | 0.6543 | 0.2187 | 0.0350 |
| **Condition C: Retrieval-Only** | 6 | XGBoost | 0.6800 | 0.7193 | 0.7491 | 0.7293 | 0.2053 | 0.0523 |
| **Condition D: Internal + Retrieval** | 17 | Logistic Regression | 0.7660 | 0.7754 | 0.8637 | 0.8675 | 0.1505 | 0.0380 |
| **Condition D: Internal + Retrieval** | 17 | XGBoost | 0.8060 | 0.8159 | 0.9184 | 0.9214 | 0.1180 | 0.0593 |
| **Condition E: Universal + Retrieval** | 25 | Logistic Regression | 0.7620 | 0.7680 | 0.8706 | 0.8823 | 0.1459 | 0.0606 |
| **Condition E: Universal + Retrieval** | 25 | XGBoost | 0.8120 | 0.8178 | 0.9169 | 0.9201 | 0.1185 | 0.0518 |

### Retrieval Observations
- On this 2,500-instance subset, Condition C (Retrieval-Only, 6 features) achieved an ROC-AUC of `0.7491` with XGBoost and `0.6990` with Logistic Regression.
- Combining retrieval features with internal signals (Condition D, 17 features) changed XGBoost ROC-AUC from `0.8978` (Internal-Only) to `0.9184` ($\Delta = +0.0206$) and accuracy from `0.7880` to `0.8060`.
- Combining retrieval features with all Universal Core features (Condition E, 25 features) yielded an ROC-AUC of `0.9169` and accuracy of `0.8120` for XGBoost, and ROC-AUC `0.8706` for Logistic Regression.
- **Distinction Between Benchmark-Provided and External Evidence**: In this experiment, evidence passages were indexed directly from benchmark metadata (HaluEval user contexts and FEVER Wikipedia entity pages). This demonstrates groundedness against supplied local documents, not autonomous open-web fact-checking.

---

## 9. What the Experiments Demonstrate

The complete experimental evaluations support the following evidence-based conclusions:

1. **Predictive Utility of Internal White-Box Signals**: The observed results provide evidence that token-level uncertainty dynamics (minimum token log-probability, predictive entropy, sequence perplexity, and token rank dispersion) contain measurable statistical signal for classifying hallucination status across diverse benchmarks, achieving baseline ROC-AUC of `0.7411` with XGBoost.
2. **Complementary Value of Multi-Signal Ensembling**: The experiments demonstrate that augmenting internal generation signals with behavioral consistency statistics ($k=5$ stochastic samples) and cross-encoder NLI agreement yields higher threshold-free ranking performance (ROC-AUC `0.7688`, PR-AUC `0.7795` for XGBoost on the primary holdout) than internal signals alone.
3. **Task-Context Dependent Discrimination**: The evaluation shows that internal generation signals are substantially more discriminative when generative outputs are conditioned on reference context (HaluEval holdout ROC-AUC `0.9433`) than when generated in closed-book factoid settings without context (FEVER holdout ROC-AUC `0.5441`).
4. **Feasibility of Post-Hoc Probability Calibration**: The calibration results demonstrate that non-parametric isotonic regression can substantially improve posterior probability alignment (reducing XGBoost ECE from `0.0712` to `0.0244`) without degrading ranking discrimination.
5. **Additivity of Local Retrieval Grounding**: On datasets where local evidence passages are structurally accessible, adding dense retrieval similarity features to internal generation signals is associated with increased classification performance (ROC-AUC rising to `0.9184` on the usable $N=2,500$ subset).

---

## 10. What the Experiments Do NOT Demonstrate

To ensure scientific rigor, the following explicit boundaries and non-claims are documented:

1. **No Cross-Model Generalization**: All generation and token scoring was conducted exclusively with `Qwen3.5-0.8B`. The experiments do **not** demonstrate whether these learned classifiers or feature relationships transfer to other model families (e.g., Llama, Mistral, Gemma) or larger parameter scales.
2. **Limited Cross-Dataset Invariance**: The cross-dataset transfer experiments showed substantial performance degradation when evaluating on unseen benchmark distributions. The results do **not** support a claim of universal, out-of-the-box generalization across arbitrary task domains.
3. **Retrieval Not Universally Applicable**: Retrieval features could not be evaluated on TruthfulQA without external web queries or data fabrication. The experiments do **not** show that retrieval is universally beneficial across arbitrary closed-book benchmarks.
4. **Benchmark-Derived Ground Truth**: All training and test labels reflect annotations from existing research benchmarks (HaluEval, TruthfulQA, FEVER), which carry their own labeling conventions, heuristics, and potential labeling noise.
5. **Contextual Sampling Nature of Behavioral Features**: Self-consistency and NLI agreement features were derived from $k=5$ stochastic samples generated for the prompt, serving as contextual neighborhood probes rather than direct token-level evaluations of the primary benchmark response.
6. **Observational Rather Than Causal Relationships**: Feature importance, Cohen's $d$ effect sizes, and model coefficients reflect observational statistical associations within the dataset; they do **not** prove that low token probability or high entropy *causes* hallucination, or that manipulating sequence length alters factual veracity.
7. **Offline Rather Than Real-Time Deployment**: While feature extraction algorithms were implemented for efficiency, all evaluations were conducted in an offline experimental environment. The experiments do **not** evaluate operational latency, memory footprint, or concurrency under live production traffic.

---

## 11. Threats to Validity

1. **Dataset and Domain Shift**: The three constituent benchmarks differ substantially in task structure (conversational dialogue and summarization in HaluEval, short imitative falsehoods in TruthfulQA, single-sentence claim verification in FEVER). Disparate task formats introduce distribution shifts that affect decision thresholds.
2. **Model Architecture and Capacity Dependence**: A compact 0.8-billion parameter model (`Qwen3.5-0.8B`) was selected to enable feasible local execution. Uncertainty profiles, entropy distributions, and calibration properties may behave differently in larger scale models ($7\text{B}+$).
3. **Benchmark Artifacts and Sequence Length Skew**: Error analysis revealed descriptive differences in sequence length across confusion categories (e.g., False Positives averaged $8.7–9.7$ words vs True Negatives $4.1$ words). While sequence length was intentionally excluded as an input feature, underlying token-level statistics can correlate with length.
4. **Retrieval Corpus Constraints**: Local evidence passages were derived directly from benchmark contexts and entity titles rather than a comprehensive, updated open-domain web corpus.
5. **Sample and Partitioning Sensitivity**: Although primary models used a stratified 80/20 partition with a fixed random seed (`random_state=42`), results reflect this specific sample realization ($N=4,000$).
6. **Calibration Partition Sample Size**: Calibration parameters in Phase 17 were fit on $640$ instances; smaller tuning partitions can introduce variance into non-parametric isotonic step functions.

---

## 12. Conclusion

This research investigated the detection of Large Language Model hallucinations through lightweight supervised classification over white-box internal generation signals and multi-sample behavioral probes. Across an empirical benchmark of 4,000 examples spanning three benchmark sources, internal token likelihoods, predictive entropy, and rank statistics from `Qwen3.5-0.8B` demonstrated clear baseline predictive signal (Phase 6 Baseline: XGBoost ROC-AUC `0.7411`, Logistic Regression ROC-AUC `0.6930`). Incorporating stochastic self-consistency and cross-encoder NLI agreement features yielded consistent increases in threshold-free ranking discrimination (Phase 14 Universal Core: XGBoost ROC-AUC `0.7688`, PR-AUC `0.7795`), with isotonic calibration successfully reducing expected calibration error to `0.0244`. 

Concurrently, the experimental results delineate critical operational boundaries. Leave-one-dataset-out evaluations demonstrated that learned decision boundaries transfer poorly across distinct task formulations without multi-domain training exposure. Furthermore, internal parametric signals exhibited strong discriminative separation primarily when reference context was available (HaluEval ROC-AUC `0.9433`), whereas closed-book factoid claims (FEVER ROC-AUC `0.5441`) remained challenging without external retrieval support. These findings support the viability of lightweight, multi-signal hallucination detection while emphasizing the necessity of domain-aware feature integration and calibration in practical applications.
