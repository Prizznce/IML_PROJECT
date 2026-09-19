# Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals

**Academic Project Report Draft**

**Authors / Team Members**:
- Kunj Patil
- Jaiprakash
- Prince
- Aditya Kumar

---

## 2. Abstract

Large Language Models (LLMs) frequently generate text that is grammatically fluent and contextually coherent yet factually false or unsupported by reference context—a phenomenon known as hallucination. While external fact-checking and secondary LLM evaluations can identify unsupported claims, they introduce substantial latency and computational cost. This project investigates whether internal generation uncertainty signals and lightweight behavioral probes can be combined toward real-time hallucination risk detection. Although designed for potential real-time integration, all evaluations in this study were conducted offline in batch mode across established benchmarks.

Using `Qwen/Qwen3.5-0.8B` as the base causal language model, we extract 11 white-box token signals directly from forward-pass logits (including token log-probabilities, predictive entropy, sequence perplexity, and token rank distributions) across a balanced benchmark of 4,000 examples spanning three benchmark sources: HaluEval ($1,500$ instances), TruthfulQA ($1,500$ instances), and FEVER ($1,000$ instances). To augment internal uncertainty, we compute 5 stochastic self-consistency features across $k=5$ sampled outputs ($20,000$ total generations) and 3 bidirectional Natural Language Inference (NLI) agreement features across $40,000$ response pairs. We also evaluate a standalone dense retrieval variant on $2,500$ instances where local evidence was available. 

Evaluating on a fixed stratified 80/20 train/test partition ($3,200$ train / $800$ holdout test), an 11-feature baseline classifier achieved an ROC-AUC of `0.7411` with XGBoost and `0.6930` with Logistic Regression. Integrating self-consistency and NLI agreement into the 19-feature Universal Core Detector changed XGBoost ROC-AUC to `0.7688` (PR-AUC `0.7795`) and Logistic Regression ROC-AUC to `0.7079` (PR-AUC `0.7124`). Non-parametric isotonic calibration reduced XGBoost Expected Calibration Error (ECE) from `0.0712` to `0.0244`. However, Leave-One-Dataset-Out (LODO) transfer evaluations revealed substantial performance drops (ROC-AUC `0.2583`–`0.5167`), demonstrating that learned decision boundaries remain strongly dataset-dependent. These findings indicate that while internal uncertainty signals provide clear predictive utility for in-domain hallucination risk estimation, cross-domain transfer and closed-book claim verification remain key challenges.

---

## 3. Introduction

Autoregressive Large Language Models (LLMs) are now widely deployed across question answering, dialogue, code generation, and automated summarization. Despite remarkable linguistic fluency, these systems routinely produce statements that contradict source text, fabricate biographical or scientific claims, or confidently assert widespread human misconceptions. In high-stakes domains such as clinical diagnosis, legal analysis, and automated tutoring, undetected hallucinations present severe risks of misinformation and diminished operational trust.

Conventional post-hoc mitigation strategies predominantly rely on external verification loops, such as calling commercial search APIs, indexing external knowledge bases, or deploying secondary "evaluator" LLMs (e.g., LLM-as-a-judge). Although effective in specific settings, these methods incur major latency penalties, high monetary inference costs, and vulnerability to external retrieval noise. 

An alternative paradigm treats the generative model not as an uninspectable black box, but as a probabilistic sequence generator whose internal state transitions reflect uncertainty. During autoregressive decoding, the model calculates a full categorical probability distribution over its token vocabulary at each step. Tokens produced during factual recall or context-grounded reasoning typically exhibit different predictive distributions—characterized by higher confidence, lower entropy, and lower rank dispersion—compared to tokens produced when the model is fabricating facts or navigating unfamiliar concepts.

This project investigates whether these internal generation signals, combined with lightweight multi-sample behavioral probes (self-consistency and natural language inference), can be integrated into tabular machine learning classifiers (Logistic Regression and XGBoost) to predict hallucination risk efficiently. By focusing on lightweight classifiers, the detection system aims to provide interpretable risk estimates with negligible computational overhead relative to full secondary generative passes. Note that while real-time inference is the motivating application and design goal, the present investigation evaluates the methodology offline across curated benchmark corpora.

---

## 4. Problem Statement

The hallucination detection task is formalized as a binary supervised risk classification problem:

Given a prompt $x$, a generated or evaluated response sequence $y = (y_0, y_1, \dots, y_{R-1})$, and an optional reference context $c$, an autoregressive causal language model $\mathcal{M}$ produces a sequence of next-token logit vectors:

$$\mathbf{z}_t = \mathcal{M}(x, y_{<t}, c) \in \mathbb{R}^V \quad \text{for } t \in \{0, 1, \dots, R-1\}$$

where $V$ represents the vocabulary size. 

From these logits, a feature extraction function $\Phi(x, y, \mathbf{z}_{0:R-1})$ extracts a $D$-dimensional feature vector $\mathbf{u} \in \mathbb{R}^D$ summarizing internal token certainty, sequence dispersion, and contextual consistency probes. The objective is to train a supervised tabular classifier $f: \mathbb{R}^D \to [0, 1]$ that outputs an estimated probability $\hat{p} = f(\mathbf{u})$ of hallucination, predicting the ground-truth binary label $z \in \{0, 1\}$:

$$z = \begin{cases} 0 & \text{if the response is factual, faithful, and supported} \\ 1 & \text{if the response is hallucinated, unfaithful, or unsupported} \end{cases}$$

The prediction is evaluated across both discrimination metrics (ROC-AUC, PR-AUC, Accuracy, F1) and probabilistic reliability metrics (Brier score, Expected Calibration Error).

---

## 5. Objectives

The project addresses seven core research objectives:

1. **Investigate Generation Uncertainty Signals**: Formulate, extract, and validate white-box token-level probabilities, predictive entropy, sequence perplexity, and token rank statistics from raw causal LM logits.
2. **Measure Behavioral Self-Consistency**: Generate stochastic output variations ($k=5$) and quantify semantic consistency using embedding cosine similarity statistics.
3. **Investigate NLI Agreement**: Evaluate pairwise bidirectional Natural Language Inference entailment, contradiction, and neutral probabilities across generated alternatives.
4. **Build Supervised Baseline Classifiers**: Train and evaluate interpretable linear (Logistic Regression) and non-linear (XGBoost) baseline models using internal signals alone.
5. **Build a Combined Multi-Signal Classifier**: Construct a unified 19-feature matrix integrating internal signals, self-consistency, and NLI agreement, evaluating improvements over baseline signals.
6. **Evaluate Probability Calibration**: Assess and improve the reliability of posterior risk probabilities using Platt scaling and non-parametric isotonic regression without test data leakage.
7. **Perform Ablation, Generalization, and Error Analysis**: Systematically ablate feature families, measure zero-shot Leave-One-Dataset-Out (LODO) transferability, and conduct a descriptive audit of error patterns.

---

## 6. Related Work / Background

The methodology bridges several distinct areas in machine learning and natural language processing:

- **LLM Hallucination Detection**: Hallucinations are broadly categorized into closed-domain unfaithfulness (contradicting an input context) and open-domain factual fabrication [REFERENCE NEEDED]. Detecting these errors traditionally relies on rule-based checks, external fact-checking databases, or LLM-as-a-judge prompting [REFERENCE NEEDED].
- **Uncertainty Estimation via Token Probabilities**: Autoregressive models compute normalized softmax distributions over vocabulary tokens at each generation step. Prior research has demonstrated that minimum token probabilities and average sequence log-probabilities correlate with generation errors [REFERENCE NEEDED].
- **Predictive Entropy and Perplexity**: Shannon entropy over the vocabulary distribution quantifies the dispersion of model beliefs. Elevated entropy or sequence perplexity often indicates that probability mass is split across multiple competing hypotheses [REFERENCE NEEDED].
- **Self-Consistency and Semantic Dispersion**: Sampling multiple outputs at non-zero temperature and evaluating semantic agreement probes whether an answer is stable across stochastic trajectories. Disagreement among samples has been associated with epistemic uncertainty [REFERENCE NEEDED].
- **Natural Language Inference (NLI) for Consistency**: Cross-encoder NLI models score bidirectional entailment and contradiction between claims. Evaluating pairwise mutual entailment provides a fine-grained measure of semantic divergence [REFERENCE NEEDED].
- **Evidence Retrieval for Fact Verification**: Retrieval-augmented architectures index external passages to evaluate claim groundedness using dense bi-encoders [REFERENCE NEEDED].
- **Lightweight Tabular Classifiers**: Gradient-boosted decision trees (XGBoost) and regularized logistic regression provide fast inference, tabular sample efficiency, and direct feature inspectability without requiring deep neural architectures [REFERENCE NEEDED].

*(Note: Bibliographic references to be added from the project's verified literature sources).*

---

## 7. Dataset Description

To prevent evaluation bias and evaluate domain robustness, experiments were conducted on a harmonized corpus of $N = 4,000$ examples assembled from three benchmark datasets:

| Dataset | Total Samples | Faithful ($z=0$) | Hallucinated ($z=1$) | Task Domain | Context Format |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **HaluEval** | 1,500 | 750 | 750 | QA, Dialogue, Summarization | Wikipedia reference text passages |
| **TruthfulQA** | 1,500 | 750 | 750 | Misconceptions, False Beliefs | Category strings and external URLs |
| **FEVER** | 1,000 | 500 | 500 | Fact Verification Claims | Wikipedia entity title pointers |
| **Combined** | **4,000** | **2,000** | **2,000** | **Multi-Domain Blend** | **Heterogeneous** |

### Canonical Schema and Preprocessing Protocols
1. **Standardized Schema**: Each sample contains six canonical fields: `id` (unique deterministic string), `source_dataset` (`halueval`, `truthfulqa`, `fever`), `prompt` (input question or claim), `response` (evaluated model answer), `context` (supporting text or pointers), and `label` (binary integer $0$ or $1$).
2. **Label Harmonization**:
   - In HaluEval, faithful responses are assigned label $0$; simulated hallucinated responses are assigned label $1$.
   - In TruthfulQA, human-curated truthful answers are labeled $0$; untruthful imitative misconceptions are labeled $1$.
   - In FEVER, claims labeled `SUPPORTS` are mapped to label $0$; claims labeled `REFUTES` are mapped to label $1$. Claims labeled `NOT ENOUGH INFO` were excluded to maintain an unambiguous binary factual truth standard.
3. **Quality Verification**: Preprocessing pipelines verified that text fields contained no missing values, prompt and response strings were stripped of formatting artifacts, and no duplicate IDs existed.

---

## 8. System Architecture

The end-to-end experimental architecture is structured into sequential processing stages:

```
┌────────────────────────────────────────────────────────┐
│               1. Benchmark Data Ingestion              │
│       HaluEval (1,500) + TruthfulQA (1,500) +          │
│       FEVER (1,000) -> 4,000 Harmonized Examples       │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│      2. Supervised Scoring of Primary Responses        │
│   Forward-pass over evaluated response using           │
│   Qwen3.5-0.8B -> 11 Internal Generation Signals       │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│        3. Behavioral Sampling & Consistency            │
│   Generate k=5 stochastic responses per prompt         │
│   (20,000 total outputs) -> 5 Consistency Signals      │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│        4. Cross-Encoder NLI Pairwise Scoring           │
│   Bidirectional scoring across 40,000 generation       │
│   pairs via MiniLM2-L6-H768 -> 3 NLI Signals           │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│       5. Universal Feature Matrix Assembly             │
│   19 Canonical Features x 4,000 Examples               │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│       6. Supervised Classification & Tuning            │
│   Stratified 80/20 Partition (3,200 Train / 800 Test)  │
│   Logistic Regression (StandardScaler) & XGBoost       │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│         7. Multi-Dimensional Evaluation               │
│   Primary Holdout Benchmark | Feature Ablation (7 cfg) │
│   Cross-Dataset LODO (3 splits) | Calibration Analysis │
│   Retrieval Variant (N=2,500) | In-Depth Error Audit   │
└────────────────────────────────────────────────────────┘
```

### Critical Architectural Distinction
It is essential to distinguish between two distinct uses of generative text:
1. **Primary Evaluated Responses**: The benchmark responses whose factual validity is being classified. Internal generation signals are extracted by passing these exact prompt-response sequences through `Qwen3.5-0.8B` in an unadulterated forward pass.
2. **Behavioral Probes**: Newly generated stochastic responses ($k=5$ per prompt at temperature $0.7$) produced specifically to measure output stability. These sampled outputs serve as contextual neighborhood probes; they are not themselves labeled or evaluated for factual correctness.

---

## 9. Internal Generation Signals

The primary detector relies on 11 internal generation signals extracted from the raw logit vectors $\mathbf{z}_t \in \mathbb{R}^V$ across the $R$ tokens of the evaluated response:

### 9.1 Token Probability Signals
1. **Mean Token Probability (`mean_token_prob`)**: Arithmetic mean of the model's assigned token probabilities:
   $$\bar{P} = \frac{1}{R} \sum_{t=0}^{R-1} P(y_t \mid x, y_{<t})$$
2. **Minimum Token Probability (`min_log_prob`)**: The lowest log-probability assigned to any individual token in the sequence:
   $$\log P_{\min} = \min_{t \in \{0, \dots, R-1\}} \log P(y_t \mid x, y_{<t})$$
   Identifies local "bottleneck" tokens where the model exhibited sharp confidence drops.
3. **Mean Log Probability (`mean_log_prob`)**: Average log-likelihood across the response sequence:
   $$\overline{\log P} = \frac{1}{R} \sum_{t=0}^{R-1} \log P(y_t \mid x, y_{<t})$$
4. **Token Probability Standard Deviation (`token_prob_std`)**: Dispersion of token probabilities across the sequence, measuring certainty volatility:
   $$\sigma_P = \sqrt{\frac{1}{R-1} \sum_{t=0}^{R-1} (P(y_t) - \bar{P})^2}$$

### 9.2 Predictive Entropy Signals
5. **Mean Predictive Entropy (`mean_entropy`)**: Average Shannon entropy of the vocabulary probability distribution:
   $$\bar{H} = \frac{1}{R} \sum_{t=0}^{R-1} H_t, \quad \text{where } H_t = -\sum_{v \in V} P(v) \log P(v)$$
   Computed in float32 with zero-limiting ($\lim_{p \to 0^+} p \log p = 0$) to eliminate numerical NaN values.
6. **Maximum Predictive Entropy (`max_entropy`)**: Peak distribution entropy observed across the sequence:
   $$H_{\max} = \max_{t} H_t$$
7. **Entropy Standard Deviation (`entropy_std`)**: Variability of predictive entropy across decoding steps:
   $$\sigma_H = \sqrt{\frac{1}{R-1} \sum_{t=0}^{R-1} (H_t - \bar{H})^2}$$

### 9.3 Sequence Perplexity
8. **Log Perplexity (`log_perplexity`)**: Log-transformed sequence perplexity, bounded for numerical stability:
   $$\text{log\_perplexity} = \log\left(\max\left(\exp(-\overline{\log P}), 10^{-12}\right)\right)$$

### 9.4 Token Rank Dispersion Signals
Token rank represents the 1-based index of the actual generated token when vocabulary logits are sorted in descending order (where rank 1 represents the greedy choice):
$$\text{Rank}(y_t) = 1 + \sum_{v \in V} \mathbb{I}(z_{t, v} > z_{t, y_t})$$
9. **Log Mean Token Rank (`log_mean_token_rank`)**: $\log(1 + \frac{1}{R}\sum_{t} \text{Rank}(y_t))$.
10. **Log Maximum Token Rank (`log_max_token_rank`)**: $\log(1 + \max_t \text{Rank}(y_t))$, capturing extreme off-beam tokens.
11. **Log Rank Standard Deviation (`log_rank_std`)**: $\log(1 + \sigma_{\text{Rank}})$, measuring rank volatility.

---

## 10. Self-Consistency

To measure whether an LLM's response reflects a stable factual belief rather than an arbitrary stochastic path, we implemented a production self-consistency pipeline:
- **Sampling Protocol**: For every prompt $x$ in the $4,000$-example dataset, `Qwen3.5-0.8B` generated $k = 5$ independent stochastic responses using temperature $T = 0.7$, top-$p = 0.9$, and maximum generation length of $128$ tokens. This produced exactly $20,000$ raw output sequences.
- **Embedding Representation**: Each generated response was embedded into a 384-dimensional dense vector using `sentence-transformers/all-MiniLM-L6-v2` with unit $L_2$ normalization.
- **Extracted Consistency Features (5)**:
  1. `exact_match_agreement`: Fraction of candidate pairs that are character-for-character identical.
  2. `mean_pairwise_similarity`: Average pairwise cosine similarity across all $\binom{5}{2} = 10$ response pairs.
  3. `min_pairwise_similarity`: Minimum cosine similarity among the 10 pairs.
  4. `max_pairwise_similarity`: Maximum cosine similarity among the 10 pairs.
  5. `pairwise_similarity_std`: Standard deviation of the pairwise similarity distribution.

---

## 11. NLI Agreement

Semantic similarity alone does not detect directional factual contradictions (e.g., "Paris is in France" versus "Paris is not in France" have high embedding cosine similarity but represent opposite truth values). To address this, we integrated cross-encoder Natural Language Inference (NLI):
- **Model**: `cross-encoder/nli-MiniLM2-L6-H768`, a sequence-pair classification model pre-trained on MNLI and SNLI.
- **Evaluation Scale**: For each example, all $10$ unordered generation pairs ($40,000$ unordered pairs across the $4,000$-example corpus) were evaluated bidirectionally:
  $$\text{Pair } (y_i, y_j) \implies (y_i \to y_j) \text{ and } (y_j \to y_i)$$
  This yielded $20$ directional forward passes per example and $80,000$ forward evaluations across the $4,000$-example corpus.
- **Softmax Normalization**: Each evaluation produces three normalized probabilities: $P(\text{entailment})$, $P(\text{neutral})$, and $P(\text{contradiction})$.
- **Extracted NLI Features (3)**:
  1. `mean_pairwise_entailment`: Average entailment probability across all 20 directed evaluations.
  2. `mean_pairwise_contradiction`: Average contradiction probability across all 20 directed evaluations.
  3. `nli_disagreement`: Composite disagreement index:
     $$\text{nli\_disagreement} = \text{mean\_pairwise\_contradiction} + 0.5 \times (1.0 - \text{mean\_pairwise\_entailment})$$

---

## 12. Retrieval-Augmented Variant

To assess whether external evidence retrieval improves hallucination detection, a standalone retrieval-augmented pipeline was constructed:

> [!NOTE]
> **Population Constraints and Experimental Scope**:
> - Evaluated strictly on the $N = 2,500$ subset comprising HaluEval ($1,500$) and FEVER ($1,000$), where local evidence passages or entity pointer annotations were available.
> - **TruthfulQA ($1,500$ instances) was excluded** because its context field contained web reference URLs rather than textual passages, and live external web queries were prohibited.
> - Evaluated on a dedicated 500-instance holdout test set ($2,000$ train / $500$ test, balanced $250/250$).
> - This variant is **standalone** and is not part of the primary 19-feature Universal Core Detector.

- **Corpus Construction**:
  - HaluEval: $1,499$ unique, clean reference passages extracted from benchmark metadata.
  - FEVER: $829$ entity title passages extracted from Wikipedia pointer annotations.
- **Retrieval Engine**: Dense semantic search with `all-MiniLM-L6-v2` indexing all $2,328$ passages, retrieving the top-$k = 3$ passages via cosine similarity against the prompt and response.
- **Extracted Retrieval Features (6)**:
  1. `top1_evidence_similarity`: Cosine similarity between prompt and the top-ranked passage.
  2. `top3_evidence_similarity`: Mean cosine similarity of top-3 retrieved passages to the prompt.
  3. `response_top1_similarity`: Cosine similarity between evaluated response and the top passage.
  4. `response_top3_similarity`: Mean cosine similarity of top-3 passages to the response.
  5. `evidence_response_margin`: Difference between response-evidence similarity and prompt-evidence similarity.
  6. `retrieval_agreement`: Binary indicator denoting whether top-ranked prompt and response passages match.

---

## 13. Machine Learning Models

Two supervised tabular learning algorithms were evaluated:
1. **Logistic Regression**:
   - Linear baseline with $L_2$ regularization.
   - Hyperparameters: $C = 1.0$, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`.
   - Feature Scaling: Features are standardized via `StandardScaler` fitted strictly on the training partition:
     $$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
2. **XGBoost (Extreme Gradient Boosting)**:
   - Non-parametric tree ensemble capturing non-linear interactions and threshold dynamics.
   - Hyperparameters: `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0.1`, `eval_metric='logloss'`, `random_state=42`.
   - Tree structures are inherently invariant to monotonic feature scaling, requiring no standardizer.

### Data Leakage Safeguards
To maintain strict statistical separation between training and test distributions:
- Partitioning was executed once deterministically (`random_state=42`) using stratified joint key `source_dataset + "_" + label`.
- Preprocessing transformers (`StandardScaler`), calibration mappings, and cross-dataset scalers were fit strictly on training splits.
- Feature definitions, token ranks, and self-consistency statistics were computed per instance without cross-sample pooling.
- Metadata and non-signal columns—specifically `id`, `prompt`, `response`, `context`, `model_input`, `source_dataset`, `forward_time`, and sequence token length `num_tokens`—were strictly excluded from the classifier feature space, preventing label leakage and length confounding.

---

## 14. Evaluation Metrics

Model performance was evaluated across complementary statistical dimensions:

### 14.1 Classification & Discrimination
- **Accuracy**: Fraction of correct binary predictions at threshold $\tau = 0.50$: $\frac{\text{TP} + \text{TN}}{N}$.
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$, measuring positive predictive value.
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$, measuring sensitivity to hallucinations.
- **F1 Score**: Harmonic mean of Precision and Recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, measuring threshold-independent ranking ability.
- **PR-AUC**: Area under the Precision-Recall curve (average precision), measuring ranking discrimination under class trade-offs.

### 14.2 Probability Calibration
- **Brier Score**: Mean squared difference between predicted probabilities $\hat{p}_i$ and binary labels $z_i$:
  $$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - z_i)^2$$
- **Expected Calibration Error (ECE)**: Weighted average difference between predicted confidence and observed accuracy across $M = 10$ equal-width probability bins $B_m$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$
- **Maximum Calibration Error (MCE)**: Maximum deviation across any individual probability bin:
  $$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

---

## 15. Experimental Results

The following subsections record the exact, validated metrics from [`docs/master_results_table.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/master_results_table.md).

### Table A: Phase 6 Baseline vs. Phase 14 Universal Core ($N=800$ Holdout Test)
| Pipeline / Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 6 Baseline (11 features)** | | | | | | | | |
| Logistic Regression | 0.6212 | 0.6120 | 0.6625 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| XGBoost | 0.6700 | 0.6518 | 0.7300 | 0.6887 | 0.7411 | 0.7322 | 0.1999 | 0.0228 |
| **Phase 14 Universal Core (19 features)** | | | | | | | | |
| Logistic Regression | 0.6288 | 0.6241 | 0.6475 | 0.6356 | 0.7079 | 0.7124 | 0.2151 | 0.0525 |
| XGBoost | 0.6788 | 0.6829 | 0.6675 | 0.6751 | 0.7688 | 0.7795 | 0.1891 | 0.0341 |

### Table B: Phase 15 Feature-Family Ablation Study ($N=800$ Holdout Test)
| Feature Configuration | Features | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Internal Only | 11 | Logistic Regression | 0.6212 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| Internal Only | 11 | XGBoost | 0.6625 | 0.6793 | 0.7397 | 0.7329 | 0.1993 | 0.0228 |
| Self-Consistency Only | 5 | Logistic Regression | 0.4900 | 0.4545 | 0.4845 | 0.4863 | 0.2515 | 0.0300 |
| Self-Consistency Only | 5 | XGBoost | 0.5100 | 0.5410 | 0.4896 | 0.4870 | 0.2567 | 0.0450 |
| NLI Only | 3 | Logistic Regression | 0.5038 | 0.5031 | 0.5050 | 0.5025 | 0.2503 | 0.0088 |
| NLI Only | 3 | XGBoost | 0.4713 | 0.4885 | 0.4912 | 0.5005 | 0.2543 | 0.0731 |
| Internal + Self-Consistency | 16 | Logistic Regression | 0.6200 | 0.6311 | 0.7036 | 0.7043 | 0.2159 | 0.0613 |
| Internal + Self-Consistency | 16 | XGBoost | 0.6512 | 0.6610 | 0.7563 | 0.7675 | 0.1918 | 0.0450 |
| Internal + NLI | 14 | Logistic Regression | 0.6225 | 0.6308 | 0.7033 | 0.6994 | 0.2161 | 0.0576 |
| Internal + NLI | 14 | XGBoost | 0.6700 | 0.6615 | 0.7659 | 0.7697 | 0.1909 | 0.0257 |
| Self-Consistency + NLI | 8 | Logistic Regression | 0.4750 | 0.4697 | 0.4874 | 0.4795 | 0.2516 | 0.0475 |
| Self-Consistency + NLI | 8 | XGBoost | 0.5062 | 0.5153 | 0.5053 | 0.5128 | 0.2541 | 0.0522 |
| All Three (Universal Core) | 19 | Logistic Regression | 0.6288 | 0.6356 | 0.7079 | 0.7124 | 0.2151 | 0.0525 |
| All Three (Universal Core) | 19 | XGBoost | 0.6788 | 0.6751 | 0.7688 | 0.7795 | 0.1891 | 0.0341 |

### Table C: Phase 16 Cross-Dataset Transfer Generalization (LODO)
| Test Dataset (Unseen) | Training Datasets | Feature Configuration | Model | Accuracy | ROC-AUC |
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

### Table D: Phase 17 Probability Calibration Analysis ($N=800$ Holdout Test)
| Model | Calibration Method | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | Uncalibrated | 0.6338 | 0.6387 | 0.7069 | 0.7096 | 0.2155 | 0.0444 | 0.1248 |
| Logistic Regression | Sigmoid / Platt | 0.6300 | 0.6373 | 0.7069 | 0.7096 | 0.2165 | 0.0547 | 0.1114 |
| Logistic Regression | Isotonic | 0.6225 | 0.6505 | 0.6971 | 0.6751 | 0.2151 | 0.0341 | 0.2473 |
| XGBoost | Uncalibrated | 0.6488 | 0.6492 | 0.7638 | 0.7787 | 0.1900 | 0.0712 | 0.1888 |
| XGBoost | Sigmoid / Platt | 0.6550 | 0.6452 | 0.7638 | 0.7787 | 0.1935 | 0.0825 | 0.1210 |
| XGBoost | Isotonic | 0.6675 | 0.6472 | 0.7641 | 0.7557 | 0.1890 | 0.0244 | 0.1256 |

### Table E: Phase 19 Retrieval-Augmented Standalone Variant ($N=500$ Test, $N=2,500$ Usable Subset)
| Condition | Features | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Condition A: Internal-Only | 11 | Logistic Regression | 0.7460 | 0.7581 | 0.8463 | 0.8486 | 0.1597 | 0.0494 |
| Condition A: Internal-Only | 11 | XGBoost | 0.7880 | 0.8000 | 0.8978 | 0.8982 | 0.1264 | 0.0323 |
| Condition B: Universal Core | 19 | Logistic Regression | 0.7580 | 0.7660 | 0.8582 | 0.8657 | 0.1538 | 0.0738 |
| Condition B: Universal Core | 19 | XGBoost | 0.7860 | 0.7864 | 0.9011 | 0.9035 | 0.1246 | 0.0482 |
| Condition C: Retrieval-Only | 6 | Logistic Regression | 0.6500 | 0.6824 | 0.6990 | 0.6543 | 0.2187 | 0.0350 |
| Condition C: Retrieval-Only | 6 | XGBoost | 0.6800 | 0.7193 | 0.7491 | 0.7293 | 0.2053 | 0.0523 |
| Condition D: Internal + Retrieval | 17 | Logistic Regression | 0.7660 | 0.7754 | 0.8637 | 0.8675 | 0.1505 | 0.0380 |
| Condition D: Internal + Retrieval | 17 | XGBoost | 0.8060 | 0.8159 | 0.9184 | 0.9214 | 0.1180 | 0.0593 |
| Condition E: Universal + Retrieval | 25 | Logistic Regression | 0.7620 | 0.7680 | 0.8706 | 0.8823 | 0.1459 | 0.0606 |
| Condition E: Universal + Retrieval | 25 | XGBoost | 0.8120 | 0.8178 | 0.9169 | 0.9201 | 0.1185 | 0.0518 |

### Table F: Phase 14 Dataset-Level Breakdown ($N=800$ Holdout Test)
| Dataset | Test Samples (N) | Task Domain | Model | Accuracy | ROC-AUC |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **HaluEval** | 300 | Dialogue, QA, Summarization (with context) | Logistic Regression | 0.8400 | 0.9311 |
| **HaluEval** | 300 | Dialogue, QA, Summarization (with context) | XGBoost | 0.8633 | 0.9433 |
| **TruthfulQA** | 300 | Misconceptions & False Beliefs (closed-book) | Logistic Regression | 0.5133 | 0.5210 |
| **TruthfulQA** | 300 | Misconceptions & False Beliefs (closed-book) | XGBoost | 0.5933 | 0.6066 |
| **FEVER** | 200 | Wikipedia Claim Verification (factoid claims) | Logistic Regression | 0.4850 | 0.4870 |
| **FEVER** | 200 | Wikipedia Claim Verification (factoid claims) | XGBoost | 0.5300 | 0.5441 |

---

## 16. Error Analysis

Phase 18 conducted a post-hoc descriptive error audit on the Phase 14 Universal Core holdout test predictions ($N=800$):

### 16.1 Overall Errors and Confusion Breakdown
- **Logistic Regression**: Total test errors numbered $297$ out of $800$ ($37.12\%$ error rate; $62.88\%$ accuracy). False Positives numbered $156$ ($39.0\%$ FPR among faithful samples); False Negatives numbered $141$ ($35.25\%$ FNR among hallucinated samples).
- **XGBoost**: Total test errors numbered $257$ out of $800$ ($32.12\%$ error rate; $67.88\%$ accuracy). False Positives numbered $124$ ($31.0\%$ FPR among faithful samples); False Negatives numbered $133$ ($33.25\%$ FNR among hallucinated samples).

### 16.2 Dataset-Level Disparities
Classification errors were distributed unevenly across datasets:

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

- On **HaluEval** ($N=300$), XGBoost achieved an error rate of `0.1367` ($41$ errors) with an ROC-AUC of `0.9433`.
- On **TruthfulQA** ($N=300$), XGBoost achieved an error rate of `0.4067` ($122$ errors) with an ROC-AUC of `0.6066`.
- On **FEVER** ($N=200$), XGBoost achieved an error rate of `0.4700` ($94$ errors) with an ROC-AUC of `0.5441`.
- Combined, TruthfulQA and FEVER accounted for $216$ of the $257$ total XGBoost errors ($84.0\%$).

### 16.3 Model Agreement & Disagreement Matrix
- Both models produced matching binary classifications on $580$ of $800$ instances ($72.50\%$ agreement rate).
- The models disagreed on $220$ instances ($27.50\%$).
- Both models were simultaneously correct on $413$ instances ($51.62\%$) and simultaneously incorrect on $167$ instances ($20.88\%$).
- Logistic Regression was correct while XGBoost was incorrect on $90$ instances ($11.25\%$).
- XGBoost was correct while Logistic Regression was incorrect on $130$ instances ($16.25\%$).

### 16.4 Descriptive Patterns and Standardized Effect Sizes
1. **Response Word Count Profile**:
   - Correctly classified faithful responses (True Negatives) averaged `4.06` words.
   - Faithful responses misclassified as hallucinations (False Positives) averaged `9.65` words for LR and `8.74` words for XGBoost.
   - Correctly identified hallucinations (True Positives) averaged `10.75` words for LR and `10.96` words for XGBoost.
   - Missed hallucinations (False Negatives) averaged `6.01` words for LR and `5.76` words for XGBoost.
2. **Confidence Dynamics Near the Decision Boundary**:
   - Model errors clustered closer to the decision boundary: False Positives averaged predicted probabilities of `0.6197` (LR) and `0.6074` (XGBoost), while False Negatives averaged `0.3795` (LR) and `0.3806` (XGBoost).
   - In contrast, confident correct predictions were separated from $0.50$: True Positives averaged `0.6842` (LR) and `0.7394` (XGBoost), while True Negatives averaged `0.3054` (LR) and `0.2078` (XGBoost).
3. **Feature Separation Effect Sizes (Cohen's $d$)**:
   - Between True Negatives and False Positives, `min_log_prob` exhibited a large effect size ($d = 1.18$), and `log_max_token_rank` exhibited $d = -1.02$.
   - Between overall Correct and Incorrect predictions, `mean_token_prob` had $d = 0.43$, `mean_entropy` had $d = -0.43$, and `log_max_token_rank` had $d = -0.42$.
   - These measurements represent observational statistical properties of the test split; they do not imply causal mechanisms.

---

## 17. Discussion

The empirical findings offer several insights into the behavior of lightweight hallucination classifiers:

1. **Information Content of Internal Token Uncertainty**: The observation that the 11-feature baseline achieved an ROC-AUC of `0.7411` with XGBoost demonstrates that autoregressive decoding dynamics contain predictive signal for hallucination detection. Tokens generated during unsupported statements tend to exhibit lower likelihood and higher entropy spikes.
2. **Multi-Signal Complementarity**: In the ablation experiments, behavioral consistency ($k=5$) and NLI agreement alone achieved near-chance classification (ROC-AUC `0.4896` and `0.4912`), but combining them with internal signals increased XGBoost ROC-AUC from `0.7397` to `0.7688`. This suggests that consistency probes act as contextual regularizers that enhance internal probability features rather than standalone detectors.
3. **Linear versus Non-Linear Modeling**: Across all feature conditions, XGBoost consistently recorded higher ranking discrimination than Logistic Regression (Universal Core ROC-AUC `0.7688` vs `0.7079`). This indicates that non-linear feature interactions and threshold splits better capture complex relationships between token ranks, entropy volatility, and pairwise similarity.
4. **Separation of Correlation from Causation**: Although features like minimum log-probability and sequence length show strong descriptive correlations with error categories, these relationships cannot be interpreted causally. Artificially shortening a response or forcing greedy token selection does not inherently make an unsupported factual claim true.

---

## 18. Calibration Discussion

In safety-critical deployment, an estimated hallucination score must reliably reflect empirical error rates. Phase 17 demonstrated that:
- Raw uncalibrated probabilities produced by XGBoost exhibited an ECE of `0.0712` and MCE of `0.1888`.
- Non-parametric isotonic regression reduced ECE to `0.0244` (a $65.7\%$ reduction) and MCE to `0.1256`, aligning predicted risks with empirical frequencies while preserving ranking discrimination (ROC-AUC `0.7641`).
- Platt scaling (sigmoid calibration) strictly preserved the original rank ordering (ROC-AUC `0.7638` for XGBoost, `0.7069` for LR), while slightly increasing ECE (`0.0825`).

### Methodological Distinction
Phase 17 used an internal 3-way nested partition ($2,560$ base-training examples, $640$ calibration tuning examples, and $800$ untouched test examples). Because base models in Phase 17 were fitted on $2,560$ examples rather than the full $3,200$ examples used in Phase 14, uncalibrated metrics (XGBoost accuracy `0.6488`) are methodologically distinct and must not be conflated with the Phase 14 primary benchmark numbers (`0.6788`).

---

## 19. Cross-Dataset Generalization Discussion

The Leave-One-Dataset-Out (LODO) experiments in Phase 16 exposed significant challenges in cross-domain transfer:
- When models trained on HaluEval and TruthfulQA were evaluated on FEVER, ROC-AUC dropped to `0.4875` (LR) and `0.5167` (XGBoost).
- When trained on HaluEval and FEVER and tested on TruthfulQA, ROC-AUC was `0.4575` (LR) and `0.4608` (XGBoost).
- When trained on TruthfulQA and FEVER and tested on HaluEval, ROC-AUC was `0.2583` (LR) and `0.4368` (XGBoost).

These substantial performance drops indicate that decision boundaries learned on one benchmark do not transfer zero-shot to another. This occurs because the three benchmarks capture fundamentally different error modes: HaluEval measures context unfaithfulness in text generation, TruthfulQA tests semantic misconceptions in open-ended answering, and FEVER assesses binary claim verification against Wikipedia facts. Without multi-benchmark exposure during training, a classifier cannot learn a unified decision threshold.

---

## 20. Limitations

The scope of this research is bounded by several explicit limitations:

1. **Single Evaluated Causal LM**: All token probability, entropy, and generation dynamics were extracted exclusively from `Qwen3.5-0.8B`. Cross-model transfer to other model families (e.g., Llama, Gemma, Mistral) or larger parameter regimes ($7\text{B}+$) was not evaluated.
2. **Limited Zero-Shot Cross-Domain Generalization**: As demonstrated in Phase 16, models trained on a subset of datasets do not reliably generalize to unseen task structures.
3. **Retrieval Non-Uniformity**: Dense evidence retrieval was feasible only on HaluEval and FEVER ($N=2,500$); TruthfulQA was excluded due to the lack of local text passages.
4. **Benchmark-Derived Ground Truth**: Annotations reflect existing benchmark labels, which may contain labeling noise, heuristic simplifications, or annotator subjectivity.
5. **Contextual Sampling Nature of Behavioral Probes**: Self-consistency and NLI agreement features represent contextual probes generated for the same prompt, rather than direct scores of the original evaluated response.
6. **No Causal Mechanism**: Identified feature associations and effect sizes (e.g., Cohen's $d$) are purely observational; they do not establish that uncertainty dynamics cause factual inaccuracy.
7. **Offline Evaluation Setting**: All experiments were conducted offline in batch mode; latency, memory footprint, and concurrency under real-time production traffic were not evaluated.
8. **Sample Size Constraints**: The total dataset comprises $4,000$ examples ($800$ holdout test instances); larger datasets may reveal additional distributional nuances.
9. **Calibration Tuning Partition Size**: Calibration parameters were estimated on $640$ tuning instances, which can introduce variance into non-parametric isotonic step functions.

---

## 21. Threats to Validity

1. **Domain and Task Formulation Shift**: Differences in task format across benchmarks (dialogue summarization vs. factoid claim verification) introduce distribution shifts in feature space.
2. **Model Parameter Scale**: A compact 0.8B parameter model was selected to ensure local computational feasibility on consumer GPU hardware. Larger models may exhibit different overconfidence or calibration profiles.
3. **Length Association Confounds**: Descriptive analysis showed that false positives had higher average word counts than true negatives. While length was excluded as an input feature, token-level sums and averages can retain weak length correlations.
4. **Retrieval Corpus Scope**: Retrieval passages were drawn from benchmark metadata rather than a comprehensive, dynamic open-web corpus.
5. **Partitioning Realization**: Primary benchmarks reflect a specific stratified 80/20 train/test realization with `random_state=42`.

---

## 22. Future Work

The following extensions are proposed as promising directions for future research:

1. **Multi-Model Cross-Validation**: Evaluate whether internal generation signals from different model families (Llama 3, Gemma 2, Mistral) share compatible feature distributions or require model-specific normalization.
2. **Scaling to Larger Parameter Regimes**: Test the extraction pipeline on 7B, 14B, and 70B parameter models to evaluate whether larger models exhibit sharper uncertainty signals.
3. **Dynamic Open-Domain Retrieval**: Integrate open-web retrieval APIs to provide external evidence for open-domain benchmarks like TruthfulQA.
4. **Real-Time Latency and Memory Benchmarking**: Measure the exact inference latency of extracting forward logits versus running secondary generative evaluation passes in a streaming production environment.
5. **Hierarchical and Domain-Conditioned Modeling**: Train multi-task or domain-aware classifiers that condition decision thresholds on detected task domains to mitigate cross-dataset degradation.

---

## 23. Conclusion

This research investigated whether Large Language Model hallucination risk can be estimated accurately and efficiently using white-box internal generation uncertainty signals, stochastic self-consistency, and cross-encoder Natural Language Inference agreement evaluated with lightweight supervised classifiers. Across a 4,000-example harmonized benchmark evaluated with `Qwen3.5-0.8B`, internal token probabilities, predictive entropy, sequence perplexity, and token rank distributions provided clear baseline predictive signal (Phase 6 Baseline: XGBoost ROC-AUC `0.7411`, Logistic Regression ROC-AUC `0.6930`). 

Integrating multi-sample self-consistency and NLI agreement into the 19-feature Universal Core Detector improved threshold-free ranking discrimination (Phase 14 Universal Core: XGBoost ROC-AUC `0.7688`, PR-AUC `0.7795`), with isotonic regression successfully reducing Expected Calibration Error from `0.0712` to `0.0244`. Simultaneously, the investigation established explicit operational boundaries: cross-dataset transfer experiments demonstrated substantial performance drops under Leave-One-Dataset-Out evaluation, confirming that decision thresholds do not generalize across distinct task formulations without multi-domain training. Furthermore, internal signals provided strong separation primarily when reference context was present (HaluEval ROC-AUC `0.9433`), whereas closed-book factoid claims (FEVER ROC-AUC `0.5441`) required external evidence retrieval to attain high accuracy. These empirical results demonstrate the feasibility of lightweight, multi-signal hallucination detection while underscoring the critical necessity of domain-aware training and rigorous probability calibration.

---

## 24. Reproducibility

To ensure strict scientific transparency and reproducibility across environments:

### 24.1 Software and Hardware Environments
The repository contains verified records for two distinct execution environments:
1. **Documented Historical Baseline Environment** (documented in `docs/environment.md` and `README.md`):
   - **Operating System**: Microsoft Windows 11 Home Single Language (64-bit, Version 10.0.26200)
   - **Hardware Mode**: CPU Execution Mode (Intel Iris Xe Graphics, discrete NVIDIA GPU none detected / CUDA unavailable)
   - **Core Packages**: Python `3.13.14`, PyTorch `2.14.0+cpu`, Scikit-Learn `1.9.1`, XGBoost `3.4.1`, Transformers `5.17.0`, Sentence-Transformers `6.0.1`, Datasets `5.0.1`, FAISS (`faiss-cpu`) `1.15.1`.
2. **Current Verification and Report-Audit Runtime** (active local environment `.venv`):
   - **Operating System**: Microsoft Windows 11 (AMD64)
   - **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, Driver 576.06, CUDA 12.4 enabled)
   - **Core Packages**: Python `3.11.9`, PyTorch `2.6.0+cu124`, Scikit-Learn `1.9.1`, XGBoost `3.2.0`, Transformers `5.17.0`, Sentence-Transformers `6.0.1`.
3. **Repository Dependency Specification**:
   - `requirements.txt` specifies unpinned package requirements (`numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `jupyter`, `tqdm`, `datasets`, `transformers`, `sentence-transformers`, `torch`, `rank-bm25`, `faiss-cpu`, `nltk`, `joblib`, `streamlit`, `requests`, `evaluate`), allowing cross-platform deployment across both CPU-only and CUDA-enabled configurations.

### 24.2 Determinism and Persistence
- **Random Seeds**: All train/test partitions, model seeds, and stochastic samplers adhered strictly to `random_state=42`.
- **Repository Artifacts**: All raw generations (`full_generations_raw.parquet`), pairwise NLI scores (`full_nli_pairwise.parquet`), feature tables (`universal_features.parquet`), and evaluation metrics are persisted locally in `experiments/` and `data/processed/`.
- **Test Suite Status**: 198 unit tests passing across all pipeline components (`pytest -q`).

---

## 25. References

Bibliographic references to be added from the project's verified literature sources.

1. Large language model hallucination surveys and taxonomies. [REFERENCE NEEDED]
2. Token likelihood and uncertainty estimation in autoregressive models. [REFERENCE NEEDED]
3. Predictive entropy and perplexity in natural language processing. [REFERENCE NEEDED]
4. Self-consistency and stochastic sampling for language model reasoning. [REFERENCE NEEDED]
5. Natural Language Inference for factual consistency evaluation. [REFERENCE NEEDED]
6. Dense passage retrieval and open-domain question answering. [REFERENCE NEEDED]
7. Gradient boosted decision trees and tabular classification. [REFERENCE NEEDED]
8. Probability calibration and Expected Calibration Error in machine learning. [REFERENCE NEEDED]

---

## 26. Appendix

### 26.1 Complete 19-Feature Universal Core Matrix
| Feature Index | Feature Name | Group | Type | Description |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `min_log_prob` | Internal Signals | Continuous | Minimum log-probability across response tokens |
| 2 | `mean_log_prob` | Internal Signals | Continuous | Arithmetic mean of token log-probabilities |
| 3 | `mean_token_prob` | Internal Signals | Continuous | Arithmetic mean of token probabilities |
| 4 | `token_prob_std` | Internal Signals | Continuous | Standard deviation of token probabilities |
| 5 | `mean_entropy` | Internal Signals | Continuous | Mean Shannon predictive entropy across tokens |
| 6 | `max_entropy` | Internal Signals | Continuous | Maximum predictive entropy observed in sequence |
| 7 | `entropy_std` | Internal Signals | Continuous | Standard deviation of predictive entropy |
| 8 | `log_perplexity` | Internal Signals | Continuous | Log-transformed sequence perplexity |
| 9 | `log_mean_token_rank` | Internal Signals | Continuous | $\log(1 + \text{mean token rank})$ |
| 10 | `log_max_token_rank` | Internal Signals | Continuous | $\log(1 + \text{max token rank})$ |
| 11 | `log_rank_std` | Internal Signals | Continuous | $\log(1 + \text{token rank standard deviation})$ |
| 12 | `exact_match_agreement` | Self-Consistency | Continuous | Fraction of candidate pairs that match identically |
| 13 | `mean_pairwise_similarity` | Self-Consistency | Continuous | Mean embedding cosine similarity among $k=5$ outputs |
| 14 | `min_pairwise_similarity` | Self-Consistency | Continuous | Minimum embedding cosine similarity among $k=5$ outputs |
| 15 | `max_pairwise_similarity` | Self-Consistency | Continuous | Maximum embedding cosine similarity among $k=5$ outputs |
| 16 | `pairwise_similarity_std` | Self-Consistency | Continuous | Standard deviation of pairwise embedding similarities |
| 17 | `mean_pairwise_entailment` | NLI Agreement | Continuous | Mean cross-encoder entailment probability across pairs |
| 18 | `mean_pairwise_contradiction`| NLI Agreement | Continuous | Mean cross-encoder contradiction probability across pairs |
| 19 | `nli_disagreement` | NLI Agreement | Continuous | Composite NLI disagreement metric |

### 26.2 Standalone Retrieval Features (Phase 19 Variant Only)
| Feature Index | Feature Name | Description |
| :---: | :--- | :--- |
| 1 | `top1_evidence_similarity` | Cosine similarity between prompt and top retrieved passage |
| 2 | `top3_evidence_similarity` | Mean cosine similarity between prompt and top-3 retrieved passages |
| 3 | `response_top1_similarity` | Cosine similarity between response and top retrieved passage |
| 4 | `response_top3_similarity` | Mean cosine similarity between response and top-3 retrieved passages |
| 5 | `evidence_response_margin` | Margin between response-evidence and prompt-evidence similarity |
| 6 | `retrieval_agreement` | Binary match indicator between top prompt and response evidence |

### 26.3 Experiment Phase Map
- **Phase 3**: Dataset ingestion, harmonization, and canonical schema definition ($N=4,000$).
- **Phase 4–5**: Internal generation signal extraction module and mathematical verification.
- **Phase 6–7**: Baseline classifiers (11 features) and sequence token length ablation.
- **Phase 10–11**: Production self-consistency generation ($20,000$ raw outputs) and embedding aggregation.
- **Phase 12**: Production pairwise NLI scoring ($40,000$ pairs) via `cross-encoder/nli-MiniLM2-L6-H768`.
- **Phase 13**: Universal feature matrix assembly (19 canonical features).
- **Phase 14**: Universal Core Detector training and primary benchmark holdout evaluation.
- **Phase 15**: Systematic feature-family ablation study across 7 conditions.
- **Phase 16**: Leave-One-Dataset-Out (LODO) cross-dataset transfer generalization.
- **Phase 17**: Leak-free probability calibration analysis (Platt scaling and isotonic regression).
- **Phase 18**: Comprehensive descriptive error analysis, length profiling, and disagreement audit.
- **Phase 19**: Standalone retrieval-augmented detector variant on $N=2,500$ usable instances.
- **Phase 20**: Final experiment inventory, master results table, research findings, and project report.

### 26.4 Classification Model Hyperparameters
- **Logistic Regression**: `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`, `StandardScaler(with_mean=True, with_std=True)` fitted on training split.
- **XGBoost Classifier**: `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0.1`, `eval_metric='logloss'`, `random_state=42`, `n_jobs=-1`.

### 26.5 Test Suite Status
All pipeline components are validated by automated unit tests:
- `tests/test_data.py`: Dataset schema, balance, and preprocessing integrity.
- `tests/test_token_signals.py`: Signal extraction math, entropy limits, and numerical stability.
- `tests/test_baseline_models.py`: Phase 6 baseline training and metrics reproduction.
- `tests/test_self_consistency.py` & `test_self_consistency_full.py`: Sampling and embedding aggregation.
- `tests/test_nli_agreement.py` & `test_nli_full.py`: Pairwise NLI mechanics and aggregation.
- `tests/test_universal_features.py` & `test_combined_models.py`: 19-feature matrix assembly and Phase 14 training.
- `tests/test_ablation_study.py`: 7 ablation conditions and delta calculations.
- `tests/test_cross_dataset.py`: 3 LODO splits and transfer evaluations.
- `tests/test_calibration.py`: Nested calibration splitting and ECE reductions.
- `tests/test_error_analysis.py`: Holdout error counts and confusion matrices.
- `tests/test_retrieval_augmented.py`: Retrieval corpus indexing and variant evaluations.
- `tests/test_final_inventory.py`, `test_master_results_table.py`, `test_final_research_findings.py`, `test_final_project_report.py`: Documentation and consistency integrity.
- **Total Passing Tests**: 198 passed in 15.89s.
