# Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%7C%20RTX%203050-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit Demo](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

Large Language Models (LLMs) frequently generate plausible-sounding falsehoods and confabulations, commonly known as **hallucinations**. Most commercial mitigation frameworks treat the LLM as an opaque black box—relying on expensive multi-agent consensus, recursive self-prompting, or latency-prohibitive open-web retrieval.

This research project introduces the **Universal Core Detector**: an ultra-lightweight, interpretable, multi-signal supervised classifier that estimates hallucination risk in real-time from the LLM's own internal generation dynamics, stochastic consistency behaviors, and Natural Language Inference (NLI) agreements.

By fusing **11 internal generation signals** (token log-probabilities, Shannon entropy, perplexity, and token rank dispersion) with **5 behavioral self-consistency signals** and **3 cross-encoder NLI agreement features**, our classifier achieves up to **0.7913 ROC-AUC** (Gemma 3 1B) and **0.7688 ROC-AUC** (Qwen 3.5 0.8B) on an 800-instance holdout test set across 4,000 curated benchmark examples. When external evidence retrieval is available, an augmented variant reaches **0.9184 ROC-AUC**.

```
                       User Question + Optional Context
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       Multi-Model Causal Generative Engine       │
             │  (Qwen 3.5 0.8B / Llama 3.2 1B / Gemma 3 1B IT)  │
             │       Greedy Decoding (T=0, bfloat16/float16)    │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                        Primary Generated Response
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐         ┌──────────────────┐
│  Stage 3 & 4:    │        │  Stage 5, 6 & 7: │         │  Stage 8 & 9:    │
│  11 Internal     │        │  5 Consistency   │         │  3 NLI Agreement │
│  Generation      │        │  Behavioral      │         │  Cross-Encoder   │
│  Signals         │        │  Probes (k=5)    │         │  Passes          │
│                  │        │                  │         │                  │
│ • min/mean log-P │        │ • k=5 stochastic │         │ • nli-MiniLM2-L6 │
│ • mean prob, std │        │   generations    │         │ • 20 directional │
│ • Shannon entropy│        │ • all-MiniLM-L6  │         │   passes (k(k-1))│
│ • log perplexity │        │ • pairwise cos-  │         │ • entailment,    │
│ • rank dispersion│        │   similarity     │         │   contradiction, │
│   (mean/max/std) │        │ • exact match    │         │   disagreement   │
└────────┬─────────┘        └────────┬─────────┘         └────────┬─────────┘
         │                           │                            │
         └───────────────────────────┼────────────────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │            Stage 10:              │
                   │  19 Canonical Universal Features  │
                   │    (Strict Length Isolation)      │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │          Stage 11 & 12:           │
                   │  Supervised Risk Classification   │
                   │ • Tuned XGBoost (Primary Non-Lin) │
                   │ • Calibrated Logistic Regression  │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Hallucination Risk Score       │
                   │   [0.0 - 1.0 Posterior Risk]      │
                   └─────────────────┬─────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│   Stage 13:      │        │   Stage 14:      │        │   Stage 15:      │
│  TreeSHAP Feature│        │  Evidence        │        │  Token Heatmap & │
│  Attributions    │        │  Groundedness    │        │  Latency Profile │
│  ("Why flagged?")│        │  Margin (Opt.)   │        │  (P(y_t), H_t)   │
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

---

## 🎯 Key Research Contributions

1. **Multi-Signal Triangulation**: Proves empirically that internal generation certainty signals (token log-probabilities, Shannon entropy, and rank dispersion) form an indispensable anchor. Standalone behavioral probes (self-consistency, NLI) degenerate to random guessing without internal grounding (~0.49 ROC-AUC), but provide significant orthogonal gain (+0.0277 ROC-AUC, +0.0474 PR-AUC) when fused into the Universal Core.
2. **Multi-Model Universality**: Validated across three modern open-weight causal architectures with dedicated full pipeline retraining on 4,000 instances each:
   - **Google Gemma 3 (1B IT)**: Reaches **0.7913 ROC-AUC**, 0.7914 PR-AUC, 71.50% Accuracy.
   - **Qwen 3.5 (0.8B)**: Reaches **0.7688 ROC-AUC**, 0.7795 PR-AUC, 67.88% Accuracy.
   - **Meta Llama 3.2 (1B Instruct)**: Reaches **0.7450 ROC-AUC**, 0.7492 PR-AUC, 66.88% Accuracy.
3. **Strict Length Isolation Safeguard**: Formally enforces an anti-confounder policy by isolating sequence length (`num_tokens`), prompt text, and dataset metadata. Phase 7 ablation proved that `num_tokens` acts as an artificial shortcut that degrades true scientific validity.
4. **Rigorous Probability Calibration**: Evaluated using a 3-way nested partition ($2,560$ train / $640$ calibration / $800$ test). Non-parametric Isotonic Regression cuts Expected Calibration Error (ECE) for XGBoost from `0.0712` down to **`0.0244`** (a **65.7% reduction**).
5. **Cross-Dataset Generalization (LODO)**: Conducts a Leave-One-Dataset-Out transfer study (Phase 16) exposing significant cross-benchmark domain shifts, proving the necessity of multi-task training data.
6. **Live Interactive Web Demo**: Full-featured Streamlit application featuring live GPU-accelerated inference across all 3 models, token-level probability/entropy heatmaps, TreeSHAP attribution, and 15-stage millisecond latency profiling.

---

## 📊 Curated Benchmark Portfolio ($N = 4,000$)

All empirical experiments were conducted across a balanced, curated benchmark population of **4,000 instances** ($50\%$ faithful, $50\%$ hallucinated) spanning three complementary task formulations:

| Benchmark Dataset | Samples (N) | Task Domain | Grounding Context |
| :--- | :---: | :--- | :--- |
| **HaluEval** | 1,500 | Dialogue, QA, Summarization | In-context source text / passages provided |
| **TruthfulQA** | 1,500 | Human Misconceptions & Falsehoods | Closed-book adversarial question answering |
| **FEVER** | 1,000 | Wikipedia Factoid Claim Verification | Claim statements evaluated without open-web access |

The dataset is partitioned into a fixed, stratified **80/20 train/test split** ($3,200$ training / $800$ holdout test instances, `random_state=42`), preserving exact class and dataset distribution balance.

---

## 🧬 The 19 Canonical Universal Features

The Universal Core feature space excludes sequence length and is partitioned into three orthogonal signal families:

### 1. Internal Generation Dynamics (11 Features)
Extracted directly from the forward pass token logits and softmax distributions of the primary response ($T=0$ greedy decoding):
- `min_log_prob`: Minimum token log-probability $\min_t \ln P(y_t)$ across the generated sequence (captures worst-case uncertainty).
- `mean_log_prob`: Length-normalized average log-probability $\frac{1}{T}\sum_{t=1}^T \ln P(y_t)$.
- `mean_token_prob`: Average probability $\frac{1}{T}\sum_{t=1}^T P(y_t)$.
- `token_prob_std`: Standard deviation of token probabilities across the sequence.
- `mean_entropy`: Average Shannon entropy $\frac{1}{T}\sum_{t=1}^T H(P_t)$ measuring vocabulary distribution flatness.
- `max_entropy`: Maximum token entropy $\max_t H(P_t)$ identifying localized hallucination pivots.
- `entropy_std`: Standard deviation of predictive entropy across the generation.
- `log_perplexity`: Natural logarithm of sequence perplexity $\ln(\text{PPL})$.
- `log_mean_token_rank`: Logarithm of average token rank in unconstrained vocabulary softmax output.
- `log_max_token_rank`: Logarithm of maximum token rank (detects tail-token sampling).
- `log_rank_std`: Volatility / dispersion of token rank values across the generation.

### 2. Behavioral Self-Consistency Probes (5 Features)
Derived from $k=5$ stochastic candidate responses sampled at $T=0.7, \text{top\_}p=0.9, \text{max\_tokens}=128$:
- `exact_match_agreement`: Fraction of candidate completions that match the primary response exactly.
- `mean_pairwise_similarity`: Mean cosine similarity across all 10 response pairs encoded with `sentence-transformers/all-MiniLM-L6-v2`.
- `min_pairwise_similarity`: Minimum pairwise cosine similarity (lowest consensus pair).
- `max_pairwise_similarity`: Maximum pairwise cosine similarity.
- `pairwise_similarity_std`: Standard deviation of pairwise semantic similarities.

### 3. Natural Language Inference Agreement (3 Features)
Computed using `cross-encoder/nli-MiniLM2-L6-H768` evaluated across all $k(k-1) = 20$ directional hypothesis-premise pairs:
- `mean_pairwise_entailment`: Average directional entailment probability.
- `mean_pairwise_contradiction`: Average directional contradiction probability.
- `nli_disagreement`: Symmetric disagreement rate: $1 - \text{Entailment} + \text{Contradiction}$.

---

## 📈 Empirical Research Results

All metrics represent exact, validated outputs extracted from persisted experiment artifacts on the identical 800-instance holdout test partition.

### 1. Primary Multi-Model Benchmark (Universal Core 19 Features)

| Base Generative LLM | Classifier | Accuracy | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Gemma 3 (1B IT)** | **XGBoost** | **0.7150** | **0.7220** | **0.7913** | **0.7914** | **0.1854** | 0.0410 |
| Google Gemma 3 (1B IT) | Logistic Regression | 0.6475 | 0.6667 | 0.7092 | 0.7054 | 0.2140 | 0.0521 |
| **Qwen 3.5 (0.8B)** | **XGBoost** | **0.6788** | **0.6751** | **0.7688** | **0.7795** | **0.1891** | **0.0341** |
| Qwen 3.5 (0.8B) | Logistic Regression | 0.6288 | 0.6356 | 0.7079 | 0.7124 | 0.2151 | 0.0525 |
| Qwen 3.5 (0.8B) [Phase 6 Baseline] | XGBoost (11 signals) | 0.6700 | 0.6887 | 0.7411 | 0.7322 | 0.1999 | 0.0228 |
| Qwen 3.5 (0.8B) [Phase 6 Baseline] | Logistic Regression (11 signals)| 0.6212 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| **Meta Llama 3.2 (1B Instruct)** | **XGBoost** | **0.6688** | **0.6834** | **0.7450** | **0.7492** | 0.1991 | 0.0392 |
| Meta Llama 3.2 (1B Instruct) | Logistic Regression | 0.6512 | 0.6610 | 0.6929 | 0.6681 | 0.2211 | **0.0319** |

---

### 2. Feature-Family Ablation Study (Phase 15)

Evaluated on the 800-instance holdout test set to isolate individual and combined feature families:

| Feature Configuration | Features | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Internal Signals Only | 11 | Logistic Regression | 0.6212 | 0.6363 | 0.6930 | 0.6618 | 0.2177 | 0.0649 |
| Internal Signals Only | 11 | XGBoost | 0.6625 | 0.6793 | 0.7397 | 0.7329 | 0.1993 | 0.0228 |
| Self-Consistency Only | 5 | Logistic Regression | 0.4900 | 0.4545 | 0.4845 | 0.4863 | 0.2515 | 0.0300 |
| Self-Consistency Only | 5 | XGBoost | 0.5100 | 0.5410 | 0.4896 | 0.4870 | 0.2567 | 0.0450 |
| NLI Agreement Only | 3 | Logistic Regression | 0.5038 | 0.5031 | 0.5050 | 0.5025 | 0.2503 | 0.0088 |
| NLI Agreement Only | 3 | XGBoost | 0.4713 | 0.4885 | 0.4912 | 0.5005 | 0.2543 | 0.0731 |
| Internal + Self-Consistency | 16 | XGBoost | 0.6512 | 0.6610 | 0.7563 | 0.7675 | 0.1918 | 0.0450 |
| Internal + NLI Agreement | 14 | XGBoost | 0.6700 | 0.6615 | 0.7659 | 0.7697 | 0.1909 | 0.0257 |
| Self-Consistency + NLI | 8 | XGBoost | 0.5062 | 0.5153 | 0.5053 | 0.5128 | 0.2541 | 0.0522 |
| **All Three / Universal Core** | **19** | **Logistic Regression** | **0.6288** | **0.6356** | **0.7079** | **0.7124** | **0.2151** | **0.0525** |
| **All Three / Universal Core** | **19** | **XGBoost** | **0.6788** | **0.6751** | **0.7688** | **0.7795** | **0.1891** | **0.0341** |

---

### 3. Task Domain Subgroup Performance (Qwen 3.5 Universal Core, N=800)

| Benchmark / Task Slice | Samples (N) | Task Characteristics | XGB Accuracy | XGB ROC-AUC | LR Accuracy | LR ROC-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **HaluEval** | 300 | Dialogue & Summarization (with context) | **86.33%** | **0.9433** | 84.00% | 0.9311 |
| **TruthfulQA** | 300 | Adversarial Human Misconceptions | **59.33%** | **0.6066** | 51.33% | 0.5210 |
| **FEVER** | 200 | Wikipedia Factoid Claims (no web access)| **53.00%** | **0.5441** | 48.50% | 0.4870 |

---

### 4. Cross-Dataset Generalization (Phase 16 Leave-One-Dataset-Out / LODO)

Models were trained exclusively on two benchmark datasets and evaluated on the 100% unseen third dataset to assess zero-shot transferability:

| Test Dataset (Held-Out) | Training Datasets | Feature Configuration | Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Universal Core (19) | XGBoost | 0.5060 | 0.5167 |
| **FEVER** ($N=1,000$) | HaluEval + TruthfulQA ($N=3,000$) | Internal Only (11) | XGBoost | 0.4830 | 0.4903 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Universal Core (19) | XGBoost | 0.4853 | 0.4608 |
| **TruthfulQA** ($N=1,500$) | HaluEval + FEVER ($N=2,500$) | Internal Only (11) | XGBoost | 0.4793 | 0.4539 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Universal Core (19) | XGBoost | 0.4320 | 0.4368 |
| **HaluEval** ($N=1,500$) | TruthfulQA + FEVER ($N=2,500$) | Internal Only (11) | XGBoost | 0.3980 | 0.3599 |

*Finding*: Multi-signal integration provides measurable buffering against out-of-domain collapse (improving HaluEval ROC-AUC from 0.3599 to 0.4368).

---

### 5. Probability Calibration Analysis (Phase 17)

Evaluated using a leak-free 3-way nested partition ($2,560$ base training / $640$ calibration tuning / $800$ holdout test):

| Model | Calibration Method | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE | MCE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost | Uncalibrated | 0.6488 | 0.6492 | 0.7638 | 0.7787 | 0.1900 | 0.0712 | 0.1888 |
| XGBoost | Sigmoid (Platt) | 0.6550 | 0.6452 | 0.7638 | 0.7787 | 0.1935 | 0.0825 | 0.1210 |
| **XGBoost** | **Isotonic Regression**| **0.6675** | **0.6472** | **0.7641** | **0.7557** | **0.1890** | **0.0244** | **0.1256** |
| Logistic Regression | Uncalibrated | 0.6338 | 0.6387 | 0.7069 | 0.7096 | 0.2155 | 0.0444 | 0.1248 |
| Logistic Regression | Sigmoid (Platt) | 0.6300 | 0.6373 | 0.7069 | 0.7096 | 0.2165 | 0.0547 | 0.1114 |
| Logistic Regression | Isotonic Regression| 0.6225 | 0.6505 | 0.6971 | 0.6751 | 0.2151 | 0.0341 | 0.2473 |

*Finding*: Non-parametric Isotonic Regression reduces XGBoost ECE by **65.7%** (from `0.0712` down to `0.0244`), producing reliable posterior probabilities.

---

### 6. Standalone Retrieval-Augmented Detector (Phase 19)

Evaluated strictly on the $N = 2,500$ population where reference evidence passages exist (HaluEval: 1,500, FEVER: 1,000) using a 500-instance holdout test split ($2,000$ train / $500$ test, balanced 250/250):

| Condition | Features | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Condition A (Internal-Only) | 11 | Logistic Regression | 0.7460 | 0.7581 | 0.8463 | 0.8486 | 0.1597 | 0.0494 |
| Condition A (Internal-Only) | 11 | XGBoost | 0.7880 | 0.8000 | 0.8978 | 0.8982 | 0.1264 | 0.0323 |
| Condition B (Universal Core) | 19 | Logistic Regression | 0.7580 | 0.7660 | 0.8582 | 0.8657 | 0.1538 | 0.0738 |
| Condition B (Universal Core) | 19 | XGBoost | 0.7860 | 0.7864 | 0.9011 | 0.9035 | 0.1246 | 0.0482 |
| Condition C (Retrieval-Only) | 6 | Logistic Regression | 0.6500 | 0.6824 | 0.6990 | 0.6543 | 0.2187 | 0.0350 |
| Condition C (Retrieval-Only) | 6 | XGBoost | 0.6800 | 0.7193 | 0.7491 | 0.7293 | 0.2053 | 0.0523 |
| **Condition D (Internal + Retrieval)** | **17** | **XGBoost** | **0.8060** | **0.8159** | **0.9184** | **0.9214** | **0.1180** | **0.0593** |
| **Condition E (Universal Core + Retrieval)** | **25** | **XGBoost** | **0.8120** | **0.8178** | **0.9169** | **0.9201** | **0.1185** | **0.0518** |

---

## ⚙️ The 15 Staged Pipeline Execution Steps

Every live inference query in the application passes through 15 staged and timed execution steps:

| Stage | Step Name | Operation Details |
| :---: | :--- | :--- |
| **1** | **Model Loading & Cache Resolution** | Resolves the selected causal LM, tokenizer, embedding model, NLI cross-encoder, and trained classifiers from the global cache. |
| **2** | **Primary Response Generation** | Deterministic greedy generation ($T=0$) with model-specific prompt templates and thinking-token suppression. |
| **3** | **Token & Logit Extraction** | Single teacher-forced forward pass producing per-token logits, softmax probability distribution, and loss tensor. |
| **4** | **Internal Signal Computation** | Calculates the 11 white-box generation signals (log-probabilities, predictive Shannon entropy, perplexity, and rank dispersion). |
| **5** | **Self-Consistency Sampling** | Generates $k=5$ stochastic responses ($T=0.7, \text{top\_}p=0.9, \text{max\_tokens}=128$) to probe behavioral stability. |
| **6** | **Embedding Model Verification** | Verifies `sentence-transformers/all-MiniLM-L6-v2` dense embedding model availability. |
| **7** | **Consistency Similarity Computation** | Encodes all candidate completions and calculates the full pairwise cosine similarity matrix and exact match score. |
| **8** | **NLI Model Verification** | Verifies `cross-encoder/nli-MiniLM2-L6-H768` cross-encoder availability and output label mappings. |
| **9** | **Pairwise Bidirectional NLI Inference** | Executes $k(k-1)=20$ directional passes measuring mean entailment, contradiction, and bidirectional disagreement. |
| **10** | **Universal Feature Vector Assembly** | Assembles the canonical 19-dimensional vector, strictly ensuring sequence length confounders are excluded. |
| **11** | **XGBoost Risk Prediction** | Generates the primary non-linear hallucination risk probability using the fitted gradient-boosted trees. |
| **12** | **Logistic Regression Prediction** | Generates the calibrated linear baseline risk probability using the standardized pipeline. |
| **13** | **Explainability & Attribution** | Calculates TreeSHAP log-odds feature contributions explaining which features flagged or cleared the response. |
| **14** | **Evidence Groundedness Analysis** | *(Optional)* Chunks reference context and computes dense retrieval cosine similarity, evidence margin, and claim agreement. |
| **15** | **Final UI Result Assembly** | Formats per-token probability and entropy tooltips, determines categorical risk tier, and tabulates millisecond stage latencies. |

---

## 🖥️ Live Streamlit Interactive Web Application

An interactive web application is included to demonstrate the detector in real time:

```bash
streamlit run app/app.py
```

### Key UI Capabilities:
1. **Dynamic Model Switcher**: Easily toggle between **Qwen 3.5 (0.8B)**, **Llama 3.2 (1B Instruct)**, and **Gemma 3 (1B IT)** with real-time parameter and benchmark status display.
2. **Interactive Token Confidence Heatmap**: Visualizes token-level uncertainty. Hover over any generated word to inspect its exact generation probability $P(y_t)$, Shannon entropy $H_t$, and token rank.
3. **Model-Native Explainability ("Why was it flagged?")**: TreeSHAP breakdown showing the exact top positive and negative feature contributions shifting the hallucination risk score.
4. **Dual Detection Modes**:
   - **Universal Core Mode (19 features)**: Context-free detection using generation dynamics, consistency, and NLI.
   - **Evidence Groundedness Mode**: Evaluates alignment against user-provided reference passages using dense semantic embeddings and evidence margins.
5. **Interactive Presets**: Pre-populated examples from HaluEval, TruthfulQA, and FEVER demonstrating both faithful and hallucinated generations.
6. **Detailed Millisecond Latency Profiler**: Granular latency table benchmarking all 15 stages.

---

## 📁 Repository Directory Structure

```
IML_PROJECT/
├── app/
│   ├── app.py                     # Streamlit demonstration application
│   └── inference.py               # Live 15-stage inference engine & caching
├── configs/                       # Configuration files and hyperparameters
├── data/
│   ├── processed/                 # Curated 4,000-instance balanced benchmark
│   └── raw/                       # Immutable raw source datasets
├── docs/                          # Comprehensive research reports and analysis
│   ├── ablation_study.md          # Phase 15 feature-family ablation report
│   ├── calibration_analysis.md    # Phase 17 probability calibration report
│   ├── cross_dataset_evaluation.md# Phase 16 Leave-One-Dataset-Out (LODO) report
│   ├── error_analysis.md          # Phase 18 comprehensive error analysis
│   ├── final_project_report.md    # Authoritative 58KB scientific report
│   ├── master_results_table.md    # Master verified experimental results
│   └── retrieval_augmented_experiment.md # Phase 19 retrieval variant
├── experiments/
│   ├── baselines/                 # Phase 6 baseline models & feature tables
│   ├── calibration/               # Phase 17 calibration partitions & artifacts
│   ├── models/                    # Model-specific feature tables & runs
│   │   ├── gemma_3_1b_it/         # Gemma 3 1B feature parquet & model results
│   │   ├── llama_3_2_1b_instruct/ # Llama 3.2 1B feature parquet & model results
│   │   └── qwen3_5_0_8b/          # Qwen 3.5 0.8B feature parquet & model results
│   └── retrieval_augmented/       # Phase 19 standalone retrieval parquet
├── models/                        # Serialized classifiers (.joblib)
│   ├── gemma_3_1b_it/             # Retrained XGBoost & LR pipelines for Gemma 3
│   ├── llama_3_2_1b_instruct/     # Retrained XGBoost & LR pipelines for Llama 3.2
│   └── qwen3_5_0_8b/              # Retrained XGBoost & LR pipelines for Qwen 3.5
├── src/
│   ├── consistency/               # Self-consistency & NLI agreement extraction
│   ├── features/                  # Canonical 19-feature vector assembly
│   ├── generation/                # Multi-model inference & prompt formatting
│   ├── retrieval/                 # Dense retrieval & evidence margin computation
│   ├── signals/                   # Token-level forward pass & logit extraction
│   ├── training/                  # Retraining pipelines & cross-validation
│   └── utils/                     # Model registry & GPU/CPU device resolution
├── tests/                         # 226 unit and integration tests (100% passing)
├── requirements.txt               # Locked project dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with 4GB+ VRAM recommended (verified on RTX 3050 Laptop GPU with CUDA 12.8; automatic CPU fallback supported)

### 2. Environment Installation

```powershell
# Clone the repository
git clone https://github.com/Prizznce/IML_PROJECT.git
cd IML_PROJECT

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Test Suite
Verify that all 226 unit and integration tests pass:

```powershell
python -m pytest tests
```

### 4. Retraining the Detector on Any Model
To retrain the complete 19-feature Universal Core pipeline on a causal LLM:

```powershell
# Retrain on Qwen 3.5 0.8B
python src/training/retrain_pipeline.py --model Qwen/Qwen3.5-0.8B

# Retrain on Llama 3.2 1B Instruct
python src/training/retrain_pipeline.py --model meta-llama/Llama-3.2-1B-Instruct

# Retrain on Google Gemma 3 1B IT
python src/training/retrain_pipeline.py --model google/gemma-3-1b-it
```

### 5. Launch the Live Demonstration
```powershell
streamlit run app/app.py
```

---

## 📑 Research Documentation Map

For in-depth mathematical formulations, ablation methodologies, and full diagnostic audits:
- [docs/final_project_report.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/final_project_report.md): The comprehensive 58KB research paper detailing theoretical foundations, split protocols, and experimental conclusions.
- [docs/master_results_table.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/master_results_table.md): Authoritative repository of all empirical benchmark tables and statistical comparisons across Phases 3–19.
- [docs/ablation_study.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/ablation_study.md): Detailed isolation and Shapley analysis of individual and combined feature families.
- [docs/calibration_analysis.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/calibration_analysis.md): Reliability diagrams, Platt scaling, and Isotonic regression calibration audit.
- [docs/error_analysis.md](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/error_analysis.md): Descriptive analysis of false positives, false negatives, confidence clustering, and model disagreements.

---

## 📜 License & Citation

This project is licensed under the MIT License. If you use this methodology or codebase in your academic research, please cite:

```bibtex
@misc{catching_llm_lying_2026,
  author = {Aditya et al.},
  title = {Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Prizznce/IML_PROJECT}}
}
```
