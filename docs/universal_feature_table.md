# Universal Combined Feature Table

**Project:** Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection  
**Phase:** Phase 13 — Build and Validate the Universal Combined Feature Table  
**Source Tables Joined:**
1. [`experiments/baselines/supervised_signals_combined.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/supervised_signals_combined.parquet) (Internal Signals)
2. [`experiments/baselines/self_consistency/full_self_consistency_aggregated.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/self_consistency/full_self_consistency_aggregated.parquet) (Self-Consistency)
3. [`experiments/baselines/nli_agreement/full_nli_aggregated.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/nli_agreement/full_nli_aggregated.parquet) (NLI Agreement)

**Primary Production Artifacts:**
- [`experiments/baselines/combined/universal_features.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/universal_features.parquet) (4,000 rows $\times$ 22 columns)
- [`experiments/baselines/combined/universal_features.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/universal_features.csv) (4,000 rows)
- [`experiments/baselines/combined/universal_features_summary.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/universal_features_summary.json)

---

## 1. Executive Summary & Architectural Overview

The Universal Combined Feature Table serves as the master supervised training foundation for the **Tier-1 Universal Core Hallucination Detector**. 

This feature table unites three orthogonal behavioral signal families extracted across all 4,000 benchmark examples:
1. **White-Box Internal Token Signals**: Confidence, log probabilities, Shannon entropy, perplexity, and token rank dynamics from `Qwen/Qwen3.5-0.8B`.
2. **Black-Box Self-Consistency Signals**: Stochastic generation consensus and bi-encoder embedding similarity across $K=5$ independent samples.
3. **Black-Box Natural Language Inference Signals**: Pairwise bidirectional premise–hypothesis logical entailment and contradiction from `cross-encoder/nli-MiniLM2-L6-H768`.

### Core Architectural Principles:
> [!IMPORTANT]
> **Universal Applicability & Data Hygiene:**
> 1. **Zero External Dependencies**: Every feature in the Universal Core is extractable for *any* arbitrary prompt and response without requiring external corpora, search engines, or database retrievals.
> 2. **Lossless ID Joining**: Joined strictly by unique benchmark example `id`.
> 3. **Strict Metadata Isolation**: Metadata columns (`id`, `source_dataset`, `label`) are preserved in the table for stratified evaluation, auditing, and error analysis, but are **strictly forbidden from entering feature matrix $X$**.
> 4. **No Length Shortcut**: `num_tokens` is excluded from the feature set following our empirical ablation findings (preventing shortcut learning on benchmark response length).
> 5. **Multicollinearity Elimination**: Exact mathematical identities and highly collinear redundant features documented during the Feature Integration Audit have been pruned.

---

## 2. The 19 Primary Feature Definitions

The Universal Core feature space consists of exactly **19 primary features** grouped into three canonical families:

```
Universal Core Feature Matrix X (19 Dimensions)
├── Group 1: Internal Generation Signals (11)
│   ├── min_log_prob
│   ├── mean_log_prob
│   ├── mean_token_prob
│   ├── token_prob_std
│   ├── mean_entropy
│   ├── max_entropy
│   ├── entropy_std
│   ├── log_perplexity       [log(max(perplexity, 1e-12))]
│   ├── log_mean_token_rank  [log1p(max(mean_token_rank, 0))]
│   ├── log_max_token_rank   [log1p(max(max_token_rank, 0))]
│   └── log_rank_std         [log1p(max(rank_std, 0))]
├── Group 2: Self-Consistency Signals (5)
│   ├── exact_match_agreement
│   ├── mean_pairwise_similarity
│   ├── min_pairwise_similarity
│   ├── max_pairwise_similarity
│   └── pairwise_similarity_std
└── Group 3: NLI Agreement Signals (3)
    ├── mean_pairwise_entailment
    ├── mean_pairwise_contradiction
    └── nli_disagreement      [mean_contra + 0.5 * mean_neutral]
```

### 2.1 Feature Group 1 — Internal Generation Signals (11)
Derived from the target response token sequence produced by `Qwen/Qwen3.5-0.8B`:
- `min_log_prob`: Lowest token log probability in the response sequence ($\le 0$).
- `mean_log_prob`: Mean token log probability across response tokens ($\le 0$).
- `mean_token_prob`: Mean token probability $p = \exp(\log p) \in [0.0, 1.0]$.
- `token_prob_std`: Standard deviation of response token probabilities ($\ge 0$).
- `mean_entropy`: Mean Shannon entropy of predictive distribution across response positions ($\ge 0$).
- `max_entropy`: Peak Shannon entropy across response positions ($\ge 0$).
- `entropy_std`: Standard deviation of token entropy values ($\ge 0$).
- `log_perplexity`: Variance-stabilized log perplexity: $\log(\max(\text{perplexity}, 10^{-12}))$.
- `log_mean_token_rank`: Log-transformed mean token rank: $\log(1 + \max(\text{mean\_token\_rank}, 0))$.
- `log_max_token_rank`: Log-transformed peak token rank: $\log(1 + \max(\text{max\_token\_rank}, 0))$.
- `log_rank_std`: Log-transformed rank standard deviation: $\log(1 + \max(\text{rank\_std}, 0))$.

### 2.2 Feature Group 2 — Self-Consistency Signals (5)
Computed across $K=5$ stochastic responses ($T=0.7, \text{top\_p}=0.9$) and embedded via `all-MiniLM-L6-v2`:
- `exact_match_agreement`: Fraction of the 10 pairs with identical normalized strings $\in [0.0, 1.0]$.
- `mean_pairwise_similarity`: Average bi-encoder cosine similarity across all 10 pairs $\in [-1.0, 1.0]$.
- `min_pairwise_similarity`: Minimum cosine similarity across all 10 pairs (worst-case sample divergence).
- `max_pairwise_similarity`: Maximum cosine similarity across all 10 pairs (closest pair agreement).
- `pairwise_similarity_std`: Standard deviation of pair cosine similarities ($\ge 0$).

### 2.3 Feature Group 3 — NLI Agreement Signals (3)
Computed across all 10 unordered candidate pairs evaluated bidirectionally (20 directional evaluations per example) via `cross-encoder/nli-MiniLM2-L6-H768`:
- `mean_pairwise_entailment`: Average symmetrized entailment probability $\in [0.0, 1.0]$.
- `mean_pairwise_contradiction`: Average symmetrized contradiction probability $\in [0.0, 1.0]$.
- `nli_disagreement`: Weighted divergence score $\bar{P}_{\text{contra}} + 0.5 \times \bar{P}_{\text{neutral}} \in [0.0, 1.0]$.

---

## 3. Deliberately Excluded Features & Multicollinearity Elimination

Following the formal Feature Integration Audit, the following features are strictly excluded from the primary Universal Core feature matrix:

| Excluded Column | Reason for Exclusion | Mathematical / Empirical Rationale |
| :--- | :--- | :--- |
| **`num_tokens`** | **Forbidden length shortcut** | Model learns dataset response length rather than hallucination behavior ($AUC$ artifact). |
| **`generation_disagreement`** | **Exact linear redundancy** | Mathematically $1.0 - \text{mean\_pairwise\_similarity}$ ($r = -1.000$). |
| **`unique_response_ratio`** | **Severe collinearity** | Collinear with `exact_match_agreement` ($r = -0.972$) and modal fraction ($r = -1.000$). |
| **`majority_response_fraction`**| **Exact collinearity** | Collinear with unique response ratio on $K=5$ ($r = -1.000$). |
| **`mean_pairwise_neutral`** | **Linear sum-to-one constraint** | $\bar{P}_{\text{entail}} + \bar{P}_{\text{contra}} + \bar{P}_{\text{neutral}} = 1.0$. Induces singular covariance matrix. |
| **`fraction_entailing_pairs`** | **Coarse discretization** | Strongly collinear with continuous `mean_pairwise_entailment` ($r = 0.915$). |
| **`fraction_contradicting_pairs`**| **Coarse discretization** | Strongly collinear with continuous `mean_pairwise_contradiction` ($r = 0.968$). |
| **`fraction_neutral_pairs`** | **Discrete sum-to-one constraint** | Discrete analogue of neutral probability; redundant with continuous features. |
| **`perplexity`, `*_token_rank`** | **Heavy-tailed non-linearity** | Replaced by variance-stabilizing log-transforms (`log_*`). |
| **Text payloads (`prompt`, etc.)** | **High-dimensional leakage** | White-box / behavioral classifier operates on scalar signal geometry. |

---

## 4. Empirical Feature Distributions & Summary Statistics ($N=4,000$)

Across the complete 4,000-example supervised benchmark dataset:

| Feature Name | Mean | Std | Min | 25% | Median | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `min_log_prob` | -6.6432 | 3.3934 | -26.7869 | -8.5492 | -5.9922 | -4.1332 | -0.0631 |
| `mean_log_prob` | -0.9996 | 0.8123 | -7.5303 | -1.3323 | -0.7397 | -0.4217 | -0.0152 |
| `mean_token_prob` | 0.5312 | 0.2452 | 0.0016 | 0.3475 | 0.5482 | 0.7225 | 0.9850 |
| `token_prob_std` | 0.3012 | 0.1066 | 0.0028 | 0.2335 | 0.3150 | 0.3789 | 0.4999 |
| `mean_entropy` | 1.1554 | 0.6974 | 0.0401 | 0.6186 | 1.0264 | 1.5721 | 4.3807 |
| `max_entropy` | 3.5516 | 1.3917 | 0.1259 | 2.5855 | 3.5670 | 4.5422 | 8.3541 |
| `entropy_std` | 0.9829 | 0.3545 | 0.0193 | 0.7423 | 0.9788 | 1.2185 | 2.4501 |
| `log_perplexity` | 0.9996 | 0.8123 | 0.0152 | 0.4217 | 0.7397 | 1.3323 | 7.5303 |
| `log_mean_token_rank` | 1.3732 | 1.0538 | 0.0000 | 0.5284 | 1.0772 | 1.9683 | 6.8407 |
| `log_max_token_rank` | 4.9080 | 2.6841 | 0.0000 | 2.7726 | 4.7958 | 6.9463 | 11.8745 |
| `log_rank_std` | 2.1287 | 1.4883 | 0.0000 | 0.9022 | 1.8601 | 3.1951 | 6.9744 |
| `exact_match_agreement` | 0.1899 | 0.3370 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 1.0000 |
| `mean_pairwise_similarity` | 0.7875 | 0.1597 | 0.1813 | 0.6990 | 0.8191 | 0.9006 | 1.0000 |
| `min_pairwise_similarity` | 0.6551 | 0.2551 | -0.1129 | 0.5089 | 0.7114 | 0.8420 | 1.0000 |
| `max_pairwise_similarity` | 0.9312 | 0.0819 | 0.3606 | 0.8956 | 0.9512 | 1.0000 | 1.0000 |
| `pairwise_similarity_std` | 0.0994 | 0.0952 | 0.0000 | 0.0335 | 0.0622 | 0.1444 | 0.4926 |
| `mean_pairwise_entailment` | 0.3381 | 0.3267 | 0.0006 | 0.0686 | 0.2104 | 0.5667 | 0.9929 |
| `mean_pairwise_contradiction` | 0.2298 | 0.2374 | 0.0005 | 0.0188 | 0.1525 | 0.3872 | 0.9958 |
| `nli_disagreement` | 0.4459 | 0.2456 | 0.0039 | 0.2938 | 0.4877 | 0.6137 | 0.9972 |

---

## 5. Mathematical Redundancy & Correlation Analysis

### 5.1 Identified Mathematical Redundancies
1. **`mean_log_prob` vs. `log_perplexity`**:
   $$\text{Perplexity} = \exp(-\text{mean\_log\_prob}) \implies \log(\text{Perplexity}) = -\text{mean\_log\_prob}$$
   - **Observed Pearson Correlation**: $r = -1.000000$ (exact identity).
   - *Status*: Retained in the 19-feature table to preserve strict compatibility with the primary baseline model (`PRIMARY_SIGNAL_FEATURES`), but identified for tree-based models or L1 regularization during classifier training.

### 5.2 High Collinearity Clusters ($|r| > 0.85$)
- **Rank Dynamics**:
  - `log_max_token_rank` $\leftrightarrow$ `log_rank_std`: $r = +0.9855$
  - `log_mean_token_rank` $\leftrightarrow$ `log_max_token_rank`: $r = +0.9688$
  - `log_mean_token_rank` $\leftrightarrow$ `log_rank_std`: $r = +0.9606$
  - `min_log_prob` $\leftrightarrow$ `log_max_token_rank`: $r = -0.9352$
- **Semantic Spread**:
  - `mean_pairwise_similarity` $\leftrightarrow$ `min_pairwise_similarity`: $r = +0.9538$
  - `min_pairwise_similarity` $\leftrightarrow$ `pairwise_similarity_std`: $r = -0.8876$
- **NLI Consensus**:
  - `mean_pairwise_entailment` $\leftrightarrow$ `nli_disagreement`: $r = -0.9087$
  - `mean_pairwise_contradiction` $\leftrightarrow$ `nli_disagreement`: $r = +0.8185$
- **Cross-Family Synergy**:
  - `exact_match_agreement` $\leftrightarrow$ `mean_pairwise_entailment`: $r = +0.8857$
    *(Generations with identical wording strongly entail one another, confirming cross-family consistency)*.

---

## 6. Per-Dataset Signal Profiles

| Metric / Signal | HaluEval ($N=1,500$) | TruthfulQA ($N=1,500$) | FEVER ($N=1,000$) | Full Benchmark ($N=4,000$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Token Probability** | 0.6341 | 0.4485 | 0.5011 | 0.5312 |
| **Mean Log Probability** | -0.6750 | -1.2407 | -1.1249 | -0.9996 |
| **Mean Shannon Entropy** | 0.8870 | 1.3785 | 1.2234 | 1.1554 |
| **Exact Match Agreement** | **0.4745** | 0.0313 | 0.0008 | 0.1899 |
| **Mean Semantic Similarity** | 0.8126 | 0.7351 | **0.8283** | 0.7875 |
| **Mean Entailment Probability** | **0.6564** | 0.1357 | 0.1640 | 0.3381 |
| **Mean Contradiction Probability**| 0.1540 | 0.2662 | **0.2891** | 0.2298 |
| **NLI Disagreement** | **0.2488** | **0.5652** | **0.5625** | 0.4459 |

---

## 7. Data Invariant Validations

All 17 production validation requirements were executed and passed:
1. **Row Count**: Exactly 4,000 rows.
2. **Unique IDs**: Exactly 4,000 unique IDs matching source benchmarks.
3. **Dataset & Label Preservation**: Zero label mutations; `source_dataset` and `label` match 1-to-1 with original supervised metadata.
4. **Column Count**: Exactly 22 columns ($3 \text{ metadata} + 19 \text{ features}$).
5. **No Forbidden Columns**: Verified zero presence of `num_tokens`, text payloads, or redundant ablation features in feature matrix.
6. **Finiteness**: Zero `NaN`, zero `Inf` across all 19 feature columns (0.0% missingness).
7. **Near-Zero Variance**: Zero features exhibit near-zero variance ($\sigma > 10^{-4}$ for all 19 features).
8. **Mathematical Identity**: Recorded $r = -1.000$ between `mean_log_prob` and `log_perplexity`.
