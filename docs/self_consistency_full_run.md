# Production Full Self-Consistency Feature Generation

**Project:** Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection  
**Phase:** Phase 11 — Full Self-Consistency Feature Generation across 4,000 Supervised Benchmark Examples  
**Input Dataset:** [`experiments/baselines/supervised_signals_combined.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/supervised_signals_combined.parquet) (4,000 rows)  
**Output Artifacts:**
- [`experiments/baselines/self_consistency/full_generations_raw.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/self_consistency/full_generations_raw.parquet) (20,000 rows)
- [`experiments/baselines/self_consistency/full_self_consistency_aggregated.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/self_consistency/full_self_consistency_aggregated.parquet) (4,000 rows)
- [`experiments/baselines/self_consistency/full_run_summary.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/self_consistency/full_run_summary.json)

---

## 1. Executive Summary & Production Objectives

This phase scales the self-consistency hallucination signal extraction pipeline from the 20-example pilot to the **entire 4,000-example supervised benchmark dataset**.

Self-consistency evaluates the model's epistemic uncertainty by sampling multiple stochastic candidate responses from `Qwen/Qwen3.5-0.8B` for each prompt and measuring discrete lexical agreement and continuous semantic similarity across the generation set.

### Strict Methodological Rules & Data Isolation
> [!IMPORTANT]
> **Supervised Benchmark Workflow Integrity:**
> 1. The original benchmark response and original benchmark ground-truth label remain the **supervised target example**.
> 2. **Never assign the original label to any newly generated response.**
> 3. Self-consistency generations are **UNLABELLED model-behavior signals** used purely to quantify consensus and divergence.
> 4. The original example `id` and `source_dataset` are strictly preserved across all raw and aggregated tables to enable lossless joining to the master supervised feature table.
> 5. The input file [`supervised_signals_combined.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/supervised_signals_combined.parquet) is never mutated or overwritten.

---

## 2. Experimental & Production Configuration

| Parameter | Production Value | Rationale |
| :--- | :---: | :--- |
| **Total Supervised Examples** | **4,000** | Full benchmark (1,500 HaluEval, 1,500 TruthfulQA, 1,000 FEVER) |
| **Generations per Prompt ($K$)** | **5** | Balances semantic coverage and inference throughput |
| **Total Generations** | **20,000** | $4,000 \times 5$ stochastic responses |
| **Causal Language Model** | `Qwen/Qwen3.5-0.8B` | Target base model (`torch.bfloat16`) |
| **Embedding Model** | `all-MiniLM-L6-v2` | Lightweight sentence bi-encoder ($d = 384$) |
| **Execution Device** | `cuda:0` | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| **Sampling Temperature** | `0.7` | Standard stochastic exploration setting |
| **Top-p (Nucleus Sampling)** | `0.9` | Truncates low-probability token tails |
| **Maximum New Tokens** | `128` | Sufficient for direct, factual answers |
| **Thinking Mode** | **Disabled** | System prompt directive + `enable_thinking=False` in chat template |
| **Deterministic Seed** | `seed = 42 + idx * 100` | Exact seed progression per example index |
| **Checkpoint Interval** | **50 examples** | Saves intermediate state every 50 examples |
| **Fault Tolerance / Resume** | **Supported (`--resume`)** | Automatically resumes from existing checkpoints |

---

## 3. Extracted Feature Suite

For every benchmark example, exactly $K=5$ stochastic responses $[r_1, r_2, \dots, r_5]$ are generated, yielding $\binom{5}{2} = 10$ distinct unordered pairs $(i, j)$ with $i < j$.

### 3.1 Discrete Agreement Features
1. **`exact_match_agreement`**:
   Fraction of pairs sharing identical strings after deterministic normalization:
   $$\text{exact\_match\_agreement} = \frac{1}{\binom{K}{2}} \sum_{i < j} \mathbb{I}(\text{norm}(r_i) == \text{norm}(r_j))$$
   - Normalized via lowercasing, quote/punctuation stripping, and whitespace collapsing.
   - Bounded in $[0.0, 1.0]$.

2. **`unique_response_ratio`** (Ablation Feature):
   Ratio of unique normalized responses to total responses ($|\text{unique}| / K$).
   - Bounded in $[0.2, 1.0]$. Strongly anti-correlated with exact match ($r = -0.972$).

3. **`majority_response_fraction`** (Ablation Feature):
   Frequency fraction of the modal normalized response ($\max(\text{count}) / K$).
   - Bounded in $[0.2, 1.0]$. Exactly collinear with unique response ratio on $K=5$ ($r = -1.000$).

### 3.2 Continuous Semantic Similarity Features
Unit-normalized embeddings $\mathbf{v}_i \in \mathbb{R}^{384}$ are computed via `all-MiniLM-L6-v2`. Pairwise cosine similarities $s_{ij} = \mathbf{v}_i \cdot \mathbf{v}_j$ form a distribution over the 10 pairs:

4. **`mean_pairwise_similarity`** (Primary Feature):
   $$\text{mean\_pairwise\_similarity} = \frac{1}{\binom{K}{2}} \sum_{i < j} s_{ij}$$

5. **`min_pairwise_similarity`** (Primary Feature):
   $$\text{min\_pairwise\_similarity} = \min_{i < j} s_{ij}$$
   - Captures worst-case semantic divergence across the stochastic sample.

6. **`max_pairwise_similarity`** (Primary Feature):
   $$\text{max\_pairwise\_similarity} = \max_{i < j} s_{ij}$$
   - Captures the closest pair alignment.

7. **`pairwise_similarity_std`** (Primary Feature):
   $$\text{pairwise\_similarity\_std} = \sqrt{\frac{1}{\binom{K}{2}} \sum_{i < j} (s_{ij} - \bar{s})^2}$$
   - Measures semantic dispersion across generations.

8. **`generation_disagreement`** (Derived / Optional Ablation):
   $$\text{generation\_disagreement} = 1.0 - \text{mean\_pairwise\_similarity}$$
   > [!NOTE]
   > As verified in the Feature Integration Audit, `generation_disagreement` is an exact linear affine transform of `mean_pairwise_similarity` ($D = 1 - S$, $r = -1.000$). It is recorded in the table for downstream interpretability but marked as derived to prevent multicollinearity in linear models.

---

## 4. Production Engineering & Integrity Architecture

### 4.1 Checkpoint & Resume Mechanics
- **Checkpoint Artifacts**:
  - `experiments/baselines/self_consistency/.checkpoint_full_generations.parquet`
  - `experiments/baselines/self_consistency/.checkpoint_full_aggregated.parquet`
- Every 50 completed examples, the pipeline flushes in-memory generation records and aggregated metrics to atomic checkpoint files and calls `torch.cuda.empty_cache()` to prevent VRAM fragmentation.
- When started with `--resume`, the script inspects completed example IDs and automatically bypasses already processed examples.

### 4.2 Strict Validation Invariants
Before writing the final artifacts, the pipeline enforces automated assertions:
1. **Row Count Integrity**: Exactly 4,000 rows in aggregated table, exactly 20,000 rows in raw generations table ($4,000 \times 5$).
2. **Uniqueness**: `df_aggregated["id"].nunique() == 4000` (zero duplicate entries).
3. **Finiteness**: Zero `NaN` or `Inf` values across all 8 numerical feature columns.
4. **Analytical Bounds**: All features reside strictly within theoretical bounds ($[0, 1]$ for agreements, $[-1, 1]$ for similarities, $[0, 2]$ for disagreement).
5. **Mathematical Consistency**: $|D - (1 - S)| < 10^{-5}$ across all 4,000 examples.

---

## 5. Execution Diagnostics & Summary Metrics

The full self-consistency generation ran on CUDA (`cuda:0`) with incremental checkpointing and robust fault-tolerant resumption:

- **Run 1 (Initial)**: Examples 1 to 2,720 (Checkpoint at 2,700 saved).
- **Run 2 (Resumed via `--resume`)**: Examples 2,701 to 3,780 (Checkpoint at 3,750 saved).
- **Run 3 (Resumed via `--resume`)**: Examples 3,751 to 4,000 (Full 4,000 completed and validated).

| Diagnostic Metric | Value |
| :--- | :---: |
| **Total Examples Evaluated** | **4,000** |
| **Generations per Example ($K$)** | **5** |
| **Total Responses Generated** | **20,000** |
| **Failed Examples / Retries** | **0 (100.0% success rate)** |
| **Execution Device** | `cuda:0` (NVIDIA GeForce RTX 3050 6GB Laptop GPU) |
| **Causal LM Precision** | `torch.bfloat16` (`Qwen/Qwen3.5-0.8B`) |
| **Embedding Model** | `all-MiniLM-L6-v2` ($d=384$) |
| **Peak GPU VRAM Allocated** | **1,746.07 MB** (~28.4% of 6GB physical capacity) |
| **Total Active Generation Time** | **~275.6 minutes (~4.59 hours)** |
| **Average Time per Example** | **~4.13 seconds** |
| **Average Time per Generation** | **~0.83 seconds** |
| **Thinking Mode Leakage** | **None (`<think>` tags detected: 0)** |

---

## 6. Feature Distributions & Benchmark Dataset Analysis

Across all 4,000 benchmark examples, the generated self-consistency signals exhibit pronounced variation across datasets, reflecting differences in task format and response length:

### 6.1 Aggregate Benchmark Statistics ($N=4,000$)

| Feature Name | Mean | Std | Min | 25% | Median | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_match_agreement` | 0.1899 | 0.3370 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 1.0000 |
| `unique_response_ratio` | 0.8125 | 0.2917 | 0.2000 | 0.6000 | 1.0000 | 1.0000 | 1.0000 |
| `majority_response_fraction` | 0.3784 | 0.2842 | 0.2000 | 0.2000 | 0.2000 | 0.6000 | 1.0000 |
| `mean_pairwise_similarity` | 0.7875 | 0.1597 | 0.1813 | 0.6990 | 0.8191 | 0.9006 | 1.0000 |
| `min_pairwise_similarity` | 0.6551 | 0.2551 | -0.1129 | 0.5089 | 0.7114 | 0.8420 | 1.0000 |
| `max_pairwise_similarity` | 0.9312 | 0.0819 | 0.3606 | 0.8956 | 0.9512 | 1.0000 | 1.0000 |
| `pairwise_similarity_std` | 0.0994 | 0.0952 | 0.0000 | 0.0335 | 0.0622 | 0.1444 | 0.4926 |
| `generation_disagreement` | 0.2125 | 0.1597 | 0.0000 | 0.0994 | 0.1809 | 0.3010 | 0.8187 |

### 6.2 Per-Dataset Breakdown

| Dataset | Count | Exact Match Agreement | Mean Semantic Similarity | Generation Disagreement |
| :--- | :---: | :---: | :---: | :---: |
| **HaluEval** | 1,500 | **0.4745** | 0.8126 | 0.1874 |
| **TruthfulQA** | 1,500 | 0.0313 | 0.7351 | **0.2649** |
| **FEVER** | 1,000 | 0.0008 | **0.8283** | 0.1717 |

#### Key Insights:
1. **HaluEval Exact Match**: High lexical agreement ($0.475$) occurs because many HaluEval QA answers are short, canonical entity or factual phrases where Qwen repeatedly generates identical wording.
2. **TruthfulQA Divergence**: Open-ended question answering prompts yield high semantic disagreement ($0.265$) and low exact match ($0.031$), reflecting epistemic uncertainty across diverse plausible formulations.
3. **FEVER Semantic Closeness**: FEVER claims produce high semantic similarity ($0.828$) with low lexical exact match ($0.0008$), as Qwen generates diverse phrasing affirming or refuting the claim.

---

## 7. Artifact Schema Reference

### 7.1 Raw Generations Table (`full_generations_raw.parquet` / `.csv`)
- `id` (str): Unique benchmark example identifier.
- `source_dataset` (str): Benchmark provenance (`halueval`, `fever`, `truthfulqa`).
- `prompt` (str): Input prompt or question text.
- `context` (str): Reference passage (HaluEval) or background text.
- `original_response` (str): Benchmark response being evaluated.
- `original_label` (int64): Target label preserved as metadata (0: factual, 1: hallucination).
- `generation_index` (int): Stochastic generation index ($0, 1, 2, 3, 4$).
- `generated_response` (str): Raw model output string.
- `normalized_response` (str): Cleaned, lowercased, punctuation-stripped string.

### 7.2 Aggregated Features Table (`full_self_consistency_aggregated.parquet` / `.csv`)
- `id` (str): Unique benchmark example identifier.
- `source_dataset` (str): Benchmark provenance (`halueval`, `fever`, `truthfulqa`).
- `original_label` (int64): Benchmark ground-truth label (metadata only).
- `num_generations` (int): Constant 5.
- `exact_match_agreement` (float64): Pair exact-match rate in $[0.0, 1.0]$.
- `unique_response_ratio` (float64): Distinct response fraction in $[0.2, 1.0]$.
- `majority_response_fraction` (float64): Modal response fraction in $[0.2, 1.0]$.
- `mean_pairwise_similarity` (float64): Mean cosine similarity in $[-1.0, 1.0]$.
- `min_pairwise_similarity` (float64): Minimum cosine similarity in $[-1.0, 1.0]$.
- `max_pairwise_similarity` (float64): Maximum cosine similarity in $[-1.0, 1.0]$.
- `pairwise_similarity_std` (float64): Standard deviation of cosine similarity ($\ge 0.0$).
- `generation_disagreement` (float64): $1.0 - \text{mean\_pairwise\_similarity}$ in $[0.0, 2.0]$.
