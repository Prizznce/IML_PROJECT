# Production Full NLI Agreement Feature Generation

**Project:** Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection  
**Phase:** Phase 12 — Full NLI Agreement Feature Generation across 4,000 Supervised Benchmark Examples  
**Input Dataset:** [`experiments/baselines/self_consistency/full_generations_raw.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/self_consistency/full_generations_raw.parquet) (20,000 candidate responses)  
**Output Directory:** [`experiments/baselines/nli_agreement/`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/nli_agreement/)  
**Primary Artifacts:**
- [`experiments/baselines/nli_agreement/full_nli_aggregated.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/nli_agreement/full_nli_aggregated.parquet) (4,000 rows)
- [`experiments/baselines/nli_agreement/full_nli_aggregated.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/nli_agreement/full_nli_aggregated.csv) (4,000 rows)
- [`experiments/baselines/nli_agreement/full_nli_pairwise.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/nli_agreement/full_nli_pairwise.parquet) (40,000 rows)
- [`experiments/baselines/nli_agreement/full_nli_summary.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/nli_agreement/full_nli_summary.json)

---

## 1. Executive Summary & Production Objectives

This phase implements production-scale Natural Language Inference (NLI) agreement feature extraction across the **entire 4,000-example supervised benchmark dataset**.

Natural Language Inference provides a rigorous semantic logic signal that complements lexical exact-match and bi-encoder embedding similarity. While embedding cosine similarity measures topic and surface alignment, NLI models capture directional logical consequence: whether one candidate response actively entails, contradicts, or remains neutral with respect to another.

### Strict Methodological Rules & Data Isolation
> [!IMPORTANT]
> **Data Integrity & Label Isolation:**
> 1. **Zero Regeneration**: The pipeline strictly reuses the 20,000 raw candidate responses previously generated and validated during Phase 11 (`full_generations_raw.parquet`).
> 2. **Unlabeled Model Behavior**: Candidate responses are strictly unlabelled behavioral samples. Original ground-truth labels are never assigned to any newly generated response.
> 3. **Metadata Preservation**: Original benchmark example `id`, `source_dataset`, and `label` are strictly preserved as metadata to guarantee lossless joins with the supervised feature table.
> 4. **Zero Mutation**: The input files and original supervised datasets remain completely read-only.
> 5. **No Downstream Leaks**: No evidence retrieval or classifier training is executed during this feature generation phase.

---

## 2. Experimental & Production Configuration

| Parameter | Production Value | Rationale |
| :--- | :---: | :--- |
| **Total Supervised Examples** | **4,000** | Full benchmark (1,500 HaluEval, 1,500 TruthfulQA, 1,000 FEVER) |
| **Candidate Responses per Example ($K$)** | **5** | From Phase 11 self-consistency raw generation outputs |
| **Unordered Pairs per Example** | **10** | $\binom{5}{2} = 10$ distinct response pairs |
| **Total Unordered Pairs** | **40,000** | $4,000 \times 10$ response pairs evaluated |
| **Directional Evaluations** | **80,000** | $40,000 \times 2$ ($A \to B$ and $B \to A$ directional passes) |
| **NLI Cross-Encoder Model** | `cross-encoder/nli-MiniLM2-L6-H768` | Pretrained cross-attention transformer ($80\text{M}$ params) |
| **Execution Device** | `cuda:0` | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| **Inference Batch Size** | **32** | Optimal balance of GPU tensor utilization and low VRAM footprint |
| **Checkpoint Interval** | **100 examples** | Saves atomic intermediate tables every 100 examples |
| **Fault Tolerance / Resume** | **Supported (`--resume`)** | Lossless resumption from existing checkpoints |

---

## 3. Mathematical Formulation & Feature Suite

For each benchmark example, let $\{r_1, r_2, \dots, r_5\}$ be the set of $K=5$ stochastic candidate responses. For every unordered pair $(i, j)$ with $1 \le i < j \le 5$, the cross-encoder computes directional logits:
$$\mathbf{z}_{i \to j} = \text{CrossEncoder}(r_i, r_j) \in \mathbb{R}^3, \quad \mathbf{z}_{j \to i} = \text{CrossEncoder}(r_j, r_i) \in \mathbb{R}^3$$

Numerically stable softmax yields directional probabilities for contradiction ($c$), entailment ($e$), and neutral ($n$):
$$\mathbf{p}_{i \to j} = \text{Softmax}(\mathbf{z}_{i \to j}), \quad \mathbf{p}_{j \to i} = \text{Softmax}(\mathbf{z}_{j \to i})$$

### 3.1 Bidirectional Pair Aggregation
The directional probabilities are averaged to produce symmetric pair-level probabilities:
$$P_{ij}^{\text{entail}} = \frac{1}{2} \left( p_{i \to j}^{\text{entail}} + p_{j \to i}^{\text{entail}} \right)$$
$$P_{ij}^{\text{contra}} = \frac{1}{2} \left( p_{i \to j}^{\text{contra}} + p_{j \to i}^{\text{contra}} \right)$$
$$P_{ij}^{\text{neutral}} = \frac{1}{2} \left( p_{i \to j}^{\text{neutral}} + p_{j \to i}^{\text{neutral}} \right)$$

The consensus pair prediction is the argmax class:
$$\hat{y}_{ij} = \arg\max_{k \in \{\text{entail}, \text{contra}, \text{neutral}\}} P_{ij}^{k}$$

### 3.2 Example-Level Extracted Features

#### Primary Features:
1. **`mean_pairwise_entailment`**:
   $$\bar{P}_{\text{entail}} = \frac{1}{10} \sum_{i < j} P_{ij}^{\text{entail}} \in [0.0, 1.0]$$
2. **`mean_pairwise_contradiction`**:
   $$\bar{P}_{\text{contra}} = \frac{1}{10} \sum_{i < j} P_{ij}^{\text{contra}} \in [0.0, 1.0]$$
3. **`nli_disagreement`**:
   $$\text{nli\_disagreement} = \bar{P}_{\text{contra}} + 0.5 \times \bar{P}_{\text{neutral}} \in [0.0, 1.0]$$
   - Direct factual contradictions receive full weight ($1.0$).
   - Neutrality (failure to entail / semantic divergence) receives half weight ($0.5$).
   - Mutual entailment incurs zero disagreement penalty ($0.0$).

#### Optional / Ablation Features:
4. **`mean_pairwise_neutral`**:
   $$\bar{P}_{\text{neutral}} = \frac{1}{10} \sum_{i < j} P_{ij}^{\text{neutral}} \in [0.0, 1.0]$$
   > [!NOTE]
   > Because $\bar{P}_{\text{entail}} + \bar{P}_{\text{contra}} + \bar{P}_{\text{neutral}} = 1.0$ identically, neutral is excluded from the primary feature set to eliminate exact collinearity.
5. **`fraction_entailing_pairs`**:
   $$\text{frac}_{\text{entail}} = \frac{1}{10} \sum_{i < j} \mathbb{I}(\hat{y}_{ij} == \text{entailment}) \in [0.0, 1.0]$$
6. **`fraction_contradicting_pairs`**:
   $$\text{frac}_{\text{contra}} = \frac{1}{10} \sum_{i < j} \mathbb{I}(\hat{y}_{ij} == \text{contradiction}) \in [0.0, 1.0]$$
7. **`fraction_neutral_pairs`**:
   $$\text{frac}_{\text{neutral}} = \frac{1}{10} \sum_{i < j} \mathbb{I}(\hat{y}_{ij} == \text{neutral}) \in [0.0, 1.0]$$

---

## 4. Production Engineering & Integrity Architecture

### 4.1 Checkpoint & Resume Architecture
- **Checkpoints**: Written atomically every 100 examples to:
  - `experiments/baselines/nli_agreement/.checkpoint_full_nli_aggregated.parquet`
  - `experiments/baselines/nli_agreement/.checkpoint_full_nli_pairwise.parquet`
- When launched with `--resume`, the pipeline inspects already completed IDs and skips them without duplicate evaluations or data loss.
- Periodic `torch.cuda.empty_cache()` prevents GPU memory fragmentation.

### 4.2 Automated Invariant Validations
Before writing final production files, the pipeline enforces strict automated assertions:
1. **Row Count Integrity**: Exactly 4,000 aggregated rows and 40,000 pairwise rows ($4,000 \times 10$).
2. **Uniqueness**: `df_aggregated["id"].nunique() == 4000` (zero duplicate IDs).
3. **Finiteness**: Zero `NaN` or `Inf` values across all 7 numerical feature columns.
4. **Range Bounds**: All probabilities, fractions, and disagreement scores reside strictly in $[0.0, 1.0]$.
5. **Exact Probability Sum Identity**: $|\bar{P}_{\text{entail}} + \bar{P}_{\text{contra}} + \bar{P}_{\text{neutral}} - 1.0| < 10^{-5}$ across all 4,000 examples.
6. **Exact Disagreement Formula Identity**: $|\text{nli\_disagreement} - (\bar{P}_{\text{contra}} + 0.5 \times \bar{P}_{\text{neutral}})| < 10^{-5}$ across all 4,000 examples.
7. **Input Integrity**: Pre-flight verification confirms that all 4,000 examples have exactly 5 candidate generations.

---

## 5. Execution Diagnostics & Summary Metrics

The production NLI agreement extraction pipeline executed end-to-end on `cuda:0` without errors or retries:

| Diagnostic Metric | Value |
| :--- | :---: |
| **Total Benchmark Examples** | **4,000** |
| **Candidate Responses Evaluated** | **20,000** |
| **Unordered Response Pairs** | **40,000** |
| **Directional NLI Inferences** | **80,000** |
| **Failed Examples / Retries** | **0 (100.0% success rate)** |
| **Execution Device** | `cuda:0` (NVIDIA GeForce RTX 3050 6GB Laptop GPU) |
| **NLI Cross-Encoder** | `cross-encoder/nli-MiniLM2-L6-H768` |
| **Model Loading Time** | **5.55 seconds** |
| **Total Inference Time** | **293.73 seconds (~4.90 minutes)** |
| **Throughput (Examples/s)** | **13.62 examples/second** |
| **Throughput (Directional Inferences/s)** | **272.36 directional evaluations/second** |
| **Average Time per Example** | **0.0734 seconds (73.4 ms)** |
| **Average Time per Pair** | **0.0073 seconds (7.3 ms)** |
| **Average Time per Directional Eval** | **0.0037 seconds (3.7 ms)** |
| **Peak GPU VRAM Allocated** | **523.89 MB** (<9% of 6GB physical capacity) |

---

## 6. Feature Distributions & Benchmark Dataset Analysis

### 6.1 Aggregate Benchmark Statistics ($N=4,000$)

| Feature Name | Mean | Std | Min | 25% | Median | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mean_pairwise_entailment` | 0.3381 | 0.3267 | 0.0006 | 0.0686 | 0.2104 | 0.5667 | 0.9929 |
| `mean_pairwise_contradiction` | 0.2298 | 0.2374 | 0.0005 | 0.0188 | 0.1525 | 0.3872 | 0.9958 |
| `mean_pairwise_neutral` | 0.4321 | 0.2914 | 0.0025 | 0.1843 | 0.4373 | 0.6745 | 0.9910 |
| `fraction_entailing_pairs` | 0.2971 | 0.3662 | 0.0000 | 0.0000 | 0.1000 | 0.6000 | 1.0000 |
| `fraction_contradicting_pairs` | 0.2121 | 0.2756 | 0.0000 | 0.0000 | 0.1000 | 0.4000 | 1.0000 |
| `fraction_neutral_pairs` | 0.4908 | 0.3709 | 0.0000 | 0.0000 | 0.5000 | 0.8000 | 1.0000 |
| `nli_disagreement` | 0.4459 | 0.2456 | 0.0039 | 0.2938 | 0.4877 | 0.6137 | 0.9972 |

### 6.2 Per-Dataset Breakdown

| Dataset | Count | Mean Entailment | Mean Contradiction | Mean Neutral | NLI Disagreement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HaluEval** | 1,500 | **0.6564** | 0.1540 | 0.1896 | **0.2488** |
| **TruthfulQA** | 1,500 | 0.1357 | 0.2662 | **0.5981** | **0.5652** |
| **FEVER** | 1,000 | 0.1640 | **0.2891** | 0.5469 | **0.5625** |

#### Key Empirical Insights:
1. **HaluEval Entailment Dominance**: For closed-context question answering, candidate responses frequently share core factual entities, yielding high mutual entailment ($0.656$) and low disagreement ($0.249$).
2. **TruthfulQA Neutral Divergence**: Open-domain responses frequently formulate differing aspects of a question without directly refuting each other, resulting in very high neutrality ($0.598$) and substantial disagreement ($0.565$).
3. **FEVER Contradiction**: Binary claim verification prompts trigger opposing truth polarities across stochastic generations, producing the highest mean pairwise contradiction rate ($0.289$) across all three benchmark domains.

---

## 7. Artifact Schema Reference

### 7.1 Pairwise Evaluations Table (`full_nli_pairwise.parquet`)
- `id` (str): Unique benchmark example identifier.
- `source_dataset` (str): Benchmark provenance (`halueval`, `fever`, `truthfulqa`).
- `original_label` (int64): Benchmark ground-truth label (metadata only).
- `pair_index_i` (int): First response index ($0 \le i < 5$).
- `pair_index_j` (int): Second response index ($0 \le j < 5, i < j$).
- `response_i` (str): Candidate response text $r_i$.
- `response_j` (str): Candidate response text $r_j$.
- `prob_entail_i_to_j` (float64): Directional probability $p(r_i \to r_j = \text{entailment})$.
- `prob_contra_i_to_j` (float64): Directional probability $p(r_i \to r_j = \text{contradiction})$.
- `prob_neutral_i_to_j` (float64): Directional probability $p(r_i \to r_j = \text{neutral})$.
- `prob_entail_j_to_i` (float64): Directional probability $p(r_j \to r_i = \text{entailment})$.
- `prob_contra_j_to_i` (float64): Directional probability $p(r_j \to r_i = \text{contradiction})$.
- `prob_neutral_j_to_i` (float64): Directional probability $p(r_j \to r_i = \text{neutral})$.
- `pair_entailment` (float64): Symmetrized pair entailment probability.
- `pair_contradiction` (float64): Symmetrized pair contradiction probability.
- `pair_neutral` (float64): Symmetrized pair neutral probability.
- `predicted_class` (str): Argmax consensus class (`entailment`, `contradiction`, or `neutral`).

### 7.2 Aggregated Features Table (`full_nli_aggregated.parquet` / `.csv`)
- `id` (str): Unique benchmark example identifier.
- `source_dataset` (str): Benchmark provenance (`halueval`, `fever`, `truthfulqa`).
- `original_label` (int64): Benchmark ground-truth label (metadata only).
- `num_generations` (int): Constant 5.
- `mean_pairwise_entailment` (float64): Average pair entailment probability in $[0.0, 1.0]$.
- `mean_pairwise_contradiction` (float64): Average pair contradiction probability in $[0.0, 1.0]$.
- `mean_pairwise_neutral` (float64): Average pair neutral probability in $[0.0, 1.0]$.
- `fraction_entailing_pairs` (float64): Fraction of pairs whose consensus prediction is entailment in $[0.0, 1.0]$.
- `fraction_contradicting_pairs` (float64): Fraction of pairs whose consensus prediction is contradiction in $[0.0, 1.0]$.
- `fraction_neutral_pairs` (float64): Fraction of pairs whose consensus prediction is neutral in $[0.0, 1.0]$.
- `nli_disagreement` (float64): $\bar{P}_{\text{contra}} + 0.5 \times \bar{P}_{\text{neutral}}$ in $[0.0, 1.0]$.
