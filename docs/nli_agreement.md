# Natural Language Inference (NLI) Agreement Signals: Methodology, Formulation, and Pilot Evaluation

## 1. Executive Summary & Phase Context

This document details the design, mathematical formulation, implementation, and empirical validation of **NLI Agreement signals**—the second additional hallucination signal group following internal signals and self-consistency.

While semantic similarity (via bi-encoder sentence embeddings) measures topical closeness, it cannot differentiate between synonymous statements and mutually exclusive factual claims (e.g., *"The champions were the New England Patriots"* vs *"The champions were the Green Bay Packers"* share high embedding similarity due to shared sporting vocabulary, yet directly contradict each other). Natural Language Inference (NLI) explicitly models directed logical relationships—**entailment**, **contradiction**, and **neutrality**—providing an essential complementary signal for hallucination detection.

---

## 2. Directionality & Pairwise Formulation

Natural Language Inference is inherently **directional**: a premise text $P$ may logically imply a hypothesis $H$ ($P \to H$), while the converse ($H \to P$) may be neutral or under-specified.

For an unordered pair of generated responses $(r_i, r_j)$ ($i < j$), we evaluate both directional hypotheses:
1. **Forward Direction ($i \to j$)**: Premise = $r_i$, Hypothesis = $r_j$
2. **Reverse Direction ($j \to i$)**: Premise = $r_j$, Hypothesis = $r_i$

The underlying cross-encoder model outputs raw logits across the three MNLI classes: Contradiction, Entailment, and Neutral. Logits are mapped to normalized probabilities via softmax:
$$\mathbf{p}_{i \to j} = \operatorname{softmax}(\operatorname{logits}(r_i, r_j))$$
$$\mathbf{p}_{j \to i} = \operatorname{softmax}(\operatorname{logits}(r_j, r_i))$$

### 2.1 Directional Aggregation

To obtain symmetric, order-invariant pairwise metrics for the unordered pair $(r_i, r_j)$, we compute the arithmetic mean across both directions:

$$\text{pair\_entailment} = \frac{p_{i \to j}^{\text{entail}} + p_{j \to i}^{\text{entail}}}{2}$$

$$\text{pair\_contradiction} = \frac{p_{i \to j}^{\text{contra}} + p_{j \to i}^{\text{contra}}}{2}$$

$$\text{pair\_neutral} = \frac{p_{i \to j}^{\text{neut}} + p_{j \to i}^{\text{neut}}}{2}$$

Because $\sum_c p_{i \to j}^c = 1.0$ and $\sum_c p_{j \to i}^c = 1.0$, the aggregated pair probabilities strictly sum to $1.0$:
$$\text{pair\_entailment} + \text{pair\_contradiction} + \text{pair\_neutral} = 1.0$$

The consensus predicted NLI class for the pair is defined as:
$$\text{predicted\_class} = \operatorname{argmax}\left(\{\text{entailment}: \text{pair\_entailment}, \text{contradiction}: \text{pair\_contradiction}, \text{neutral}: \text{pair\_neutral}\}\right)$$

---

## 3. Example-Level Aggregated NLI Features

For a prompt with $K = 5$ candidate generations, there are $M = \binom{K}{2} = 10$ distinct unordered response pairs.

We extract the following 7 scalar features per example:

### 3.1 Mean Pairwise Probabilities
- **Mean Pairwise Entailment**:
  $$\text{mean\_pairwise\_entailment} = \frac{1}{M} \sum_{m=1}^M \text{pair\_entailment}_m$$
- **Mean Pairwise Contradiction**:
  $$\text{mean\_pairwise\_contradiction} = \frac{1}{M} \sum_{m=1}^M \text{pair\_contradiction}_m$$
- **Mean Pairwise Neutral**:
  $$\text{mean\_pairwise\_neutral} = \frac{1}{M} \sum_{m=1}^M \text{pair\_neutral}_m$$

### 3.2 Discrete Pair Fractions
- **Fraction Entailing Pairs**:
  $$\text{fraction\_entailing\_pairs} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}(\text{predicted\_class}_m == \text{'entailment'})$$
- **Fraction Contradicting Pairs**:
  $$\text{fraction\_contradicting\_pairs} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}(\text{predicted\_class}_m == \text{'contradiction'})$$
- **Fraction Neutral Pairs**:
  $$\text{fraction\_neutral_pairs} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}(\text{predicted\_class}_m == \text{'neutral'})$$

### 3.3 NLI Disagreement Score
We define the scalar **NLI Disagreement Score** as:
$$\text{nli\_disagreement} = \text{mean\_pairwise\_contradiction} + 0.5 \times \text{mean\_pairwise\_neutral}$$

- **Exact interpretation**:
  - Direct factual collision (contradiction) receives full penalty (weight $1.0$).
  - Informational divergence or failure to entail (neutrality) receives intermediate penalty (weight $0.5$).
  - Mutual entailment receives zero penalty (weight $0.0$).
- **Properties**:
  - Equivalently: $\text{nli\_disagreement} = 1.0 - \text{mean\_pairwise\_entailment} - 0.5 \times \text{mean\_pairwise\_neutral}$.
  - Strictly bounded in $[0.0, 1.0]$.
  - Continuous, linear, deterministic, and free from division-by-zero singularities.

---

## 4. Experimental Setup & Model Details

### 4.1 Model Selection
- **Model**: `cross-encoder/nli-MiniLM2-L6-H768` (MiniLMv2 architecture, 6 transformer layers, hidden dimension 768, ~80M parameters).
- **Execution Device**: CUDA GPU (`cuda:0`, NVIDIA RTX 3050 Laptop).
- **Inference Mode**: Zero-shot feature extraction with frozen weights (no fine-tuning).
- **Resolved Label Indices**:
  - `0: contradiction`
  - `1: entailment`
  - `2: neutral`

### 4.2 Pilot Data Controls
- **Input Data**: The exact 100 generated candidate responses from the Phase 7 self-consistency pilot ([`pilot_generations_raw.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/self_consistency/pilot_generations_raw.parquet)).
- **No new generations were created**: Guarantees direct experimental comparability between self-consistency and NLI signals.
- **Evaluation scale**:
  - 20 benchmark examples
  - 5 generations per example
  - 10 unordered pairs per example $\implies 200$ total response pairs
  - 2 directional evaluations per pair $\implies 400$ total directional inference calls.

---

## 5. Pilot Empirical Results & Diagnostics

The pilot evaluation executed on CUDA GPU with batch size 32:

### 5.1 Aggregate Diagnostic Metrics

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **NLI Model Name** | `cross-encoder/nli-MiniLM2-L6-H768` | 6 layers, ~80M params |
| **Device** | `cuda:0` | NVIDIA GeForce RTX 3050 |
| **Model Loading Time** | 7.553 s | Initial weights instantiation |
| **Total Inference Time** | **1.293 s** | 400 directional evaluations |
| **Average Pair Inference Time** | **0.0065 s (6.5 ms)** | Evaluates both directions |
| **Average Directional Inference Time** | **0.0032 s (3.2 ms)** | Per text pair |
| **Total Examples** | 20 | Identical seed-42 sample |
| **Total Pair Count** | 200 | $20 \times 10$ pairs |
| **Total Directional Evaluations** | 400 | $200 \times 2$ |
| **Failed Evaluations** | **0** | 100% completion rate |
| **Mean Entailment Probability** | **0.4175** | Average pairwise entailment |
| **Mean Contradiction Probability** | **0.2306** | Average pairwise contradiction |
| **Mean Neutral Probability** | **0.3519** | Average pairwise neutrality |
| **Mean Fraction Entailing Pairs** | **0.3400 (34.0%)** | Majority consensus entailment |
| **Mean Fraction Contradicting Pairs** | **0.2250 (22.5%)** | Majority consensus contradiction |
| **Mean Fraction Neutral Pairs** | **0.4350 (43.5%)** | Majority consensus neutrality |
| **Mean NLI Disagreement Score** | **0.4066** | $\text{contra} + 0.5 \times \text{neutral}$ |
| **Peak GPU VRAM Allocated** | **481.33 MB** | Extremely lightweight footprint |

### 5.2 Per-Dataset Breakdown

| Dataset | Count | Mean Entailment | Mean Contradiction | Mean Neutral | Frac Entail | Frac Contra | Frac Neutral | NLI Disagreement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HaluEval** | 12 | **0.5989** | 0.2082 | 0.1928 | 0.5333 | 0.2167 | 0.2500 | **0.3047** |
| **FEVER** | 6 | 0.1427 | 0.1952 | **0.6621** | 0.0500 | 0.1833 | **0.7667** | **0.5263** |
| **TruthfulQA** | 2 | 0.1531 | **0.4710** | 0.3758 | 0.0500 | **0.4000** | 0.5500 | **0.6590** |

#### Domain Observations:
1. **HaluEval**: High mean entailment ($0.599$) and $53.3\%$ entailing pairs. When given passage context, Qwen generations tend to converge toward consistent, mutually entailing factual answers.
2. **FEVER**: Heavy neutral dominance ($66.2\%$ mean probability, $76.7\%$ neutral pairs). Because FEVER prompts generate complex multi-sentence rebuttals with differing historical details, the NLI model correctly flags them as mutually neutral rather than strictly entailing.
3. **TruthfulQA**: Highest contradiction probability ($0.471$) and highest disagreement score ($0.659$). Confirmatory of TruthfulQA's adversarial, deceptive prompts where LLM sampling oscillates between common falsehoods, corporate brands, and refutations.

---

## 6. Qualitative Pairwise Inspection

Below are three representative qualitative cases illustrating the NLI model's ability to distinguish factual agreement, direct collision, and paraphrasing:

### Case 1: Direct Literal Consensus (`truthfulqa_152_corr_0`, Pair (0, 1))
- **Response A**: `"Microsoft."`
- **Response B**: `"Microsoft."`
- **Probabilities**:
  - $P(\text{Entailment}) = \mathbf{0.9913}$
  - $P(\text{Contradiction}) = 0.0007$
  - $P(\text{Neutral}) = 0.0080$
- **Predicted Class**: `entailment`
- **Analysis**: Identical responses exhibit near-certain mutual entailment.

### Case 2: Clear Factual Contradiction (`halueval-8758`, Pair (0, 3))
- **Prompt**: *"Kevin Smith was a member of a team that defeated what American Football Conference champions?"*
- **Response A**: `"New England Patriots."`
- **Response B**: `"The Green Bay Packers."`
- **Probabilities**:
  - $P(\text{Entailment}) = 0.0004$
  - $P(\text{Contradiction}) = \mathbf{0.9985}$
  - $P(\text{Neutral}) = 0.0012$
- **Predicted Class**: `contradiction`
- **Analysis**: While sentence embeddings reported $0.765$ similarity due to shared NFL context, the NLI cross-encoder correctly identifies that the entity assertions are mutually exclusive ($99.85\%$ contradiction).

### Case 3: Paraphrased Factual Agreement (`halueval-6985`, Pair (2, 3))
- **Prompt**: Factual query regarding the performer of the song *B.L.O.W.*
- **Response A**: `"B.L.O.W. is a song by Tory Lanez, who is a Canadian rapper."`
- **Response B**: `"B.L.O.W. is a song by Canadian rapper Tory Lanez."`
- **Probabilities**:
  - $P(\text{Entailment}) = \mathbf{0.9916}$
  - $P(\text{Contradiction}) = 0.0007$
  - $P(\text{Neutral}) = 0.0077$
- **Predicted Class**: `entailment`
- **Analysis**: The two responses use different syntactic constructions (relative clause vs pre-nominal adjective phrase) but assert the exact same proposition. The NLI model assigns $99.16\%$ entailment, capturing true factual invariant consensus.

---

## 7. Output Artifacts & Data Schema

Outputs are persisted in [`experiments/baselines/nli_agreement/`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/nli_agreement/):

1. **`pilot_nli_pairwise.parquet` / `.csv`** (200 rows):
   - `id`: Example identifier.
   - `source_dataset`: Benchmark source.
   - `original_label`: Benchmark ground truth (metadata only).
   - `pair_index_i`: First generation index ($0 \dots 3$).
   - `pair_index_j`: Second generation index ($1 \dots 4$, $i < j$).
   - `response_i`: Raw generated text for response $i$.
   - `response_j`: Raw generated text for response $j$.
   - `prob_entail_i_to_j`, `prob_contra_i_to_j`, `prob_neutral_i_to_j`: Directional probabilities for $i \to j$.
   - `prob_entail_j_to_i`, `prob_contra_j_to_i`, `prob_neutral_j_to_i`: Directional probabilities for $j \to i$.
   - `pair_entailment`: Mean directional entailment.
   - `pair_contradiction`: Mean directional contradiction.
   - `pair_neutral`: Mean directional neutral.
   - `predicted_class`: Consensus predicted class (`'entailment'`, `'contradiction'`, `'neutral'`).

2. **`pilot_nli_aggregated.parquet` / `.csv`** (20 rows):
   - `id`: Example identifier.
   - `source_dataset`: Benchmark source.
   - `original_label`: Benchmark label.
   - `num_generations`: $5$.
   - `mean_pairwise_entailment`: Example mean entailment.
   - `mean_pairwise_contradiction`: Example mean contradiction.
   - `mean_pairwise_neutral`: Example mean neutral.
   - `fraction_entailing_pairs`: Proportion of entailing pairs.
   - `fraction_contradicting_pairs`: Proportion of contradicting pairs.
   - `fraction_neutral_pairs`: Proportion of neutral pairs.
   - `nli_disagreement`: Scalar disagreement score.

3. **`pilot_nli_summary.json`**:
   - Machine-readable summary of runtime latency, memory, aggregate probabilities, and per-dataset distributions.

---

## 8. Methodological Limitations & Scientific Disclaimers

> [!CAUTION]
> ### Scientific Disclaimers
> 1. **NLI Agreement is NOT Factual Truth**: If an LLM consistently hallucinates a falsehood across all stochastic samples, the generated responses will mutually entail each other with zero contradiction. NLI agreement measures logical consistency across candidate samples, not ground-truth veracity.
> 2. **Evaluation-Only Pilot**: This pilot is designed strictly to validate feature extraction efficiency, directional aggregation logic, and signal quality on 20 examples. Classification metrics (Accuracy, ROC-AUC) cannot and should not be inferred from this pilot.
> 3. **Label Isolation Preserved**: The benchmark's `original_label` was not used to condition, filter, or compute NLI features.
