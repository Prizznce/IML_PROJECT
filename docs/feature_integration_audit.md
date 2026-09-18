# Feature Integration Audit: Canonical Schema Design for Combined Hallucination Detection

**Project:** Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection  
**Phase:** Feature Integration Audit & Final Supervised Schema Specification  
**Status:** Completed & Validated  
**Artifacts Generated:**
- [`docs/feature_integration_audit.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/feature_integration_audit.md)
- [`configs/combined_features.yaml`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/configs/combined_features.yaml)
- [`tests/test_feature_integration.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/tests/test_feature_integration.py)

---

## 1. Executive Summary & Audit Objectives

Following the successful pilot implementations of all four proposal-required signal groups:
1. **Internal Generation Signals** (white-box token probabilities, entropy, perplexity, rank statistics via Qwen3.5-0.8B)
2. **Self-Consistency Signals** (black-box discrete agreement and continuous semantic similarity across $K=5$ stochastic samples)
3. **NLI Agreement Signals** (bidirectional premise $\leftrightarrow$ hypothesis logical inference via CrossEncoder)
4. **Evidence Retrieval Agreement Signals** (dense passage retrieval, query-evidence similarity, response support, and evidence margin via bi-encoder)

This **Feature Integration Audit** designs the exact, final supervised feature schema before executing any resource-intensive full-scale generation or scoring across the 4,000 benchmark examples.

### Core Methodological Invariant: Strict Label Isolation
> [!IMPORTANT]
> **Fundamental Grounding Principle:**
> The final supervised classifier evaluates whether the **ORIGINAL benchmark response** produced for an input prompt was factual (`label = 0`) or a hallucination (`label = 1`).
>
> - Under **no circumstances** is an original benchmark ground-truth label ever attached to an independently generated stochastic response.
> - Stochastic generations ($K=5$) exist purely as **unlabelled consensus candidates** to quantify the model's epistemic uncertainty.
> - Both the Supervised Benchmark Evaluation workflow and the Real-Time / Demo workflow maintain strict boundary separation between inputs, candidate responses, and target labels.

---

## 2. Comprehensive Audit of the Four Feature Families

### 2.1 Family A: Internal Generation Signals (11 Primary Features)

Extracted directly from the base causal LM (`Qwen/Qwen3.5-0.8B`) during forward evaluation over the concatenated prompt and original response:

$$\mathcal{L}_{int} = \{ \text{min\_log\_prob}, \text{mean\_log\_prob}, \text{mean\_token\_prob}, \text{token\_prob\_std}, \text{mean\_entropy}, \text{max\_entropy}, \text{entropy\_std}, \log(\text{ppl}), \log(1 + \bar{r}), \log(1 + r_{max}), \log(1 + \sigma_r) \}$$

- **Variance-Stabilizing Log Transforms**: Applied to heavy-tailed rank and perplexity metrics:
  - $\text{log\_perplexity} = \log(\max(\text{perplexity}, 10^{-12}))$
  - $\text{log\_mean\_token\_rank} = \log(1 + \max(\text{mean\_token\_rank}, 0))$
  - $\text{log\_max\_token\_rank} = \log(1 + \max(\text{max\_token\_rank}, 0))$
  - $\text{log\_rank\_std} = \log(1 + \max(\text{rank\_std}, 0))$
- **Sequence Length Exclusion Policy**: Sequence length (`num_tokens`) is strictly an **optional ablation feature**. The controlled ablation experiment confirmed that `num_tokens` acts as an empirical dataset shortcut (improving in-domain TruthfulQA while deteriorating out-of-domain transfer). It is **strictly forbidden** from the primary combined feature set.
- **Availability**: **Universal**. Extractable for 100% of benchmark examples.

---

### 2.2 Family B: Self-Consistency Signals (8 Candidate Features)

Extracted across $K=5$ stochastic generations sampled at temperature $T=0.7$, $\text{top\_p}=0.9$ using deterministic text normalization and `sentence-transformers/all-MiniLM-L6-v2`:

1. `exact_match_agreement`: Fraction of the $\binom{K}{2}$ pairs sharing identical normalized strings.
2. `unique_response_ratio`: Ratio of unique normalized responses to total responses ($|\text{unique}| / K$).
3. `majority_response_fraction`: Frequency fraction of the modal normalized response ($\max(\text{count}) / K$).
4. `mean_pairwise_similarity`: Average cosine similarity across all $\binom{K}{2}$ unordered pairs.
5. `min_pairwise_similarity`: Minimum cosine similarity across pairs (worst-case divergence).
6. `max_pairwise_similarity`: Maximum cosine similarity across pairs (best-case alignment).
7. `pairwise_similarity_std`: Standard deviation of cosine similarity across pairs.
8. `generation_disagreement`: $1.0 - \text{mean\_pairwise\_similarity}$.

- **Redundancy & Mathematical Identity**:
  - `generation_disagreement` is an exact linear affine transformation of `mean_pairwise_similarity` ($D = 1 - S$). In linear models (Logistic Regression), the Pearson correlation is strictly $-1.000$. Retaining both creates exact collinearity.
  - `unique_response_ratio` and `majority_response_fraction` correlate at $r = -1.000$ on the pilot dataset and at $|r| = 0.972$ with `exact_match_agreement`.
- **Availability**: **Universal**. Extractable for any prompt via stochastic sampling.

---

### 2.3 Family C: NLI Agreement Signals (7 Candidate Features)

Evaluates premise $\leftrightarrow$ hypothesis logical inference across all unordered candidate pairs using `cross-encoder/nli-MiniLM2-L6-H768`:

- **Bidirectional Aggregation Confirmation**:
  Pairwise values **ARE already bidirectionally aggregated**. For each unordered pair $(i, j)$ with $i < j$, the cross-encoder evaluates both directional inferences:
  1. Forward: Premise $r_i \to$ Hypothesis $r_j$
  2. Backward: Premise $r_j \to$ Hypothesis $r_i$
  
  The directional probabilities are averaged to produce order-invariant pair probabilities:
  $$\text{pair\_entailment} = \frac{1}{2} \left( P(r_i \models r_j) + P(r_j \models r_i) \right)$$
  $$\text{pair\_contradiction} = \frac{1}{2} \left( P(r_i \perp r_j) + P(r_j \perp r_i) \right)$$
  $$\text{pair\_neutral} = \frac{1}{2} \left( P(r_i \sim r_j) + P(r_j \sim r_i) \right)$$

1. `mean_pairwise_entailment`: Average pair entailment probability across all pairs.
2. `mean_pairwise_contradiction`: Average pair contradiction probability across all pairs.
3. `mean_pairwise_neutral`: Average pair neutral probability across all pairs.
4. `fraction_entailing_pairs`: Fraction of pairs whose consensus argmax class is entailment.
5. `fraction_contradicting_pairs`: Fraction of pairs whose consensus argmax class is contradiction.
6. `fraction_neutral_pairs`: Fraction of pairs whose consensus argmax class is neutral.
7. `nli_disagreement`: $\text{mean\_pairwise\_contradiction} + 0.5 \times \text{mean\_pairwise\_neutral}$.

- **Redundancy & Linear Dependencies**:
  - **Sum-to-One Identity**: Because softmax probabilities sum to 1.0 and directional averaging is linear, $\text{mean\_entailment} + \text{mean\_contradiction} + \text{mean\_neutral} \equiv 1.000$. Including all three introduces rank deficiency in linear classifiers. Retaining `entailment` and `contradiction` captures full continuous information without singularity.
  - **Discrete Argmax Fractions**: `fraction_entailing_pairs` correlates strongly with continuous entailment ($r = 0.915$), and `fraction_contradicting_pairs` correlates strongly with continuous contradiction ($r = 0.968$).
- **Availability**: **Universal**. Extractable for any set of stochastic generations.

---

### 2.4 Family D: Evidence Retrieval Agreement Signals (6 Candidate Features)

Measures dense retrieval relevance and factual groundedness between queries, responses, and reference text using `all-MiniLM-L6-v2`:

1. `top1_evidence_similarity`: $s(q, e_{(1)}) = \mathbf{v}_q \cdot \mathbf{v}_{e_{(1)}}$
2. `mean_top3_evidence_similarity`: $\frac{1}{3}\sum_{j=1}^3 s(q, e_{(j)})$
3. `response_top1_evidence_similarity`: $s(r, e_{(1)}) = \mathbf{v}_r \cdot \mathbf{v}_{e_{(1)}}$
4. `mean_response_top3_evidence_similarity`: $\frac{1}{3}\sum_{j=1}^3 s(r, e_{(j)})$
5. `evidence_margin`: $\max(0.0, s(q, e_{(1)}) - s(q, e_{(2)}))$
6. `retrieval_agreement`: $\max(0.0, s(r, e_{(1)})) \times \max(0.0, s(q, e_{(1)}))$

- **Dataset Provenance & Context Reality**:
  The three project benchmarks exhibit fundamentally different context representations:
  - **HaluEval**: Real textual reference passages (1,499 unique passages) are present in the local processed dataset $\implies$ **Available**.
  - **FEVER**: The local dataset provides only structured document/sentence pointers (`[[[annotation_id, evidence_id, "Wiki_Title", sentence_id]]]`), without the actual Wikipedia sentence text $\implies$ **Missing / Unavailable**.
  - **TruthfulQA**: The local dataset provides external Wikipedia reference URLs/citations, but not fetched article bodies $\implies$ **Missing / Unavailable**.
- **Classification**: **Dataset-Dependent / Partially Unavailable**. 8 of the 20 pilot examples (40%) lack authentic textual evidence in the local dataset.

---

## 3. Data Leakage & Integrity Audit

A comprehensive security and scientific integrity audit confirmed that no improper information reaches the feature table:

| Potential Leakage Vector | Audit Finding | Verification Mechanism |
| :--- | :---: | :--- |
| **Ground-Truth Label Leakage** | **Zero Leakage** | None of the feature extractors (`token_signals`, `self_consistency`, `nli_agreement`, `evidence_retrieval`) take `label` or `original_label` as inputs. Target `label` is strictly separated into vector `y`. |
| **Future Test Set Leakage** | **Zero Leakage** | All scalers (`StandardScaler`) are fit strictly on training splits within an `sklearn.pipeline.Pipeline` or explicit `fit(X_train)` before `transform(X_test)`. Verified in `tests/test_baseline_models.py`. |
| **Target Dataset Statistics Leakage** | **Zero Leakage** | Cross-dataset evaluators fit exclusively on source datasets; target datasets are strictly held out. Verified in `tests/test_cross_dataset.py`. |
| **Generated Response Label Confusion** | **Zero Leakage** | Generated responses are treated strictly as unlabelled consensus samples. Ground-truth labels are exclusively preserved for the original benchmark response. |
| **Dataset Provenance Shortcut Leakage** | **Protected** | `source_dataset` is stored exclusively in `metadata_df` and is explicitly listed in `forbidden_feature_columns`. |
| **Retrieval Corpus Target Overlap** | **Audited & Documented** | The evidence corpus consists strictly of reference passages (`context`), not benchmark responses (`response`). However, for HaluEval examples, the gold reference passage is present in the corpus. In open production settings, an external Wikipedia dump may not contain gold text. |

---

## 4. Empirical Feature Correlation & Redundancy Analysis

Using the 20-example pilot dataset containing aligned outputs across all four signal families, we computed intra-family and cross-family Pearson correlations.

### 4.1 Intra-Family Redundancies

#### Self-Consistency Family
```
                            exact_match  unique_ratio  majority_frac  mean_sim  min_sim  max_sim   std_sim  disagreement
exact_match_agreement             1.000        -0.972          0.972     0.335    0.268    0.408    -0.034        -0.335
unique_response_ratio            -0.972         1.000         -1.000    -0.239   -0.159   -0.483    -0.138         0.239
majority_response_fraction        0.972        -1.000          1.000     0.239    0.159    0.483     0.138        -0.239
mean_pairwise_similarity          0.335        -0.239          0.239     1.000    0.958    0.525    -0.760        -1.000
generation_disagreement          -0.335         0.239         -0.239    -1.000   -0.958   -0.525     0.760         1.000
```
- **Finding 1:** `generation_disagreement` and `mean_pairwise_similarity` have $r = -1.000$. They are mathematically identical under linear reflection ($D = 1 - S$).
- **Finding 2:** `unique_response_ratio` and `majority_response_fraction` have $r = -1.000$ and $r = -0.972$ with `exact_match_agreement`.

#### NLI Family
```
                              mean_entail  mean_contra  mean_neutral  frac_entail  frac_contra  frac_neutral  nli_disagree
mean_pairwise_entailment            1.000       -0.502        -0.691        0.915       -0.413        -0.514        -0.904
mean_pairwise_contradiction        -0.502        1.000        -0.278       -0.327        0.968        -0.408         0.823
mean_pairwise_neutral              -0.691       -0.278         1.000       -0.743       -0.350         0.912         0.317
fraction_entailing_pairs            0.915       -0.327        -0.743        1.000       -0.241        -0.714        -0.762
fraction_contradicting_pairs       -0.413        0.968        -0.350       -0.241        1.000        -0.507         0.749
fraction_neutral_pairs             -0.514       -0.408         0.912       -0.714       -0.507         1.000         0.136
nli_disagreement                   -0.904        0.823         0.317       -0.762        0.749         0.136         1.000
```
- **Finding 1:** Mean probabilities sum to 1.0. Including all three produces a singular covariance matrix.
- **Finding 2:** Continuous probabilities correlate strongly with discrete argmax fractions ($r = 0.915$ and $r = 0.968$). Continuous probabilities provide richer gradient information without discretization artifacts.

#### Evidence Retrieval Family
```
                                        top1_q_sim  mean_top3_q  top1_r_sim  mean_top3_r  margin  agreement
top1_evidence_similarity                     1.000        0.947       0.761        0.647   0.878      0.888
mean_top3_evidence_similarity                0.947        1.000       0.771        0.711   0.682      0.857
response_top1_evidence_similarity            0.761        0.771       1.000        0.950   0.602      0.942
mean_response_top3_evidence_similarity       0.647        0.711       0.950        1.000   0.442      0.853
evidence_margin                              0.878        0.682       0.602        0.442   1.000      0.765
retrieval_agreement                          0.888        0.857       0.942        0.853   0.765      1.000
```
- **Finding:** Top-1 and Mean Top-3 similarities are strongly correlated ($r = 0.947$ for query, $r = 0.950$ for response). Top-1 similarity directly matches the rank-1 evidence passage evaluated by `retrieval_agreement`.

---

### 4.2 Cross-Family Complementarity

Cross-family correlations between internal signals and external consistency/retrieval features confirm that these groups capture **orthogonal, non-redundant dimensions** of hallucination:

```
                     exact_match  mean_sim  disagree  mean_entail  mean_contra  nli_disagree  top1_q_sim  top1_r_sim  ret_agreement
mean_token_prob            0.269     0.237    -0.237        0.548        0.015        -0.352       0.431       0.480          0.439
mean_entropy              -0.360    -0.245     0.245       -0.664        0.056         0.464      -0.562      -0.502         -0.508
log_perplexity            -0.144    -0.200     0.200       -0.297       -0.113         0.139      -0.282      -0.514         -0.415
log_mean_token_rank       -0.255     0.110    -0.110       -0.259       -0.199         0.072      -0.340      -0.431         -0.405
```

- **Confidence vs. Consistency**: While higher token confidence (`mean_token_prob`) moderately tracks semantic entailment ($r = 0.548$) and retrieval agreement ($r = 0.439$), the correlations remain in the moderate range ($|r| \in [0.2, 0.6]$).
- **Entropy vs. Divergence**: Elevated token entropy (`mean_entropy`) moderately tracks NLI disagreement ($r = 0.464$) and decreased retrieval agreement ($r = -0.508$).
- **Conclusion**: Combining internal token distributions with black-box consensus and retrieval grounding provides genuine multi-signal variance that a single signal family cannot represent.

---

## 5. Missing-Value Strategy: Comparative Analysis & Recommendation

Because evidence retrieval is available for HaluEval but unavailable for FEVER (structured pointers) and TruthfulQA (URLs) in the local benchmark split, we evaluated four distinct architectural strategies.

### 5.1 Strategy Evaluation Matrix

| Criterion | 1. NaN + Imputation | 2. Explicit Missingness Indicators | 3. Dataset-Specific Models | 4. Modular Tiered Architecture (Recommended) |
| :--- | :---: | :---: | :---: | :---: |
| **Leakage Risk** | Low (if fit on train) | **Fatal Shortcut Leakage** | Low | **Zero Leakage** |
| **Shortcut Susceptibility** | Moderate (constant bias) | **Extremely High** (Flag = `is_halueval`) | Low | **Zero** (Tier 1 is 100% complete) |
| **Cross-Dataset Validity** | Distorts out-of-domain logits | Incompatible out-of-domain | Cannot evaluate zero-shot | **Strictly Preserved** (Tier 1 evaluates cleanly) |
| **Logistic Regression Compatibility** | Compatible | Compatible | Compatible | **Fully Compatible** |
| **XGBoost Compatibility** | Natively handles NaNs | Splits on indicator flag | Compatible | **Fully Compatible** |
| **Operational Simplicity** | Single model, imputed data | Single model, augmented data | Multiple isolated models | Clean two-tier evaluation |

### 5.2 Why Explicit Missingness Indicators Are Prohibited
> [!CAUTION]
> **Fatal Leakage Hazard of Missingness Indicators:**
> In our benchmark configuration, an indicator flag $M_{\text{retrieval}} \in \{0, 1\}$ would equal $1$ if and only if $\text{source\_dataset} == \text{'halueval'}$.
> For FEVER and TruthfulQA, $M_{\text{retrieval}} \equiv 0$.
> 
> A gradient-boosted tree (XGBoost) would immediately select $M_{\text{retrieval}}$ as its root split to partition HaluEval from FEVER/TruthfulQA, bypassing feature learning and inducing **catastrophic shortcut learning**. In cross-dataset evaluation (e.g. Train on FEVER, Test on HaluEval), the indicator value is completely inverted out-of-distribution, rendering the evaluation invalid.

### 5.3 Recommended Strategy: Modular Tiered Architecture with Constant Neutral Imputation

We recommend a **Two-Tier Hierarchical Evaluation Framework**:

#### Tier 1: Universal Multi-Signal Model (Core 26 Features / 19 Primary)
- **Signal Families**: Internal Signals (11) + Self-Consistency (5 primary) + NLI Agreement (3 primary).
- **Properties**:
  - Requires **zero external document stores, Wikipedia dumps, or web retrieval**.
  - **100% available** across all 4,000 examples in HaluEval, FEVER, and TruthfulQA.
  - **Zero missing values** (no imputation artifacts, no shortcut masks).
  - Acts as the definitive **universal baseline** for zero-shot cross-dataset transfer evaluation.

#### Tier 2: Retrieval-Augmented Model (Ablation / Grounded Setting)
- **Signal Families**: Tier 1 Core + Evidence Retrieval Signals (4 primary).
- **Properties**:
  - Evaluated on domains where genuine ground-truth reference passages exist (HaluEval, and future FEVER Wikipedia dumps).
  - For cross-dataset or combined evaluation across all benchmarks, missing retrieval features in non-retrieval datasets are mapped to a **neutral ungrounded constant** ($s = 0.0, \text{margin} = 0.0, \text{agreement} = 0.0$), reflecting an unindexed corpus without binary missingness masks.
  - Evaluated as an ablation study to isolate the **marginal lift of external knowledge grounding** over intrinsic model uncertainty and self-consistency.

---

## 6. Proposed Canonical Final Feature Table

The proposed final supervised feature schema comprises:
- **3 Metadata Columns** (`id`, `source_dataset`, `label`)
- **19 Primary Universal Features** (11 Internal + 5 Self-Consistency + 3 NLI)
- **4 Primary Retrieval Features** (Tier 2 Retrieval-Augmented)
- **9 Optional Ablation / Derived Features** (including `num_tokens`)

| Feature Family | Feature Name | Operational Meaning | Availability | Primary / Optional | Leakage Risk |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Metadata** | `id` | Unique example identifier | Universal | Metadata Only | Strict Exclusion |
| **Metadata** | `source_dataset` | Origin benchmark (`halueval`, `fever`, `truthfulqa`) | Universal | Metadata Only | Strict Exclusion |
| **Metadata** | `label` | Ground truth (0: factual, 1: hallucination) | Universal | Target Variable | Strict Exclusion |
| **Internal** | `min_log_prob` | Minimum token log probability | Universal | Primary Tier 1 | Zero |
| **Internal** | `mean_log_prob` | Average token log probability | Universal | Primary Tier 1 | Zero |
| **Internal** | `mean_token_prob` | Average token probability $[0, 1]$ | Universal | Primary Tier 1 | Zero |
| **Internal** | `token_prob_std` | Standard deviation of token probabilities | Universal | Primary Tier 1 | Zero |
| **Internal** | `mean_entropy` | Average predictive distribution entropy | Universal | Primary Tier 1 | Zero |
| **Internal** | `max_entropy` | Peak predictive entropy | Universal | Primary Tier 1 | Zero |
| **Internal** | `entropy_std` | Standard deviation of predictive entropy | Universal | Primary Tier 1 | Zero |
| **Internal** | `log_perplexity` | Log-transformed sequence perplexity | Universal | Primary Tier 1 | Zero |
| **Internal** | `log_mean_token_rank` | Log-transformed average token rank | Universal | Primary Tier 1 | Zero |
| **Internal** | `log_max_token_rank` | Log-transformed maximum token rank | Universal | Primary Tier 1 | Zero |
| **Internal** | `log_rank_std` | Log-transformed token rank std dev | Universal | Primary Tier 1 | Zero |
| **Internal (Abl)** | `num_tokens` | Total response token count | Universal | Optional Ablation | Length-Shortcut |
| **Consistency** | `exact_match_agreement` | Fraction of identical generation pairs | Universal | Primary Tier 1 | Zero |
| **Consistency** | `mean_pairwise_similarity` | Average cosine similarity across pairs | Universal | Primary Tier 1 | Zero |
| **Consistency** | `min_pairwise_similarity` | Minimum pairwise cosine similarity | Universal | Primary Tier 1 | Zero |
| **Consistency** | `max_pairwise_similarity` | Maximum pairwise cosine similarity | Universal | Primary Tier 1 | Zero |
| **Consistency** | `pairwise_similarity_std` | Standard deviation of pairwise similarities | Universal | Primary Tier 1 | Zero |
| **Consistency (Abl)** | `unique_response_ratio` | Ratio of unique responses ($r = -0.972$) | Universal | Optional Ablation | Zero |
| **Consistency (Abl)** | `majority_response_fraction` | Modal response frequency ($r = 0.972$) | Universal | Optional Ablation | Zero |
| **Consistency (Abl)** | `generation_disagreement` | $1.0 - \text{mean\_pairwise\_similarity}$ | Universal | Derived / Ablation | Zero |
| **NLI** | `mean_pairwise_entailment` | Average bidirectional entailment prob | Universal | Primary Tier 1 | Zero |
| **NLI** | `mean_pairwise_contradiction` | Average bidirectional contradiction prob | Universal | Primary Tier 1 | Zero |
| **NLI** | `nli_disagreement` | $\text{contra} + 0.5 \times \text{neutral}$ | Universal | Primary Tier 1 | Zero |
| **NLI (Abl)** | `mean_pairwise_neutral` | Average bidirectional neutral prob | Universal | Optional Ablation | Collinear ($\sum=1$) |
| **NLI (Abl)** | `fraction_entailing_pairs` | Fraction of pairs with entailment argmax | Universal | Optional Ablation | Zero |
| **NLI (Abl)** | `fraction_contradicting_pairs`| Fraction of pairs with contradiction argmax | Universal | Optional Ablation | Zero |
| **NLI (Abl)** | `fraction_neutral_pairs` | Fraction of pairs with neutral argmax | Universal | Optional Ablation | Collinear ($\sum=1$) |
| **Retrieval** | `top1_evidence_similarity` | Query $\leftrightarrow$ Rank-1 evidence cosine sim | Dataset-Dependent | Primary Tier 2 | Zero |
| **Retrieval** | `response_top1_evidence_similarity` | Response $\leftrightarrow$ Rank-1 evidence cosine sim | Dataset-Dependent | Primary Tier 2 | Zero |
| **Retrieval** | `evidence_margin` | Rank-1 sim $-$ Rank-2 sim | Dataset-Dependent | Primary Tier 2 | Zero |
| **Retrieval** | `retrieval_agreement` | $\max(0, s(r, e_1)) \times \max(0, s(q, e_1))$ | Dataset-Dependent | Primary Tier 2 | Zero |
| **Retrieval (Abl)** | `mean_top3_evidence_similarity` | Mean query similarity across top-3 | Dataset-Dependent | Optional Ablation | Zero |
| **Retrieval (Abl)** | `mean_response_top3_evidence_similarity` | Mean response similarity across top-3 | Dataset-Dependent | Optional Ablation | Zero |

### Strict Forbidden Columns
The following columns MUST NEVER enter model feature matrix $X$:
```python
FORBIDDEN_COLUMNS = {
    # Identifiers & Targets
    "id", "source_dataset", "label",
    # Raw Text Payloads
    "prompt", "context", "response", "model_input", "query_text",
    "retrieved_rank_1_text", "retrieved_rank_2_text", "retrieved_rank_3_text",
    # Diagnostic / Runtime Metadata
    "forward_time_s", "num_generations", "evidence_status",
    "retrieved_rank_1", "retrieved_rank_2", "retrieved_rank_3",
    # Shortcut Ablations (Forbidden from primary model)
    "num_tokens",
}
```

---

## 7. Audit Conclusion & Phase Sign-Off

1. **Schema Finalized**: Canonical definitions and metadata protections codified in [`configs/combined_features.yaml`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/configs/combined_features.yaml).
2. **Mathematical Invariants Verified**: All 13 schema and pilot identity tests passed in [`tests/test_feature_integration.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/tests/test_feature_integration.py).
3. **No Premature Computation**: Neither 4,000 stochastic responses nor full-scale NLI/retrieval were triggered.
4. **Clean Baseline Integrity**: All completed experiments (baseline, calibration, cross-dataset, self-consistency, NLI, evidence retrieval) remain completely unmodified.
