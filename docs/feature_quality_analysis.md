# Feature Quality, Distribution, and Leakage Analysis Report

## 1. Executive Summary

This report delivers a comprehensive read-only audit of the completed 4,000-example supervised benchmark signal dataset:
[`experiments/baselines/supervised_signals_combined.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/supervised_signals_combined.parquet)

The dataset was generated using unadulterated causal forward passes with `Qwen/Qwen3.5-0.8B` (`bfloat16`) on the local NVIDIA GeForce RTX 3050 Laptop GPU. Every row evaluates the **original labeled benchmark response** conditioned on the prompt and context, guaranteeing that the ground-truth binary label strictly characterizes the scored response sequence.

---

## 2. Dataset Overview

### 2.1 Dimensions and Provenance
- **Total Records**: Exactly **4,000** rows.
- **Total Columns**: **22** columns.
- **Missing / Null Values**: **0** across all 22 columns (100% complete).
- **NaN / Infinite Values**: **0** across all 14 numerical signal features.

### 2.2 Dataset Breakdown & Balance
| Source Dataset | Total Rows | Faithful (Label 0) | Hallucinated (Label 1) | Balance Ratio | Task Type |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **HaluEval** | 1,500 | 750 | 750 | 50.0% / 50.0% | Question Answering with Reference Context |
| **TruthfulQA** | 1,500 | 750 | 750 | 50.0% / 50.0% | Open-domain Misconceptions & False Beliefs |
| **FEVER** | 1,000 | 500 | 500 | 50.0% / 50.0% | Fact Verification / Claim Support |
| **Combined Total** | **4,000** | **2,000** | **2,000** | **50.0% / 50.0%** | Multi-benchmark Canonical Collection |

### 2.3 Column Inventory and Data Types
| Column Name | Data Type | Role | Included in Classifier? |
| :--- | :--- | :--- | :---: |
| `id` | `object` (str) | Canonical benchmark identifier | **NO** (ID metadata) |
| `source_dataset` | `object` (str) | Provenance tag (`halueval`, `truthfulqa`, `fever`) | **NO** (Partition metadata) |
| `label` | `int64` | Ground truth (0 = Faithful, 1 = Hallucinated) | **NO** (Target variable) |
| `prompt` | `object` (str) | Benchmark instruction / query | **NO** (Text payload) |
| `context` | `object` (str) | Reference passage or evidence metadata | **NO** (Text payload) |
| `response` | `object` (str) | Evaluated benchmark response | **NO** (Text payload) |
| `model_input` | `object` (str) | Conditioning prefix fed to LLM | **NO** (Text payload) |
| `forward_time_s` | `float64` | Execution latency in seconds | **NO** (Hardware runtime metric) |
| `num_tokens` | `int64` | Sequence length of response tokens | **YES** (Candidate feature) |
| `mean_token_prob` | `float64` | Arithmetic mean of token probabilities | **YES** (Signal feature) |
| `min_token_prob` | `float64` | Minimum token probability across response | **YES** (Signal feature) |
| `mean_log_prob` | `float64` | Mean token log probability | **YES** (Signal feature) |
| `min_log_prob` | `float64` | Minimum token log probability | **YES** (Signal feature) |
| `mean_entropy` | `float64` | Mean predictive Shannon entropy | **YES** (Signal feature) |
| `max_entropy` | `float64` | Peak predictive Shannon entropy in sequence | **YES** (Signal feature) |
| `entropy_std` | `float64` | Standard deviation of token entropy | **YES** (Signal feature) |
| `token_prob_std` | `float64` | Standard deviation of token probabilities | **YES** (Signal feature) |
| `perplexity` | `float64` | Sequence perplexity $\exp(-\text{mean\_log\_prob})$ | **YES** (Signal feature) |
| `mean_token_rank` | `float64` | Mean vocabulary rank of target tokens | **YES** (Signal feature) |
| `max_token_rank` | `int64` | Worst-case vocabulary rank in sequence | **YES** (Signal feature) |
| `min_token_rank` | `int64` | Best-case vocabulary rank in sequence | **YES** (Signal feature) |
| `rank_std` | `float64` | Standard deviation of token ranks | **YES** (Signal feature) |

---

## 3. Feature Distributions & Heavy-Tail Analysis

Summary statistics computed across all 4,000 samples:

| Feature | Min | 1% | 25% | Median | Mean | 75% | 99% | Max | Std | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `num_tokens` | 1.0000 | 1.0000 | 6.0000 | 10.0000 | 10.3715 | 14.0000 | 34.0000 | 54.0000 | 6.6141 | +1.52 |
| `mean_token_prob` | 0.0000 | 0.0002 | 0.2482 | 0.4920 | 0.4737 | 0.6824 | 0.9789 | 0.9992 | 0.2620 | -0.01 |
| `min_token_prob` | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 0.0742 | 0.0093 | 0.9307 | 0.9985 | 0.1976 | **+3.04** |
| `mean_log_prob` | -18.0536 | -9.0710 | -3.9674 | -2.1470 | -2.6500 | -1.0292 | -0.0220 | -0.0008 | 2.0717 | -1.07 |
| `min_log_prob` | -25.2238 | -18.7347 | -12.1163 | -8.5581 | -8.3895 | -4.6785 | -0.0718 | -0.0015 | 4.8109 | **-0.03** |
| `mean_entropy` | 0.0077 | 0.0948 | 0.7467 | 1.3773 | 1.6144 | 2.4062 | 3.9980 | 5.1310 | 1.0309 | +0.56 |
| `max_entropy` | 0.0134 | 0.2802 | 2.3659 | 3.5542 | 3.5952 | 4.8854 | 6.8643 | 7.5417 | 1.6491 | +0.01 |
| `entropy_std` | 0.0000 | 0.0000 | 0.7855 | 1.1749 | 1.1403 | 1.5068 | 2.3174 | 3.4114 | 0.5164 | -0.06 |
| `token_prob_std` | 0.0000 | 0.0000 | 0.2785 | 0.3686 | 0.3325 | 0.4171 | 0.6477 | 0.7070 | 0.1339 | -0.82 |
| `perplexity` | 1.0008 | 1.0222 | 2.7989 | 8.5594 | 17,798.28 | 52.8478 | 8,699.21 | 6.93 × 10⁷ | 1.10 × 10⁶ | **+63.24** |
| `mean_token_rank` | 1.0000 | 1.0000 | 2.0000 | 10.0000 | 222.38 | 126.82 | 2,966.70 | 22,266.11 | 878.44 | **+12.99** |
| `max_token_rank` | 1.0000 | 1.0000 | 7.0000 | 62.5000 | 1,779.32 | 928.50 | 28,878.47 | 195,678.00 | 7,076.51 | **+12.91** |
| `min_token_rank` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.29 | 1.0000 | 28.01 | 5,599.00 | 94.13 | **+53.83** |
| `rank_std` | 0.0000 | 0.0000 | 1.6415 | 18.7079 | 555.97 | 303.67 | 8,404.52 | 50,516.41 | 2,221.91 | **+12.50** |

### Key Heavy-Tail Observations:
1. **Perplexity Extremes**: The median perplexity is **8.56**, and 75% of examples have perplexity $\le 52.85$. However, the maximum is **69,276,710** (skewness = +63.24). This occurs because for strongly refuted or unnatural statements, a single token probability approaches $10^{-8}$, driving negative mean log-prob down to -18.05 and causing $\exp(-\text{mean\_log\_prob})$ to surge into millions.
2. **Rank Heavy Tails**: `max_token_rank` has a median of **62.5**, but a 99th percentile of **28,878** and a max of **195,678** (near the Qwen vocabulary boundary of 248,320).
3. **Log-space Well-Behaved Nature**: Whereas `min_token_prob` has positive skewness (+3.04) compressed near 0.0, its logarithmic counterpart `min_log_prob` has near-zero skewness (**-0.0342**), indicating that log-probabilities provide vastly superior linearity for downstream linear models.

---

## 4. Class-Wise Comparison: Faithful (0) vs. Hallucinated (1)

Statistics computed separately for Faithful ($N=2,000$) and Hallucinated ($N=2,000$):

| Feature | Median (0) | Median (1) | Mean (0) | Mean (1) | Std (0) | Std (1) | Cohen's d | AUC | Cliff's Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `num_tokens` | 7.0000 | 11.0000 | 8.1595 | 12.5835 | 5.3483 | 7.0086 | **+0.7097** | **0.7025** | +0.4050 |
| `min_token_prob` | 0.0007 | 0.0001 | 0.1353 | 0.0130 | 0.2578 | 0.0647 | **-0.6503** | **0.3784** | -0.2432 |
| `min_log_prob` | -7.2551 | -9.3831 | -7.3423 | -9.4367 | 5.2573 | 4.0581 | **-0.4460** | **0.3784** | -0.2432 |
| `max_token_rank` | 29.5000 | 105.0000 | 1,810.80 | 1,747.83 | 8,521.74 | 5,249.49 | -0.0089 | **0.6056** | +0.2111 |
| `rank_std` | 9.4842 | 31.7167 | 572.58 | 539.36 | 2,682.97 | 1,636.27 | -0.0150 | **0.6022** | +0.2045 |
| `token_prob_std` | 0.3460 | 0.3811 | 0.3079 | 0.3572 | 0.1542 | 0.1043 | **+0.3743** | **0.5951** | +0.1902 |
| `max_entropy` | 3.3034 | 3.7411 | 3.3654 | 3.8251 | 1.7551 | 1.5014 | **+0.2815** | **0.5746** | +0.1492 |
| `mean_token_rank` | 7.1714 | 14.1548 | 230.04 | 214.72 | 1,046.07 | 670.32 | -0.0174 | **0.5774** | +0.1547 |
| `perplexity` | 8.3798 | 8.5861 | 819.99 | 34,776.57 | 10,431.04 | 1.55 × 10⁶ | +0.0310 | 0.5396 | +0.0792 |
| `entropy_std` | 1.1569 | 1.1978 | 1.1022 | 1.1785 | 0.5536 | 0.4733 | +0.1483 | 0.5301 | +0.0603 |
| `mean_entropy` | 1.4136 | 1.3578 | 1.5974 | 1.6315 | 1.0580 | 1.0029 | +0.0330 | 0.5147 | +0.0293 |
| `mean_token_prob` | 0.4729 | 0.5017 | 0.4816 | 0.4658 | 0.2909 | 0.2293 | -0.0607 | 0.4860 | -0.0281 |
| `mean_log_prob` | -2.1257 | -2.1501 | -2.6361 | -2.6640 | 2.3097 | 1.8030 | -0.0135 | 0.4604 | -0.0792 |
| `min_token_rank` | 1.0000 | 1.0000 | 3.9905 | 4.5860 | 42.6997 | 126.10 | +0.0063 | 0.4767 | -0.0466 |

*Note on AUC Interpretation: An AUC $< 0.50$ indicates that the feature is inversely associated with hallucination (i.e., higher values indicate Faithful responses). Inverting gives the discriminative power: for `min_token_prob`, $1 - 0.3784 = 0.6216$.*

---

## 5. Correlation Analysis & Feature Redundancy

A full correlation analysis was performed evaluating both linear (Pearson $r$) and monotonic rank-based (Spearman $\rho$) dependencies:

### Highly Correlated & Approximately Redundant Feature Pairs
| Feature 1 | Feature 2 | Spearman $\rho$ | Pearson $r$ | Structural Mechanism / Cause |
| :--- | :--- | :---: | :---: | :--- |
| `mean_log_prob` | `perplexity` | **-1.0000** | -0.1192 | Exact analytical inverse identity: $\text{PPL} = \exp(-\text{mean\_log\_prob})$. |
| `min_token_prob` | `min_log_prob` | **+1.0000** | +0.5976 | Exact monotonic identity: $\text{min\_log\_prob} = \log(\text{min\_token\_prob})$. |
| `mean_token_rank` | `max_token_rank` | **+0.9797** | +0.8997 | Sequence maximum rank strongly dominates the sequence mean rank. |
| `max_token_rank` | `rank_std` | **+0.9789** | +0.9796 | High maximum rank directly inflates sample variance. |
| `mean_token_rank` | `rank_std` | **+0.9541** | +0.9593 | Rank variance scales with the overall average rank magnitude. |
| `min_log_prob` | `max_token_rank` | **-0.9422** | -0.4252 | Lowest probability token corresponds to the worst-case rank in vocabulary. |
| `mean_token_prob` | `perplexity` | **-0.9364** | -0.0293 | High mean probability directly produces low sequence perplexity. |
| `mean_token_prob` | `mean_log_prob` | **+0.9364** | +0.8889 | Strong monotonic agreement across average token confidence. |
| `mean_token_prob` | `mean_entropy` | **-0.8570** | -0.8448 | Higher probability mass concentration corresponds to lower entropy. |
| `mean_entropy` | `max_entropy` | **+0.8248** | +0.8011 | Baseline predictive uncertainty shifts whole entropy distribution upward. |
| `max_entropy` | `entropy_std` | **+0.8196** | +0.7905 | Spikes in maximum entropy inflate sequence entropy dispersion. |

### Redundancy Implications:
1. **Multicollinearity Hazard for Logistic Regression**: Retaining both `mean_log_prob` and `perplexity`, or both `min_token_prob` and `min_log_prob`, creates severe collinearity. `min_log_prob` and `mean_log_prob` should be preferred over raw probabilities and perplexities in linear models due to bounded numerical stability.
2. **Rank Metrics**: `max_token_rank` and `mean_token_rank` capture essentially the same rank bottleneck ($r > 0.97$). Using $\log(1 + \text{max\_token\_rank})$ captures the worst-case token anomaly effectively.

---

## 6. Outlier Analysis & Extreme Observations

Outliers were quantified using Tukey’s classical Interquartile Range method ($1.5 \times \text{IQR}$) and severe outlier fences ($3.0 \times \text{IQR}$):

| Feature | IQR | 1.5× Upper Fence | Count ($> 1.5\times\text{IQR}$) | Outlier % | Count ($> 3.0\times\text{IQR}$) | Max Observed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `perplexity` | 50.0490 | 127.92 | 593 | 14.8% | **453** | 69,276,710.0 |
| `max_token_rank` | 921.5000 | 2,310.75 | 657 | 16.4% | **433** | 195,678.0 |
| `mean_token_rank` | 124.8205 | 314.05 | 629 | 15.7% | **418** | 22,266.1 |
| `rank_std` | 302.0247 | 756.70 | 649 | 16.2% | **417** | 50,516.4 |
| `min_token_prob` | 0.0093 | 0.0232 | 807 | 20.2% | **723** | 0.9985 |
| `token_prob_std` | 0.1386 | 0.6250 | 345 | 8.6% | 0 | 0.7070 |
| `num_tokens` | 8.0000 | 26.00 | 93 | 2.3% | 18 | 54.0 |
| `mean_log_prob` | 2.9382 | 3.3781 | 59 | 1.5% | 0 | -0.0008 |
| `entropy_std` | 0.7212 | 2.5886 | 15 | 0.4% | 0 | 3.4114 |
| `mean_entropy` | 1.6595 | 4.8953 | 3 | 0.1% | 0 | 5.1310 |
| `min_log_prob` | 7.4378 | 6.4783 | 2 | 0.1% | 0 | -0.0015 |
| `mean_token_prob` | 0.4342 | 1.3338 | 0 | 0.0% | 0 | 0.9992 |
| `max_entropy` | 2.5194 | 8.6645 | 0 | 0.0% | 0 | 7.5417 |

### Actionable Takeaways:
- **Never feed raw `perplexity` into Logistic Regression**: The extreme value $6.93 \times 10^7$ will distort weight optimization.
- **Logarithmic Compression**: Applying $\log(1 + x)$ to `perplexity`, `max_token_rank`, `mean_token_rank`, and `rank_std` compresses the heavy tails into Gaussian-like distributions.
- **Outliers represent real signal, not corrupted data**: These extreme perplexities and ranks correspond to catastrophic factual contradictions where the model assigned near-zero likelihood to the response tokens. They must **not be deleted**, but rather **transformed**.

---

## 7. Dataset-Specific Behavior Analysis

Evaluating the features separately within each source benchmark illuminates critical domain differences:

### 7.1 HaluEval ($N = 1,500$: 750 Faithful, 750 Hallucinated)
*Structure: Question Answering conditioned on reference passages.*

| Feature | Median (0) | Median (1) | Mean (0) | Mean (1) | Separability (AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `num_tokens` | 4.0000 | 14.0000 | 3.8973 | 15.9813 | **0.9628** |
| `min_token_prob` | 0.2441 | 0.0002 | 0.3367 | 0.0084 | **0.1089** ($1-\text{AUC} = \mathbf{0.8911}$) |
| `mean_token_rank` | 1.2000 | 3.8409 | 25.0243 | 52.3589 | **0.7976** |
| `perplexity` | 1.5018 | 4.4539 | 1,276.11 | 92,388.02 | **0.7603** |
| `max_entropy` | 1.7221 | 2.6324 | 1.8203 | 2.8034 | **0.7267** |
| `mean_token_prob` | 0.7937 | 0.6281 | 0.6962 | 0.6003 | **0.2892** ($1-\text{AUC} = \mathbf{0.7108}$) |
| `mean_entropy` | 0.5702 | 0.7511 | 0.7737 | 0.8521 | **0.6144** |

**Observation**: In HaluEval, internal generation signals display **immense discriminative separation**:
- Faithful answers have a median `min_token_prob` of **0.2441** vs **0.0002** for hallucinated answers.
- `mean_token_rank` is **1.20** (almost pure top-1 argmax) for faithful answers vs **3.84** for hallucinations.
- *Note on `num_tokens`*: In HaluEval, hallucinated answers are naturally longer descriptions (mean 15.98 tokens) while faithful answers are concise entity extractions (mean 3.89 tokens). While `num_tokens` achieves AUC 0.96 within HaluEval, using it as a standalone predictor risks length-shortcut learning.

---

### 7.2 TruthfulQA ($N = 1,500$: 750 Faithful, 750 Hallucinated)
*Structure: Questions designed to probe widespread misconceptions and conspiracy theories.*

| Feature | Median (0) | Median (1) | Mean (0) | Mean (1) | Separability (AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `num_tokens` | 10.0000 | 9.0000 | 9.7840 | 9.4760 | **0.4806** (Well matched) |
| `mean_token_prob` | 0.4585 | 0.5298 | 0.4448 | 0.5014 | **0.5835** |
| `max_entropy` | 3.5693 | 3.7787 | 3.5234 | 3.7544 | **0.5583** |
| `perplexity` | 8.7549 | 6.5209 | 626.24 | 115.20 | **0.4430** |
| `min_token_prob` | 0.0005 | 0.0007 | 0.0240 | 0.0264 | **0.5241** |
| `mean_token_rank` | 8.5901 | 9.9706 | 91.7307 | 76.2402 | **0.5115** |
| `mean_entropy` | 1.5142 | 1.4693 | 1.6117 | 1.6032 | **0.4882** |

**Observation**: TruthfulQA exhibits the classic **"confident myth" anomaly**:
- The model exhibits slightly higher probability and lower perplexity on common human myths (Label 1) than on nuanced factual corrections (Label 0).
- Because language models are pre-trained on internet text saturated with common misconceptions, the model can generate popular falsehoods with fluent confidence. This confirms that token confidence alone cannot solve TruthfulQA, highlighting the need for retrieval/external knowledge augmentation.

---

### 7.3 FEVER ($N = 1,000$: 500 Faithful, 500 Hallucinated)
*Structure: Fact-verification claims evaluated without the supporting Wikipedia evidence article.*

| Feature | Median (0) | Median (1) | Mean (0) | Mean (1) | Separability (AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `num_tokens` | 11.0000 | 11.0000 | 12.1160 | 12.1480 | **0.5145** (Identical length) |
| `mean_token_rank` | 300.2888 | 349.5000 | 745.0189 | 665.9779 | **0.5325** |
| `perplexity` | 97.4049 | 100.8551 | 426.4425 | 351.4637 | **0.5168** |
| `max_entropy` | 5.5501 | 5.5539 | 5.4459 | 5.4636 | **0.5086** |
| `mean_entropy` | 2.8305 | 2.8817 | 2.8116 | 2.8429 | **0.5079** |
| `mean_token_prob` | 0.1988 | 0.2052 | 0.2151 | 0.2105 | **0.4978** |
| `min_token_prob` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.4753** |

**Observation**: In FEVER, claims are evaluated zero-shot without Wikipedia evidence articles:
- The model exhibits high uncertainty (mean entropy ~2.84, median perplexity ~100) on both supported and refuted claims.
- Without external retrieval, parametric signals alone hover near AUC ~0.50–0.53. This validates the project thesis: internal generation signals are strongest when reference context is present (HaluEval), but external retrieval/verification is necessary for out-of-context claims (FEVER).

---

## 8. Potential Data Leakage Audit

A thorough audit was performed to guarantee that the supervised signal dataset contains zero leakage:

1. **ID Duplication**: Exactly **0 duplicate IDs** (all 4,000 IDs are distinct).
2. **Prompt-Response Duplication**: Exactly **0 duplicate (prompt, response) pairs**.
3. **Direct Feature Leakage**:
   - No numerical feature has $|r| > 0.35$ with the label.
   - The label is never used in the forward pass or in feature computation.
4. **Prompt Text Cleanliness**:
   - `model_input` strings contain **0 occurrences** of the words `"faithful"` or `"hallucinated"`.
   - The system prompt enforces a neutral directive: `"Answer the question directly, factually, and concisely. Do not provide preamble or internal thinking."`
5. **Categorical / Provenance Leakage**:
   - `source_dataset` is explicitly excluded from model inputs and classifier features.
   - `forward_time_s` (hardware latency) is strictly excluded from classifier training.

**Verdict**: The dataset is completely clean and free of target leakage.

---

## 9. Train / Test Split & Experimental Design Plan

To ensure scientific rigor, we recommend an evaluation plan that separates **within-dataset validation** from **cross-dataset generalization**:

### 9.1 Primary In-Distribution Evaluation: Stratified 80/20 Split
- **Methodology**: 5-fold Stratified Cross-Validation on the combined dataset (or a fixed 80/20 train/test split stratified across both `label` and `source_dataset`).
- **Composition per split**:
  - Training (80%, $N=3,200$): Exactly 1,200 HaluEval, 1,200 TruthfulQA, 800 FEVER (50% balanced labels).
  - Test (20%, $N=800$): Exactly 300 HaluEval, 300 TruthfulQA, 200 FEVER (50% balanced labels).
- **Metric Suite**: ROC-AUC, PR-AUC, Balanced Accuracy, Expected Calibration Error (ECE), and Brier Score.

### 9.2 Secondary Generalization Evaluation: Leave-One-Dataset-Out (LODO)
To test whether internal signals learn generalizable hallucination patterns:
1. **Experiment 1 (Passage-conditioned to Out-of-context)**: Train on `HaluEval`, evaluate zero-shot on `TruthfulQA` and `FEVER`.
2. **Experiment 2 (Fact-verification to Misconceptions)**: Train on `FEVER + HaluEval`, evaluate on `TruthfulQA`.
3. **Experiment 3 (TruthfulQA to QA)**: Train on `TruthfulQA`, evaluate on `HaluEval`.

---

## 10. Feature Transformation & Engineering Recommendations

For the subsequent modeling phase:

### 10.1 Transformations for Linear Models (Logistic Regression)
1. **Perplexity**: Apply log-transformation:
   $$\tilde{x}_{\text{ppl}} = \log(\text{perplexity}) = -\text{mean\_log\_prob}$$
   *(Note: `mean_log_prob` is already the exact negative log-perplexity and is naturally bounded!)*
2. **Token Ranks**: Apply logarithmic compression:
   $$\tilde{x}_{\text{mean\_rank}} = \log(1 + \text{mean\_token\_rank})$$
   $$\tilde{x}_{\text{max\_rank}} = \log(1 + \text{max\_token\_rank})$$
3. **Minimum Token Probability**: Use `min_log_prob` rather than `min_token_prob` to avoid non-linear compression near zero.
4. **Standardization**: Apply `StandardScaler` ($\mu = 0, \sigma = 1$) fitted strictly on the training folds.

### 10.2 Transformations for Tree Models (XGBoost)
- Tree-based splits are invariant to monotonic scaling, but using log-transformed rank and perplexity features prevents numerical float overflow during gradient boosting calculations.

---

## 11. Initial Classifier Feature Set Design

### Included Signal Features (11 Features)
1. `min_log_prob` (Uncertainty floor: minimum log-probability of any response token)
2. `mean_log_prob` (Overall sequence confidence: negative cross-entropy)
3. `mean_token_prob` (Average raw probability)
4. `token_prob_std` (Probability dispersion across tokens)
5. `mean_entropy` (Average predictive entropy)
6. `max_entropy` (Worst-case single-token predictive entropy)
7. `entropy_std` (Entropy dispersion across tokens)
8. `log_perplexity` ($\log(\text{perplexity})$)
9. `log_mean_token_rank` ($\log(1 + \text{mean\_token\_rank})$)
10. `log_max_token_rank` ($\log(1 + \text{max\_token\_rank})$)
11. `log_rank_std` ($\log(1 + \text{rank\_std})$)

*Optional Length Baseline Feature*:
- `num_tokens` (To be evaluated both **with** and **without** to test for length shortcut learning).

### Excluded Columns (Strictly Omitted from Classifier Inputs)
- `id` (Identifiers)
- `source_dataset` (Dataset provenance)
- `label` (Target variable)
- `prompt` (Raw text payload)
- `context` (Raw text payload)
- `response` (Raw text payload)
- `model_input` (Conditioning string)
- `forward_time_s` (Hardware runtime latency)

---

## 12. Conclusion & Readiness Assessment

- **Dataset Integrity**: **100% complete**. 4,000 rows, perfectly balanced (50/50), zero missing values, zero NaNs, zero Infs, zero duplicate pairs, zero target leakage.
- **Signal Validity**: Strong, statistically significant separation observed between faithful and hallucinated responses in passage-conditioned tasks (`min_token_prob`, `mean_token_rank`, `max_entropy`).
- **Theoretical Coherence**: The contrast between HaluEval (strong internal signals), TruthfulQA (confident misconceptions), and FEVER (out-of-context uncertainty) precisely mirrors the foundational literature.
- **Readiness**: **The supervised signal dataset is 100% READY for the modeling phase (Phase 6).**
