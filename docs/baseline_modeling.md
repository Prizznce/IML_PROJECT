# Baseline Classification Modeling Pipeline (Phase 6)

## 1. Overview & Scientific Objectives

Phase 6 establishes reproducible, supervised classification baselines for detecting Large Language Model (LLM) hallucinations using internal generation signals extracted from `Qwen/Qwen3.5-0.8B`. 

The modeling pipeline operates on the canonical 4,000-example supervised feature dataset:
[`experiments/baselines/supervised_signals_combined.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/supervised_signals_combined.parquet)

In this dataset, internal generation signals (token likelihoods, predictive entropy, sequence perplexity, and token rank distributions) were extracted strictly over the **original labeled benchmark responses** (from HaluEval, TruthfulQA, and FEVER), completely eliminating label misalignment.

The modeling pipeline is implemented in:
- [`src/training/baseline_models.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/training/baseline_models.py)
- [`src/training/__init__.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/training/__init__.py)
- Unit tests: [`tests/test_baseline_models.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/tests/test_baseline_models.py)

---

## 2. Feature Selection Policy & Rationale

### 2.1 The 11 Primary Internal Signal Features

The primary baseline uses strictly **11 internal signal features**:

| Feature Name | Source Column | Transformation | Theoretical Rationale |
| :--- | :--- | :--- | :--- |
| `min_log_prob` | `min_log_prob` | Identity | Captures the weakest token transition in the sequence. A sharp drop in log-probability often flags factual drift or unsupported entity hallucination. |
| `mean_log_prob` | `mean_log_prob` | Identity | Average log-likelihood across the response sequence; measures global sequence confidence. |
| `mean_token_prob` | `mean_token_prob` | Identity | Arithmetic mean of token generation probabilities, complementing geometric mean (mean log-prob). |
| `token_prob_std` | `token_prob_std` | Identity | Dispersion of token probabilities; identifies sequences with erratic confidence fluctuations. |
| `mean_entropy` | `mean_entropy` | Identity | Average Shannon entropy $H(P_t) = -\sum p_v \log p_v$; quantifies predictive distribution spread across vocabulary. |
| `max_entropy` | `max_entropy` | Identity | Peak Shannon entropy; flags focal decision points where the model experienced maximum uncertainty. |
| `entropy_std` | `entropy_std` | Identity | Variability of predictive uncertainty across the sequence. |
| `log_perplexity` | `perplexity` | $\log(\text{perplexity})$ | Sequence perplexity $\exp(-\text{mean\_log\_prob})$. Stabilized in log-space to compress heavy tails (up to $6.9 \times 10^7$). |
| `log_mean_token_rank` | `mean_token_rank` | $\log(1 + \text{mean\_token\_rank})$ | Mean position of the target token in the sorted vocabulary distribution. Log-scaled to tame heavy-tailed ranks. |
| `log_max_token_rank` | `max_token_rank` | $\log(1 + \text{max\_token\_rank})$ | Worst-case vocabulary rank in the sequence. Directly signals rare, out-of-distribution, or uncharacteristic token selections. |
| `log_rank_std` | `rank_std` | $\log(1 + \text{rank\_std})$ | Standard deviation of token ranks, measuring instability in vocabulary ranking positions. |

### 2.2 Mathematical Transformations on Raw Features
Raw token rank and perplexity metrics exhibit extreme heavy-tailed skewness across 4,000 benchmark samples (perplexity skewness = +63.24, max rank skewness = +12.91). To ensure numerical stability and satisfy linearity assumptions:
$$\text{log\_perplexity} = \log(\max(\text{perplexity}, 10^{-12}))$$
$$\text{log\_mean\_token\_rank} = \log(1 + \max(\text{mean\_token\_rank}, 0))$$
$$\text{log\_max\_token\_rank} = \log(1 + \max(\text{max\_token\_rank}, 0))$$
$$\text{log\_rank\_std} = \log(1 + \max(\text{rank\_std}, 0))$$

### 2.3 Strict Exclusion of Sequence Length (`num_tokens`)
A critical finding from our feature quality and leakage analysis ([`docs/feature_quality_analysis.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/feature_quality_analysis.md)) is that response sequence length (`num_tokens`) exhibits strong dataset-dependent correlation with benchmark labels (specifically in HaluEval, where hallucinated responses differ systematically in length from faithful answers).

If `num_tokens` is provided to the baseline classifier:
1. The model exploits token count as a trivial **shortcut feature** (spurious correlation) rather than learning true uncertainty dynamics.
2. The model fails to generalize across tasks or to open-ended generation where response length is unconstrained.

**Policy**: `num_tokens` is **strictly excluded** from the primary baseline feature set. It is only permitted in isolated ablation experiments designed explicitly to quantify shortcut reliance.

### 2.4 Quarantined and Forbidden Columns
The following columns are explicitly excluded from model input matrices:
- `id`: Benchmark sample identifier (arbitrary string).
- `source_dataset`: Benchmark origin (used strictly for stratification).
- `label`: Ground-truth target variable.
- `prompt`, `context`, `response`, `model_input`: Raw text payloads (not tabular signals).
- `forward_time_s`: Hardware inference execution latency.
- `num_tokens`: Response length (quarantined).

---

## 3. Preprocessing Strategy & Leakage Prevention

### 3.1 Separate Pipelines for Logistic Regression and XGBoost

| Consideration | Logistic Regression | XGBoost |
| :--- | :--- | :--- |
| **Model Nature** | Linear parametric model with L2 regularization | Non-parametric gradient-boosted decision tree ensemble |
| **Scale Sensitivity** | **High**: Gradient descent and L2 penalty assume features share commensurate variance. | **Invariant**: Tree split decisions depend strictly on monotonic rank order ($x_j \le \theta$). |
| **Preprocessing Applied** | Log transformations + `StandardScaler` (zero-mean, unit-variance) | Log transformations only (no scaling needed) |
| **Implementation** | Embedded inside `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])` | Directly fed transformed feature DataFrame to `XGBClassifier` |

### 3.2 Strict Prevention of Data Leakage
A common flaw in machine learning workflows is applying `StandardScaler.fit()` or global normalization before splitting into train and test sets. This introduces **lookahead leakage**: the mean and standard deviation of the test partition bleed into the training representation.

Our implementation guarantees zero data leakage:
1. Splitting into `(X_train, X_test, y_train, y_test)` occurs **before** any model fitting.
2. `StandardScaler` is wrapped in a scikit-learn `Pipeline`. When `pipeline.fit(X_train, y_train)` is executed, the scaler computes $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ strictly from the training partition.
3. During evaluation, `pipeline.predict(X_test)` transforms test features using the cached training statistics without observing test distribution parameters.

---

## 4. Dataset Partitioning & Stratification

### 4.1 Stratified 80/20 Train/Test Split
The 4,000-example combined dataset is partitioned into:
- **Training Set (80%)**: 3,200 examples.
- **Test Set (20%)**: 800 examples.
- **Random Seed**: Deterministic `random_state = 42`.

### 4.2 Joint Stratification Strategy
Our dataset combines three heterogeneous benchmarks with distinct task formats and query structures:
- HaluEval (1,500 samples, QA with context)
- TruthfulQA (1,500 samples, adversarial misconceptions)
- FEVER (1,000 samples, claim verification)

If stratification were performed only on `label`, a random split could unevenly distribute benchmark origins (e.g., placing 70% of FEVER in train and 30% in test).

To guarantee identical benchmark representation and class balance across partitions, we construct a composite stratification key:
$$\text{stratify\_key} = \text{source\_dataset} + \text{"\_"} + \text{label}$$

This yields 6 distinct stratum buckets:
1. `halueval_0`: 750 samples $\rightarrow$ 600 train, 150 test
2. `halueval_1`: 750 samples $\rightarrow$ 600 train, 150 test
3. `truthfulqa_0`: 750 samples $\rightarrow$ 600 train, 150 test
4. `truthfulqa_1`: 750 samples $\rightarrow$ 600 train, 150 test
5. `fever_0`: 500 samples $\rightarrow$ 400 train, 100 test
6. `fever_1`: 500 samples $\rightarrow$ 400 train, 100 test

### 4.3 Why `source_dataset` is Not a Model Feature
`source_dataset` is metadata denoting benchmark provenance. If exposed to the classifier as an input feature (e.g., one-hot encoded), the model would learn benchmark-specific base rates and task biases instead of intrinsic confidence indicators. It is strictly retained in `metadata` for stratification and post-hoc subgroup error analysis.

---

## 5. Evaluation Metrics & Formulation

Each classifier is evaluated across 8 complementary statistical metrics:

### 5.1 Classification Metrics
1. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision**:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity)**:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1 Score**:
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 5.2 Threshold-Independent Ranking Metrics
5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
   Computes the integral of True Positive Rate against False Positive Rate across all discrimination thresholds. Evaluates global ranking discrimination between faithful and hallucinated sequences.
6. **PR-AUC (Precision-Recall Area Under Curve / Average Precision)**:
   Evaluates precision across recall levels. Essential for measuring positive-class (hallucination) discrimination fidelity.

### 5.3 Probabilistic Calibration Metrics
7. **Brier Score**:
   Measures mean squared error between predicted probability $p_i \in [0, 1]$ and ground truth $y_i \in \{0, 1\}$:
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
8. **Expected Calibration Error (ECE)**:
   Evaluates whether predicted confidence probabilities reflect true empirical accuracy. Predicted probabilities are partitioned into $M$ equal-width intervals $B_1, \dots, B_M$ (default: $M=10$, width 0.1):
   $$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(y_i == 1)$$
   $$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i$$
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   A transparent NumPy implementation is built into `src/training/baseline_models.py` without requiring external unverified packages.

### 5.4 Confusion Matrix
Provides explicit counts of:
- **True Positives ($TP$)**: Correctly identified hallucinations.
- **True Negatives ($TN$)**: Correctly identified faithful responses.
- **False Positives ($FP$)**: Faithful responses mistakenly flagged as hallucinations (Type I error).
- **False Negatives ($FN$)**: Hallucinations that escaped detection (Type II error).

---

## 6. Planned Baseline Experiments (Execution Roadmap)

The baseline experiment will be executed in subsequent steps of Phase 6:

### 6.1 Execution Command
The baseline classification pipeline will be executed via the dedicated CLI:

```bash
.\.venv\Scripts\python.exe src/training/baseline_models.py --data experiments/baselines/supervised_signals_combined.parquet --output-dir experiments/baselines/results --test-size 0.2 --random-state 42
```

### 6.2 Planned Experimental Steps
1. **Primary Baseline Comparison**:
   - Train Logistic Regression (StandardScaler + L2 regularization, `max_iter=1000`).
   - Train XGBoost (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0.1`).
   - Compare discrimination (ROC-AUC, PR-AUC, F1) and calibration (Brier Score, ECE) on the 800-sample test set.
2. **Feature Importance & Signal Ranking**:
   - Inspect Logistic Regression normalized coefficients $\beta_j$.
   - Inspect XGBoost gain and split importances to identify which internal signals provide the strongest hallucination signatures.
3. **Sequence-Length Ablation Study**:
   - Execute pipeline with `--include-num-tokens`.
   - Measure delta in performance ($\Delta \text{ROC-AUC}$, $\Delta F_1$) to quantify length-bias reliance.
4. **Cross-Benchmark Generalization (Out-of-Domain Transfer)**:
   - Train on two benchmarks (e.g., HaluEval + FEVER) and test on the unseen benchmark (e.g., TruthfulQA) to assess cross-task signal transferability.
5. **Post-Hoc Probability Calibration**:
   - Apply Platt scaling / Isotonic regression on validation splits to minimize ECE.
