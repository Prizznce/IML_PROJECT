# Cross-Dataset Evaluation Report (Phase 6)

## 1. Purpose of Cross-Dataset Evaluation

The primary baseline established in Phase 6 evaluated classifiers on a combined test partition that shared the same source distribution as the training data (in-domain stratified evaluation). While this benchmark validated that internal logit signals (token probabilities, predictive entropy, sequence perplexity, and vocabulary rank dispersion) correlate with hallucination labels within the combined pool, it does not assess **out-of-domain transferability**.

In real-world deployment, an LLM hallucination detector encounters diverse task formats, prompting styles, and grounding conditions. Cross-dataset evaluation measures whether a classifier trained on one or more benchmark domains can generalize to an unseen domain without re-training or domain adaptation.

---

## 2. Experimental Protocol: Leave-One-Dataset-Out (LODO)

To evaluate domain transfer under strict scientific controls, we employ a **Leave-One-Dataset-Out (LODO)** evaluation protocol across the three canonical benchmarks:
- **HaluEval** (Question Answering with Reference Context)
- **TruthfulQA** (Adversarial Misconceptions and False Beliefs)
- **FEVER** (Fact Verification / Isolated Claim Verification)

### 2.1 The Three Transfer Configurations

| Experiment | Training Sources | Training Samples ($N_{\text{train}}$) | Unseen Target Dataset | Test Samples ($N_{\text{test}}$) | Target Task Type |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **Exp 1** | HaluEval + TruthfulQA | 3,000 (1,500 / 1,500) | **FEVER** | 1,000 (500 / 500) | Isolated claim verification without context |
| **Exp 2** | HaluEval + FEVER | 2,500 (1,250 / 1,250) | **TruthfulQA** | 1,500 (750 / 750) | Open-domain questions probing common false beliefs |
| **Exp 3** | TruthfulQA + FEVER | 2,500 (1,250 / 1,250) | **HaluEval** | 1,500 (750 / 750) | Context-grounded question answering |

### 2.2 Strict Leakage Prevention & Feature Policy
1. **Zero Target Contamination**:
   The target dataset is completely withheld from all model fitting, feature scaling, and parameter selection.
2. **Feature Set (11 Primary Internal Signals)**:
   - Token Probabilities: `min_log_prob`, `mean_log_prob`, `mean_token_prob`, `token_prob_std`
   - Predictive Uncertainty: `mean_entropy`, `max_entropy`, `entropy_std`
   - Sequence Statistics: `log_perplexity` ($\log(\text{perplexity})$)
   - Rank Dispersion: `log_mean_token_rank` ($\log(1 + \text{mean\_token\_rank})$), `log_max_token_rank` ($\log(1 + \text{max\_token\_rank})$), `log_rank_std` ($\log(1 + \text{rank\_std})$)
3. **Quarantined Columns**:
   `num_tokens` is **strictly excluded** from all transfer models, along with `id`, `source_dataset`, `label`, `prompt`, `context`, `response`, `model_input`, and `forward_time_s`.
4. **Preprocessing Execution**:
   For Logistic Regression, `StandardScaler` is fitted **strictly on the training partition** $\mathbf{X}_{\text{train}}$ and transforms $\mathbf{X}_{\text{test}}$ using the training parameters ($\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}}$). Tree-based XGBoost models operate directly on unscaled features from $\mathbf{X}_{\text{train}}$.

### 2.3 Estimator Configurations
- **Logistic Regression**: Scikit-learn `Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])`.
- **XGBoost**: `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, eval_metric='logloss')`.

---

## 3. Quantitative Cross-Dataset Results

Machine-readable JSON metrics for all three experiments are stored in:
[`experiments/baselines/results/cross_dataset/cross_dataset_results.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/results/cross_dataset/cross_dataset_results.json)

### 3.1 Complete Performance Table

| Target Dataset | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE ($M=10$) | Confusion Matrix ($TN, FP, FN, TP$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FEVER**<br>($N=1,000$)<br>Train: HaluEval + TruthfulQA | Logistic Regression<br>XGBoost | 0.5060<br>0.4980 | 0.5038<br>0.4981 | **0.7860**<br>0.5340 | **0.6141**<br>0.5154 | 0.4868<br>0.4967 | 0.4818<br>0.4921 | 0.2952<br>0.2642 | 0.1716<br>0.0950 | TN: 113, FP: 387, FN: 107, TP: 393<br>TN: 231, FP: 269, FN: 233, TP: 267 |
| **TruthfulQA**<br>($N=1,500$)<br>Train: HaluEval + FEVER | Logistic Regression<br>XGBoost | 0.4660<br>0.4773 | 0.4723<br>0.4855 | 0.5800<br>**0.7573** | 0.5206<br>**0.5917** | 0.4439<br>0.4460 | 0.4544<br>0.4531 | 0.3248<br>0.3663 | 0.2299<br>0.2934 | TN: 264, FP: 486, FN: 315, TP: 435<br>TN: 148, FP: 602, FN: 182, TP: 568 |
| **HaluEval**<br>($N=1,500$)<br>Train: TruthfulQA + FEVER | Logistic Regression<br>XGBoost | 0.2140<br>0.3333 | 0.2136<br>0.3494 | 0.2133<br>0.3867 | 0.2135<br>0.3671 | 0.1601<br>0.3316 | 0.3366<br>0.4094 | 0.3563<br>0.3132 | 0.4194<br>0.2905 | TN: 161, FP: 589, FN: 590, TP: 160<br>TN: 210, FP: 540, FN: 460, TP: 290 |

---

## 4. In-Domain Baseline vs. Cross-Dataset Performance

To understand domain transferability, we compare the **in-domain test partition performance** (from Phase 6 primary baseline) against the **cross-dataset transfer performance**:

| Evaluation Paradigm | Training Data | Evaluation Target | LR ROC-AUC | XGB ROC-AUC | LR Accuracy | XGB Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **In-Domain (Stratified Hold-Out)** | 80% Combined Pool | 20% Combined Test ($N=800$) | **0.6930** | **0.7411** | **62.13%** | **67.00%** |
| *In-Domain Subgroup: HaluEval* | 80% Combined Pool | 20% HaluEval Test ($N=300$) | **0.9467** | **0.9357** | **84.67%** | **86.33%** |
| *In-Domain Subgroup: TruthfulQA* | 80% Combined Pool | 20% TruthfulQA Test ($N=300$) | **0.5044** | **0.5983** | **51.33%** | **57.67%** |
| *In-Domain Subgroup: FEVER* | 80% Combined Pool | 20% FEVER Test ($N=200$) | **0.4656** | **0.4854** | **44.50%** | **52.00%** |
| **Cross-Dataset Transfer (LODO)** | HaluEval + TruthfulQA | **FEVER** ($N=1,000$) | 0.4868 | 0.4967 | 50.60% | 49.80% |
| **Cross-Dataset Transfer (LODO)** | HaluEval + FEVER | **TruthfulQA** ($N=1,500$) | 0.4439 | 0.4460 | 46.60% | 47.73% |
| **Cross-Dataset Transfer (LODO)** | TruthfulQA + FEVER | **HaluEval** ($N=1,500$) | 0.1601 | 0.3316 | 21.40% | 33.33% |

---

## 5. Per-Dataset Interpretation & Analysis of Domain Shift

### 5.1 Transfer to FEVER (Target: FEVER; Train: HaluEval + TruthfulQA)
- **Measured Metrics**:
  - Logistic Regression: Accuracy = **50.60%**, ROC-AUC = **0.4868**, Recall = **78.60%**, Precision = **50.38%**.
  - XGBoost: Accuracy = **49.80%**, ROC-AUC = **0.4967**, Recall = **53.40%**, Precision = **49.81%**.
- **Factual Interpretation**:
  Performance on FEVER hovers at chance level (ROC-AUC $\approx 0.49$, Accuracy $\approx 50\%$). FEVER consists of isolated factual assertions without conditioning passages. When trained on HaluEval (which has context passages) and TruthfulQA (conversational misconception queries), the models do not find transferable logit signatures that discriminate refuted versus supported claims in FEVER.

### 5.2 Transfer to TruthfulQA (Target: TruthfulQA; Train: HaluEval + FEVER)
- **Measured Metrics**:
  - Logistic Regression: Accuracy = **46.60%**, ROC-AUC = **0.4439**, F1 = **0.5206**.
  - XGBoost: Accuracy = **47.73%**, ROC-AUC = **0.4460**, F1 = **0.5917**.
- **Factual Interpretation**:
  Transfer performance on TruthfulQA falls slightly below random chance (ROC-AUC $\approx 0.44$). As observed during the feature analysis, TruthfulQA queries probe common human myths and misconceptions. When the model generates a widespread false belief learned during pre-training, it often outputs tokens with low entropy and low rank (high confidence). Models trained on HaluEval and FEVER expect factual hallucinations to exhibit elevated uncertainty, leading to misclassification of fluent misconceptions.

### 5.3 Transfer to HaluEval (Target: HaluEval; Train: TruthfulQA + FEVER)
- **Measured Metrics**:
  - Logistic Regression: Accuracy = **21.40%**, ROC-AUC = **0.1601**, Brier = **0.3563**, ECE = **0.4194**.
  - XGBoost: Accuracy = **33.33%**, ROC-AUC = **0.3316**, Brier = **0.3132**, ECE = **0.2905**.
- **Factual Interpretation**:
  Transfer to HaluEval exhibits severe performance degradation with inverted discrimination (ROC-AUC $< 0.35$).
  - In HaluEval, questions are accompanied by a **reference context passage**. When generating answers conditioned on context, faithful responses exhibit high probability and low rank because they verbatim or semantically reproduce context tokens. Hallucinations depart from the context, producing marked spikes in entropy and vocabulary rank.
  - In TruthfulQA and FEVER, **no reference passage** is provided; responses are generated unconditionally from parametric memory, resulting in fundamentally different baseline log-probabilities and entropy distributions.
  - When a classifier trained strictly on unconditioned text (TruthfulQA + FEVER) is applied to context-grounded text (HaluEval), the distributional shift in feature scales and baselines causes the learned decision thresholds to misclassify context-grounded faithful responses as hallucinations, resulting in high false alarm rates (FP = 589 for LR, 540 for XGBoost out of 750 negative samples).

---

## 6. Discussion: Mechanisms of Dataset and Task Shift

1. **Context Grounding vs. Open-Domain Generation**:
   The presence or absence of a conditioning document alters the model's forward logit dynamics:
   - Grounded generation (HaluEval) concentrates probability mass on context tokens.
   - Ungrounded generation (TruthfulQA, FEVER) distributes probability mass across general vocabulary.
   Classifiers trained without awareness of prompt conditioning structure cannot normalize for this task-level baseline shift.
2. **Memorized Misconceptions vs. Generation Errors**:
   The nature of the hallucination differs:
   - In HaluEval, hallucinations are synthetic factual discrepancies.
   - In TruthfulQA, hallucinations are culturally entrenched false beliefs that the model has seen frequently during web pre-training.
   Internal signals measure epistemic confidence, not objective truth; when a model is confidently incorrect, raw logit signals align with the error rather than the factual truth.
3. **Threshold Brittleness across Domains**:
   Both linear decision boundaries (Logistic Regression) and orthogonal axis splits (XGBoost) fit absolute numerical thresholds on the training distribution. When evaluated on a domain with shifted feature baselines, fixed thresholds fail unless normalized by task type or conditioning length.

---

## 7. Limitations & Scientific Scope

1. **No Target Adaptation**:
   This experiment evaluated pure zero-shot domain transfer without domain adaptation, unlabelled target domain calibration, or task conditioning.
2. **Fixed Parameterization**:
   Models were evaluated with fixed baseline hyperparameters without tuning for cross-domain robustness.
3. **Scope of Claim**:
   These results do not demonstrate that internal signals cannot be used for cross-domain detection; rather, they show that **raw unnormalized internal signals from an unaugmented classifier do not transfer zero-shot across divergent task formats without domain alignment or grounding-aware normalization**.
