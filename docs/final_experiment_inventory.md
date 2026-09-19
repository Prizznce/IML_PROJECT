# Final Experiment Inventory & Methodological Audit

This document provides a comprehensive inventory and methodological audit of all experimental pipelines, datasets, models, metrics, and artifact files completed across **Phases 3–19** of the Hallucination Detection project.

---

## 1. Executive Status

All 11 planned experimental areas have been fully executed, validated, and persisted with zero missing files and 100% test pass rates:
- **Total Supervised Population**: $N = 4,000$ examples across three benchmark sources (HaluEval: 1,500, TruthfulQA: 1,500, FEVER: 1,000).
- **Primary Reference Benchmark (Phase 14)**: Universal Core 19-Feature Detector evaluated on a fixed, stratified 80/20 train/holdout split ($3,200$ train / $800$ holdout test, `random_state=42`).
- **Full Signal Generation Executed**:
  - Internal generation signals extracted for all 4,000 examples (token log-probabilities, entropy, perplexity, rank statistics).
  - Production Self-Consistency executed for all 4,000 examples $\times$ 5 generations ($20,000$ raw generations).
  - Production NLI agreement executed across all $40,000$ unordered pairs ($4,000$ examples $\times 10$ pairs).
  - Complete 19-feature universal feature matrix built and verified.
- **Ablation, Generalization, Calibration & Error Profiles**:
  - 7 feature-family ablation conditions evaluated on the primary 800-instance holdout.
  - Leave-One-Dataset-Out cross-dataset generalization evaluated across 24 experimental configurations.
  - Probability calibration evaluated with zero test-leakage nested splitting (Platt scaling and isotonic regression).
  - Detailed descriptive error analysis completed across all confusion categories, text lengths, and model disagreements.
- **Retrieval-Augmented Variant**: Evaluated as a standalone variant on the usable $N = 2,500$ population (HaluEval + FEVER) across 5 feature conditions ($N = 500$ holdout test).

---

## 2. Comprehensive Experiment Inventory Table

| Phase | Experiment Area | Artifact Directory | Key Output Files | Primary Model(s) | Dataset(s) & Population | Train / Test Setup | Random State | Direct Comparability to Phase 14? |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **3** | Data Ingestion & Harmonization | `data/processed/` | `combined_processed.parquet`, `halueval_processed.parquet`, `truthfulqa_processed.parquet`, `fever_processed.parquet` | N/A (Data Pipeline) | HaluEval (1,500), TruthfulQA (1,500), FEVER (1,000); Total $N=4,000$ | Full population (1:1 balanced, 2,000/2,000) | N/A | Foundational data for all experiments |
| **4–5** | Internal Token Signals | `experiments/baselines/` | `supervised_signals_combined.parquet`, `supervised_signals_combined.csv` | Qwen3.5-0.8B | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Full population extraction | N/A | Feature foundation for Group 1 (11 features) |
| **6–7** | Baseline Classifiers & Token Length Ablation | `experiments/baselines/results/` | `baseline_metrics.json`, `num_tokens_ablation/baseline_metrics.json` | Logistic Regression, XGBoost | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Stratified 80/20 ($3,200$ train / $800$ holdout) | 42 | **Directly Comparable** (serves as the 11-feature baseline) |
| **10–11**| Full Self-Consistency Generation | `experiments/baselines/self_consistency/` | `full_generations_raw.parquet` ($20,000$ rows), `full_self_consistency_aggregated.parquet` ($4,000$ rows) | Qwen3.5-0.8B ($k=5$ samples, temp 0.7) | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Full population generation & aggregation | 42 | Feature foundation for Group 2 (5 features) |
| **12** | Full NLI Agreement Feature Extraction | `experiments/baselines/nli_agreement/` | `full_nli_pairwise.parquet` ($40,000$ rows), `full_nli_aggregated.parquet` ($4,000$ rows) | `cross-encoder/nli-MiniLM2-L6-H768` | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Full population pairwise bidirectional scoring ($40,000$ pairs) | N/A | Feature foundation for Group 3 (3 features) |
| **13** | Universal Feature Matrix Assembly | `experiments/baselines/combined/` | `universal_features.parquet` ($4,000$ rows, 22 cols), `universal_features.csv` | N/A (Feature Engineering) | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Full population assembly of 19 canonical signals | N/A | Canonical 19-feature matrix used in Phases 14–18 |
| **14** | Universal Core Detector Training | `experiments/baselines/combined/` | `combined_model_results.json`, `combined_model_results.csv`, `combined_test_predictions.parquet` | Logistic Regression, XGBoost | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Stratified 80/20 ($3,200$ train / $800$ holdout) | 42 | **PRIMARY REFERENCE BENCHMARK** |
| **15** | Feature-Family Ablation Study | `experiments/ablation/` | `ablation_results.json`, `ablation_results.csv` (14 rows), `ablation_predictions.parquet` | Logistic Regression, XGBoost | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Stratified 80/20 ($3,200$ train / $800$ holdout; 7 feature conditions) | 42 | **Directly Comparable** (evaluated on identical 800 test instances) |
| **16** | Cross-Dataset Transfer Generalization | `experiments/cross_dataset/` | `cross_dataset_results.json`, `cross_dataset_results.csv` (24 rows), `cross_dataset_predictions.parquet` | Logistic Regression, XGBoost | LODO across FEVER ($1,000$), TruthfulQA ($1,500$), HaluEval ($1,500$) | Leave-One-Dataset-Out (Train on 2 datasets, test on 1 unseen) | 42 | **Not Directly Comparable** (different training and holdout populations) |
| **17** | Probability Calibration Analysis | `experiments/calibration/` | `calibration_results.json`, `calibration_results.csv`, `calibration_reliability_tables.csv`, `calibration_predictions.parquet` | Logistic Regression, XGBoost (Uncalibrated, Platt, Isotonic) | HaluEval, TruthfulQA, FEVER ($N=4,000$) | Nested Split: 2,560 model-fit / 640 calibration / 800 holdout | 42 | **Methodologically Distinct** (base model fit on 2,560 vs full 3,200) |
| **18** | Comprehensive Error Analysis | `experiments/error_analysis/` | `error_analysis_results.json`, `error_category_statistics.csv`, `error_by_dataset.csv`, `model_disagreement.csv`, 7 PNG plots | Logistic Regression, XGBoost | Phase 14 Holdout Set ($N=800$) | Post-hoc descriptive analysis on Phase 14 test predictions | 42 | **Directly Comparable** (describes exact Phase 14 predictions) |
| **19** | Retrieval-Augmented Detector Variant | `experiments/retrieval_augmented/` | `retrieval_features.parquet`, `retrieval_experiment_results.json`, `retrieval_experiment_results.csv` (10 rows), `retrieval_predictions.parquet` | Logistic Regression, XGBoost | Usable subset: HaluEval ($1,500$) + FEVER ($1,000$); Total $N=2,500$ | Stratified 80/20 ($2,000$ train / $500$ holdout; 5 feature conditions) | 42 | **Not Directly Comparable** (different population and holdout size) |

---

## 3. Primary Reference Experiment Definition

The **Phase 14 Universal Core Detector** serves as the primary scientific reference point of this investigation:
- **Input Table**: [`experiments/baselines/combined/universal_features.parquet`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/universal_features.parquet) ($N = 4,000$).
- **Features (19 Canonical Signals)**:
  - *Internal Token Statistics (11)*: `min_log_prob`, `mean_log_prob`, `mean_token_prob`, `token_prob_std`, `mean_entropy`, `max_entropy`, `entropy_std`, `log_perplexity`, `log_mean_token_rank`, `log_max_token_rank`, `log_rank_std`.
  - *Self-Consistency Signals (5)*: `exact_match_agreement`, `mean_pairwise_similarity`, `min_pairwise_similarity`, `max_pairwise_similarity`, `pairwise_similarity_std`.
  - *NLI Agreement Signals (3)*: `mean_pairwise_entailment`, `mean_pairwise_contradiction`, `nli_disagreement`.
- **Partitioning Protocol**:
  - Deterministic stratified 80/20 train/test split with `random_state=42` on joint stratification key `source_dataset + "_" + label`.
  - **Training Split ($N = 3,200$)**: 1,600 label 0, 1,600 label 1 (HaluEval: 1,200, TruthfulQA: 1,200, FEVER: 800).
  - **Holdout Test Split ($N = 800$)**: 400 label 0, 400 label 1 (HaluEval: 300, TruthfulQA: 300, FEVER: 200).
- **Primary Model Architectures & Configurations**:
  - *Logistic Regression*: `Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42))])`. Scaler fitted strictly on the 3,200 training instances.
  - *XGBoost*: `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=0.1, eval_metric='logloss', random_state=42, n_jobs=-1)`.
- **Primary Holdout Performance Results ($N = 800$)**:
  - *Logistic Regression*: Accuracy = `0.6288`, Precision = `0.6241`, Recall = `0.6475`, F1 = `0.6356`, ROC-AUC = `0.7079`, PR-AUC = `0.7124`, Brier = `0.2151`, ECE = `0.0525`.
  - *XGBoost*: Accuracy = `0.6788`, Precision = `0.6829`, Recall = `0.6675`, F1 = `0.6751`, ROC-AUC = `0.7688`, PR-AUC = `0.7795`, Brier = `0.1891`, ECE = `0.0341`.

---

## 4. Dataset Inventory & Provenance

| Dataset | Total Samples | Label 0 (Faithful) | Label 1 (Hallucinated) | Source Tasks | Context Payload Type | Retrieval Feasibility | Primary Role |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **HaluEval** | 1,500 | 750 | 750 | QA / Dialogue / Summarization | Genuine reference text passages (Wikipedia paragraphs) | **Fully Supported** (Local corpus: 1,499 unique passages) | Core benchmark & retrieval evaluation |
| **TruthfulQA** | 1,500 | 750 | 750 | Imitative falsehoods / misconceptions | External reference URLs and category strings | **Structurally Unavailable** (No local text; web search prohibited) | Core benchmark; excluded from retrieval variant |
| **FEVER** | 1,000 | 500 | 500 | Fact verification against Wikipedia claims | Wikipedia pointer annotations (`[[page, line]]`) | **Partially Supported** (Local corpus: 829 clean entity titles) | Core benchmark & retrieval evaluation |
| **Combined** | 4,000 | 2,000 | 2,000 | Multi-domain benchmark blend | Heterogeneous | 2,500 usable for local retrieval | Universal Core Detector foundation |

---

## 5. Model Inventory

| Model Identifier | Type | Input Dimension | Preprocessing | Hyperparameters | Execution Environment |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Qwen3.5-0.8B** | Base Causal LM | N/A | Chat Template / Prompt Formatting | Greedy generation for baseline; Temperature 0.7 ($k=5$) for self-consistency | CUDA (`cuda:0`, bfloat16/float16) |
| **cross-encoder/nli-MiniLM2-L6-H768** | Cross-Encoder NLI | Sequence pairs | HuggingFace Tokenizer | Max length 512, bidirectional pairs, softmax normalization | CUDA (`cuda:0`, PyTorch) |
| **sentence-transformers/all-MiniLM-L6-v2**| Dense Bi-Encoder | Text chunks | Unit L2 normalization | Embedding dimension 384, cosine similarity ranking | CUDA (`cuda:0`) |
| **Logistic Regression (Baseline/Core)** | Supervised Classifier | 11, 14, 16, 17, 19, 25 | `StandardScaler` (train split only) | $C=1.0$, `solver='lbfgs'`, `max_iter=1000`, `random_state=42` | CPU (scikit-learn 1.9.1) |
| **XGBoost (Baseline/Core)** | Gradient Boosted Trees | 11, 14, 16, 17, 19, 25 | None (tree-based invariance) | `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0.1`, `random_state=42` | CPU multi-threaded (xgboost 3.1.1) |
| **Platt Calibrator** | Sigmoid Calibrator | 1 (Probabilities) | Out-of-fold calibration partition probability input | `_SigmoidCalibration()` fit on 640 calibration instances | CPU (scikit-learn) |
| **Isotonic Calibrator** | Non-parametric Isotonic Regression | 1 (Probabilities) | Out-of-fold calibration partition probability input | `IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)` fit on 640 instances | CPU (scikit-learn) |

---

## 6. Metric Inventory Across Experimental Areas

| Experiment Area | Accuracy | F1 Score | ROC-AUC | PR-AUC | Brier Score | ECE | MCE | Cohen's d | Cox Slope / Intercept | Deltas Reported |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 6: Baselines** | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | Deltas vs num_tokens ablation |
| **Phase 14: Universal Core** | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | Deltas vs Phase 6 Baseline ($\Delta_{\text{vs base}}$) |
| **Phase 15: Ablation** | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | Deltas vs Universal Core ($\Delta_{\text{vs all\_three}}$) |
| **Phase 16: Cross-Dataset** | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | Deltas vs Internal-Only within experiment |
| **Phase 17: Calibration** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | Deltas vs Uncalibrated ($\Delta_{\text{calib} - \text{uncalib}}$) |
| **Phase 18: Error Analysis** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | Cohen's d across all 19 features |
| **Phase 19: Retrieval Variant** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Deltas vs Internal-Only and Universal Core |

---

## 7. Comparability Notes

To preserve scientific rigor, the following comparability distinctions must be strictly observed in all reporting:

1. **Phase 14 vs. Phase 15 (Directly Comparable)**:
   - Evaluated on the exact same 800 holdout instances from `universal_features.parquet`.
   - Condition `all_three` in Phase 15 is mathematically identical to the Universal Core Detector in Phase 14.
2. **Phase 14 vs. Phase 6 (Directly Comparable Baseline Comparison)**:
   - Phase 14 and Phase 6 share the identical 3,200 train / 800 holdout split (`random_state=42`).
   - Phase 14 adds self-consistency (5) and NLI (3) features to the 11 baseline internal signals.
3. **Phase 14 vs. Phase 16 (NOT Directly Comparable)**:
   - Phase 16 uses a Leave-One-Dataset-Out (LODO) paradigm where models are trained on 100% of two datasets ($2,500$ or $3,000$ instances) and tested on 100% of the third ($1,000$ or $1,500$ instances). The test sets and sample distributions differ fundamentally from the 800-instance mixed holdout.
4. **Phase 14 vs. Phase 17 (Methodologically Distinct Calibration Setup)**:
   - Phase 17 evaluates on the same 800 holdout instances, but its base models were fitted on an inner subset of $2,560$ instances ($80\%$ of the $3,200$ training set) to reserve $640$ instances ($20\%$) for leak-free calibration training. Uncalibrated metrics in Phase 17 reflect a 2,560-sample base model and should not be confused with the 3,200-sample Phase 14 models.
5. **Phase 14 vs. Phase 19 (NOT Directly Comparable)**:
   - Phase 19 is restricted to the $2,500$ instances where local evidence retrieval was structurally feasible (HaluEval + FEVER). Its holdout test set contains $500$ instances ($250/250$), completely omitting TruthfulQA. Comparisons in Phase 19 must be made strictly among the 5 conditions evaluated on those same 500 instances.

---

## 8. Methodological Caveats & Limitations

1. **Local Context vs. Open-Web Retrieval**:
   - In Phase 19, HaluEval evidence is derived from benchmark-provided reference context, and FEVER evidence is derived from annotated Wikipedia article titles. This represents groundedness against available benchmark context rather than open-domain web fact-checking.
2. **TruthfulQA Retrieval Incompatibility**:
   - Because TruthfulQA lacks a local evidence passage corpus, retrieval features cannot be computed without external web queries (which were prohibited) or text fabrication. TruthfulQA is excluded from the retrieval variant to avoid dataset-identifying missingness indicators.
3. **Token Length Association**:
   - In Phase 18 error analysis, False Positives exhibited higher average word counts ($8.7 - 9.7$ words) than True Negatives ($4.1 - 5.1$ words). This is an empirical descriptive association, not a causal factor; sequence length was intentionally excluded as an input feature.
4. **Confidence Interpretation**:
   - Higher confidence indicates that output probabilities are farther from $0.50$, not that the model possesses internal knowledge of factual truth.
5. **Zero Remote Leakage**:
   - The repository remains entirely local; no commits have been pushed to remote.

---

## 9. Missing or Unfinished Items

- **No missing artifacts**: All 11 experiment areas have generated complete CSV, Parquet, JSON, and Markdown summaries.
- **No missing baseline runs**: All pilot and production runs are accounted for.
- **Next Step**: Proceed to Phase 21 (Final Technical Report & Research Synthesis) and Phase 22 (Streamlit Interactive Demonstration Application).

---

## 10. Recommended Source Files for the Final Technical Report

When compiling the final project report, cite the following authoritative artifact files:

1. **Executive Summary & Primary Benchmark**:
   - [`experiments/baselines/combined/combined_model_results.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/combined_model_results.json)
   - [`experiments/baselines/combined/combined_model_results.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/baselines/combined/combined_model_results.csv)
2. **Feature-Family Ablation Study**:
   - [`experiments/ablation/ablation_summary.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/ablation/ablation_summary.md)
   - [`experiments/ablation/ablation_results.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/ablation/ablation_results.csv)
3. **Cross-Dataset Transfer Evaluation**:
   - [`experiments/cross_dataset/cross_dataset_summary.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/cross_dataset/cross_dataset_summary.md)
   - [`experiments/cross_dataset/cross_dataset_results.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/cross_dataset/cross_dataset_results.csv)
4. **Probability Calibration Analysis**:
   - [`experiments/calibration/calibration_summary.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/calibration_summary.md)
   - [`experiments/calibration/calibration_results.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/calibration_results.csv)
   - Reliability Plots: [`experiments/calibration/logistic_regression_reliability.png`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/logistic_regression_reliability.png), [`experiments/calibration/xgboost_reliability.png`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/calibration/xgboost_reliability.png)
5. **Detailed Error Analysis**:
   - [`experiments/error_analysis/error_analysis_summary.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/error_analysis/error_analysis_summary.md)
   - [`experiments/error_analysis/error_category_statistics.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/error_analysis/error_category_statistics.csv)
   - [`experiments/error_analysis/feature_error_comparison.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/error_analysis/feature_error_comparison.csv)
   - [`experiments/error_analysis/model_disagreement.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/error_analysis/model_disagreement.csv)
6. **Retrieval-Augmented Variant**:
   - [`experiments/retrieval_augmented/retrieval_summary.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/retrieval_augmented/retrieval_summary.md)
   - [`experiments/retrieval_augmented/retrieval_experiment_results.csv`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/retrieval_augmented/retrieval_experiment_results.csv)
   - [`experiments/retrieval_augmented/retrieval_corpus_manifest.json`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/experiments/retrieval_augmented/retrieval_corpus_manifest.json)
