# Retrieval-Augmented Hallucination Detection Variant

This document details the design, methodology, and empirical evaluation of the **Retrieval-Augmented Hallucination Detection Variant**, evaluated as a standalone comparative experimental branch against the Universal Core Hallucination Detector.

---

## 1. Background & Research Motivation

A preliminary retrieval pilot evaluated the feasibility of dense evidence retrieval using `sentence-transformers/all-MiniLM-L6-v2`. That pilot demonstrated that attempting to query a HaluEval-derived evidence corpus on instances from FEVER or TruthfulQA produces severe cross-domain missingness. 

In Phase 19, we establish a clean, dataset-appropriate evidence retrieval methodology designed to avoid structural missingness and dataset-identifying leakage.

---

## 2. Evidence Corpus Construction & Retrieval Design

### Evidence Sources by Benchmark

1. **HaluEval ($N = 1,500$)**:
   - *Source*: The benchmark-provided `context` field containing full factual reference passages (derived from Wikipedia passages in HotpotQA).
   - *Corpus Size*: **1,499 unique text passages**.
   - *Retrieval Query*: The benchmark query/question (`prompt`).
2. **FEVER ($N = 1,000$)**:
   - *Source*: The benchmark-provided `context` field containing Wikipedia article pointer tuples (`[[[page_id, line_id, "Page_Title", line_number]]]`).
   - *Corpus Size*: **829 unique clean Wikipedia article titles** extracted and normalized from the pointer annotations. External Wikipedia full-text articles were not downloaded to comply with repository safety constraints.
   - *Retrieval Query*: The factual verification claim (`response`).
3. **TruthfulQA ($N = 1,500$) — Documented Benchmark Limitation**:
   - *Status*: **Excluded from the retrieval-augmented experiment**.
   - *Reason*: The `context` field in TruthfulQA contains external reference URLs and brief category annotations rather than local evidence passages. Performing web retrieval was prohibited, and local evidence could not be constructed without fabrication.
   - *Fairness Precaution*: Including TruthfulQA with zeroed or missing retrieval features would allow the classifier to infer dataset identity from retrieval availability. To prevent this confounder, TruthfulQA was excluded from the retrieval evaluation.

### Critical Methodological Distinction: Benchmark Context vs. External Verification
> [!IMPORTANT]
> The evidence corpora constructed here consist of **benchmark-provided reference context**. This setup evaluates retrieval groundedness against available benchmark knowledge, rather than open-domain external fact verification across an uncurated external web index.

---

## 3. Retrieval Feature Definitions ($k = 3$)

Dense retrieval was executed using `sentence-transformers/all-MiniLM-L6-v2` with unit-normalized embeddings and cosine similarity:

1. `top1_evidence_similarity`: Cosine similarity between query embedding and rank-1 retrieved evidence chunk.
2. `top3_evidence_similarity`: Mean cosine similarity between query embedding and top-3 retrieved evidence chunks.
3. `response_top1_similarity`: Cosine similarity between response embedding and rank-1 retrieved evidence chunk.
4. `response_top3_similarity`: Mean cosine similarity between response embedding and top-3 retrieved evidence chunks.
5. `evidence_response_margin`: Difference between rank-1 and rank-2 query-evidence similarity ($s_1 - s_2 \ge 0.0$).
6. `retrieval_agreement`: Dual groundedness metric bounded in $[0.0, 1.0]$:
   $$\text{retrieval\_agreement} = \max(0.0, s_{q,1}) \times \max(0.0, s_{r,1})$$

### Zero-Leakage Audit
- No ground-truth benchmark labels (`0` or `1`) were present in any retrieval corpus.
- No target responses or candidate generations were indexed in the evidence corpora.
- The ground-truth label was never used as an input to query formatting, retrieval, or feature extraction.

---

## 4. Experimental Dataset & Partitioning Protocol

- **Usable Evaluation Dataset**: $N = 2,500$ instances (HaluEval: 1,500, FEVER: 1,000).
- **Class Balance**: Exactly $1,250$ non-hallucinated instances (label 0) and $1,250$ hallucinated instances (label 1), representing a perfectly balanced 50.0% / 50.0% distribution.
- **Deterministic Stratified Partitioning**:
  - Stratification Key: `source_dataset + "_" + label` with `random_state=42`.
  - **Training Partition**: $N = 2,000$ instances ($1,000$ label 0, $1,000$ label 1; $1,200$ HaluEval, $800$ FEVER).
  - **Holdout Test Partition**: $N = 500$ instances ($250$ label 0, $250$ label 1; $300$ HaluEval, $200$ FEVER).
- **Identical Holdout Test Set**: All 5 conditions and both model families were trained on the identical 2,000 training instances and evaluated on the **exact same 500 test instances**.
- **Preprocessing Isolation**: `StandardScaler` for Logistic Regression was fitted strictly on the training partition.

---

## 5. Model Comparison Across 5 Feature Conditions ($N = 500$)

Deltas are computed relative to:
- Condition A (Internal-Only): $\Delta_{\text{vs int}} = \text{Metric} - \text{Metric}_{\text{Cond A}}$
- Condition B (Universal Core): $\Delta_{\text{vs univ}} = \text{Metric} - \text{Metric}_{\text{Cond B}}$

| Condition | Feats | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier | ECE | $\Delta$(vs Int) ROC | $\Delta$(vs Univ) ROC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A (Internal-Only)** | 11 | Logistic Regression | 0.7460 | 0.7581 | 0.8463 | 0.8486 | 0.1597 | 0.0494 | +0.0000 | -0.0119 |
| | 11 | XGBoost | 0.7880 | 0.8000 | 0.8978 | 0.8982 | 0.1264 | 0.0323 | +0.0000 | -0.0033 |
| **Condition B (Universal Core)** | 19 | Logistic Regression | 0.7580 | 0.7660 | 0.8582 | 0.8657 | 0.1538 | 0.0738 | +0.0119 | +0.0000 |
| | 19 | XGBoost | 0.7860 | 0.7864 | 0.9011 | 0.9035 | 0.1246 | 0.0482 | +0.0033 | +0.0000 |
| **Condition C (Retrieval-Only)** | 6 | Logistic Regression | 0.6500 | 0.6824 | 0.6990 | 0.6543 | 0.2187 | 0.0350 | -0.1473 | -0.1592 |
| | 6 | XGBoost | 0.6800 | 0.7193 | 0.7491 | 0.7293 | 0.2053 | 0.0523 | -0.1487 | -0.1520 |
| **Condition D (Internal + Retrieval)** | 17 | Logistic Regression | 0.7660 | 0.7754 | 0.8637 | 0.8675 | 0.1505 | 0.0380 | +0.0174 | +0.0055 |
| | 17 | XGBoost | 0.8060 | 0.8159 | 0.9184 | 0.9214 | 0.1180 | 0.0593 | +0.0206 | +0.0173 |
| **Condition E (Universal + Retrieval)** | 25 | Logistic Regression | 0.7620 | 0.7680 | 0.8706 | 0.8823 | 0.1459 | 0.0606 | +0.0243 | +0.0124 |
| | 25 | XGBoost | 0.8120 | 0.8178 | 0.9169 | 0.9201 | 0.1185 | 0.0518 | +0.0191 | +0.0159 |

---

## 6. Descriptive Retrieval Quality Analysis

Average retrieval metrics across the usable evaluation dataset:

| Dataset | Top-1 Q Sim | Top-3 Q Sim | Top-1 R Sim | Top-3 R Sim | Evidence Margin | Retrieval Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HaluEval ($N=1,500$)** | 0.6428 | 0.4859 | 0.4423 | 0.3090 | 0.2180 | 0.2918 |
| **FEVER ($N=1,000$)** | 0.6955 | 0.4911 | 0.6955 | 0.4911 | 0.2801 | 0.4951 |

### Descriptive Observations:
- In FEVER, the claim text serves as both query and response, resulting in identical query-evidence and response-evidence similarities ($0.6955$).
- In HaluEval, the question query and the model response are distinct texts; query similarity to the reference passage was $0.6428$, while response similarity to the passage was $0.4423$.
- High similarity alone does not establish factual correctness, as hallucinated responses often share topical vocabulary with reference passages.

---

## 7. Comparative Error Analysis: Condition E vs. Condition B

Evaluating predictions on the identical 500 test instances between **Universal Core** (Condition B) and **Universal Core + Retrieval** (Condition E):

### Logistic Regression
- **Prediction Agreement**: $452 / 500$ ($90.4\%$)
- **Both Correct**: $356$ instances ($71.2\%$)
- **Both Incorrect**: $96$ instances ($19.2\%$)
- **Condition B Correct / Condition E Incorrect**: $23$ instances ($4.6\%$)
- **Condition B Incorrect / Condition E Correct**: $25$ instances ($5.0\%$)
- **Net Accuracy Delta**: $+0.0040$ ($+2$ instances)

### XGBoost
- **Prediction Agreement**: $435 / 500$ ($87.0\%$)
- **Both Correct**: $367$ instances ($73.4\%$)
- **Both Incorrect**: $68$ instances ($13.6\%$)
- **Condition B Correct / Condition E Incorrect**: $26$ instances ($5.2\%$)
- **Condition B Incorrect / Condition E Correct**: $39$ instances ($7.8\%$)
- **Net Accuracy Delta**: $+0.0260$ ($+13$ instances)

---

## 8. Artifacts Generated

The following artifacts have been created in `experiments/retrieval_augmented/`:

- `retrieval_corpus_manifest.json`: Manifest of evidence corpora sizes and exclusion justifications.
- `retrieval_features.parquet`: Full feature matrix for the 2,500 usable instances (19 universal + 6 retrieval features).
- `retrieval_features.csv`: CSV format of the retrieval-augmented feature table.
- `retrieval_experiment_results.json`: Full metrics, deltas, and retrieval quality summaries.
- `retrieval_experiment_results.csv`: Tabular results across the 10 experimental runs.
- `retrieval_predictions.parquet`: Instance-level test predictions ($5,000$ rows: 500 test samples $\times$ 5 conditions $\times$ 2 models).
- `retrieval_summary.md`: Executive summary markdown.
- `retrieval_error_analysis.csv`: Detailed error analysis table comparing Condition E and Condition B.
