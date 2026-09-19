# Phase 20 — Step 5: Final Technical Consistency Audit of Academic Report

**Target Document**: [`docs/final_project_report.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/final_project_report.md)  
**Audit Date**: September 18, 2026  
**Auditor**: Antigravity Automated Verification Engine  
**Audit Scope**: Strict factual consistency, mathematical integrity, code-level verification, environment verification, scientific tone, and citation compliance.

---

## 1. Executive Summary

A comprehensive, line-by-line technical consistency audit was conducted on the academic project report draft ([`docs/final_project_report.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/final_project_report.md)). The report was verified against:
1. Ground-truth source code modules in `src/`.
2. Authoritative serialized experiment artifacts and parquet tables in `experiments/` and `data/processed/`.
3. The consolidated master results table in [`docs/master_results_table.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/master_results_table.md).
4. Repository environment configurations in [`docs/environment.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/environment.md), [`README.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/README.md), and [`requirements.txt`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/requirements.txt).
5. The local Python virtual environment runtime (`.venv`).

### Overall Audit Verdict: **PASSED WITH MINOR FACTUAL CORRECTIONS**
- **Experiment Re-runs**: **0** (No models retrained, no experiments rerun).
- **Metric Alterations**: **0** (Every numerical metric in Tables A–F strictly matches primary experiment outputs).
- **Factual Corrections Applied**: Revised Section 24 to accurately differentiate the repository's documented CPU baseline environment from the current local CUDA verification runtime; clarified explicit total unordered NLI pair counts ($40,000$ pairs); explicitly enumerated excluded non-signal columns under data leakage safeguards; updated unit test suite count to 198 passing tests.

---

## 2. Item-by-Item Verification & Audit Results

### 2.1 Model Identity
- **Check**: Verify every mention of the generation model against actual source code:
  - [`src/signals/score_labeled_responses.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/signals/score_labeled_responses.py#L421): `--model-id default="Qwen/Qwen3.5-0.8B"`
  - [`src/generation/generate_signals.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/generation/generate_signals.py#L458): `--model-id default="Qwen/Qwen3.5-0.8B"`
  - [`src/consistency/self_consistency.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/consistency/self_consistency.py#L6): `Qwen/Qwen3.5-0.8B`
  - [`docs/generation_pipeline.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/generation_pipeline.md#L4): `Qwen/Qwen3.5-0.8B`
- **Report Status**: Every mention in `docs/final_project_report.md` designates `Qwen/Qwen3.5-0.8B` (or `Qwen3.5-0.8B`). Zero mentions of outdated or alternative identifiers (e.g., `Qwen2.5-0.5B` or Llama) exist.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.2 Python and Library Environment
- **Check**: Verify environment specifications against `requirements.txt`, `docs/environment.md`, and the active virtual environment (`.venv`).
- **Audit Findings**:
  - `requirements.txt` specifies unpinned dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `transformers`, `sentence-transformers`, `torch`, etc.).
  - `docs/environment.md` and `README.md` document the repository's historical CPU baseline: Microsoft Windows 11 Home Single Language (64-bit, Version 10.0.26200), CPU execution mode (Intel Iris Xe Graphics, CUDA unavailable), Python `3.13.14`, PyTorch `2.14.0+cpu`, Scikit-Learn `1.9.1`, XGBoost `3.4.1`, Transformers `5.17.0`, Sentence-Transformers `6.0.1`, Datasets `5.0.1`, FAISS (`faiss-cpu`) `1.15.1`.
  - The current audit-time active virtual environment (`.venv`) runs: Microsoft Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, CUDA 12.4, driver 576.06), Python `3.11.9`, PyTorch `2.6.0+cu124`, Scikit-Learn `1.9.1`, XGBoost `3.2.0`, Transformers `5.17.0`, Sentence-Transformers `6.0.1`.
  - **Issue Identified**: The previous report draft listed assumed placeholder versions (`Scikit-Learn 1.6.1`, `XGBoost 3.1.1`, `Transformers 4.49.0`, `Sentence-Transformers 3.4.1`).
  - **Correction Made**: Section 24 was rewritten to explicitly distinguish between (A) the documented historical CPU baseline environment in `docs/environment.md` and (B) the verified active audit-time CUDA execution runtime in `.venv`.
- **Verdict**: **CORRECTED & VERIFIED**.

### 2.3 Dataset Counts & Composition
- **Check**: Verify benchmark sample sizes, class balance, and handling of unverified labels.
  - HaluEval: $1,500$ instances ($750$ faithful / $750$ hallucinated).
  - TruthfulQA: $1,500$ instances ($750$ truthful / $750$ untruthful).
  - FEVER: $1,000$ instances ($500$ `SUPPORTS` / $500$ `REFUTES`).
  - Combined Corpus: $4,000$ instances ($2,000$ label $0$ / $2,000$ label $1$, exact 50/50 balance).
  - Label Filtering: Claims labeled `NOT ENOUGH INFO` in FEVER were excluded to maintain an unambiguous binary ground truth.
- **Report Status**: Sections 2, 7, and Table in Section 7 match these numbers exactly.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.4 Primary Modeling Split (Phase 14)
- **Check**: Verify training and test split specifications.
  - Train Set: $3,200$ examples ($80.0\%$).
  - Holdout Test Set: $800$ examples ($20.0\%$).
  - Partitioning Seed: `random_state=42`.
  - Stratification Key: Joint key `source_dataset + "_" + label` to preserve exact class and domain ratios.
- **Report Status**: Sections 2, 8, 13, and 15 strictly adhere to this protocol.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.5 Universal Core Feature Matrix
- **Check**: Verify exactly 19 primary features and ensure strict exclusion of non-signal/metadata fields.
  - 11 Internal Signals: `min_log_prob`, `mean_log_prob`, `mean_token_prob`, `token_prob_std`, `mean_entropy`, `max_entropy`, `entropy_std`, `log_perplexity`, `log_mean_token_rank`, `log_max_token_rank`, `log_rank_std`.
  - 5 Self-Consistency Signals: `exact_match_agreement`, `mean_pairwise_similarity`, `min_pairwise_similarity`, `max_pairwise_similarity`, `pairwise_similarity_std`.
  - 3 NLI Signals: `mean_pairwise_entailment`, `mean_pairwise_contradiction`, `nli_disagreement`.
  - Excluded Features: `source_dataset`, `id`, `prompt`, `response`, `context`, `model_input`, `forward_time`, `num_tokens`, and all retrieval features.
- **Report Status**: Confirmed in Section 8, Section 13 (Data Leakage Safeguards), and Section 26.1 (Appendix Feature Table).
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.6 Self-Consistency Generation Protocol
- **Check**: Verify sampling parameters, generation counts, and conceptual probe distinction.
  - Number of stochastic samples: $k = 5$.
  - Sampling hyperparameters: Temperature $T = 0.7$, top-$p = 0.9$, `max_new_tokens = 128`.
  - Total generated candidates: $4,000 \times 5 = 20,000$ raw outputs.
  - Conceptual Role: The report explicitly differentiates primary evaluated responses (original benchmark responses whose veracity is predicted) from behavioral probes ($k=5$ stochastic outputs generated solely to measure neighborhood stability).
- **Report Status**: Sections 2, 8.1, 10, and 26.3 state these exact parameters and distinctions.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.7 Natural Language Inference (NLI) Protocol
- **Check**: Verify candidate pair counts, directional passes, and model identity.
  - Evaluated candidates per prompt: $5$.
  - Unordered pairs per example: $\binom{5}{2} = 10$.
  - Total unordered pairs across corpus: $4,000 \times 10 = 40,000$ pairs.
  - Directional evaluations per example: $10 \times 2 = 20$ directed pairs.
  - Total directional forward passes: $4,000 \times 20 = 80,000$ passes.
  - Cross-encoder Model: `cross-encoder/nli-MiniLM2-L6-H768`.
- **Report Status**: Section 11 specifies these exact parameters and equations.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.8 Retrieval-Augmented Standalone Variant (Phase 19)
- **Check**: Verify subset scope, holdout split, retrieval engine, and absence of open-web claims.
  - Sample Population: Strictly $N = 2,500$ instances (HaluEval $1,500$ + FEVER $1,000$).
  - TruthfulQA Exclusion: Excluded ($1,500$ instances) because its metadata contains URL links rather than text passages, and live open-web requests were prohibited.
  - Evaluation Split: Dedicated $2,000$ train / $500$ holdout test partition (balanced $250/250$).
  - Retrieval Scope: Dense semantic retrieval with `sentence-transformers/all-MiniLM-L6-v2` over $2,328$ local passages ($1,499$ HaluEval, $829$ FEVER), extracting top-$k = 3$ passages.
  - Number of Retrieval Features: Exactly 6 features.
  - Model Status: Standalone experimental variant; not part of the 19-feature Universal Core Detector.
- **Report Status**: Stated in Note Box in Section 12, Table E in Section 15, Section 20.3, and Section 26.2.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.9 Numerical Results Cross-Check
- **Check**: Audit every numerical value in Tables A–F against [`docs/master_results_table.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/master_results_table.md).
  - Table A (Phase 6 Baseline vs Phase 14 Universal Core):
    - LR Baseline: Acc `0.6212`, F1 `0.6363`, ROC-AUC `0.6930`, PR-AUC `0.6618`, Brier `0.2177`, ECE `0.0649`.
    - XGB Baseline: Acc `0.6700`, F1 `0.6887`, ROC-AUC `0.7411`, PR-AUC `0.7322`, Brier `0.1999`, ECE `0.0228`.
    - LR Universal Core: Acc `0.6288`, F1 `0.6356`, ROC-AUC `0.7079`, PR-AUC `0.7124`, Brier `0.2151`, ECE `0.0525`.
    - XGB Universal Core: Acc `0.6788`, F1 `0.6751`, ROC-AUC `0.7688`, PR-AUC `0.7795`, Brier `0.1891`, ECE `0.0341`.
  - Table B (Phase 15 Ablation): All 14 model-configuration entries match `master_results_table.md` to 4 decimal places.
  - Table C (Phase 16 LODO Cross-Dataset Generalization): All 24 cross-dataset evaluations match `master_results_table.md`.
  - Table D (Phase 17 Calibration): All 6 entries (Uncalibrated, Platt, Isotonic) match `master_results_table.md`.
  - Table E (Phase 19 Retrieval Variant): All 10 entries across 5 conditions match `master_results_table.md`.
  - Table F (Phase 14 Dataset-Level Breakdown): All 6 entries match `master_results_table.md`.
- **Verdict**: **100% NUMERICAL MATCH / VERIFIED**.

### 2.10 Calibration Methodology Distinction (Phase 17)
- **Check**: Verify that Phase 17 nested partition is clearly differentiated from Phase 14 primary benchmark.
  - Phase 17 Partition: $2,560$ base-training, $640$ calibration, $800$ test.
  - Non-conflation: The report explicitly explains that Phase 17 uncalibrated numbers (XGBoost Acc `0.6488`) are trained on $2,560$ examples and must not be conflated with the Phase 14 primary benchmark (`0.6788`).
- **Report Status**: Section 18 ("Methodological Distinction") explicitly clarifies this structure.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.11 Error Analysis & Scientific Non-Causal Phrasing (Phase 18)
- **Check**: Verify test error counts, word length profiling, probability dynamics, and absence of causal claims.
  - LR errors: $297/800$ ($37.12\%$), FP: $156$, FN: $141$.
  - XGB errors: $257/800$ ($32.12\%$), FP: $124$, FN: $133$.
  - True Negatives avg word count: `4.06` words.
  - True Positives avg word count: `10.75` (LR), `10.96` (XGB).
  - FP predicted prob: `0.6197` (LR), `0.6074` (XGB); FN predicted prob: `0.3795` (LR), `0.3806` (XGB).
  - Cohen's $d$: $d = 1.18$ (`min_log_prob`), $d = -1.02$ (`log_max_token_rank`).
  - Scientific Wording: Section 16.3 and Section 17.4 explicitly state that these relationships are descriptive correlations of the test partition and do not imply causal mechanisms.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.12 References & Bibliography
- **Check**: Search repository for bibtex or bibliography files. Ensure no fabricated citations.
  - Repository Search: No `.bib` or external citation database files exist in the repository.
  - Report Handling: Section 6 and Section 25 use explicit `[REFERENCE NEEDED]` placeholders accompanied by the mandatory disclaimer:  
    `Bibliographic references to be added from the project's verified literature sources.`
  - No synthetic author names, paper titles, or DOIs were fabricated.
- **Verdict**: **VERIFIED / COMPLIANT**.

### 2.13 Team Information
- **Check**: Verify team member names against repository documentation.
  - Author List:
    - Kunj Patil
    - Jaiprakash
    - Prince
    - Aditya Kumar
  - No unauthorized affiliations, institutional titles, or student IDs were added.
- **Verdict**: **VERIFIED / CONSISTENT**.

### 2.14 Scientific Tone and Real-Time Qualification
- **Check**: Audit for prohibited competitive or superlative language ("best model", "winner", "superior", "optimal", "proves", "guarantees", "production-ready").
  - Prohibited competitive terms: **0 found** across the entire report.
  - Real-Time Qualification: While the official project title retains *"Real-Time Hallucination Detection"*, the Abstract (Section 2) and Introduction (Section 3) explicitly state:  
    `Although designed for potential real-time integration, all evaluations in this study were conducted offline in batch mode across established benchmarks.`
- **Verdict**: **VERIFIED / COMPLIANT**.

---

## 3. Detailed Summary of Corrections Made

The following factual corrections were applied directly to [`docs/final_project_report.md`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/docs/final_project_report.md):

1. **Section 11 (NLI Agreement)**:
   - Added explicit clause: `($40,000$ unordered pairs across the $4,000$-example corpus)` to clarify the total number of unordered pairs evaluated alongside the 80,000 directional evaluations.
2. **Section 13 (Machine Learning Models — Data Leakage Safeguards)**:
   - Rephrased `"To guarantee complete separation"` to `"To maintain strict statistical separation"`.
   - Explicitly listed excluded non-signal columns: `id`, `prompt`, `response`, `context`, `model_input`, `source_dataset`, `forward_time`, and `num_tokens`.
3. **Section 24 (Reproducibility)**:
   - Replaced unverified placeholder versions (`Scikit-Learn 1.6.1, XGBoost 3.1.1, Transformers 4.49.0`) with the verified dual-environment documentation:
     - Documented historical CPU baseline from `docs/environment.md` (Python 3.13.14, torch 2.14.0+cpu, scikit-learn 1.9.1, xgboost 3.4.1).
     - Active verification runtime in `.venv` (Python 3.11.9, torch 2.6.0+cu124, scikit-learn 1.9.1, xgboost 3.2.0, CUDA 12.4).
     - Clarified that `requirements.txt` specifies unpinned dependencies for portable cross-environment deployment.
4. **Section 26.5 (Test Suite Status)**:
   - Updated passing test count to 198 to include all documentation and consistency unit tests (`tests/test_final_project_report.py`).

---

## 4. Environment Verification Summary

| Component | Documented Baseline (`docs/environment.md`) | Current Active Runtime (`.venv`) | Verification Source |
| :--- | :--- | :--- | :--- |
| **OS** | Windows 11 Home Single Language (64-bit) | Windows 11 Home Single Language (64-bit) | System / `docs/environment.md` |
| **Python** | `3.13.14` | `3.11.9` | `sys.version` in `.venv` |
| **PyTorch** | `2.14.0+cpu` | `2.6.0+cu124` | `torch.__version__` |
| **CUDA** | Unavailable (CPU mode) | CUDA 12.4 (RTX 3050 Laptop GPU, 6GB) | `torch.cuda.is_available()` |
| **Scikit-Learn** | `1.9.1` | `1.9.1` | `sklearn.__version__` |
| **XGBoost** | `3.4.1` | `3.2.0` | `xgboost.__version__` |
| **Transformers** | `5.17.0` | `5.17.0` | `transformers.__version__` |
| **Sentence-Transformers** | `6.0.1` | `6.0.1` | `sentence_transformers.__version__` |

---

## 5. Certification of Compliance

- [x] **Zero Metric Alteration**: All metric values remain identical to verified experiment outputs.
- [x] **Zero Experiment Re-execution**: No training scripts, scoring pipelines, or sampling routines were triggered.
- [x] **Zero Citation Fabrication**: Placeholder tags `[REFERENCE NEEDED]` and disclaimers were maintained.
- [x] **No Git Commits or Pushes**: Local working tree preserved without executing `git commit` or `git push`.
