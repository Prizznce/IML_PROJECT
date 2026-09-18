# Retrieval-Augmented Hallucination Detection Variant — Summary

## 1. Experimental Setup & Evidence Corpora
- **Usable Dataset**: $N = 2,500$ instances (HaluEval: 1,500, FEVER: 1,000)
- **Holdout Test Set**: $N = 500$ instances (HaluEval: 300, FEVER: 200; 250 label 0, 250 label 1)
- **HaluEval Evidence Corpus**: 1499 benchmark reference passages
- **FEVER Evidence Corpus**: 829 clean Wikipedia article titles from context pointers
- **TruthfulQA Exclusion**: Documented benchmark limitation; context contains URLs rather than local text
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Dense cosine retrieval, top-k=3)

## 2. Model Performance Across 5 Feature Conditions ($N = 500$ Holdout Set)

| Condition | Feats | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier | ECE | Δ(vs Int) ROC | Δ(vs Univ) ROC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Condition A (Internal-Only) | 11 | logistic_regression | 0.7460 | 0.7581 | 0.8463 | 0.8486 | 0.1597 | 0.0494 | +0.0000 | -0.0119 |
| Condition A (Internal-Only) | 11 | xgboost | 0.7880 | 0.8000 | 0.8978 | 0.8982 | 0.1264 | 0.0323 | +0.0000 | -0.0033 |
| Condition B (Universal Core) | 19 | logistic_regression | 0.7580 | 0.7660 | 0.8582 | 0.8657 | 0.1538 | 0.0738 | +0.0119 | +0.0000 |
| Condition B (Universal Core) | 19 | xgboost | 0.7860 | 0.7864 | 0.9011 | 0.9035 | 0.1246 | 0.0482 | +0.0033 | +0.0000 |
| Condition C (Retrieval-Only) | 6 | logistic_regression | 0.6500 | 0.6824 | 0.6990 | 0.6543 | 0.2187 | 0.0350 | -0.1473 | -0.1592 |
| Condition C (Retrieval-Only) | 6 | xgboost | 0.6800 | 0.7193 | 0.7491 | 0.7293 | 0.2053 | 0.0523 | -0.1487 | -0.1520 |
| Condition D (Internal + Retrieval) | 17 | logistic_regression | 0.7660 | 0.7754 | 0.8637 | 0.8675 | 0.1505 | 0.0380 | +0.0174 | +0.0055 |
| Condition D (Internal + Retrieval) | 17 | xgboost | 0.8060 | 0.8159 | 0.9184 | 0.9214 | 0.1180 | 0.0593 | +0.0206 | +0.0173 |
| Condition E (Universal + Retrieval) | 25 | logistic_regression | 0.7620 | 0.7680 | 0.8706 | 0.8823 | 0.1459 | 0.0606 | +0.0243 | +0.0124 |
| Condition E (Universal + Retrieval) | 25 | xgboost | 0.8120 | 0.8178 | 0.9169 | 0.9201 | 0.1185 | 0.0518 | +0.0191 | +0.0159 |

## 3. Methodological Observations
- **Fairness Rule**: All 5 conditions evaluated on the exact same 500 holdout instances.
- **Zero Leakage**: No labels, target responses, or generated responses in retrieval corpora.
- **Standalone Variant**: Retrieval signals are evaluated as a separate variant and are not incorporated into the Universal Core Detector.
