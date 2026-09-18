# Comprehensive Probability Calibration Analysis Summary (Phase 17)

## 1. Experimental Split Protocol

- **Total Dataset**: 4,000 samples
- **Holdout Test Set (Untouched)**: Exactly 800 samples (same 800 IDs as Phases 14-16)
- **Total Training Partition**: 3,200 samples
  - **Model-Fitting Subset**: 2,560 samples (80% of training data)
  - **Calibration-Fitting Subset**: 640 samples (20% of training data)
- **Zero Leakage**: Base models and scalers were fit on the model subset; calibrators were fit on the calibration subset; test set was strictly evaluated.

---

## 2. Overall Model Performance Across Calibration Methods

$$\Delta = \text{Calibrated Metric} - \text{Uncalibrated Metric}$$
*(For Brier, ECE, and MCE, negative delta indicates error reduction)*

| Model | Calibration | Accuracy | F1 | ROC-AUC | PR-AUC | Brier ($\Delta$) | ECE ($\Delta$) | MCE ($\Delta$) | Slope | Intercept |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_regression` | **uncalibrated** | 0.6338 | 0.6387 | 0.7069 | 0.7096 | 0.2155 (+0.0000) | 0.0444 (+0.0000) | 0.1248 (+0.0000) | 0.938 | -0.009 |
| `logistic_regression` | **sigmoid** | 0.6300 | 0.6373 | 0.7069 | 0.7096 | 0.2165 (+0.0010) | 0.0547 (+0.0102) | 0.1114 (-0.0133) | 0.957 | -0.022 |
| `logistic_regression` | **isotonic** | 0.6225 | 0.6505 | 0.6971 | 0.6751 | 0.2151 (-0.0003) | 0.0341 (-0.0104) | 0.2473 (+0.1226) | 0.540 | -0.044 |
| `xgboost` | **uncalibrated** | 0.6488 | 0.6492 | 0.7638 | 0.7787 | 0.1900 (+0.0000) | 0.0712 (+0.0000) | 0.1888 (+0.0000) | 1.042 | -0.012 |
| `xgboost` | **sigmoid** | 0.6550 | 0.6452 | 0.7638 | 0.7787 | 0.1935 (+0.0034) | 0.0825 (+0.0113) | 0.1210 (-0.0678) | 0.953 | 0.041 |
| `xgboost` | **isotonic** | 0.6675 | 0.6472 | 0.7641 | 0.7557 | 0.1890 (-0.0010) | 0.0244 (-0.0468) | 0.1256 (-0.0631) | 0.796 | 0.027 |

---

## 3. Subgroup Calibration Breakdown on Holdout Test Set

| Model | Calibration | Dataset | N | Brier | ECE | MCE | Accuracy | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_regression` | uncalibrated | **fever** | 200 | 0.2715 | 0.1245 | 0.8207 | 0.4850 | 0.4850 |
| `logistic_regression` | uncalibrated | **halueval** | 300 | 0.1235 | 0.1280 | 0.2106 | 0.8433 | 0.9278 |
| `logistic_regression` | uncalibrated | **truthfulqa** | 300 | 0.2700 | 0.1121 | 0.6225 | 0.5233 | 0.5232 |
| `logistic_regression` | sigmoid | **fever** | 200 | 0.2731 | 0.1361 | 0.8106 | 0.4800 | 0.4850 |
| `logistic_regression` | sigmoid | **halueval** | 300 | 0.1252 | 0.1437 | 0.2082 | 0.8400 | 0.9278 |
| `logistic_regression` | sigmoid | **truthfulqa** | 300 | 0.2700 | 0.1190 | 0.8275 | 0.5200 | 0.5232 |
| `logistic_regression` | isotonic | **fever** | 200 | 0.2664 | 0.1080 | 0.9583 | 0.4650 | 0.4692 |
| `logistic_regression` | isotonic | **halueval** | 300 | 0.1251 | 0.0928 | 0.2789 | 0.8300 | 0.9194 |
| `logistic_regression` | isotonic | **truthfulqa** | 300 | 0.2710 | 0.0845 | 0.8372 | 0.5200 | 0.5228 |
| `xgboost` | uncalibrated | **fever** | 200 | 0.2483 | 0.0827 | 0.3881 | 0.5050 | 0.5460 |
| `xgboost` | uncalibrated | **halueval** | 300 | 0.0930 | 0.0858 | 0.2084 | 0.8633 | 0.9466 |
| `xgboost` | uncalibrated | **truthfulqa** | 300 | 0.2482 | 0.0978 | 0.3401 | 0.5300 | 0.5834 |
| `xgboost` | sigmoid | **fever** | 200 | 0.2525 | 0.0733 | 0.8014 | 0.5300 | 0.5460 |
| `xgboost` | sigmoid | **halueval** | 300 | 0.0931 | 0.0699 | 0.1541 | 0.8700 | 0.9466 |
| `xgboost` | sigmoid | **truthfulqa** | 300 | 0.2545 | 0.1312 | 0.2851 | 0.5233 | 0.5834 |
| `xgboost` | isotonic | **fever** | 200 | 0.2461 | 0.0263 | 0.0943 | 0.5500 | 0.5577 |
| `xgboost` | isotonic | **halueval** | 300 | 0.0941 | 0.0514 | 0.1810 | 0.8700 | 0.9442 |
| `xgboost` | isotonic | **truthfulqa** | 300 | 0.2460 | 0.0689 | 0.2500 | 0.5433 | 0.5825 |

---

## 4. Key Methodological Findings

- **Brier Score vs. Classification Metrics**: Post-hoc calibration specifically adjusts probability alignment without fundamentally changing binary decision boundaries or rank ordering (ROC-AUC remains nearly constant).
- **Platt Scaling (Sigmoid) Effect**: Provides smooth, parametric adjustment that prevents extreme overconfidence while preserving monotonic rankings.
- **Isotonic Regression Effect**: Non-parametric piecewise constant calibration allows flexible mapping for non-sigmoid distortion.
- **Non-Causality Disclaimer**: Calibration curves describe empirical confidence alignment and do not establish causal origins of hallucination generation.
