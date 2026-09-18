# Hallucination Detector Error Analysis Summary

## 1. Experimental Overview
- **Evaluated Test Samples**: 800 (Phase 14 Holdout Set)
- **Class Balance**: 400 Non-Hallucinated (0), 400 Hallucinated (1)
- **Dataset Distribution**: HaluEval (300), TruthfulQA (300), FEVER (200)
- **Models Analyzed**: Logistic Regression and XGBoost (Universal 19-Feature Detector)

## 2. Confusion Category Statistics

| Model | Category | Count | Total % | Label % | Mean Prob | Median Prob | Mean Resp Words |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | TP | 259 | 32.4% | 64.8% | 0.6842 | 0.6648 | 10.7 |
| Logistic Regression | TN | 244 | 30.5% | 61.0% | 0.3054 | 0.3102 | 4.1 |
| Logistic Regression | FP | 156 | 19.5% | 39.0% | 0.6197 | 0.6053 | 9.7 |
| Logistic Regression | FN | 141 | 17.6% | 35.2% | 0.3795 | 0.3988 | 6.0 |
| XGBoost | TP | 267 | 33.4% | 66.8% | 0.7250 | 0.6841 | 10.1 |
| XGBoost | TN | 276 | 34.5% | 69.0% | 0.2761 | 0.3465 | 5.1 |
| XGBoost | FP | 124 | 15.5% | 31.0% | 0.6133 | 0.5828 | 8.7 |
| XGBoost | FN | 133 | 16.6% | 33.2% | 0.4095 | 0.4222 | 6.9 |

## 3. Per-Dataset Error & Rate Breakdown

| Model | Dataset | Total | TP | FP | TN | FN | Error Rate | FP Rate | FN Rate | Accuracy | F1 | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | overall | 800 | 259 | 156 | 244 | 141 | 0.3713 | 0.3900 | 0.3525 | 0.6288 | 0.6356 | 0.7079 |
| Logistic Regression | fever | 200 | 60 | 63 | 37 | 40 | 0.5150 | 0.6300 | 0.4000 | 0.4850 | 0.5381 | 0.4870 |
| Logistic Regression | halueval | 300 | 121 | 19 | 131 | 29 | 0.1600 | 0.1267 | 0.1933 | 0.8400 | 0.8345 | 0.9311 |
| Logistic Regression | truthfulqa | 300 | 78 | 74 | 76 | 72 | 0.4867 | 0.4933 | 0.4800 | 0.5133 | 0.5166 | 0.5210 |
| XGBoost | overall | 800 | 267 | 124 | 276 | 133 | 0.3212 | 0.3100 | 0.3325 | 0.6787 | 0.6751 | 0.7688 |
| XGBoost | fever | 200 | 50 | 44 | 56 | 50 | 0.4700 | 0.4400 | 0.5000 | 0.5300 | 0.5155 | 0.5441 |
| XGBoost | halueval | 300 | 128 | 19 | 131 | 22 | 0.1367 | 0.1267 | 0.1467 | 0.8633 | 0.8620 | 0.9433 |
| XGBoost | truthfulqa | 300 | 89 | 61 | 89 | 61 | 0.4067 | 0.4067 | 0.4067 | 0.5933 | 0.5933 | 0.6066 |

## 4. Model Disagreement Summary

- **Agreement Count**: 580 / 800 (72.5%)
- **Disagreement Count**: 220 / 800 (27.5%)
- **Both Correct**: 413 (51.6%)
- **Both Incorrect**: 167 (20.9%)
- **LR Correct / XGB Incorrect**: 90 (11.2%)
- **LR Incorrect / XGB Correct**: 130 (16.2%)

### Top Disagreement Instances (Largest Absolute Probability Difference)

| ID | Dataset | Label | LR Prob | XGB Prob | |ΔProb| | Prompt Preview | Response Preview |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| truthfulqa_381_inc_4 | truthfulqa | 1 | 0.1051 | 0.5956 | 0.4905 | What are circumstances where you are req... | For a divorce... |
| halueval-3702 | halueval | 0 | 0.5932 | 0.1321 | 0.4611 | Which band is from the US, Pylon or Rive... | Rivermaya... |
| halueval-2882 | halueval | 0 | 0.3844 | 0.8141 | 0.4297 | Which Cracker album included  musical ac... | O' Cracker Where Art Thou?... |
| halueval-5060 | halueval | 0 | 0.5117 | 0.0820 | 0.4297 | In Norway, Pinnekjøtt, Stick Meat, is a ... | Scandinavia... |
| truthfulqa_303_corr_3 | truthfulqa | 0 | 0.1785 | 0.5957 | 0.4172 | Americans drink more beer than people in... | Italy and France... |
| halueval-9408 | halueval | 1 | 0.4680 | 0.8788 | 0.4108 | Which dog type is of the spitz type, Ari... | Ariegeois is not of the spitz type.... |
| halueval-6230 | halueval | 0 | 0.5585 | 0.1512 | 0.4073 | What type of chemist was the American wh... | organic chemist... |
| halueval-2748 | halueval | 0 | 0.7828 | 0.3765 | 0.4064 | Who directed the film for which Danny Mi... | Alan Parker,... |
| halueval-4632 | halueval | 0 | 0.6281 | 0.2244 | 0.4036 | Which song by Desiigner was featured on ... | Panda... |
| truthfulqa_439_corr_1 | truthfulqa | 0 | 0.3402 | 0.7367 | 0.3966 | Based in Chestnut Hill, a basketball tea... | Boston College Eagles... |
