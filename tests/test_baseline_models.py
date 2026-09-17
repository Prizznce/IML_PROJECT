"""
Unit tests for baseline classification modeling pipeline.

Tests feature extraction, forbidden column exclusion, log transformations,
joint stratification, absence of test-set data leakage in scaling,
pipeline construction, evaluation metrics, and ECE calculation on synthetic data.
Does NOT download external LLMs or require GPU hardware.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.baseline_models import (
    PRIMARY_SIGNAL_FEATURES,
    FORBIDDEN_COLUMNS,
    prepare_feature_dataframe,
    split_dataset,
    build_logistic_regression_pipeline,
    build_xgboost_model,
    compute_ece,
    evaluate_predictions,
    run_baseline_experiment,
)


class TestBaselineModels(unittest.TestCase):
    """Test suite for baseline modeling and evaluation components."""

    def setUp(self):
        # Create synthetic supervised dataframe matching canonical schema
        np.random.seed(42)
        n = 100
        sources = ["halueval"] * 40 + ["truthfulqa"] * 40 + ["fever"] * 20
        labels = ([0, 1] * 50)
        
        self.mock_df = pd.DataFrame({
            "id": [f"ex_{i}" for i in range(n)],
            "source_dataset": sources,
            "label": labels,
            "prompt": [f"Prompt {i}" for i in range(n)],
            "context": [f"Context {i}" for i in range(n)],
            "response": [f"Response {i}" for i in range(n)],
            "model_input": [f"Input {i}" for i in range(n)],
            "forward_time_s": np.random.uniform(0.05, 0.15, n),
            "num_tokens": np.random.randint(5, 30, n),
            "mean_token_prob": np.random.uniform(0.1, 0.9, n),
            "min_token_prob": np.random.uniform(0.0001, 0.5, n),
            "mean_log_prob": np.random.uniform(-5.0, -0.1, n),
            "min_log_prob": np.random.uniform(-15.0, -1.0, n),
            "mean_entropy": np.random.uniform(0.2, 3.0, n),
            "max_entropy": np.random.uniform(1.0, 6.0, n),
            "entropy_std": np.random.uniform(0.1, 1.5, n),
            "token_prob_std": np.random.uniform(0.05, 0.4, n),
            "perplexity": np.random.uniform(1.5, 500.0, n),
            "mean_token_rank": np.random.uniform(1.0, 50.0, n),
            "max_token_rank": np.random.uniform(10.0, 1000.0, n),
            "min_token_rank": np.ones(n, dtype=int),
            "rank_std": np.random.uniform(5.0, 200.0, n),
        })

    def test_feature_selection_and_forbidden_columns(self):
        """Verify exactly 11 features are selected and forbidden columns are excluded."""
        X, y, meta = prepare_feature_dataframe(self.mock_df, include_num_tokens=False)

        # Exact 11 features
        self.assertEqual(len(X.columns), 11)
        self.assertEqual(list(X.columns), PRIMARY_SIGNAL_FEATURES)

        # num_tokens MUST NOT be present
        self.assertNotIn("num_tokens", X.columns)

        # Ensure NO forbidden column exists in feature matrix
        for forbidden in FORBIDDEN_COLUMNS:
            self.assertNotIn(forbidden, X.columns)

        # Verify y contains correct labels
        self.assertEqual(len(y), len(self.mock_df))
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    def test_log_transformations(self):
        """Verify mathematical correctness of log transformations."""
        X, _, _ = prepare_feature_dataframe(self.mock_df)

        expected_log_ppl = np.log(self.mock_df["perplexity"])
        np.testing.assert_allclose(X["log_perplexity"].values, expected_log_ppl.values, rtol=1e-5)

        expected_log_mean_rank = np.log1p(self.mock_df["mean_token_rank"])
        np.testing.assert_allclose(X["log_mean_token_rank"].values, expected_log_mean_rank.values, rtol=1e-5)

        expected_log_max_rank = np.log1p(self.mock_df["max_token_rank"])
        np.testing.assert_allclose(X["log_max_token_rank"].values, expected_log_max_rank.values, rtol=1e-5)

        expected_log_rank_std = np.log1p(self.mock_df["rank_std"])
        np.testing.assert_allclose(X["log_rank_std"].values, expected_log_rank_std.values, rtol=1e-5)

        # Check no NaN or inf in transformed features
        self.assertFalse(X.isnull().any().any())
        self.assertFalse(np.isinf(X.values).any())

    def test_train_test_split_and_stratification(self):
        """Verify 80/20 split and joint source+label stratification."""
        X, y, meta = prepare_feature_dataframe(self.mock_df)
        X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
            X, y, meta, test_size=0.2, random_state=42
        )

        # Sizes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

        # Stratification: check class balance
        self.assertEqual((y_train == 0).sum(), 40)
        self.assertEqual((y_train == 1).sum(), 40)
        self.assertEqual((y_test == 0).sum(), 10)
        self.assertEqual((y_test == 1).sum(), 10)

        # Stratification: check source_dataset distribution
        # Halueval: 40 total -> 32 train, 8 test
        self.assertEqual((meta_train["source_dataset"] == "halueval").sum(), 32)
        self.assertEqual((meta_test["source_dataset"] == "halueval").sum(), 8)

        # FEVER: 20 total -> 16 train, 4 test
        self.assertEqual((meta_train["source_dataset"] == "fever").sum(), 16)
        self.assertEqual((meta_test["source_dataset"] == "fever").sum(), 4)

    def test_no_data_leakage_in_scaling(self):
        """Verify StandardScaler in Pipeline fits ONLY on training data."""
        X, y, meta = prepare_feature_dataframe(self.mock_df)
        X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y, meta, test_size=0.2, random_state=42)

        pipeline = build_logistic_regression_pipeline(max_iter=100, random_state=42)
        pipeline.fit(X_train, y_train)

        scaler = pipeline.named_steps["scaler"]
        # Scaler mean_ must equal X_train mean, NOT combined mean
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0).values, rtol=1e-5)
        # Scaler scale_ must equal X_train std (population std)
        np.testing.assert_allclose(scaler.scale_, X_train.std(axis=0, ddof=0).values, rtol=1e-5)

        # Ensure predictions work on test set
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)
        self.assertEqual(len(preds), len(X_test))
        self.assertEqual(probs.shape, (len(X_test), 2))

    def test_xgboost_model_construction(self):
        """Verify XGBoost model instantiation, fitting, and prediction."""
        X, y, meta = prepare_feature_dataframe(self.mock_df)
        X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y, meta, test_size=0.2, random_state=42)

        xgb = build_xgboost_model(n_estimators=10, max_depth=3, random_state=42)
        xgb.fit(X_train, y_train)

        preds = xgb.predict(X_test)
        probs = xgb.predict_proba(X_test)

        self.assertEqual(len(preds), len(X_test))
        self.assertEqual(probs.shape, (len(X_test), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_compute_ece(self):
        """Verify Expected Calibration Error binning calculation."""
        # Case 1: Perfectly calibrated probabilities
        # Bin 1: prob=0.1, true=0 (acc=0, conf=0.1, diff=0.1)
        # Bin 9: prob=0.9, true=1 (acc=1, conf=0.9, diff=0.1)
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.array([0.0] * 50 + [1.0] * 50)
        ece_perfect = compute_ece(y_true, y_prob, n_bins=10)
        self.assertAlmostEqual(ece_perfect, 0.0, places=5)

        # Case 2: Completely uncalibrated probabilities (inverted)
        # Predicting 1.0 when true is 0, and 0.0 when true is 1
        y_prob_inverted = np.array([1.0] * 50 + [0.0] * 50)
        ece_worst = compute_ece(y_true, y_prob_inverted, n_bins=10)
        self.assertAlmostEqual(ece_worst, 1.0, places=5)

        # Case 3: Empty input
        self.assertEqual(compute_ece([], []), 0.0)

    def test_evaluate_predictions_metrics(self):
        """Verify comprehensive evaluation metrics suite."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.7, 0.8, 0.9, 0.4])

        metrics = evaluate_predictions(y_true, y_pred, y_prob, n_bins_ece=5)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("confusion_matrix", metrics)

        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"], 2)
        self.assertEqual(cm["tn"], 1)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["fn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 3 / 5)

    def test_run_baseline_experiment_mock(self):
        """Verify end-to-end execution of run_baseline_experiment with synthetic data."""
        results = run_baseline_experiment(
            data_path=self.mock_df,
            output_dir=None,
            test_size=0.2,
            random_state=42,
            include_num_tokens=False,
        )

        self.assertIn("logistic_regression", results)
        self.assertIn("xgboost", results)
        self.assertIn("split_summary", results)

        lr_res = results["logistic_regression"]
        xgb_res = results["xgboost"]

        # Both models produced all expected metrics
        for res in [lr_res, xgb_res]:
            self.assertIn("accuracy", res)
            self.assertIn("roc_auc", res)
            self.assertIn("pr_auc", res)
            self.assertIn("ece", res)
            self.assertIn("brier_score", res)
            self.assertIn("confusion_matrix", res)

        # Verify num_tokens was not used
        self.assertEqual(results["split_summary"]["num_features"], 11)
        self.assertNotIn("num_tokens", results["split_summary"]["features"])


if __name__ == "__main__":
    unittest.main()
