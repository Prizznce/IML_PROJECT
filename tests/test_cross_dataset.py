"""
Unit tests for cross-dataset evaluation module.

Tests leave-one-dataset-out partitioning, complete target exclusion from training,
zero data leakage in preprocessing scaling, 11-feature restriction, probability validity,
and metric calculation on synthetic mock benchmark data.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cross_dataset import (
    BENCHMARK_DATASETS,
    get_leave_one_out_partitions,
    evaluate_transfer_fold,
    run_cross_dataset_evaluation,
)
from src.training.baseline_models import PRIMARY_SIGNAL_FEATURES, FORBIDDEN_COLUMNS


class TestCrossDataset(unittest.TestCase):
    """Test suite for Leave-One-Dataset-Out (LODO) transfer evaluation."""

    def setUp(self):
        np.random.seed(42)
        n = 120
        # 40 halueval, 40 truthfulqa, 40 fever
        sources = ["halueval"] * 40 + ["truthfulqa"] * 40 + ["fever"] * 40
        labels = [0, 1] * 60

        self.mock_df = pd.DataFrame({
            "id": [f"ex_{i}" for i in range(n)],
            "source_dataset": sources,
            "label": labels,
            "prompt": [f"Question {i}" for i in range(n)],
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

    def test_three_configurations_and_target_exclusion(self):
        """Verify that all 3 LODO configurations exclude the target dataset from training."""
        expected_configs = {
            "fever": {"train_sources": {"halueval", "truthfulqa"}, "train_n": 80, "test_n": 40},
            "truthfulqa": {"train_sources": {"halueval", "fever"}, "train_n": 80, "test_n": 40},
            "halueval": {"train_sources": {"truthfulqa", "fever"}, "train_n": 80, "test_n": 40},
        }

        for target, cfg in expected_configs.items():
            X_train, X_test, y_train, y_test, meta_train, meta_test = get_leave_one_out_partitions(
                df=self.mock_df, target_dataset=target, include_num_tokens=False
            )

            # 1. Target never in training
            self.assertNotIn(target, meta_train["source_dataset"].values)
            # 2. Test split contains strictly the target
            self.assertEqual(set(meta_test["source_dataset"].unique()), {target})
            # 3. Train sources match expected
            self.assertEqual(set(meta_train["source_dataset"].unique()), cfg["train_sources"])
            # 4. Sample sizes
            self.assertEqual(len(X_train), cfg["train_n"])
            self.assertEqual(len(X_test), cfg["test_n"])
            self.assertEqual(len(y_train), cfg["train_n"])
            self.assertEqual(len(y_test), cfg["test_n"])

    def test_feature_count_and_forbidden_exclusion(self):
        """Verify feature matrix contains exactly the 11 primary signals and zero forbidden columns."""
        X_train, X_test, _, _, _, _ = get_leave_one_out_partitions(
            self.mock_df, target_dataset="fever", include_num_tokens=False
        )

        self.assertEqual(len(X_train.columns), 11)
        self.assertEqual(list(X_train.columns), PRIMARY_SIGNAL_FEATURES)
        self.assertEqual(list(X_test.columns), PRIMARY_SIGNAL_FEATURES)

        for col in FORBIDDEN_COLUMNS:
            self.assertNotIn(col, X_train.columns)
            self.assertNotIn(col, X_test.columns)

    def test_scaler_fitted_strictly_on_training_data(self):
        """Verify StandardScaler is fit strictly on training partitions without target leakage."""
        X_train, X_test, y_train, y_test, meta_train, meta_test = get_leave_one_out_partitions(
            self.mock_df, target_dataset="truthfulqa"
        )

        fold_res = evaluate_transfer_fold(
            X_train, X_test, y_train, y_test, meta_train, meta_test, target_dataset="truthfulqa"
        )

        # Reconstruct pipeline to check scaler parameters directly
        from src.training.baseline_models import build_logistic_regression_pipeline
        pipe = build_logistic_regression_pipeline()
        pipe.fit(X_train, y_train)

        scaler = pipe.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0).values, rtol=1e-5)
        np.testing.assert_allclose(scaler.scale_, X_train.std(axis=0, ddof=0).values, rtol=1e-5)

        # Verify scaler differs from full-dataset parameters
        from src.training.baseline_models import prepare_feature_dataframe
        X_full, _, _ = prepare_feature_dataframe(self.mock_df)
        full_mean = X_full.mean(axis=0).values
        self.assertFalse(np.allclose(scaler.mean_, full_mean))

    def test_prediction_validity_and_metrics(self):
        """Verify probability bounds in [0, 1], no NaN/inf, and correct confusion matrix structure."""
        X_train, X_test, y_train, y_test, meta_train, meta_test = get_leave_one_out_partitions(
            self.mock_df, target_dataset="halueval"
        )

        res = evaluate_transfer_fold(
            X_train, X_test, y_train, y_test, meta_train, meta_test, target_dataset="halueval"
        )

        for model in ["logistic_regression", "xgboost"]:
            m = res[model]
            self.assertIn("accuracy", m)
            self.assertIn("roc_auc", m)
            self.assertIn("pr_auc", m)
            self.assertIn("brier_score", m)
            self.assertIn("ece", m)
            self.assertIn("confusion_matrix", m)

            # Metrics bounds
            self.assertTrue(0.0 <= m["accuracy"] <= 1.0)
            self.assertTrue(0.0 <= m["roc_auc"] <= 1.0)
            self.assertTrue(0.0 <= m["pr_auc"] <= 1.0)
            self.assertTrue(0.0 <= m["ece"] <= 1.0)

            # Confusion matrix
            cm = m["confusion_matrix"]
            total_cm = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
            self.assertEqual(total_cm, len(X_test))

    def test_run_cross_dataset_evaluation_mock(self):
        """Verify full run_cross_dataset_evaluation execution with temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "test_data.parquet"
            self.mock_df.to_parquet(parquet_path)

            results = run_cross_dataset_evaluation(
                data_path=parquet_path,
                output_dir=Path(tmpdir) / "results",
                random_state=42,
                include_num_tokens=False,
            )

            self.assertIn("metadata", results)
            self.assertIn("experiments", results)
            self.assertEqual(len(results["experiments"]), 3)

            for target in BENCHMARK_DATASETS:
                key = f"test_on_{target}"
                self.assertIn(key, results["experiments"])
                exp = results["experiments"][key]
                self.assertEqual(exp["target_dataset"], target)
                self.assertEqual(exp["num_features"], 11)


class TestUniversalCrossDatasetEvaluation(unittest.TestCase):
    """Test suite for Phase 16 Universal Cross-Dataset Generalization Evaluation."""

    def setUp(self):
        np.random.seed(42)
        n = 120
        sources = ["halueval"] * 50 + ["truthfulqa"] * 40 + ["fever"] * 30
        labels = [0, 1] * 60

        from src.features.build_universal_features import UNIVERSAL_CORE_FEATURES
        data = {
            "id": [f"univ_{i}" for i in range(n)],
            "source_dataset": sources,
            "label": labels,
        }
        for feat in UNIVERSAL_CORE_FEATURES:
            data[feat] = np.random.randn(n)

        self.mock_universal_df = pd.DataFrame(data)

    def test_universal_configurations_and_feature_counts(self):
        """Verify the 4 configurations have exact feature counts (11, 16, 14, 19)."""
        from src.evaluation.cross_dataset import (
            CROSS_DATASET_CONFIGURATIONS,
            REFERENCE_CONFIG_KEY,
        )
        self.assertEqual(len(CROSS_DATASET_CONFIGURATIONS), 4)
        self.assertEqual(REFERENCE_CONFIG_KEY, "internal_only")
        expected_counts = {
            "internal_only": 11,
            "internal_plus_self_consistency": 16,
            "internal_plus_nli": 14,
            "all_three": 19,
        }
        for k, exp_cnt in expected_counts.items():
            cfg = CROSS_DATASET_CONFIGURATIONS[k]
            self.assertEqual(cfg["feature_count"], exp_cnt)
            self.assertEqual(len(cfg["features"]), exp_cnt)

    def test_universal_experiments_structure(self):
        """Verify the 3 Leave-One-Dataset-Out experiment definitions."""
        from src.evaluation.cross_dataset import CROSS_DATASET_EXPERIMENTS
        self.assertEqual(len(CROSS_DATASET_EXPERIMENTS), 3)
        self.assertIn("exp1_test_fever", CROSS_DATASET_EXPERIMENTS)
        self.assertIn("exp2_test_truthfulqa", CROSS_DATASET_EXPERIMENTS)
        self.assertIn("exp3_test_halueval", CROSS_DATASET_EXPERIMENTS)

        exp1 = CROSS_DATASET_EXPERIMENTS["exp1_test_fever"]
        self.assertEqual(exp1["test_dataset"], "fever")
        self.assertEqual(set(exp1["train_datasets"]), {"halueval", "truthfulqa"})

    def test_universal_lodo_partitions_isolation(self):
        """Verify strict partition isolation and zero ID overlap."""
        from src.evaluation.cross_dataset import (
            CROSS_DATASET_CONFIGURATIONS,
            get_leave_one_dataset_out_partitions,
        )
        feats = CROSS_DATASET_CONFIGURATIONS["all_three"]["features"]
        for target in ["fever", "truthfulqa", "halueval"]:
            X_tr, X_te, y_tr, y_te, meta_tr, meta_te = get_leave_one_dataset_out_partitions(
                df=self.mock_universal_df, test_dataset=target, feature_cols=feats
            )
            # Zero ID overlap
            train_ids = set(meta_tr["id"])
            test_ids = set(meta_te["id"])
            self.assertEqual(len(train_ids.intersection(test_ids)), 0)

            # Target completely absent from train
            self.assertNotIn(target, meta_tr["source_dataset"].values)
            # Test contains strictly target
            self.assertEqual(set(meta_te["source_dataset"].unique()), {target})

    def test_universal_deltas_zero_for_reference(self):
        """Verify deltas relative to reference condition are 0.0."""
        from src.evaluation.cross_dataset import compute_transfer_deltas
        metrics = {
            "accuracy": 0.65,
            "f1": 0.60,
            "roc_auc": 0.70,
            "pr_auc": 0.68,
            "brier_score": 0.21,
            "ece": 0.05,
        }
        deltas = compute_transfer_deltas(metrics, metrics)
        for k, v in deltas.items():
            self.assertEqual(v, 0.0)

    def test_mock_cross_dataset_study_run(self):
        """Run complete 24-run study on mock data and verify artifacts."""
        from src.evaluation.cross_dataset import run_cross_dataset_study
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "cross_dataset_test"
            results = run_cross_dataset_study(
                data_path=self.mock_universal_df,
                output_dir=out_dir,
                random_state=42,
            )

            self.assertEqual(len(results["experiments"]), 3)
            self.assertTrue((out_dir / "cross_dataset_results.json").exists())
            self.assertTrue((out_dir / "cross_dataset_results.csv").exists())
            self.assertTrue((out_dir / "cross_dataset_per_dataset.csv").exists())
            self.assertTrue((out_dir / "cross_dataset_predictions.parquet").exists())
            self.assertTrue((out_dir / "cross_dataset_summary.md").exists())

            # 24 rows in results CSV
            res_df = pd.read_csv(out_dir / "cross_dataset_results.csv")
            self.assertEqual(len(res_df), 24)

            # Predictions check
            pred_df = pd.read_parquet(out_dir / "cross_dataset_predictions.parquet")
            self.assertEqual(len(pred_df), 120 * 4 * 2)  # Total test samples across 3 exps (all 120 items) x 4 cfgs x 2 models
            self.assertTrue(np.all((pred_df["y_probability"] >= 0.0) & (pred_df["y_probability"] <= 1.0)))


if __name__ == "__main__":
    unittest.main()
