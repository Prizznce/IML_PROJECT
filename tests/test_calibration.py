"""
Unit and Integration Tests for Probability Calibration Analysis (Phase 17).

Validates nested split isolation (model-fit, calibration-fit, untouched test set),
zero ID leakage, 10-bin reliability calculations, ECE/MCE/Brier metric bounds,
and post-hoc calibrator outputs.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import IsotonicRegression, _SigmoidCalibration

from src.evaluation.calibration import (
    compute_calibration_curve_and_metrics,
    evaluate_model_prob_metrics,
    run_calibration_study,
    split_training_into_model_and_calib,
)
from src.features.build_universal_features import UNIVERSAL_CORE_FEATURES
from src.training.combined_models import split_dataset


@pytest.fixture
def mock_universal_dataset() -> pd.DataFrame:
    """Generate synthetic universal features dataset for calibration testing."""
    np.random.seed(42)
    n = 100
    sources = ["halueval"] * 40 + ["truthfulqa"] * 40 + ["fever"] * 20
    labels = [0, 1] * 50

    data = {
        "id": [f"item_{i}" for i in range(n)],
        "source_dataset": sources,
        "label": labels,
    }
    for feat in UNIVERSAL_CORE_FEATURES:
        data[feat] = np.random.randn(n)

    return pd.DataFrame(data)


class TestCalibrationPartitionsAndIsolation:
    """Test nested splitting and zero-leakage guarantees."""

    def test_nested_split_sizes_and_zero_overlap(self, mock_universal_dataset):
        """Verify outer test and inner calibration splits have zero ID overlap."""
        df = mock_universal_dataset
        X_full = df[list(UNIVERSAL_CORE_FEATURES)]
        y_full = df["label"]
        meta_full = df[["id", "source_dataset", "label"]]

        # Outer split (80% train, 20% test)
        X_tr, X_te, y_tr, y_te, meta_tr, meta_te = split_dataset(
            X_full, y_full, meta_full, test_size=0.2, random_state=42
        )
        assert len(X_te) == 20
        assert len(X_tr) == 80

        # Inner split on train (80% model, 20% calib)
        X_mod, X_cal, y_mod, y_cal, meta_mod, meta_cal = split_training_into_model_and_calib(
            X_tr, y_tr, meta_tr, calib_size=0.2, random_state=42
        )
        assert len(X_mod) == 64
        assert len(X_cal) == 16

        test_ids = set(meta_te["id"])
        model_ids = set(meta_mod["id"])
        calib_ids = set(meta_cal["id"])

        # Test set completely isolated
        assert len(test_ids.intersection(model_ids)) == 0
        assert len(test_ids.intersection(calib_ids)) == 0
        assert len(model_ids.intersection(calib_ids)) == 0
        assert len(test_ids) + len(model_ids) + len(calib_ids) == len(df)


class TestCalibrationMetricsCalculation:
    """Test reliability table, ECE, MCE, Brier score, and Cox slope/intercept."""

    def test_perfect_calibration_low_ece(self):
        """Perfect predictions yield 0.0 ECE and 0.0 MCE."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.array([0.0] * 50 + [1.0] * 50)

        calib = compute_calibration_curve_and_metrics(y_true, y_prob, n_bins=10)
        assert pytest.approx(calib["ece"], abs=1e-5) == 0.0
        assert pytest.approx(calib["mce"], abs=1e-5) == 0.0
        assert pytest.approx(calib["brier_score"], abs=1e-5) == 0.0

    def test_calibration_metric_bounds_and_structure(self):
        """Metrics are bounded in [0, 1] and bins sum to sample count."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_prob = np.random.uniform(0.0, 1.0, n)

        calib = compute_calibration_curve_and_metrics(y_true, y_prob, n_bins=10)

        assert 0.0 <= calib["ece"] <= 1.0
        assert 0.0 <= calib["mce"] <= 1.0
        assert 0.0 <= calib["brier_score"] <= 1.0
        assert calib["mce"] >= calib["ece"]  # MCE is upper bound on ECE
        assert len(calib["bins"]) == 10

        total_binned = sum(b["sample_count"] for b in calib["bins"])
        assert total_binned == n

    def test_calibrators_fit_and_predict_bounds(self):
        """Sigmoid and Isotonic calibrators produce valid probabilities in [0, 1]."""
        np.random.seed(42)
        n = 80
        prob_calib = np.random.uniform(0.1, 0.9, n)
        y_calib = np.random.randint(0, 2, n)

        sig = _SigmoidCalibration().fit(prob_calib, y_calib)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(prob_calib, y_calib)

        test_prob = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
        sig_pred = sig.predict(test_prob)
        iso_pred = iso.predict(test_prob)

        assert np.all((sig_pred >= 0.0) & (sig_pred <= 1.0))
        assert np.all((iso_pred >= 0.0) & (iso_pred <= 1.0))


class TestEndToEndCalibrationStudy:
    """Test full execution of calibration study on mock data."""

    def test_mock_calibration_run(self, mock_universal_dataset, tmp_path):
        """Run complete calibration pipeline on mock dataset and verify artifacts."""
        out_dir = tmp_path / "calibration_test"
        results = run_calibration_study(
            data_path=mock_universal_dataset,
            output_dir=out_dir,
            random_state=42,
        )

        assert "models" in results
        assert "logistic_regression" in results["models"]
        assert "xgboost" in results["models"]

        # Check files
        assert (out_dir / "calibration_results.json").exists()
        assert (out_dir / "calibration_results.csv").exists()
        assert (out_dir / "calibration_reliability_tables.csv").exists()
        assert (out_dir / "calibration_predictions.parquet").exists()
        assert (out_dir / "calibration_summary.md").exists()
        assert (out_dir / "logistic_regression_reliability.png").exists()
        assert (out_dir / "xgboost_reliability.png").exists()

        # Check CSV rows: 2 models x 3 methods = 6 rows
        res_df = pd.read_csv(out_dir / "calibration_results.csv")
        assert len(res_df) == 6

        # Check reliability table: 6 models/methods x 10 bins = 60 rows
        rel_df = pd.read_csv(out_dir / "calibration_reliability_tables.csv")
        assert len(rel_df) == 60

        # Check predictions: 20 test samples x 2 models x 3 methods = 120 rows
        pred_df = pd.read_parquet(out_dir / "calibration_predictions.parquet")
        assert len(pred_df) == 20 * 2 * 3
