"""
Unit and Integration Tests for Universal Hallucination Detector Ablation Study (Phase 15).

Verifies condition specifications, feature set integrity, split invariance, delta
mathematics, and prediction record formatting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation_study import (
    ABLATION_CONDITIONS,
    REFERENCE_CONDITION_KEY,
    compute_ablation_deltas,
    run_ablation_study,
    validate_ablation_conditions,
)
from src.features.build_universal_features import (
    FORBIDDEN_FEATURE_COLUMNS,
    INTERNAL_SIGNAL_FEATURES,
    NLI_AGREEMENT_FEATURES,
    SELF_CONSISTENCY_FEATURES,
    UNIVERSAL_CORE_FEATURES,
)


@pytest.fixture
def mock_universal_dataframe() -> pd.DataFrame:
    """Generate synthetic universal feature table for testing."""
    np.random.seed(42)
    n = 60
    data = {
        "id": [f"ex_{i}" for i in range(n)],
        "source_dataset": np.random.choice(["halueval", "truthfulqa", "fever"], size=n),
        "label": np.random.choice([0, 1], size=n),
    }

    for feat in UNIVERSAL_CORE_FEATURES:
        data[feat] = np.random.randn(n)

    return pd.DataFrame(data)


class TestAblationConditionSpecifications:
    """Test validity of 7 ablation condition definitions."""

    def test_condition_count_and_keys(self):
        """Must have exactly 7 defined conditions."""
        assert len(ABLATION_CONDITIONS) == 7
        expected_keys = {
            "internal_only",
            "self_consistency_only",
            "nli_only",
            "internal_plus_self_consistency",
            "internal_plus_nli",
            "self_consistency_plus_nli",
            "all_three",
        }
        assert set(ABLATION_CONDITIONS.keys()) == expected_keys

    def test_reference_condition_is_internal_only(self):
        """Reference condition must be internal_only."""
        assert REFERENCE_CONDITION_KEY == "internal_only"
        assert ABLATION_CONDITIONS["internal_only"]["is_reference"] is True
        assert ABLATION_CONDITIONS["internal_only"]["feature_count"] == 11

    def test_condition_feature_counts(self):
        """Validate feature count in each condition."""
        expected_counts = {
            "internal_only": 11,
            "self_consistency_only": 5,
            "nli_only": 3,
            "internal_plus_self_consistency": 16,
            "internal_plus_nli": 14,
            "self_consistency_plus_nli": 8,
            "all_three": 19,
        }
        for k, expected_count in expected_counts.items():
            assert ABLATION_CONDITIONS[k]["feature_count"] == expected_count
            assert len(ABLATION_CONDITIONS[k]["features"]) == expected_count

    def test_no_forbidden_columns_in_any_condition(self):
        """No ablation condition may contain any forbidden column."""
        for k, cond in ABLATION_CONDITIONS.items():
            overlap = set(cond["features"]).intersection(FORBIDDEN_FEATURE_COLUMNS)
            assert len(overlap) == 0, f"Condition {k} contains forbidden columns: {overlap}"
            assert "num_tokens" not in cond["features"]
            assert "source_dataset" not in cond["features"]
            assert "label" not in cond["features"]
            assert "id" not in cond["features"]

    def test_validate_ablation_conditions_helper(self):
        """Helper function executes without assertion errors."""
        validate_ablation_conditions()


class TestAblationDeltaMathematics:
    """Test delta calculations relative to the reference baseline."""

    def test_delta_zero_for_reference_condition(self):
        """Condition A relative to Condition A must produce strictly 0.0 delta."""
        ref_metrics = {
            "accuracy": 0.65,
            "f1": 0.62,
            "roc_auc": 0.70,
            "pr_auc": 0.68,
            "brier_score": 0.21,
            "ece": 0.05,
        }
        deltas = compute_ablation_deltas(ref_metrics, ref_metrics)
        for k, v in deltas.items():
            assert v == 0.0, f"Delta {k} is not 0.0: {v}"

    def test_delta_arithmetic(self):
        """Verify delta = ablation - reference."""
        ref = {"accuracy": 0.60, "f1": 0.55, "roc_auc": 0.65, "pr_auc": 0.62, "brier_score": 0.22, "ece": 0.06}
        abl = {"accuracy": 0.68, "f1": 0.65, "roc_auc": 0.75, "pr_auc": 0.72, "brier_score": 0.19, "ece": 0.04}

        deltas = compute_ablation_deltas(abl, ref)
        assert pytest.approx(deltas["delta_accuracy"], abs=1e-5) == 0.08
        assert pytest.approx(deltas["delta_f1"], abs=1e-5) == 0.10
        assert pytest.approx(deltas["delta_roc_auc"], abs=1e-5) == 0.10
        assert pytest.approx(deltas["delta_pr_auc"], abs=1e-5) == 0.10
        assert pytest.approx(deltas["delta_brier"], abs=1e-5) == -0.03
        assert pytest.approx(deltas["delta_ece"], abs=1e-5) == -0.02


class TestEndToEndAblationPipeline:
    """Test end-to-end execution of ablation pipeline on synthetic data."""

    def test_mock_ablation_run(self, mock_universal_dataframe, tmp_path):
        """Run complete ablation pipeline on mock dataset and verify artifacts."""
        out_dir = tmp_path / "ablation_test"
        results = run_ablation_study(
            data_path=mock_universal_dataframe,
            output_dir=out_dir,
            test_size=0.2,
            random_state=42,
        )

        assert results["num_conditions"] == 7
        assert len(results["conditions"]) == 7

        # Check generated files
        assert (out_dir / "ablation_results.json").exists()
        assert (out_dir / "ablation_results.csv").exists()
        assert (out_dir / "ablation_per_dataset.csv").exists()
        assert (out_dir / "ablation_test_predictions.parquet").exists()
        assert (out_dir / "ablation_summary.md").exists()

        # Check CSV rows
        overall_csv = pd.read_csv(out_dir / "ablation_results.csv")
        assert len(overall_csv) == 14  # 7 conditions x 2 models

        # Check predictions parquet
        pred_df = pd.read_parquet(out_dir / "ablation_test_predictions.parquet")
        n_test = results["test_samples"]
        assert len(pred_df) == 7 * 2 * n_test
        assert set(pred_df["condition"].unique()) == set(ABLATION_CONDITIONS.keys())
        assert set(pred_df["model"].unique()) == {"logistic_regression", "xgboost"}
        assert np.all((pred_df["y_probability"] >= 0.0) & (pred_df["y_probability"] <= 1.0))
