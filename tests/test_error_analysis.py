"""
Unit tests for Phase 18: Hallucination Detector Error Analysis.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    PRIMARY_19_FEATURES,
    EXPECTED_TEST_SAMPLE_COUNT,
    EXPECTED_DATASET_COUNTS,
    EXPECTED_LABEL_COUNTS,
    load_and_validate_error_analysis_data,
    assign_error_categories,
    compute_confidence,
    compute_cohen_d,
    compute_error_category_statistics,
    compute_dataset_error_metrics,
    compute_feature_error_effects,
    compute_model_disagreement,
    run_error_analysis_pipeline,
)


class TestErrorAnalysisDataValidation:
    """Tests for data loading, schema integrity, and validation assertions."""

    def test_production_data_loading_and_invariants(self):
        preds_path = "experiments/baselines/combined/combined_test_predictions.parquet"
        proc_path = "data/processed/combined_processed.parquet"
        feat_path = "experiments/baselines/combined/universal_features.parquet"

        df = load_and_validate_error_analysis_data(preds_path, proc_path, feat_path)

        assert len(df) == EXPECTED_TEST_SAMPLE_COUNT
        assert df["id"].nunique() == EXPECTED_TEST_SAMPLE_COUNT

        # Check label balance
        lbl_counts = df["label"].value_counts().to_dict()
        assert lbl_counts[0] == EXPECTED_LABEL_COUNTS[0]
        assert lbl_counts[1] == EXPECTED_LABEL_COUNTS[1]

        # Check dataset balance
        ds_counts = df["source_dataset"].value_counts().to_dict()
        for ds, exp_cnt in EXPECTED_DATASET_COUNTS.items():
            assert ds_counts[ds] == exp_cnt

        # Check columns
        required_cols = [
            "id",
            "source_dataset",
            "label",
            "lr_probability",
            "lr_prediction",
            "xgb_probability",
            "xgb_prediction",
            "prompt",
            "response",
            "context",
        ] + PRIMARY_19_FEATURES

        for col in required_cols:
            assert col in df.columns

        # Check probabilities
        assert (df["lr_probability"] >= 0.0).all() and (df["lr_probability"] <= 1.0).all()
        assert (df["xgb_probability"] >= 0.0).all() and (df["xgb_probability"] <= 1.0).all()

    def test_validation_fails_on_bad_sample_count(self, tmp_path):
        bad_df = pd.DataFrame({
            "id": ["sample_1"],
            "source_dataset": ["halueval"],
            "label": [0],
            "lr_probability": [0.2],
            "lr_prediction": [0],
            "xgb_probability": [0.3],
            "xgb_prediction": [0],
        })
        bad_preds_path = tmp_path / "bad_preds.parquet"
        bad_df.to_parquet(bad_preds_path)

        proc_path = "data/processed/combined_processed.parquet"
        feat_path = "experiments/baselines/combined/universal_features.parquet"

        with pytest.raises(ValueError, match="Expected 800 test predictions"):
            load_and_validate_error_analysis_data(bad_preds_path, proc_path, feat_path)


class TestErrorCategoryCalculations:
    """Tests for confusion categories, probabilities, and confidence."""

    def test_confusion_category_assignment(self):
        synth_df = pd.DataFrame({
            "label": [1, 0, 0, 1],
            "pred": [1, 0, 1, 0],
            "prob": [0.9, 0.1, 0.8, 0.2],
        })
        cats = assign_error_categories(synth_df, "pred", "label").tolist()
        assert cats == ["TP", "TN", "FP", "FN"]

    def test_confidence_calculation(self):
        probs = np.array([0.9, 0.1, 0.8, 0.2, 0.5])
        preds = np.array([1, 0, 1, 0, 1])
        labels = np.array([1, 0, 0, 1, 1])
        confs = compute_confidence(probs, preds, labels)

        # All confidences must be in [0.5, 1.0]
        assert (confs >= 0.5).all()
        assert (confs <= 1.0).all()

        # Check values
        np.testing.assert_allclose(confs, [0.9, 0.9, 0.8, 0.8, 0.5])

    def test_error_category_statistics_sums(self):
        synth_df = pd.DataFrame({
            "label": [1, 0, 0, 1, 1, 0],
            "pred": [1, 0, 1, 0, 1, 0],
            "prob": [0.8, 0.2, 0.7, 0.3, 0.9, 0.1],
            "prompt": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "response": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "context": ["c1", "c2", "", None, "c5", "c6"],
        })
        stats = compute_error_category_statistics(synth_df, "Model", "prob", "pred")
        total_count = sum(s["count"] for s in stats.values())
        assert total_count == 6


class TestMathematicalMetricsAndEffectSizes:
    """Tests for Cohen's d effect size and dataset error rates."""

    def test_cohen_d_identical_groups(self):
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = compute_cohen_d(g1, g2)
        assert abs(d) < 1e-6

    def test_cohen_d_known_shift(self):
        np.random.seed(42)
        g1 = np.random.normal(loc=1.0, scale=1.0, size=1000)
        g2 = np.random.normal(loc=0.0, scale=1.0, size=1000)
        d = compute_cohen_d(g1, g2)
        # Should be approximately 1.0
        assert 0.9 < d < 1.1

    def test_dataset_error_metrics_rates(self):
        synth_df = pd.DataFrame({
            "source_dataset": ["A", "A", "B", "B"],
            "label": [0, 1, 0, 1],
            "pred": [0, 0, 1, 1],
            "prob": [0.1, 0.4, 0.7, 0.8],
        })
        metrics = compute_dataset_error_metrics(synth_df, "prob", "pred")
        overall = next(m for m in metrics if m["slice"] == "overall")
        assert overall["total_samples"] == 4
        assert overall["tp"] == 1
        assert overall["tn"] == 1
        assert overall["fp"] == 1
        assert overall["fn"] == 1
        assert overall["error_rate"] == 0.5


class TestModelDisagreementLogic:
    """Tests for inter-model agreement and confusion comparisons."""

    def test_disagreement_counts_and_partitions(self):
        synth_df = pd.DataFrame({
            "label": [1, 0, 1, 0],
            "lr_prediction": [1, 0, 0, 1],
            "xgb_prediction": [1, 0, 1, 0],
            "lr_probability": [0.8, 0.2, 0.3, 0.7],
            "xgb_probability": [0.9, 0.1, 0.6, 0.4],
            "id": ["s1", "s2", "s3", "s4"],
            "source_dataset": ["ds", "ds", "ds", "ds"],
            "prompt": ["p1", "p2", "p3", "p4"],
            "response": ["r1", "r2", "r3", "r4"],
        })
        summary, top_diffs = compute_model_disagreement(synth_df)

        assert summary["total_samples"] == 4
        assert summary["agreement_count"] == 2
        assert summary["disagreement_count"] == 2
        assert summary["both_correct_count"] == 2  # s1 (both 1), s2 (both 0)
        assert summary["both_wrong_count"] == 0
        assert summary["lr_correct_xgb_wrong_count"] == 0
        assert summary["lr_wrong_xgb_correct_count"] == 2  # s3, s4
        assert len(top_diffs) == 4


class TestEndToEndErrorAnalysisPipeline:
    """Tests running the complete error analysis pipeline and verifying file creation."""

    def test_full_pipeline_execution(self, tmp_path):
        out_dir = tmp_path / "error_analysis_test"
        results = run_error_analysis_pipeline(output_dir=out_dir)

        assert results["metadata"]["total_test_samples"] == 800

        # Check all required files exist
        expected_files = [
            "error_analysis_summary.md",
            "error_analysis_results.json",
            "error_category_statistics.csv",
            "error_by_dataset.csv",
            "probability_by_error_category.csv",
            "feature_error_comparison.csv",
            "model_disagreement.csv",
            "top_false_positives.csv",
            "top_false_negatives.csv",
            "low_confidence_correct.csv",
            "top_model_disagreements.csv",
            "confusion_matrix_lr.png",
            "confusion_matrix_xgb.png",
            "probability_distribution_lr.png",
            "probability_distribution_xgb.png",
            "error_rate_by_dataset_lr.png",
            "error_rate_by_dataset_xgb.png",
            "feature_differences_comparison.png",
        ]

        for fname in expected_files:
            file_path = out_dir / fname
            assert file_path.exists(), f"Missing expected output: {fname}"
            assert file_path.stat().st_size > 0, f"Empty file: {fname}"
