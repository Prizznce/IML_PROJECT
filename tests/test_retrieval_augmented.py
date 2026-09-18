"""
Unit tests for Phase 19: Retrieval-Augmented Hallucination Detection Variant.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.retrieval.retrieval_augmented_experiment import (
    CONDITION_FEATURE_MAP,
    RETRIEVAL_FEATURES,
    build_dataset_evidence_corpora,
    clean_fever_title,
    compute_ece_mce,
    evaluate_predictions,
    train_and_evaluate_retrieval_models,
)
from src.features.build_universal_features import (
    INTERNAL_SIGNAL_FEATURES,
    UNIVERSAL_CORE_FEATURES,
)


class TestCorpusConstructionAndIntegrity:
    """Tests for dataset-appropriate evidence corpora construction."""

    def test_clean_fever_title(self):
        raw = "Temple_Grandin_-LRB-film-RRB-"
        cleaned = clean_fever_title(raw)
        assert cleaned == "Temple Grandin (film)"

    def test_production_corpora_construction(self):
        corpora, manifest = build_dataset_evidence_corpora()

        assert "halueval" in corpora
        assert "fever" in corpora
        assert len(corpora["halueval"]) == 1499
        assert len(corpora["fever"]) == 829

        # Verify TruthfulQA is documented as excluded
        assert manifest["truthfulqa"]["status"] == "excluded_from_retrieval_experiment"

        # Leakage check: no labels or binary indicator strings
        for title in corpora["fever"]:
            assert "label=" not in title
            assert "hallucinated" not in title.lower()

        for passage in corpora["halueval"][:50]:
            assert "label=" not in passage
            assert "hallucinated" not in passage.lower()


class TestConditionFeatureDefinitions:
    """Tests for feature counts, composition, and nesting across the 5 conditions."""

    def test_condition_feature_counts(self):
        assert len(RETRIEVAL_FEATURES) == 6
        assert len(INTERNAL_SIGNAL_FEATURES) == 11
        assert len(UNIVERSAL_CORE_FEATURES) == 19

        assert len(CONDITION_FEATURE_MAP["internal_only"]) == 11
        assert len(CONDITION_FEATURE_MAP["universal_core"]) == 19
        assert len(CONDITION_FEATURE_MAP["retrieval_only"]) == 6
        assert len(CONDITION_FEATURE_MAP["internal_plus_retrieval"]) == 17
        assert len(CONDITION_FEATURE_MAP["universal_plus_retrieval"]) == 25

    def test_nesting_integrity(self):
        # internal_only is subset of universal_core
        assert set(CONDITION_FEATURE_MAP["internal_only"]).issubset(
            set(CONDITION_FEATURE_MAP["universal_core"])
        )
        # retrieval_only is disjoint from universal_core
        assert set(RETRIEVAL_FEATURES).isdisjoint(set(UNIVERSAL_CORE_FEATURES))
        # universal_plus_retrieval is union of universal_core and retrieval
        expected_union = set(UNIVERSAL_CORE_FEATURES).union(set(RETRIEVAL_FEATURES))
        assert set(CONDITION_FEATURE_MAP["universal_plus_retrieval"]) == expected_union


class TestCalibrationMetricsAndBounds:
    """Tests for ECE, MCE, and metric bounds."""

    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        ece, mce = compute_ece_mce(y_true, y_prob, n_bins=10)
        assert ece == 0.0
        assert mce == 0.0

    def test_evaluate_predictions_bounds(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.6, 0.4])

        metrics = evaluate_predictions(y_true, y_pred, y_prob)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["mce"] <= 1.0
        assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == 4


class TestModelTrainingAndDeltas:
    """Tests for training across 5 conditions and computing deltas."""

    def test_synthetic_train_and_evaluate(self):
        np.random.seed(42)
        all_cols = UNIVERSAL_CORE_FEATURES + RETRIEVAL_FEATURES
        n_train = 40
        n_test = 20

        # Build synthetic data
        train_data = {col: np.random.randn(n_train) for col in all_cols}
        train_data["id"] = [f"train_{i}" for i in range(n_train)]
        train_data["source_dataset"] = ["halueval"] * 20 + ["fever"] * 20
        train_data["label"] = [0, 1] * (n_train // 2)
        train_df = pd.DataFrame(train_data)

        test_data = {col: np.random.randn(n_test) for col in all_cols}
        test_data["id"] = [f"test_{i}" for i in range(n_test)]
        test_data["source_dataset"] = ["halueval"] * 10 + ["fever"] * 10
        test_data["label"] = [0, 1] * (n_test // 2)
        test_df = pd.DataFrame(test_data)

        results_df, preds_df = train_and_evaluate_retrieval_models(train_df, test_df)

        # 5 conditions * 2 models = 10 rows
        assert len(results_df) == 10
        assert set(results_df["model"].unique()) == {"logistic_regression", "xgboost"}
        assert len(results_df["condition"].unique()) == 5

        # Check deltas exist
        for delta_col in [
            "delta_vs_internal_roc_auc",
            "delta_vs_universal_roc_auc",
            "delta_vs_internal_accuracy",
            "delta_vs_universal_accuracy",
        ]:
            assert delta_col in results_df.columns
            assert not results_df[delta_col].isna().any()

        # Check prediction counts: 10 configs * 20 test instances = 200 rows
        assert len(preds_df) == 200
        assert (preds_df["y_probability"] >= 0.0).all() and (preds_df["y_probability"] <= 1.0).all()
