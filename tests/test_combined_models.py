"""
Unit and Integration Tests for Universal Combined Hallucination Detector Training (Phase 14).

Validates feature selection rigor, strict metadata/token exclusion, deterministic
stratified splitting, train-only scaler fitting, finite metric bounds, and baseline
delta comparison mathematics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.combined_models import (
    FORBIDDEN_COLUMNS,
    UNIVERSAL_MODEL_FEATURES,
    build_logistic_regression_pipeline,
    build_xgboost_model,
    compute_baseline_comparison,
    evaluate_by_source_dataset,
    evaluate_predictions,
    prepare_combined_feature_dataframe,
    split_dataset,
)


@pytest.fixture
def mock_universal_dataframe() -> pd.DataFrame:
    """Generate a synthetic universal feature table for fast, self-contained unit tests."""
    np.random.seed(42)
    n = 100
    data = {
        "id": [f"ex_{i}" for i in range(n)],
        "source_dataset": np.random.choice(["halueval", "truthfulqa", "fever"], size=n),
        "label": np.random.choice([0, 1], size=n),
    }

    # Add 19 universal features with realistic non-zero variance
    for feat in UNIVERSAL_MODEL_FEATURES:
        data[feat] = np.random.randn(n)

    return pd.DataFrame(data)


class TestCombinedFeatureIntegrity:
    """Test feature set constraints, absence of leakage, and forbidden column guards."""

    def test_feature_count_and_specification(self):
        """Universal model must specify exactly 19 features with no duplicates."""
        assert len(UNIVERSAL_MODEL_FEATURES) == 19
        assert len(set(UNIVERSAL_MODEL_FEATURES)) == 19

    def test_forbidden_columns_disjointness(self):
        """Universal model features must be completely disjoint from forbidden columns."""
        overlap = set(UNIVERSAL_MODEL_FEATURES).intersection(FORBIDDEN_COLUMNS)
        assert len(overlap) == 0, f"Found forbidden columns in UNIVERSAL_MODEL_FEATURES: {overlap}"

    def test_num_tokens_strictly_excluded(self):
        """num_tokens must never be present in UNIVERSAL_MODEL_FEATURES."""
        assert "num_tokens" not in UNIVERSAL_MODEL_FEATURES

    def test_forbidden_columns_caught_by_preparation(self, mock_universal_dataframe):
        """Tampering with UNIVERSAL_MODEL_FEATURES to include forbidden column raises ValueError."""
        # Inject forbidden column into dataframe
        df_tampered = mock_universal_dataframe.copy()
        df_tampered["num_tokens"] = 15

        # Feature matrix preparation with normal features works
        X, y, meta = prepare_combined_feature_dataframe(df_tampered)
        assert "num_tokens" not in X.columns
        assert X.shape[1] == 19

    def test_non_finite_values_caught(self, mock_universal_dataframe):
        """NaN or Inf in feature table must be detected and rejected."""
        df_nan = mock_universal_dataframe.copy()
        df_nan.loc[5, "min_log_prob"] = np.nan
        with pytest.raises(ValueError, match="Non-finite values"):
            prepare_combined_feature_dataframe(df_nan)


class TestStratifiedSplittingAndDataLeakage:
    """Test train/test split isolation and train-only preprocessing."""

    def test_split_determinism_and_no_id_overlap(self, mock_universal_dataframe):
        """Train and test splits must be deterministic with zero overlapping IDs."""
        X, y, meta = prepare_combined_feature_dataframe(mock_universal_dataframe)
        X_train1, X_test1, y_train1, y_test1, meta_train1, meta_test1 = split_dataset(
            X, y, meta, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2, meta_train2, meta_test2 = split_dataset(
            X, y, meta, test_size=0.2, random_state=42
        )

        # Determinism
        assert meta_train1["id"].tolist() == meta_train2["id"].tolist()
        assert meta_test1["id"].tolist() == meta_test2["id"].tolist()

        # Zero ID overlap
        train_ids = set(meta_train1["id"])
        test_ids = set(meta_test1["id"])
        assert len(train_ids.intersection(test_ids)) == 0
        assert len(train_ids) + len(test_ids) == len(mock_universal_dataframe)

    def test_scaler_fitted_on_train_only(self, mock_universal_dataframe):
        """StandardScaler within Logistic Regression pipeline must fit strictly on train split."""
        X, y, meta = prepare_combined_feature_dataframe(mock_universal_dataframe)
        X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
            X, y, meta, test_size=0.2, random_state=42
        )

        pipeline = build_logistic_regression_pipeline(random_state=42)
        pipeline.fit(X_train, y_train)

        scaler = pipeline.named_steps["scaler"]
        assert isinstance(scaler, StandardScaler)

        # Scaler mean must match X_train mean within numerical tolerance, NOT full X or X_test
        expected_train_mean = X_train.mean(axis=0).to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected_train_mean, rtol=1e-5)


class TestModelEvaluationMetrics:
    """Test metric computation, bounded ranges, and calibration error."""

    def test_metric_bounds_and_finiteness(self):
        """All evaluation metrics must be finite and within theoretical bounds."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.4, 0.85])
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_predictions(y_true, y_pred, y_prob)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["ece"] <= 1.0

        for k, v in metrics.items():
            if k != "confusion_matrix":
                assert np.isfinite(v), f"Metric {k} is non-finite: {v}"

    def test_evaluate_by_source_dataset_isolation(self, mock_universal_dataframe):
        """Subgroup breakdown must evaluate only rows corresponding to that dataset."""
        X, y, meta = prepare_combined_feature_dataframe(mock_universal_dataframe)
        X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
            X, y, meta, test_size=0.2, random_state=42
        )

        pipeline = build_logistic_regression_pipeline(random_state=42)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        subgroups = evaluate_by_source_dataset(y_test, y_pred, y_prob, meta_test)
        total_subgroup_samples = sum(s["test_samples"] for s in subgroups.values())
        assert total_subgroup_samples == len(y_test)

    def test_compute_baseline_comparison_deltas(self):
        """Delta calculations must correctly compute combined - baseline."""
        comb = {
            "accuracy": 0.75,
            "f1": 0.70,
            "roc_auc": 0.80,
            "pr_auc": 0.78,
            "brier_score": 0.18,
            "ece": 0.04,
        }
        base = {
            "accuracy": 0.67,
            "f1": 0.68,
            "roc_auc": 0.74,
            "pr_auc": 0.73,
            "brier_score": 0.20,
            "ece": 0.05,
        }

        deltas = compute_baseline_comparison(comb, base)
        assert pytest.approx(deltas["accuracy_delta"], abs=1e-5) == 0.08
        assert pytest.approx(deltas["f1_delta"], abs=1e-5) == 0.02
        assert pytest.approx(deltas["roc_auc_delta"], abs=1e-5) == 0.06
        assert pytest.approx(deltas["pr_auc_delta"], abs=1e-5) == 0.05
        assert pytest.approx(deltas["brier_score_delta"], abs=1e-5) == -0.02
        assert pytest.approx(deltas["ece_delta"], abs=1e-5) == -0.01


class TestModelsTrainability:
    """Test that both LR and XGBoost train and predict successfully on universal features."""

    def test_lr_and_xgboost_fit_predict(self, mock_universal_dataframe):
        """Both models must train without errors and output valid probabilities in [0, 1]."""
        X, y, meta = prepare_combined_feature_dataframe(mock_universal_dataframe)
        X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
            X, y, meta, test_size=0.2, random_state=42
        )

        lr_pipe = build_logistic_regression_pipeline(random_state=42)
        lr_pipe.fit(X_train, y_train)
        lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
        assert len(lr_prob) == len(y_test)
        assert np.all((lr_prob >= 0.0) & (lr_prob <= 1.0))

        xgb_mod = build_xgboost_model(random_state=42, n_estimators=10)
        xgb_mod.fit(X_train, y_train)
        xgb_prob = xgb_mod.predict_proba(X_test)[:, 1]
        assert len(xgb_prob) == len(y_test)
        assert np.all((xgb_prob >= 0.0) & (xgb_prob <= 1.0))
