"""
Training and baseline modeling package for hallucination detection.
"""

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

__all__ = [
    "PRIMARY_SIGNAL_FEATURES",
    "FORBIDDEN_COLUMNS",
    "prepare_feature_dataframe",
    "split_dataset",
    "build_logistic_regression_pipeline",
    "build_xgboost_model",
    "compute_ece",
    "evaluate_predictions",
    "run_baseline_experiment",
]
