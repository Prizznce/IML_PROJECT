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
from src.training.calibration import (
    compute_calibration_data,
    plot_reliability_diagram,
    run_primary_calibration_analysis,
)
from src.training.cross_dataset import (
    get_leave_one_out_partitions,
    evaluate_transfer_fold,
    run_cross_dataset_evaluation,
)
from src.training.combined_models import (
    UNIVERSAL_MODEL_FEATURES,
    prepare_combined_feature_dataframe,
    evaluate_by_source_dataset,
    compute_baseline_comparison,
    run_combined_experiment,
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
    "compute_calibration_data",
    "plot_reliability_diagram",
    "run_primary_calibration_analysis",
    "get_leave_one_out_partitions",
    "evaluate_transfer_fold",
    "run_cross_dataset_evaluation",
    "UNIVERSAL_MODEL_FEATURES",
    "prepare_combined_feature_dataframe",
    "evaluate_by_source_dataset",
    "compute_baseline_comparison",
    "run_combined_experiment",
]


