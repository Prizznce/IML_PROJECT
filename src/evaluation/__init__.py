"""
Evaluation and ablation analysis module for hallucination detection.
"""

from src.evaluation.ablation_study import (
    ABLATION_CONDITIONS,
    REFERENCE_CONDITION_KEY,
    compute_ablation_deltas,
    run_ablation_study,
    validate_ablation_conditions,
)
from src.evaluation.cross_dataset import (
    CROSS_DATASET_CONFIGURATIONS,
    CROSS_DATASET_EXPERIMENTS,
    compute_transfer_deltas,
    get_leave_one_dataset_out_partitions,
    run_cross_dataset_study,
)
from src.evaluation.calibration import (
    compute_calibration_curve_and_metrics,
    evaluate_model_prob_metrics,
    split_training_into_model_and_calib,
    plot_comparative_reliability_diagram,
    run_calibration_study,
)
from src.evaluation.error_analysis import (
    PRIMARY_19_FEATURES,
    assign_error_categories,
    compute_confidence,
    compute_dataset_error_metrics,
    compute_error_category_statistics,
    compute_feature_error_effects,
    compute_model_disagreement,
    extract_confusion_examples,
    load_and_validate_error_analysis_data,
    run_error_analysis_pipeline,
)

__all__ = [
    "ABLATION_CONDITIONS",
    "REFERENCE_CONDITION_KEY",
    "compute_ablation_deltas",
    "run_ablation_study",
    "validate_ablation_conditions",
    "CROSS_DATASET_EXPERIMENTS",
    "CROSS_DATASET_CONFIGURATIONS",
    "compute_transfer_deltas",
    "get_leave_one_dataset_out_partitions",
    "run_cross_dataset_study",
    "compute_calibration_curve_and_metrics",
    "evaluate_model_prob_metrics",
    "split_training_into_model_and_calib",
    "plot_comparative_reliability_diagram",
    "run_calibration_study",
    "PRIMARY_19_FEATURES",
    "assign_error_categories",
    "compute_confidence",
    "compute_dataset_error_metrics",
    "compute_error_category_statistics",
    "compute_feature_error_effects",
    "compute_model_disagreement",
    "extract_confusion_examples",
    "load_and_validate_error_analysis_data",
    "run_error_analysis_pipeline",
]
