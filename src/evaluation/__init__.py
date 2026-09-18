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

__all__ = [
    "ABLATION_CONDITIONS",
    "REFERENCE_CONDITION_KEY",
    "compute_ablation_deltas",
    "run_ablation_study",
    "validate_ablation_conditions",
]
