"""
Feature engineering and assembly module for hallucination detection.
"""

from src.features.build_universal_features import (
    METADATA_COLUMNS,
    INTERNAL_SIGNAL_FEATURES,
    SELF_CONSISTENCY_FEATURES,
    NLI_AGREEMENT_FEATURES,
    UNIVERSAL_CORE_FEATURES,
    FORBIDDEN_FEATURE_COLUMNS,
    compute_internal_log_transforms,
    build_universal_feature_dataframe,
    validate_universal_feature_dataframe,
    build_and_save_universal_features,
)

__all__ = [
    "METADATA_COLUMNS",
    "INTERNAL_SIGNAL_FEATURES",
    "SELF_CONSISTENCY_FEATURES",
    "NLI_AGREEMENT_FEATURES",
    "UNIVERSAL_CORE_FEATURES",
    "FORBIDDEN_FEATURE_COLUMNS",
    "compute_internal_log_transforms",
    "build_universal_feature_dataframe",
    "validate_universal_feature_dataframe",
    "build_and_save_universal_features",
]
