"""
Streamlit demonstration application package for LLM hallucination detection.
"""

from app.inference import (
    UNIVERSAL_CORE_FEATURES,
    run_live_inference,
    get_trained_classifiers,
    assemble_feature_vector,
    format_risk_level,
    predict_hallucination_risk,
)

__all__ = [
    "UNIVERSAL_CORE_FEATURES",
    "run_live_inference",
    "get_trained_classifiers",
    "assemble_feature_vector",
    "format_risk_level",
    "predict_hallucination_risk",
]
