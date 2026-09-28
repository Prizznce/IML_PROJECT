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
    get_token_confidence_band,
    format_token_details,
    compute_feature_contributions,
    compute_reference_evidence_agreement,
)

__all__ = [
    "UNIVERSAL_CORE_FEATURES",
    "run_live_inference",
    "get_trained_classifiers",
    "assemble_feature_vector",
    "format_risk_level",
    "predict_hallucination_risk",
    "get_token_confidence_band",
    "format_token_details",
    "compute_feature_contributions",
    "compute_reference_evidence_agreement",
]
