"""
Evidence retrieval module for hallucination detection.
"""

from src.retrieval.evidence_retrieval import (
    compute_cosine_similarity,
    rank_top_k,
    calculate_evidence_margin,
    calculate_retrieval_agreement,
    format_query_text,
    build_evidence_corpus,
    load_retrieval_model,
    extract_retrieval_features,
    run_evidence_retrieval_pilot,
)

__all__ = [
    "compute_cosine_similarity",
    "rank_top_k",
    "calculate_evidence_margin",
    "calculate_retrieval_agreement",
    "format_query_text",
    "build_evidence_corpus",
    "load_retrieval_model",
    "extract_retrieval_features",
    "run_evidence_retrieval_pilot",
]
