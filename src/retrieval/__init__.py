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
from src.retrieval.retrieval_augmented_experiment import (
    CONDITION_FEATURE_MAP,
    RETRIEVAL_FEATURES,
    build_dataset_evidence_corpora,
    extract_dense_retrieval_features,
    run_retrieval_augmented_experiment,
    train_and_evaluate_retrieval_models,
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
    "RETRIEVAL_FEATURES",
    "CONDITION_FEATURE_MAP",
    "build_dataset_evidence_corpora",
    "extract_dense_retrieval_features",
    "train_and_evaluate_retrieval_models",
    "run_retrieval_augmented_experiment",
]
