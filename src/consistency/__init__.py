"""
Self-consistency module for hallucination detection.
"""

from src.consistency.self_consistency import (
    normalize_response,
    calculate_exact_match_agreement,
    calculate_unique_response_ratio,
    calculate_majority_response_fraction,
    calculate_pairwise_semantic_similarity,
    calculate_generation_disagreement,
    extract_self_consistency_signals,
    generate_stochastic_responses,
    run_self_consistency_pilot,
)

__all__ = [
    "normalize_response",
    "calculate_exact_match_agreement",
    "calculate_unique_response_ratio",
    "calculate_majority_response_fraction",
    "calculate_pairwise_semantic_similarity",
    "calculate_generation_disagreement",
    "extract_self_consistency_signals",
    "generate_stochastic_responses",
    "run_self_consistency_pilot",
]
