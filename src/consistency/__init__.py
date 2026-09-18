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
    run_self_consistency_full,
)
from src.consistency.nli_agreement import (
    load_nli_model,
    get_nli_label_indices,
    compute_softmax_probabilities,
    aggregate_directional_probabilities,
    calculate_nli_disagreement,
    extract_pairwise_nli_predictions,
    aggregate_example_nli_signals,
    run_nli_agreement_pilot,
    run_nli_agreement_full,
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
    "run_self_consistency_full",
    "load_nli_model",
    "get_nli_label_indices",
    "compute_softmax_probabilities",
    "aggregate_directional_probabilities",
    "calculate_nli_disagreement",
    "extract_pairwise_nli_predictions",
    "aggregate_example_nli_signals",
    "run_nli_agreement_pilot",
    "run_nli_agreement_full",
]
