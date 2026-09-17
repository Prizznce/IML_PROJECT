"""
Unit tests for self-consistency signal extraction and evaluation.

Verifies:
- Deterministic response normalization
- Exact-match agreement rate
- Unique-response ratio
- Majority-response fraction
- Pairwise semantic similarity (mean, min, max, std)
- Generation disagreement calculation
- Edge cases: identical responses, distinct responses, K=1, K=0, empty inputs
- Division-by-zero guards and numerical finiteness
"""

import math
import unittest
import numpy as np

from src.consistency.self_consistency import (
    normalize_response,
    calculate_exact_match_agreement,
    calculate_unique_response_ratio,
    calculate_majority_response_fraction,
    calculate_pairwise_semantic_similarity,
    calculate_generation_disagreement,
    extract_self_consistency_signals,
)


class TestResponseNormalization(unittest.TestCase):
    """Test deterministic response normalization."""

    def test_lowercasing_and_whitespace(self):
        self.assertEqual(normalize_response("  Paris  "), "paris")
        self.assertEqual(normalize_response("PARIS"), "paris")
        self.assertEqual(normalize_response("pArIs"), "paris")

    def test_collapse_whitespace(self):
        self.assertEqual(normalize_response("Paris   in   France"), "paris in france")
        self.assertEqual(normalize_response("Paris\t\nin\nFrance"), "paris in france")

    def test_punctuation_handling(self):
        self.assertEqual(normalize_response("Paris."), "paris")
        self.assertEqual(normalize_response("Paris!"), "paris")
        self.assertEqual(normalize_response("Paris, France?"), "paris france")
        self.assertEqual(normalize_response('"Paris"'), "paris")
        self.assertEqual(normalize_response("'Paris'"), "paris")
        self.assertEqual(normalize_response("`Paris`"), "paris")

    def test_apostrophes_and_contractions(self):
        # Contractions: apostrophe removed, letters kept intact
        self.assertEqual(normalize_response("It's true."), "its true")
        self.assertEqual(normalize_response("don't know"), "dont know")

    def test_empty_and_none_inputs(self):
        self.assertEqual(normalize_response(""), "")
        self.assertEqual(normalize_response("   "), "")
        self.assertEqual(normalize_response(None), "")
        self.assertEqual(normalize_response("..."), "")

    def test_deterministic_idempotence(self):
        sample = "  The quick brown fox, jumps over 10 lazy dogs!  "
        norm1 = normalize_response(sample)
        norm2 = normalize_response(norm1)
        self.assertEqual(norm1, norm2)


class TestDiscreteAgreementMetrics(unittest.TestCase):
    """Test exact-match agreement, unique ratio, and majority fraction."""

    def test_all_identical_responses(self):
        responses = ["Paris", "paris.", " PARIS! ", '"Paris"', "  paris  "]
        self.assertAlmostEqual(calculate_exact_match_agreement(responses), 1.0)
        self.assertAlmostEqual(calculate_unique_response_ratio(responses), 1.0 / 5.0)
        self.assertAlmostEqual(calculate_majority_response_fraction(responses), 1.0)

    def test_all_distinct_responses(self):
        responses = ["Paris", "London", "Berlin", "Rome", "Madrid"]
        self.assertAlmostEqual(calculate_exact_match_agreement(responses), 0.0)
        self.assertAlmostEqual(calculate_unique_response_ratio(responses), 1.0)
        self.assertAlmostEqual(calculate_majority_response_fraction(responses), 1.0 / 5.0)

    def test_partial_agreement(self):
        # 3 "paris", 2 "london" -> C(5, 2) = 10 pairs
        # Matching pairs: C(3, 2) + C(2, 2) = 3 + 1 = 4 pairs
        responses = ["Paris", "paris.", "Paris!", "London", "london!"]
        self.assertAlmostEqual(calculate_exact_match_agreement(responses), 4.0 / 10.0)
        self.assertAlmostEqual(calculate_unique_response_ratio(responses), 2.0 / 5.0)
        self.assertAlmostEqual(calculate_majority_response_fraction(responses), 3.0 / 5.0)

    def test_single_response_edge_case(self):
        # K=1
        responses = ["Paris"]
        self.assertEqual(calculate_exact_match_agreement(responses), 1.0)
        self.assertEqual(calculate_unique_response_ratio(responses), 1.0)
        self.assertEqual(calculate_majority_response_fraction(responses), 1.0)

    def test_empty_responses_edge_case(self):
        # K=0
        responses = []
        self.assertEqual(calculate_exact_match_agreement(responses), 1.0)
        self.assertEqual(calculate_unique_response_ratio(responses), 0.0)
        self.assertEqual(calculate_majority_response_fraction(responses), 0.0)


class TestPairwiseSemanticSimilarity(unittest.TestCase):
    """Test pairwise cosine similarity and edge cases with precomputed/mock embeddings."""

    def test_identical_embeddings(self):
        # 4 identical 4-dimensional vectors
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.tile(vec, (4, 1))
        responses = ["r1", "r2", "r3", "r4"]

        stats = calculate_pairwise_semantic_similarity(responses=responses, embeddings=embeddings)
        self.assertAlmostEqual(stats["mean_pairwise_similarity"], 1.0, places=5)
        self.assertAlmostEqual(stats["min_pairwise_similarity"], 1.0, places=5)
        self.assertAlmostEqual(stats["max_pairwise_similarity"], 1.0, places=5)
        self.assertAlmostEqual(stats["pairwise_similarity_std"], 0.0, places=5)

    def test_orthogonal_embeddings(self):
        # 2 orthogonal vectors
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        responses = ["r1", "r2"]

        stats = calculate_pairwise_semantic_similarity(responses=responses, embeddings=embeddings)
        self.assertAlmostEqual(stats["mean_pairwise_similarity"], 0.0, places=5)
        self.assertAlmostEqual(stats["min_pairwise_similarity"], 0.0, places=5)
        self.assertAlmostEqual(stats["max_pairwise_similarity"], 0.0, places=5)
        self.assertAlmostEqual(stats["pairwise_similarity_std"], 0.0, places=5)

    def test_known_pairwise_values(self):
        # 3 vectors: v1=[1, 0], v2=[1, 0], v3=[0, 1]
        # Pairwise cos: (0, 1)=1.0, (0, 2)=0.0, (1, 2)=0.0
        # Mean = 1/3, Min = 0.0, Max = 1.0
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        responses = ["r1", "r2", "r3"]

        stats = calculate_pairwise_semantic_similarity(responses=responses, embeddings=embeddings)
        self.assertAlmostEqual(stats["mean_pairwise_similarity"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(stats["min_pairwise_similarity"], 0.0, places=5)
        self.assertAlmostEqual(stats["max_pairwise_similarity"], 1.0, places=5)
        expected_std = np.std([1.0, 0.0, 0.0])
        self.assertAlmostEqual(stats["pairwise_similarity_std"], expected_std, places=5)

    def test_single_item_edge_case(self):
        embeddings = np.array([[1.0, 0.5]], dtype=np.float32)
        responses = ["only_one"]
        stats = calculate_pairwise_semantic_similarity(responses=responses, embeddings=embeddings)
        self.assertEqual(stats["mean_pairwise_similarity"], 1.0)
        self.assertEqual(stats["min_pairwise_similarity"], 1.0)
        self.assertEqual(stats["max_pairwise_similarity"], 1.0)
        self.assertEqual(stats["pairwise_similarity_std"], 0.0)

    def test_missing_model_and_embeddings_raises(self):
        with self.assertRaises(ValueError):
            calculate_pairwise_semantic_similarity(responses=["a", "b"])


class TestGenerationDisagreement(unittest.TestCase):
    """Test scalar generation disagreement score calculation."""

    def test_disagreement_formula(self):
        self.assertAlmostEqual(calculate_generation_disagreement(1.0), 0.0)
        self.assertAlmostEqual(calculate_generation_disagreement(0.8), 0.2)
        self.assertAlmostEqual(calculate_generation_disagreement(0.0), 1.0)
        self.assertAlmostEqual(calculate_generation_disagreement(-0.5), 1.5)

    def test_finite_output(self):
        val = calculate_generation_disagreement(0.723)
        self.assertTrue(math.isfinite(val))


class TestExtractSelfConsistencySignals(unittest.TestCase):
    """Test end-to-end signal extraction dictionary output and schema."""

    def test_extract_signals_with_mock_embeddings(self):
        responses = ["Paris", "paris", "London"]
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        signals = extract_self_consistency_signals(responses=responses, embeddings=embeddings)

        expected_keys = {
            "num_generations",
            "exact_match_agreement",
            "unique_response_ratio",
            "majority_response_fraction",
            "mean_pairwise_similarity",
            "min_pairwise_similarity",
            "max_pairwise_similarity",
            "pairwise_similarity_std",
            "generation_disagreement",
        }
        self.assertEqual(set(signals.keys()), expected_keys)
        self.assertEqual(signals["num_generations"], 3)

        # 1 match out of 3 pairs = 1/3
        self.assertAlmostEqual(signals["exact_match_agreement"], 1.0 / 3.0, places=5)
        # 2 unique out of 3 = 2/3
        self.assertAlmostEqual(signals["unique_response_ratio"], 2.0 / 3.0, places=5)
        # majority count = 2 out of 3 = 2/3
        self.assertAlmostEqual(signals["majority_response_fraction"], 2.0 / 3.0, places=5)

        # Pairwise similarities: (0, 1)=1.0, (0, 2)=0.0, (1, 2)=0.0 -> mean = 1/3
        self.assertAlmostEqual(signals["mean_pairwise_similarity"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(signals["generation_disagreement"], 1.0 - 1.0 / 3.0, places=5)

        # All values should be finite
        for k, v in signals.items():
            self.assertTrue(math.isfinite(v), f"Value for {k} is not finite: {v}")


if __name__ == "__main__":
    unittest.main()
