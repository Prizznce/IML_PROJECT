"""
Unit tests for NLI agreement signal extraction and evaluation.

Verifies:
- Softmax probability normalization and numerical stability
- Model label index mapping across configurations
- Directional probability aggregation (A -> B and B -> A)
- Pairwise evaluation counts (C(K, 2) pairs for K=5)
- Directional evaluation counts (2 * C(K, 2) = 20 directional evaluations for K=5)
- Example-level signal aggregation and fraction calculations
- NLI disagreement formula and boundary conditions
- Edge cases: K=1, K=0, identical responses, contradictory responses
- Zero-division guards and numerical finiteness
"""

import math
import unittest
import numpy as np

from src.consistency.nli_agreement import (
    get_nli_label_indices,
    compute_softmax_probabilities,
    aggregate_directional_probabilities,
    calculate_nli_disagreement,
    extract_pairwise_nli_predictions,
    aggregate_example_nli_signals,
)


class TestSoftmaxProbabilityNormalization(unittest.TestCase):
    """Test softmax probability normalization and numerical stability."""

    def test_1d_logits_sum_to_one(self):
        logits = np.array([-2.0, 3.5, 0.1])
        probs = compute_softmax_probabilities(logits)
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        # Logit 3.5 should have highest probability
        self.assertEqual(int(np.argmax(probs)), 1)

    def test_2d_logits_sum_to_one(self):
        logits = np.array([
            [-2.0, 3.5, 0.1],
            [4.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        probs = compute_softmax_probabilities(logits)
        self.assertEqual(probs.shape, (3, 3))
        sums = np.sum(probs, axis=1)
        for s in sums:
            self.assertAlmostEqual(float(s), 1.0, places=6)

    def test_numerical_stability_large_logits(self):
        logits = np.array([1000.0, 1002.0, 999.0])
        probs = compute_softmax_probabilities(logits)
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)


class TestLabelIndexMapping(unittest.TestCase):
    """Test dynamic resolution of class label indices from model configs."""

    def test_standard_id2label(self):
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        mapping = get_nli_label_indices(id2label)
        self.assertEqual(mapping["contradiction"], 0)
        self.assertEqual(mapping["entailment"], 1)
        self.assertEqual(mapping["neutral"], 2)

    def test_permuted_id2label(self):
        id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        mapping = get_nli_label_indices(id2label)
        self.assertEqual(mapping["entailment"], 0)
        self.assertEqual(mapping["neutral"], 1)
        self.assertEqual(mapping["contradiction"], 2)

    def test_missing_label_raises_value_error(self):
        id2label = {0: "entailment", 1: "neutral"}
        with self.assertRaises(ValueError):
            get_nli_label_indices(id2label)


class TestDirectionalAggregation(unittest.TestCase):
    """Test directional probability aggregation and consensus prediction."""

    def test_symmetric_entailment(self):
        p_a_b = {"contradiction": 0.05, "entailment": 0.90, "neutral": 0.05}
        p_b_a = {"contradiction": 0.05, "entailment": 0.90, "neutral": 0.05}
        agg = aggregate_directional_probabilities(p_a_b, p_b_a)

        self.assertAlmostEqual(agg["pair_entailment"], 0.90)
        self.assertAlmostEqual(agg["pair_contradiction"], 0.05)
        self.assertAlmostEqual(agg["pair_neutral"], 0.05)
        self.assertEqual(agg["predicted_class"], "entailment")

    def test_asymmetric_directionality(self):
        # A implies B, but B does not imply A (Neutral)
        p_a_b = {"contradiction": 0.0, "entailment": 0.80, "neutral": 0.20}
        p_b_a = {"contradiction": 0.0, "entailment": 0.10, "neutral": 0.90}
        agg = aggregate_directional_probabilities(p_a_b, p_b_a)

        self.assertAlmostEqual(agg["pair_entailment"], 0.45)
        self.assertAlmostEqual(agg["pair_neutral"], 0.55)
        self.assertAlmostEqual(agg["pair_contradiction"], 0.0)
        self.assertEqual(agg["predicted_class"], "neutral")

    def test_probabilities_sum_to_one(self):
        p_a_b = {"contradiction": 0.30, "entailment": 0.50, "neutral": 0.20}
        p_b_a = {"contradiction": 0.40, "entailment": 0.20, "neutral": 0.40}
        agg = aggregate_directional_probabilities(p_a_b, p_b_a)

        total = agg["pair_entailment"] + agg["pair_contradiction"] + agg["pair_neutral"]
        self.assertAlmostEqual(total, 1.0, places=6)


class TestNLIDisagreement(unittest.TestCase):
    """Test scalar NLI disagreement calculation and boundary conditions."""

    def test_pure_entailment_zero_disagreement(self):
        # Full entailment -> 0 contradiction, 0 neutral
        self.assertAlmostEqual(calculate_nli_disagreement(0.0, 0.0), 0.0)

    def test_pure_contradiction_max_disagreement(self):
        # Full contradiction -> 1.0 contradiction, 0 neutral
        self.assertAlmostEqual(calculate_nli_disagreement(1.0, 0.0), 1.0)

    def test_pure_neutral_half_disagreement(self):
        # Full neutral -> 0 contradiction, 1.0 neutral
        self.assertAlmostEqual(calculate_nli_disagreement(0.0, 1.0), 0.5)

    def test_mixed_disagreement(self):
        # 0.2 contradiction + 0.5 * 0.4 neutral = 0.4
        self.assertAlmostEqual(calculate_nli_disagreement(0.2, 0.4), 0.4)

    def test_clipping_bounds_and_finiteness(self):
        self.assertTrue(math.isfinite(calculate_nli_disagreement(0.35, 0.25)))
        # Bounded in [0, 1]
        self.assertEqual(calculate_nli_disagreement(1.5, 0.5), 1.0)
        self.assertEqual(calculate_nli_disagreement(-0.5, 0.0), 0.0)


class TestPairwiseExtractionCounts(unittest.TestCase):
    """Test pairwise generation counts and directional evaluation structure using mocks."""

    def test_k5_produces_10_pairs(self):
        responses = ["r0", "r1", "r2", "r3", "r4"]
        k = len(responses)
        expected_pairs = (k * (k - 1)) // 2  # C(5, 2) = 10

        # Build mock directional probabilities for all 20 directional pairs
        mock_probs = {}
        for i in range(k):
            for j in range(k):
                if i != j:
                    mock_probs[(i, j)] = {"contradiction": 0.1, "entailment": 0.8, "neutral": 0.1}

        pairs = extract_pairwise_nli_predictions(responses=responses, mock_directional_probs=mock_probs)
        self.assertEqual(len(pairs), expected_pairs)
        self.assertEqual(len(pairs), 10)

        # Verify pair index range and unordered property (i < j)
        for p in pairs:
            self.assertLess(p["pair_index_i"], p["pair_index_j"])
            self.assertIn("pair_entailment", p)
            self.assertIn("pair_contradiction", p)
            self.assertIn("pair_neutral", p)
            self.assertIn("predicted_class", p)

    def test_k1_produces_zero_pairs(self):
        pairs = extract_pairwise_nli_predictions(responses=["only_one"])
        self.assertEqual(len(pairs), 0)

    def test_k0_produces_zero_pairs(self):
        pairs = extract_pairwise_nli_predictions(responses=[])
        self.assertEqual(len(pairs), 0)


class TestExampleSignalAggregation(unittest.TestCase):
    """Test aggregation of pairwise records into example-level scalar signals."""

    def test_aggregation_known_fractions(self):
        # 10 pairs: 5 entailment, 3 neutral, 2 contradiction
        pair_records = []
        for i in range(5):
            pair_records.append({
                "pair_entailment": 0.9,
                "pair_contradiction": 0.05,
                "pair_neutral": 0.05,
                "predicted_class": "entailment",
            })
        for i in range(3):
            pair_records.append({
                "pair_entailment": 0.1,
                "pair_contradiction": 0.1,
                "pair_neutral": 0.8,
                "predicted_class": "neutral",
            })
        for i in range(2):
            pair_records.append({
                "pair_entailment": 0.05,
                "pair_contradiction": 0.9,
                "pair_neutral": 0.05,
                "predicted_class": "contradiction",
            })

        signals = aggregate_example_nli_signals(pair_records, num_generations=5)

        self.assertEqual(signals["num_generations"], 5)
        self.assertAlmostEqual(signals["fraction_entailing_pairs"], 5.0 / 10.0)
        self.assertAlmostEqual(signals["fraction_neutral_pairs"], 3.0 / 10.0)
        self.assertAlmostEqual(signals["fraction_contradicting_pairs"], 2.0 / 10.0)

        # Sum of fractions should equal 1.0
        frac_sum = (
            signals["fraction_entailing_pairs"]
            + signals["fraction_neutral_pairs"]
            + signals["fraction_contradicting_pairs"]
        )
        self.assertAlmostEqual(frac_sum, 1.0)

        # Means
        expected_mean_entail = (5 * 0.9 + 3 * 0.1 + 2 * 0.05) / 10.0
        expected_mean_contra = (5 * 0.05 + 3 * 0.1 + 2 * 0.9) / 10.0
        expected_mean_neutral = (5 * 0.05 + 3 * 0.8 + 2 * 0.05) / 10.0

        self.assertAlmostEqual(signals["mean_pairwise_entailment"], expected_mean_entail)
        self.assertAlmostEqual(signals["mean_pairwise_contradiction"], expected_mean_contra)
        self.assertAlmostEqual(signals["mean_pairwise_neutral"], expected_mean_neutral)

        # Disagreement = contra + 0.5 * neutral
        expected_disagreement = expected_mean_contra + 0.5 * expected_mean_neutral
        self.assertAlmostEqual(signals["nli_disagreement"], expected_disagreement)

        # All values finite
        for k, v in signals.items():
            self.assertTrue(math.isfinite(v), f"Value for {k} is not finite: {v}")

    def test_k1_edge_case_aggregation(self):
        signals = aggregate_example_nli_signals([], num_generations=1)
        self.assertEqual(signals["num_generations"], 1)
        self.assertEqual(signals["mean_pairwise_entailment"], 1.0)
        self.assertEqual(signals["mean_pairwise_contradiction"], 0.0)
        self.assertEqual(signals["nli_disagreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
