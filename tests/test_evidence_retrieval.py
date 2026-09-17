"""
Unit tests for evidence retrieval and retrieval-agreement signal extraction.

Verifies:
- Cosine similarity computation across query and corpus embeddings
- Top-k ranking and deterministic tie-breaking
- Evidence margin calculation
- Retrieval-agreement score formulation and boundary conditions
- Empty corpus and fewer-than-k evidence items handling
- Zero-division guards and numerical finiteness
- Query formatting across HaluEval, FEVER, and TruthfulQA
- Feature extraction schema with mocked embeddings (zero external network requirement)
"""

import math
import unittest
import numpy as np

from src.retrieval.evidence_retrieval import (
    compute_cosine_similarity,
    rank_top_k,
    calculate_evidence_margin,
    calculate_retrieval_agreement,
    format_query_text,
    extract_retrieval_features,
)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity calculations and boundary conditions."""

    def test_unit_vectors(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        corpus = np.array([
            [1.0, 0.0],   # identical
            [0.0, 1.0],   # orthogonal
            [-1.0, 0.0],  # opposing
        ], dtype=np.float32)

        sims = compute_cosine_similarity(query, corpus)
        self.assertEqual(len(sims), 3)
        self.assertAlmostEqual(float(sims[0]), 1.0, places=5)
        self.assertAlmostEqual(float(sims[1]), 0.0, places=5)
        self.assertAlmostEqual(float(sims[2]), -1.0, places=5)

    def test_unnormalized_vectors(self):
        query = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        corpus = np.array([
            [6.0, 8.0],    # collinear, same direction -> 1.0
            [-3.0, -4.0],  # collinear, opposite direction -> -1.0
        ], dtype=np.float32)

        sims = compute_cosine_similarity(query, corpus)
        self.assertAlmostEqual(float(sims[0]), 1.0, places=5)
        self.assertAlmostEqual(float(sims[1]), -1.0, places=5)

    def test_zero_vector_guard(self):
        query = np.array([0.0, 0.0], dtype=np.float32)
        corpus = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = compute_cosine_similarity(query, corpus)
        self.assertTrue(np.all(sims == 0.0))

    def test_empty_corpus(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        corpus = np.empty((0, 2), dtype=np.float32)
        sims = compute_cosine_similarity(query, corpus)
        self.assertEqual(len(sims), 0)


class TestTopKRanking(unittest.TestCase):
    """Test top-k ranking, descending order, and deterministic tie-breaking."""

    def test_ranking_descending(self):
        sims = np.array([0.2, 0.9, 0.7, 0.4], dtype=np.float32)
        indices, scores = rank_top_k(sims, k=3)
        self.assertEqual(indices, [1, 2, 3])
        self.assertAlmostEqual(scores[0], 0.9, places=5)
        self.assertAlmostEqual(scores[1], 0.7, places=5)
        self.assertAlmostEqual(scores[2], 0.4, places=5)

    def test_deterministic_tie_breaking(self):
        # Index 0 and Index 2 have identical scores (0.8)
        sims = np.array([0.8, 0.5, 0.8, 0.1], dtype=np.float32)
        indices, scores = rank_top_k(sims, k=2)
        self.assertEqual(indices, [0, 2])
        self.assertAlmostEqual(scores[0], 0.8, places=5)
        self.assertAlmostEqual(scores[1], 0.8, places=5)

    def test_fewer_than_k_items(self):
        sims = np.array([0.5, 0.8], dtype=np.float32)
        indices, scores = rank_top_k(sims, k=5)
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices, [1, 0])

    def test_empty_similarities(self):
        sims = np.array([], dtype=np.float32)
        indices, scores = rank_top_k(sims, k=3)
        self.assertEqual(indices, [])
        self.assertEqual(scores, [])


class TestEvidenceMargin(unittest.TestCase):
    """Test evidence margin calculation and edge cases."""

    def test_distinct_top_two(self):
        margin = calculate_evidence_margin([0.85, 0.60, 0.40])
        self.assertAlmostEqual(margin, 0.25, places=5)

    def test_tied_top_two(self):
        margin = calculate_evidence_margin([0.70, 0.70, 0.50])
        self.assertAlmostEqual(margin, 0.0, places=5)

    def test_single_item(self):
        margin = calculate_evidence_margin([0.90])
        self.assertEqual(margin, 0.0)

    def test_empty_list(self):
        margin = calculate_evidence_margin([])
        self.assertEqual(margin, 0.0)


class TestRetrievalAgreement(unittest.TestCase):
    """Test retrieval agreement formula, boundaries, and robustness."""

    def test_perfect_dual_agreement(self):
        score = calculate_retrieval_agreement(1.0, 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_relevant_query_unsupported_response(self):
        score = calculate_retrieval_agreement(0.85, 0.0)
        self.assertAlmostEqual(score, 0.0)

    def test_negative_response_similarity(self):
        score = calculate_retrieval_agreement(0.85, -0.4)
        self.assertAlmostEqual(score, 0.0)

    def test_irrelevant_query_similarity(self):
        score = calculate_retrieval_agreement(0.0, 0.9)
        self.assertAlmostEqual(score, 0.0)

    def test_intermediate_values(self):
        # 0.6 * 0.5 = 0.3
        score = calculate_retrieval_agreement(0.6, 0.5)
        self.assertAlmostEqual(score, 0.3, places=5)

    def test_finite_and_bounded(self):
        score = calculate_retrieval_agreement(0.723, 0.814)
        self.assertTrue(math.isfinite(score))
        self.assertTrue(0.0 <= score <= 1.0)


class TestQueryFormatting(unittest.TestCase):
    """Test query representation formatting by dataset provenance."""

    def test_halueval_uses_prompt(self):
        q = format_query_text(prompt="What team won?", original_response="Patriots", source_dataset="halueval")
        self.assertEqual(q, "What team won?")

    def test_truthfulqa_uses_prompt(self):
        q = format_query_text(prompt="Is the earth flat?", original_response="No", source_dataset="truthfulqa")
        self.assertEqual(q, "Is the earth flat?")

    def test_fever_uses_claim(self):
        # For FEVER, prompt is generic verification instruction, claim is the query
        q = format_query_text(
            prompt="Verify whether the following claim is supported by factual evidence.",
            original_response="Saturn is the second-largest planet.",
            source_dataset="fever",
        )
        self.assertEqual(q, "Saturn is the second-largest planet.")


class TestFeatureExtractionWithMocks(unittest.TestCase):
    """Test end-to-end feature extraction using synthetic mock vectors."""

    def test_mock_feature_extraction(self):
        # Corpus with 4 2D vectors
        corpus_texts = ["doc_A", "doc_B", "doc_C", "doc_D"]
        corpus_embs = np.array([
            [1.0, 0.0],  # rank 1 for query [1, 0]
            [0.8, 0.6],  # rank 2
            [0.0, 1.0],  # rank 3
            [-1.0, 0.0], # rank 4
        ], dtype=np.float32)

        query_emb = np.array([1.0, 0.0], dtype=np.float32)
        response_emb = np.array([1.0, 0.0], dtype=np.float32)

        features = extract_retrieval_features(
            query_text="query",
            response_text="response",
            corpus_texts=corpus_texts,
            corpus_embeddings=corpus_embs,
            query_emb=query_emb,
            response_emb=response_emb,
            k=3,
        )

        expected_keys = {
            "retrieved_rank_1",
            "retrieved_rank_2",
            "retrieved_rank_3",
            "retrieved_rank_1_text",
            "retrieved_rank_2_text",
            "retrieved_rank_3_text",
            "top1_evidence_similarity",
            "mean_top3_evidence_similarity",
            "response_top1_evidence_similarity",
            "mean_response_top3_evidence_similarity",
            "evidence_margin",
            "retrieval_agreement",
        }
        self.assertEqual(set(features.keys()), expected_keys)

        # Top 1 should be doc_A with similarity 1.0
        self.assertEqual(features["retrieved_rank_1_text"], "doc_A")
        self.assertAlmostEqual(features["top1_evidence_similarity"], 1.0, places=5)
        self.assertAlmostEqual(features["response_top1_evidence_similarity"], 1.0, places=5)
        # Margin = 1.0 - 0.8 = 0.2
        self.assertAlmostEqual(features["evidence_margin"], 0.2, places=5)
        # Agreement = 1.0 * 1.0 = 1.0
        self.assertAlmostEqual(features["retrieval_agreement"], 1.0, places=5)

        for k, v in features.items():
            if isinstance(v, (int, float)):
                self.assertTrue(math.isfinite(v), f"Value for {k} is not finite: {v}")

    def test_empty_corpus_feature_extraction(self):
        features = extract_retrieval_features(
            query_text="query",
            response_text="response",
            corpus_texts=[],
            corpus_embeddings=np.empty((0, 2), dtype=np.float32),
            query_emb=np.array([1.0, 0.0]),
            response_emb=np.array([1.0, 0.0]),
        )
        self.assertEqual(features["retrieved_rank_1"], "none")
        self.assertEqual(features["top1_evidence_similarity"], 0.0)
        self.assertEqual(features["retrieval_agreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
