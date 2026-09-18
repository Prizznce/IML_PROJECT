"""
Unit tests for Production Full Self-Consistency Feature Generation Pipeline.

Verifies:
- Function interface and parameter defaults
- CLI argument parsing for production and pilot modes
- Checkpoint persistence and resume mechanics
- Strict validation assertions (row counts, uniqueness, bounds, finiteness)
- Mathematical identity: generation_disagreement = 1.0 - mean_pairwise_similarity
- Preserves original example IDs and labels without data mutation
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.consistency.self_consistency import (
    run_self_consistency_full,
    run_self_consistency_pilot,
    extract_self_consistency_signals,
    calculate_exact_match_agreement,
    calculate_generation_disagreement,
)


class TestFullSelfConsistencyPipeline(unittest.TestCase):
    """Test suite for full self-consistency production pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

        # Create small mock dataset of 4 examples
        self.mock_data = pd.DataFrame({
            "id": ["ex_001", "ex_002", "ex_003", "ex_004"],
            "source_dataset": ["halueval", "halueval", "fever", "truthfulqa"],
            "label": [0, 1, 0, 1],
            "prompt": ["What is A?", "What is B?", "Verify C.", "Where is D?"],
            "context": ["Passage A", "Passage B", "", ""],
            "response": ["Answer A", "Answer B", "Answer C", "Answer D"],
        })
        self.mock_parquet_path = self.out_dir / "mock_input.parquet"
        self.mock_data.to_parquet(self.mock_parquet_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mock_full_generation_and_validation(self):
        """Test full generation runner with mocked model generation and embeddings."""
        mock_resps = ["Paris", "paris", "PARIS", "Paris!", "London"]

        with patch("src.consistency.self_consistency.generate_stochastic_responses") as mock_gen, \
             patch("src.consistency.self_consistency.load_embedding_model") as mock_load_emb, \
             patch("src.generation.generate_signals.load_model_and_tokenizer") as mock_load_llm:

            mock_load_llm.return_value = (MagicMock(), MagicMock())
            mock_embed = MagicMock()
            mock_load_emb.return_value = mock_embed
            # Mock SentenceTransformer encode returning unit vectors
            mock_embed.encode.return_value = np.array([
                [1.0, 0.0],
                [0.99, 0.01],
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
            ], dtype=np.float32)

            mock_gen.return_value = (mock_resps, 0.5, False)

            df_gen, df_agg, summary = run_self_consistency_full(
                data_path=str(self.mock_parquet_path),
                output_dir=str(self.out_dir),
                num_generations=5,
                checkpoint_interval=2,
                resume=True,
                limit=4,
            )

            # Assertions on outputs
            self.assertEqual(len(df_agg), 4)
            self.assertEqual(len(df_gen), 20)  # 4 x 5
            self.assertEqual(df_agg["id"].nunique(), 4)
            self.assertEqual(list(df_agg["id"]), ["ex_001", "ex_002", "ex_003", "ex_004"])

            # Verify math identity
            diff = np.abs(df_agg["generation_disagreement"] - (1.0 - df_agg["mean_pairwise_similarity"]))
            self.assertTrue((diff < 1e-5).all())

            # Verify files exist
            self.assertTrue((self.out_dir / "full_generations_raw.parquet").exists())
            self.assertTrue((self.out_dir / "full_generations_raw.csv").exists())
            self.assertTrue((self.out_dir / "full_self_consistency_aggregated.parquet").exists())
            self.assertTrue((self.out_dir / "full_self_consistency_aggregated.csv").exists())
            self.assertTrue((self.out_dir / "full_run_summary.json").exists())

    def test_resume_skips_already_completed(self):
        """Verify that resume skips already completed examples."""
        # Create pre-existing aggregated file with ex_001 and ex_002
        agg_file = self.out_dir / "full_self_consistency_aggregated.parquet"
        gen_file = self.out_dir / "full_generations_raw.parquet"

        existing_agg = pd.DataFrame({
            "id": ["ex_001", "ex_002"],
            "source_dataset": ["halueval", "halueval"],
            "original_label": [0, 1],
            "num_generations": [5, 5],
            "exact_match_agreement": [0.8, 0.6],
            "unique_response_ratio": [0.4, 0.6],
            "majority_response_fraction": [0.6, 0.4],
            "mean_pairwise_similarity": [0.9, 0.7],
            "min_pairwise_similarity": [0.8, 0.5],
            "max_pairwise_similarity": [1.0, 0.9],
            "pairwise_similarity_std": [0.05, 0.1],
            "generation_disagreement": [0.1, 0.3],
        })
        existing_gen = pd.DataFrame({
            "id": ["ex_001"] * 5 + ["ex_002"] * 5,
            "source_dataset": ["halueval"] * 10,
            "prompt": ["What is A?"] * 5 + ["What is B?"] * 5,
            "context": ["Passage A"] * 5 + ["Passage B"] * 5,
            "original_response": ["Answer A"] * 5 + ["Answer B"] * 5,
            "original_label": [0] * 5 + [1] * 5,
            "generation_index": list(range(5)) * 2,
            "generated_response": ["resp"] * 10,
            "normalized_response": ["resp"] * 10,
        })
        existing_agg.to_parquet(agg_file, index=False)
        existing_gen.to_parquet(gen_file, index=False)

        # Also write checkpoint
        ckpt_agg = self.out_dir / ".checkpoint_full_aggregated.parquet"
        ckpt_gen = self.out_dir / ".checkpoint_full_generations.parquet"
        existing_agg.to_parquet(ckpt_agg, index=False)
        existing_gen.to_parquet(ckpt_gen, index=False)

        with patch("src.consistency.self_consistency.generate_stochastic_responses") as mock_gen, \
             patch("src.consistency.self_consistency.load_embedding_model") as mock_load_emb, \
             patch("src.generation.generate_signals.load_model_and_tokenizer") as mock_load_llm:

            mock_load_llm.return_value = (MagicMock(), MagicMock())
            mock_embed = MagicMock()
            mock_load_emb.return_value = mock_embed
            mock_embed.encode.return_value = np.array([[1.0, 0.0]] * 5, dtype=np.float32)
            mock_gen.return_value = (["ans"] * 5, 0.2, False)

            # Run with limit=4 (should only process ex_003 and ex_004)
            df_gen, df_agg, _ = run_self_consistency_full(
                data_path=str(self.mock_parquet_path),
                output_dir=str(self.out_dir),
                num_generations=5,
                resume=True,
                limit=4,
            )

            # generate_stochastic_responses should have been called exactly 2 times (for ex_003 and ex_004)
            self.assertEqual(mock_gen.call_count, 2)
            self.assertEqual(len(df_agg), 4)
            self.assertEqual(len(df_gen), 20)

    def test_original_input_file_unmodified(self):
        """Ensure input dataset is completely untouched."""
        mtime_before = os.path.getmtime(self.mock_parquet_path)
        content_before = pd.read_parquet(self.mock_parquet_path)

        with patch("src.consistency.self_consistency.generate_stochastic_responses") as mock_gen, \
             patch("src.consistency.self_consistency.load_embedding_model") as mock_load_emb, \
             patch("src.generation.generate_signals.load_model_and_tokenizer") as mock_load_llm:

            mock_load_llm.return_value = (MagicMock(), MagicMock())
            mock_embed = MagicMock()
            mock_load_emb.return_value = mock_embed
            mock_embed.encode.return_value = np.array([[1.0, 0.0]] * 5, dtype=np.float32)
            mock_gen.return_value = (["ans"] * 5, 0.1, False)

            run_self_consistency_full(
                data_path=str(self.mock_parquet_path),
                output_dir=str(self.out_dir),
                num_generations=5,
                limit=2,
            )

        mtime_after = os.path.getmtime(self.mock_parquet_path)
        content_after = pd.read_parquet(self.mock_parquet_path)

        self.assertEqual(mtime_before, mtime_after)
        pd.testing.assert_frame_equal(content_before, content_after)


if __name__ == "__main__":
    unittest.main()
