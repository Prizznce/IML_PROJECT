"""
Unit tests for Production Full NLI Agreement Feature Generation Pipeline.

Verifies:
- Production pipeline runner interface and argument defaults
- Checkpoint persistence and resume mechanics
- Strict validation assertions (row counts, uniqueness, bounds, finiteness)
- Mathematical identities:
  mean_entailment + mean_contradiction + mean_neutral == 1.0
  nli_disagreement == mean_pairwise_contradiction + 0.5 * mean_pairwise_neutral
- Pre-validation of exactly 5 candidate generations per example
- Non-mutation of input generations dataset
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.consistency.nli_agreement import (
    run_nli_agreement_full,
    calculate_nli_disagreement,
    aggregate_example_nli_signals,
)


class TestFullNLIPipeline(unittest.TestCase):
    """Test suite for production full NLI agreement pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

        # Create mock 4-example dataset with exactly 5 generations each (20 rows)
        records = []
        for i in range(4):
            ex_id = f"ex_{i+1:03d}"
            for g in range(5):
                records.append({
                    "id": ex_id,
                    "source_dataset": "halueval" if i < 2 else "truthfulqa",
                    "original_label": i % 2,
                    "generation_index": g,
                    "generated_response": f"Response {g} for {ex_id}",
                    "normalized_response": f"response {g} for {ex_id}",
                })
        self.mock_raw_df = pd.DataFrame(records)
        self.mock_raw_path = self.out_dir / "mock_generations.parquet"
        self.mock_raw_df.to_parquet(self.mock_raw_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mock_full_nli_pipeline_and_validation(self):
        """Test full NLI pipeline with mocked cross-encoder inference."""
        with patch("src.consistency.nli_agreement.load_nli_model") as mock_load:
            mock_model = MagicMock()
            mock_label_indices = {"contradiction": 0, "entailment": 1, "neutral": 2}
            mock_load.return_value = (mock_model, mock_label_indices)

            # Mock model.predict returning logits for 20 directional pairs per example
            # Shape (20, 3) logits
            def mock_predict(pairs, batch_size=32, convert_to_numpy=True):
                n = len(pairs)
                # Return balanced logits
                logits = np.zeros((n, 3), dtype=np.float32)
                logits[:, 0] = -1.0  # contradiction
                logits[:, 1] = 2.0   # entailment
                logits[:, 2] = 0.5   # neutral
                return logits

            mock_model.predict.side_effect = mock_predict

            df_pair, df_agg, summary = run_nli_agreement_full(
                input_generations_path=str(self.mock_raw_path),
                output_dir=str(self.out_dir),
                checkpoint_interval=2,
                resume=True,
                limit=4,
            )

            # Assertions on outputs
            self.assertEqual(len(df_agg), 4)
            self.assertEqual(len(df_pair), 40)  # 4 examples x 10 pairs
            self.assertEqual(df_agg["id"].nunique(), 4)

            # Probability sum identity
            prob_sum = (
                df_agg["mean_pairwise_entailment"]
                + df_agg["mean_pairwise_contradiction"]
                + df_agg["mean_pairwise_neutral"]
            )
            self.assertTrue((np.abs(prob_sum - 1.0) < 1e-5).all())

            # Disagreement identity
            expected_dis = (
                df_agg["mean_pairwise_contradiction"]
                + 0.5 * df_agg["mean_pairwise_neutral"]
            )
            self.assertTrue((np.abs(df_agg["nli_disagreement"] - expected_dis) < 1e-5).all())

            # Artifact files created
            self.assertTrue((self.out_dir / "full_nli_aggregated.parquet").exists())
            self.assertTrue((self.out_dir / "full_nli_aggregated.csv").exists())
            self.assertTrue((self.out_dir / "full_nli_pairwise.parquet").exists())
            self.assertTrue((self.out_dir / "full_nli_summary.json").exists())

    def test_resume_skips_already_completed(self):
        """Verify that checkpoint resume skips already processed examples."""
        # Create intermediate checkpoint with ex_001 and ex_002
        ckpt_agg_path = self.out_dir / ".checkpoint_full_nli_aggregated.parquet"
        ckpt_pair_path = self.out_dir / ".checkpoint_full_nli_pairwise.parquet"

        existing_agg = pd.DataFrame({
            "id": ["ex_001", "ex_002"],
            "source_dataset": ["halueval", "halueval"],
            "original_label": [0, 1],
            "num_generations": [5, 5],
            "mean_pairwise_entailment": [0.7, 0.6],
            "mean_pairwise_contradiction": [0.1, 0.2],
            "mean_pairwise_neutral": [0.2, 0.2],
            "fraction_entailing_pairs": [0.8, 0.6],
            "fraction_contradicting_pairs": [0.1, 0.2],
            "fraction_neutral_pairs": [0.1, 0.2],
            "nli_disagreement": [0.2, 0.3],
        })
        existing_pair = pd.DataFrame([{
            "id": f"ex_{i+1:03d}",
            "source_dataset": "halueval",
            "original_label": 0,
            "pair_index_i": 0,
            "pair_index_j": 1,
            "response_i": "a",
            "response_j": "b",
            "prob_entail_i_to_j": 0.7,
            "prob_contra_i_to_j": 0.1,
            "prob_neutral_i_to_j": 0.2,
            "prob_entail_j_to_i": 0.7,
            "prob_contra_j_to_i": 0.1,
            "prob_neutral_j_to_i": 0.2,
            "pair_entailment": 0.7,
            "pair_contradiction": 0.1,
            "pair_neutral": 0.2,
            "predicted_class": "entailment",
        } for i in range(2) for _ in range(10)])

        existing_agg.to_parquet(ckpt_agg_path, index=False)
        existing_pair.to_parquet(ckpt_pair_path, index=False)

        with patch("src.consistency.nli_agreement.load_nli_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, {"contradiction": 0, "entailment": 1, "neutral": 2})

            def mock_predict(pairs, batch_size=32, convert_to_numpy=True):
                logits = np.zeros((len(pairs), 3), dtype=np.float32)
                logits[:, 1] = 2.0
                return logits

            mock_model.predict.side_effect = mock_predict

            df_pair, df_agg, _ = run_nli_agreement_full(
                input_generations_path=str(self.mock_raw_path),
                output_dir=str(self.out_dir),
                resume=True,
                limit=4,
            )

            # Only ex_003 and ex_004 should have been inferred
            self.assertEqual(mock_model.predict.call_count, 2)
            self.assertEqual(len(df_agg), 4)
            self.assertEqual(len(df_pair), 40)

    def test_bad_generation_count_raises(self):
        """Ensure pipeline catches examples with missing or extra generations."""
        bad_df = self.mock_raw_df.iloc[:-1]  # Drop one generation
        bad_path = self.out_dir / "bad_generations.parquet"
        bad_df.to_parquet(bad_path, index=False)

        with self.assertRaises(ValueError):
            run_nli_agreement_full(
                input_generations_path=str(bad_path),
                output_dir=str(self.out_dir),
            )

    def test_original_input_file_unmodified(self):
        """Ensure input dataset is completely untouched."""
        mtime_before = os.path.getmtime(self.mock_raw_path)
        content_before = pd.read_parquet(self.mock_raw_path)

        with patch("src.consistency.nli_agreement.load_nli_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, {"contradiction": 0, "entailment": 1, "neutral": 2})
            mock_model.predict.return_value = np.zeros((20, 3), dtype=np.float32)

            run_nli_agreement_full(
                input_generations_path=str(self.mock_raw_path),
                output_dir=str(self.out_dir),
                limit=2,
            )

        mtime_after = os.path.getmtime(self.mock_raw_path)
        content_after = pd.read_parquet(self.mock_raw_path)

        self.assertEqual(mtime_before, mtime_after)
        pd.testing.assert_frame_equal(content_before, content_after)


if __name__ == "__main__":
    unittest.main()
