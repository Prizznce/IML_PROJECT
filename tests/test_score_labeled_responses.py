"""
Unit tests for supervised benchmark-response scoring pipeline.

Tests prompt formatting, signal extraction with mock model, validation assertions,
and checkpoint/resume mechanisms without requiring GPU hardware or model downloads.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.score_labeled_responses import (
    format_scoring_prompt,
    score_single_response,
    score_labeled_dataset,
    validate_scored_dataset,
)


class MockChatTokenizer:
    """Mock tokenizer providing encode, decode, and apply_chat_template."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.all_special_ids = [0, 1]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        # Simple whitespace tokenization mapping words to ids
        return [(hash(w) % (self.vocab_size - 2)) + 2 for w in text.split()]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok_{t}" for t in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        rendered = ""
        for msg in messages:
            rendered += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
            else:
                rendered += "<think>\n"
        return rendered


class MockCausalLM(nn.Module):
    """Deterministic mock causal LM returning consistent logits."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **kwargs):
        batch_size, seq_len = input_ids.shape
        # Deterministic logits favoring token_id
        logits = torch.ones(batch_size, seq_len, self.vocab_size, dtype=torch.float32) * -5.0
        for b in range(batch_size):
            for t in range(seq_len):
                tok = input_ids[b, t].item()
                logits[b, t, tok] = 5.0  # High logit for target

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


class TestScoreLabeledResponses(unittest.TestCase):
    """Test suite for score_labeled_responses pipeline."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_tok = MockChatTokenizer(vocab_size=50)
        self.mock_model = MockCausalLM(vocab_size=50)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_format_scoring_prompt(self):
        """Test prompt formatting across HaluEval, TruthfulQA, and FEVER."""
        # HaluEval
        h_prompt = format_scoring_prompt(
            prompt="Who won the match?",
            context="Team A defeated Team B.",
            source_dataset="halueval",
            use_chat_template=True,
            tokenizer=self.mock_tok,
        )
        self.assertIn("Passage: Team A defeated Team B.", h_prompt)
        self.assertIn("Question: Who won the match?", h_prompt)
        self.assertTrue(h_prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

        # TruthfulQA
        t_prompt = format_scoring_prompt(
            prompt="What happens if you swallow gum?",
            source_dataset="truthfulqa",
            use_chat_template=True,
            tokenizer=self.mock_tok,
        )
        self.assertIn("Question: What happens if you swallow gum?", t_prompt)

        # FEVER
        f_prompt = format_scoring_prompt(
            prompt="Verify claim.",
            source_dataset="fever",
            use_chat_template=True,
            tokenizer=self.mock_tok,
        )
        self.assertIn("Verify whether the following claim is supported by factual evidence.", f_prompt)

    def test_score_single_response(self):
        """Test signal extraction and rank dispersion over an original labeled response."""
        prompt = "<|im_start|>user\nQuestion: What is X?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        response = "X is a widget."

        metrics = score_single_response(
            model=self.mock_model,
            tokenizer=self.mock_tok,
            prompt=prompt,
            response=response,
            device="cpu",
        )

        self.assertGreater(metrics["num_tokens"], 0)
        self.assertGreaterEqual(metrics["mean_token_prob"], 0.0)
        self.assertLessEqual(metrics["mean_token_prob"], 1.0)
        self.assertGreaterEqual(metrics["mean_entropy"], 0.0)
        self.assertGreater(metrics["perplexity"], 0.0)
        self.assertIn("mean_token_rank", metrics)
        self.assertIn("max_token_rank", metrics)
        self.assertIn("rank_std", metrics)

    def test_validate_scored_dataset(self):
        """Test validation checks on valid and invalid scored DataFrames."""
        valid_data = {
            "id": ["ex_1", "ex_2"],
            "source_dataset": ["halueval", "fever"],
            "label": [0, 1],
            "prompt": ["P1", "P2"],
            "response": ["R1", "R2"],
            "num_tokens": [10, 12],
            "mean_token_prob": [0.8, 0.6],
            "min_token_prob": [0.3, 0.2],
            "mean_log_prob": [-0.3, -0.6],
            "min_log_prob": [-1.2, -1.8],
            "mean_entropy": [0.5, 0.9],
            "max_entropy": [1.5, 2.2],
            "entropy_std": [0.3, 0.4],
            "token_prob_std": [0.1, 0.15],
            "perplexity": [1.35, 1.82],
            "mean_token_rank": [1.0, 1.2],
            "max_token_rank": [1, 2],
            "min_token_rank": [1, 1],
            "rank_std": [0.0, 0.4],
        }
        df_valid = pd.DataFrame(valid_data)
        checks = validate_scored_dataset(df_valid, expected_count=2)
        self.assertTrue(all(checks.values()), f"Checks failed: {checks}")

        # Invalid data: NaN prob
        df_invalid = df_valid.copy()
        df_invalid.loc[0, "mean_token_prob"] = float("nan")
        checks_invalid = validate_scored_dataset(df_invalid, expected_count=2)
        self.assertFalse(checks_invalid["no_nan_or_inf_in_signals"])

    def test_checkpoint_and_resume(self):
        """Verify checkpoint saving and resumption skipping already processed examples."""
        input_data = {
            "id": ["ex_1", "ex_2", "ex_3", "ex_4"],
            "source_dataset": ["halueval", "halueval", "truthfulqa", "fever"],
            "label": [0, 1, 0, 1],
            "prompt": ["P1", "P2", "P3", "P4"],
            "context": ["C1", "C2", "", ""],
            "response": ["Ans1", "Ans2", "Ans3", "Ans4"],
        }
        input_file = os.path.join(self.test_dir, "test_input.parquet")
        pd.DataFrame(input_data).to_parquet(input_file, index=False)

        out_parquet = os.path.join(self.test_dir, "test_out.parquet")
        out_csv = os.path.join(self.test_dir, "test_out.csv")
        ckpt_file = os.path.join(self.test_dir, "test_out.checkpoint.parquet")

        # Step 1: Pre-create a checkpoint containing ex_1 and ex_2
        partial_records = [
            {
                "id": "ex_1",
                "source_dataset": "halueval",
                "label": 0,
                "prompt": "P1",
                "context": "C1",
                "response": "Ans1",
                "model_input": "prompt_1",
                "forward_time_s": 0.01,
                "num_tokens": 5,
                "mean_token_prob": 0.9,
                "min_token_prob": 0.5,
                "mean_log_prob": -0.1,
                "min_log_prob": -0.7,
                "mean_entropy": 0.3,
                "max_entropy": 1.0,
                "entropy_std": 0.2,
                "token_prob_std": 0.1,
                "perplexity": 1.1,
                "mean_token_rank": 1.0,
                "max_token_rank": 1,
                "min_token_rank": 1,
                "rank_std": 0.0,
            },
            {
                "id": "ex_2",
                "source_dataset": "halueval",
                "label": 1,
                "prompt": "P2",
                "context": "C2",
                "response": "Ans2",
                "model_input": "prompt_2",
                "forward_time_s": 0.01,
                "num_tokens": 6,
                "mean_token_prob": 0.8,
                "min_token_prob": 0.4,
                "mean_log_prob": -0.2,
                "min_log_prob": -0.9,
                "mean_entropy": 0.4,
                "max_entropy": 1.2,
                "entropy_std": 0.2,
                "token_prob_std": 0.1,
                "perplexity": 1.2,
                "mean_token_rank": 1.0,
                "max_token_rank": 1,
                "min_token_rank": 1,
                "rank_std": 0.0,
            },
        ]
        pd.DataFrame(partial_records).to_parquet(ckpt_file, index=False)

        # Step 2: Run scoring with resume=True
        df_out, summary = score_labeled_dataset(
            input_path=input_file,
            output_parquet=out_parquet,
            output_csv=out_csv,
            checkpoint_path=ckpt_file,
            device="cpu",
            checkpoint_interval=1,
            resume=True,
            model=self.mock_model,
            tokenizer=self.mock_tok,
        )

        # Verify all 4 examples are present
        self.assertEqual(len(df_out), 4)
        self.assertEqual(set(df_out["id"]), {"ex_1", "ex_2", "ex_3", "ex_4"})
        # Verify newly scored count is 2 (only ex_3 and ex_4)
        self.assertEqual(summary["newly_scored"], 2)
        # Verify checkpoint file was removed on completion
        self.assertFalse(os.path.exists(ckpt_file))
        # Verify output files exist
        self.assertTrue(os.path.exists(out_parquet))
        self.assertTrue(os.path.exists(out_csv))


if __name__ == "__main__":
    unittest.main()
