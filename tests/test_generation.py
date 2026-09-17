"""
Unit tests for generation pipeline utilities and validation logic.

Tests input formatting, canonical dataset conditioning, and data integrity
assertions using mock data and mock tokenizers without external model downloads
or GPU hardware requirements.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.generate_signals import (
    format_model_input,
    validate_pilot_signals,
)


class MockChatTokenizer:
    """Mock tokenizer providing apply_chat_template for testing conditioning formatting."""

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


class TestGenerationPipeline(unittest.TestCase):
    """Test suite for generation formatting and validation assertions."""

    def test_format_model_input_raw(self):
        """Verify v1 legacy formatting for each dataset provenance."""
        # HaluEval with context
        h_input = format_model_input(
            prompt="What is X?",
            context="X is a widget.",
            source_dataset="halueval",
            use_chat_template=False,
        )
        self.assertIn("Passage: X is a widget.", h_input)
        self.assertIn("Question: What is X?", h_input)
        self.assertTrue(h_input.endswith("Answer:"))

        # FEVER with claim
        f_input = format_model_input(
            prompt="Verify the claim.",
            original_response="The sky is blue.",
            source_dataset="fever",
            use_chat_template=False,
        )
        self.assertIn("Claim: The sky is blue.", f_input)
        self.assertTrue(f_input.endswith("Answer:"))

        # TruthfulQA
        t_input = format_model_input(
            prompt="Why is grass green?",
            source_dataset="truthfulqa",
            use_chat_template=False,
        )
        self.assertIn("Question: Why is grass green?", t_input)
        self.assertTrue(t_input.endswith("Answer:"))

    def test_format_model_input_chat_template(self):
        """Verify v2 ChatML formatting with suppressed reasoning."""
        mock_tok = MockChatTokenizer()
        chat_input = format_model_input(
            prompt="What is X?",
            context="Reference text.",
            source_dataset="halueval",
            use_chat_template=True,
            tokenizer=mock_tok,
        )
        self.assertIn("<|im_start|>system\nAnswer the question directly", chat_input)
        self.assertIn("<|im_start|>user\nPassage: Reference text.\n\nQuestion: What is X?<|im_end|>", chat_input)
        self.assertTrue(chat_input.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_validate_pilot_signals_valid(self):
        """Verify validation passes on completely valid signal data."""
        data = {
            "id": [f"ex_{i}" for i in range(10)],
            "source_dataset": ["halueval"] * 5 + ["truthfulqa"] * 3 + ["fever"] * 2,
            "label": [0, 1] * 5,
            "prompt": ["Sample prompt"] * 10,
            "generated_response": ["Sample valid response text."] * 10,
            "num_generated_tokens": [25] * 10,
            "mean_token_prob": [0.85] * 10,
            "min_token_prob": [0.20] * 10,
            "mean_log_prob": [-0.25] * 10,
            "min_log_prob": [-1.50] * 10,
            "mean_entropy": [0.65] * 10,
            "max_entropy": [2.50] * 10,
            "entropy_std": [0.40] * 10,
            "token_prob_std": [0.15] * 10,
            "perplexity": [1.35] * 10,
        }
        df = pd.DataFrame(data)
        checks = validate_pilot_signals(df, expected_count=10, check_no_think_tags=True)
        self.assertTrue(all(checks.values()), f"Failed checks: {checks}")

    def test_validate_pilot_signals_catches_think_tag(self):
        """Verify validation catches leaked <think> tag in v2."""
        data = {
            "id": [f"ex_{i}" for i in range(2)],
            "source_dataset": ["halueval", "fever"],
            "label": [0, 1],
            "prompt": ["P1", "P2"],
            "generated_response": ["Clean response", "<think>Leaked thinking</think> answer"],
            "num_generated_tokens": [10, 15],
            "mean_token_prob": [0.8, 0.7],
            "min_token_prob": [0.3, 0.2],
            "mean_log_prob": [-0.3, -0.4],
            "min_log_prob": [-1.2, -1.5],
            "mean_entropy": [0.5, 0.6],
            "max_entropy": [2.0, 2.5],
            "entropy_std": [0.3, 0.4],
            "token_prob_std": [0.1, 0.15],
            "perplexity": [1.4, 1.6],
        }
        df = pd.DataFrame(data)
        checks = validate_pilot_signals(df, expected_count=2, check_no_think_tags=True)
        self.assertFalse(checks["no_reasoning_think_tags"])

    def test_validate_pilot_signals_catches_nan(self):
        """Verify validation catches NaN or invalid values."""
        data = {
            "id": ["ex_0"],
            "source_dataset": ["halueval"],
            "label": [0],
            "prompt": ["P"],
            "generated_response": ["R"],
            "num_generated_tokens": [10],
            "mean_token_prob": [float("nan")],
            "min_token_prob": [0.2],
            "mean_log_prob": [-0.3],
            "min_log_prob": [-1.2],
            "mean_entropy": [0.5],
            "max_entropy": [2.0],
            "entropy_std": [0.3],
            "token_prob_std": [0.1],
            "perplexity": [1.4],
        }
        df = pd.DataFrame(data)
        checks = validate_pilot_signals(df, expected_count=1, check_no_think_tags=True)
        self.assertFalse(checks["no_nan_or_inf_in_signals"])


if __name__ == "__main__":
    unittest.main()
