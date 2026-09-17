"""
Unit tests for raw-logit token and sequence signal extraction module.

Tests mathematical correctness, causal alignment, prompt token exclusion,
and numerical stability using synthetic logits and mock causal LM modules
without external network or model download dependencies.
"""

import math
import sys
import unittest
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.token_signals import (
    TokenSignal,
    SequenceSignals,
    compute_token_signals_from_logits,
    extract_raw_sequence_signals,
)


class DummyTokenizer:
    """Mock tokenizer for deterministic testing."""
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.all_special_ids = [0, 1]  # 0: <pad>, 1: <eos>

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        # Simple whitespace tokenizer or char ordinals
        tokens = [ord(c) % self.vocab_size for c in text.split()]
        if add_special_tokens:
            return [1] + tokens
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(f"tok_{t}" for t in token_ids)


class MockCausalOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class MockCausalLM(nn.Module):
    """Mock causal language model that returns pre-configured or deterministic logits."""
    def __init__(self, vocab_size: int = 100, custom_logits: Optional[torch.Tensor] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.custom_logits = custom_logits

    def forward(self, input_ids: torch.Tensor, **kwargs):
        batch_size, seq_len = input_ids.shape
        if self.custom_logits is not None:
            return MockCausalOutput(self.custom_logits)
        # Default: small random logits
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float32)
        return MockCausalOutput(logits)


class TestTokenSignals(unittest.TestCase):

    def test_probability_and_log_probability_calculation(self):
        """Test exact softmax probability and log_softmax values."""
        # 1 step, vocab size 4
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], dtype=torch.float32)
        target_token = [0]  # token id 0 has logit 2.0

        signals = compute_token_signals_from_logits(logits, target_token)

        expected_probs = F.softmax(logits[0], dim=-1)
        expected_log_probs = F.log_softmax(logits[0], dim=-1)

        self.assertEqual(signals.num_tokens, 1)
        self.assertAlmostEqual(signals.tokens[0].probability, expected_probs[0].item(), places=5)
        self.assertAlmostEqual(signals.tokens[0].log_probability, expected_log_probs[0].item(), places=5)
        self.assertEqual(signals.tokens[0].rank, 1)  # logit 2.0 is the largest (rank 1)

    def test_entropy_calculation_uniform_and_peaked(self):
        """Test entropy for theoretical bounds: uniform ln(K) and deterministic 0."""
        vocab_size = 4
        # Case A: Uniform distribution over 4 tokens -> H = ln(4) nats
        uniform_logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        target = [2]
        sig_uniform = compute_token_signals_from_logits(uniform_logits, target)
        expected_entropy = math.log(4)
        self.assertAlmostEqual(sig_uniform.tokens[0].entropy, expected_entropy, places=5)
        self.assertAlmostEqual(sig_uniform.mean_entropy, expected_entropy, places=5)

        # Case B: Peaked/deterministic distribution -> H approaches 0
        peaked_logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]], dtype=torch.float32)
        sig_peaked = compute_token_signals_from_logits(peaked_logits, [0])
        self.assertAlmostEqual(sig_peaked.tokens[0].entropy, 0.0, places=4)
        self.assertAlmostEqual(sig_peaked.tokens[0].probability, 1.0, places=4)

    def test_perplexity_calculation(self):
        """Test perplexity against analytical values."""
        # Uniform distribution over 8 tokens -> perplexity must equal exactly 8.0
        vocab_size = 8
        logits = torch.zeros((3, vocab_size), dtype=torch.float32)  # 3 tokens
        targets = [1, 3, 5]
        signals = compute_token_signals_from_logits(logits, targets)

        self.assertEqual(signals.num_tokens, 3)
        self.assertAlmostEqual(signals.perplexity, 8.0, places=4)

        # Deterministic predictions -> perplexity must equal 1.0
        peaked_logits = torch.tensor([
            [50.0, 0.0],
            [0.0, 50.0]
        ], dtype=torch.float32)
        targets_peaked = [0, 1]
        sig_peaked = compute_token_signals_from_logits(peaked_logits, targets_peaked)
        self.assertAlmostEqual(sig_peaked.perplexity, 1.0, places=3)

    def test_token_alignment_causal_shift(self):
        """
        Verify causal alignment: Logit at index t predicts token at index t+1.
        Prompt tokens = [10, 20] (len P=2)
        Response tokens = [30, 40, 50] (len R=3)
        Full sequence = [10, 20, 30, 40, 50] (total len N=5)
        Position P-1 (idx 1) predicts response token 0 (30).
        Position P (idx 2) predicts response token 1 (40).
        Position P+1 (idx 3) predicts response token 2 (50).
        """
        vocab_size = 100
        total_len = 5
        # Construct synthetic logits where each position puts 100.0 on a specific distinct token
        logits = torch.full((1, total_len, vocab_size), -50.0, dtype=torch.float32)
        # Position 0 predicts token 20
        logits[0, 0, 20] = 50.0
        # Position 1 (P-1) predicts token 30 (first response token!)
        logits[0, 1, 30] = 50.0
        # Position 2 predicts token 40 (second response token!)
        logits[0, 2, 40] = 50.0
        # Position 3 predicts token 50 (third response token!)
        logits[0, 3, 50] = 50.0

        model = MockCausalLM(vocab_size=vocab_size, custom_logits=logits)

        # Tokenizer mock that directly returns the desired IDs
        class ExactTokenizer:
            def encode(self, text, add_special_tokens=True):
                if text == "PROMPT":
                    return [10, 20]
                elif text == "RESPONSE":
                    return [30, 40, 50]
                return []

            def decode(self, ids):
                return str(ids[0])

        tok = ExactTokenizer()
        signals = extract_raw_sequence_signals(model, tok, prompt="PROMPT", response="RESPONSE")

        self.assertEqual(signals.num_tokens, 3)
        # All 3 response tokens should have near 100% probability because we aligned with positions 1, 2, 3
        for t in signals.tokens:
            self.assertAlmostEqual(t.probability, 1.0, places=3)
            self.assertEqual(t.rank, 1)

        self.assertEqual(signals.tokens[0].token_id, 30)
        self.assertEqual(signals.tokens[1].token_id, 40)
        self.assertEqual(signals.tokens[2].token_id, 50)

    def test_exclusion_of_prompt_tokens(self):
        """
        Ensure prompt tokens are NOT included in sequence statistics.
        If prompt tokens had probability 0 and response tokens had probability 1,
        mean_token_prob must be 1.0, not contaminated by prompt.
        """
        vocab_size = 10
        # Full seq: prompt=[1, 2], response=[3]
        total_len = 3
        logits = torch.full((1, total_len, vocab_size), -10.0, dtype=torch.float32)
        # Position 0 (predicting prompt token 2): low probability for token 2
        logits[0, 0, 9] = 10.0  # puts prob on 9, so prompt token 2 gets ~0 prob
        # Position 1 (P-1, predicting response token 3): high prob for token 3
        logits[0, 1, 3] = 20.0

        model = MockCausalLM(vocab_size=vocab_size, custom_logits=logits)

        class MockTok:
            def encode(self, text, add_special_tokens=True):
                return [1, 2] if text == "P" else [3]

            def decode(self, ids):
                return str(ids[0])

        signals = extract_raw_sequence_signals(model, MockTok(), prompt="P", response="R")

        self.assertEqual(signals.num_tokens, 1)
        self.assertAlmostEqual(signals.mean_token_prob, 1.0, places=3)
        self.assertAlmostEqual(signals.perplexity, 1.0, places=2)

    def test_numerical_stability_nan_and_inf(self):
        """Verify that -inf / zero probabilities do not produce NaN via 0 * log(0)."""
        vocab_size = 5
        logits = torch.tensor([
            [10.0, -float("inf"), -float("inf"), -float("inf"), -float("inf")],
            [0.0, -1e9, -1e9, 0.0, -1e9]
        ], dtype=torch.float32)
        targets = [0, 3]

        signals = compute_token_signals_from_logits(logits, targets)

        self.assertEqual(signals.num_tokens, 2)
        for t in signals.tokens:
            self.assertFalse(math.isnan(t.entropy), f"Entropy is NaN at step {t.step}")
            self.assertFalse(math.isinf(t.entropy), f"Entropy is Inf at step {t.step}")
            self.assertFalse(math.isnan(t.probability), f"Prob is NaN at step {t.step}")
        self.assertFalse(math.isnan(signals.mean_entropy))
        self.assertFalse(math.isnan(signals.perplexity))

    def test_empty_response_handling(self):
        """Verify graceful handling when response is empty."""
        logits = torch.empty((0, 50), dtype=torch.float32)
        signals = compute_token_signals_from_logits(logits, [])
        self.assertEqual(signals.num_tokens, 0)
        self.assertEqual(signals.mean_token_prob, 0.0)
        self.assertEqual(signals.perplexity, 1.0)


if __name__ == "__main__":
    unittest.main()
