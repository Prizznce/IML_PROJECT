"""
Raw-logit signal extraction module for real-time LLM hallucination detection.

Computes mathematically rigorous token-level and sequence-level internal generation
signals from unadulterated causal language model forward passes:
  - Raw token probabilities: P(y_t | x, y_<t)
  - Token log probabilities: log P(y_t | x, y_<t)
  - Predictive entropy: H_t = -sum_v P(v) log P(v)
  - Token rank in unconstrained vocabulary distribution
  - Sequence-level dispersion statistics (min/mean prob, min/mean log-prob,
    mean/max entropy, entropy std, prob std, sequence perplexity).

Strictly isolates generated response tokens from prompt tokens and enforces
causal shifting (logit at index t predicts token at index t+1).
"""

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F


@dataclass
class TokenSignal:
    """Detailed internal generation signals for an individual generated token."""
    step: int
    token_id: int
    token_text: str
    probability: float
    log_probability: float
    entropy: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "probability": self.probability,
            "log_probability": self.log_probability,
            "entropy": self.entropy,
            "rank": self.rank,
        }


@dataclass
class SequenceSignals:
    """Aggregated sequence-level signals computed over response tokens."""
    num_tokens: int
    mean_token_prob: float
    min_token_prob: float
    mean_log_prob: float
    min_log_prob: float
    mean_entropy: float
    max_entropy: float
    entropy_std: float
    token_prob_std: float
    perplexity: float
    tokens: List[TokenSignal] = field(default_factory=list)

    def to_dict(self, include_tokens: bool = False) -> Dict[str, Any]:
        """Convert sequence metrics to a flat dictionary suitable for tabular ML."""
        data = {
            "num_tokens": self.num_tokens,
            "mean_token_prob": self.mean_token_prob,
            "min_token_prob": self.min_token_prob,
            "mean_log_prob": self.mean_log_prob,
            "min_log_prob": self.min_log_prob,
            "mean_entropy": self.mean_entropy,
            "max_entropy": self.max_entropy,
            "entropy_std": self.entropy_std,
            "token_prob_std": self.token_prob_std,
            "perplexity": self.perplexity,
        }
        if include_tokens:
            data["tokens"] = [t.to_dict() for t in self.tokens]
        return data


def compute_token_signals_from_logits(
    response_logits: torch.Tensor,
    target_token_ids: Sequence[int],
    tokenizer: Optional[Any] = None,
    exclude_special_tokens: bool = False,
) -> SequenceSignals:
    """
    Compute token and sequence signals given aligned response logits and target tokens.

    Parameters:
        response_logits (torch.Tensor): Logits of shape (R, Vocab_Size) or (1, R, Vocab_Size),
            where each row t corresponds to the prediction for target_token_ids[t].
        target_token_ids (Sequence[int]): Target generated token IDs of length R.
        tokenizer (Optional[Any]): Tokenizer for decoding token text. If None,
            repr of token_id is used.
        exclude_special_tokens (bool): If True and tokenizer is provided, tokens
            matching tokenizer.all_special_ids are ignored in sequence stats.

    Returns:
        SequenceSignals: Complete token-level and sequence-level metrics.
    """
    if response_logits.dim() == 3:
        if response_logits.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got shape {response_logits.shape}")
        response_logits = response_logits[0]

    num_steps = response_logits.size(0)
    if num_steps != len(target_token_ids):
        raise ValueError(
            f"Dimension mismatch: response_logits has {num_steps} steps, "
            f"but target_token_ids has {len(target_token_ids)} tokens."
        )

    if num_steps == 0:
        return SequenceSignals(
            num_tokens=0,
            mean_token_prob=0.0,
            min_token_prob=0.0,
            mean_log_prob=0.0,
            min_log_prob=0.0,
            mean_entropy=0.0,
            max_entropy=0.0,
            entropy_std=0.0,
            token_prob_std=0.0,
            perplexity=1.0,
            tokens=[],
        )

    special_ids = set()
    if exclude_special_tokens and tokenizer is not None and hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)

    token_signals: List[TokenSignal] = []
    included_probs: List[float] = []
    included_log_probs: List[float] = []
    included_entropies: List[float] = []

    # Ensure float32 precision for numerical stability
    logits_f32 = response_logits.float()

    for step_idx in range(num_steps):
        target_id = int(target_token_ids[step_idx])
        step_logit = logits_f32[step_idx]

        # Log-softmax and softmax
        log_probs = F.log_softmax(step_logit, dim=-1)
        probs = F.softmax(step_logit, dim=-1)

        target_log_prob = log_probs[target_id].item()
        target_prob = probs[target_id].item()

        # Numerically stable predictive entropy:
        # H = -sum_{v} P(v) * log P(v)
        # Handle lim (p -> 0) p * log(p) = 0 to prevent 0 * (-inf) -> NaN
        safe_p_log_p = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
        entropy = -safe_p_log_p.sum().item()
        # Clamp tiny negative floating artifacts (e.g. -0.0)
        if entropy < 0.0:
            entropy = 0.0

        # Rank of target token (1-based, 1 = argmax/top-1 prediction)
        target_val = step_logit[target_id]
        rank = (step_logit > target_val).sum().item() + 1

        # Decode token text
        if tokenizer is not None:
            try:
                token_text = tokenizer.decode([target_id])
            except Exception:
                token_text = str(target_id)
        else:
            token_text = str(target_id)

        sig = TokenSignal(
            step=step_idx,
            token_id=target_id,
            token_text=token_text,
            probability=target_prob,
            log_probability=target_log_prob,
            entropy=entropy,
            rank=rank,
        )
        token_signals.append(sig)

        if target_id not in special_ids:
            included_probs.append(target_prob)
            included_log_probs.append(target_log_prob)
            included_entropies.append(entropy)

    # If all tokens were filtered out as special tokens, fall back to all tokens
    if not included_probs:
        included_probs = [t.probability for t in token_signals]
        included_log_probs = [t.log_probability for t in token_signals]
        included_entropies = [t.entropy for t in token_signals]

    count = len(included_probs)
    mean_token_prob = sum(included_probs) / count
    min_token_prob = min(included_probs)
    mean_log_prob = sum(included_log_probs) / count
    min_log_prob = min(included_log_probs)
    mean_entropy = sum(included_entropies) / count
    max_entropy = max(included_entropies)

    entropy_std = statistics.stdev(included_entropies) if count > 1 else 0.0
    token_prob_std = statistics.stdev(included_probs) if count > 1 else 0.0

    # Perplexity = exp(-mean_log_prob)
    # Clamp exponent to prevent overflow in extreme underflow cases
    try:
        perplexity = math.exp(min(-mean_log_prob, 100.0))
    except OverflowError:
        perplexity = float("inf")

    return SequenceSignals(
        num_tokens=count,
        mean_token_prob=mean_token_prob,
        min_token_prob=min_token_prob,
        mean_log_prob=mean_log_prob,
        min_log_prob=min_log_prob,
        mean_entropy=mean_entropy,
        max_entropy=max_entropy,
        entropy_std=entropy_std,
        token_prob_std=token_prob_std,
        perplexity=perplexity,
        tokens=token_signals,
    )


def extract_raw_sequence_signals(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: Optional[Union[str, torch.device]] = None,
    exclude_special_tokens: bool = False,
) -> SequenceSignals:
    """
    Extract raw-logit signals by performing a forward pass over prompt + response.

    This function operates on RAW model logits from a complete forward pass,
    completely independent of generation-time sampling transforms (temperature,
    top-k, top-p, min-p, repetition penalty).

    Causal Alignment:
      If sequence has N tokens [x_0, ..., x_{P-1}, y_0, ..., y_{R-1}] where:
        Prompt tokens = [x_0, ..., x_{P-1}] (length P)
        Response tokens = [y_0, ..., y_{R-1}] (length R)
      Then:
        The first response token y_0 is predicted by the logit at index P - 1.
        Response token y_t is predicted by the logit at index P - 1 + t.
        Prompt tokens [0 .. P-1] are strictly excluded from response metrics.

    Parameters:
        model: Causal language model (e.g. Qwen3_5ForCausalLM).
        tokenizer: Fast or standard tokenizer.
        prompt (str): Input query / conditioning prompt.
        response (str): Generated candidate response or ground-truth answer.
        device: Target execution device. If None, uses model parameter device.
        exclude_special_tokens (bool): If True, skips special tokens in aggregate stats.

    Returns:
        SequenceSignals: Complete token-level and sequence-level metrics.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Tokenize prompt and response
    # Prompt is encoded with default special tokens (e.g. BOS if applicable)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # Response is encoded without re-adding BOS
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    p_len = len(prompt_ids)
    r_len = len(response_ids)

    if r_len == 0:
        return SequenceSignals(
            num_tokens=0,
            mean_token_prob=0.0,
            min_token_prob=0.0,
            mean_log_prob=0.0,
            min_log_prob=0.0,
            mean_entropy=0.0,
            max_entropy=0.0,
            entropy_std=0.0,
            token_prob_std=0.0,
            perplexity=1.0,
            tokens=[],
        )

    full_input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    total_len = full_input_ids.size(1)

    model.eval()
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits  # shape: (1, total_len, vocab_size)

    # Causal shift extraction:
    # Logits predicting response tokens y_0 ... y_{R-1} start at index p_len - 1
    # up to total_len - 2 (inclusive), which has length r_len.
    # Target tokens are full_input_ids[:, p_len : total_len].
    response_logits = logits[0, p_len - 1 : total_len - 1, :]

    return compute_token_signals_from_logits(
        response_logits=response_logits,
        target_token_ids=response_ids,
        tokenizer=tokenizer,
        exclude_special_tokens=exclude_special_tokens,
    )
