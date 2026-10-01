"""
Unit tests validating multi-model architecture and support for Llama 3.2 1B alongside Qwen 3.5 0.8B.
Tests model slugging, directory routing, chat template fallback, token extraction, and classifier retrieval.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.generation.generate_signals import format_model_input
from src.signals.score_labeled_responses import format_scoring_prompt
from app.inference import get_trained_classifiers
from src.utils.model_registry import (
    DEFAULT_MODEL_ID,
    SUPPORTED_MODELS,
    get_hf_token,
    get_model_experiment_dir,
    get_model_slug,
    get_saved_model_dir,
    get_universal_features_path,
)


class MockQwenTokenizer:
    """Mock tokenizer supporting Qwen's enable_thinking parameter."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        res = "<im_start>"
        for m in messages:
            res += f"[{m['role']}]: {m['content']} "
        if add_generation_prompt:
            res += "<im_start>assistant\n"
            if not enable_thinking:
                res += "<think></think>"
        return res


class MockLlamaTokenizer:
    """Mock tokenizer simulating Llama 3.2 (does NOT accept enable_thinking)."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
        res = "<|begin_of_text|>"
        for m in messages:
            res += f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>"
        if add_generation_prompt:
            res += "<|start_header_id|>assistant<|end_header_id|>\n"
        return res


class MockBaseTokenizer:
    """Mock tokenizer without apply_chat_template (base models)."""
    pass


def test_model_slug_generation():
    """Verify clean, deterministic slugs for all supported models."""
    assert get_model_slug("Qwen/Qwen3.5-0.8B") == "qwen3_5_0_8b"
    assert get_model_slug("meta-llama/Llama-3.2-1B-Instruct") == "llama_3_2_1b_instruct"
    assert get_model_slug("meta-llama/Llama-3.2-1B") == "llama_3_2_1b"
    assert get_model_slug("google/gemma-3-1b-it") == "gemma_3_1b_it"
    assert get_model_slug("custom-org/My_Model-V1") == "my_model_v1"


def test_model_directory_resolution():
    """Verify distinct, isolated experiment and model directories."""
    qwen_exp = get_model_experiment_dir("Qwen/Qwen3.5-0.8B")
    llama_exp = get_model_experiment_dir("meta-llama/Llama-3.2-1B-Instruct")

    assert qwen_exp != llama_exp
    assert "llama_3_2_1b_instruct" in str(llama_exp)

    llama_save = get_saved_model_dir("meta-llama/Llama-3.2-1B-Instruct")
    assert "llama_3_2_1b_instruct" in str(llama_save)


def test_qwen_chat_template_formatting():
    """Verify Qwen chat template utilizes enable_thinking=False."""
    tok = MockQwenTokenizer()
    formatted = format_model_input(
        prompt="Capital of France?",
        context="Geography",
        source_dataset="halueval",
        use_chat_template=True,
        tokenizer=tok,
    )
    assert "<think></think>" in formatted
    assert "Capital of France?" in formatted


def test_llama_chat_template_fallback():
    """Verify Llama chat template gracefully handles and drops enable_thinking without errors."""
    tok = MockLlamaTokenizer()
    formatted = format_model_input(
        prompt="Capital of France?",
        context="Geography",
        source_dataset="halueval",
        use_chat_template=True,
        tokenizer=tok,
    )
    assert "<|start_header_id|>user<|end_header_id|>" in formatted
    assert "Capital of France?" in formatted
    assert "<think>" not in formatted


def test_base_model_plain_format_fallback():
    """Verify base models lacking chat templates fall back to clean prompt: Answer: format."""
    tok = MockBaseTokenizer()
    formatted = format_model_input(
        prompt="Capital of France?",
        context="",
        source_dataset="truthfulqa",
        use_chat_template=True,
        tokenizer=tok,
    )
    assert "Question: Capital of France?\nAnswer:" in formatted


def test_score_labeled_responses_prompt_formatting():
    """Verify score_labeled_responses format_scoring_prompt handles both Qwen and Llama."""
    qwen_tok = MockQwenTokenizer()
    llama_tok = MockLlamaTokenizer()

    q_out = format_scoring_prompt("Prompt A", tokenizer=qwen_tok, use_chat_template=True)
    l_out = format_scoring_prompt("Prompt A", tokenizer=llama_tok, use_chat_template=True)

    assert "Prompt A" in q_out
    assert "Prompt A" in l_out
    assert "<think></think>" in q_out
    assert "<|start_header_id|>" in l_out


def test_get_trained_classifiers_qwen():
    """Verify existing Qwen baseline classifiers load and evaluate cleanly."""
    lr, xgb = get_trained_classifiers(model_id="Qwen/Qwen3.5-0.8B")
    assert lr is not None
    assert xgb is not None
    assert hasattr(lr, "predict_proba")
    assert hasattr(xgb, "predict_proba")


def test_get_trained_classifiers_llama_fallback():
    """Verify Llama falls back gracefully to baseline classifiers if not yet retrained."""
    lr, xgb = get_trained_classifiers(model_id="meta-llama/Llama-3.2-1B-Instruct")
    assert lr is not None
    assert xgb is not None
    assert hasattr(lr, "predict_proba")
    assert hasattr(xgb, "predict_proba")
