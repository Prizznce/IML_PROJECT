"""
Pilot generation and raw-logit signal extraction pipeline.

Executes deterministic generation on a representative sample of benchmark prompts
using Qwen/Qwen3.5-0.8B on CUDA GPU and extracts token-level and sequence-level
internal generation signals using the raw-logit signal extraction module.
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.signals.token_signals import extract_raw_sequence_signals, SequenceSignals

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_signals")


def format_model_input(
    prompt: str,
    context: str = "",
    source_dataset: str = "",
    original_response: str = "",
    use_chat_template: bool = True,
    tokenizer: Optional[AutoTokenizer] = None,
    system_instruction: str = "Answer the question directly, factually, and concisely. Do not provide preamble or internal thinking.",
) -> str:
    """
    Format conditioning text for the causal language model based on dataset provenance.

    Parameters:
        prompt (str): Input query or task instruction.
        context (str): Reference passage, evidence annotations, or metadata.
        source_dataset (str): Benchmark source ('halueval', 'fever', 'truthfulqa').
        original_response (str): Benchmark ground-truth answer or claim.
        use_chat_template (bool): If True, format via tokenizer.apply_chat_template
            with enable_thinking=False to suppress reasoning scratchpads.
        tokenizer (AutoTokenizer, optional): Tokenizer instance for applying chat templates.
        system_instruction (str): Directive enforcing direct, concise factual answers.

    Returns:
        str: Cleanly structured conditioning prompt for generation.
    """
    prompt_str = str(prompt).strip()
    context_str = str(context).strip() if context else ""
    orig_resp = str(original_response).strip() if original_response else ""

    if source_dataset == "halueval":
        user_content = f"Passage: {context_str}\n\nQuestion: {prompt_str}" if context_str else f"Question: {prompt_str}"
    elif source_dataset == "fever":
        user_content = f"{prompt_str}\nClaim: {orig_resp}" if orig_resp else f"{prompt_str}"
    elif source_dataset == "truthfulqa":
        user_content = f"Question: {prompt_str}"
    else:
        user_content = f"Context: {context_str}\n\nQuestion: {prompt_str}" if context_str else f"Question: {prompt_str}"

    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # Fallback to plain prompt format (v1 baseline)
    return f"{user_content}\nAnswer:"


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load tokenizer and causal language model with verified device placement.

    Parameters:
        model_id: Hugging Face model repository ID.
        device: CUDA device identifier.
        torch_dtype: Tensor precision (bfloat16 recommended for Ampere+).

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading tokenizer for '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading causal model '{model_id}' in {torch_dtype} on {device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map=device,
        )
    except (ValueError, ImportError):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
                local_files_only=True,
            )
        if torch.cuda.is_available() and "cuda" in str(device):
            model = model.to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            local_files_only=True,
        )
        if torch.cuda.is_available() and "cuda" in str(device):
            model = model.to(device)
    model.eval()

    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    vram_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    logger.info(f"Model successfully loaded on {device_name}. VRAM allocated: {vram_alloc:.2f} MB")

    return model, tokenizer


def validate_pilot_signals(
    df: pd.DataFrame,
    expected_count: int = 30,
    check_no_think_tags: bool = True,
) -> Dict[str, bool]:
    """
    Validate statistical and schema integrity of the pilot generation signals dataset.

    Checks:
      1. Exactly expected_count rows.
      2. No duplicate IDs.
      3. No missing generated responses.
      4. No NaN or inf values in any signal columns.
      5. Perplexity strictly positive (> 0).
      6. Probabilities bounded in [0, 1].
      7. Entropy non-negative (>= 0).
      8. Labels remain binary {0, 1}.
      9. source_dataset values belong to known benchmark set.
      10. (v2) Generated responses do not contain leaked reasoning <think> tags.

    Returns:
        Dict[str, bool]: Check names mapped to boolean pass/fail status.
    """
    checks: Dict[str, bool] = {}

    # 1. Row count
    checks["expected_row_count"] = (len(df) == expected_count)

    # 2. Unique IDs
    checks["unique_ids"] = (df["id"].nunique() == len(df))

    # 3. Non-empty generated responses
    has_empty_responses = (df["generated_response"].astype(str).str.strip() == "").sum() > 0
    checks["no_empty_responses"] = not has_empty_responses

    # 4. No NaN / inf in signal columns
    signal_cols = [
        "num_generated_tokens",
        "mean_token_prob",
        "min_token_prob",
        "mean_log_prob",
        "min_log_prob",
        "mean_entropy",
        "max_entropy",
        "entropy_std",
        "token_prob_std",
        "perplexity",
    ]
    has_nan_or_inf = False
    for col in signal_cols:
        if col not in df.columns:
            has_nan_or_inf = True
            break
        vals = df[col].to_numpy()
        if np.isnan(vals).any() or np.isinf(vals).any():
            has_nan_or_inf = True
            break
    checks["no_nan_or_inf_in_signals"] = not has_nan_or_inf

    # 5. Perplexity > 0
    checks["perplexity_positive"] = bool((df["perplexity"] > 0).all())

    # 6. Probabilities in [0, 1]
    prob_valid = bool(
        ((df["mean_token_prob"] >= 0.0) & (df["mean_token_prob"] <= 1.0 + 1e-5)).all()
        and ((df["min_token_prob"] >= 0.0) & (df["min_token_prob"] <= 1.0 + 1e-5)).all()
    )
    checks["probabilities_bounded_0_1"] = prob_valid

    # 7. Entropy >= 0
    entropy_valid = bool(
        ((df["mean_entropy"] >= -1e-5)).all()
        and ((df["max_entropy"] >= -1e-5)).all()
    )
    checks["entropy_non_negative"] = entropy_valid

    # 8. Binary labels {0, 1}
    unique_labels = set(df["label"].unique())
    checks["binary_labels"] = unique_labels.issubset({0, 1})

    # 9. Valid source datasets
    valid_sources = {"halueval", "truthfulqa", "fever"}
    checks["valid_source_datasets"] = set(df["source_dataset"].unique()).issubset(valid_sources)

    # 10. No <think> reasoning tags leaked into generated responses
    if check_no_think_tags:
        has_think_tags = (
            df["generated_response"].astype(str).str.contains("<think>")
            | df["generated_response"].astype(str).str.contains("</think>")
        ).sum() > 0
        checks["no_reasoning_think_tags"] = not has_think_tags

    return checks


def run_pilot_generation(
    data_path: str = "data/processed/combined_processed.parquet",
    sample_size: int = 10,
    seed: int = 42,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    device: str = "cuda:0",
    max_new_tokens: int = 128,
    output_parquet: str = "experiments/baselines/pilot_generation_signals_v2.parquet",
    output_csv: str = "experiments/baselines/pilot_generation_signals_v2.csv",
    use_chat_template: bool = True,
    system_instruction: str = "Answer the question directly, factually, and concisely. Do not provide preamble or internal thinking.",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute deterministic generation and signal extraction on a pilot sample.

    Parameters:
        data_path: Path to canonical combined processed parquet.
        sample_size: Number of examples to sample (default: 10).
        seed: Random seed for deterministic sampling.
        model_id: Hugging Face model identifier.
        device: CUDA device identifier.
        max_new_tokens: Maximum tokens to generate per prompt (default: 128).
        output_parquet: File path for output Parquet artifact.
        output_csv: File path for output CSV artifact.
        use_chat_template: If True, uses ChatML with enable_thinking=False.
        system_instruction: Direct factual answer directive.

    Returns:
        Tuple of (results_dataframe, performance_metrics_dict).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input dataset not found at '{data_path}'")

    logger.info(f"Reading dataset from '{data_path}'...")
    df_raw = pd.read_parquet(data_path)
    logger.info(f"Total rows in dataset: {len(df_raw)}. Drawing sample of {sample_size} (seed={seed})...")

    # Deterministic sampling
    pilot_sample = df_raw.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    logger.info(f"Sample breakdown: {pilot_sample['source_dataset'].value_counts().to_dict()} | "
                f"Labels: {pilot_sample['label'].value_counts().to_dict()}")

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)

    records: List[Dict[str, Any]] = []
    total_generated_tokens = 0
    step_times: List[float] = []

    logger.info(f"Starting deterministic generation (do_sample=False, max_new_tokens={max_new_tokens}, chat_template={use_chat_template})...")
    start_all_time = time.perf_counter()

    for idx, row in pilot_sample.iterrows():
        t_start = time.perf_counter()

        ex_id = str(row["id"])
        source = str(row["source_dataset"])
        label = int(row["label"])
        prompt = str(row["prompt"])
        context = str(row["context"]) if "context" in row and pd.notna(row["context"]) else ""
        orig_resp = str(row["response"]) if "response" in row and pd.notna(row["response"]) else ""

        # Format input conditioning prompt
        model_input = format_model_input(
            prompt=prompt,
            context=context,
            source_dataset=source,
            original_response=orig_resp,
            use_chat_template=use_chat_template,
            tokenizer=tokenizer,
            system_instruction=system_instruction,
        )

        # Tokenize prompt input
        enc = tokenizer(model_input, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        input_len = input_ids.shape[1]

        # Deterministic generation
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )

        # Slice generated response tokens
        gen_token_ids = gen_out[0][input_len:]
        num_new_tokens = len(gen_token_ids)
        total_generated_tokens += num_new_tokens

        # Decode generated text
        raw_gen_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
        clean_gen_text = raw_gen_text.strip()

        has_think_open = "<think>" in raw_gen_text
        has_think_close = "</think>" in raw_gen_text
        reached_limit = (num_new_tokens == max_new_tokens)

        # Extract internal generation signals using unadulterated forward pass
        signals: SequenceSignals = extract_raw_sequence_signals(
            model=model,
            tokenizer=tokenizer,
            prompt=model_input,
            response=raw_gen_text,
            device=device,
        )

        t_elapsed = time.perf_counter() - t_start
        step_times.append(t_elapsed)

        record = {
            "id": ex_id,
            "source_dataset": source,
            "label": label,
            "prompt": prompt,
            "context": context,
            "response": orig_resp,
            "original_response": orig_resp,
            "model_input": model_input,
            "generated_response": clean_gen_text,
            "num_generated_tokens": signals.num_tokens,
            "mean_token_prob": signals.mean_token_prob,
            "min_token_prob": signals.min_token_prob,
            "mean_log_prob": signals.mean_log_prob,
            "min_log_prob": signals.min_log_prob,
            "mean_entropy": signals.mean_entropy,
            "max_entropy": signals.max_entropy,
            "entropy_std": signals.entropy_std,
            "token_prob_std": signals.token_prob_std,
            "perplexity": signals.perplexity,
            "has_think": has_think_open,
            "has_think_close": has_think_close,
            "reached_limit": reached_limit,
            "generation_time_s": t_elapsed,
        }
        records.append(record)

        # Real-time progress display
        preview = clean_gen_text.replace("\n", " ")[:60]
        logger.info(
            f"[{idx + 1:02d}/{sample_size:02d}] ID: {ex_id} ({source}) | "
            f"Tokens: {signals.num_tokens} | Time: {t_elapsed:.2f}s | "
            f"Prob: {signals.mean_token_prob:.3f} | Entropy: {signals.mean_entropy:.3f} | "
            f"Limit: {reached_limit} | Think: {has_think_open} | "
            f"Preview: '{preview}'"
        )

    total_generation_time = time.perf_counter() - start_all_time
    avg_time_per_example = total_generation_time / sample_size if sample_size > 0 else 0.0
    avg_tokens_per_example = total_generated_tokens / sample_size if sample_size > 0 else 0.0

    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    results_df = pd.DataFrame(records)

    metrics = {
        "sample_size": sample_size,
        "total_generation_time_s": total_generation_time,
        "avg_time_per_example_s": avg_time_per_example,
        "total_generated_tokens": total_generated_tokens,
        "avg_tokens_per_example": avg_tokens_per_example,
        "min_generated_tokens": int(results_df["num_generated_tokens"].min()),
        "max_generated_tokens": int(results_df["num_generated_tokens"].max()),
        "think_count": int(results_df["has_think"].sum()),
        "think_close_count": int(results_df["has_think_close"].sum()),
        "reached_limit_count": int(results_df["reached_limit"].sum()),
        "mean_token_prob": float(results_df["mean_token_prob"].mean()),
        "mean_entropy": float(results_df["mean_entropy"].mean()),
        "mean_perplexity": float(results_df["perplexity"].mean()),
        "peak_vram_allocated_mb": peak_allocated_mb,
        "peak_vram_reserved_mb": peak_reserved_mb,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save artifacts
    results_df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved pilot Parquet artifact to '{output_parquet}' ({os.path.getsize(output_parquet)} bytes)")

    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Saved pilot CSV artifact to '{output_csv}' ({os.path.getsize(output_csv)} bytes)")

    return results_df, metrics


def main():
    parser = argparse.ArgumentParser(description="Deterministic generation and signal extraction pilot (v2).")
    parser.add_argument("--data-path", default="data/processed/combined_processed.parquet", help="Input dataset path")
    parser.add_argument("--sample-size", type=int, default=10, help="Pilot sample size (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model ID")
    parser.add_argument("--device", default="cuda:0", help="Execution device")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate (default: 128)")
    parser.add_argument("--output-parquet", default="experiments/baselines/pilot_generation_signals_v2.parquet")
    parser.add_argument("--output-csv", default="experiments/baselines/pilot_generation_signals_v2.csv")
    parser.add_argument("--raw-prompt", action="store_true", help="Use legacy raw prompt rather than ChatML template")
    args = parser.parse_args()

    results_df, metrics = run_pilot_generation(
        data_path=args.data_path,
        sample_size=args.sample_size,
        seed=args.seed,
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
        use_chat_template=not args.raw_prompt,
    )

    logger.info("=" * 60)
    logger.info("PILOT GENERATION PERFORMANCE SUMMARY (v2)")
    logger.info("=" * 60)
    logger.info(f"Total rows processed: {metrics['sample_size']}")
    logger.info(f"Total generation time: {metrics['total_generation_time_s']:.2f} s")
    logger.info(f"Average time / example: {metrics['avg_time_per_example_s']:.2f} s")
    logger.info(f"Total generated tokens: {metrics['total_generated_tokens']}")
    logger.info(f"Average generated tokens / example: {metrics['avg_tokens_per_example']:.1f}")
    logger.info(f"Min tokens: {metrics['min_generated_tokens']} | Max tokens: {metrics['max_generated_tokens']}")
    logger.info(f"Reached token limit ({args.max_new_tokens}): {metrics['reached_limit_count']} / {metrics['sample_size']}")
    logger.info(f"Responses containing <think>: {metrics['think_count']} / {metrics['sample_size']}")
    logger.info(f"Responses containing </think>: {metrics['think_close_count']} / {metrics['sample_size']}")
    logger.info(f"Mean token probability: {metrics['mean_token_prob']:.4f}")
    logger.info(f"Mean entropy: {metrics['mean_entropy']:.4f}")
    logger.info(f"Mean perplexity: {metrics['mean_perplexity']:.4f}")
    logger.info(f"Peak VRAM allocated: {metrics['peak_vram_allocated_mb']:.2f} MB")
    logger.info(f"Peak VRAM reserved:  {metrics['peak_vram_reserved_mb']:.2f} MB")

    logger.info("=" * 60)
    logger.info("RUNNING QUALITY & INTEGRITY VALIDATIONS")
    logger.info("=" * 60)
    validation_results = validate_pilot_signals(
        results_df,
        expected_count=args.sample_size,
        check_no_think_tags=not args.raw_prompt,
    )
    all_passed = True
    for check_name, passed in validation_results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        logger.info(f"  [{status}] {check_name}")

    if not all_passed:
        logger.error("Validation checks FAILED! Inspect anomalies above.")
        sys.exit(1)
    else:
        logger.info("ALL VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
