"""
Supervised Signal Extraction Pipeline for Labeled Benchmark Responses.

Calculates internal generation signals (probabilities, log-probs, predictive entropy,
perplexity, token rank dispersion) strictly over the ORIGINAL labeled response tokens
from benchmark datasets (HaluEval, TruthfulQA, FEVER).

Preserves the exact ground-truth benchmark label (0 = Faithful, 1 = Hallucinated)
without label mismatch caused by generating arbitrary new text. Includes periodic
checkpointing and interrupt-resume capability for large-scale runs.
"""

import argparse
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.signals.token_signals import extract_raw_sequence_signals, SequenceSignals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("score_labeled_responses")


def format_scoring_prompt(
    prompt: str,
    context: str = "",
    source_dataset: str = "",
    use_chat_template: bool = True,
    tokenizer: Optional[Any] = None,
    system_instruction: str = "Answer the question directly, factually, and concisely. Do not provide preamble or internal thinking.",
) -> str:
    """
    Format conditioning prefix for evaluating an original benchmark response.

    Parameters:
        prompt: Benchmark question or task instruction.
        context: Background passage, evidence citation, or metadata.
        source_dataset: Benchmark origin ('halueval', 'truthfulqa', 'fever').
        use_chat_template: If True, uses ChatML with enable_thinking=False.
        tokenizer: Tokenizer instance for chat template formatting.
        system_instruction: System directive for factual responses.

    Returns:
        str: Prompt string preceding the response continuation.
    """
    prompt_str = str(prompt).strip()
    context_str = str(context).strip() if context else ""

    if source_dataset == "halueval":
        user_content = f"Passage: {context_str}\n\nQuestion: {prompt_str}" if context_str else f"Question: {prompt_str}"
    elif source_dataset == "fever":
        user_content = f"Verify whether the following claim is supported by factual evidence.\nClaim: "
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

    return f"{user_content}\nAnswer:"


def score_single_response(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Score a single labeled response sequence given conditioning prompt.

    Computes token-level raw logit signals and sequence-level aggregations
    including rank dispersion statistics.
    """
    signals: SequenceSignals = extract_raw_sequence_signals(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        response=response,
        device=device,
    )

    ranks = [t.rank for t in signals.tokens] if signals.tokens else []
    mean_token_rank = float(sum(ranks) / len(ranks)) if ranks else 1.0
    max_token_rank = int(max(ranks)) if ranks else 1
    min_token_rank = int(min(ranks)) if ranks else 1
    rank_std = float(statistics.stdev(ranks)) if len(ranks) > 1 else 0.0

    return {
        "num_tokens": signals.num_tokens,
        "mean_token_prob": signals.mean_token_prob,
        "min_token_prob": signals.min_token_prob,
        "mean_log_prob": signals.mean_log_prob,
        "min_log_prob": signals.min_log_prob,
        "mean_entropy": signals.mean_entropy,
        "max_entropy": signals.max_entropy,
        "entropy_std": signals.entropy_std,
        "token_prob_std": signals.token_prob_std,
        "perplexity": signals.perplexity,
        "mean_token_rank": mean_token_rank,
        "max_token_rank": max_token_rank,
        "min_token_rank": min_token_rank,
        "rank_std": rank_std,
    }


def validate_scored_dataset(df: pd.DataFrame, expected_count: Optional[int] = None) -> Dict[str, bool]:
    """
    Validate schema and mathematical integrity of the scored dataset.
    """
    checks: Dict[str, bool] = {}

    if expected_count is not None:
        checks["expected_row_count"] = (len(df) == expected_count)

    checks["unique_ids"] = (df["id"].nunique() == len(df))
    checks["no_empty_responses"] = bool((df["response"].astype(str).str.strip() != "").all())

    signal_cols = [
        "num_tokens", "mean_token_prob", "min_token_prob",
        "mean_log_prob", "min_log_prob", "mean_entropy",
        "max_entropy", "entropy_std", "token_prob_std", "perplexity",
        "mean_token_rank", "max_token_rank", "min_token_rank", "rank_std"
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

    checks["perplexity_positive"] = bool((df["perplexity"] > 0).all())
    checks["probabilities_bounded_0_1"] = bool(
        ((df["mean_token_prob"] >= 0.0) & (df["mean_token_prob"] <= 1.0 + 1e-5)).all()
        and ((df["min_token_prob"] >= 0.0) & (df["min_token_prob"] <= 1.0 + 1e-5)).all()
    )
    checks["entropy_non_negative"] = bool(
        (df["mean_entropy"] >= -1e-5).all() and (df["max_entropy"] >= -1e-5).all()
    )
    checks["binary_labels"] = set(df["label"].unique()).issubset({0, 1})
    valid_sources = {"halueval", "truthfulqa", "fever"}
    checks["valid_source_datasets"] = set(df["source_dataset"].unique()).issubset(valid_sources)

    return checks


def score_labeled_dataset(
    input_path: str = "data/processed/combined_processed.parquet",
    output_parquet: str = "experiments/baselines/supervised_signals_combined.parquet",
    output_csv: str = "experiments/baselines/supervised_signals_combined.csv",
    checkpoint_path: Optional[str] = None,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    device: str = "cuda:0",
    limit: Optional[int] = None,
    seed: int = 42,
    checkpoint_interval: int = 100,
    resume: bool = True,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score labeled benchmark responses using unadulterated forward passes.

    Parameters:
        input_path: Path to processed parquet benchmark.
        output_parquet: Destination path for final parquet artifact.
        output_csv: Destination path for final CSV artifact.
        checkpoint_path: Intermediate checkpoint file path.
        model_id: Hugging Face model ID.
        device: CUDA device identifier.
        limit: Optional sample size limit (for dry-runs or pilot verification).
        seed: Random seed if limit is used.
        checkpoint_interval: Save progress every N completed examples.
        resume: If True, detects and resumes from existing checkpoint.
        model: Pre-loaded model instance (optional).
        tokenizer: Pre-loaded tokenizer instance (optional).

    Returns:
        Tuple of (results_dataframe, summary_metrics).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input benchmark not found at '{input_path}'")

    if checkpoint_path is None:
        checkpoint_path = f"{output_parquet}.checkpoint.parquet"

    logger.info(f"Loading input dataset from '{input_path}'...")
    df_input = pd.read_parquet(input_path)
    total_raw_rows = len(df_input)
    logger.info(f"Loaded {total_raw_rows} examples from '{input_path}'.")

    if limit is not None and limit < total_raw_rows:
        logger.info(f"Subsampling {limit} examples (seed={seed})...")
        df_input = df_input.sample(n=limit, random_state=seed).reset_index(drop=True)
    else:
        df_input = df_input.reset_index(drop=True)

    target_count = len(df_input)

    # Resume handling
    completed_records: List[Dict[str, Any]] = []
    completed_ids: Set[str] = set()

    if resume and os.path.exists(checkpoint_path):
        try:
            df_ckpt = pd.read_parquet(checkpoint_path)
            completed_records = df_ckpt.to_dict(orient="records")
            completed_ids = set(df_ckpt["id"].astype(str))
            logger.info(f"Resuming from checkpoint '{checkpoint_path}': {len(completed_ids)}/{target_count} already completed.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint '{checkpoint_path}' ({e}). Starting fresh.")
            completed_records = []
            completed_ids = set()

    # Filter out already scored examples
    remaining_indices = [i for i, row in df_input.iterrows() if str(row["id"]) not in completed_ids]
    logger.info(f"Examples remaining to score: {len(remaining_indices)} of {target_count}.")

    if not remaining_indices and completed_records:
        logger.info("All examples already scored in checkpoint. Assembling final outputs...")
        results_df = pd.DataFrame(completed_records)
        os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
        results_df.to_parquet(output_parquet, index=False)
        results_df.to_csv(output_csv, index=False, encoding="utf-8")
        return results_df, {"total_scored": len(results_df), "resumed_complete": True}

    # Load model & tokenizer if not passed
    if model is None or tokenizer is None:
        logger.info(f"Loading tokenizer for '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model '{model_id}' in bfloat16 on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start_all = time.perf_counter()
    newly_scored = 0

    for idx in remaining_indices:
        t_ex_start = time.perf_counter()
        row = df_input.iloc[idx]

        ex_id = str(row["id"])
        source = str(row["source_dataset"])
        label = int(row["label"])
        prompt = str(row["prompt"])
        context = str(row["context"]) if "context" in row and pd.notna(row["context"]) else ""
        resp = str(row["response"]).strip()

        # Format input conditioning prefix
        scoring_prompt = format_scoring_prompt(
            prompt=prompt,
            context=context,
            source_dataset=source,
            use_chat_template=True,
            tokenizer=tokenizer,
        )

        # Forward pass & signal extraction strictly over original response
        metrics = score_single_response(
            model=model,
            tokenizer=tokenizer,
            prompt=scoring_prompt,
            response=resp,
            device=device,
        )

        dt = time.perf_counter() - t_ex_start

        record = {
            "id": ex_id,
            "source_dataset": source,
            "label": label,
            "prompt": prompt,
            "context": context,
            "response": resp,
            "model_input": scoring_prompt,
            "forward_time_s": dt,
            **metrics,
        }
        completed_records.append(record)
        completed_ids.add(ex_id)
        newly_scored += 1

        # Real-time progress logging
        if newly_scored % 10 == 0 or newly_scored == len(remaining_indices) or newly_scored <= 5:
            logger.info(
                f"[{len(completed_records):04d}/{target_count:04d}] ID: {ex_id} ({source}) | "
                f"Label: {label} | Tokens: {metrics['num_tokens']} | Time: {dt*1000:.1f}ms | "
                f"Prob: {metrics['mean_token_prob']:.3f} | Entropy: {metrics['mean_entropy']:.3f} | "
                f"Rank: {metrics['mean_token_rank']:.1f} | PPL: {metrics['perplexity']:.2f}"
            )

        # Periodic checkpoint save
        if newly_scored % checkpoint_interval == 0:
            df_temp = pd.DataFrame(completed_records)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            df_temp.to_parquet(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved: {len(completed_records)} records -> '{checkpoint_path}'")

    total_time = time.perf_counter() - t_start_all
    avg_time_per_ex = total_time / newly_scored if newly_scored > 0 else 0.0

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    peak_res = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0

    results_df = pd.DataFrame(completed_records)

    # Save final artifacts
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results_df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved final scored Parquet artifact to '{output_parquet}' ({len(results_df)} rows, {os.path.getsize(output_parquet)} bytes)")

    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Saved final scored CSV artifact to '{output_csv}' ({len(results_df)} rows, {os.path.getsize(output_csv)} bytes)")

    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Removed transient checkpoint '{checkpoint_path}'.")
        except OSError:
            pass

    summary = {
        "total_rows": len(results_df),
        "newly_scored": newly_scored,
        "total_time_s": total_time,
        "avg_time_ms": avg_time_per_ex * 1000,
        "peak_vram_alloc_mb": peak_alloc,
        "peak_vram_res_mb": peak_res,
        "mean_token_prob": float(results_df["mean_token_prob"].mean()),
        "mean_entropy": float(results_df["mean_entropy"].mean()),
        "mean_perplexity": float(results_df["perplexity"].mean()),
        "mean_token_rank": float(results_df["mean_token_rank"].mean()),
    }

    return results_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Supervised signal extraction pipeline scoring original benchmark responses."
    )
    parser.add_argument(
        "--dataset",
        default="combined",
        choices=["combined", "halueval", "truthfulqa", "fever"],
        help="Target benchmark dataset to score (default: combined)",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional override path for input dataset",
    )
    parser.add_argument(
        "--output-parquet",
        default=None,
        help="Optional override path for output parquet",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional override path for output CSV",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional override path for checkpoint file",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model ID")
    parser.add_argument("--device", default="cuda:0", help="Execution device")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint save frequency")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")
    args = parser.parse_args()

    # Resolve default paths
    input_path = args.input_path
    if input_path is None:
        if args.dataset == "combined":
            input_path = "data/processed/combined_processed.parquet"
        else:
            input_path = f"data/processed/{args.dataset}_processed.parquet"

    output_parquet = args.output_parquet
    if output_parquet is None:
        output_parquet = f"experiments/baselines/supervised_signals_{args.dataset}.parquet"

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = f"experiments/baselines/supervised_signals_{args.dataset}.csv"

    results_df, summary = score_labeled_dataset(
        input_path=input_path,
        output_parquet=output_parquet,
        output_csv=output_csv,
        checkpoint_path=args.checkpoint_path,
        model_id=args.model_id,
        device=args.device,
        limit=args.limit,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
    )

    logger.info("=" * 60)
    logger.info("SUPERVISED SCORING SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    logger.info("=" * 60)
    logger.info("VALIDATING SCORED DATASET")
    logger.info("=" * 60)
    val_results = validate_scored_dataset(results_df, expected_count=args.limit)
    all_passed = True
    for check_name, passed in val_results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        logger.info(f"  [{status}] {check_name}")

    if not all_passed:
        logger.error("Validation failed!")
        sys.exit(1)
    logger.info("ALL VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
