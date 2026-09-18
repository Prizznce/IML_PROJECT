"""
NLI Agreement Signal Extraction and Evaluation Pipeline.

Implements bidirectional pairwise Natural Language Inference (NLI) across independent
LLM generations to measure entailment, contradiction, and neutrality.

Evaluates directional premise -> hypothesis pairs using a lightweight cross-encoder model
(cross-encoder/nli-MiniLM2-L6-H768), aggregates directional probabilities, and extracts
example-level consensus and disagreement features.
"""

import argparse
import itertools
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nli_agreement")


# ==============================================================================
# 1. Model Loading & Label Mapping
# ==============================================================================

def get_nli_label_indices(config_or_id2label: Union[Any, Dict[int, str]]) -> Dict[str, int]:
    """
    Dynamically resolve index mapping for contradiction, entailment, and neutral.

    Parameters:
        config_or_id2label: HuggingFace model config or id2label dictionary.

    Returns:
        Dict[str, int]: Canonical mapping {'contradiction': idx, 'entailment': idx, 'neutral': idx}.
    """
    id2label: Dict[int, str]
    if hasattr(config_or_id2label, "id2label"):
        id2label = {int(k): str(v).lower() for k, v in config_or_id2label.id2label.items()}
    elif isinstance(config_or_id2label, dict):
        id2label = {int(k): str(v).lower() for k, v in config_or_id2label.items()}
    else:
        # Default fallback standard MNLI
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    mapping: Dict[str, int] = {}
    for idx, label_str in id2label.items():
        if "contra" in label_str:
            mapping["contradiction"] = idx
        elif "entail" in label_str:
            mapping["entailment"] = idx
        elif "neut" in label_str:
            mapping["neutral"] = idx

    # Verify all 3 classes resolved
    for required in ["contradiction", "entailment", "neutral"]:
        if required not in mapping:
            raise ValueError(f"Could not resolve label index for '{required}' from id2label: {id2label}")

    return mapping


def load_nli_model(
    model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
    device: str = "cuda:0",
) -> Tuple[CrossEncoder, Dict[str, int]]:
    """
    Load cross-encoder NLI model with device placement and resolved label indices.

    Parameters:
        model_name: HuggingFace model repository identifier.
        device: CUDA or CPU target device string.

    Returns:
        Tuple of (CrossEncoder instance, label_indices_dict).
    """
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    logger.info(f"Loading CrossEncoder '{model_name}' on device '{dev}'...")
    t0 = time.perf_counter()
    model = CrossEncoder(model_name, device=dev)
    load_time = time.perf_counter() - t0
    logger.info(f"CrossEncoder successfully loaded in {load_time:.2f}s.")

    label_indices = get_nli_label_indices(model.model.config)
    logger.info(f"Resolved NLI label indices: {label_indices}")

    return model, label_indices


# ==============================================================================
# 2. Probability Computation & Directional Aggregation
# ==============================================================================

def compute_softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities along the last axis in a numerically stable manner.

    Parameters:
        logits: Numpy array of shape (N, 3) or (3,).

    Returns:
        Numpy array of probabilities summing to 1.0.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        shifted = arr - np.max(arr)
        exp_arr = np.exp(shifted)
        sum_exp = np.sum(exp_arr)
        return exp_arr / (sum_exp if sum_exp > 0 else 1.0)
    else:
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        exp_arr = np.exp(shifted)
        sum_exp = np.sum(exp_arr, axis=-1, keepdims=True)
        sum_exp[sum_exp == 0] = 1.0
        return exp_arr / sum_exp


def aggregate_directional_probabilities(
    probs_a_to_b: Dict[str, float],
    probs_b_to_a: Dict[str, float],
) -> Dict[str, float]:
    """
    Aggregate directional NLI probabilities for an unordered response pair (A, B).

    Formula:
        pair_entailment    = mean(entail(A -> B), entail(B -> A))
        pair_contradiction = mean(contra(A -> B), contra(B -> A))
        pair_neutral       = mean(neutral(A -> B), neutral(B -> A))

    Parameters:
        probs_a_to_b: Softmax probabilities for premise A, hypothesis B.
        probs_b_to_a: Softmax probabilities for premise B, hypothesis A.

    Returns:
        Dict[str, float]: Aggregated probabilities and consensus predicted class.
    """
    p_entail = 0.5 * (probs_a_to_b["entailment"] + probs_b_to_a["entailment"])
    p_contra = 0.5 * (probs_a_to_b["contradiction"] + probs_b_to_a["contradiction"])
    p_neutral = 0.5 * (probs_a_to_b["neutral"] + probs_b_to_a["neutral"])

    # Determine predicted class via argmax of aggregated probabilities
    class_probs = {
        "entailment": p_entail,
        "contradiction": p_contra,
        "neutral": p_neutral,
    }
    pred_class = max(class_probs.items(), key=lambda x: x[1])[0]

    return {
        "pair_entailment": float(p_entail),
        "pair_contradiction": float(p_contra),
        "pair_neutral": float(p_neutral),
        "predicted_class": pred_class,
    }


def calculate_nli_disagreement(
    mean_pairwise_contradiction: float,
    mean_pairwise_neutral: float,
) -> float:
    """
    Calculate scalar NLI disagreement score from pairwise contradiction and neutrality.

    Formula:
        nli_disagreement = mean_pairwise_contradiction + 0.5 * mean_pairwise_neutral

    Interpretation:
        - Quantifies departure from mutual logical entailment.
        - Direct factual contradiction incurs full disagreement penalty (weight 1.0).
        - Neutrality (failure to entail / semantic divergence) incurs intermediate penalty (weight 0.5).
        - Mutual entailment incurs zero penalty (weight 0.0).
        - Strictly bounded in [0.0, 1.0] since probabilities sum to 1.0.

    Parameters:
        mean_pairwise_contradiction: Average pairwise contradiction probability.
        mean_pairwise_neutral: Average pairwise neutral probability.

    Returns:
        float: Bounded scalar disagreement score in [0.0, 1.0].
    """
    score = mean_pairwise_contradiction + 0.5 * mean_pairwise_neutral
    return float(np.clip(score, 0.0, 1.0))


# ==============================================================================
# 3. Pairwise & Example-Level Signal Extraction
# ==============================================================================

def extract_pairwise_nli_predictions(
    responses: List[str],
    model: Optional[CrossEncoder] = None,
    label_indices: Optional[Dict[str, int]] = None,
    mock_directional_probs: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Evaluate all unordered pairs of responses bidirectionally.

    For K responses, produces C(K, 2) pair records. Each pair evaluates both:
      i -> j (Premise: r_i, Hypothesis: r_j)
      j -> i (Premise: r_j, Hypothesis: r_i)

    Parameters:
        responses: List of raw candidate response strings.
        model: Loaded CrossEncoder instance.
        label_indices: Dict mapping class names to logit column indices.
        mock_directional_probs: Optional pre-computed directional probabilities for mocking.
        batch_size: CrossEncoder batch size.

    Returns:
        List of dictionaries detailing pair-level probabilities and predictions.
    """
    k = len(responses)
    if k < 2:
        return []

    # Construct all distinct unordered index pairs (i, j) with i < j
    unordered_pairs = list(itertools.combinations(range(k), 2))

    directional_results: Dict[Tuple[int, int], Dict[str, float]] = {}

    if mock_directional_probs is not None:
        directional_results = mock_directional_probs
    else:
        if model is None or label_indices is None:
            raise ValueError("CrossEncoder model and label_indices required when not using mock probabilities.")

        # Build list of directional text pairs: (i -> j) and (j -> i)
        pair_requests: List[Tuple[int, int, str, str]] = []
        for i, j in unordered_pairs:
            pair_requests.append((i, j, responses[i], responses[j]))
            pair_requests.append((j, i, responses[j], responses[i]))

        eval_tuples = [(req[2], req[3]) for req in pair_requests]

        # Batched inference
        raw_logits = model.predict(eval_tuples, batch_size=batch_size, convert_to_numpy=True)
        all_probs = compute_softmax_probabilities(raw_logits)

        c_idx = label_indices["contradiction"]
        e_idx = label_indices["entailment"]
        n_idx = label_indices["neutral"]

        for idx_req, (prem_i, hyp_j, _, _) in enumerate(pair_requests):
            row_p = all_probs[idx_req]
            directional_results[(prem_i, hyp_j)] = {
                "contradiction": float(row_p[c_idx]),
                "entailment": float(row_p[e_idx]),
                "neutral": float(row_p[n_idx]),
            }

    pair_records: List[Dict[str, Any]] = []
    for i, j in unordered_pairs:
        p_i_to_j = directional_results[(i, j)]
        p_j_to_i = directional_results[(j, i)]

        agg = aggregate_directional_probabilities(p_i_to_j, p_j_to_i)

        pair_records.append({
            "pair_index_i": i,
            "pair_index_j": j,
            "response_i": responses[i],
            "response_j": responses[j],
            "prob_entail_i_to_j": p_i_to_j["entailment"],
            "prob_contra_i_to_j": p_i_to_j["contradiction"],
            "prob_neutral_i_to_j": p_i_to_j["neutral"],
            "prob_entail_j_to_i": p_j_to_i["entailment"],
            "prob_contra_j_to_i": p_j_to_i["contradiction"],
            "prob_neutral_j_to_i": p_j_to_i["neutral"],
            "pair_entailment": agg["pair_entailment"],
            "pair_contradiction": agg["pair_contradiction"],
            "pair_neutral": agg["pair_neutral"],
            "predicted_class": agg["predicted_class"],
        })

    return pair_records


def aggregate_example_nli_signals(
    pair_records: List[Dict[str, Any]],
    num_generations: int = 5,
) -> Dict[str, Union[int, float]]:
    """
    Aggregate pairwise NLI evaluations into example-level scalar features.

    Parameters:
        pair_records: List of pairwise evaluation dictionaries.
        num_generations: Number of candidate responses evaluated.

    Returns:
        Dict of aggregated example-level NLI metrics.
    """
    m = len(pair_records)
    if m == 0:
        # Edge case: K < 2
        return {
            "num_generations": num_generations,
            "mean_pairwise_entailment": 1.0 if num_generations == 1 else 0.0,
            "mean_pairwise_contradiction": 0.0,
            "mean_pairwise_neutral": 0.0,
            "fraction_entailing_pairs": 1.0 if num_generations == 1 else 0.0,
            "fraction_contradicting_pairs": 0.0,
            "fraction_neutral_pairs": 0.0,
            "nli_disagreement": 0.0,
        }

    entailments = [r["pair_entailment"] for r in pair_records]
    contradictions = [r["pair_contradiction"] for r in pair_records]
    neutrals = [r["pair_neutral"] for r in pair_records]
    pred_classes = [r["predicted_class"] for r in pair_records]

    mean_entail = float(np.mean(entailments))
    mean_contra = float(np.mean(contradictions))
    mean_neutral = float(np.mean(neutrals))

    frac_entail = float(sum(1 for c in pred_classes if c == "entailment") / m)
    frac_contra = float(sum(1 for c in pred_classes if c == "contradiction") / m)
    frac_neutral = float(sum(1 for c in pred_classes if c == "neutral") / m)

    disagreement = calculate_nli_disagreement(mean_contra, mean_neutral)

    return {
        "num_generations": num_generations,
        "mean_pairwise_entailment": mean_entail,
        "mean_pairwise_contradiction": mean_contra,
        "mean_pairwise_neutral": mean_neutral,
        "fraction_entailing_pairs": frac_entail,
        "fraction_contradicting_pairs": frac_contra,
        "fraction_neutral_pairs": frac_neutral,
        "nli_disagreement": disagreement,
    }


# ==============================================================================
# 4. Pilot Runner & Artifact Persistence
# ==============================================================================

def run_nli_agreement_pilot(
    input_generations_path: str = "experiments/baselines/self_consistency/pilot_generations_raw.parquet",
    output_dir: str = "experiments/baselines/nli_agreement",
    model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
    device: str = "cuda:0",
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Execute NLI agreement evaluation on the 20-example self-consistency pilot outputs.

    Parameters:
        input_generations_path: Parquet file path of self-consistency raw generations.
        output_dir: Target directory for NLI artifacts.
        model_name: HuggingFace model repository identifier.
        device: CUDA or CPU device string.
        batch_size: Batch size for model inference.

    Returns:
        Tuple of (df_pairwise, df_aggregated, summary_metrics).
    """
    if not os.path.exists(input_generations_path):
        raise FileNotFoundError(f"Input generations dataset not found at '{input_generations_path}'")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw generations from '{input_generations_path}'...")
    df_raw = pd.read_parquet(input_generations_path)
    logger.info(f"Loaded {len(df_raw)} generation rows across {df_raw['id'].nunique()} unique examples.")

    # Load NLI model
    t_load_start = time.perf_counter()
    model, label_indices = load_nli_model(model_name=model_name, device=device)
    model_loading_time = time.perf_counter() - t_load_start

    # Track GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    pairwise_records: List[Dict[str, Any]] = []
    aggregated_records: List[Dict[str, Any]] = []

    unique_ids = df_raw["id"].unique().tolist()
    total_examples = len(unique_ids)
    failed_evaluations = 0

    logger.info(f"Evaluating {total_examples} examples with C(5, 2) = 10 pairs each (total 200 pairs, 400 directional evaluations)...")
    t_infer_start = time.perf_counter()

    for idx, ex_id in enumerate(unique_ids):
        sub_df = df_raw[df_raw["id"] == ex_id].sort_values("generation_index").reset_index(drop=True)
        source_dataset = str(sub_df["source_dataset"].iloc[0])
        original_label = int(sub_df["original_label"].iloc[0])
        responses = sub_df["generated_response"].tolist()

        try:
            pair_recs = extract_pairwise_nli_predictions(
                responses=responses,
                model=model,
                label_indices=label_indices,
                batch_size=batch_size,
            )

            # Store pair records
            for p in pair_recs:
                rec = {
                    "id": ex_id,
                    "source_dataset": source_dataset,
                    "original_label": original_label,
                    **p,
                }
                pairwise_records.append(rec)

            # Extract example-level aggregated signals
            signals = aggregate_example_nli_signals(
                pair_records=pair_recs,
                num_generations=len(responses),
            )

            aggregated_records.append({
                "id": ex_id,
                "source_dataset": source_dataset,
                "original_label": original_label,
                "num_generations": signals["num_generations"],
                "mean_pairwise_entailment": signals["mean_pairwise_entailment"],
                "mean_pairwise_contradiction": signals["mean_pairwise_contradiction"],
                "mean_pairwise_neutral": signals["mean_pairwise_neutral"],
                "fraction_entailing_pairs": signals["fraction_entailing_pairs"],
                "fraction_contradicting_pairs": signals["fraction_contradicting_pairs"],
                "fraction_neutral_pairs": signals["fraction_neutral_pairs"],
                "nli_disagreement": signals["nli_disagreement"],
            })

            logger.info(
                f"[{idx + 1:02d}/{total_examples}] {ex_id} ({source_dataset}) | "
                f"Entail: {signals['mean_pairwise_entailment']:.3f} | "
                f"Contra: {signals['mean_pairwise_contradiction']:.3f} | "
                f"Neut: {signals['mean_pairwise_neutral']:.3f} | "
                f"Disagreement: {signals['nli_disagreement']:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed evaluation for example {ex_id}: {e}", exc_info=True)
            failed_evaluations += 1

    total_infer_time = time.perf_counter() - t_infer_start
    total_pairs = len(pairwise_records)
    total_directional = total_pairs * 2

    df_pairwise = pd.DataFrame(pairwise_records)
    df_aggregated = pd.DataFrame(aggregated_records)

    # Save artifacts
    pairwise_parquet_path = out_path / "pilot_nli_pairwise.parquet"
    pairwise_csv_path = out_path / "pilot_nli_pairwise.csv"
    agg_parquet_path = out_path / "pilot_nli_aggregated.parquet"
    agg_csv_path = out_path / "pilot_nli_aggregated.csv"
    summary_json_path = out_path / "pilot_nli_summary.json"

    df_pairwise.to_parquet(pairwise_parquet_path, index=False)
    df_pairwise.to_csv(pairwise_csv_path, index=False)
    df_aggregated.to_parquet(agg_parquet_path, index=False)
    df_aggregated.to_csv(agg_csv_path, index=False)

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Per-dataset breakdown
    per_dataset_summary: Dict[str, Dict[str, float]] = {}
    for ds_name, grp in df_aggregated.groupby("source_dataset"):
        per_dataset_summary[str(ds_name)] = {
            "count": int(len(grp)),
            "mean_pairwise_entailment": round(float(grp["mean_pairwise_entailment"].mean()), 4),
            "mean_pairwise_contradiction": round(float(grp["mean_pairwise_contradiction"].mean()), 4),
            "mean_pairwise_neutral": round(float(grp["mean_pairwise_neutral"].mean()), 4),
            "fraction_entailing_pairs": round(float(grp["fraction_entailing_pairs"].mean()), 4),
            "fraction_contradicting_pairs": round(float(grp["fraction_contradicting_pairs"].mean()), 4),
            "fraction_neutral_pairs": round(float(grp["fraction_neutral_pairs"].mean()), 4),
            "mean_nli_disagreement": round(float(grp["nli_disagreement"].mean()), 4),
        }

    summary_metrics: Dict[str, Any] = {
        "nli_model_name": model_name,
        "device": device,
        "model_loading_time_s": round(model_loading_time, 3),
        "total_inference_time_s": round(total_infer_time, 3),
        "avg_pair_inference_time_s": round(total_infer_time / max(total_pairs, 1), 4),
        "avg_directional_inference_time_s": round(total_infer_time / max(total_directional, 1), 4),
        "total_examples": total_examples,
        "total_pair_count": total_pairs,
        "total_directional_evaluations": total_directional,
        "failed_evaluations": failed_evaluations,
        "mean_entailment_probability": round(float(df_aggregated["mean_pairwise_entailment"].mean()), 4),
        "mean_contradiction_probability": round(float(df_aggregated["mean_pairwise_contradiction"].mean()), 4),
        "mean_neutral_probability": round(float(df_aggregated["mean_pairwise_neutral"].mean()), 4),
        "mean_fraction_entailing_pairs": round(float(df_aggregated["fraction_entailing_pairs"].mean()), 4),
        "mean_fraction_contradicting_pairs": round(float(df_aggregated["fraction_contradicting_pairs"].mean()), 4),
        "mean_fraction_neutral_pairs": round(float(df_aggregated["fraction_neutral_pairs"].mean()), 4),
        "mean_nli_disagreement": round(float(df_aggregated["nli_disagreement"].mean()), 4),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
        "per_dataset_summary": per_dataset_summary,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"NLI pilot completed successfully. Summary saved to '{summary_json_path}'.")
    logger.info(f"Summary metrics: {json.dumps(summary_metrics, indent=2)}")

    return df_pairwise, df_aggregated, summary_metrics


# ==============================================================================
# 5. Production Full Runner & Artifact Persistence
# ==============================================================================

def run_nli_agreement_full(
    input_generations_path: str = "experiments/baselines/self_consistency/full_generations_raw.parquet",
    output_dir: str = "experiments/baselines/nli_agreement",
    model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
    device: str = "cuda:0",
    batch_size: int = 32,
    checkpoint_interval: int = 100,
    resume: bool = True,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Execute production NLI agreement feature extraction across the full 4,000-example benchmark.

    Evaluates C(5, 2) = 10 response pairs per example bidirectionally (20 directional inferences/ex).
    Produces 40,000 unordered pairs and 80,000 directional evaluations across 4,000 examples.

    Parameters:
        input_generations_path: Parquet file containing 20,000 raw candidate responses.
        output_dir: Destination directory for production NLI artifacts.
        model_name: CrossEncoder NLI model repository ID.
        device: Target execution device string ('cuda:0' or 'cpu').
        batch_size: CrossEncoder inference batch size.
        checkpoint_interval: Number of completed examples between atomic disk checkpoints.
        resume: If True, resumes seamlessly from intermediate checkpoints.
        limit: Optional maximum number of examples for smoke-testing and dry-runs.

    Returns:
        Tuple of (df_pairwise, df_aggregated, summary_metrics).
    """
    if not os.path.exists(input_generations_path):
        raise FileNotFoundError(f"Input generations dataset not found at '{input_generations_path}'")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_parquet_path = out_path / "full_nli_aggregated.parquet"
    agg_csv_path = out_path / "full_nli_aggregated.csv"
    pairwise_parquet_path = out_path / "full_nli_pairwise.parquet"
    summary_json_path = out_path / "full_nli_summary.json"

    ckpt_agg_path = out_path / ".checkpoint_full_nli_aggregated.parquet"
    ckpt_pair_path = out_path / ".checkpoint_full_nli_pairwise.parquet"

    logger.info(f"Loading raw generations dataset from '{input_generations_path}'...")
    df_raw = pd.read_parquet(input_generations_path)

    # Pre-validation: Verify exactly 5 generations per completed example
    logger.info("Validating input dataset: checking exactly 5 candidate responses per example...")
    gen_counts = df_raw.groupby("id").size()
    if not (gen_counts == 5).all():
        bad_ids = gen_counts[gen_counts != 5]
        raise ValueError(
            f"Integrity check failed: {len(bad_ids)} examples do not have exactly 5 generations! "
            f"Min={gen_counts.min()}, Max={gen_counts.max()}"
        )
    logger.info(f"Input dataset validated: {len(gen_counts)} examples with exactly 5 generations each.")

    # Unique example IDs in dataset order
    unique_ids = df_raw["id"].unique().tolist()
    if limit is not None and limit > 0:
        logger.info(f"Applying limit of {limit} examples for dry-run/smoke-test.")
        unique_ids = unique_ids[:limit]

    target_count = len(unique_ids)
    target_ids = set(unique_ids)

    pairwise_records: List[Dict[str, Any]] = []
    aggregated_records: List[Dict[str, Any]] = []
    completed_ids = set()

    # Check for completed outputs first
    if resume and agg_parquet_path.exists() and pairwise_parquet_path.exists():
        try:
            df_existing_agg = pd.read_parquet(agg_parquet_path)
            df_existing_pair = pd.read_parquet(pairwise_parquet_path)
            existing_ids = set(df_existing_agg["id"].astype(str))
            if target_ids.issubset(existing_ids):
                logger.info(f"All {target_count} target examples already completed in '{agg_parquet_path}'. Loading existing results.")
                df_target_agg = df_existing_agg[df_existing_agg["id"].isin(target_ids)].copy().reset_index(drop=True)
                df_target_pair = df_existing_pair[df_existing_pair["id"].isin(target_ids)].copy().reset_index(drop=True)
                summary = {}
                if summary_json_path.exists():
                    with open(summary_json_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                return df_target_pair, df_target_agg, summary
        except Exception as e:
            logger.warning(f"Could not read existing completed files: {e}. Falling back to checkpoint inspection.")

    # Check for intermediate checkpoints
    if resume and ckpt_agg_path.exists() and ckpt_pair_path.exists():
        try:
            df_ckpt_agg = pd.read_parquet(ckpt_agg_path)
            df_ckpt_pair = pd.read_parquet(ckpt_pair_path)
            completed_ids = set(df_ckpt_agg["id"].astype(str))
            aggregated_records = df_ckpt_agg.to_dict(orient="records")
            pairwise_records = df_ckpt_pair.to_dict(orient="records")
            logger.info(f"Resuming from checkpoint: {len(completed_ids)}/{target_count} examples already completed.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint files: {e}. Starting fresh.")
            completed_ids = set()
            pairwise_records = []
            aggregated_records = []

    remaining_ids = [ex_id for ex_id in unique_ids if ex_id not in completed_ids]
    logger.info(f"Remaining examples to evaluate: {len(remaining_ids)} of {target_count}.")

    # Group df_raw by ID for fast retrieval
    df_raw_target = df_raw[df_raw["id"].isin(target_ids)].copy()
    raw_by_id = {k: v for k, v in df_raw_target.groupby("id")}

    failed_evaluations = 0
    t_infer_start = time.perf_counter()
    model_loading_time = 0.0

    if remaining_ids:
        dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        logger.info(f"Loading CrossEncoder '{model_name}' on device '{dev}'...")
        t_load_start = time.perf_counter()
        model, label_indices = load_nli_model(model_name=model_name, device=dev)
        model_loading_time = time.perf_counter() - t_load_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for step_idx, ex_id in enumerate(remaining_ids):
            sub_df = raw_by_id[ex_id].sort_values("generation_index").reset_index(drop=True)
            source_dataset = str(sub_df["source_dataset"].iloc[0])
            original_label = int(sub_df["original_label"].iloc[0])
            responses = sub_df["generated_response"].tolist()

            try:
                pair_recs = extract_pairwise_nli_predictions(
                    responses=responses,
                    model=model,
                    label_indices=label_indices,
                    batch_size=batch_size,
                )

                for p in pair_recs:
                    pairwise_records.append({
                        "id": ex_id,
                        "source_dataset": source_dataset,
                        "original_label": original_label,
                        **p,
                    })

                signals = aggregate_example_nli_signals(
                    pair_records=pair_recs,
                    num_generations=len(responses),
                )

                aggregated_records.append({
                    "id": ex_id,
                    "source_dataset": source_dataset,
                    "original_label": original_label,
                    "num_generations": signals["num_generations"],
                    "mean_pairwise_entailment": signals["mean_pairwise_entailment"],
                    "mean_pairwise_contradiction": signals["mean_pairwise_contradiction"],
                    "mean_pairwise_neutral": signals["mean_pairwise_neutral"],
                    "fraction_entailing_pairs": signals["fraction_entailing_pairs"],
                    "fraction_contradicting_pairs": signals["fraction_contradicting_pairs"],
                    "fraction_neutral_pairs": signals["fraction_neutral_pairs"],
                    "nli_disagreement": signals["nli_disagreement"],
                })

                completed_ids.add(ex_id)

                if (step_idx + 1) % 50 == 0 or (step_idx + 1) == len(remaining_ids):
                    elapsed_so_far = time.perf_counter() - t_infer_start
                    avg_pace = elapsed_so_far / (step_idx + 1)
                    rem_time_s = avg_pace * (len(remaining_ids) - (step_idx + 1))
                    rem_time_min = rem_time_s / 60.0
                    logger.info(
                        f"[{len(completed_ids)}/{target_count}] ({step_idx + 1}/{len(remaining_ids)}) "
                        f"{ex_id} ({source_dataset}) | "
                        f"Entail: {signals['mean_pairwise_entailment']:.3f} | "
                        f"Contra: {signals['mean_pairwise_contradiction']:.3f} | "
                        f"Neut: {signals['mean_pairwise_neutral']:.3f} | "
                        f"Dis: {signals['nli_disagreement']:.3f} | "
                        f"Pace: {avg_pace:.3f}s/ex | ETA: {rem_time_min:.1f}m"
                    )

                # Periodic Checkpoint
                if (step_idx + 1) % checkpoint_interval == 0:
                    logger.info(f"Saving NLI checkpoint at {len(completed_ids)} completed examples...")
                    df_ckpt_p = pd.DataFrame(pairwise_records)
                    df_ckpt_a = pd.DataFrame(aggregated_records)
                    df_ckpt_p.to_parquet(ckpt_pair_path, index=False)
                    df_ckpt_a.to_parquet(ckpt_agg_path, index=False)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed evaluation for example {ex_id}: {e}", exc_info=True)
                failed_evaluations += 1

    total_infer_time = time.perf_counter() - t_infer_start

    # --------------------------------------------------------------------------
    # Assemble & Validate Final DataFrames
    # --------------------------------------------------------------------------
    df_pairwise = pd.DataFrame(pairwise_records)
    df_aggregated = pd.DataFrame(aggregated_records)

    logger.info("Executing comprehensive validation assertions on NLI features...")

    # 1. Row count validation
    assert len(df_aggregated) == target_count, (
        f"Mismatch in aggregated examples: expected {target_count}, got {len(df_aggregated)}"
    )
    expected_pairs = target_count * 10
    assert len(df_pairwise) == expected_pairs, (
        f"Mismatch in pairwise evaluations: expected {expected_pairs}, got {len(df_pairwise)}"
    )

    # 2. Duplicate ID validation
    assert df_aggregated["id"].nunique() == target_count, (
        f"Duplicate IDs detected in aggregated table! Unique: {df_aggregated['id'].nunique()}, Total: {len(df_aggregated)}"
    )

    # 3. Numeric completeness & finiteness
    numeric_cols = [
        "mean_pairwise_entailment",
        "mean_pairwise_contradiction",
        "mean_pairwise_neutral",
        "fraction_entailing_pairs",
        "fraction_contradicting_pairs",
        "fraction_neutral_pairs",
        "nli_disagreement",
    ]
    for col in numeric_cols:
        assert not df_aggregated[col].isna().any(), f"NaN values detected in column '{col}'"
        assert np.all(np.isfinite(df_aggregated[col].values)), f"Non-finite values detected in column '{col}'"

    # 4. Valid range checks
    for col in numeric_cols:
        assert (df_aggregated[col] >= 0.0 - 1e-6).all(), f"Values below 0.0 in '{col}': min={df_aggregated[col].min()}"
        assert (df_aggregated[col] <= 1.0 + 1e-6).all(), f"Values above 1.0 in '{col}': max={df_aggregated[col].max()}"

    # 5. Probabilities sum to 1.0 identity
    prob_sums = (
        df_aggregated["mean_pairwise_entailment"]
        + df_aggregated["mean_pairwise_contradiction"]
        + df_aggregated["mean_pairwise_neutral"]
    )
    sum_diff = np.abs(prob_sums - 1.0)
    assert (sum_diff < 1e-5).all(), f"Entail + Contra + Neutral does not equal 1.0! Max diff={sum_diff.max()}"

    # 6. Disagreement formula identity: disagreement = contra + 0.5 * neutral
    expected_disagreement = (
        df_aggregated["mean_pairwise_contradiction"]
        + 0.5 * df_aggregated["mean_pairwise_neutral"]
    )
    dis_diff = np.abs(df_aggregated["nli_disagreement"] - expected_disagreement)
    assert (dis_diff < 1e-5).all(), f"nli_disagreement formula mismatch! Max diff={dis_diff.max()}"

    logger.info("All validation assertions passed successfully.")

    # --------------------------------------------------------------------------
    # Save Final Artifacts
    # --------------------------------------------------------------------------
    logger.info(f"Saving final aggregated NLI artifact to '{agg_parquet_path}'...")
    df_aggregated.to_parquet(agg_parquet_path, index=False)
    df_aggregated.to_csv(agg_csv_path, index=False)

    logger.info(f"Saving final pairwise NLI artifact to '{pairwise_parquet_path}'...")
    df_pairwise.to_parquet(pairwise_parquet_path, index=False)

    # Clean up checkpoints
    if ckpt_pair_path.exists():
        ckpt_pair_path.unlink()
        logger.info(f"Cleaned up transient checkpoint '{ckpt_pair_path.name}'.")
    if ckpt_agg_path.exists():
        ckpt_agg_path.unlink()
        logger.info(f"Cleaned up transient checkpoint '{ckpt_agg_path.name}'.")

    # Metrics Summary
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    total_pairs = len(df_pairwise)
    total_directional = total_pairs * 2

    per_dataset_summary: Dict[str, Dict[str, float]] = {}
    for ds_name, grp in df_aggregated.groupby("source_dataset"):
        per_dataset_summary[str(ds_name)] = {
            "count": int(len(grp)),
            "mean_pairwise_entailment": round(float(grp["mean_pairwise_entailment"].mean()), 4),
            "mean_pairwise_contradiction": round(float(grp["mean_pairwise_contradiction"].mean()), 4),
            "mean_pairwise_neutral": round(float(grp["mean_pairwise_neutral"].mean()), 4),
            "fraction_entailing_pairs": round(float(grp["fraction_entailing_pairs"].mean()), 4),
            "fraction_contradicting_pairs": round(float(grp["fraction_contradicting_pairs"].mean()), 4),
            "fraction_neutral_pairs": round(float(grp["fraction_neutral_pairs"].mean()), 4),
            "mean_nli_disagreement": round(float(grp["nli_disagreement"].mean()), 4),
        }

    summary_metrics: Dict[str, Any] = {
        "nli_model_name": model_name,
        "device": device,
        "input_generations_path": input_generations_path,
        "total_examples": target_count,
        "total_pair_count": total_pairs,
        "total_directional_evaluations": total_directional,
        "failed_evaluations": failed_evaluations,
        "model_loading_time_s": round(model_loading_time, 3),
        "total_inference_time_s": round(total_infer_time, 3),
        "avg_example_inference_time_s": round(total_infer_time / max(target_count, 1), 4),
        "avg_pair_inference_time_s": round(total_infer_time / max(total_pairs, 1), 5),
        "avg_directional_inference_time_s": round(total_infer_time / max(total_directional, 1), 5),
        "mean_entailment_probability": round(float(df_aggregated["mean_pairwise_entailment"].mean()), 4),
        "mean_contradiction_probability": round(float(df_aggregated["mean_pairwise_contradiction"].mean()), 4),
        "mean_neutral_probability": round(float(df_aggregated["mean_pairwise_neutral"].mean()), 4),
        "mean_fraction_entailing_pairs": round(float(df_aggregated["fraction_entailing_pairs"].mean()), 4),
        "mean_fraction_contradicting_pairs": round(float(df_aggregated["fraction_contradicting_pairs"].mean()), 4),
        "mean_fraction_neutral_pairs": round(float(df_aggregated["fraction_neutral_pairs"].mean()), 4),
        "mean_nli_disagreement": round(float(df_aggregated["nli_disagreement"].mean()), 4),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
        "per_dataset_summary": per_dataset_summary,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Production NLI summary saved to '{summary_json_path}'.")

    return df_pairwise, df_aggregated, summary_metrics


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pairwise NLI agreement signal extraction.")
    parser.add_argument("--mode", type=str, choices=["pilot", "full"], default="full", help="Execution mode ('pilot' or 'full').")
    parser.add_argument("--data", type=str, default=None, help="Path to input raw generations parquet.")
    parser.add_argument("--output-dir", type=str, default="experiments/baselines/nli_agreement", help="Output directory.")
    parser.add_argument("--model-name", type=str, default="cross-encoder/nli-MiniLM2-L6-H768", help="NLI CrossEncoder model ID.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint interval for full mode.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry-runs / smoke tests.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoints (default: True).")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing checkpoints.")

    args = parser.parse_args()

    if args.mode == "pilot":
        data_path = args.data or "experiments/baselines/self_consistency/pilot_generations_raw.parquet"
        run_nli_agreement_pilot(
            input_generations_path=data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        data_path = args.data or "experiments/baselines/self_consistency/full_generations_raw.parquet"
        run_nli_agreement_full(
            input_generations_path=data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            resume=not args.no_resume,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
