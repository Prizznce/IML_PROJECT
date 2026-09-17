"""
Self-Consistency Signal Extraction and Evaluation Pipeline.

Implements stochastic sampling, deterministic text normalization, discrete agreement
metrics, and pairwise semantic similarity to quantify self-consistency across
independent LLM generations from Qwen/Qwen3.5-0.8B.

Designed for both:
1. Supervised experimental evaluation (scoring stochastic generations for benchmark prompts
   while strictly preserving the benchmark ground-truth label without label leakage).
2. Real-time / demo inference workflows (generating multiple responses for arbitrary user prompts
   and computing consensus features for downstream hallucination classifiers).
"""

import argparse
import itertools
import json
import logging
import math
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.generation.generate_signals import format_model_input

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("self_consistency")


# ==============================================================================
# 1. Deterministic Normalization
# ==============================================================================

def normalize_response(text: Optional[str]) -> str:
    """
    Deterministic response normalization function for exact-match comparison.

    Transforms text by:
    1. Coercing input to string (handling None/NaN gracefully).
    2. Stripping leading and trailing whitespace.
    3. Lowercasing all characters.
    4. Removing quotes and apostrophes.
    5. Replacing remaining punctuation marks with whitespace.
    6. Collapsing repeated whitespace characters into a single space.
    7. Final strip.

    Parameters:
        text (Optional[str]): Raw response text.

    Returns:
        str: Cleanly normalized string.
    """
    if text is None or pd.isna(text):
        return ""
    s = str(text).strip().lower()
    if not s:
        return ""

    # Remove quotes, backticks, and apostrophes
    s = re.sub(r"[\"'`]", "", s)

    # Replace remaining punctuation characters with space
    punct_chars = string.punctuation.replace("'", "").replace('"', "").replace("`", "")
    s = re.sub(r"[" + re.escape(punct_chars) + r"]", " ", s)

    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ==============================================================================
# 2. Discrete Agreement Metrics
# ==============================================================================

def calculate_exact_match_agreement(responses: List[str]) -> float:
    """
    Calculate the fraction of generation pairs that exactly agree after normalization.

    Formula:
        For K generations, there are C(K, 2) = K*(K-1)/2 distinct unordered pairs.
        agreement = (number of pairs (i, j) where norm(r_i) == norm(r_j)) / C(K, 2)

    Edge cases:
        If K <= 1: returns 1.0 (a single response or empty set is trivially self-consistent).

    Parameters:
        responses (List[str]): List of generated response strings.

    Returns:
        float: Exact match agreement rate in [0.0, 1.0].
    """
    k = len(responses)
    if k <= 1:
        return 1.0

    normalized = [normalize_response(r) for r in responses]
    num_pairs = (k * (k - 1)) // 2
    matching_pairs = 0

    for i in range(k):
        for j in range(i + 1, k):
            if normalized[i] == normalized[j]:
                matching_pairs += 1

    return float(matching_pairs / num_pairs)


def calculate_unique_response_ratio(responses: List[str]) -> float:
    """
    Calculate the ratio of unique normalized responses to total generations.

    Formula:
        unique_ratio = |unique(norm(responses))| / K

    Properties:
        - In [1/K, 1.0] for K >= 1.
        - If all responses agree: 1/K.
        - If all responses differ: 1.0.

    Parameters:
        responses (List[str]): List of generated response strings.

    Returns:
        float: Unique response ratio in [0.0, 1.0].
    """
    k = len(responses)
    if k == 0:
        return 0.0

    normalized = [normalize_response(r) for r in responses]
    unique_count = len(set(normalized))
    return float(unique_count / k)


def calculate_majority_response_fraction(responses: List[str]) -> float:
    """
    Calculate the frequency fraction of the most common normalized response.

    Formula:
        majority_fraction = max_frequency(norm(responses)) / K

    Properties:
        - In [1/K, 1.0] for K >= 1.
        - If all responses agree: 1.0.
        - If all responses are distinct: 1/K.

    Parameters:
        responses (List[str]): List of generated response strings.

    Returns:
        float: Majority response fraction in [0.0, 1.0].
    """
    k = len(responses)
    if k == 0:
        return 0.0

    normalized = [normalize_response(r) for r in responses]
    counts: Dict[str, int] = {}
    for r in normalized:
        counts[r] = counts.get(r, 0) + 1

    max_count = max(counts.values())
    return float(max_count / k)


# ==============================================================================
# 3. Pairwise Semantic Similarity & Disagreement Score
# ==============================================================================

def calculate_pairwise_semantic_similarity(
    responses: List[str],
    embedding_model: Optional[Any] = None,
    embeddings: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate pairwise cosine similarity statistics across generated responses.

    Parameters:
        responses (List[str]): List of generated responses.
        embedding_model (Any, optional): SentenceTransformer model instance.
        embeddings (np.ndarray, optional): Pre-computed embeddings of shape (K, D).
            If provided, embedding_model is not required.
        device (str, optional): Computation device ('cuda' or 'cpu').

    Returns:
        Dict[str, float]: Dictionary containing:
            - mean_pairwise_similarity
            - min_pairwise_similarity
            - max_pairwise_similarity
            - pairwise_similarity_std
    """
    k = len(responses)
    if k <= 1:
        return {
            "mean_pairwise_similarity": 1.0,
            "min_pairwise_similarity": 1.0,
            "max_pairwise_similarity": 1.0,
            "pairwise_similarity_std": 0.0,
        }

    if embeddings is None:
        if embedding_model is None:
            raise ValueError("Either embedding_model or precomputed embeddings must be provided.")
        embeddings = embedding_model.encode(
            responses,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )

    # Ensure embeddings are 2D numpy array with unit L2 norm
    embs = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embs = embs / norms

    # Compute pairwise cosine similarity matrix
    sim_matrix = np.dot(norm_embs, norm_embs.T)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    # Extract upper triangular pairs (i < j)
    triu_indices = np.triu_indices(k, k=1)
    pairwise_sims = sim_matrix[triu_indices]

    if len(pairwise_sims) == 0:
        return {
            "mean_pairwise_similarity": 1.0,
            "min_pairwise_similarity": 1.0,
            "max_pairwise_similarity": 1.0,
            "pairwise_similarity_std": 0.0,
        }

    mean_sim = float(np.mean(pairwise_sims))
    min_sim = float(np.min(pairwise_sims))
    max_sim = float(np.max(pairwise_sims))
    std_sim = float(np.std(pairwise_sims))

    return {
        "mean_pairwise_similarity": mean_sim,
        "min_pairwise_similarity": min_sim,
        "max_pairwise_similarity": max_sim,
        "pairwise_similarity_std": std_sim,
    }


def calculate_generation_disagreement(mean_pairwise_similarity: float) -> float:
    """
    Calculate scalar generation disagreement score from mean pairwise similarity.

    Formula:
        generation_disagreement = 1.0 - mean_pairwise_similarity

    Interpretation:
        - Measures semantic divergence across stochastic generations.
        - When all generations share identical semantic representations (mean similarity = 1.0),
          disagreement is exactly 0.0.
        - As responses diverge semantically, mean similarity drops and disagreement rises.
        - Valid finite output bounded in [0.0, 2.0] for cosine similarity in [-1.0, 1.0].

    Parameters:
        mean_pairwise_similarity (float): Average pairwise cosine similarity.

    Returns:
        float: Scalar disagreement score.
    """
    return float(1.0 - mean_pairwise_similarity)


def extract_self_consistency_signals(
    responses: List[str],
    embedding_model: Optional[Any] = None,
    embeddings: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Dict[str, Union[int, float]]:
    """
    Extract complete suite of self-consistency signals for a set of generations.

    Parameters:
        responses (List[str]): List of generated candidate responses.
        embedding_model (Any, optional): SentenceTransformer model.
        embeddings (np.ndarray, optional): Precomputed response embeddings.
        device (str, optional): Compute device.

    Returns:
        Dict[str, Union[int, float]]: All 7 self-consistency metrics.
    """
    k = len(responses)
    exact_match = calculate_exact_match_agreement(responses)
    unique_ratio = calculate_unique_response_ratio(responses)
    majority_frac = calculate_majority_response_fraction(responses)

    sim_stats = calculate_pairwise_semantic_similarity(
        responses=responses,
        embedding_model=embedding_model,
        embeddings=embeddings,
        device=device,
    )

    disagreement = calculate_generation_disagreement(sim_stats["mean_pairwise_similarity"])

    return {
        "num_generations": k,
        "exact_match_agreement": exact_match,
        "unique_response_ratio": unique_ratio,
        "majority_response_fraction": majority_frac,
        "mean_pairwise_similarity": sim_stats["mean_pairwise_similarity"],
        "min_pairwise_similarity": sim_stats["min_pairwise_similarity"],
        "max_pairwise_similarity": sim_stats["max_pairwise_similarity"],
        "pairwise_similarity_std": sim_stats["pairwise_similarity_std"],
        "generation_disagreement": disagreement,
    }


# ==============================================================================
# 4. Model Loading & Generation Routine
# ==============================================================================

def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda:0",
) -> Any:
    """
    Load lightweight sentence transformer model for pairwise semantic similarity.

    Parameters:
        model_name: Repository ID of sentence embedding model.
        device: CUDA or CPU device string.

    Returns:
        SentenceTransformer model instance.
    """
    from sentence_transformers import SentenceTransformer
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    logger.info(f"Loading SentenceTransformer '{model_name}' on device '{dev}'...")
    model = SentenceTransformer(model_name, device=dev)
    logger.info(f"SentenceTransformer '{model_name}' successfully loaded.")
    return model


def generate_stochastic_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    context: str = "",
    source_dataset: str = "",
    original_response: str = "",
    num_generations: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    seed: Optional[int] = None,
    system_instruction: str = "Answer the question directly, factually, and concisely. Do not provide preamble or internal thinking.",
) -> Tuple[List[str], float, bool]:
    """
    Generate multiple stochastic responses for a given prompt using Qwen3.5-0.8B.

    Parameters:
        model: Loaded CausalLM instance.
        tokenizer: Loaded AutoTokenizer instance.
        prompt: User question or instruction.
        context: Reference passage or background context.
        source_dataset: Provenance identifier ('halueval', 'fever', 'truthfulqa').
        original_response: Benchmark reference claim or ground truth.
        num_generations: Number of stochastic generations K (default: 5).
        temperature: Sampling temperature (default: 0.7).
        top_p: Nucleus sampling threshold (default: 0.9).
        max_new_tokens: Maximum tokens to generate per response (default: 128).
        device: CUDA device identifier.
        seed: Optional RNG seed for deterministic generation.
        system_instruction: Direct factual answer directive.

    Returns:
        Tuple of:
            - List of raw generated response strings (len = num_generations).
            - Elapsed generation time in seconds.
            - Boolean flag indicating whether any response contained reasoning <think> tags.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Format model input with chat template and enable_thinking=False
    model_input = format_model_input(
        prompt=prompt,
        context=context,
        source_dataset=source_dataset,
        original_response=original_response,
        use_chat_template=True,
        tokenizer=tokenizer,
        system_instruction=system_instruction,
    )

    enc = tokenizer(model_input, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    input_len = input_ids.shape[1]

    t_start = time.perf_counter()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    t_elapsed = time.perf_counter() - t_start

    raw_responses: List[str] = []
    has_think_tags = False

    for i in range(num_generations):
        gen_tokens = gen_out[i][input_len:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        raw_responses.append(decoded)
        if "<think>" in decoded or "</think>" in decoded:
            has_think_tags = True

    return raw_responses, t_elapsed, has_think_tags


# ==============================================================================
# 5. Pilot Runner & Artifact Persistence
# ==============================================================================

def run_self_consistency_pilot(
    data_path: str = "data/processed/combined_processed.parquet",
    sample_size: int = 20,
    seed: int = 42,
    num_generations: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda:0",
    output_dir: str = "experiments/baselines/self_consistency",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Execute the 20-example pilot experiment for self-consistency signal extraction.

    Parameters:
        data_path: Path to canonical combined dataset.
        sample_size: Number of examples to evaluate (exactly 20).
        seed: Random seed for deterministic sample selection and generation.
        num_generations: Number of stochastic generations per prompt (5).
        temperature: Sampling temperature (0.7).
        top_p: Nucleus sampling threshold (0.9).
        max_new_tokens: Maximum tokens per generation (128).
        model_id: Causal LLM model identifier.
        embed_model_id: SentenceTransformer model identifier.
        device: Target execution device.
        output_dir: Output directory for pilot artifacts.

    Returns:
        Tuple of:
            - df_generations: Generation-level DataFrame (100 rows).
            - df_aggregated: Example-level aggregated DataFrame (20 rows).
            - summary_metrics: Dictionary of diagnostic and timing statistics.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at '{data_path}'")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading input dataset from '{data_path}'...")
    df_raw = pd.read_parquet(data_path)
    logger.info(f"Total dataset size: {len(df_raw)}. Selecting {sample_size} examples (seed={seed})...")

    pilot_sample = df_raw.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    logger.info(f"Sample breakdown by source: {pilot_sample['source_dataset'].value_counts().to_dict()} | "
                f"by label: {pilot_sample['label'].value_counts().to_dict()}")

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load generation model and tokenizer
    from src.generation.generate_signals import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)

    # Load embedding model
    embed_model = load_embedding_model(model_name=embed_model_id, device=device)

    generation_records: List[Dict[str, Any]] = []
    aggregated_records: List[Dict[str, Any]] = []

    total_generation_time = 0.0
    any_think_tags_found = False
    failed_generations = 0

    logger.info(f"Starting pilot generation: {sample_size} prompts x {num_generations} generations = {sample_size * num_generations} total responses.")
    start_total_time = time.perf_counter()

    for idx, row in pilot_sample.iterrows():
        ex_id = str(row["id"])
        source = str(row["source_dataset"])
        orig_label = int(row["label"])
        prompt = str(row["prompt"])
        context = str(row["context"]) if "context" in row and pd.notna(row["context"]) else ""
        orig_resp = str(row["response"]) if "response" in row and pd.notna(row["response"]) else ""

        # Deterministic seed progression per example
        ex_seed = seed + idx * 100

        try:
            raw_resps, gen_time, think_leak = generate_stochastic_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                context=context,
                source_dataset=source,
                original_response=orig_resp,
                num_generations=num_generations,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                device=device,
                seed=ex_seed,
            )
            total_generation_time += gen_time
            if think_leak:
                any_think_tags_found = True

            # Record each individual generation
            for gen_idx, resp_text in enumerate(raw_resps):
                generation_records.append({
                    "id": ex_id,
                    "source_dataset": source,
                    "prompt": prompt,
                    "context": context,
                    "original_response": orig_resp,
                    "original_label": orig_label,
                    "generation_index": gen_idx,
                    "generated_response": resp_text,
                    "normalized_response": normalize_response(resp_text),
                })

            # Calculate example-level self-consistency signals
            signals = extract_self_consistency_signals(
                responses=raw_resps,
                embedding_model=embed_model,
                device=device,
            )

            aggregated_records.append({
                "id": ex_id,
                "source_dataset": source,
                "original_label": orig_label,
                "num_generations": signals["num_generations"],
                "exact_match_agreement": signals["exact_match_agreement"],
                "unique_response_ratio": signals["unique_response_ratio"],
                "majority_response_fraction": signals["majority_response_fraction"],
                "mean_pairwise_similarity": signals["mean_pairwise_similarity"],
                "min_pairwise_similarity": signals["min_pairwise_similarity"],
                "max_pairwise_similarity": signals["max_pairwise_similarity"],
                "pairwise_similarity_std": signals["pairwise_similarity_std"],
                "generation_disagreement": signals["generation_disagreement"],
            })

            logger.info(
                f"[{idx + 1:02d}/{sample_size}] {ex_id} ({source}) | "
                f"EM Agreement: {signals['exact_match_agreement']:.3f} | "
                f"Unique Ratio: {signals['unique_response_ratio']:.3f} | "
                f"Mean Sim: {signals['mean_pairwise_similarity']:.3f} | "
                f"Disagreement: {signals['generation_disagreement']:.3f} ({gen_time:.2f}s)"
            )

        except Exception as e:
            logger.error(f"Failed generation for example {ex_id}: {e}", exc_info=True)
            failed_generations += 1

    overall_elapsed = time.perf_counter() - start_total_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    df_generations = pd.DataFrame(generation_records)
    df_aggregated = pd.DataFrame(aggregated_records)

    # Save structured artifacts
    gen_parquet_path = out_path / "pilot_generations_raw.parquet"
    gen_csv_path = out_path / "pilot_generations_raw.csv"
    agg_parquet_path = out_path / "pilot_self_consistency_aggregated.parquet"
    agg_csv_path = out_path / "pilot_self_consistency_aggregated.csv"
    summary_json_path = out_path / "pilot_summary_metrics.json"

    df_generations.to_parquet(gen_parquet_path, index=False)
    df_generations.to_csv(gen_csv_path, index=False)
    df_aggregated.to_parquet(agg_parquet_path, index=False)
    df_aggregated.to_csv(agg_csv_path, index=False)

    summary_metrics: Dict[str, Any] = {
        "total_examples": len(df_aggregated),
        "generations_per_example": num_generations,
        "total_generations": len(df_generations),
        "failed_generations": failed_generations,
        "total_generation_time_s": round(total_generation_time, 2),
        "overall_elapsed_time_s": round(overall_elapsed, 2),
        "avg_generation_time_per_example_s": round(total_generation_time / max(len(df_aggregated), 1), 3),
        "avg_generation_time_per_response_s": round(total_generation_time / max(len(df_generations), 1), 3),
        "mean_exact_match_agreement": round(float(df_aggregated["exact_match_agreement"].mean()), 4),
        "mean_unique_response_ratio": round(float(df_aggregated["unique_response_ratio"].mean()), 4),
        "mean_majority_response_fraction": round(float(df_aggregated["majority_response_fraction"].mean()), 4),
        "mean_pairwise_similarity": round(float(df_aggregated["mean_pairwise_similarity"].mean()), 4),
        "mean_generation_disagreement": round(float(df_aggregated["generation_disagreement"].mean()), 4),
        "any_think_tags_leaked": any_think_tags_found,
        "peak_gpu_memory_mb": round(peak_vram_mb, 2),
        "sampling_parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "model_id": model_id,
            "embed_model_id": embed_model_id,
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"Pilot completed successfully. Summary saved to '{summary_json_path}'.")
    logger.info(f"Summary metrics: {json.dumps(summary_metrics, indent=2)}")

    return df_generations, df_aggregated, summary_metrics


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self-consistency signal extraction pipeline.")
    parser.add_argument("--data", type=str, default="data/processed/combined_processed.parquet", help="Path to input parquet.")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of pilot examples.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--num-generations", type=int, default=5, help="Stochastic generations per prompt.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per generation.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-0.8B", help="Model repository ID.")
    parser.add_argument("--embed-model-id", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model ID.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device or cpu.")
    parser.add_argument("--output-dir", type=str, default="experiments/baselines/self_consistency", help="Output directory.")

    args = parser.parse_args()

    run_self_consistency_pilot(
        data_path=args.data,
        sample_size=args.sample_size,
        seed=args.seed,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        embed_model_id=args.embed_model_id,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
