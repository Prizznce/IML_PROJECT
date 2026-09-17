"""
Evidence Retrieval and Retrieval-Agreement Signal Extraction Pipeline.

Implements lightweight dense retrieval using sentence embeddings to retrieve top-k
evidence chunks from a reference corpus and computes query-evidence and response-evidence
similarity signals to evaluate evidence groundedness for hallucination detection.
"""

import argparse
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
from sentence_transformers import SentenceTransformer

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
logger = logging.getLogger("evidence_retrieval")


# ==============================================================================
# 1. Similarity & Ranking Primitives
# ==============================================================================

def compute_cosine_similarity(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarities between a query embedding and corpus embeddings.

    Parameters:
        query_emb: 1D or 2D numpy array representing query embedding(s).
        corpus_embs: 2D numpy array of shape (N, D) representing corpus embeddings.

    Returns:
        1D numpy array of cosine similarities in [-1.0, 1.0], length N.
    """
    q = np.asarray(query_emb, dtype=np.float32)
    c = np.asarray(corpus_embs, dtype=np.float32)

    if c.size == 0 or len(c.shape) < 2 or c.shape[0] == 0:
        return np.array([], dtype=np.float32)

    if q.ndim == 2:
        q = q.squeeze(0)

    # Unit-normalize query
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0 or not np.isfinite(q_norm):
        return np.zeros(c.shape[0], dtype=np.float32)
    q_unit = q / q_norm

    # Unit-normalize corpus
    c_norms = np.linalg.norm(c, axis=1, keepdims=True)
    c_norms[c_norms == 0.0] = 1.0
    c_unit = c / c_norms

    # Dot product gives cosine similarity
    sims = np.dot(c_unit, q_unit)
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


def rank_top_k(
    similarities: np.ndarray,
    k: int = 3,
) -> Tuple[List[int], List[float]]:
    """
    Retrieve top-k indices and similarity values in descending order.

    Deterministic tie-breaking: earlier indices in the corpus are preferred upon exact ties.

    Parameters:
        similarities: 1D array of similarity scores.
        k: Maximum number of items to retrieve (default: 3).

    Returns:
        Tuple of (top_k_indices, top_k_similarities).
    """
    n = len(similarities)
    if n == 0 or k <= 0:
        return [], []

    actual_k = min(k, n)

    # Sort descending with deterministic stable sort
    sorted_indices = np.argsort(-similarities, kind="stable")[:actual_k]
    top_indices = [int(idx) for idx in sorted_indices]
    top_sims = [float(similarities[idx]) for idx in top_indices]

    return top_indices, top_sims


def calculate_evidence_margin(top_similarities: List[float]) -> float:
    """
    Calculate evidence margin (relevance gap between rank-1 and rank-2 evidence).

    Formula:
        evidence_margin = top1_similarity - top2_similarity

    Edge cases:
        If fewer than 2 evidence items exist, returns 0.0.

    Parameters:
        top_similarities: List of retrieved similarities in descending order.

    Returns:
        float: Difference in similarity between rank 1 and rank 2 (>= 0.0).
    """
    if len(top_similarities) < 2:
        return 0.0
    margin = float(top_similarities[0] - top_similarities[1])
    return max(0.0, margin)


def calculate_retrieval_agreement(
    query_similarity: float,
    response_similarity: float,
) -> float:
    """
    Calculate scalar retrieval agreement between query, response, and retrieved evidence.

    Formula:
        retrieval_agreement = max(0.0, response_similarity) * max(0.0, query_similarity)

    Interpretation:
        - Evaluates dual groundedness:
          1. The retrieved chunk must be relevant to the query (query_similarity > 0).
          2. The response must be supported by that retrieved chunk (response_similarity > 0).
        - If query similarity is low or negative (no relevant evidence found in the corpus),
          any accidental overlap between response and chunk does not constitute grounded agreement.
        - If response similarity is low or negative (response contradicts or diverges from evidence),
          retrieval agreement drops to near zero.
        - Strictly bounded in [0.0, 1.0].
        - Completely finite and immune to division-by-zero singularities.

    Parameters:
        query_similarity: Cosine similarity between query and rank-1 retrieved chunk.
        response_similarity: Cosine similarity between response and rank-1 retrieved chunk.

    Returns:
        float: Bounded agreement score in [0.0, 1.0].
    """
    q_pos = max(0.0, float(query_similarity))
    r_pos = max(0.0, float(response_similarity))
    return float(np.clip(q_pos * r_pos, 0.0, 1.0))


# ==============================================================================
# 2. Corpus Construction & Query Formatting
# ==============================================================================

def format_query_text(
    prompt: str,
    original_response: str = "",
    source_dataset: str = "",
) -> str:
    """
    Format query representation for evidence retrieval based on dataset provenance.

    Parameters:
        prompt: Benchmark question or directive prompt.
        original_response: Benchmark response or claim.
        source_dataset: Originating benchmark dataset ('halueval', 'fever', 'truthfulqa').

    Returns:
        str: Factual query text to encode for retrieval.
    """
    src = str(source_dataset).lower()
    p_str = str(prompt).strip()
    r_str = str(original_response).strip()

    if src == "fever":
        # In FEVER, prompt is generic verification instruction; the claim itself is the query
        return r_str if r_str else p_str
    return p_str


def build_evidence_corpus(
    data_path: str = "data/processed/combined_processed.parquet",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Construct the textual evidence corpus from grounded passages available in project data.

    Preserves provenance and deduplicates identical text chunks.
    Documents the dataset-specific context formats:
      - HaluEval: Real passage text included in context.
      - TruthfulQA: Context contains reference URLs only (not usable textual evidence).
      - FEVER: Context contains Wikipedia pointer tuples (not raw sentence text).

    Parameters:
        data_path: Path to canonical combined processed dataset.

    Returns:
        Tuple of (list_of_passage_texts, list_of_metadata_dicts).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'")

    logger.info(f"Reading dataset from '{data_path}' to construct evidence corpus...")
    df = pd.read_parquet(data_path)

    # Extract non-empty text passages
    seen_texts = set()
    corpus_texts: List[str] = []
    corpus_meta: List[Dict[str, Any]] = []

    # Focus on HaluEval where context is authentic textual reference passage
    halu_rows = df[df["source_dataset"] == "halueval"]
    for idx, row in halu_rows.iterrows():
        ctx = str(row.get("context", "")).strip()
        if ctx and ctx not in seen_texts:
            seen_texts.add(ctx)
            chunk_id = f"halueval_chunk_{len(corpus_texts):04d}"
            corpus_texts.append(ctx)
            corpus_meta.append({
                "chunk_id": chunk_id,
                "source_example_id": str(row.get("id", "")),
                "source_dataset": "halueval",
                "char_length": len(ctx),
            })

    logger.info(f"Evidence corpus built successfully with {len(corpus_texts)} unique text passages.")
    return corpus_texts, corpus_meta


# ==============================================================================
# 3. Model Loading & Feature Extraction
# ==============================================================================

def load_retrieval_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda:0",
) -> Tuple[SentenceTransformer, float]:
    """
    Load dense retrieval embedding model.

    Parameters:
        model_name: HuggingFace model repository identifier.
        device: CUDA or CPU device string.

    Returns:
        Tuple of (SentenceTransformer instance, loading_time_s).
    """
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    logger.info(f"Loading retrieval embedding model '{model_name}' on device '{dev}'...")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=dev)
    load_time = time.perf_counter() - t0
    logger.info(f"Retrieval embedding model loaded in {load_time:.2f}s.")
    return model, load_time


def extract_retrieval_features(
    query_text: str,
    response_text: str,
    corpus_texts: List[str],
    corpus_embeddings: np.ndarray,
    corpus_meta: Optional[List[Dict[str, Any]]] = None,
    model: Optional[SentenceTransformer] = None,
    query_emb: Optional[np.ndarray] = None,
    response_emb: Optional[np.ndarray] = None,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Perform dense retrieval and extract complete suite of retrieval agreement signals.

    Parameters:
        query_text: Query text string.
        response_text: Response text to evaluate for evidence support.
        corpus_texts: List of textual passages in corpus.
        corpus_embeddings: 2D array of normalized corpus embeddings (N, D).
        corpus_meta: Optional metadata dicts for corpus chunks.
        model: Loaded SentenceTransformer model.
        query_emb: Optional pre-computed query embedding.
        response_emb: Optional pre-computed response embedding.
        k: Top-k evidence chunks to retrieve (default: 3).

    Returns:
        Dict containing top-k ranks, similarities, margins, and agreement metrics.
    """
    n_corpus = len(corpus_texts)
    if n_corpus == 0 or corpus_embeddings.size == 0:
        # Empty corpus edge case
        return {
            "retrieved_rank_1": "none",
            "retrieved_rank_2": "none",
            "retrieved_rank_3": "none",
            "retrieved_rank_1_text": "",
            "retrieved_rank_2_text": "",
            "retrieved_rank_3_text": "",
            "top1_evidence_similarity": 0.0,
            "mean_top3_evidence_similarity": 0.0,
            "response_top1_evidence_similarity": 0.0,
            "mean_response_top3_evidence_similarity": 0.0,
            "evidence_margin": 0.0,
            "retrieval_agreement": 0.0,
        }

    # Encode query and response if not provided
    if query_emb is None:
        if model is None:
            raise ValueError("Either model or query_emb must be provided.")
        query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]

    if response_emb is None:
        if model is None:
            raise ValueError("Either model or response_emb must be provided.")
        response_emb = model.encode([response_text], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Compute query similarities across corpus
    q_sims = compute_cosine_similarity(query_emb, corpus_embeddings)
    top_indices, top_q_sims = rank_top_k(q_sims, k=k)

    # Compute response similarities to retrieved evidence items
    top_corpus_embs = corpus_embeddings[top_indices]
    r_sims = compute_cosine_similarity(response_emb, top_corpus_embs)

    top1_q_sim = float(top_q_sims[0]) if top_q_sims else 0.0
    mean_top3_q_sim = float(np.mean(top_q_sims)) if top_q_sims else 0.0
    top1_r_sim = float(r_sims[0]) if len(r_sims) > 0 else 0.0
    mean_top3_r_sim = float(np.mean(r_sims)) if len(r_sims) > 0 else 0.0
    margin = calculate_evidence_margin(top_q_sims)
    agreement = calculate_retrieval_agreement(top1_q_sim, top1_r_sim)

    # Build retrieved chunk representations
    retrieved_ranks: List[str] = []
    retrieved_texts: List[str] = []

    for rank_idx, c_idx in enumerate(top_indices):
        chunk_id = corpus_meta[c_idx]["chunk_id"] if corpus_meta else f"chunk_{c_idx}"
        retrieved_ranks.append(chunk_id)
        retrieved_texts.append(corpus_texts[c_idx])

    # Pad if fewer than k
    while len(retrieved_ranks) < k:
        retrieved_ranks.append("none")
        retrieved_texts.append("")

    return {
        "retrieved_rank_1": retrieved_ranks[0],
        "retrieved_rank_2": retrieved_ranks[1],
        "retrieved_rank_3": retrieved_ranks[2],
        "retrieved_rank_1_text": retrieved_texts[0],
        "retrieved_rank_2_text": retrieved_texts[1],
        "retrieved_rank_3_text": retrieved_texts[2],
        "top1_evidence_similarity": top1_q_sim,
        "mean_top3_evidence_similarity": mean_top3_q_sim,
        "response_top1_evidence_similarity": top1_r_sim,
        "mean_response_top3_evidence_similarity": mean_top3_r_sim,
        "evidence_margin": margin,
        "retrieval_agreement": agreement,
    }


# ==============================================================================
# 4. Pilot Runner & Artifact Persistence
# ==============================================================================

def run_evidence_retrieval_pilot(
    input_data_path: str = "experiments/baselines/self_consistency/pilot_generations_raw.parquet",
    corpus_data_path: str = "data/processed/combined_processed.parquet",
    output_dir: str = "experiments/baselines/evidence_retrieval",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda:0",
    top_k: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute evidence retrieval and agreement extraction on the 20 pilot examples.

    Parameters:
        input_data_path: Path to self-consistency raw generations parquet.
        corpus_data_path: Path to canonical processed dataset for corpus extraction.
        output_dir: Output artifact directory.
        model_name: Dense retrieval model identifier.
        device: CUDA or CPU device identifier.
        top_k: Top-k evidence chunks to retrieve.

    Returns:
        Tuple of (df_results, summary_metrics).
    """
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Pilot dataset not found at '{input_data_path}'")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Build evidence corpus
    corpus_texts, corpus_meta = build_evidence_corpus(data_path=corpus_data_path)

    # 2. Load retrieval model
    model, load_time = load_retrieval_model(model_name=model_name, device=device)

    # 3. Encode evidence corpus
    logger.info(f"Encoding {len(corpus_texts)} corpus evidence passages on {device}...")
    t_enc_start = time.perf_counter()
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    corpus_enc_time = time.perf_counter() - t_enc_start
    logger.info(f"Corpus encoded in {corpus_enc_time:.2f}s. Shape: {corpus_embeddings.shape}")

    # 4. Load the 20 pilot examples
    df_raw = pd.read_parquet(input_data_path)
    pilot_examples = df_raw.drop_duplicates(subset=["id"]).sort_values("id").reset_index(drop=True)
    total_examples = len(pilot_examples)
    logger.info(f"Evaluating {total_examples} pilot examples...")

    results: List[Dict[str, Any]] = []
    t_retrieval_start = time.perf_counter()

    for idx, row in pilot_examples.iterrows():
        ex_id = str(row["id"])
        source_dataset = str(row["source_dataset"])
        original_label = int(row["original_label"])
        prompt = str(row["prompt"])
        orig_resp = str(row["original_response"])

        query_text = format_query_text(
            prompt=prompt,
            original_response=orig_resp,
            source_dataset=source_dataset,
        )

        # Context provenance classification
        if source_dataset == "halueval":
            evidence_status = "available"
        elif source_dataset == "fever":
            evidence_status = "insufficient_wikipedia_pointers_only"
        elif source_dataset == "truthfulqa":
            evidence_status = "insufficient_url_citations_only"
        else:
            evidence_status = "unknown"

        features = extract_retrieval_features(
            query_text=query_text,
            response_text=orig_resp,
            corpus_texts=corpus_texts,
            corpus_embeddings=corpus_embeddings,
            corpus_meta=corpus_meta,
            model=model,
            k=top_k,
        )

        rec = {
            "id": ex_id,
            "source_dataset": source_dataset,
            "original_label": original_label,
            "prompt": prompt,
            "original_response": orig_resp,
            "query_text": query_text,
            "evidence_status": evidence_status,
            **features,
        }
        results.append(rec)

        logger.info(
            f"[{idx + 1:02d}/{total_examples}] {ex_id} ({source_dataset}) | "
            f"Status: {evidence_status} | "
            f"Top1 Q-Sim: {features['top1_evidence_similarity']:.3f} | "
            f"Top1 R-Sim: {features['response_top1_evidence_similarity']:.3f} | "
            f"Margin: {features['evidence_margin']:.3f} | "
            f"Agreement: {features['retrieval_agreement']:.3f}"
        )

    retrieval_time = time.perf_counter() - t_retrieval_start
    total_query_and_retrieval_time = corpus_enc_time + retrieval_time

    df_results = pd.DataFrame(results)

    # Save artifacts
    parquet_path = out_path / "pilot_retrieval_results.parquet"
    csv_path = out_path / "pilot_retrieval_results.csv"
    summary_json_path = out_path / "pilot_retrieval_summary.json"

    df_results.to_parquet(parquet_path, index=False)
    df_results.to_csv(csv_path, index=False)

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Dataset breakdown
    per_dataset_summary: Dict[str, Dict[str, Any]] = {}
    for ds_name, grp in df_results.groupby("source_dataset"):
        per_dataset_summary[str(ds_name)] = {
            "count": int(len(grp)),
            "evidence_status": str(grp["evidence_status"].iloc[0]),
            "mean_top1_evidence_similarity": round(float(grp["top1_evidence_similarity"].mean()), 4),
            "mean_top3_evidence_similarity": round(float(grp["mean_top3_evidence_similarity"].mean()), 4),
            "mean_response_top1_similarity": round(float(grp["response_top1_evidence_similarity"].mean()), 4),
            "mean_response_top3_similarity": round(float(grp["mean_response_top3_evidence_similarity"].mean()), 4),
            "mean_evidence_margin": round(float(grp["evidence_margin"].mean()), 4),
            "mean_retrieval_agreement": round(float(grp["retrieval_agreement"].mean()), 4),
        }

    successful_count = int((df_results["evidence_status"] == "available").sum())
    insufficient_count = int((df_results["evidence_status"] != "available").sum())

    summary_metrics: Dict[str, Any] = {
        "number_of_examples": total_examples,
        "number_of_usable_evidence_items": len(corpus_texts),
        "number_with_successful_retrieval": successful_count,
        "number_with_insufficient_evidence": insufficient_count,
        "retrieval_model": model_name,
        "device": device,
        "model_loading_time_s": round(load_time, 3),
        "corpus_encoding_time_s": round(corpus_enc_time, 3),
        "query_retrieval_time_s": round(retrieval_time, 3),
        "total_retrieval_embedding_time_s": round(total_query_and_retrieval_time, 3),
        "avg_retrieval_time_per_query_s": round(retrieval_time / max(total_examples, 1), 4),
        "mean_top1_evidence_similarity": round(float(df_results["top1_evidence_similarity"].mean()), 4),
        "mean_top3_evidence_similarity": round(float(df_results["mean_top3_evidence_similarity"].mean()), 4),
        "mean_response_evidence_similarity": round(float(df_results["response_top1_evidence_similarity"].mean()), 4),
        "mean_response_top3_similarity": round(float(df_results["mean_response_top3_evidence_similarity"].mean()), 4),
        "mean_evidence_margin": round(float(df_results["evidence_margin"].mean()), 4),
        "mean_retrieval_agreement": round(float(df_results["retrieval_agreement"].mean()), 4),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
        "per_dataset_summary": per_dataset_summary,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"Retrieval pilot completed successfully. Summary saved to '{summary_json_path}'.")
    logger.info(f"Summary metrics: {json.dumps(summary_metrics, indent=2)}")

    return df_results, summary_metrics


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evidence retrieval and agreement pipeline.")
    parser.add_argument("--data", type=str, default="experiments/baselines/self_consistency/pilot_generations_raw.parquet", help="Path to input pilot generations.")
    parser.add_argument("--corpus-data", type=str, default="data/processed/combined_processed.parquet", help="Path to dataset for corpus extraction.")
    parser.add_argument("--output-dir", type=str, default="experiments/baselines/evidence_retrieval", help="Output directory.")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model identifier.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k evidence chunks to retrieve.")

    args = parser.parse_args()

    run_evidence_retrieval_pilot(
        input_data_path=args.data,
        corpus_data_path=args.corpus_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
