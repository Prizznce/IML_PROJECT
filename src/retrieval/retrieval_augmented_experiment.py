"""
Retrieval-Augmented Hallucination Detection Variant Pipeline.

Implements and evaluates the standalone retrieval-augmented experimental variant
across HaluEval and FEVER (N=2,500), comparing 5 feature conditions across
Logistic Regression and XGBoost classifiers on an identical holdout split (N=500).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.build_universal_features import (
    INTERNAL_SIGNAL_FEATURES,
    UNIVERSAL_CORE_FEATURES,
)
from src.retrieval.evidence_retrieval import (
    calculate_evidence_margin,
    calculate_retrieval_agreement,
    compute_cosine_similarity,
    rank_top_k,
)

logger = logging.getLogger("retrieval_augmented_experiment")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

RETRIEVAL_FEATURES: list[str] = [
    "top1_evidence_similarity",
    "top3_evidence_similarity",
    "response_top1_similarity",
    "response_top3_similarity",
    "evidence_response_margin",
    "retrieval_agreement",
]

CONDITION_FEATURE_MAP: dict[str, list[str]] = {
    "internal_only": INTERNAL_SIGNAL_FEATURES,
    "universal_core": UNIVERSAL_CORE_FEATURES,
    "retrieval_only": RETRIEVAL_FEATURES,
    "internal_plus_retrieval": INTERNAL_SIGNAL_FEATURES + RETRIEVAL_FEATURES,
    "universal_plus_retrieval": UNIVERSAL_CORE_FEATURES + RETRIEVAL_FEATURES,
}

CONDITION_DISPLAY_NAMES: dict[str, str] = {
    "internal_only": "Condition A (Internal-Only)",
    "universal_core": "Condition B (Universal Core)",
    "retrieval_only": "Condition C (Retrieval-Only)",
    "internal_plus_retrieval": "Condition D (Internal + Retrieval)",
    "universal_plus_retrieval": "Condition E (Universal + Retrieval)",
}


def clean_fever_title(title_raw: str) -> str:
    """Clean Wikipedia page title from FEVER pointer syntax."""
    t = str(title_raw).replace("_", " ")
    t = t.replace("-LRB-", "(").replace("-RRB-", ")")
    t = t.replace("-COLON-", ":").replace("-SLASH-", "/")
    return re.sub(r"\s+", " ", t).strip()


def build_dataset_evidence_corpora(
    processed_data_path: str | Path = "data/processed/combined_processed.parquet",
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """
    Construct separate, dataset-appropriate evidence corpora for HaluEval and FEVER.
    - HaluEval: Unique reference passages in 'context'.
    - FEVER: Unique clean Wikipedia article titles parsed from 'context' pointers.
    - TruthfulQA: Excluded due to containing reference URLs rather than textual evidence.
    """
    path = Path(processed_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    df = pd.read_parquet(path)

    # 1. HaluEval Evidence Corpus
    halu_df = df[df["source_dataset"] == "halueval"]
    halu_passages = []
    halu_seen = set()
    for ctx in halu_df["context"]:
        ctx_str = str(ctx).strip()
        if ctx_str and ctx_str not in halu_seen:
            halu_seen.add(ctx_str)
            halu_passages.append(ctx_str)

    # 2. FEVER Evidence Corpus
    fever_df = df[df["source_dataset"] == "fever"]
    fever_titles = []
    fever_seen = set()
    for ctx in fever_df["context"]:
        try:
            data = ast.literal_eval(str(ctx))
            for annotator in data:
                for item in annotator:
                    if len(item) >= 3 and item[2]:
                        clean_t = clean_fever_title(item[2])
                        if clean_t and clean_t not in fever_seen:
                            fever_seen.add(clean_t)
                            fever_titles.append(clean_t)
        except Exception:
            continue

    corpora = {
        "halueval": halu_passages,
        "fever": fever_titles,
    }

    manifest = {
        "source_dataset_file": str(path),
        "halueval": {
            "corpus_size": len(halu_passages),
            "evidence_type": "benchmark_reference_passages",
            "sample_evidence": halu_passages[:2] if halu_passages else [],
        },
        "fever": {
            "corpus_size": len(fever_titles),
            "evidence_type": "clean_wikipedia_article_titles_from_context_pointers",
            "sample_evidence": fever_titles[:2] if fever_titles else [],
        },
        "truthfulqa": {
            "status": "excluded_from_retrieval_experiment",
            "reason": (
                "TruthfulQA context field contains external reference URLs and brief keywords "
                "rather than a local evidence corpus. Excluded to prevent artificial missingness "
                "from acting as a dataset-identifying confounder."
            ),
        },
    }

    logger.info(
        f"Built corpora: HaluEval={len(halu_passages)} passages, FEVER={len(fever_titles)} titles."
    )
    return corpora, manifest


def extract_dense_retrieval_features(
    usable_df: pd.DataFrame,
    corpora: dict[str, list[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda:0",
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Encode corpora and queries, retrieve top-k candidates, and compute 6 retrieval signals.
    """
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    logger.info(f"Loading SentenceTransformer '{model_name}' on device '{dev}'...")
    model = SentenceTransformer(model_name, device=dev)

    # Encode corpora
    corpus_embs: dict[str, np.ndarray] = {}
    for ds_name, texts in corpora.items():
        logger.info(f"Encoding {len(texts)} chunks for {ds_name} corpus...")
        embs = model.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=dev,
            show_progress_bar=False,
        )
        corpus_embs[ds_name] = embs

    # Prepare queries and responses
    # FEVER query: claim in 'response'; HaluEval query: question in 'prompt'
    queries: list[str] = []
    responses: list[str] = []
    for _, row in usable_df.iterrows():
        ds = str(row["source_dataset"]).lower()
        if ds == "fever":
            queries.append(str(row["response"]).strip())
        else:
            queries.append(str(row["prompt"]).strip())
        responses.append(str(row["response"]).strip())

    logger.info(f"Encoding {len(queries)} queries and {len(responses)} responses...")
    query_embs = model.encode(
        queries,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=dev,
        show_progress_bar=False,
    )
    response_embs = model.encode(
        responses,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=dev,
        show_progress_bar=False,
    )

    # Perform top-k retrieval and compute features
    feature_records: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(usable_df.iterrows()):
        ds = str(row["source_dataset"]).lower()
        c_texts = corpora[ds]
        c_embs = corpus_embs[ds]

        q_emb = query_embs[idx]
        r_emb = response_embs[idx]

        # Cosine similarity between query and corpus items
        q_sims = compute_cosine_similarity(q_emb, c_embs)
        top_indices, top_q_sims = rank_top_k(q_sims, k=top_k)

        # Response similarity to top retrieved evidence
        top_c_embs = c_embs[top_indices]
        r_sims = compute_cosine_similarity(r_emb, top_c_embs)

        top1_q = float(top_q_sims[0]) if top_q_sims else 0.0
        top3_q = float(np.mean(top_q_sims)) if top_q_sims else 0.0
        top1_r = float(r_sims[0]) if len(r_sims) > 0 else 0.0
        top3_r = float(np.mean(r_sims)) if len(r_sims) > 0 else 0.0
        margin = calculate_evidence_margin(top_q_sims)
        agreement = calculate_retrieval_agreement(top1_q, top1_r)

        feature_records.append({
            "id": row["id"],
            "source_dataset": row["source_dataset"],
            "label": row["label"],
            "top1_evidence_similarity": top1_q,
            "top3_evidence_similarity": top3_q,
            "response_top1_similarity": top1_r,
            "response_top3_similarity": top3_r,
            "evidence_response_margin": margin,
            "retrieval_agreement": agreement,
        })

    return pd.DataFrame(feature_records)


def compute_ece_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """Compute Expected Calibration Error and Maximum Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_size / n_samples) * diff
            if diff > mce:
                mce = diff

    return float(ece), float(mce)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute classification performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece, mce = compute_ece_mce(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "min_prob": float(np.min(y_prob)),
        "max_prob": float(np.max(y_prob)),
    }


def train_and_evaluate_retrieval_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train and evaluate Logistic Regression and XGBoost on the 5 feature conditions.
    Returns:
        (results_df, predictions_df)
    """
    results: list[dict[str, Any]] = []
    preds_records: list[dict[str, Any]] = []

    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    for cond_key, feat_cols in CONDITION_FEATURE_MAP.items():
        X_train = train_df[feat_cols].to_numpy(dtype=float)
        X_test = test_df[feat_cols].to_numpy(dtype=float)

        # 1. Logistic Regression
        lr_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
        ])
        lr_model.fit(X_train, y_train)
        lr_probs = lr_model.predict_proba(X_test)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)

        lr_metrics = evaluate_predictions(y_test, lr_preds, lr_probs)
        lr_metrics.update({
            "condition": cond_key,
            "condition_display": CONDITION_DISPLAY_NAMES[cond_key],
            "model": "logistic_regression",
            "num_features": len(feat_cols),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        })
        results.append(lr_metrics)

        for i, row in test_df.reset_index(drop=True).iterrows():
            preds_records.append({
                "id": row["id"],
                "source_dataset": row["source_dataset"],
                "label": int(row["label"]),
                "condition": cond_key,
                "model": "logistic_regression",
                "y_probability": float(lr_probs[i]),
                "y_prediction": int(lr_preds[i]),
            })

        # 2. XGBoost
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        xgb_preds = (xgb_probs >= 0.5).astype(int)

        xgb_metrics = evaluate_predictions(y_test, xgb_preds, xgb_probs)
        xgb_metrics.update({
            "condition": cond_key,
            "condition_display": CONDITION_DISPLAY_NAMES[cond_key],
            "model": "xgboost",
            "num_features": len(feat_cols),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        })
        results.append(xgb_metrics)

        for i, row in test_df.reset_index(drop=True).iterrows():
            preds_records.append({
                "id": row["id"],
                "source_dataset": row["source_dataset"],
                "label": int(row["label"]),
                "condition": cond_key,
                "model": "xgboost",
                "y_probability": float(xgb_probs[i]),
                "y_prediction": int(xgb_preds[i]),
            })

    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame(preds_records)

    # Compute deltas relative to internal_only and universal_core
    for model_name in ["logistic_regression", "xgboost"]:
        m_mask = results_df["model"] == model_name

        # Internal-only baseline
        internal_row = results_df[m_mask & (results_df["condition"] == "internal_only")].iloc[0]
        # Universal-core baseline
        universal_row = results_df[m_mask & (results_df["condition"] == "universal_core")].iloc[0]

        for metric in ["accuracy", "f1", "roc_auc", "pr_auc", "brier_score", "ece", "mce"]:
            results_df.loc[m_mask, f"delta_vs_internal_{metric}"] = (
                results_df.loc[m_mask, metric] - internal_row[metric]
            )
            results_df.loc[m_mask, f"delta_vs_universal_{metric}"] = (
                results_df.loc[m_mask, metric] - universal_row[metric]
            )

    return results_df, preds_df


def analyze_retrieval_quality_and_errors(
    test_df: pd.DataFrame,
    preds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive profiling of retrieval quality metrics and error categories.
    """
    merged = test_df.copy()

    # Add predictions for Condition E (universal_plus_retrieval) and Condition B (universal_core)
    for model_name, prefix in [("logistic_regression", "lr"), ("xgboost", "xgb")]:
        cond_b_preds = preds_df[
            (preds_df["model"] == model_name) & (preds_df["condition"] == "universal_core")
        ].set_index("id")
        cond_e_preds = preds_df[
            (preds_df["model"] == model_name) & (preds_df["condition"] == "universal_plus_retrieval")
        ].set_index("id")

        merged[f"{prefix}_cond_b_prob"] = merged["id"].map(cond_b_preds["y_probability"])
        merged[f"{prefix}_cond_b_pred"] = merged["id"].map(cond_b_preds["y_prediction"])
        merged[f"{prefix}_cond_e_prob"] = merged["id"].map(cond_e_preds["y_probability"])
        merged[f"{prefix}_cond_e_pred"] = merged["id"].map(cond_e_preds["y_prediction"])

        # Error category for Condition E
        y = merged["label"].to_numpy()
        p = merged[f"{prefix}_cond_e_pred"].to_numpy()
        cats = np.empty(len(merged), dtype=object)
        cats[(y == 1) & (p == 1)] = "TP"
        cats[(y == 0) & (p == 0)] = "TN"
        cats[(y == 0) & (p == 1)] = "FP"
        cats[(y == 1) & (p == 0)] = "FN"
        merged[f"{prefix}_cond_e_category"] = cats

    return merged


def run_retrieval_augmented_experiment(
    processed_data_path: str | Path = "data/processed/combined_processed.parquet",
    universal_features_path: str | Path = "experiments/baselines/combined/universal_features.parquet",
    output_dir: str | Path = "experiments/retrieval_augmented",
    device: str = "cuda:0",
) -> dict[str, Any]:
    """
    Main orchestration entrypoint for Phase 19.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc_path = Path(processed_data_path)
    univ_path = Path(universal_features_path)

    # 1. Corpora construction
    corpora, manifest = build_dataset_evidence_corpora(proc_path)
    with open(out_dir / "retrieval_corpus_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. Filter usable dataset (HaluEval + FEVER)
    all_proc_df = pd.read_parquet(proc_path)
    usable_df = all_proc_df[all_proc_df["source_dataset"].isin(["halueval", "fever"])].copy()
    usable_df = usable_df.sort_values("id").reset_index(drop=True)

    logger.info(
        f"Usable dataset size: {len(usable_df)} (HaluEval: {(usable_df['source_dataset']=='halueval').sum()}, "
        f"FEVER: {(usable_df['source_dataset']=='fever').sum()})"
    )

    # 3. Extract dense retrieval features
    retrieval_feats_df = extract_dense_retrieval_features(
        usable_df=usable_df,
        corpora=corpora,
        device=device,
        top_k=3,
    )

    # 4. Merge with existing 19 universal features
    univ_df = pd.read_parquet(univ_path)
    merged_full = usable_df[["id", "source_dataset", "label"]].merge(
        univ_df[["id"] + UNIVERSAL_CORE_FEATURES], on="id", how="inner"
    )
    merged_full = merged_full.merge(
        retrieval_feats_df[["id"] + RETRIEVAL_FEATURES], on="id", how="inner"
    )

    if len(merged_full) != len(usable_df):
        raise ValueError(
            f"Feature merge count mismatch: expected {len(usable_df)}, got {len(merged_full)}"
        )

    # Save feature tables
    merged_full.to_parquet(out_dir / "retrieval_features.parquet", index=False)
    merged_full.to_csv(out_dir / "retrieval_features.csv", index=False)

    # 5. Deterministic stratified 80/20 train/test split on usable 2,500 dataset
    strat_key = merged_full["source_dataset"] + "_" + merged_full["label"].astype(str)
    train_df, test_df = train_test_split(
        merged_full,
        test_size=0.2,
        random_state=42,
        stratify=strat_key,
    )
    train_df = train_df.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    logger.info(
        f"Partitioned usable dataset: Train N={len(train_df)} (label 0: {(train_df['label']==0).sum()}, label 1: {(train_df['label']==1).sum()}), "
        f"Test N={len(test_df)} (label 0: {(test_df['label']==0).sum()}, label 1: {(test_df['label']==1).sum()})"
    )

    # 6. Train and evaluate models across 5 conditions
    results_df, preds_df = train_and_evaluate_retrieval_models(train_df, test_df)

    results_df.to_csv(out_dir / "retrieval_experiment_results.csv", index=False)
    preds_df.to_parquet(out_dir / "retrieval_predictions.parquet", index=False)

    # 7. Error analysis and retrieval quality profiling
    error_analysis_df = analyze_retrieval_quality_and_errors(test_df, preds_df)
    error_analysis_df.to_csv(out_dir / "retrieval_error_analysis.csv", index=False)

    # 8. Compile master results JSON
    master_results = {
        "metadata": {
            "experiment": "Phase 19 — Retrieval-Augmented Hallucination Detection Variant",
            "usable_total_samples": len(usable_df),
            "excluded_samples": int((all_proc_df["source_dataset"] == "truthfulqa").sum()),
            "excluded_reason": "TruthfulQA context field contains reference URLs rather than textual evidence.",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "retrieval_features": RETRIEVAL_FEATURES,
            "conditions_evaluated": list(CONDITION_FEATURE_MAP.keys()),
        },
        "retrieval_quality_summary": {
            "halueval": {
                "mean_top1_evidence_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "halueval", "top1_evidence_similarity"].mean()),
                "mean_top3_evidence_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "halueval", "top3_evidence_similarity"].mean()),
                "mean_response_top1_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "halueval", "response_top1_similarity"].mean()),
                "mean_response_top3_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "halueval", "response_top3_similarity"].mean()),
                "mean_evidence_response_margin": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "halueval", "evidence_response_margin"].mean()),
                "mean_retrieval_agreement": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "halueval", "retrieval_agreement"].mean()),
            },
            "fever": {
                "mean_top1_evidence_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "fever", "top1_evidence_similarity"].mean()),
                "mean_top3_evidence_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "fever", "top3_evidence_similarity"].mean()),
                "mean_response_top1_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "fever", "response_top1_similarity"].mean()),
                "mean_response_top3_similarity": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "fever", "response_top3_similarity"].mean()),
                "mean_evidence_response_margin": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "fever", "evidence_response_margin"].mean()),
                "mean_retrieval_agreement": float(retrieval_feats_df.loc[retrieval_feats_df["source_dataset"] == "fever", "retrieval_agreement"].mean()),
            },
        },
        "results_records": results_df.to_dict(orient="records"),
    }

    with open(out_dir / "retrieval_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    # 9. Format executive summary markdown
    generate_summary_markdown(out_dir / "retrieval_summary.md", results_df, manifest)

    return master_results


def generate_summary_markdown(
    filepath: Path,
    results_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    """Format markdown summary table."""
    lines: list[str] = [
        "# Retrieval-Augmented Hallucination Detection Variant — Summary",
        "",
        "## 1. Experimental Setup & Evidence Corpora",
        f"- **Usable Dataset**: $N = 2,500$ instances (HaluEval: 1,500, FEVER: 1,000)",
        f"- **Holdout Test Set**: $N = 500$ instances (HaluEval: 300, FEVER: 200; 250 label 0, 250 label 1)",
        f"- **HaluEval Evidence Corpus**: {manifest['halueval']['corpus_size']} benchmark reference passages",
        f"- **FEVER Evidence Corpus**: {manifest['fever']['corpus_size']} clean Wikipedia article titles from context pointers",
        f"- **TruthfulQA Exclusion**: Documented benchmark limitation; context contains URLs rather than local text",
        "- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Dense cosine retrieval, top-k=3)",
        "",
        "## 2. Model Performance Across 5 Feature Conditions ($N = 500$ Holdout Set)",
        "",
        "| Condition | Feats | Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier | ECE | Δ(vs Int) ROC | Δ(vs Univ) ROC |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in results_df.iterrows():
        d_int = f"{row['delta_vs_internal_roc_auc']:+.4f}"
        d_univ = f"{row['delta_vs_universal_roc_auc']:+.4f}"
        lines.append(
            f"| {row['condition_display']} | {row['num_features']} | {row['model']} | "
            f"{row['accuracy']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
            f"{row['pr_auc']:.4f} | {row['brier_score']:.4f} | {row['ece']:.4f} | "
            f"{d_int} | {d_univ} |"
        )

    lines.extend([
        "",
        "## 3. Methodological Observations",
        "- **Fairness Rule**: All 5 conditions evaluated on the exact same 500 holdout instances.",
        "- **Zero Leakage**: No labels, target responses, or generated responses in retrieval corpora.",
        "- **Standalone Variant**: Retrieval signals are evaluated as a separate variant and are not incorporated into the Universal Core Detector.",
        "",
    ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run retrieval-augmented hallucination detector experiment.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device.")
    parser.add_argument("--output-dir", type=str, default="experiments/retrieval_augmented", help="Output directory.")
    args = parser.parse_args()

    run_retrieval_augmented_experiment(device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
