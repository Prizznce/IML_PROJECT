"""
End-to-End Retraining Pipeline for LLM Hallucination Detection.

Orchestrates the complete multi-signal training workflow for any target LLM
(e.g., meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-1B, Qwen/Qwen3.5-0.8B):
  Step 1: Extract 11 internal generation signals (probabilities, entropy, perplexity, ranks).
  Step 2: Generate stochastic responses and extract 5 self-consistency consensus signals.
  Step 3: Extract 3 pairwise Natural Language Inference (NLI) agreement signals.
  Step 4: Assemble canonical 19-feature universal dataset.
  Step 5: Train and calibrate Logistic Regression and XGBoost classifiers.
  Step 6: Serialize model artifacts (joblib/json) and evaluate test metrics.

Strictly isolates model-specific artifacts to experiments/models/{model_slug}/
and models/{model_slug}/, keeping existing baseline results intact.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
import torch

from src.consistency.nli_agreement import run_nli_agreement_full
from src.consistency.self_consistency import run_self_consistency_full
from src.features.build_universal_features import (
    UNIVERSAL_CORE_FEATURES,
    build_and_save_universal_features,
)
from src.signals.score_labeled_responses import score_labeled_dataset
from src.training.combined_models import (
    build_logistic_regression_pipeline,
    build_xgboost_model,
    compute_ece,
    prepare_combined_feature_dataframe,
    split_dataset,
)
from src.utils.model_registry import (
    DEFAULT_MODEL_ID,
    get_hf_token,
    get_model_experiment_dir,
    get_model_slug,
    get_saved_model_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("retrain_pipeline")


def retrain_for_model(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
    dataset: str = "combined",
    limit: Optional[int] = None,
    seed: int = 42,
    test_size: float = 0.2,
    skip_signals: bool = False,
    skip_consistency: bool = False,
    skip_nli: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Execute full multi-signal retraining pipeline for a specific LLM.

    Parameters:
        model_id: Hugging Face model repository ID.
        hf_token: Optional Hugging Face token for gated models (e.g. Llama 3.2).
        device: Compute device ('cuda:0' or 'cpu'). Defaults to CUDA if available.
        dataset: Target benchmark dataset ('combined', 'halueval', etc.).
        limit: Optional sample limit (for quick pilot testing or development).
        seed: Random seed for deterministic reproducibility.
        test_size: Test fraction for stratified split.
        skip_signals: If True, skips step 1 if output file exists.
        skip_consistency: If True, skips step 2 if output file exists.
        skip_nli: If True, skips step 3 if output file exists.
        resume: If True, uses existing checkpoint files.

    Returns:
        Dict containing training summary, evaluation metrics, and artifact paths.
    """
    t_start = time.perf_counter()
    slug = get_model_slug(model_id)
    token = get_hf_token(hf_token)

    if device is None:
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    logger.info("=" * 70)
    logger.info(f"RETRAINING PIPELINE FOR MODEL: {model_id} (slug: {slug})")
    logger.info(f"Target device: {target_device} | Limit: {limit} | Seed: {seed}")
    logger.info("=" * 70)

    # Directories
    exp_dir = get_model_experiment_dir(model_id)
    model_save_dir = get_saved_model_dir(model_id)
    combined_exp_dir = exp_dir / "combined"
    sc_exp_dir = exp_dir / "self_consistency"
    nli_exp_dir = exp_dir / "nli_agreement"

    for d in (combined_exp_dir, sc_exp_dir, nli_exp_dir, model_save_dir):
        d.mkdir(parents=True, exist_ok=True)

    input_data_path = f"data/processed/{dataset}_processed.parquet"
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Processed input benchmark not found: '{input_data_path}'")

    signals_parquet = exp_dir / f"supervised_signals_{dataset}.parquet"
    signals_csv = exp_dir / f"supervised_signals_{dataset}.csv"

    # --------------------------------------------------------------------------
    # Step 1: Internal Signal Extraction
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 1: Internal Generation Signal Extraction ---")
    if skip_signals and signals_parquet.exists():
        logger.info(f"Step 1 skipped: Reusing existing signals at '{signals_parquet}'")
        df_signals = pd.read_parquet(signals_parquet)
    else:
        logger.info(f"Scoring benchmark responses using model '{model_id}'...")
        df_signals, signals_summary = score_labeled_dataset(
            input_path=input_data_path,
            output_parquet=str(signals_parquet),
            output_csv=str(signals_csv),
            model_id=model_id,
            device=target_device,
            limit=limit,
            seed=seed,
            resume=resume,
            token=token,
        )
        logger.info(f"Step 1 completed: {len(df_signals)} examples scored.")

    # --------------------------------------------------------------------------
    # Step 2: Self-Consistency Probing (k=5 stochastic responses)
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 2: Behavioral Self-Consistency Probing ---")
    sc_parquet = sc_exp_dir / "full_self_consistency_aggregated.parquet"
    sc_raw_parquet = sc_exp_dir / "full_generations_raw.parquet"

    if skip_consistency and sc_parquet.exists():
        logger.info(f"Step 2 skipped: Reusing existing self-consistency at '{sc_parquet}'")
        df_sc = pd.read_parquet(sc_parquet)
    else:
        logger.info(f"Generating stochastic responses and computing similarity signals...")
        df_gen, df_sc, sc_summary = run_self_consistency_full(
            data_path=str(signals_parquet),
            output_dir=str(sc_exp_dir),
            model_id=model_id,
            device=target_device,
            limit=limit,
            seed=seed,
            resume=resume,
            token=token,
        )
        logger.info(f"Step 2 completed: {len(df_sc)} aggregated consistency records.")

    # --------------------------------------------------------------------------
    # Step 3: Natural Language Inference Agreement
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 3: Pairwise NLI Agreement Scoring ---")
    nli_parquet = nli_exp_dir / "full_nli_aggregated.parquet"

    if skip_nli and nli_parquet.exists():
        logger.info(f"Step 3 skipped: Reusing existing NLI agreement at '{nli_parquet}'")
        df_nli = pd.read_parquet(nli_parquet)
    else:
        logger.info(f"Extracting pairwise NLI agreement signals across responses...")
        df_pairs, df_nli, nli_summary = run_nli_agreement_full(
            input_generations_path=str(sc_raw_parquet),
            output_dir=str(nli_exp_dir),
            device=target_device,
            limit=limit,
            resume=resume,
        )
        logger.info(f"Step 3 completed: {len(df_nli)} NLI records.")

    # --------------------------------------------------------------------------
    # Step 4: Universal Feature Assembly (19 features)
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 4: Canonical 19-Feature Universal Table Assembly ---")
    universal_parquet = combined_exp_dir / "universal_features.parquet"
    df_universal, feat_summary = build_and_save_universal_features(
        internal_signals_path=str(signals_parquet),
        self_consistency_path=str(sc_parquet),
        nli_agreement_path=str(nli_parquet),
        output_dir=str(combined_exp_dir),
    )
    logger.info(f"Step 4 completed: Universal feature table shape: {df_universal.shape}")

    # --------------------------------------------------------------------------
    # Step 5: Supervised Model Training (Logistic Regression & XGBoost)
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 5: Supervised Classifier Training & Split Evaluation ---")
    X, y, meta = prepare_combined_feature_dataframe(df_universal)

    X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
        X, y, meta, test_size=test_size, random_state=seed
    )
    logger.info(f"Dataset split: {len(X_train)} train, {len(X_test)} test (test_size={test_size}).")

    # Fit Logistic Regression Pipeline
    logger.info("Fitting Logistic Regression pipeline with StandardScaler...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=seed)
    lr_pipeline.fit(X_train, y_train)

    # Fit XGBoost Classifier
    logger.info("Fitting XGBoost Classifier...")
    xgb_model = build_xgboost_model(random_state=seed)
    xgb_model.fit(X_train, y_train)

    # Evaluate on holdout test split
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

    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)

    def _eval(p, y_true):
        preds = (p >= 0.5).astype(int)
        cm = confusion_matrix(y_true, preds).ravel()
        tn, fp, fn, tp = cm if len(cm) == 4 else (0, 0, 0, 0)
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else 0.5,
            "pr_auc": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else 0.5,
            "brier_score": float(brier_score_loss(y_true, p)),
            "ece": float(compute_ece(y_true.to_numpy(), p)),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        }

    lr_metrics = _eval(lr_prob, y_test)
    xgb_metrics = _eval(xgb_prob, y_test)

    # --------------------------------------------------------------------------
    # Step 6: Artifact Serialization & Summary
    # --------------------------------------------------------------------------
    logger.info("\n--- STEP 6: Serializing Model Checkpoints and Evaluation Results ---")

    # Serialize in models/{slug}/ and experiments/models/{slug}/combined/
    for dest_dir in (model_save_dir, combined_exp_dir):
        joblib.dump(lr_pipeline, dest_dir / "lr_pipeline.joblib")
        joblib.dump(xgb_model, dest_dir / "xgb_model.joblib")

    # Save test predictions
    predictions_df = pd.DataFrame({
        "id": meta_test["id"].to_numpy(),
        "source_dataset": meta_test["source_dataset"].to_numpy(),
        "label": y_test.to_numpy(),
        "lr_probability": lr_prob,
        "lr_prediction": lr_pred,
        "xgb_probability": xgb_prob,
        "xgb_prediction": xgb_pred,
    })
    predictions_df.to_parquet(combined_exp_dir / "combined_test_predictions.parquet", index=False)
    predictions_df.to_csv(combined_exp_dir / "combined_test_predictions.csv", index=False)

    total_pipeline_time = time.perf_counter() - t_start

    final_results = {
        "model_id": model_id,
        "model_slug": slug,
        "training_time_s": round(total_pipeline_time, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": len(UNIVERSAL_CORE_FEATURES),
        "features": list(UNIVERSAL_CORE_FEATURES),
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "artifacts": {
            "signals_parquet": str(signals_parquet),
            "universal_features_parquet": str(universal_parquet),
            "lr_model_joblib": str(model_save_dir / "lr_pipeline.joblib"),
            "xgb_model_joblib": str(model_save_dir / "xgb_model.joblib"),
        },
    }

    results_json_path = combined_exp_dir / "combined_model_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    logger.info("=" * 70)
    logger.info("RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"LR  Accuracy: {lr_metrics['accuracy']:.4f} | F1: {lr_metrics['f1']:.4f} | ROC-AUC: {lr_metrics['roc_auc']:.4f} | ECE: {lr_metrics['ece']:.4f}")
    logger.info(f"XGB Accuracy: {xgb_metrics['accuracy']:.4f} | F1: {xgb_metrics['f1']:.4f} | ROC-AUC: {xgb_metrics['roc_auc']:.4f} | ECE: {xgb_metrics['ece']:.4f}")
    logger.info(f"Fitted classifiers saved to '{model_save_dir}'")
    logger.info("=" * 70)

    return final_results


def main():
    parser = argparse.ArgumentParser(description="End-to-End Retraining Pipeline for LLM Hallucination Detection.")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face model ID to train for.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face user token for gated model access.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda:0' or 'cpu').")
    parser.add_argument("--dataset", type=str, default="combined", help="Benchmark dataset ('combined', 'halueval', etc.).")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit for quick validation runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction.")
    parser.add_argument("--skip-signals", action="store_true", help="Skip signal scoring if parquet exists.")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip self-consistency if parquet exists.")
    parser.add_argument("--skip-nli", action="store_true", help="Skip NLI scoring if parquet exists.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from intermediate checkpoints.")

    args = parser.parse_args()

    retrain_for_model(
        model_id=args.model_id,
        hf_token=args.hf_token,
        device=args.device,
        dataset=args.dataset,
        limit=args.limit,
        seed=args.seed,
        test_size=args.test_size,
        skip_signals=args.skip_signals,
        skip_consistency=args.skip_consistency,
        skip_nli=args.skip_nli,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
