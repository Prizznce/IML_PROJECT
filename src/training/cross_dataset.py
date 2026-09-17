"""
Cross-Dataset Evaluation Module for LLM Hallucination Detection Baselines.

Implements strict Leave-One-Dataset-Out (LODO) transfer evaluation across
HaluEval, TruthfulQA, and FEVER:
1. Train on HaluEval + TruthfulQA  -> Test on FEVER
2. Train on HaluEval + FEVER       -> Test on TruthfulQA
3. Train on TruthfulQA + FEVER     -> Test on HaluEval

Guarantees:
- Zero data leakage: Target dataset is completely withheld from training and preprocessing.
- StandardScaler is fitted strictly on the training sources.
- Uses strictly the 11 primary internal signal features (num_tokens is excluded).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd

from src.training.baseline_models import (
    PRIMARY_SIGNAL_FEATURES,
    prepare_feature_dataframe,
    build_logistic_regression_pipeline,
    build_xgboost_model,
    evaluate_predictions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cross_dataset_evaluation")


BENCHMARK_DATASETS: List[str] = ["halueval", "truthfulqa", "fever"]


def get_leave_one_out_partitions(
    df: pd.DataFrame,
    target_dataset: str,
    include_num_tokens: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Partition dataset into training sources and an unseen target evaluation dataset.

    Parameters:
        df: Full supervised feature dataset.
        target_dataset: Name of dataset to withhold for testing ('halueval', 'truthfulqa', or 'fever').
        include_num_tokens: Whether to include num_tokens (must remain False for primary baseline).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, meta_train, meta_test).
    """
    target_clean = str(target_dataset).lower().strip()
    available_sources = set(df["source_dataset"].str.lower().unique())

    if target_clean not in available_sources:
        raise ValueError(
            f"Target dataset '{target_clean}' not found in dataset sources: {available_sources}"
        )

    # 1. Prepare features and enforce column exclusions
    X, y, meta = prepare_feature_dataframe(df, include_num_tokens=include_num_tokens)

    # 2. Strict partition by source_dataset
    test_mask = (meta["source_dataset"].str.lower() == target_clean)
    train_mask = ~test_mask

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    meta_train = meta[train_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    meta_test = meta[test_mask].reset_index(drop=True)

    # Sanity checks: Verify complete target exclusion
    assert target_clean not in meta_train["source_dataset"].str.lower().values, (
        f"Data leakage detected: Target dataset '{target_clean}' found in training split!"
    )
    assert set(meta_test["source_dataset"].str.lower().unique()) == {target_clean}, (
        f"Contamination: Test split contains non-target sources: {meta_test['source_dataset'].unique()}"
    )

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def evaluate_transfer_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    target_dataset: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train baseline classifiers on training datasets and evaluate on unseen target dataset.

    Parameters:
        X_train, X_test: Feature matrices.
        y_train, y_test: Ground-truth target series.
        meta_train, meta_test: Metadata dataframes.
        target_dataset: Target dataset name.
        random_state: Deterministic random state for models.

    Returns:
        Dict containing fold metrics for Logistic Regression and XGBoost.
    """
    train_sources = sorted(list(meta_train["source_dataset"].str.lower().unique()))
    train_source_counts = {
        src: int((meta_train["source_dataset"].str.lower() == src).sum())
        for src in train_sources
    }
    train_label_dist = {
        int(lbl): int((y_train == lbl).sum())
        for lbl in sorted(y_train.unique())
    }
    test_label_dist = {
        int(lbl): int((y_test == lbl).sum())
        for lbl in sorted(y_test.unique())
    }

    logger.info(
        f"Transfer experiment: Target='{target_dataset}' (N={len(X_test)}) | "
        f"Train sources={train_sources} (N={len(X_train)})"
    )

    # 1. Logistic Regression (StandardScaler fit ONLY on X_train)
    lr_pipeline = build_logistic_regression_pipeline(random_state=random_state)
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob)

    # 2. XGBoost (Trained strictly on X_train)
    xgb_model = build_xgboost_model(random_state=random_state)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_prob)

    return {
        "target_dataset": target_dataset,
        "train_sources": train_sources,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_source_counts": train_source_counts,
        "train_label_distribution": train_label_dist,
        "test_label_distribution": test_label_dist,
        "num_features": len(X_train.columns),
        "features_used": list(X_train.columns),
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
    }


def run_cross_dataset_evaluation(
    data_path: Union[str, Path] = "experiments/baselines/supervised_signals_combined.parquet",
    output_dir: Union[str, Path] = "experiments/baselines/results/cross_dataset",
    random_state: int = 42,
    include_num_tokens: bool = False,
) -> Dict[str, Any]:
    """
    Execute complete Leave-One-Dataset-Out cross-dataset evaluation protocol:
      1. Train on HaluEval + TruthfulQA -> Test on FEVER
      2. Train on HaluEval + FEVER       -> Test on TruthfulQA
      3. Train on TruthfulQA + FEVER     -> Test on HaluEval

    Parameters:
        data_path: Path to supervised signal parquet dataset.
        output_dir: Output directory for saving cross_dataset_results.json.
        random_state: Seed for models.
        include_num_tokens: If True, includes num_tokens (must remain False for primary).

    Returns:
        Dict containing results for all three transfer experiments.
    """
    d_path = Path(data_path)
    if not d_path.exists():
        raise FileNotFoundError(f"Dataset not found at {d_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {d_path}...")
    df = pd.read_parquet(d_path)

    experiments_results: Dict[str, Any] = {}

    for target in BENCHMARK_DATASETS:
        logger.info("=" * 60)
        logger.info(f"LEAVE-ONE-DATASET-OUT: TARGET = {target.upper()}")
        logger.info("=" * 60)

        X_train, X_test, y_train, y_test, meta_train, meta_test = get_leave_one_out_partitions(
            df=df,
            target_dataset=target,
            include_num_tokens=include_num_tokens,
        )

        fold_res = evaluate_transfer_fold(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            meta_train=meta_train,
            meta_test=meta_test,
            target_dataset=target,
            random_state=random_state,
        )

        experiments_results[f"test_on_{target}"] = fold_res

        # Log immediate summary
        for model in ["logistic_regression", "xgboost"]:
            m = fold_res[model]
            logger.info(
                f"  [{model.upper()}] Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | "
                f"ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | "
                f"Brier={m['brier_score']:.4f} | ECE={m['ece']:.4f}"
            )

    full_results = {
        "metadata": {
            "dataset": str(data_path),
            "random_state": random_state,
            "evaluation_protocol": "Leave-One-Dataset-Out (LODO)",
            "num_features": len(PRIMARY_SIGNAL_FEATURES) + (1 if include_num_tokens else 0),
            "features_used": PRIMARY_SIGNAL_FEATURES + (["num_tokens"] if include_num_tokens else []),
            "include_num_tokens": include_num_tokens,
            "target_datasets": BENCHMARK_DATASETS,
        },
        "experiments": experiments_results,
    }

    results_file = out_dir / "cross_dataset_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Saved complete cross-dataset results to {results_file}")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Evaluation (Phase 6)")
    parser.add_argument(
        "--data",
        type=str,
        default="experiments/baselines/supervised_signals_combined.parquet",
        help="Path to supervised signal dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baselines/results/cross_dataset",
        help="Directory to save cross-dataset evaluation metrics",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible training",
    )
    parser.add_argument(
        "--include-num-tokens",
        action="store_true",
        help="Include num_tokens (for ablation only; default: False)",
    )

    args = parser.parse_args()
    run_cross_dataset_evaluation(
        data_path=args.data,
        output_dir=args.output_dir,
        random_state=args.random_state,
        include_num_tokens=args.include_num_tokens,
    )


if __name__ == "__main__":
    main()
