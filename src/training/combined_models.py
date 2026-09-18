"""
Universal Combined Hallucination Detector Training and Evaluation Pipeline.

Trains and validates supervised classifiers (Logistic Regression with StandardScaler
and XGBoost) using the 19 canonical features of the Universal Core Detector:
  - 11 Internal generation signals (token prob/entropy/perplexity/rank statistics)
  - 5 Self-consistency signals (exact match, pairwise cosine similarity statistics)
  - 3 NLI agreement signals (mean entailment, mean contradiction, NLI disagreement)

Enforces strict isolation of sequence length (num_tokens), metadata, and text payloads,
and uses identical stratified 80/20 train/test splitting (random_state=42) for
rigorous direct comparability against the 11-feature primary baseline.
"""

import argparse
import json
import logging
import os
import sys
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
    FORBIDDEN_FEATURE_COLUMNS,
    INTERNAL_SIGNAL_FEATURES,
    METADATA_COLUMNS,
    NLI_AGREEMENT_FEATURES,
    SELF_CONSISTENCY_FEATURES,
    UNIVERSAL_CORE_FEATURES,
)
from src.training.baseline_models import compute_ece

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("combined_models")

# 19 universal features for the core combined classifier
UNIVERSAL_MODEL_FEATURES: List[str] = list(UNIVERSAL_CORE_FEATURES)

# Forbidden columns that must NEVER be used in the model feature matrix
FORBIDDEN_COLUMNS: Set[str] = set(FORBIDDEN_FEATURE_COLUMNS)


def prepare_combined_feature_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract and validate the 19 universal features from the combined feature table.

    Verifies absence of forbidden columns, absence of NaNs/Infs, and isolates
    metadata (id, source_dataset, label) from the feature matrix X.

    Parameters:
        df: Input DataFrame containing universal features.

    Returns:
        Tuple of (X, y, metadata).
    """
    # Defensive copy
    df_work = df.copy()

    # Verify all 19 features are present
    missing_features = [f for f in UNIVERSAL_MODEL_FEATURES if f not in df_work.columns]
    if missing_features:
        raise KeyError(f"Required universal features missing from dataset: {missing_features}")

    # Verify no forbidden columns are accidentally in the feature list
    forbidden_in_features = set(UNIVERSAL_MODEL_FEATURES).intersection(FORBIDDEN_COLUMNS)
    if forbidden_in_features:
        raise ValueError(f"Forbidden columns detected in model feature list: {forbidden_in_features}")

    if "label" not in df_work.columns:
        raise KeyError("Target column 'label' missing from dataset.")

    X = df_work[UNIVERSAL_MODEL_FEATURES].copy()
    y = df_work["label"].astype(int).copy()

    # Verify finite values
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values (NaN, Inf, -Inf) detected in feature matrix X.")

    meta_cols = [c for c in ["id", "source_dataset", "label"] if c in df_work.columns]
    metadata = df_work[meta_cols].copy()

    return X, y, metadata


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified 80/20 train/test split preserving joint dataset and label distributions.

    Uses the exact composite stratification key `source_dataset + "_" + label`
    identical to the Phase 6 baseline pipeline, guaranteeing matching train/test partitions.

    Parameters:
        X: Feature matrix.
        y: Target label series.
        metadata: Metadata dataframe containing 'source_dataset'.
        test_size: Hold-out test fraction (default: 0.2).
        random_state: Seed for deterministic splitting (default: 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, meta_train, meta_test).
    """
    if "source_dataset" not in metadata.columns:
        raise KeyError("Metadata must contain 'source_dataset' column for joint stratification.")

    stratify_key = metadata["source_dataset"].astype(str) + "_" + y.astype(str)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_key,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        meta_train.reset_index(drop=True),
        meta_test.reset_index(drop=True),
    )


def build_logistic_regression_pipeline(
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with StandardScaler and LogisticRegression.

    Parameters:
        C: Inverse regularization parameter (default: 1.0).
        solver: Optimization algorithm (default: 'lbfgs').
        max_iter: Maximum iterations for convergence (default: 1000).
        random_state: Seed for reproducibility (default: 42).

    Returns:
        Pipeline with StandardScaler and LogisticRegression.
    """
    scaler = StandardScaler()
    clf = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    return Pipeline([("scaler", scaler), ("classifier", clf)])


def build_xgboost_model(
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0.1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBClassifier:
    """
    Construct an XGBClassifier with baseline hyperparameters.

    Parameters:
        n_estimators: Number of boosting rounds (default: 100).
        max_depth: Maximum tree depth (default: 4).
        learning_rate: Boosting learning rate (default: 0.05).
        subsample: Subsample ratio of training instances (default: 0.8).
        colsample_bytree: Subsample ratio of columns per tree (default: 0.8).
        gamma: Minimum loss reduction for partitioning (default: 0.1).
        random_state: Seed for reproducibility (default: 42).
        n_jobs: Worker threads (default: -1).

    Returns:
        Configured XGBClassifier.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Union[np.ndarray, pd.Series, List[float]],
    n_bins_ece: int = 10,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics on model predictions.

    Parameters:
        y_true: Binary ground-truth targets (0 or 1).
        y_pred: Binary hard predictions (0 or 1).
        y_prob: Predicted probabilities for class 1.
        n_bins_ece: Number of bins for Expected Calibration Error (default: 10).

    Returns:
        Dictionary of computed performance metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    y_pr = np.asarray(y_prob, dtype=float)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_t, y_pr))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, y_pr))
    except ValueError:
        pr_auc = 0.5

    brier = float(brier_score_loss(y_t, y_pr))
    ece = compute_ece(y_t, y_pr, n_bins=n_bins_ece)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_by_source_dataset(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metadata_test: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate test predictions broken down by source dataset.

    Parameters:
        y_true: Test-set true labels.
        y_pred: Test-set predictions.
        y_prob: Test-set predicted probabilities.
        metadata_test: Test-set metadata containing 'source_dataset'.

    Returns:
        Dictionary mapping dataset name to evaluation metrics and sample counts.
    """
    results_by_source = {}
    datasets = metadata_test["source_dataset"].unique()

    for ds in sorted(datasets):
        mask = (metadata_test["source_dataset"] == ds).to_numpy()
        ds_y_true = y_true[mask]
        ds_y_pred = y_pred[mask]
        ds_y_prob = y_prob[mask]

        ds_metrics = evaluate_predictions(ds_y_true, ds_y_pred, ds_y_prob)
        ds_metrics["test_samples"] = int(np.sum(mask))
        results_by_source[str(ds)] = ds_metrics

    return results_by_source


def compute_baseline_comparison(
    combined_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute absolute metric deltas: combined - baseline.

    Parameters:
        combined_metrics: Dictionary of combined model metrics.
        baseline_metrics: Dictionary of baseline model metrics.

    Returns:
        Dictionary of absolute deltas for Accuracy, F1, ROC-AUC, PR-AUC, Brier score, and ECE.
    """
    metrics_to_compare = ["accuracy", "f1", "roc_auc", "pr_auc", "brier_score", "ece"]
    deltas = {}
    for m in metrics_to_compare:
        if m in combined_metrics and m in baseline_metrics:
            deltas[f"{m}_delta"] = float(combined_metrics[m] - baseline_metrics[m])
    return deltas


def run_combined_experiment(
    data_path: Union[str, Path, pd.DataFrame] = "experiments/baselines/combined/universal_features.parquet",
    baseline_summary_path: Optional[Union[str, Path]] = "experiments/baselines/results/baseline_results_summary.json",
    output_dir: Union[str, Path] = "experiments/baselines/combined",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute the full supervised training and evaluation for the Universal Core Detector.

    Parameters:
        data_path: Path to universal features parquet/csv or DataFrame.
        baseline_summary_path: Path to baseline results summary JSON for delta comparison.
        output_dir: Directory to save results and prediction files.
        test_size: Stratified test fraction (default: 0.2).
        random_state: Reproducibility seed (default: 42).

    Returns:
        Dictionary containing all evaluation results, subgroup metrics, and baseline comparisons.
    """
    if isinstance(data_path, pd.DataFrame):
        df = data_path.copy()
    else:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Universal feature dataset not found at {path}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

    logger.info(f"Loaded universal feature table: {len(df)} rows, {len(df.columns)} columns.")

    X, y, meta = prepare_combined_feature_dataframe(df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
        X, y, meta, test_size=test_size, random_state=random_state
    )

    logger.info(f"Stratified split completed: Train={len(X_train)} samples, Test={len(X_test)} samples.")
    logger.info(f"Train class balance: {dict(y_train.value_counts())}")
    logger.info(f"Test class balance:  {dict(y_test.value_counts())}")
    logger.info(f"Using exactly {len(X.columns)} features: {list(X.columns)}")

    # 1. Train Logistic Regression Pipeline (StandardScaler + LogisticRegression)
    logger.info("Training Logistic Regression pipeline (StandardScaler + LogisticRegression)...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=random_state)
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob)
    lr_subgroups = evaluate_by_source_dataset(y_test, lr_pred, lr_prob, meta_test)

    # 2. Train XGBoost Model (XGBClassifier)
    logger.info("Training XGBoost baseline configuration (XGBClassifier)...")
    xgb_model = build_xgboost_model(random_state=random_state)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_prob)
    xgb_subgroups = evaluate_by_source_dataset(y_test, xgb_pred, xgb_prob, meta_test)

    # 3. Load baseline results for direct delta calculation if available
    baseline_data = None
    baseline_lr_deltas = {}
    baseline_xgb_deltas = {}
    subgroup_lr_deltas = {}
    subgroup_xgb_deltas = {}

    if baseline_summary_path is not None and Path(baseline_summary_path).exists():
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        base_overall = baseline_data.get("overall_metrics", {})
        base_lr = base_overall.get("logistic_regression", {})
        base_xgb = base_overall.get("xgboost", {})

        baseline_lr_deltas = compute_baseline_comparison(lr_metrics, base_lr)
        baseline_xgb_deltas = compute_baseline_comparison(xgb_metrics, base_xgb)

        base_subgroups = baseline_data.get("subgroup_breakdown_by_source", {})
        for ds in sorted(lr_subgroups.keys()):
            if ds in base_subgroups:
                ds_base_lr = base_subgroups[ds].get("logistic_regression", {})
                ds_base_xgb = base_subgroups[ds].get("xgboost", {})
                subgroup_lr_deltas[ds] = compute_baseline_comparison(lr_subgroups[ds], ds_base_lr)
                subgroup_xgb_deltas[ds] = compute_baseline_comparison(xgb_subgroups[ds], ds_base_xgb)

    # Assemble comprehensive results dictionary
    results = {
        "experiment_metadata": {
            "dataset": str(data_path),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_size_fraction": test_size,
            "random_state": random_state,
            "num_features": len(X.columns),
            "features_used": list(X.columns),
            "feature_families": {
                "internal_signals": list(INTERNAL_SIGNAL_FEATURES),
                "self_consistency": list(SELF_CONSISTENCY_FEATURES),
                "nli_agreement": list(NLI_AGREEMENT_FEATURES),
            },
            "train_label_distribution": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "test_label_distribution": {str(k): int(v) for k, v in y_test.value_counts().items()},
            "test_dataset_distribution": {str(k): int(v) for k, v in meta_test["source_dataset"].value_counts().items()},
        },
        "overall_metrics": {
            "logistic_regression": lr_metrics,
            "xgboost": xgb_metrics,
        },
        "subgroup_breakdown_by_source": {
            "logistic_regression": lr_subgroups,
            "xgboost": xgb_subgroups,
        },
        "baseline_comparison_deltas": {
            "logistic_regression": baseline_lr_deltas,
            "xgboost": baseline_xgb_deltas,
            "subgroup_deltas": {
                "logistic_regression": subgroup_lr_deltas,
                "xgboost": subgroup_xgb_deltas,
            },
        },
    }

    # Save output artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predictions table
    predictions_df = pd.DataFrame({
        "id": meta_test["id"].to_numpy(),
        "source_dataset": meta_test["source_dataset"].to_numpy(),
        "label": y_test.to_numpy(),
        "lr_probability": lr_prob,
        "lr_prediction": lr_pred,
        "xgb_probability": xgb_prob,
        "xgb_prediction": xgb_pred,
        # Canonical aliases pointing to primary model (XGBoost)
        "y_probability": xgb_prob,
        "y_prediction": xgb_pred,
    })

    pred_parquet = out_dir / "combined_test_predictions.parquet"
    pred_csv = out_dir / "combined_test_predictions.csv"
    predictions_df.to_parquet(pred_parquet, index=False)
    predictions_df.to_csv(pred_csv, index=False)
    logger.info(f"Saved test predictions to {pred_parquet} and {pred_csv}")

    # 2. JSON summary
    results_json = out_dir / "combined_model_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved experiment results to {results_json}")

    # 3. CSV summary table
    summary_rows = []
    # Overall rows
    for model_name, metrics, deltas in [
        ("logistic_regression", lr_metrics, baseline_lr_deltas),
        ("xgboost", xgb_metrics, baseline_xgb_deltas),
    ]:
        row = {
            "model": model_name,
            "subset": "overall",
            "samples": len(y_test),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier_score": metrics["brier_score"],
            "ece": metrics["ece"],
            "tp": metrics["confusion_matrix"]["tp"],
            "fp": metrics["confusion_matrix"]["fp"],
            "tn": metrics["confusion_matrix"]["tn"],
            "fn": metrics["confusion_matrix"]["fn"],
            "accuracy_delta": deltas.get("accuracy_delta", np.nan),
            "f1_delta": deltas.get("f1_delta", np.nan),
            "roc_auc_delta": deltas.get("roc_auc_delta", np.nan),
            "pr_auc_delta": deltas.get("pr_auc_delta", np.nan),
            "brier_score_delta": deltas.get("brier_score_delta", np.nan),
            "ece_delta": deltas.get("ece_delta", np.nan),
        }
        summary_rows.append(row)

    # Subgroup rows
    for model_name, subgroups, sub_deltas in [
        ("logistic_regression", lr_subgroups, subgroup_lr_deltas),
        ("xgboost", xgb_subgroups, subgroup_xgb_deltas),
    ]:
        for ds, m in subgroups.items():
            d = sub_deltas.get(ds, {})
            row = {
                "model": model_name,
                "subset": ds,
                "samples": m["test_samples"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "brier_score": m["brier_score"],
                "ece": m["ece"],
                "tp": m["confusion_matrix"]["tp"],
                "fp": m["confusion_matrix"]["fp"],
                "tn": m["confusion_matrix"]["tn"],
                "fn": m["confusion_matrix"]["fn"],
                "accuracy_delta": d.get("accuracy_delta", np.nan),
                "f1_delta": d.get("f1_delta", np.nan),
                "roc_auc_delta": d.get("roc_auc_delta", np.nan),
                "pr_auc_delta": d.get("pr_auc_delta", np.nan),
                "brier_score_delta": d.get("brier_score_delta", np.nan),
                "ece_delta": d.get("ece_delta", np.nan),
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    results_csv = out_dir / "combined_model_results.csv"
    summary_df.to_csv(results_csv, index=False)
    logger.info(f"Saved metric summary table to {results_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Universal Combined Hallucination Detector Training (Phase 14)")
    parser.add_argument(
        "--data",
        type=str,
        default="experiments/baselines/combined/universal_features.parquet",
        help="Path to universal features parquet dataset",
    )
    parser.add_argument(
        "--baseline-summary",
        type=str,
        default="experiments/baselines/results/baseline_results_summary.json",
        help="Path to Phase 6 baseline summary JSON for delta comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baselines/combined",
        help="Output directory for results and predictions",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Stratified test split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and training (default: 42)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 14: UNIVERSAL COMBINED HALLUCINATION DETECTOR TRAINING")
    logger.info("=" * 60)

    results = run_combined_experiment(
        data_path=args.data,
        baseline_summary_path=args.baseline_summary,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Print summary to console
    logger.info("=" * 60)
    logger.info("OVERALL COMBINED MODEL EVALUATION")
    logger.info("=" * 60)
    for model_name in ["logistic_regression", "xgboost"]:
        metrics = results["overall_metrics"][model_name]
        deltas = results["baseline_comparison_deltas"][model_name]
        logger.info(f"\n[{model_name.upper()}]")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}  (Delta: {deltas.get('accuracy_delta', 0.0):+.4f})")
        logger.info(f"  Precision:   {metrics['precision']:.4f}")
        logger.info(f"  Recall:      {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:    {metrics['f1']:.4f}  (Delta: {deltas.get('f1_delta', 0.0):+.4f})")
        logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}  (Delta: {deltas.get('roc_auc_delta', 0.0):+.4f})")
        logger.info(f"  PR-AUC:      {metrics['pr_auc']:.4f}  (Delta: {deltas.get('pr_auc_delta', 0.0):+.4f})")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}  (Delta: {deltas.get('brier_score_delta', 0.0):+.4f})")
        logger.info(f"  ECE:         {metrics['ece']:.4f}  (Delta: {deltas.get('ece_delta', 0.0):+.4f})")
        cm = metrics["confusion_matrix"]
        logger.info(f"  Confusion:   TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")


if __name__ == "__main__":
    main()
