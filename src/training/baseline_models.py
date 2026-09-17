"""
Baseline Classification Pipeline for LLM Hallucination Detection.

Trains and evaluates baseline classifiers (Logistic Regression and XGBoost) using
internal generation signals (token probabilities, predictive entropy, perplexity,
and rank dispersion statistics) extracted from Qwen3.5-0.8B.

Strictly excludes sequence length (num_tokens) from the primary baseline to prevent
length-shortcut learning, and stratifies across both benchmark origin and class label.
"""

import argparse
import logging
import math
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("baseline_models")

# 11 strictly internal signal features for primary baseline
PRIMARY_SIGNAL_FEATURES: List[str] = [
    "min_log_prob",
    "mean_log_prob",
    "mean_token_prob",
    "token_prob_std",
    "mean_entropy",
    "max_entropy",
    "entropy_std",
    "log_perplexity",
    "log_mean_token_rank",
    "log_max_token_rank",
    "log_rank_std",
]

# Forbidden columns that must NEVER be passed as features to the primary baseline
FORBIDDEN_COLUMNS: Set[str] = {
    "id",
    "source_dataset",
    "label",
    "prompt",
    "context",
    "response",
    "model_input",
    "forward_time_s",
    "num_tokens",
}


def prepare_feature_dataframe(
    df: pd.DataFrame,
    include_num_tokens: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract and transform internal signal features from the raw supervised dataset.

    Applies variance-stabilizing log transformations to heavy-tailed features:
      - log_perplexity = log(perplexity)
      - log_mean_token_rank = log1p(mean_token_rank)
      - log_max_token_rank = log1p(max_token_rank)
      - log_rank_std = log1p(rank_std)

    Parameters:
        df: Input supervised signal dataset.
        include_num_tokens: If True, adds num_tokens (for ablation studies only;
            MUST remain False for the primary baseline).

    Returns:
        Tuple of (X_features, y_labels, metadata_df).
    """
    # Defensive copy to avoid mutating source dataset
    df_work = df.copy()

    # Log transformations for heavy-tailed metrics
    df_work["log_perplexity"] = np.log(np.maximum(df_work["perplexity"].astype(float), 1e-12))
    df_work["log_mean_token_rank"] = np.log1p(np.maximum(df_work["mean_token_rank"].astype(float), 0.0))
    df_work["log_max_token_rank"] = np.log1p(np.maximum(df_work["max_token_rank"].astype(float), 0.0))
    df_work["log_rank_std"] = np.log1p(np.maximum(df_work["rank_std"].astype(float), 0.0))

    feature_cols = list(PRIMARY_SIGNAL_FEATURES)
    if include_num_tokens:
        feature_cols.append("num_tokens")
    else:
        # Strict enforcement: num_tokens MUST NOT be in feature columns
        assert "num_tokens" not in feature_cols, "Policy violation: num_tokens found in primary baseline!"

    # Verification of feature selection integrity
    forbidden_intersect = set(feature_cols).intersection(FORBIDDEN_COLUMNS - ({"num_tokens"} if include_num_tokens else set()))
    if forbidden_intersect:
        raise ValueError(f"Forbidden columns detected in model feature set: {forbidden_intersect}")

    missing_cols = [c for c in feature_cols if c not in df_work.columns]
    if missing_cols:
        raise KeyError(f"Required signal feature columns missing from dataset: {missing_cols}")

    X = df_work[feature_cols].copy()
    y = df_work["label"].astype(int).copy()

    meta_cols = [c for c in ["id", "source_dataset", "prompt", "response", "num_tokens"] if c in df_work.columns]
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

    Combines source_dataset and label into a composite stratification key
    (e.g., 'halueval_0', 'truthfulqa_1', 'fever_0') so that training and test
    partitions have exactly matched task and class balances.

    Parameters:
        X: Feature matrix.
        y: Target label series.
        metadata: Metadata dataframe containing 'source_dataset'.
        test_size: Fraction for hold-out test set (default: 0.2).
        random_state: Seed for deterministic splitting.

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
    penalty: Optional[str] = None,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
    class_weight: Optional[Union[str, dict]] = None,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with StandardScaler and LogisticRegression.

    Ensures that StandardScaler is fit ONLY on training data during cross-validation
    or train/test evaluation, eliminating data leakage from feature centering.

    Parameters:
        C: Inverse regularization strength.
        penalty: Regularization norm (optional, default uses standard L2 without warning).
        solver: Optimization algorithm.
        max_iter: Maximum optimization iterations.
        random_state: Reproducibility seed.
        class_weight: Class reweighting scheme.

    Returns:
        sklearn.pipeline.Pipeline
    """
    scaler = StandardScaler()
    lr_kwargs: Dict[str, Any] = {
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": random_state,
        "class_weight": class_weight,
    }
    if penalty is not None:
        lr_kwargs["penalty"] = penalty

    clf = LogisticRegression(**lr_kwargs)
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
    Construct an XGBClassifier baseline model.

    Tree-based models operate directly on numerical signal features without
    requiring feature standardization.

    Parameters:
        n_estimators: Number of gradient boosted trees.
        max_depth: Maximum tree depth for base learners.
        learning_rate: Boosting learning rate (eta).
        subsample: Subsample ratio of the training instances.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        gamma: Minimum loss reduction required to make a further partition.
        random_state: Seed for deterministic execution.
        n_jobs: Number of parallel CPU threads (-1 uses all cores).

    Returns:
        XGBClassifier instance.
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


def compute_ece(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Union[np.ndarray, pd.Series, List[float]],
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) using transparent uniform binning.

    Partitions predicted confidence probabilities p in [0, 1] into M equal-width
    bins B_1, ..., B_M:
      acc(B_m) = (1 / |B_m|) * sum_{i in B_m} I(y_i == 1)
      conf(B_m) = (1 / |B_m|) * sum_{i in B_m} p_i
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Parameters:
        y_true: Ground-truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (1).
        n_bins: Number of probability intervals (default: 10).

    Returns:
        float: Expected Calibration Error value in [0, 1].
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("Dimensions of y_true and y_prob must match.")

    n = len(y_true_arr)
    if n == 0:
        return 0.0

    # Probability bounds clipping for numerical safety
    y_prob_clamped = np.clip(y_prob_arr, 0.0, 1.0)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob_clamped >= lower) & (y_prob_clamped <= upper)
        else:
            in_bin = (y_prob_clamped >= lower) & (y_prob_clamped < upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true_arr[in_bin])
            bin_conf = np.mean(y_prob_clamped[in_bin])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Union[np.ndarray, pd.Series, List[float]],
    n_bins_ece: int = 10,
) -> Dict[str, Any]:
    """
    Compute full evaluation metric suite on classifier predictions.

    Calculates:
      - Accuracy
      - Precision
      - Recall
      - F1 Score
      - ROC-AUC
      - PR-AUC (Average Precision)
      - Brier Score
      - Expected Calibration Error (ECE)
      - Confusion Matrix (TP, FP, TN, FN)

    Parameters:
        y_true: Ground-truth binary labels.
        y_pred: Hard class predictions (0 or 1).
        y_prob: Soft probability estimates for class 1.
        n_bins_ece: Bins for calibration error.

    Returns:
        Dict[str, Any]: Formatted performance metrics.
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


def run_baseline_experiment(
    data_path: Union[str, Path, pd.DataFrame] = "experiments/baselines/supervised_signals_combined.parquet",
    output_dir: Optional[Union[str, Path]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    include_num_tokens: bool = False,
) -> Dict[str, Any]:
    """
    Run the end-to-end baseline modeling pipeline.

    Loads dataset, applies log transformations, enforces strict feature policies,
    executes stratified split, trains Logistic Regression and XGBoost baselines,
    and evaluates both models.

    Parameters:
        data_path: Path to Parquet/CSV file, or an existing DataFrame.
        output_dir: Optional directory to store results and metrics JSON.
        test_size: Fraction of samples reserved for test set.
        random_state: Seed for splitting and training.
        include_num_tokens: If True, include num_tokens (ablation only; default False).

    Returns:
        Dict with evaluation metrics for 'logistic_regression' and 'xgboost'.
    """
    import json

    if isinstance(data_path, pd.DataFrame):
        df = data_path.copy()
    else:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

    logger.info(f"Loaded dataset with {len(df)} rows.")

    X, y, meta = prepare_feature_dataframe(df, include_num_tokens=include_num_tokens)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
        X, y, meta, test_size=test_size, random_state=random_state
    )

    logger.info(f"Stratified split: Train={len(X_train)} samples, Test={len(X_test)} samples.")
    logger.info(f"Features used ({len(X.columns)}): {list(X.columns)}")

    # 1. Logistic Regression
    logger.info("Training Logistic Regression pipeline (StandardScaler + LogisticRegression)...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=random_state)
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob)

    # 2. XGBoost
    logger.info("Training XGBoost baseline (XGBClassifier)...")
    xgb_model = build_xgboost_model(random_state=random_state)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_prob)

    results = {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "split_summary": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_features": len(X.columns),
            "features": list(X.columns),
            "include_num_tokens": include_num_tokens,
            "random_state": random_state,
            "test_size_fraction": test_size,
        },
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_file = out_path / "baseline_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved baseline metrics to {metrics_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline Classification Pipeline (Phase 6)")
    parser.add_argument(
        "--data",
        type=str,
        default="experiments/baselines/supervised_signals_combined.parquet",
        help="Path to supervised signals parquet/csv dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baselines/results",
        help="Directory to save baseline evaluation metrics",
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
        help="Random seed for splitting and modeling (default: 42)",
    )
    parser.add_argument(
        "--include-num-tokens",
        action="store_true",
        help="Include num_tokens (ablation only; forbidden in primary baseline)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 6: BASELINE CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Data path: {args.data}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Include num_tokens: {args.include_num_tokens}")

    results = run_baseline_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        include_num_tokens=args.include_num_tokens,
    )

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for model_name in ["logistic_regression", "xgboost"]:
        metrics = results[model_name]
        logger.info(f"\n[{model_name.upper()}]")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:   {metrics['precision']:.4f}")
        logger.info(f"  Recall:      {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:    {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"  ECE:         {metrics['ece']:.4f}")
        cm = metrics["confusion_matrix"]
        logger.info(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")


if __name__ == "__main__":
    main()

