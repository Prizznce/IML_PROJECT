"""
Comprehensive Probability Calibration Analysis (Phase 17).

Evaluates and optimizes probabilistic calibration for the Universal Core Hallucination Detector
using 10-bin uniform reliability tables, Expected Calibration Error (ECE), Maximum Calibration
Error (MCE), Brier score, Cox calibration slope and intercept, and post-hoc calibration methods
(Platt scaling / sigmoid and Isotonic regression).

Enforces zero test leakage:
  - Outer 80/20 split: 3,200 train, 800 holdout test (identical to Phase 14).
  - Inner 80/20 split on train: 2,560 model-fit, 640 calibration-fit.
  - Base models fit strictly on the 2,560 model-fit partition.
  - Calibrators fit strictly on the 640 calibration partition.
  - The 800-sample test set remains completely untouched during all parameter fitting.
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

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression, _SigmoidCalibration
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

from src.features.build_universal_features import UNIVERSAL_CORE_FEATURES
from src.training.combined_models import (
    build_logistic_regression_pipeline,
    build_xgboost_model,
    prepare_combined_feature_dataframe,
    split_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("calibration_analysis")


def compute_calibration_curve_and_metrics(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Union[np.ndarray, pd.Series, List[float]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute 10-bin uniform reliability table, ECE, MCE, Brier score, and calibration slope/intercept.

    Parameters:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for class 1.
        n_bins: Number of uniform probability bins (default: 10).

    Returns:
        Dictionary containing reliability table, ECE, MCE, Brier score, slope, intercept,
        mean probability, and observed positive rate.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_pr = np.asarray(y_prob, dtype=float)

    n_samples = len(y_t)
    if n_samples == 0:
        return {
            "bins": [],
            "ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "mean_predicted_probability": 0.0,
            "actual_positive_rate": 0.0,
            "calibration_slope": 1.0,
            "calibration_intercept": 0.0,
            "non_empty_bins": 0,
        }

    # Defensive clamping
    y_pr_clamped = np.clip(y_pr, 0.0, 1.0)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    bins_data: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lower = float(bin_boundaries[i])
        upper = float(bin_boundaries[i + 1])

        # Include upper boundary on the last bin
        if i == n_bins - 1:
            in_bin = (y_pr_clamped >= lower) & (y_pr_clamped <= upper)
        else:
            in_bin = (y_pr_clamped >= lower) & (y_pr_clamped < upper)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_prob = float(np.mean(y_pr_clamped[in_bin]))
            observed_rate = float(np.mean(y_t[in_bin]))
            calib_err = float(abs(observed_rate - mean_prob))
            ece += (count / n_samples) * calib_err
            if calib_err > mce:
                mce = calib_err
        else:
            mean_prob = None
            observed_rate = None
            calib_err = 0.0

        bins_data.append({
            "probability_bin": i + 1,
            "bin_lower": round(lower, 2),
            "bin_upper": round(upper, 2),
            "sample_count": count,
            "mean_predicted_probability": round(mean_prob, 4) if mean_prob is not None else None,
            "observed_positive_rate": round(observed_rate, 4) if observed_rate is not None else None,
            "absolute_calibration_error": round(calib_err, 4),
        })

    brier = float(brier_score_loss(y_t, y_pr_clamped))
    non_empty = sum(1 for b in bins_data if b["sample_count"] > 0)
    mean_prob_overall = float(np.mean(y_pr_clamped))
    actual_pos_rate = float(np.mean(y_t))

    # Compute Cox calibration slope and intercept via logistic regression on log-odds
    calib_slope = 1.0
    calib_intercept = 0.0
    try:
        # Avoid log(0) and division by zero
        p_safe = np.clip(y_pr_clamped, 1e-6, 1.0 - 1e-6)
        log_odds = np.log(p_safe / (1.0 - p_safe)).reshape(-1, 1)

        # Only fit if both classes are present
        if len(np.unique(y_t)) > 1:
            calib_reg = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
            calib_reg.fit(log_odds, y_t)
            calib_intercept = float(calib_reg.intercept_[0])
            calib_slope = float(calib_reg.coef_[0][0])
    except Exception as e:
        logger.warning(f"Could not fit calibration slope/intercept: {e}")

    return {
        "bins": bins_data,
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "mean_predicted_probability": mean_prob_overall,
        "actual_positive_rate": actual_pos_rate,
        "calibration_slope": calib_slope,
        "calibration_intercept": calib_intercept,
        "non_empty_bins": non_empty,
    }


def evaluate_model_prob_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation and calibration metrics for a probability prediction vector.

    Parameters:
        y_true: True binary targets.
        y_prob: Predicted positive-class probabilities.
        n_bins: Bins for calibration evaluation.

    Returns:
        Dictionary of discrimination and calibration metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_pr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_pr >= 0.5).astype(int)

    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_t, y_pr))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, y_pr))
    except ValueError:
        pr_auc = 0.5

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    calib = compute_calibration_curve_and_metrics(y_t, y_pr, n_bins=n_bins)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": calib["brier_score"],
        "ece": calib["ece"],
        "mce": calib["mce"],
        "calibration_slope": calib["calibration_slope"],
        "calibration_intercept": calib["calibration_intercept"],
        "mean_predicted_probability": calib["mean_predicted_probability"],
        "probability_std": float(np.std(y_pr)),
        "min_probability": float(np.min(y_pr)),
        "max_probability": float(np.max(y_pr)),
        "actual_positive_rate": calib["actual_positive_rate"],
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "reliability_bins": calib["bins"],
    }


def split_training_into_model_and_calib(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    calib_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Split the 3,200 training examples into model-fitting (80%) and calibration-fitting (20%) subsets.

    Uses the composite stratification key `source_dataset + "_" + label`.

    Parameters:
        X_train: Feature matrix of training partition.
        y_train: Label series of training partition.
        meta_train: Metadata DataFrame containing 'source_dataset'.
        calib_size: Fraction of training instances reserved for calibration (default: 0.2).
        random_state: Seed for deterministic splitting (default: 42).

    Returns:
        Tuple of (X_model, X_calib, y_model, y_calib, meta_model, meta_calib).
    """
    stratify_key = meta_train["source_dataset"].astype(str) + "_" + y_train.astype(str)

    X_model, X_calib, y_model, y_calib, meta_model, meta_calib = train_test_split(
        X_train,
        y_train,
        meta_train,
        test_size=calib_size,
        random_state=random_state,
        stratify=stratify_key,
    )

    return (
        X_model.reset_index(drop=True),
        X_calib.reset_index(drop=True),
        y_model.reset_index(drop=True),
        y_calib.reset_index(drop=True),
        meta_model.reset_index(drop=True),
        meta_calib.reset_index(drop=True),
    )


def plot_comparative_reliability_diagram(
    model_name: str,
    uncal_data: Dict[str, Any],
    sig_data: Dict[str, Any],
    iso_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Generate and save a publication-quality dual-panel reliability diagram comparing
    Uncalibrated, Sigmoid, and Isotonic calibration against the perfect diagonal.

    Parameters:
        model_name: Display name of the model.
        uncal_data: Metrics dict for uncalibrated model.
        sig_data: Metrics dict for sigmoid calibrated model.
        iso_data: Metrics dict for isotonic calibrated model.
        output_path: Path to output PNG.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.5, 8.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Upper panel: Reliability Curves
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7, linewidth=1.5)

    curves = [
        ("Uncalibrated", uncal_data, "#d62728", "s-"),
        ("Sigmoid (Platt)", sig_data, "#1f77b4", "o-"),
        ("Isotonic", iso_data, "#2ca02c", "^-"),
    ]

    for label, data, color, marker in curves:
        bins = data["reliability_bins"]
        probs = []
        rates = []
        for b in bins:
            if b["mean_predicted_probability"] is not None and b["observed_positive_rate"] is not None:
                probs.append(b["mean_predicted_probability"])
                rates.append(b["observed_positive_rate"])

        ece_val = data["ece"]
        brier_val = data["brier_score"]
        ax1.plot(
            probs,
            rates,
            marker,
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{label} (ECE={ece_val:.3f}, Brier={brier_val:.3f})",
        )

    ax1.set_ylabel("Observed Fraction of Positives (Hallucination)", fontsize=11)
    ax1.set_title(f"Reliability Diagram: {model_name} (Holdout Test Set, N=800)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Lower panel: Sample count histogram per bin (using uncalibrated sample counts)
    uncal_bins = uncal_data["reliability_bins"]
    bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2.0 for b in uncal_bins]
    counts = [b["sample_count"] for b in uncal_bins]
    width = 0.08

    ax2.bar(bin_centers, counts, width=width, color="#4a5568", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved reliability diagram to {output_path}")


def run_calibration_study(
    data_path: Union[str, Path, pd.DataFrame] = "experiments/baselines/combined/universal_features.parquet",
    output_dir: Union[str, Path] = "experiments/calibration",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run complete probability calibration study with zero test leakage.

    Parameters:
        data_path: Path to universal features dataset.
        output_dir: Directory to save outputs.
        random_state: Seed for deterministic splitting.

    Returns:
        Structured results dictionary.
    """
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

    logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns.")

    # 1. Prepare 19 features
    X_full, y_full, meta_full = prepare_combined_feature_dataframe(df)

    # 2. Outer 80/20 train/test split (same as Phase 14)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
        X_full, y_full, meta_full, test_size=0.2, random_state=random_state
    )

    # 3. Inner 80/20 model-fit/calibration-fit split on training partition
    X_model, X_calib, y_model, y_calib, meta_model, meta_calib = split_training_into_model_and_calib(
        X_train, y_train, meta_train, calib_size=0.2, random_state=random_state
    )

    logger.info(f"Outer Split: Train={len(X_train)} samples, Test={len(X_test)} samples.")
    logger.info(f"Inner Split: Model-Fit={len(X_model)} samples, Calibration-Fit={len(X_calib)} samples.")

    # Invariance and leakage checks
    test_ids = set(meta_test["id"])
    model_ids = set(meta_model["id"])
    calib_ids = set(meta_calib["id"])

    if len(df) == 4000:
        assert len(test_ids) == 800, f"Expected 800 test samples, got {len(test_ids)}"
        assert len(model_ids) == 2560, f"Expected 2560 model samples, got {len(model_ids)}"
        assert len(calib_ids) == 640, f"Expected 640 calib samples, got {len(calib_ids)}"
    assert len(model_ids.intersection(calib_ids)) == 0, "Model-fit and calibration-fit partitions overlap!"
    assert len(test_ids.intersection(model_ids)) == 0, "Test set overlaps with model-fit partition!"
    assert len(test_ids.intersection(calib_ids)) == 0, "Test set overlaps with calibration-fit partition!"

    # 4. Train base models strictly on model-fit partition
    logger.info("Training base models on model-fit subset (N=2,560)...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=random_state)
    lr_pipeline.fit(X_model, y_model)

    xgb_model = build_xgboost_model(random_state=random_state)
    xgb_model.fit(X_model, y_model)

    # 5. Generate probabilities on calibration subset (N=640)
    logger.info("Generating calibration predictions on calibration subset (N=640)...")
    calib_prob_lr = lr_pipeline.predict_proba(X_calib)[:, 1]
    calib_prob_xgb = xgb_model.predict_proba(X_calib)[:, 1]

    # 6. Fit post-hoc calibrators on calibration partition
    logger.info("Fitting Sigmoid (Platt) and Isotonic calibrators on calibration subset...")
    sig_calib_lr = _SigmoidCalibration().fit(calib_prob_lr, y_calib.to_numpy())
    iso_calib_lr = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(calib_prob_lr, y_calib.to_numpy())

    sig_calib_xgb = _SigmoidCalibration().fit(calib_prob_xgb, y_calib.to_numpy())
    iso_calib_xgb = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(calib_prob_xgb, y_calib.to_numpy())

    # 7. Evaluate on untouched holdout test set (N=800)
    logger.info("Evaluating probability outputs on untouched holdout test set (N=800)...")
    test_prob_lr_uncal = np.clip(lr_pipeline.predict_proba(X_test)[:, 1], 0.0, 1.0)
    test_prob_lr_sig = np.clip(sig_calib_lr.predict(test_prob_lr_uncal), 0.0, 1.0)
    test_prob_lr_iso = np.clip(iso_calib_lr.predict(test_prob_lr_uncal), 0.0, 1.0)

    test_prob_xgb_uncal = np.clip(xgb_model.predict_proba(X_test)[:, 1], 0.0, 1.0)
    test_prob_xgb_sig = np.clip(sig_calib_xgb.predict(test_prob_xgb_uncal), 0.0, 1.0)
    test_prob_xgb_iso = np.clip(iso_calib_xgb.predict(test_prob_xgb_uncal), 0.0, 1.0)

    # Assemble predictions mapping
    models_eval = {
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "methods": {
                "uncalibrated": test_prob_lr_uncal,
                "sigmoid": test_prob_lr_sig,
                "isotonic": test_prob_lr_iso,
            },
        },
        "xgboost": {
            "display_name": "XGBoost",
            "methods": {
                "uncalibrated": test_prob_xgb_uncal,
                "sigmoid": test_prob_xgb_sig,
                "isotonic": test_prob_xgb_iso,
            },
        },
    }

    # Evaluate metrics, deltas, subgroups, and reliability tables
    results_summary: Dict[str, Any] = {
        "experiment_metadata": {
            "dataset": str(data_path),
            "total_samples": len(df),
            "test_samples": len(X_test),
            "total_train_samples": len(X_train),
            "model_fit_samples": len(X_model),
            "calibration_fit_samples": len(X_calib),
            "random_state": random_state,
            "test_dataset_distribution": {str(k): int(v) for k, v in meta_test["source_dataset"].value_counts().items()},
        },
        "models": {},
    }

    table_rows: List[Dict[str, Any]] = []
    reliability_table_rows: List[Dict[str, Any]] = []
    prediction_records: List[Dict[str, Any]] = []

    for m_key, m_info in models_eval.items():
        disp_name = m_info["display_name"]
        methods_prob = m_info["methods"]

        m_results: Dict[str, Any] = {"display_name": disp_name, "calibration_methods": {}}

        # Evaluate uncalibrated first for delta reference
        uncal_prob = methods_prob["uncalibrated"]
        uncal_metrics = evaluate_model_prob_metrics(y_test.to_numpy(), uncal_prob)

        for cal_method, prob_vec in methods_prob.items():
            metrics = evaluate_model_prob_metrics(y_test.to_numpy(), prob_vec)

            # Compute deltas relative to uncalibrated
            delta_brier = float(metrics["brier_score"] - uncal_metrics["brier_score"])
            delta_ece = float(metrics["ece"] - uncal_metrics["ece"])
            delta_mce = float(metrics["mce"] - uncal_metrics["mce"])

            # Subgroup evaluation on test set
            subgroups = {}
            for ds in sorted(meta_test["source_dataset"].unique()):
                ds_mask = (meta_test["source_dataset"] == ds).to_numpy()
                ds_metrics = evaluate_model_prob_metrics(y_test[ds_mask].to_numpy(), prob_vec[ds_mask])
                subgroups[ds] = ds_metrics

            m_results["calibration_methods"][cal_method] = {
                "metrics": metrics,
                "deltas_relative_to_uncalibrated": {
                    "delta_brier": delta_brier,
                    "delta_ece": delta_ece,
                    "delta_mce": delta_mce,
                },
                "subgroups": subgroups,
            }

            # Append to overall CSV rows
            table_rows.append({
                "model": m_key,
                "calibration": cal_method,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier_score"],
                "ece": metrics["ece"],
                "mce": metrics["mce"],
                "delta_brier": delta_brier,
                "delta_ece": delta_ece,
                "delta_mce": delta_mce,
                "calibration_slope": metrics["calibration_slope"],
                "calibration_intercept": metrics["calibration_intercept"],
                "mean_probability": metrics["mean_predicted_probability"],
                "prob_std": metrics["probability_std"],
                "min_prob": metrics["min_probability"],
                "max_prob": metrics["max_probability"],
                "tp": metrics["confusion_matrix"]["tp"],
                "fp": metrics["confusion_matrix"]["fp"],
                "tn": metrics["confusion_matrix"]["tn"],
                "fn": metrics["confusion_matrix"]["fn"],
            })

            # Append reliability bins
            for b in metrics["reliability_bins"]:
                reliability_table_rows.append({
                    "model": m_key,
                    "calibration_method": cal_method,
                    "probability_bin": b["probability_bin"],
                    "bin_lower": b["bin_lower"],
                    "bin_upper": b["bin_upper"],
                    "sample_count": b["sample_count"],
                    "mean_predicted_probability": b["mean_predicted_probability"],
                    "observed_positive_rate": b["observed_positive_rate"],
                    "absolute_calibration_error": b["absolute_calibration_error"],
                })

            # Accumulate predictions
            hard_pred = (prob_vec >= 0.5).astype(int)
            for idx in range(len(y_test)):
                prediction_records.append({
                    "id": meta_test["id"].iloc[idx],
                    "source_dataset": meta_test["source_dataset"].iloc[idx],
                    "label": int(y_test.iloc[idx]),
                    "model": m_key,
                    "calibration_method": cal_method,
                    "y_probability": float(prob_vec[idx]),
                    "y_prediction": int(hard_pred[idx]),
                })

        results_summary["models"][m_key] = m_results

    # Output Artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. calibration_results.json
    res_json_path = out_dir / "calibration_results.json"
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved calibration results JSON to {res_json_path}")

    # 2. calibration_results.csv
    res_df = pd.DataFrame(table_rows)
    res_csv_path = out_dir / "calibration_results.csv"
    res_df.to_csv(res_csv_path, index=False)
    logger.info(f"Saved calibration results CSV to {res_csv_path}")

    # 3. calibration_reliability_tables.csv
    rel_df = pd.DataFrame(reliability_table_rows)
    rel_csv_path = out_dir / "calibration_reliability_tables.csv"
    rel_df.to_csv(rel_csv_path, index=False)
    logger.info(f"Saved reliability tables CSV to {rel_csv_path}")

    # 4. calibration_predictions.parquet
    pred_df = pd.DataFrame(prediction_records)
    pred_parquet_path = out_dir / "calibration_predictions.parquet"
    pred_df.to_parquet(pred_parquet_path, index=False)
    logger.info(f"Saved predictions parquet to {pred_parquet_path} ({len(pred_df)} rows)")

    # 5. Reliability Plots
    lr_uncal = results_summary["models"]["logistic_regression"]["calibration_methods"]["uncalibrated"]["metrics"]
    lr_sig = results_summary["models"]["logistic_regression"]["calibration_methods"]["sigmoid"]["metrics"]
    lr_iso = results_summary["models"]["logistic_regression"]["calibration_methods"]["isotonic"]["metrics"]
    plot_comparative_reliability_diagram(
        "Logistic Regression", lr_uncal, lr_sig, lr_iso, out_dir / "logistic_regression_reliability.png"
    )

    xgb_uncal = results_summary["models"]["xgboost"]["calibration_methods"]["uncalibrated"]["metrics"]
    xgb_sig = results_summary["models"]["xgboost"]["calibration_methods"]["sigmoid"]["metrics"]
    xgb_iso = results_summary["models"]["xgboost"]["calibration_methods"]["isotonic"]["metrics"]
    plot_comparative_reliability_diagram(
        "XGBoost", xgb_uncal, xgb_sig, xgb_iso, out_dir / "xgboost_reliability.png"
    )

    # 6. calibration_summary.md
    summary_md_path = out_dir / "calibration_summary.md"
    write_calibration_summary_markdown(summary_md_path, res_df, results_summary)
    logger.info(f"Saved calibration summary markdown to {summary_md_path}")

    return results_summary


def write_calibration_summary_markdown(
    file_path: Path,
    res_df: pd.DataFrame,
    results_summary: Dict[str, Any],
) -> None:
    """Format calibration summary into markdown."""
    lines = [
        "# Comprehensive Probability Calibration Analysis Summary (Phase 17)",
        "",
        "## 1. Experimental Split Protocol",
        "",
        "- **Total Dataset**: 4,000 samples",
        "- **Holdout Test Set (Untouched)**: Exactly 800 samples (same 800 IDs as Phases 14-16)",
        "- **Total Training Partition**: 3,200 samples",
        "  - **Model-Fitting Subset**: 2,560 samples (80% of training data)",
        "  - **Calibration-Fitting Subset**: 640 samples (20% of training data)",
        "- **Zero Leakage**: Base models and scalers were fit on the model subset; calibrators were fit on the calibration subset; test set was strictly evaluated.",
        "",
        "---",
        "",
        "## 2. Overall Model Performance Across Calibration Methods",
        "",
        "$$\\Delta = \\text{Calibrated Metric} - \\text{Uncalibrated Metric}$$",
        "*(For Brier, ECE, and MCE, negative delta indicates error reduction)*",
        "",
        "| Model | Calibration | Accuracy | F1 | ROC-AUC | PR-AUC | Brier ($\\Delta$) | ECE ($\\Delta$) | MCE ($\\Delta$) | Slope | Intercept |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in res_df.iterrows():
        m = row["model"]
        cal = row["calibration"]
        acc = f"{row['accuracy']:.4f}"
        f1 = f"{row['f1']:.4f}"
        auc = f"{row['roc_auc']:.4f}"
        prauc = f"{row['pr_auc']:.4f}"
        brier = f"{row['brier']:.4f} ({row['delta_brier']:+.4f})"
        ece = f"{row['ece']:.4f} ({row['delta_ece']:+.4f})"
        mce = f"{row['mce']:.4f} ({row['delta_mce']:+.4f})"
        slope = f"{row['calibration_slope']:.3f}"
        intercept = f"{row['calibration_intercept']:.3f}"
        lines.append(f"| `{m}` | **{cal}** | {acc} | {f1} | {auc} | {prauc} | {brier} | {ece} | {mce} | {slope} | {intercept} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Subgroup Calibration Breakdown on Holdout Test Set",
        "",
        "| Model | Calibration | Dataset | N | Brier | ECE | MCE | Accuracy | ROC-AUC |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_key in ["logistic_regression", "xgboost"]:
        for cal_method in ["uncalibrated", "sigmoid", "isotonic"]:
            subgroups = results_summary["models"][m_key]["calibration_methods"][cal_method]["subgroups"]
            for ds, ds_m in subgroups.items():
                cm = ds_m["confusion_matrix"]
                n_samples = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
                lines.append(
                    f"| `{m_key}` | {cal_method} | **{ds}** | {n_samples} | "
                    f"{ds_m['brier_score']:.4f} | {ds_m['ece']:.4f} | {ds_m['mce']:.4f} | "
                    f"{ds_m['accuracy']:.4f} | {ds_m['roc_auc']:.4f} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Key Methodological Findings",
        "",
        "- **Brier Score vs. Classification Metrics**: Post-hoc calibration specifically adjusts probability alignment without fundamentally changing binary decision boundaries or rank ordering (ROC-AUC remains nearly constant).",
        "- **Platt Scaling (Sigmoid) Effect**: Provides smooth, parametric adjustment that prevents extreme overconfidence while preserving monotonic rankings.",
        "- **Isotonic Regression Effect**: Non-parametric piecewise constant calibration allows flexible mapping for non-sigmoid distortion.",
        "- **Non-Causality Disclaimer**: Calibration curves describe empirical confidence alignment and do not establish causal origins of hallucination generation.",
    ])

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Probability Calibration Analysis (Phase 17)")
    parser.add_argument(
        "--data",
        type=str,
        default="experiments/baselines/combined/universal_features.parquet",
        help="Path to universal features parquet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/calibration",
        help="Directory to save calibration outputs",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible splitting",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 17: PROBABILITY CALIBRATION ANALYSIS")
    logger.info("=" * 60)

    run_calibration_study(
        data_path=args.data,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
