"""
Probability Calibration Analysis for LLM Hallucination Detection Baselines.

Provides rigorous calibration evaluation for probabilistic classifiers:
- Reliability diagrams / calibration curves using uniform probability binning
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Detailed bin-by-bin calibration data tables
"""

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
from sklearn.metrics import brier_score_loss

from src.training.baseline_models import (
    PRIMARY_SIGNAL_FEATURES,
    prepare_feature_dataframe,
    split_dataset,
    build_logistic_regression_pipeline,
    build_xgboost_model,
)

logger = logging.getLogger("calibration_analysis")


def compute_calibration_data(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Union[np.ndarray, pd.Series, List[float]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Partition predicted probabilities into M equal-width bins and compute
    reliability statistics, ECE, MCE, and Brier score.

    Parameters:
        y_true: Ground-truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (hallucination).
        n_bins: Number of equal-width bins between 0.0 and 1.0 (default: 10).

    Returns:
        Dict containing:
            - bins: List of dicts with bin-level metrics
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - brier_score: Mean squared error of probabilities
            - total_samples: Total number of evaluated samples
            - non_empty_bins: Number of bins containing at least 1 sample
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("Lengths of y_true and y_prob must match.")

    n_total = len(y_true_arr)
    if n_total == 0:
        return {
            "bins": [],
            "ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "total_samples": 0,
            "non_empty_bins": 0,
        }

    # Defensive clipping to [0, 1]
    y_prob_clamped = np.clip(y_prob_arr, 0.0, 1.0)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    bins_data: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    total_binned_samples = 0

    for i in range(n_bins):
        lower = float(bin_boundaries[i])
        upper = float(bin_boundaries[i + 1])

        # For the final bin, include the upper boundary 1.0
        if i == n_bins - 1:
            in_bin = (y_prob_clamped >= lower) & (y_prob_clamped <= upper)
        else:
            in_bin = (y_prob_clamped >= lower) & (y_prob_clamped < upper)

        count = int(np.sum(in_bin))
        total_binned_samples += count

        if count > 0:
            mean_prob = float(np.mean(y_prob_clamped[in_bin]))
            acc = float(np.mean(y_true_arr[in_bin]))
            gap = float(abs(acc - mean_prob))
            ece += (count / n_total) * gap
            if gap > mce:
                mce = gap
        else:
            mean_prob = None
            acc = None
            gap = 0.0

        bins_data.append({
            "bin_index": i,
            "bin_lower": round(lower, 2),
            "bin_upper": round(upper, 2),
            "sample_count": count,
            "sample_fraction": round(count / n_total, 4),
            "mean_predicted_probability": round(mean_prob, 4) if mean_prob is not None else None,
            "observed_fraction_positive": round(acc, 4) if acc is not None else None,
            "absolute_calibration_gap": round(gap, 4),
        })

    # Validate that every sample was assigned to exactly one bin
    assert total_binned_samples == n_total, f"Bin count sum ({total_binned_samples}) != total samples ({n_total})"

    brier = float(brier_score_loss(y_true_arr, y_prob_clamped))
    non_empty = sum(1 for b in bins_data if b["sample_count"] > 0)

    return {
        "bins": bins_data,
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "total_samples": n_total,
        "non_empty_bins": non_empty,
    }


def plot_reliability_diagram(
    calib_data: Dict[str, Any],
    model_name: str,
    output_path: Union[str, Path],
) -> None:
    """
    Generate and save a publication-quality reliability diagram using matplotlib.

    Upper subplot: Calibration curve vs. perfectly calibrated diagonal.
    Lower subplot: Histogram of sample counts per probability bin.

    Parameters:
        calib_data: Calibration metrics dict from compute_calibration_data.
        model_name: Display name of the model (e.g., 'Logistic Regression').
        output_path: Path to output PNG image.
    """
    import matplotlib.pyplot as plt

    bins = calib_data["bins"]
    ece = calib_data["ece"]
    mce = calib_data["mce"]
    brier = calib_data["brier_score"]

    bin_centers = []
    mean_probs = []
    observed_accs = []
    sample_counts = []
    bin_labels = []

    for b in bins:
        center = (b["bin_lower"] + b["bin_upper"]) / 2.0
        bin_centers.append(center)
        bin_labels.append(f"{b['bin_lower']:.1f}-{b['bin_upper']:.1f}")
        sample_counts.append(b["sample_count"])
        if b["mean_predicted_probability"] is not None:
            mean_probs.append(b["mean_predicted_probability"])
            observed_accs.append(b["observed_fraction_positive"])
        else:
            mean_probs.append(center)
            observed_accs.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # 1. Upper Plot: Reliability Diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7, linewidth=1.5)

    valid_mask = [not np.isnan(a) for a in observed_accs]
    valid_probs = [p for p, v in zip(mean_probs, valid_mask) if v]
    valid_accs = [a for a, v in zip(observed_accs, valid_mask) if v]

    ax1.plot(
        valid_probs,
        valid_accs,
        marker="o",
        color="#1f77b4" if "Logistic" in model_name else "#2ca02c",
        linewidth=2,
        label=f"{model_name} (ECE = {ece:.3f}, MCE = {mce:.3f}, Brier = {brier:.3f})",
    )

    # Plot bars connecting observed accuracy to perfect diagonal (calibration gap)
    for p, a in zip(valid_probs, valid_accs):
        ax1.plot([p, p], [p, a], color="red", linestyle=":", alpha=0.6, linewidth=1.2)

    ax1.set_ylabel("Observed Fraction Positive (Accuracy)", fontsize=11)
    ax1.set_title(f"Reliability Diagram: {model_name} (Primary 11 Signals)", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # 2. Lower Plot: Sample Distribution per Bin
    bar_width = 0.08
    ax2.bar(
        bin_centers,
        sample_counts,
        width=bar_width,
        color="#4c72b0" if "Logistic" in model_name else "#55a868",
        edgecolor="black",
        alpha=0.8,
    )
    ax2.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.set_ylim(0, max(sample_counts) * 1.15 if max(sample_counts) > 0 else 10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Value labels on bars
    for c, count in zip(bin_centers, sample_counts):
        if count > 0:
            ax2.text(c, count + 3, str(count), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close(fig)
    logger.info(f"Saved reliability diagram plot to {out_p}")


def run_primary_calibration_analysis(
    data_path: Union[str, Path] = "experiments/baselines/supervised_signals_combined.parquet",
    output_dir: Union[str, Path] = "experiments/baselines/results/calibration",
    test_size: float = 0.2,
    random_state: int = 42,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Execute full calibration analysis for PRIMARY 11-feature baseline models.

    Parameters:
        data_path: Path to supervised signal parquet dataset.
        output_dir: Output directory for plots and calibration metrics JSON.
        test_size: Stratified hold-out fraction (default: 0.2).
        random_state: Random seed for deterministic reproducibility (default: 42).
        n_bins: Number of calibration bins (default: 10).

    Returns:
        Dict with calibration data for Logistic Regression and XGBoost.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_path = Path(data_path)
    if not d_path.exists():
        raise FileNotFoundError(f"Dataset not found at {d_path}")

    logger.info(f"Loading dataset from {d_path}...")
    df = pd.read_parquet(d_path)

    # 1. Feature preparation & stratified split (strictly identical to primary baseline)
    X, y, meta = prepare_feature_dataframe(df, include_num_tokens=False)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_dataset(
        X, y, meta, test_size=test_size, random_state=random_state
    )

    logger.info(f"Reproduced exact split: Train={len(X_train)}, Test={len(X_test)}")
    assert len(X_test) == 800, f"Expected 800 test samples, got {len(X_test)}"
    assert len(X_train) == 3200, f"Expected 3200 train samples, got {len(X_train)}"

    # 2. Fit Logistic Regression pipeline & predict probabilities
    logger.info("Fitting Logistic Regression pipeline on X_train...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=random_state)
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

    # 3. Fit XGBoost model & predict probabilities
    logger.info("Fitting XGBoost model on X_train...")
    xgb_model = build_xgboost_model(random_state=random_state)
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # 4. Compute calibration tables & metrics
    lr_calib = compute_calibration_data(y_test, lr_probs, n_bins=n_bins)
    xgb_calib = compute_calibration_data(y_test, xgb_probs, n_bins=n_bins)

    # 5. Plot reliability diagrams
    plot_reliability_diagram(
        lr_calib,
        model_name="Logistic Regression",
        output_path=out_dir / "logistic_regression_reliability.png",
    )
    plot_reliability_diagram(
        xgb_calib,
        model_name="XGBoost",
        output_path=out_dir / "xgboost_reliability.png",
    )

    results = {
        "metadata": {
            "dataset": str(data_path),
            "test_size": test_size,
            "random_state": random_state,
            "n_bins": n_bins,
            "num_features": len(PRIMARY_SIGNAL_FEATURES),
            "features_used": PRIMARY_SIGNAL_FEATURES,
            "include_num_tokens": False,
            "total_test_samples": len(y_test),
        },
        "logistic_regression": lr_calib,
        "xgboost": xgb_calib,
    }

    metrics_file = out_dir / "calibration_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved calibration metrics to {metrics_file}")

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    run_primary_calibration_analysis()


if __name__ == "__main__":
    main()
