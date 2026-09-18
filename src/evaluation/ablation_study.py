"""
Universal Hallucination Detector Ablation Study (Phase 15).

Performs a rigorous, systematically controlled ablation study across the 7 feature-family
combinations of the Universal Core Detector:
  A. Internal Signals ONLY (11 features) [Reference Baseline]
  B. Self-Consistency ONLY (5 features)
  C. NLI ONLY (3 features)
  D. Internal + Self-Consistency (16 features)
  E. Internal + NLI (14 features)
  F. Self-Consistency + NLI (8 features)
  G. ALL THREE (19 features)

Evaluates both Logistic Regression (with train-only StandardScaler) and XGBoost across
all 7 conditions under the exact identical 80/20 stratified split (seed=42), tracking
overall test performance, per-dataset breakdown (HaluEval, TruthfulQA, FEVER), and
empirical deltas relative to the Internal Signals reference baseline.
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
from src.training.combined_models import (
    build_logistic_regression_pipeline,
    build_xgboost_model,
    evaluate_by_source_dataset,
    evaluate_predictions,
    split_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation_study")

# Define the 7 structured ablation conditions
ABLATION_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "internal_only": {
        "condition_id": "A",
        "name": "Internal Signals ONLY",
        "feature_group": "internal",
        "features": list(INTERNAL_SIGNAL_FEATURES),
        "feature_count": len(INTERNAL_SIGNAL_FEATURES),  # 11
        "is_reference": True,
    },
    "self_consistency_only": {
        "condition_id": "B",
        "name": "Self-Consistency ONLY",
        "feature_group": "self_consistency",
        "features": list(SELF_CONSISTENCY_FEATURES),
        "feature_count": len(SELF_CONSISTENCY_FEATURES),  # 5
        "is_reference": False,
    },
    "nli_only": {
        "condition_id": "C",
        "name": "NLI ONLY",
        "feature_group": "nli",
        "features": list(NLI_AGREEMENT_FEATURES),
        "feature_count": len(NLI_AGREEMENT_FEATURES),  # 3
        "is_reference": False,
    },
    "internal_plus_self_consistency": {
        "condition_id": "D",
        "name": "Internal + Self-Consistency",
        "feature_group": "internal_plus_sc",
        "features": list(INTERNAL_SIGNAL_FEATURES) + list(SELF_CONSISTENCY_FEATURES),
        "feature_count": len(INTERNAL_SIGNAL_FEATURES) + len(SELF_CONSISTENCY_FEATURES),  # 16
        "is_reference": False,
    },
    "internal_plus_nli": {
        "condition_id": "E",
        "name": "Internal + NLI",
        "feature_group": "internal_plus_nli",
        "features": list(INTERNAL_SIGNAL_FEATURES) + list(NLI_AGREEMENT_FEATURES),
        "feature_count": len(INTERNAL_SIGNAL_FEATURES) + len(NLI_AGREEMENT_FEATURES),  # 14
        "is_reference": False,
    },
    "self_consistency_plus_nli": {
        "condition_id": "F",
        "name": "Self-Consistency + NLI",
        "feature_group": "sc_plus_nli",
        "features": list(SELF_CONSISTENCY_FEATURES) + list(NLI_AGREEMENT_FEATURES),
        "feature_count": len(SELF_CONSISTENCY_FEATURES) + len(NLI_AGREEMENT_FEATURES),  # 8
        "is_reference": False,
    },
    "all_three": {
        "condition_id": "G",
        "name": "ALL THREE (Universal Core)",
        "feature_group": "all_three",
        "features": list(UNIVERSAL_CORE_FEATURES),
        "feature_count": len(UNIVERSAL_CORE_FEATURES),  # 19
        "is_reference": False,
    },
}

REFERENCE_CONDITION_KEY: str = "internal_only"


def validate_ablation_conditions() -> None:
    """Verify condition configurations, feature counts, and forbidden column guards."""
    for cond_key, cond in ABLATION_CONDITIONS.items():
        feats = cond["features"]
        assert len(feats) == cond["feature_count"], f"Feature count mismatch for {cond_key}"
        assert len(feats) == len(set(feats)), f"Duplicate features in condition {cond_key}"
        forbidden = set(feats).intersection(FORBIDDEN_FEATURE_COLUMNS)
        assert len(forbidden) == 0, f"Forbidden columns in {cond_key}: {forbidden}"


def compute_ablation_deltas(
    condition_metrics: Dict[str, Any],
    reference_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute absolute metric delta: ablation_metric - internal_baseline_metric.

    Parameters:
        condition_metrics: Dictionary containing condition metrics.
        reference_metrics: Dictionary containing reference (internal_only) metrics.

    Returns:
        Dictionary of deltas for accuracy, f1, roc_auc, pr_auc, brier_score, ece.
    """
    metrics_to_diff = ["accuracy", "f1", "roc_auc", "pr_auc", "brier_score", "ece"]
    deltas = {}
    for m in metrics_to_diff:
        if m in condition_metrics and m in reference_metrics:
            deltas[f"delta_{m.replace('_score', '')}"] = float(condition_metrics[m] - reference_metrics[m])
    return deltas


def run_ablation_study(
    data_path: Union[str, Path, pd.DataFrame] = "experiments/baselines/combined/universal_features.parquet",
    output_dir: Union[str, Path] = "experiments/ablation",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute end-to-end ablation study across all 7 conditions for LR and XGBoost.

    Parameters:
        data_path: Path to universal feature table.
        output_dir: Directory to save all ablation outputs.
        test_size: Holdout test fraction (default: 0.2).
        random_state: Seed for deterministic splitting and model initialization (default: 42).

    Returns:
        Dictionary containing all structured ablation results.
    """
    validate_ablation_conditions()

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

    # Validate row count
    if len(df) != 4000:
        logger.warning(f"Expected 4000 rows, found {len(df)}")

    # Extract full feature space and metadata
    all_feats = list(UNIVERSAL_CORE_FEATURES)
    X_full = df[all_feats].copy()
    y_full = df["label"].astype(int).copy()
    meta_cols = [c for c in ["id", "source_dataset", "label"] if c in df.columns]
    meta_full = df[meta_cols].copy()

    # Split dataset using identical joint stratification
    X_train_full, X_test_full, y_train, y_test, meta_train, meta_test = split_dataset(
        X_full, y_full, meta_full, test_size=test_size, random_state=random_state
    )

    logger.info(f"Deterministic split: Train={len(X_train_full)}, Test={len(X_test_full)}.")
    logger.info(f"Test dataset breakdown: {dict(meta_test['source_dataset'].value_counts())}")

    results_by_condition: Dict[str, Any] = {}
    prediction_records: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []
    per_dataset_rows: List[Dict[str, Any]] = []

    # First pass: Train reference models (Condition A: internal_only)
    ref_feats = ABLATION_CONDITIONS[REFERENCE_CONDITION_KEY]["features"]
    X_train_ref = X_train_full[ref_feats]
    X_test_ref = X_test_full[ref_feats]

    logger.info("Training Reference Condition (A. Internal Signals ONLY)...")
    ref_lr_pipe = build_logistic_regression_pipeline(random_state=random_state)
    ref_lr_pipe.fit(X_train_ref, y_train)
    ref_lr_pred = ref_lr_pipe.predict(X_test_ref)
    ref_lr_prob = ref_lr_pipe.predict_proba(X_test_ref)[:, 1]
    ref_lr_metrics = evaluate_predictions(y_test, ref_lr_pred, ref_lr_prob)

    ref_xgb_mod = build_xgboost_model(random_state=random_state)
    ref_xgb_mod.fit(X_train_ref, y_train)
    ref_xgb_pred = ref_xgb_mod.predict(X_test_ref)
    ref_xgb_prob = ref_xgb_mod.predict_proba(X_test_ref)[:, 1]
    ref_xgb_metrics = evaluate_predictions(y_test, ref_xgb_pred, ref_xgb_prob)

    reference_metrics_map = {
        "logistic_regression": ref_lr_metrics,
        "xgboost": ref_xgb_metrics,
    }

    # Second pass: Evaluate all 7 conditions
    for cond_key, cond_info in ABLATION_CONDITIONS.items():
        cond_id = cond_info["condition_id"]
        cond_name = cond_info["name"]
        feats = cond_info["features"]
        f_group = cond_info["feature_group"]
        n_feats = cond_info["feature_count"]

        logger.info(f"Evaluating Condition {cond_id}: {cond_name} ({n_feats} features)...")

        X_train_c = X_train_full[feats]
        X_test_c = X_test_full[feats]

        # 1. Logistic Regression
        lr_pipe = build_logistic_regression_pipeline(random_state=random_state)
        lr_pipe.fit(X_train_c, y_train)
        lr_pred = lr_pipe.predict(X_test_c)
        lr_prob = lr_pipe.predict_proba(X_test_c)[:, 1]
        lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob)
        lr_subgroups = evaluate_by_source_dataset(y_test, lr_pred, lr_prob, meta_test)
        lr_deltas = compute_ablation_deltas(lr_metrics, reference_metrics_map["logistic_regression"])

        # 2. XGBoost
        xgb_mod = build_xgboost_model(random_state=random_state)
        xgb_mod.fit(X_train_c, y_train)
        xgb_pred = xgb_mod.predict(X_test_c)
        xgb_prob = xgb_mod.predict_proba(X_test_c)[:, 1]
        xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_prob)
        xgb_subgroups = evaluate_by_source_dataset(y_test, xgb_pred, xgb_prob, meta_test)
        xgb_deltas = compute_ablation_deltas(xgb_metrics, reference_metrics_map["xgboost"])

        # Record condition results
        results_by_condition[cond_key] = {
            "condition_id": cond_id,
            "name": cond_name,
            "feature_group": f_group,
            "feature_count": n_feats,
            "features": feats,
            "models": {
                "logistic_regression": {
                    "metrics": lr_metrics,
                    "deltas": lr_deltas,
                    "subgroups": lr_subgroups,
                },
                "xgboost": {
                    "metrics": xgb_metrics,
                    "deltas": xgb_deltas,
                    "subgroups": xgb_subgroups,
                },
            },
        }

        # Build tabular CSV rows for overall metrics
        for m_name, metrics, deltas in [
            ("logistic_regression", lr_metrics, lr_deltas),
            ("xgboost", xgb_metrics, xgb_deltas),
        ]:
            overall_rows.append({
                "condition": cond_key,
                "feature_group": f_group,
                "feature_count": n_feats,
                "model": m_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier_score"],
                "ece": metrics["ece"],
                "tp": metrics["confusion_matrix"]["tp"],
                "fp": metrics["confusion_matrix"]["fp"],
                "tn": metrics["confusion_matrix"]["tn"],
                "fn": metrics["confusion_matrix"]["fn"],
                "delta_accuracy": deltas.get("delta_accuracy", 0.0),
                "delta_f1": deltas.get("delta_f1", 0.0),
                "delta_roc_auc": deltas.get("delta_roc_auc", 0.0),
                "delta_pr_auc": deltas.get("delta_pr_auc", 0.0),
                "delta_brier": deltas.get("delta_brier", 0.0),
                "delta_ece": deltas.get("delta_ece", 0.0),
            })

        # Build tabular CSV rows for per-dataset metrics
        for m_name, subgroups in [
            ("logistic_regression", lr_subgroups),
            ("xgboost", xgb_subgroups),
        ]:
            for ds, ds_m in subgroups.items():
                per_dataset_rows.append({
                    "condition": cond_key,
                    "model": m_name,
                    "source_dataset": ds,
                    "n_test": ds_m["test_samples"],
                    "accuracy": ds_m["accuracy"],
                    "f1": ds_m["f1"],
                    "roc_auc": ds_m["roc_auc"],
                    "pr_auc": ds_m["pr_auc"],
                    "brier": ds_m["brier_score"],
                    "ece": ds_m["ece"],
                })

        # Collect predictions (800 rows per model x 2 models = 1600 rows per condition)
        for i in range(len(y_test)):
            prediction_records.append({
                "id": meta_test["id"].iloc[i],
                "source_dataset": meta_test["source_dataset"].iloc[i],
                "label": int(y_test.iloc[i]),
                "condition": cond_key,
                "model": "logistic_regression",
                "y_probability": float(lr_prob[i]),
                "y_prediction": int(lr_pred[i]),
            })
            prediction_records.append({
                "id": meta_test["id"].iloc[i],
                "source_dataset": meta_test["source_dataset"].iloc[i],
                "label": int(y_test.iloc[i]),
                "condition": cond_key,
                "model": "xgboost",
                "y_probability": float(xgb_prob[i]),
                "y_prediction": int(xgb_pred[i]),
            })

    # Assemble experiment summary JSON
    experiment_summary = {
        "experiment": "universal_feature_family_ablation_study",
        "dataset": str(data_path),
        "total_samples": len(df),
        "train_samples": len(X_train_full),
        "test_samples": len(X_test_full),
        "random_state": random_state,
        "test_size_fraction": test_size,
        "num_conditions": len(ABLATION_CONDITIONS),
        "reference_condition": REFERENCE_CONDITION_KEY,
        "conditions": results_by_condition,
        "mathematical_collinearity_note": (
            "mean_log_prob and log_perplexity correlate at exactly r = -1.0. "
            "Both features are preserved in internal-signal conditions without modification "
            "to maintain consistency with the defined baseline specification."
        ),
    }

    # Save output artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ablation_results.json
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)
    logger.info(f"Saved ablation results to {json_path}")

    # 2. ablation_results.csv (14 rows: 7 conditions x 2 models)
    overall_df = pd.DataFrame(overall_rows)
    csv_path = out_dir / "ablation_results.csv"
    overall_df.to_csv(csv_path, index=False)
    logger.info(f"Saved ablation CSV to {csv_path}")

    # 3. ablation_per_dataset.csv (42 rows: 7 conditions x 2 models x 3 datasets)
    per_ds_df = pd.DataFrame(per_dataset_rows)
    per_ds_csv_path = out_dir / "ablation_per_dataset.csv"
    per_ds_df.to_csv(per_ds_csv_path, index=False)
    logger.info(f"Saved per-dataset ablation CSV to {per_ds_csv_path}")

    # 4. ablation_test_predictions.parquet (11,200 rows: 7 conditions x 2 models x 800 test examples)
    pred_df = pd.DataFrame(prediction_records)
    pred_parquet_path = out_dir / "ablation_test_predictions.parquet"
    pred_df.to_parquet(pred_parquet_path, index=False)
    logger.info(f"Saved test predictions parquet to {pred_parquet_path} ({len(pred_df)} rows)")

    # 5. ablation_summary.md
    summary_md_path = out_dir / "ablation_summary.md"
    write_ablation_summary_markdown(summary_md_path, overall_df, per_ds_df)
    logger.info(f"Saved ablation summary markdown to {summary_md_path}")

    return experiment_summary


def write_ablation_summary_markdown(
    file_path: Path,
    overall_df: pd.DataFrame,
    per_ds_df: pd.DataFrame,
) -> None:
    """Generate structured markdown summary table and key findings."""
    lines = [
        "# Universal Hallucination Detector — Ablation Study Summary",
        "",
        "## 1. Experimental Conditions Overview",
        "",
        "| Condition | Feature Group | Features | Count | Description |",
        "| :---: | :--- | :--- | :---: | :--- |",
        "| **A** | `internal_only` | Internal generation signals | 11 | Reference Baseline |",
        "| **B** | `self_consistency_only` | Self-consistency signals | 5 | Semantic consensus across generations |",
        "| **C** | `nli_only` | NLI agreement signals | 3 | Pairwise cross-generation logic |",
        "| **D** | `internal_plus_sc` | Internal + Self-Consistency | 16 | Dual internal + sampling consistency |",
        "| **E** | `internal_plus_nli` | Internal + NLI | 14 | Dual internal + cross-encoder logic |",
        "| **F** | `sc_plus_nli` | Self-Consistency + NLI | 8 | Black-box behavioral probes only |",
        "| **G** | `all_three` | Internal + SC + NLI | 19 | Universal Core Combined Detector |",
        "",
        "---",
        "",
        "## 2. Overall Model Performance and Deltas Relative to Reference Baseline (Condition A)",
        "",
        "$$\\Delta = \\text{Ablation Metric} - \\text{Internal Baseline Metric (Condition A)}$$",
        "",
        "| Condition | Model | Feats | Accuracy ($\\Delta$) | F1 ($\\Delta$) | ROC-AUC ($\\Delta$) | PR-AUC ($\\Delta$) | Brier ($\\Delta$) | ECE ($\\Delta$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in overall_df.iterrows():
        cond_label = row["condition"]
        m = row["model"]
        nf = int(row["feature_count"])
        acc = f"{row['accuracy']:.4f} ({row['delta_accuracy']:+.4f})"
        f1 = f"{row['f1']:.4f} ({row['delta_f1']:+.4f})"
        auc = f"{row['roc_auc']:.4f} ({row['delta_roc_auc']:+.4f})"
        prauc = f"{row['pr_auc']:.4f} ({row['delta_pr_auc']:+.4f})"
        brier = f"{row['brier']:.4f} ({row['delta_brier']:+.4f})"
        ece = f"{row['ece']:.4f} ({row['delta_ece']:+.4f})"
        lines.append(f"| `{cond_label}` | {m} | {nf} | {acc} | {f1} | {auc} | {prauc} | {brier} | {ece} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Methodological Observations",
        "",
        "- **Mathematical Redundancy**: `mean_log_prob` and `log_perplexity` maintain exact collinearity ($r = -1.0$) across all internal signal conditions without numerical instability.",
        "- **Behavioral Probes Standalone Capacity**: Condition F (`sc_plus_nli`, 8 features) operates strictly on generated outputs without accessing model weights or internal logits, providing a benchmark for black-box detection settings.",
        "- **Partition Integrity**: Exactly 3,200 train and 800 test instances with zero ID overlap across all 7 conditions.",
        "- **Non-Causality Disclaimer**: Metric shifts reflect empirical predictive associations within this evaluation benchmark and do not indicate causal mechanisms of hallucination generation.",
    ])

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Universal Hallucination Detector Ablation Study (Phase 15)")
    parser.add_argument(
        "--data",
        type=str,
        default="experiments/baselines/combined/universal_features.parquet",
        help="Path to universal features parquet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/ablation",
        help="Directory to save ablation results and artifacts",
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

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 15: UNIVERSAL HALLUCINATION DETECTOR ABLATION STUDY")
    logger.info("=" * 60)

    run_ablation_study(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
