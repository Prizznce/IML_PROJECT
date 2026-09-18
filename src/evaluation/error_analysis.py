"""
Comprehensive post-hoc error analysis for the Universal Core Hallucination Detector.

Evaluates Logistic Regression and XGBoost predictions on the fixed Phase 14
holdout test set (N=800) across error categories (TP, TN, FP, FN), per-dataset
performance, probability distributions, text-length characteristics, feature-level
effect sizes, and inter-model disagreement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)

PRIMARY_19_FEATURES: list[str] = [
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
    "exact_match_agreement",
    "mean_pairwise_similarity",
    "min_pairwise_similarity",
    "max_pairwise_similarity",
    "pairwise_similarity_std",
    "mean_pairwise_entailment",
    "mean_pairwise_contradiction",
    "nli_disagreement",
]

EXPECTED_TEST_SAMPLE_COUNT: int = 800
EXPECTED_DATASET_COUNTS: dict[str, int] = {
    "halueval": 300,
    "truthfulqa": 300,
    "fever": 200,
}
EXPECTED_LABEL_COUNTS: dict[int, int] = {
    0: 400,
    1: 400,
}


def load_and_validate_error_analysis_data(
    predictions_path: str | Path,
    processed_data_path: str | Path,
    features_path: str | Path,
) -> pd.DataFrame:
    """
    Load test predictions, validate against strict Phase 14 invariants, and join
    text and feature payloads for post-hoc descriptive analysis.
    """
    preds_path = Path(predictions_path)
    proc_path = Path(processed_data_path)
    feat_path = Path(features_path)

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {proc_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Universal features not found: {feat_path}")

    # 1. Load predictions
    preds_df = pd.read_parquet(preds_path)
    if len(preds_df) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLE_COUNT} test predictions, found {len(preds_df)}"
        )

    if preds_df["id"].nunique() != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError("Duplicate IDs found in test predictions.")

    # Validate dataset distribution
    ds_counts = preds_df["source_dataset"].value_counts().to_dict()
    for ds, count in EXPECTED_DATASET_COUNTS.items():
        if ds_counts.get(ds, 0) != count:
            raise ValueError(
                f"Dataset {ds} count mismatch: expected {count}, found {ds_counts.get(ds, 0)}"
            )

    # Validate label distribution
    lbl_counts = preds_df["label"].value_counts().to_dict()
    for lbl, count in EXPECTED_LABEL_COUNTS.items():
        if lbl_counts.get(lbl, 0) != count:
            raise ValueError(
                f"Label {lbl} count mismatch: expected {count}, found {lbl_counts.get(lbl, 0)}"
            )

    # Validate probability bounds
    for prob_col in ["lr_probability", "xgb_probability"]:
        if prob_col in preds_df.columns:
            probs = preds_df[prob_col]
            if (probs < 0.0).any() or (probs > 1.0).any() or probs.isna().any():
                raise ValueError(f"Probabilities in {prob_col} out of [0, 1] or contains NaN.")

    # 2. Join text payloads
    proc_df = pd.read_parquet(proc_path)[["id", "prompt", "response", "context"]]
    merged = preds_df.merge(proc_df, on="id", how="inner")
    if len(merged) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"Merged dataset count mismatch: expected {EXPECTED_TEST_SAMPLE_COUNT}, got {len(merged)}"
        )

    # 3. Join 19 feature columns
    feat_df = pd.read_parquet(feat_path)[["id"] + PRIMARY_19_FEATURES]
    final_df = merged.merge(feat_df, on="id", how="inner")
    if len(final_df) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"Final feature-merged dataset count mismatch: expected {EXPECTED_TEST_SAMPLE_COUNT}, got {len(final_df)}"
        )

    return final_df


def assign_error_categories(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "label",
) -> pd.Series:
    """
    Assign confusion categories: TP, TN, FP, FN.
    """
    y_true = df[target_col].to_numpy()
    y_pred = df[pred_col].to_numpy()

    categories = np.empty(len(df), dtype=object)
    categories[(y_true == 1) & (y_pred == 1)] = "TP"
    categories[(y_true == 0) & (y_pred == 0)] = "TN"
    categories[(y_true == 0) & (y_pred == 1)] = "FP"
    categories[(y_true == 1) & (y_pred == 0)] = "FN"

    return pd.Series(categories, index=df.index)


def compute_confidence(
    probability: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    label: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Compute confidence metric:
    - In general: max(p, 1 - p)
    - Specifically:
      - FP (label=0, pred=1): confidence = p
      - FN (label=1, pred=0): confidence = 1 - p
      - TP (label=1, pred=1): confidence = p
      - TN (label=0, pred=0): confidence = 1 - p
    """
    p = np.asarray(probability, dtype=float)
    return np.maximum(p, 1.0 - p)


def compute_error_category_statistics(
    df: pd.DataFrame,
    model_name: str,
    prob_col: str,
    pred_col: str,
    label_col: str = "label",
) -> dict[str, Any]:
    """
    Compute distribution, probability statistics, and length metrics across TP, TN, FP, FN.
    """
    cat_series = assign_error_categories(df, pred_col, target_col=label_col)
    probs = df[prob_col].to_numpy()
    confs = compute_confidence(probs, df[pred_col].to_numpy(), df[label_col].to_numpy())

    total_samples = len(df)
    categories = ["TP", "TN", "FP", "FN"]
    stats: dict[str, Any] = {}

    for cat in categories:
        mask = (cat_series == cat).to_numpy()
        count = int(np.sum(mask))
        fraction = float(count / total_samples) if total_samples > 0 else 0.0

        # Subgroup fraction within true label
        if cat in ["TP", "FN"]:
            label_total = int(np.sum(df[label_col] == 1))
        else:
            label_total = int(np.sum(df[label_col] == 0))
        label_fraction = float(count / label_total) if label_total > 0 else 0.0

        cat_probs = probs[mask]
        cat_confs = confs[mask]

        # Probability stats
        if count > 0:
            p_mean = float(np.mean(cat_probs))
            p_median = float(np.median(cat_probs))
            p_std = float(np.std(cat_probs, ddof=1)) if count > 1 else 0.0
            p_min = float(np.min(cat_probs))
            p_max = float(np.max(cat_probs))
            conf_mean = float(np.mean(cat_confs))
            conf_median = float(np.median(cat_confs))
            conf_std = float(np.std(cat_confs, ddof=1)) if count > 1 else 0.0
        else:
            p_mean = p_median = p_std = p_min = p_max = 0.0
            conf_mean = conf_median = conf_std = 0.0

        # Text length metrics
        cat_df = df[mask]
        resp_chars = cat_df["response"].astype(str).str.len().to_numpy()
        resp_words = cat_df["response"].astype(str).str.split().str.len().to_numpy()
        prompt_words = cat_df["prompt"].astype(str).str.split().str.len().to_numpy()

        # Context words when context exists
        has_context_mask = cat_df["context"].notna() & (cat_df["context"].astype(str).str.strip() != "")
        context_words = (
            cat_df.loc[has_context_mask, "context"]
            .astype(str)
            .str.split()
            .str.len()
            .to_numpy()
        )

        stats[cat] = {
            "count": count,
            "fraction_of_total": fraction,
            "fraction_of_label": label_fraction,
            "probability": {
                "mean": p_mean,
                "median": p_median,
                "std": p_std,
                "min": p_min,
                "max": p_max,
            },
            "confidence": {
                "mean": conf_mean,
                "median": conf_median,
                "std": conf_std,
            },
            "text_lengths": {
                "response_char_len": {
                    "mean": float(np.mean(resp_chars)) if len(resp_chars) > 0 else 0.0,
                    "median": float(np.median(resp_chars)) if len(resp_chars) > 0 else 0.0,
                    "std": float(np.std(resp_chars, ddof=1)) if len(resp_chars) > 1 else 0.0,
                },
                "response_word_count": {
                    "mean": float(np.mean(resp_words)) if len(resp_words) > 0 else 0.0,
                    "median": float(np.median(resp_words)) if len(resp_words) > 0 else 0.0,
                    "std": float(np.std(resp_words, ddof=1)) if len(resp_words) > 1 else 0.0,
                },
                "prompt_word_count": {
                    "mean": float(np.mean(prompt_words)) if len(prompt_words) > 0 else 0.0,
                    "median": float(np.median(prompt_words)) if len(prompt_words) > 0 else 0.0,
                    "std": float(np.std(prompt_words, ddof=1)) if len(prompt_words) > 1 else 0.0,
                },
                "context_word_count": {
                    "valid_count": int(np.sum(has_context_mask)),
                    "mean": float(np.mean(context_words)) if len(context_words) > 0 else 0.0,
                    "median": float(np.median(context_words)) if len(context_words) > 0 else 0.0,
                    "std": float(np.std(context_words, ddof=1)) if len(context_words) > 1 else 0.0,
                },
            },
        }

    return stats


def compute_dataset_error_metrics(
    df: pd.DataFrame,
    prob_col: str,
    pred_col: str,
    label_col: str = "label",
    dataset_col: str = "source_dataset",
) -> list[dict[str, Any]]:
    """
    Compute per-dataset and overall error rates, FPR, FNR, accuracy, F1, ROC-AUC,
    and mean predicted probabilities.
    """
    results: list[dict[str, Any]] = []

    slices: list[tuple[str, pd.DataFrame]] = [("overall", df)]
    for ds_name, sub_df in df.groupby(dataset_col):
        slices.append((str(ds_name), sub_df))

    for slice_name, slice_df in slices:
        y_true = slice_df[label_col].to_numpy()
        y_pred = slice_df[pred_col].to_numpy()
        y_prob = slice_df[prob_col].to_numpy()

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        total = len(slice_df)

        error_rate = float((fp + fn) / total) if total > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        accuracy = float((tp + tn) / total) if total > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # ROC-AUC requires both classes present
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        else:
            roc_auc = float("nan")

        mean_prob = float(np.mean(y_prob))
        mean_prob_lbl0 = float(np.mean(y_prob[y_true == 0])) if np.any(y_true == 0) else float("nan")
        mean_prob_lbl1 = float(np.mean(y_prob[y_true == 1])) if np.any(y_true == 1) else float("nan")

        results.append({
            "slice": slice_name,
            "total_samples": total,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "error_rate": error_rate,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc,
            "mean_probability": mean_prob,
            "mean_probability_label_0": mean_prob_lbl0,
            "mean_probability_label_1": mean_prob_lbl1,
        })

    return results


def extract_confusion_examples(
    df: pd.DataFrame,
    prob_col: str,
    pred_col: str,
    top_n: int = 10,
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """
    Identify:
    - 10 highest-confidence false positives
    - 10 highest-confidence false negatives
    - 10 lowest-confidence correct predictions
    - 10 highest-confidence correct predictions
    """
    working_df = df.copy()
    working_df["category"] = assign_error_categories(working_df, pred_col, target_col=label_col)
    working_df["confidence"] = compute_confidence(
        working_df[prob_col].to_numpy(),
        working_df[pred_col].to_numpy(),
        working_df[label_col].to_numpy(),
    )

    output_cols = [
        "id",
        "source_dataset",
        "label",
        pred_col,
        prob_col,
        "confidence",
        "prompt",
        "response",
        "context",
    ]

    # False Positives: label=0, pred=1. Sorted by confidence (i.e. prob) descending
    fps = (
        working_df[working_df["category"] == "FP"]
        .sort_values(by="confidence", ascending=False)
        .head(top_n)[output_cols]
        .rename(columns={pred_col: "predicted_label", prob_col: "probability"})
    )

    # False Negatives: label=1, pred=0. Sorted by confidence (i.e. 1 - prob) descending
    fns = (
        working_df[working_df["category"] == "FN"]
        .sort_values(by="confidence", ascending=False)
        .head(top_n)[output_cols]
        .rename(columns={pred_col: "predicted_label", prob_col: "probability"})
    )

    # Lowest-confidence correct predictions: category in [TP, TN]. Sorted by confidence ascending
    correct = working_df[working_df["category"].isin(["TP", "TN"])]
    low_conf_correct = (
        correct.sort_values(by="confidence", ascending=True)
        .head(top_n)[output_cols]
        .rename(columns={pred_col: "predicted_label", prob_col: "probability"})
    )

    # Highest-confidence correct predictions: category in [TP, TN]. Sorted by confidence descending
    high_conf_correct = (
        correct.sort_values(by="confidence", ascending=False)
        .head(top_n)[output_cols]
        .rename(columns={pred_col: "predicted_label", prob_col: "probability"})
    )

    return {
        "highest_confidence_false_positives": fps,
        "highest_confidence_false_negatives": fns,
        "lowest_confidence_correct": low_conf_correct,
        "highest_confidence_correct": high_conf_correct,
    }


def compute_cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups:
    d = (mean1 - mean2) / s_pooled
    where s_pooled = sqrt( ((n1 - 1)*s1^2 + (n2 - 1)*s2^2) / (n1 + n2 - 2) )
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = float(np.mean(group1)), float(np.mean(group2))
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var <= 1e-12:
        return 0.0

    return (mean1 - mean2) / np.sqrt(pooled_var)


def compute_feature_error_effects(
    df: pd.DataFrame,
    pred_col: str,
    feature_cols: list[str] = PRIMARY_19_FEATURES,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Compute mean, median, std, and standardized mean difference (Cohen's d)
    comparing:
    1. TP vs FN (label 1: predicted 1 vs predicted 0)
    2. TN vs FP (label 0: predicted 0 vs predicted 1)
    3. Correct vs Incorrect overall
    """
    cats = assign_error_categories(df, pred_col, target_col=label_col)
    rows: list[dict[str, Any]] = []

    tp_mask = (cats == "TP").to_numpy()
    fn_mask = (cats == "FN").to_numpy()
    tn_mask = (cats == "TN").to_numpy()
    fp_mask = (cats == "FP").to_numpy()
    correct_mask = (cats.isin(["TP", "TN"])).to_numpy()
    incorrect_mask = (cats.isin(["FP", "FN"])).to_numpy()

    for feat in feature_cols:
        vals = df[feat].to_numpy(dtype=float)

        tp_vals, fn_vals = vals[tp_mask], vals[fn_mask]
        tn_vals, fp_vals = vals[tn_mask], vals[fp_mask]
        corr_vals, incorr_vals = vals[correct_mask], vals[incorrect_mask]

        rows.append({
            "feature": feat,
            # TP vs FN
            "tp_mean": float(np.mean(tp_vals)),
            "tp_median": float(np.median(tp_vals)),
            "fn_mean": float(np.mean(fn_vals)),
            "fn_median": float(np.median(fn_vals)),
            "cohen_d_tp_vs_fn": float(compute_cohen_d(tp_vals, fn_vals)),
            # TN vs FP
            "tn_mean": float(np.mean(tn_vals)),
            "tn_median": float(np.median(tn_vals)),
            "fp_mean": float(np.mean(fp_vals)),
            "fp_median": float(np.median(fp_vals)),
            "cohen_d_tn_vs_fp": float(compute_cohen_d(tn_vals, fp_vals)),
            # Correct vs Incorrect
            "correct_mean": float(np.mean(corr_vals)),
            "correct_median": float(np.median(corr_vals)),
            "incorrect_mean": float(np.mean(incorr_vals)),
            "incorrect_median": float(np.median(incorr_vals)),
            "cohen_d_correct_vs_incorrect": float(compute_cohen_d(corr_vals, incorr_vals)),
        })

    return pd.DataFrame(rows)


def compute_model_disagreement(
    df: pd.DataFrame,
    lr_pred_col: str = "lr_prediction",
    lr_prob_col: str = "lr_probability",
    xgb_pred_col: str = "xgb_prediction",
    xgb_prob_col: str = "xgb_probability",
    label_col: str = "label",
    top_n: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Compare Logistic Regression vs XGBoost predictions on the same 800 examples:
    - agreement / disagreement counts & rates
    - both correct, both wrong, LR correct/XGB wrong, LR wrong/XGB correct
    - Top N largest absolute probability disagreements
    """
    y_true = df[label_col].to_numpy()
    lr_pred = df[lr_pred_col].to_numpy()
    xgb_pred = df[xgb_pred_col].to_numpy()
    lr_prob = df[lr_prob_col].to_numpy()
    xgb_prob = df[xgb_prob_col].to_numpy()
    total = len(df)

    agree_mask = lr_pred == xgb_pred
    disagree_mask = ~agree_mask

    both_correct = int(np.sum((lr_pred == y_true) & (xgb_pred == y_true)))
    both_wrong = int(np.sum((lr_pred != y_true) & (xgb_pred != y_true)))
    lr_corr_xgb_wrong = int(np.sum((lr_pred == y_true) & (xgb_pred != y_true)))
    lr_wrong_xgb_corr = int(np.sum((lr_pred != y_true) & (xgb_pred == y_true)))

    summary: dict[str, Any] = {
        "total_samples": total,
        "agreement_count": int(np.sum(agree_mask)),
        "agreement_rate": float(np.mean(agree_mask)),
        "disagreement_count": int(np.sum(disagree_mask)),
        "disagreement_rate": float(np.mean(disagree_mask)),
        "both_correct_count": both_correct,
        "both_correct_rate": float(both_correct / total),
        "both_wrong_count": both_wrong,
        "both_wrong_rate": float(both_wrong / total),
        "lr_correct_xgb_wrong_count": lr_corr_xgb_wrong,
        "lr_correct_xgb_wrong_rate": float(lr_corr_xgb_wrong / total),
        "lr_wrong_xgb_correct_count": lr_wrong_xgb_corr,
        "lr_wrong_xgb_correct_rate": float(lr_wrong_xgb_corr / total),
    }

    diff_df = df.copy()
    diff_df["prob_diff"] = np.abs(lr_prob - xgb_prob)

    top_disagreements = (
        diff_df.sort_values(by="prob_diff", ascending=False)
        .head(top_n)[
            [
                "id",
                "source_dataset",
                "label",
                lr_prob_col,
                xgb_prob_col,
                "prob_diff",
                lr_pred_col,
                xgb_pred_col,
                "prompt",
                "response",
            ]
        ]
        .rename(
            columns={
                lr_prob_col: "lr_probability",
                xgb_prob_col: "xgb_probability",
                "prob_diff": "absolute_probability_difference",
            }
        )
    )

    return summary, top_disagreements


def generate_error_analysis_visualizations(
    df: pd.DataFrame,
    output_dir: str | Path,
    lr_pred_col: str = "lr_prediction",
    lr_prob_col: str = "lr_probability",
    xgb_pred_col: str = "xgb_prediction",
    xgb_prob_col: str = "xgb_probability",
    label_col: str = "label",
) -> list[str]:
    """
    Generate the 7 required matplotlib visualization figures under output_dir:
    1. confusion_matrix_lr.png
    2. confusion_matrix_xgb.png
    3. probability_distribution_lr.png
    4. probability_distribution_xgb.png
    5. error_rate_by_dataset_lr.png
    6. error_rate_by_dataset_xgb.png
    7. feature_differences_comparison.png
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    # 1. Confusion Matrix - Logistic Regression
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_lr = np.zeros((2, 2), dtype=int)
    for i in [0, 1]:
        for j in [0, 1]:
            cm_lr[i, j] = int(np.sum((df[label_col] == i) & (df[lr_pred_col] == j)))
    cax = ax.matshow(cm_lr, cmap="Blues")
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm_lr):
        ax.text(j, i, f"{val}\n({val/len(df):.1%})", ha="center", va="center", color="black" if val < 200 else "white")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 0", "Pred: 1"])
    ax.set_yticklabels(["True: 0", "True: 1"])
    ax.set_title("Confusion Matrix — Logistic Regression", pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    p1 = str(out_path / "confusion_matrix_lr.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    generated_files.append(p1)

    # 2. Confusion Matrix - XGBoost
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_xgb = np.zeros((2, 2), dtype=int)
    for i in [0, 1]:
        for j in [0, 1]:
            cm_xgb[i, j] = int(np.sum((df[label_col] == i) & (df[xgb_pred_col] == j)))
    cax = ax.matshow(cm_xgb, cmap="Greens")
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm_xgb):
        ax.text(j, i, f"{val}\n({val/len(df):.1%})", ha="center", va="center", color="black" if val < 200 else "white")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 0", "Pred: 1"])
    ax.set_yticklabels(["True: 0", "True: 1"])
    ax.set_title("Confusion Matrix — XGBoost", pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    p2 = str(out_path / "confusion_matrix_xgb.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    generated_files.append(p2)

    # 3. Probability Distribution by Category - Logistic Regression
    fig, ax = plt.subplots(figsize=(8, 5))
    cats_lr = assign_error_categories(df, lr_pred_col, target_col=label_col)
    for cat in ["TP", "TN", "FP", "FN"]:
        subset_p = df.loc[cats_lr == cat, lr_prob_col]
        ax.hist(subset_p, bins=20, alpha=0.5, label=f"{cat} (N={len(subset_p)})", density=True)
    ax.set_title("Probability Distribution by Error Category — Logistic Regression")
    ax.set_xlabel("Predicted Probability (P(label=1))")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = str(out_path / "probability_distribution_lr.png")
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    generated_files.append(p3)

    # 4. Probability Distribution by Category - XGBoost
    fig, ax = plt.subplots(figsize=(8, 5))
    cats_xgb = assign_error_categories(df, xgb_pred_col, target_col=label_col)
    for cat in ["TP", "TN", "FP", "FN"]:
        subset_p = df.loc[cats_xgb == cat, xgb_prob_col]
        ax.hist(subset_p, bins=20, alpha=0.5, label=f"{cat} (N={len(subset_p)})", density=True)
    ax.set_title("Probability Distribution by Error Category — XGBoost")
    ax.set_xlabel("Predicted Probability (P(label=1))")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p4 = str(out_path / "probability_distribution_xgb.png")
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    generated_files.append(p4)

    # 5. Error Rate by Dataset - Logistic Regression
    fig, ax = plt.subplots(figsize=(7, 5))
    ds_metrics_lr = compute_dataset_error_metrics(df, lr_prob_col, lr_pred_col, label_col)
    ds_names = [m["slice"] for m in ds_metrics_lr if m["slice"] != "overall"]
    err_rates = [m["error_rate"] for m in ds_metrics_lr if m["slice"] != "overall"]
    fprs = [m["false_positive_rate"] for m in ds_metrics_lr if m["slice"] != "overall"]
    fnrs = [m["false_negative_rate"] for m in ds_metrics_lr if m["slice"] != "overall"]
    x = np.arange(len(ds_names))
    width = 0.25
    ax.bar(x - width, err_rates, width, label="Error Rate", alpha=0.8)
    ax.bar(x, fprs, width, label="FP Rate", alpha=0.8)
    ax.bar(x + width, fnrs, width, label="FN Rate", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("Rate")
    ax.set_title("Error Rates by Dataset — Logistic Regression")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p5 = str(out_path / "error_rate_by_dataset_lr.png")
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    generated_files.append(p5)

    # 6. Error Rate by Dataset - XGBoost
    fig, ax = plt.subplots(figsize=(7, 5))
    ds_metrics_xgb = compute_dataset_error_metrics(df, xgb_prob_col, xgb_pred_col, label_col)
    err_rates_xgb = [m["error_rate"] for m in ds_metrics_xgb if m["slice"] != "overall"]
    fprs_xgb = [m["false_positive_rate"] for m in ds_metrics_xgb if m["slice"] != "overall"]
    fnrs_xgb = [m["false_negative_rate"] for m in ds_metrics_xgb if m["slice"] != "overall"]
    ax.bar(x - width, err_rates_xgb, width, label="Error Rate", alpha=0.8)
    ax.bar(x, fprs_xgb, width, label="FP Rate", alpha=0.8)
    ax.bar(x + width, fnrs_xgb, width, label="FN Rate", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("Rate")
    ax.set_title("Error Rates by Dataset — XGBoost")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p6 = str(out_path / "error_rate_by_dataset_xgb.png")
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    generated_files.append(p6)

    # 7. Feature Mean Comparison (Standardized Difference)
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_effects = compute_feature_error_effects(df, xgb_pred_col)
    y_pos = np.arange(len(feat_effects))
    ax.barh(y_pos, feat_effects["cohen_d_correct_vs_incorrect"], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_effects["feature"])
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Cohen's d (Correct vs Incorrect Predictions — XGBoost)")
    ax.set_title("Feature Standardized Mean Differences: Correct vs Incorrect")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    p7 = str(out_path / "feature_differences_comparison.png")
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    generated_files.append(p7)

    return generated_files


def run_error_analysis_pipeline(
    predictions_path: str | Path = "experiments/baselines/combined/combined_test_predictions.parquet",
    processed_data_path: str | Path = "data/processed/combined_processed.parquet",
    features_path: str | Path = "experiments/baselines/combined/universal_features.parquet",
    output_dir: str | Path = "experiments/error_analysis",
) -> dict[str, Any]:
    """
    Execute full Phase 18 Error Analysis pipeline.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate data
    df = load_and_validate_error_analysis_data(
        predictions_path, processed_data_path, features_path
    )

    # 2. Error category statistics
    lr_cat_stats = compute_error_category_statistics(df, "Logistic Regression", "lr_probability", "lr_prediction")
    xgb_cat_stats = compute_error_category_statistics(df, "XGBoost", "xgb_probability", "xgb_prediction")

    # Build error_category_statistics.csv
    cat_rows: list[dict[str, Any]] = []
    for model_name, model_stats in [("Logistic Regression", lr_cat_stats), ("XGBoost", xgb_cat_stats)]:
        for cat, s in model_stats.items():
            cat_rows.append({
                "model": model_name,
                "category": cat,
                "count": s["count"],
                "fraction_of_total": s["fraction_of_total"],
                "fraction_of_label": s["fraction_of_label"],
                "prob_mean": s["probability"]["mean"],
                "prob_median": s["probability"]["median"],
                "prob_std": s["probability"]["std"],
                "prob_min": s["probability"]["min"],
                "prob_max": s["probability"]["max"],
                "conf_mean": s["confidence"]["mean"],
                "conf_median": s["confidence"]["median"],
                "resp_char_len_mean": s["text_lengths"]["response_char_len"]["mean"],
                "resp_word_count_mean": s["text_lengths"]["response_word_count"]["mean"],
                "prompt_word_count_mean": s["text_lengths"]["prompt_word_count"]["mean"],
                "context_word_count_mean": s["text_lengths"]["context_word_count"]["mean"],
            })
    cat_stats_df = pd.DataFrame(cat_rows)
    cat_stats_df.to_csv(out_dir / "error_category_statistics.csv", index=False)

    # Probability by error category CSV
    prob_cat_rows: list[dict[str, Any]] = []
    for model_name, model_stats in [("Logistic Regression", lr_cat_stats), ("XGBoost", xgb_cat_stats)]:
        for cat, s in model_stats.items():
            prob_cat_rows.append({
                "model": model_name,
                "category": cat,
                "count": s["count"],
                "mean_probability": s["probability"]["mean"],
                "median_probability": s["probability"]["median"],
                "std_probability": s["probability"]["std"],
                "min_probability": s["probability"]["min"],
                "max_probability": s["probability"]["max"],
            })
    prob_cat_df = pd.DataFrame(prob_cat_rows)
    prob_cat_df.to_csv(out_dir / "probability_by_error_category.csv", index=False)

    # 3. Per-dataset error metrics
    lr_ds_metrics = compute_dataset_error_metrics(df, "lr_probability", "lr_prediction")
    xgb_ds_metrics = compute_dataset_error_metrics(df, "xgb_probability", "xgb_prediction")
    ds_rows: list[dict[str, Any]] = []
    for m in lr_ds_metrics:
        row = {"model": "Logistic Regression"}
        row.update(m)
        ds_rows.append(row)
    for m in xgb_ds_metrics:
        row = {"model": "XGBoost"}
        row.update(m)
        ds_rows.append(row)
    ds_df = pd.DataFrame(ds_rows)
    ds_df.to_csv(out_dir / "error_by_dataset.csv", index=False)

    # 4. Confusion examples
    lr_examples = extract_confusion_and_save(df, "lr_probability", "lr_prediction", "lr", out_dir)
    xgb_examples = extract_confusion_and_save(df, "xgb_probability", "xgb_prediction", "xgb", out_dir)

    # 5. Feature error comparison
    lr_feat_effects = compute_feature_error_effects(df, "lr_prediction")
    lr_feat_effects.insert(0, "model", "Logistic Regression")
    xgb_feat_effects = compute_feature_error_effects(df, "xgb_prediction")
    xgb_feat_effects.insert(0, "model", "XGBoost")
    combined_feat_effects = pd.concat([lr_feat_effects, xgb_feat_effects], ignore_index=True)
    combined_feat_effects.to_csv(out_dir / "feature_error_comparison.csv", index=False)

    # 6. Model disagreement
    disagreement_summary, top_disagreements = compute_model_disagreement(df)
    pd.DataFrame([disagreement_summary]).to_csv(out_dir / "model_disagreement.csv", index=False)
    top_disagreements.to_csv(out_dir / "top_model_disagreements.csv", index=False)

    # 7. Generate plots
    plot_files = generate_error_analysis_visualizations(df, out_dir)

    # 8. Compile master JSON
    full_results: dict[str, Any] = {
        "metadata": {
            "total_test_samples": len(df),
            "label_distribution": df["label"].value_counts().to_dict(),
            "dataset_distribution": df["source_dataset"].value_counts().to_dict(),
            "features_analyzed": PRIMARY_19_FEATURES,
        },
        "logistic_regression": {
            "error_categories": lr_cat_stats,
            "dataset_metrics": lr_ds_metrics,
        },
        "xgboost": {
            "error_categories": xgb_cat_stats,
            "dataset_metrics": xgb_ds_metrics,
        },
        "model_disagreement": disagreement_summary,
        "plots_generated": [str(Path(p).name) for p in plot_files],
    }

    with open(out_dir / "error_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # 9. Generate summary markdown
    generate_summary_markdown(
        out_dir / "error_analysis_summary.md",
        full_results,
        ds_df,
        cat_stats_df,
        disagreement_summary,
        top_disagreements,
    )

    return full_results


def extract_confusion_and_save(
    df: pd.DataFrame,
    prob_col: str,
    pred_col: str,
    model_tag: str,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Extract confusion examples and write CSV artifacts.
    """
    examples = extract_confusion_examples(df, prob_col, pred_col, top_n=10)

    # Add model tag column
    for k, sub_df in examples.items():
        sub_df.insert(1, "model", model_tag)

    # If first model, save directly; if second, append
    fp_path = out_dir / "top_false_positives.csv"
    fn_path = out_dir / "top_false_negatives.csv"
    low_conf_path = out_dir / "low_confidence_correct.csv"

    if model_tag == "lr":
        examples["highest_confidence_false_positives"].to_csv(fp_path, index=False)
        examples["highest_confidence_false_negatives"].to_csv(fn_path, index=False)
        examples["lowest_confidence_correct"].to_csv(low_conf_path, index=False)
    else:
        examples["highest_confidence_false_positives"].to_csv(fp_path, mode="a", header=False, index=False)
        examples["highest_confidence_false_negatives"].to_csv(fn_path, mode="a", header=False, index=False)
        examples["lowest_confidence_correct"].to_csv(low_conf_path, mode="a", header=False, index=False)

    return examples


def generate_summary_markdown(
    filepath: Path,
    results: dict[str, Any],
    ds_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    disagreement_summary: dict[str, Any],
    top_disagreements: pd.DataFrame,
) -> None:
    """
    Format executive markdown summary of error analysis.
    """
    lines: list[str] = [
        "# Hallucination Detector Error Analysis Summary",
        "",
        "## 1. Experimental Overview",
        f"- **Evaluated Test Samples**: {results['metadata']['total_test_samples']} (Phase 14 Holdout Set)",
        f"- **Class Balance**: 400 Non-Hallucinated (0), 400 Hallucinated (1)",
        f"- **Dataset Distribution**: HaluEval (300), TruthfulQA (300), FEVER (200)",
        "- **Models Analyzed**: Logistic Regression and XGBoost (Universal 19-Feature Detector)",
        "",
        "## 2. Confusion Category Statistics",
        "",
        "| Model | Category | Count | Total % | Label % | Mean Prob | Median Prob | Mean Resp Words |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in cat_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['category']} | {row['count']} | "
            f"{row['fraction_of_total']:.1%} | {row['fraction_of_label']:.1%} | "
            f"{row['prob_mean']:.4f} | {row['prob_median']:.4f} | "
            f"{row['resp_word_count_mean']:.1f} |"
        )

    lines.extend([
        "",
        "## 3. Per-Dataset Error & Rate Breakdown",
        "",
        "| Model | Dataset | Total | TP | FP | TN | FN | Error Rate | FP Rate | FN Rate | Accuracy | F1 | ROC-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, row in ds_df.iterrows():
        roc_str = f"{row['roc_auc']:.4f}" if not np.isnan(row["roc_auc"]) else "N/A"
        lines.append(
            f"| {row['model']} | {row['slice']} | {row['total_samples']} | "
            f"{row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | "
            f"{row['error_rate']:.4f} | {row['false_positive_rate']:.4f} | {row['false_negative_rate']:.4f} | "
            f"{row['accuracy']:.4f} | {row['f1']:.4f} | {roc_str} |"
        )

    lines.extend([
        "",
        "## 4. Model Disagreement Summary",
        "",
        f"- **Agreement Count**: {disagreement_summary['agreement_count']} / {disagreement_summary['total_samples']} ({disagreement_summary['agreement_rate']:.1%})",
        f"- **Disagreement Count**: {disagreement_summary['disagreement_count']} / {disagreement_summary['total_samples']} ({disagreement_summary['disagreement_rate']:.1%})",
        f"- **Both Correct**: {disagreement_summary['both_correct_count']} ({disagreement_summary['both_correct_rate']:.1%})",
        f"- **Both Incorrect**: {disagreement_summary['both_wrong_count']} ({disagreement_summary['both_wrong_rate']:.1%})",
        f"- **LR Correct / XGB Incorrect**: {disagreement_summary['lr_correct_xgb_wrong_count']} ({disagreement_summary['lr_correct_xgb_wrong_rate']:.1%})",
        f"- **LR Incorrect / XGB Correct**: {disagreement_summary['lr_wrong_xgb_correct_count']} ({disagreement_summary['lr_wrong_xgb_correct_rate']:.1%})",
        "",
        "### Top Disagreement Instances (Largest Absolute Probability Difference)",
        "",
        "| ID | Dataset | Label | LR Prob | XGB Prob | |ΔProb| | Prompt Preview | Response Preview |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |",
    ])

    for _, row in top_disagreements.head(10).iterrows():
        p_prev = str(row["prompt"])[:40].replace("\n", " ") + "..."
        r_prev = str(row["response"])[:40].replace("\n", " ") + "..."
        lines.append(
            f"| {row['id']} | {row['source_dataset']} | {row['label']} | "
            f"{row['lr_probability']:.4f} | {row['xgb_probability']:.4f} | "
            f"{row['absolute_probability_difference']:.4f} | {p_prev} | {r_prev} |"
        )

    lines.append("")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
