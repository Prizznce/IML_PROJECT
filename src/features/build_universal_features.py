"""
Universal Combined Feature Table Builder.

Constructs the canonical 19-feature supervised table for the Tier-1 Universal Core
Hallucination Detector by joining:
  1. Internal Generation Signals (11 features) from Qwen3.5-0.8B
  2. Self-Consistency Signals (5 features) across K=5 stochastic responses
  3. NLI Agreement Signals (3 features) from cross-encoder/nli-MiniLM2-L6-H768

Strictly preserves metadata ('id', 'source_dataset', 'label') while forbidding
shortcut length features (num_tokens) and text payloads from entering feature space X.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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
logger = logging.getLogger("build_universal_features")

# ==============================================================================
# Feature Schema Definitions
# ==============================================================================

METADATA_COLUMNS: List[str] = [
    "id",
    "source_dataset",
    "label",
]

# Group 1: 11 internal signal features
INTERNAL_SIGNAL_FEATURES: List[str] = [
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

# Group 2: 5 primary self-consistency features
SELF_CONSISTENCY_FEATURES: List[str] = [
    "exact_match_agreement",
    "mean_pairwise_similarity",
    "min_pairwise_similarity",
    "max_pairwise_similarity",
    "pairwise_similarity_std",
]

# Group 3: 3 primary NLI agreement features
NLI_AGREEMENT_FEATURES: List[str] = [
    "mean_pairwise_entailment",
    "mean_pairwise_contradiction",
    "nli_disagreement",
]

# Canonical 19 features of the Universal Core Detector
UNIVERSAL_CORE_FEATURES: List[str] = (
    INTERNAL_SIGNAL_FEATURES + SELF_CONSISTENCY_FEATURES + NLI_AGREEMENT_FEATURES
)

# Forbidden columns that must NEVER enter feature matrix X
FORBIDDEN_FEATURE_COLUMNS: Set[str] = {
    # Metadata & labels
    "id",
    "source_dataset",
    "label",
    "original_label",
    # Raw text payloads
    "prompt",
    "context",
    "response",
    "original_response",
    "model_input",
    "query_text",
    "generated_response",
    "normalized_response",
    # Diagnostics & timing
    "forward_time_s",
    "num_generations",
    # Raw untransformed heavy-tailed metrics
    "perplexity",
    "mean_token_rank",
    "max_token_rank",
    "min_token_rank",
    "rank_std",
    "min_token_prob",
    # Forbidden shortcut length feature
    "num_tokens",
    # Redundant / collinear self-consistency ablation features
    "unique_response_ratio",
    "majority_response_fraction",
    "generation_disagreement",
    # Redundant NLI ablation features
    "mean_pairwise_neutral",
    "fraction_entailing_pairs",
    "fraction_contradicting_pairs",
    "fraction_neutral_pairs",
}


# ==============================================================================
# Feature Construction Functions
# ==============================================================================

def compute_internal_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply variance-stabilizing log transformations to heavy-tailed internal signals.

    Matches baseline_models.py exactly:
      log_perplexity = log(max(perplexity, 1e-12))
      log_mean_token_rank = log1p(max(mean_token_rank, 0))
      log_max_token_rank = log1p(max(max_token_rank, 0))
      log_rank_std = log1p(max(rank_std, 0))

    Parameters:
        df: DataFrame containing raw internal token statistics.

    Returns:
        DataFrame with the 4 log-transformed columns added.
    """
    df_out = df.copy()
    df_out["log_perplexity"] = np.log(np.maximum(df_out["perplexity"].astype(float), 1e-12))
    df_out["log_mean_token_rank"] = np.log1p(np.maximum(df_out["mean_token_rank"].astype(float), 0.0))
    df_out["log_max_token_rank"] = np.log1p(np.maximum(df_out["max_token_rank"].astype(float), 0.0))
    df_out["log_rank_std"] = np.log1p(np.maximum(df_out["rank_std"].astype(float), 0.0))
    return df_out


def build_universal_feature_dataframe(
    df_internal: pd.DataFrame,
    df_self_consistency: pd.DataFrame,
    df_nli: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble and validate the universal combined feature DataFrame.

    Joins strictly on unique example ID ('id').

    Parameters:
        df_internal: Supervised signals table containing internal token signals.
        df_self_consistency: Aggregated self-consistency features table.
        df_nli: Aggregated NLI agreement features table.

    Returns:
        pd.DataFrame: Merged DataFrame with metadata and 19 universal features.
    """
    # 1. Verify exact ID alignment
    ids_internal = set(df_internal["id"].astype(str))
    ids_sc = set(df_self_consistency["id"].astype(str))
    ids_nli = set(df_nli["id"].astype(str))

    if ids_internal != ids_sc:
        diff_sc = ids_internal.symmetric_difference(ids_sc)
        raise ValueError(f"ID mismatch between internal signals and self-consistency ({len(diff_sc)} differing IDs)")
    if ids_internal != ids_nli:
        diff_nli = ids_internal.symmetric_difference(ids_nli)
        raise ValueError(f"ID mismatch between internal signals and NLI agreement ({len(diff_nli)} differing IDs)")

    # 2. Transform internal features
    df_internal_transformed = compute_internal_log_transforms(df_internal)

    # 3. Select subsets
    df_base = df_internal_transformed[METADATA_COLUMNS + INTERNAL_SIGNAL_FEATURES].copy()
    df_base["id"] = df_base["id"].astype(str)

    df_sc_sub = df_self_consistency[["id"] + SELF_CONSISTENCY_FEATURES].copy()
    df_sc_sub["id"] = df_sc_sub["id"].astype(str)

    df_nli_sub = df_nli[["id"] + NLI_AGREEMENT_FEATURES].copy()
    df_nli_sub["id"] = df_nli_sub["id"].astype(str)

    # 4. Merge sequentially on 'id'
    merged = df_base.merge(df_sc_sub, on="id", how="inner", validate="one_to_one")
    merged = merged.merge(df_nli_sub, on="id", how="inner", validate="one_to_one")

    # 5. Order columns canonically
    final_cols = METADATA_COLUMNS + UNIVERSAL_CORE_FEATURES
    df_universal = merged[final_cols].copy()

    return df_universal


def validate_universal_feature_dataframe(
    df: pd.DataFrame,
    df_orig: Optional[pd.DataFrame] = None,
    expected_count: int = 4000,
) -> Dict[str, Any]:
    """
    Execute comprehensive validation assertions on the universal feature table.

    Parameters:
        df: The combined universal feature DataFrame.
        df_orig: Optional original supervised signals DataFrame for metadata matching.
        expected_count: Expected total example count.

    Returns:
        Dict[str, Any]: Summary diagnostics, per-feature statistics, and correlation audit.
    """
    logger.info("Executing comprehensive validation on universal feature table...")

    # 1. Row count check
    assert len(df) == expected_count, (
        f"Row count mismatch: expected {expected_count}, got {len(df)}"
    )

    # 2. Unique IDs check
    assert df["id"].nunique() == expected_count, (
        f"Unique ID count mismatch: expected {expected_count}, got {df['id'].nunique()}"
    )

    # 3. No duplicate IDs
    assert not df["id"].duplicated().any(), "Duplicate IDs detected in universal feature table"

    # 4. Exact feature count: 19 features + 3 metadata = 22 columns
    assert len(df.columns) == len(METADATA_COLUMNS) + len(UNIVERSAL_CORE_FEATURES), (
        f"Column count mismatch: expected {len(METADATA_COLUMNS) + len(UNIVERSAL_CORE_FEATURES)}, got {len(df.columns)}"
    )
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    assert len(feature_cols) == 19, f"Expected exactly 19 feature columns, got {len(feature_cols)}"
    assert set(feature_cols) == set(UNIVERSAL_CORE_FEATURES), (
        f"Feature columns do not match expected set: differing={set(feature_cols).symmetric_difference(set(UNIVERSAL_CORE_FEATURES))}"
    )

    # 5. Strict policy check: No forbidden columns in feature set
    forbidden_overlap = set(feature_cols).intersection(FORBIDDEN_FEATURE_COLUMNS)
    assert not forbidden_overlap, f"Forbidden columns detected in feature set: {forbidden_overlap}"
    assert "num_tokens" not in feature_cols, "Policy violation: num_tokens found in universal feature set!"

    # 6. Check against original supervised dataset if provided
    if df_orig is not None:
        assert len(df_orig) == expected_count, "Original dataset count mismatch"
        assert set(df["id"]) == set(df_orig["id"]), "ID set does not match original supervised dataset"
        df_check = df.merge(df_orig[["id", "source_dataset", "label"]], on="id", suffixes=("", "_orig"))
        assert (df_check["source_dataset"] == df_check["source_dataset_orig"]).all(), "source_dataset mismatch with original"
        assert (df_check["label"] == df_check["label_orig"]).all(), "label mismatch with original"

    # 7. Numerical completeness: Zero NaN or Inf values
    for col in UNIVERSAL_CORE_FEATURES:
        assert not df[col].isna().any(), f"NaN detected in feature '{col}'"
        assert np.all(np.isfinite(df[col].values)), f"Non-finite value detected in feature '{col}'"

    # 8. Expected numerical ranges
    # Probability bounds
    prob_cols = [
        "mean_token_prob", "token_prob_std",
        "exact_match_agreement",
        "mean_pairwise_entailment", "mean_pairwise_contradiction", "nli_disagreement"
    ]
    for col in prob_cols:
        assert (df[col] >= 0.0 - 1e-6).all(), f"Values < 0 in '{col}': min={df[col].min()}"
        assert (df[col] <= 1.0 + 1e-6).all(), f"Values > 1 in '{col}': max={df[col].max()}"

    # Cosine similarity bounds [-1.0, 1.0]
    sim_cols = ["mean_pairwise_similarity", "min_pairwise_similarity", "max_pairwise_similarity"]
    for col in sim_cols:
        assert (df[col] >= -1.0 - 1e-6).all(), f"Cosine similarity < -1.0 in '{col}': min={df[col].min()}"
        assert (df[col] <= 1.0 + 1e-6).all(), f"Cosine similarity > 1.0 in '{col}': max={df[col].max()}"

    # Non-negative metrics
    non_neg_cols = [
        "mean_entropy", "max_entropy", "entropy_std", "pairwise_similarity_std",
        "log_perplexity", "log_mean_token_rank", "log_max_token_rank", "log_rank_std"
    ]
    for col in non_neg_cols:
        assert (df[col] >= 0.0 - 1e-6).all(), f"Non-negative feature < 0 in '{col}': min={df[col].min()}"

    # Log probability bounds (log prob <= 0)
    for col in ["min_log_prob", "mean_log_prob"]:
        assert (df[col] <= 0.0 + 1e-6).all(), f"Log probability > 0 in '{col}': max={df[col].max()}"

    # 9. Near-zero variance check
    feature_stds = df[UNIVERSAL_CORE_FEATURES].std()
    near_zero_var = feature_stds[feature_stds < 1e-4]
    assert len(near_zero_var) == 0, f"Near-zero variance detected in features: {near_zero_var.to_dict()}"

    # 10. Summary statistics and correlation matrix
    desc = df[UNIVERSAL_CORE_FEATURES].describe().to_dict()
    corr_matrix = df[UNIVERSAL_CORE_FEATURES].corr()

    # Find any exact collinear pairs (|r| > 0.999)
    collinear_pairs: List[Tuple[str, str, float]] = []
    for i, f1 in enumerate(UNIVERSAL_CORE_FEATURES):
        for j, f2 in enumerate(UNIVERSAL_CORE_FEATURES):
            if i < j:
                r_val = float(corr_matrix.loc[f1, f2])
                if abs(r_val) > 0.999:
                    collinear_pairs.append((f1, f2, r_val))

    if collinear_pairs:
        logger.warning(
            f"Identified exact mathematical redundancies among features: {collinear_pairs}. "
            f"(Documented: log_perplexity is mathematically -mean_log_prob)."
        )

    logger.info("All universal feature table validation assertions passed successfully.")

    # Convert describe stats to clean serializable dict
    per_feature_stats: Dict[str, Dict[str, float]] = {}
    for feat in UNIVERSAL_CORE_FEATURES:
        s = df[feat]
        per_feature_stats[feat] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "25%": round(float(s.quantile(0.25)), 6),
            "median": round(float(s.median()), 6),
            "75%": round(float(s.quantile(0.75)), 6),
            "max": round(float(s.max()), 6),
        }

    # Per dataset count
    dataset_counts = {str(k): int(v) for k, v in df["source_dataset"].value_counts().items()}
    label_distribution = {str(k): int(v) for k, v in df["label"].value_counts().items()}

    summary: Dict[str, Any] = {
        "table_name": "universal_features",
        "total_rows": len(df),
        "total_unique_ids": df["id"].nunique(),
        "total_columns": len(df.columns),
        "metadata_columns": METADATA_COLUMNS,
        "feature_count": len(UNIVERSAL_CORE_FEATURES),
        "feature_columns": UNIVERSAL_CORE_FEATURES,
        "feature_families": {
            "internal_signals": INTERNAL_SIGNAL_FEATURES,
            "self_consistency": SELF_CONSISTENCY_FEATURES,
            "nli_agreement": NLI_AGREEMENT_FEATURES,
        },
        "missingness_rate": 0.0,
        "near_zero_variance_features": list(near_zero_var.index),
        "exact_collinear_pairs": collinear_pairs,
        "dataset_distribution": dataset_counts,
        "label_distribution": label_distribution,
        "per_feature_statistics": per_feature_stats,
    }

    return summary


def build_and_save_universal_features(
    internal_signals_path: str = "experiments/baselines/supervised_signals_combined.parquet",
    self_consistency_path: str = "experiments/baselines/self_consistency/full_self_consistency_aggregated.parquet",
    nli_agreement_path: str = "experiments/baselines/nli_agreement/full_nli_aggregated.parquet",
    output_dir: str = "experiments/baselines/combined",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load source feature tables, merge into canonical universal feature table, validate, and save.

    Parameters:
        internal_signals_path: Path to supervised_signals_combined.parquet.
        self_consistency_path: Path to full_self_consistency_aggregated.parquet.
        nli_agreement_path: Path to full_nli_aggregated.parquet.
        output_dir: Destination directory for combined outputs.

    Returns:
        Tuple of (df_universal, summary_dict).
    """
    logger.info(f"Loading internal signals from '{internal_signals_path}'...")
    df_internal = pd.read_parquet(internal_signals_path)

    logger.info(f"Loading self-consistency from '{self_consistency_path}'...")
    df_sc = pd.read_parquet(self_consistency_path)

    logger.info(f"Loading NLI agreement from '{nli_agreement_path}'...")
    df_nli = pd.read_parquet(nli_agreement_path)

    # Build combined DataFrame
    df_universal = build_universal_feature_dataframe(
        df_internal=df_internal,
        df_self_consistency=df_sc,
        df_nli=df_nli,
    )

    # Validate against source dataset
    summary = validate_universal_feature_dataframe(
        df=df_universal,
        df_orig=df_internal,
        expected_count=len(df_internal),
    )

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "universal_features.parquet"
    csv_path = out_dir / "universal_features.csv"
    summary_path = out_dir / "universal_features_summary.json"

    logger.info(f"Saving universal feature table to '{parquet_path}'...")
    df_universal.to_parquet(parquet_path, index=False)

    logger.info(f"Saving universal feature table to '{csv_path}'...")
    df_universal.to_csv(csv_path, index=False)

    logger.info(f"Saving summary metadata to '{summary_path}'...")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Universal combined feature table build complete.")
    return df_universal, summary


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build canonical universal combined feature table.")
    parser.add_argument(
        "--internal-data",
        type=str,
        default="experiments/baselines/supervised_signals_combined.parquet",
        help="Path to internal supervised signals parquet.",
    )
    parser.add_argument(
        "--self-consistency-data",
        type=str,
        default="experiments/baselines/self_consistency/full_self_consistency_aggregated.parquet",
        help="Path to self-consistency aggregated parquet.",
    )
    parser.add_argument(
        "--nli-data",
        type=str,
        default="experiments/baselines/nli_agreement/full_nli_aggregated.parquet",
        help="Path to NLI agreement aggregated parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baselines/combined",
        help="Output directory.",
    )

    args = parser.parse_args()

    build_and_save_universal_features(
        internal_signals_path=args.internal_data,
        self_consistency_path=args.self_consistency_data,
        nli_agreement_path=args.nli_data,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
