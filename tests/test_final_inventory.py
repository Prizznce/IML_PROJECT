"""
Unit tests validating the final experiment inventory document and repository artifact integrity.
"""

import os
from pathlib import Path
import pytest


def test_final_experiment_inventory_exists_and_complete():
    doc_path = Path("docs/final_experiment_inventory.md")
    assert doc_path.exists(), "docs/final_experiment_inventory.md does not exist."
    assert doc_path.stat().st_size > 1000, "docs/final_experiment_inventory.md is too short."

    content = doc_path.read_text(encoding="utf-8")

    # Verify all 10 required sections are present
    required_sections = [
        "1. Executive Status",
        "2. Comprehensive Experiment Inventory Table",
        "3. Primary Reference Experiment Definition",
        "4. Dataset Inventory & Provenance",
        "5. Model Inventory",
        "6. Metric Inventory Across Experimental Areas",
        "7. Comparability Notes",
        "8. Methodological Caveats & Limitations",
        "9. Missing or Unfinished Items",
        "10. Recommended Source Files for the Final Technical Report",
    ]

    for sec in required_sections:
        assert sec in content, f"Missing required section: {sec}"


def test_inventory_referenced_artifacts_exist():
    # Core artifacts referenced across all 11 experiment areas
    core_artifacts = [
        "data/processed/combined_processed.parquet",
        "data/processed/halueval_processed.parquet",
        "data/processed/truthfulqa_processed.parquet",
        "data/processed/fever_processed.parquet",
        "experiments/baselines/supervised_signals_combined.parquet",
        "experiments/baselines/results/baseline_metrics.json",
        "experiments/baselines/self_consistency/full_generations_raw.parquet",
        "experiments/baselines/self_consistency/full_self_consistency_aggregated.parquet",
        "experiments/baselines/nli_agreement/full_nli_pairwise.parquet",
        "experiments/baselines/nli_agreement/full_nli_aggregated.parquet",
        "experiments/baselines/combined/universal_features.parquet",
        "experiments/baselines/combined/combined_test_predictions.parquet",
        "experiments/baselines/combined/combined_model_results.json",
        "experiments/ablation/ablation_results.csv",
        "experiments/cross_dataset/cross_dataset_results.csv",
        "experiments/calibration/calibration_results.csv",
        "experiments/error_analysis/error_analysis_results.json",
        "experiments/retrieval_augmented/retrieval_experiment_results.csv",
        "experiments/retrieval_augmented/retrieval_corpus_manifest.json",
    ]

    for art in core_artifacts:
        p = Path(art)
        assert p.exists(), f"Core artifact missing from repository: {art}"
        assert p.stat().st_size > 0, f"Core artifact is empty: {art}"
