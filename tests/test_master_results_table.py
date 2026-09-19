"""
Unit tests validating docs/master_results_table.md structure, sections, and key experiment labels.
"""

from pathlib import Path
import pytest


def test_master_results_table_exists_and_complete():
    doc_path = Path("docs/master_results_table.md")
    assert doc_path.exists(), "docs/master_results_table.md does not exist."
    assert doc_path.stat().st_size > 1000, "docs/master_results_table.md is too short."

    content = doc_path.read_text(encoding="utf-8")

    # Verify all 8 required sections are present
    required_sections = [
        "1. Primary Benchmark",
        "2. Feature-Family Ablation",
        "3. Cross-Dataset Generalization",
        "4. Calibration",
        "5. Retrieval-Augmented Standalone Variant",
        "6. Dataset-Level Primary Benchmark",
        "7. Error Analysis Summary",
        "8. Final Experiment Map",
    ]

    for sec in required_sections:
        assert sec in content, f"Missing required section in master table: {sec}"


def test_master_results_table_key_labels_and_notes():
    content = Path("docs/master_results_table.md").read_text(encoding="utf-8")

    # Verify key experiment labels and models
    key_terms = [
        "Phase 14 Universal Core",
        "Phase 6 Baseline",
        "Logistic Regression",
        "XGBoost",
        "All Three / Universal Core",
        "Comparability Note",
        "Standalone Variant",
        "HaluEval",
        "TruthfulQA",
        "FEVER",
        "Brier Score",
        "ECE",
        "ROC-AUC",
        "PR-AUC",
    ]

    for term in key_terms:
        assert term in content, f"Missing key term in master table: {term}"
