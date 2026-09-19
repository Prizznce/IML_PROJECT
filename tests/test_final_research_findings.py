"""
Unit tests validating docs/final_research_findings.md structure, sections, and scientific neutrality.
"""

from pathlib import Path
import pytest


def test_final_research_findings_exists_and_complete():
    doc_path = Path("docs/final_research_findings.md")
    assert doc_path.exists(), "docs/final_research_findings.md does not exist."
    assert doc_path.stat().st_size > 2000, "docs/final_research_findings.md is too short."

    content = doc_path.read_text(encoding="utf-8")

    # Verify all 12 required sections are present
    required_sections = [
        "1. Research Objective",
        "2. Experimental Setup",
        "3. Primary Findings",
        "4. Feature-Family Findings",
        "5. Cross-Dataset Generalization",
        "6. Calibration Findings",
        "7. Error-Analysis Findings",
        "8. Retrieval Findings",
        "9. What the Experiments Demonstrate",
        "10. What the Experiments Do NOT Demonstrate",
        "11. Threats to Validity",
        "12. Conclusion",
    ]

    for sec in required_sections:
        assert sec in content, f"Missing required section in findings document: {sec}"


def test_final_research_findings_key_content_and_neutrality():
    content = Path("docs/final_research_findings.md").read_text(encoding="utf-8")

    # Verify key models, datasets, and concepts are documented
    key_terms = [
        "Qwen3.5-0.8B",
        "HaluEval",
        "TruthfulQA",
        "FEVER",
        "Phase 6 Baseline",
        "Phase 14 Universal Core",
        "Logistic Regression",
        "XGBoost",
        "ROC-AUC",
        "PR-AUC",
        "Brier Score",
        "Expected Calibration Error",
        "Platt",
        "Isotonic",
    ]

    for term in key_terms:
        assert term in content, f"Missing required concept or term: {term}"

    # Verify absence of unscientific competitive framing
    prohibited_terms = [
        "best model",
        "clear winner",
        "optimal classifier",
    ]
    for pterm in prohibited_terms:
        assert pterm not in content.lower(), f"Found unscientific competitive phrasing: {pterm}"
