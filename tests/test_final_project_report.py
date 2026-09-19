"""
Unit tests validating docs/final_project_report.md structure, sections, content, and citation integrity.
"""

from pathlib import Path
import pytest


def test_final_project_report_exists_and_complete():
    doc_path = Path("docs/final_project_report.md")
    assert doc_path.exists(), "docs/final_project_report.md does not exist."
    assert doc_path.stat().st_size > 5000, "docs/final_project_report.md is too short."

    content = doc_path.read_text(encoding="utf-8")

    # Verify project title
    expected_title = "Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals"
    assert expected_title in content, f"Missing project title: {expected_title}"

    # Verify key sections exist
    required_sections = [
        "## 2. Abstract",
        "## 3. Introduction",
        "## 4. Problem Statement",
        "## 5. Objectives",
        "## 6. Related Work / Background",
        "## 7. Dataset Description",
        "## 8. System Architecture",
        "## 9. Internal Generation Signals",
        "## 10. Self-Consistency",
        "## 11. NLI Agreement",
        "## 12. Retrieval-Augmented Variant",
        "## 13. Machine Learning Models",
        "## 14. Evaluation Metrics",
        "## 15. Experimental Results",
        "## 16. Error Analysis",
        "## 17. Discussion",
        "## 18. Calibration Discussion",
        "## 19. Cross-Dataset Generalization Discussion",
        "## 20. Limitations",
        "## 21. Threats to Validity",
        "## 22. Future Work",
        "## 23. Conclusion",
        "## 24. Reproducibility",
        "## 25. References",
        "## 26. Appendix",
    ]

    for sec in required_sections:
        assert sec in content, f"Missing required section in final report: {sec}"


def test_final_project_report_key_entities_and_citations():
    content = Path("docs/final_project_report.md").read_text(encoding="utf-8")

    # Verify required models and datasets
    key_entities = [
        "Qwen3.5-0.8B",
        "HaluEval",
        "TruthfulQA",
        "FEVER",
        "Logistic Regression",
        "XGBoost",
    ]

    for entity in key_entities:
        assert entity in content, f"Missing key entity in final report: {entity}"

    # Verify team members
    team_members = [
        "Kunj Patil",
        "Jaiprakash",
        "Prince",
        "Aditya Kumar",
    ]
    for member in team_members:
        assert member in content, f"Missing team member: {member}"

    # Verify absence of fabricated citations and presence of disclaimer
    assert "Bibliographic references to be added from the project's verified literature sources." in content
    assert "[REFERENCE NEEDED]" in content

    # Verify absence of competitive superlatives
    prohibited_phrases = [
        "best model",
        "clear winner",
        "optimal classifier",
    ]
    for phrase in prohibited_phrases:
        assert phrase not in content.lower(), f"Found unscientific competitive phrasing: {phrase}"


def test_final_report_consistency_audit_exists():
    audit_path = Path("docs/final_report_consistency_audit.md")
    assert audit_path.exists(), "docs/final_report_consistency_audit.md does not exist."
    assert audit_path.stat().st_size > 2000, "docs/final_report_consistency_audit.md is too short."

    content = audit_path.read_text(encoding="utf-8")
    assert "Phase 20 — Step 5: Final Technical Consistency Audit of Academic Report" in content
    assert "Qwen/Qwen3.5-0.8B" in content
    assert "PASSED WITH MINOR FACTUAL CORRECTIONS" in content
    assert "Zero Metric Alteration" in content
    assert "Zero Experiment Re-execution" in content

