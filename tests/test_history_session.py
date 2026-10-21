"""
Unit tests for VERITAS Session & Audit History Subsystem.

Tests:
1. Session Initialization: Unique session ID generation and empty history list initialization.
2. Detector Activity Recording: Storing prompt, context, result, confidence score, risk percentage, timestamp.
3. Session Lifecycle Controls: History clearance, record deletion, and session resets.
4. Export Integrity: JSON and CSV exports.
5. UI Component Rendering: History view and sidebar integration.
"""

import json
from unittest.mock import patch, MagicMock
import pytest
import streamlit as st

from app.session import (
    init_session,
    get_session_id,
    get_session_history,
    record_detector_activity,
    clear_session_history,
    reset_session,
    delete_history_record,
    get_session_analytics,
    export_history_to_json,
    export_history_to_csv,
    calculate_confidence_score,
)
from app.components.history import render_history_view
from app.components.sidebar import render_sidebar


@pytest.fixture(autouse=True)
def clean_streamlit_session():
    """Ensure clean session state for every test."""
    # Clear session state keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    yield
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def test_init_session_creates_unique_id_and_empty_history():
    """Verify init_session creates a unique session ID and initializes an empty history list."""
    session_id = init_session()
    assert session_id.startswith("SES-"), f"Session ID should start with SES-, got {session_id}"
    assert len(session_id) >= 12
    assert "history" in st.session_state
    assert isinstance(st.session_state["history"], list)
    assert len(st.session_state["history"]) == 0


def test_init_session_preserves_existing_session():
    """Verify init_session preserves existing session unless forced."""
    first_id = init_session()
    second_id = init_session()
    assert first_id == second_id


def test_reset_session_creates_new_id_and_clears_history():
    """Verify reset_session creates a fresh session ID and clears history."""
    old_id = init_session()
    st.session_state["history"].append({"test": "data"})
    assert len(st.session_state["history"]) == 1

    new_id = reset_session()
    assert new_id != old_id
    assert new_id.startswith("SES-")
    assert len(st.session_state["history"]) == 0


def test_confidence_score_calculation():
    """Verify confidence score is high when risk is extreme, and lower near decision boundary."""
    # Near 0.0 (certainly faithful) -> high confidence
    conf_low_risk = calculate_confidence_score(0.02)
    assert conf_low_risk > 90.0

    # Near 1.0 (certainly hallucinated) -> high confidence
    conf_high_risk = calculate_confidence_score(0.98)
    assert conf_high_risk > 90.0

    # Near 0.5 (uncertain/decision boundary) -> lower confidence
    conf_uncertain = calculate_confidence_score(0.50)
    assert 50.0 <= conf_uncertain <= 55.0


def test_record_detector_activity():
    """Verify recording detector activity appends input, result, confidence score, and timestamp."""
    init_session()
    mock_inference_result = {
        "predicted_risk_xgb": 0.084,
        "predicted_risk_lr": 0.12,
        "generated_response": "Apollo 11 was the American spaceflight that first landed humans on the Moon.",
        "internal_signals": {
            "num_tokens": 42,
            "perplexity": 3.14,
        },
        "total_latency_ms": 115.4,
    }

    rec = record_detector_activity(
        prompt="Who landed on the Moon?",
        context="Apollo 11 landed Neil Armstrong and Buzz Aldrin.",
        result=mock_inference_result,
        model_name="Qwen 3.5 (0.8B)",
        threshold_low=35.0,
        threshold_high=65.0,
        is_evidence_mode=True,
    )

    history = get_session_history()
    assert len(history) == 1
    assert rec["id"].startswith("REC-")
    assert rec["input"]["prompt"] == "Who landed on the Moon?"
    assert rec["input"]["context"] == "Apollo 11 landed Neil Armstrong and Buzz Aldrin."
    assert rec["input"]["is_evidence_mode"] is True

    # Result checks
    assert rec["result"]["tier"] == "Faithful"
    assert rec["xgb_risk_pct"] == 8.4
    assert rec["lr_risk_pct"] == 12.0
    assert rec["combined_risk_pct"] == 10.2
    assert rec["result"]["xgb_risk_pct"] == 8.4
    assert rec["result"]["lr_risk_pct"] == 12.0
    assert rec["result"]["combined_risk_pct"] == 10.2
    assert rec["confidence_score"] > 80.0
    assert "timestamp" in rec
    assert rec["model"] == "Qwen 3.5 (0.8B)"

    # Telemetry checks
    assert rec["telemetry"]["tokens"] == 42
    assert rec["telemetry"]["latency_ms"] == 115.4


def test_record_detector_activity_high_risk():
    """Verify recording high risk detection correctly tags High Risk tier."""
    init_session()
    mock_result = {
        "predicted_risk_xgb": 0.825,
        "predicted_risk_lr": 0.80,
        "generated_response": "Humans walked on Mars in 1985.",
        "internal_signals": {"num_tokens": 20, "perplexity": 18.2},
        "total_latency_ms": 95.0,
    }

    rec = record_detector_activity(
        prompt="When did humans land on Mars?",
        result=mock_result,
        threshold_low=35.0,
        threshold_high=65.0,
    )

    assert rec["tier"] == "High Risk"
    assert rec["xgb_risk_pct"] == 82.5
    assert rec["lr_risk_pct"] == 80.0
    assert rec["combined_risk_pct"] == 81.2
    assert rec["result"]["tier_color"] == "rose"


def test_delete_history_record_and_clear_history():
    """Verify deleting a specific record and clearing all history."""
    init_session()
    mock_res = {"predicted_risk_xgb": 0.1, "total_latency_ms": 50.0}
    r1 = record_detector_activity(prompt="Query 1", result=mock_res)
    r2 = record_detector_activity(prompt="Query 2", result=mock_res)

    assert len(get_session_history()) == 2
    deleted = delete_history_record(r1["id"])
    assert deleted is True
    assert len(get_session_history()) == 1
    assert get_session_history()[0]["id"] == r2["id"]

    clear_session_history()
    assert len(get_session_history()) == 0


def test_session_analytics_calculation():
    """Verify get_session_analytics aggregates metrics accurately."""
    init_session()
    # 1 faithful, 1 high risk
    record_detector_activity(
        prompt="Q1",
        result={
            "predicted_risk_xgb": 0.10,
            "predicted_risk_lr": 0.10,
            "total_latency_ms": 100.0,
            "internal_signals": {"num_tokens": 10},
        },
    )
    record_detector_activity(
        prompt="Q2",
        result={
            "predicted_risk_xgb": 0.90,
            "predicted_risk_lr": 0.90,
            "total_latency_ms": 200.0,
            "internal_signals": {"num_tokens": 20},
        },
    )

    analytics = get_session_analytics()
    assert analytics["total_checks"] == 2
    assert analytics["faithful_count"] == 1
    assert analytics["high_risk_count"] == 1
    assert analytics["faithful_pct"] == 50.0
    assert analytics["high_risk_pct"] == 50.0
    assert analytics["avg_risk_pct"] == 50.0
    assert analytics["avg_latency_ms"] == 150.0


def test_export_history_json_and_csv():
    """Verify JSON and CSV exports contain session ID and records."""
    init_session()
    record_detector_activity(
        prompt="Export prompt test",
        result={"predicted_risk_xgb": 0.25, "total_latency_ms": 120.0},
    )

    json_str = export_history_to_json()
    payload = json.loads(json_str)
    assert payload["session_id"].startswith("SES-")
    assert payload["total_records"] == 1
    assert payload["records"][0]["prompt"].startswith("Export prompt test")

    csv_str = export_history_to_csv()
    assert "Record ID,Session ID,Timestamp" in csv_str
    assert "Combined Risk Pct" in csv_str
    assert "XGB Risk Pct" in csv_str
    assert "LR Risk Pct" in csv_str
    assert "Export prompt test" in csv_str


def test_render_history_view_empty_and_populated():
    """Verify render_history_view executes without exceptions."""
    init_session()
    def mock_columns(spec, *args, **kwargs):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [MagicMock() for _ in range(n)]

    # 1. Test empty state rendering
    with patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=mock_columns), \
         patch("streamlit.button", return_value=False):
        render_history_view()

    # 2. Test populated state rendering
    record_detector_activity(
        prompt="Test Query",
        result={"predicted_risk_xgb": 0.15, "total_latency_ms": 80.0, "internal_signals": {"num_tokens": 15}},
    )
    with patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=mock_columns), \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.selectbox", return_value="All Tiers"), \
         patch("streamlit.download_button"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.container"), \
         patch("streamlit.expander"):
        render_history_view()


def test_render_sidebar_renders_clean_navigation():
    """Verify render_sidebar renders clean navigation system without intrusive preview blocks."""
    rendered_markdown = []
    def mock_md(content, **kwargs):
        rendered_markdown.append(content)

    init_session()
    with patch("streamlit.sidebar"), \
         patch("streamlit.markdown", side_effect=mock_md), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.toggle"):
        render_sidebar()

    all_content = " ".join(rendered_markdown)
    assert "VeritasLM" in all_content or "VERITAS" in all_content
    assert "NAVIGATION" in all_content
    # Ensure extra preview box is not present in sidebar
    assert "sidebar-history-scroll-box" not in all_content
    assert "sidebar-history-empty" not in all_content


def test_record_benchmark_activity_all_failed_tier():
    """Verify that a benchmark where all models encountered errors is marked as Failed, not Faithful."""
    from app.session import record_benchmark_activity

    init_session()
    failed_benchmarks = {
        "Qwen 3.5 (0.8B)": {"error": "CUDA out of memory"},
        "Llama 3.2 (1B Instruct)": {"error": "Gated repo 403 Forbidden"},
    }
    rec = record_benchmark_activity(
        prompt="Test Prompt",
        benchmark_results=failed_benchmarks,
    )
    assert rec["tier"] == "Failed", f"Expected 'Failed' for all-error benchmark, got {rec['tier']}"
    assert rec["tier_color"] == "rose"
    assert rec["best_model"] == "N/A"


def test_history_load_detector_synchronizes_input_fields():
    """Verify loading a history card into Detector synchronizes all input field session keys."""
    from app.components.history import _render_history_record_card

    init_session()
    # Populate existing fields with old text
    st.session_state["main_prompt_input_field"] = "OLD_PROMPT"
    st.session_state["main_context_input_field"] = "OLD_CONTEXT"

    rec = record_detector_activity(
        prompt="NEW_LOADED_PROMPT",
        context="NEW_LOADED_CONTEXT",
        result={"predicted_risk_xgb": 0.25, "total_latency_ms": 100.0, "internal_signals": {"num_tokens": 10}},
    )

    # Simulate clicking "Load Query into Detector"
    def mock_button(label, **kwargs):
        if label == "Load Query into Detector":
            return True
        return False

    with patch("streamlit.markdown"), \
         patch("streamlit.container"), \
         patch("streamlit.expander"), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.code"), \
         patch("streamlit.caption"), \
         patch("streamlit.button", side_effect=mock_button), \
         patch("streamlit.rerun"):
        _render_history_record_card(rec, 0)

    assert st.session_state["user_prompt"] == "NEW_LOADED_PROMPT"
    assert st.session_state["main_prompt_input_field"] == "NEW_LOADED_PROMPT"
    assert st.session_state["user_context"] == "NEW_LOADED_CONTEXT"
    assert st.session_state["main_context_input_field"] == "NEW_LOADED_CONTEXT"
    assert st.session_state["active_nav"] == "Detector"

