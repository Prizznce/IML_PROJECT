"""
Unit tests for VeritasLM System Health & Neural Pipeline Telemetry Component.

Verifies:
1. Accurate compute engine reflection (CUDA vs Host CPU).
2. Setting engine to Host CPU in detector session state reflects everywhere in System Health.
3. Live status badge, Executive Telemetry Ribbon, and Pillar 1 update accurately based on active engine.
4. Action buttons (Purge Tensor Cache, Reset Peak Watermark, Run Diagnostic Self-Test, Configure Engine) execute properly without errors.
5. Epistemic Watchdog telemetry (risk distribution, entropy collapse, dynamic probe agreement rate).
6. Diagnostics self-test evaluates on the active target compute device.
"""

from unittest.mock import patch, MagicMock
import pytest
import streamlit as st
import torch

from app.session import init_session, record_detector_activity
from app.components.system_health import render_system_health_view


@pytest.fixture(autouse=True)
def clean_streamlit_session():
    """Ensure clean session state for every test."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    yield
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def test_system_health_reflects_cpu_engine_when_selected_explicitly():
    """Verify that passing selected_device='cpu' renders Host CPU across header, ribbon, and pillars."""
    init_session()
    markdown_outputs = []

    with patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False):
        render_system_health_view(
            selected_device="cpu",
            cuda_available=True,
            gpu_name="NVIDIA GeForce RTX 4060",
            cpu_name="13th Gen Intel Core i7",
            cpu_display="13th Gen Intel Core i7 (24 vCPUs)",
        )

    full_output = " ".join(markdown_outputs)

    # Header badge must reflect HOST CPU
    assert "HOST CPU (13th Gen Intel Core i7)" in full_output
    assert "var(--amber-text)" in full_output

    # Ribbon must reflect Host Memory Mode & CPU
    assert "HOST MEMORY" in full_output
    assert "Host Float32 Vectorized Engine" in full_output

    # Pillar 1 must show CPU hardware and Float32 policy
    assert "Single Precision (Float32 Native / CPU SIMD Vectorized)" in full_output
    assert "Host CPU Processor (13th Gen Intel Core i7 (24 vCPUs))" in full_output

    # Pillar 2 Primary Generator must show HOST CPU target and official Qwen architecture
    assert "HOST CPU (13th Gen Intel Core i7)" in full_output
    assert "Qwen3_5ForConditionalGeneration" in full_output


def test_qwen_official_architecture_label():
    """Verify that Qwen 3.5 official architecture label is Qwen3_5ForConditionalGeneration."""
    from app.config import get_model_specs
    specs = get_model_specs("Qwen 3.5 (0.8B)")
    assert "Qwen3_5ForConditionalGeneration" in specs["arch"]

    markdown_outputs = []
    with patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False):
        render_system_health_view(selected_model_name="Qwen 3.5 (0.8B)")

    full_output = " ".join(markdown_outputs)
    assert "Qwen3_5ForConditionalGeneration" in full_output


def test_system_health_reflects_cpu_engine_from_detector_session_state():
    """Verify that when user selects Host CPU in Detector via hardware_choice, System Health auto-reflects CPU."""
    init_session()
    st.session_state["hardware_choice"] = "Host CPU: 13th Gen Intel Core i7 (Fallback)"

    markdown_outputs = []
    with patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False), \
         patch("app.components.system_health.get_hardware_info", return_value={
             "cuda_available": True,
             "gpu_name": "NVIDIA GeForce RTX 4060",
             "gpu_display": "NVIDIA GeForce RTX 4060 (8.0 GB)",
             "cpu_name": "13th Gen Intel Core i7",
             "cpu_display": "13th Gen Intel Core i7 (24 vCPUs)",
         }):
        # Called with no hardware args (as from navigation router)
        render_system_health_view()

    full_output = " ".join(markdown_outputs)

    # Must reflect CPU despite cuda_available being True
    assert "HOST CPU (13th Gen Intel Core i7)" in full_output
    assert "HOST MEMORY" in full_output
    assert "Host Float32 Vectorized Engine" in full_output


def test_system_health_reflects_cuda_engine_when_selected():
    """Verify that when CUDA engine is active, System Health displays CUDA details."""
    init_session()
    markdown_outputs = []

    with patch("torch.cuda.is_available", return_value=True), \
         patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False):
        render_system_health_view(
            selected_device="cuda:0",
            cuda_available=True,
            gpu_name="NVIDIA GeForce RTX 4060",
            gpu_display="NVIDIA GeForce RTX 4060 (8.0 GB)",
            cpu_name="13th Gen Intel Core i7",
            cpu_display="13th Gen Intel Core i7 (24 vCPUs)",
        )

    full_output = " ".join(markdown_outputs)

    assert "CUDA:0 (NVIDIA GeForce RTX 4060)" in full_output
    assert "var(--emerald-text)" in full_output
    assert "TENSOR VRAM" in full_output
    assert "Autocast BF16 / FP16 Native" in full_output
    assert "Dynamic Mixed Precision (bfloat16 / float16 with float32 fallbacks)" in full_output


def test_system_health_diagnostics_executes_on_cpu_when_cpu_active():
    """Verify diagnostics tool tests on CPU and records HOST CPU label when CPU engine is active."""
    init_session()
    st.session_state.pop("health_diag_results", None)

    markdown_outputs = []
    with patch("streamlit.button", return_value=True), \
         patch("streamlit.spinner") as mock_spinner, \
         patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]):
        render_system_health_view(
            selected_device="cpu",
            cuda_available=True,
            gpu_name="NVIDIA GeForce RTX 4060",
            cpu_name="13th Gen Intel Core i7",
        )

    results = st.session_state.get("health_diag_results", {})
    assert results.get("all_passed") is True
    assert results.get("t1_pass") is True
    assert "HOST CPU (13th Gen Intel Core i7)" in results.get("device_label", "")

    full_output = " ".join(markdown_outputs)
    assert "Verified tensor computation on HOST CPU (13th Gen Intel Core i7)" in full_output


def test_purge_tensor_cache_button_action():
    """Verify Purge Tensor Cache button triggers garbage collection and shows appropriate toast."""
    init_session()

    # Case 1: CPU mode purge
    button_mock = MagicMock(side_effect=lambda label, **kw: label == "Purge Tensor Cache")
    toast_calls = []

    with patch("streamlit.button", button_mock), \
         patch("streamlit.toast", side_effect=lambda msg, **kw: toast_calls.append(msg)), \
         patch("streamlit.rerun") as mock_rerun, \
         patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]):
        render_system_health_view(selected_device="cpu")

    assert mock_rerun.called
    assert any("Host memory reclaimed" in call for call in toast_calls)


def test_reset_peak_watermark_button_action():
    """Verify Reset Peak Watermark button triggers reset and shows toast."""
    init_session()

    button_mock = MagicMock(side_effect=lambda label, **kw: label == "Reset Peak Watermark")
    toast_calls = []

    with patch("streamlit.button", button_mock), \
         patch("streamlit.toast", side_effect=lambda msg, **kw: toast_calls.append(msg)), \
         patch("streamlit.rerun") as mock_rerun, \
         patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]):
        render_system_health_view(selected_device="cpu")

    assert mock_rerun.called
    assert any("Peak tensor watermark reset" in call for call in toast_calls)


def test_no_configure_engine_button_in_system_health():
    """Verify that Configure Engine button is not rendered in System Health as per design requirements."""
    init_session()
    button_labels = []

    with patch("streamlit.button", side_effect=lambda label, **kw: button_labels.append(label) or False), \
         patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]):
        render_system_health_view()

    assert "Configure Engine" not in button_labels


def test_first_card_in_section_3_renders_clean_html_without_raw_code_blocks():
    """Verify the first card in Section 3 (Risk Distribution) renders pure HTML without raw markup leaking."""
    import markdown_it
    init_session()
    markdown_outputs = []

    with patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False):
        render_system_health_view()

    # Find the markdown output corresponding to the risk distribution card
    dist_card_md = next(m for m in markdown_outputs if "Session Hallucination Risk Distribution" in m)

    # Render through markdown-it (Streamlit's internal renderer)
    rendered_html = markdown_it.MarkdownIt().render(dist_card_md)

    # Assert no code block tags are produced
    assert "<pre>" not in rendered_html
    assert "<code>" not in rendered_html
    assert "&lt;div" not in rendered_html
    assert "<div class=\"health-dist-bar-container\"" in rendered_html


def test_risk_distribution_empty_state_and_populated_state():
    """Verify risk distribution renders clear empty placeholder when 0 audits, and segments when populated."""
    init_session()

    # Empty state: 0 audits
    markdown_outputs_empty = []
    with patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs_empty.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False):
        render_system_health_view()

    empty_output = " ".join(markdown_outputs_empty)
    assert "No session audits recorded yet" in empty_output

    # Populated state: 2 audits
    record_detector_activity(prompt="Test 1", result={
        "predicted_risk_xgb": 0.1,
        "predicted_risk_lr": 0.1,
        "total_latency_ms": 100.0,
    }, threshold_low=35.0, threshold_high=65.0)

    record_detector_activity(prompt="Test 2", result={
        "predicted_risk_xgb": 0.8,
        "predicted_risk_lr": 0.8,
        "total_latency_ms": 120.0,
    }, threshold_low=35.0, threshold_high=65.0)

    markdown_outputs_pop = []
    with patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs_pop.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False):
        render_system_health_view()

    pop_output = " ".join(markdown_outputs_pop)
    assert "2 SAMPLES" in pop_output
    assert "Faithful" in pop_output
    assert "1 (50.0%)" in pop_output


def test_probe_agreement_rate_calculation():
    """Verify probe agreement rate displays dynamically when session audits exist."""
    init_session()

    # 1 agreeing audit, 1 disagreeing audit
    record_detector_activity(prompt="Audit 1 - Agree", result={
        "predicted_risk_xgb": 0.1,  # Faithful
        "predicted_risk_lr": 0.2,   # Faithful
        "total_latency_ms": 80.0,
    }, threshold_low=35.0, threshold_high=65.0)

    record_detector_activity(prompt="Audit 2 - Disagree", result={
        "predicted_risk_xgb": 0.1,  # Faithful
        "predicted_risk_lr": 0.75,  # High Risk
        "total_latency_ms": 90.0,
    }, threshold_low=35.0, threshold_high=65.0)

    markdown_outputs = []
    with patch("streamlit.markdown", side_effect=lambda text, **kw: markdown_outputs.append(str(text))), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else spec)]), \
         patch("streamlit.button", return_value=False):
        render_system_health_view()

    full_output = " ".join(markdown_outputs)
    # 1 out of 2 agree -> 50.0%
    assert "50.0% (1/2 Session Audits)" in full_output
