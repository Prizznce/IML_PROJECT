"""
VeritasLM: Real-Time Neural Hallucination Observability & Defense Platform.
Linear & Railway inspired developer UX architecture.
"""

import datetime
from pathlib import Path
import sys

# Ensure repository root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

from app.assets import ICON_PNG_PATH
from app.styles import apply_custom_styles
from app.config import (
    MODEL_OPTIONS,
    get_model_specs,
    get_hardware_info,
    get_cached_models,
)
from app.inference import run_live_inference
from app.components.dialogs import configure_infrastructure_dialog
from app.components.sidebar import render_sidebar
from app.components.header import render_header, render_infrastructure_strip
from app.components.controls import render_input_controls
from app.components.dashboard import render_dashboard
from app.components.about import render_about_view
from app.components.history import render_history_view
from app.components.benchmark_mode import render_benchmark_mode
from app.components.system_health import render_system_health_view
from app.session import init_session, record_detector_activity

# ==============================================================================
# 1. Page Configuration & Custom Design System
# ==============================================================================

st.set_page_config(
    page_title="VeritasLM · Neural Hallucination Observability Platform",
    page_icon=str(ICON_PNG_PATH) if ICON_PNG_PATH.exists() else "◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_custom_styles()

# ==============================================================================
# 2. Session State & Infrastructure Runtime Resolution
# ==============================================================================

if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = "Qwen 3.5 (0.8B)"
if "threshold_low" not in st.session_state:
    st.session_state["threshold_low"] = 35
if "threshold_high" not in st.session_state:
    st.session_state["threshold_high"] = 65
if "hardware_choice" not in st.session_state:
    st.session_state["hardware_choice"] = "auto"
# Session lifecycle: unique session ID and empty history initialization
init_session()

# Hardware resolution
hardware = get_hardware_info()
cuda_available = hardware["cuda_available"]
gpu_name = hardware["gpu_name"]
gpu_display = hardware["gpu_display"]
cpu_name = hardware["cpu_name"]
cpu_display = hardware["cpu_display"]

hw_choice = str(st.session_state.get("hardware_choice", "auto"))
is_cpu_override = hw_choice.startswith("Host CPU") or "Host CPU" in hw_choice

if cuda_available and not is_cpu_override:
    selected_device = "cuda:0"
    dev_short_label = f"CUDA:0 ({gpu_name})"
    dev_color = "#34D399"
else:
    selected_device = "cpu"
    dev_short_label = f"HOST CPU ({cpu_name})"
    dev_color = "#FBBF24"

selected_model_name = st.session_state["selected_model_name"]
active_model_id = MODEL_OPTIONS.get(selected_model_name, "Qwen/Qwen3.5-0.8B")
threshold_low = st.session_state["threshold_low"]
threshold_high = st.session_state["threshold_high"]
active_specs = get_model_specs(selected_model_name)

# ==============================================================================
# 3. Sidebar (Identity, Navigation, Profile)
# ==============================================================================

render_sidebar()

# ==============================================================================
# 4. View Router & Navigation Dispatch
# ==============================================================================

if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Detector"

active_nav = st.session_state["active_nav"]

# ------------------------------------------------------------------------------
# 1. DETECTOR (Primary Hallucination Observability & Defense Experience)
# ------------------------------------------------------------------------------
if active_nav == "Detector":
    render_header(
        selected_model_name=selected_model_name,
        cuda_available=cuda_available,
        selected_device=selected_device,
        gpu_name=gpu_name,
        cpu_name=cpu_name,
        cpu_display=cpu_display,
        active_specs=active_specs,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        configure_dialog_fn=lambda: configure_infrastructure_dialog(
            cuda_available=cuda_available,
            gpu_display=gpu_display,
            cpu_display=cpu_display,
        ),
    )

    analyze_clicked, prompt_text, context_text, is_evidence_mode = render_input_controls()

    if analyze_clicked:
        clean_p = prompt_text.strip()
        if not clean_p:
            st.warning("Please provide a prompt or select a sample preset before executing.")
        else:
            st.session_state["user_prompt"] = clean_p
            st.session_state["user_context"] = context_text.strip()

            with st.spinner("Executing 15-stage neural inference & consistency pipeline..."):
                try:
                    cached_models = get_cached_models(model_id=active_model_id, target_device=selected_device)
                    result = run_live_inference(
                        prompt=clean_p,
                        context=context_text.strip() if context_text.strip() else None,
                        device=selected_device,
                        causal_model=cached_models["causal_model"],
                        tokenizer=cached_models["tokenizer"],
                        embedding_model=cached_models["embedding_model"],
                        nli_model=cached_models["nli_model"],
                        nli_label_indices=cached_models["nli_label_indices"],
                        lr_pipeline=cached_models["lr_pipeline"],
                        xgb_model=cached_models["xgb_model"],
                        is_evidence_mode=is_evidence_mode,
                        max_new_tokens=int(st.session_state.get("pref_max_tokens", 128)),
                        temperature=float(st.session_state.get("pref_temperature", 0.0)),
                    )
                    st.session_state["inference_result"] = result

                    # Record detector activity into session audit history
                    try:
                        record_detector_activity(
                            prompt=clean_p,
                            result=result,
                            context=context_text.strip() if context_text else None,
                            model_name=selected_model_name,
                            threshold_low=float(threshold_low),
                            threshold_high=float(threshold_high),
                            is_evidence_mode=is_evidence_mode,
                        )
                    except Exception as err:
                        st.warning(f"Session history recording warning: {str(err)}")
                except Exception as exc:
                    st.error(f"Inference execution failed: {str(exc)}")
                    st.session_state["inference_result"] = None

    render_dashboard(
        res=st.session_state.get("inference_result"),
        selected_model_name=selected_model_name,
        active_model_id=active_model_id,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        is_evidence_mode=is_evidence_mode,
    )

# ------------------------------------------------------------------------------
# 2. BENCHMARK MODE (Multi-Model Simultaneous Evaluation & Comparison)
# ------------------------------------------------------------------------------
elif active_nav == "Benchmark Mode":
    render_benchmark_mode(
        cuda_available=cuda_available,
        selected_device=selected_device,
        gpu_name=gpu_name,
        cpu_name=cpu_name,
        cpu_display=cpu_display,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )

# ------------------------------------------------------------------------------
# 3. SYSTEM HEALTH (Neural Runtime, Multi-Model Registry & Epistemic Telemetry)
# ------------------------------------------------------------------------------
elif active_nav == "System Health":
    render_system_health_view(
        selected_model_name=selected_model_name,
        active_model_id=active_model_id,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )

# ------------------------------------------------------------------------------
# 4. HISTORY (Complete Session Detection Audit & Observability Dossier)
# ------------------------------------------------------------------------------
elif active_nav == "History":
    render_history_view()

# ------------------------------------------------------------------------------
# 5. ABOUT (Comprehensive Platform & Empirical Validation Dossier)
# ------------------------------------------------------------------------------
elif active_nav == "About":
    render_about_view()

# ------------------------------------------------------------------------------
# 7. PROFILE (Unimplemented Backend: Clean minimal structure, empty internal content)
# ------------------------------------------------------------------------------
elif active_nav == "Profile":
    st.markdown(
        """
        <div class="linear-header">
            <div class="linear-title-row">
                <div>
                    <h1 class="linear-title">Profile</h1>
                </div>
            </div>
        </div>
        <div class="empty-view-container"></div>
        """,
        unsafe_allow_html=True,
    )

