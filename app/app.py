"""
VERITAS: Real-Time Neural Hallucination Observability & Defense Platform.
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
from app.components.benchmarks import render_benchmark_proof

# ==============================================================================
# 1. Page Configuration & Custom Design System
# ==============================================================================

st.set_page_config(
    page_title="VERITAS · Neural Hallucination Observability Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme_choice" not in st.session_state:
    st.session_state["theme_choice"] = "Dark Mode"

apply_custom_styles(st.session_state.get("theme_choice", "Dark Mode"))

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
if "history" not in st.session_state:
    st.session_state["history"] = [
        {
            "prompt": "Who was the first person to walk on the Moon?",
            "full_prompt": "Who was the first person to walk on the Moon?",
            "full_context": "",
            "model": "Qwen 3.5 (0.8B)",
            "risk_pct": 8.4,
            "tier": "Faithful",
            "time": "10:14 AM",
        },
        {
            "prompt": "Who was the first human to walk on Mars?",
            "full_prompt": "Who was the first human to walk on Mars?",
            "full_context": "",
            "model": "Qwen 3.5 (0.8B)",
            "risk_pct": 79.2,
            "tier": "High Risk",
            "time": "10:22 AM",
        },
    ]

# Hardware resolution
hardware = get_hardware_info()
cuda_available = hardware["cuda_available"]
gpu_name = hardware["gpu_name"]
gpu_display = hardware["gpu_display"]
cpu_name = hardware["cpu_name"]
cpu_display = hardware["cpu_display"]

if cuda_available and st.session_state.get("hardware_choice") != "Host CPU (Fallback)":
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
# 3. Sidebar (Identity, Audit History, Theme Toggle)
# ==============================================================================

render_sidebar()

# ==============================================================================
# 4. Main Canvas: Header & Infrastructure Quick Strip
# ==============================================================================

render_header(
    selected_model_name=selected_model_name,
    cuda_available=cuda_available,
    selected_device=selected_device,
    gpu_name=gpu_name,
    cpu_name=cpu_name,
    cpu_display=cpu_display,
)

render_infrastructure_strip(
    selected_model_name=selected_model_name,
    active_specs=active_specs,
    dev_color=dev_color,
    dev_short_label=dev_short_label,
    threshold_low=threshold_low,
    threshold_high=threshold_high,
    configure_dialog_fn=lambda: configure_infrastructure_dialog(
        cuda_available=cuda_available,
        gpu_display=gpu_display,
        cpu_display=cpu_display,
    ),
)

# ==============================================================================
# 5. Interactive Input & Action Bar (Run Button + Engine Popover + Export Format)
# ==============================================================================

analyze_clicked, prompt_text, context_text, is_evidence_mode = render_input_controls()

# ==============================================================================
# 6. Live Pipeline Execution
# ==============================================================================

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

                # Log to session audit history for sidebar tracking
                try:
                    xgb_val = float(result.get("predicted_risk_xgb", result.get("xgb_hallucination_probability", 0.0)) or 0.0)
                    pct_val = round(xgb_val * 100.0, 1)
                    tier_label = (
                        "Faithful"
                        if pct_val < float(threshold_low)
                        else ("High Risk" if pct_val >= float(threshold_high) else "Indeterminate")
                    )
                    new_hist = {
                        "prompt": clean_p[:45] + ("..." if len(clean_p) > 45 else ""),
                        "full_prompt": clean_p,
                        "full_context": context_text.strip() if context_text else "",
                        "model": selected_model_name,
                        "risk_pct": pct_val,
                        "tier": tier_label,
                        "time": datetime.datetime.now().strftime("%I:%M %p"),
                    }
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].insert(0, new_hist)
                    st.session_state["history"] = st.session_state["history"][:20]
                except Exception:
                    pass
            except Exception as exc:
                st.error(f"Inference execution failed: {str(exc)}")
                st.session_state["inference_result"] = None

# ==============================================================================
# 7. Telemetry HUD & Results Dashboard
# ==============================================================================

render_dashboard(
    res=st.session_state.get("inference_result"),
    selected_model_name=selected_model_name,
    active_model_id=active_model_id,
    threshold_low=threshold_low,
    threshold_high=threshold_high,
    is_evidence_mode=is_evidence_mode,
)

# ==============================================================================
# 8. Empirical Model Validation & Multi-Task Benchmark Proof
# ==============================================================================

render_benchmark_proof()
