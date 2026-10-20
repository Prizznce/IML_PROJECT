"""
Header and Infrastructure Telemetry Strip Components.
"""

from typing import Callable, Dict, Any, Optional
import streamlit as st


def render_header(
    selected_model_name: str,
    cuda_available: bool,
    selected_device: str,
    gpu_name: str,
    cpu_name: str,
    cpu_display: str,
    active_specs: Optional[Dict[str, Any]] = None,
    threshold_low: int = 35,
    threshold_high: int = 65,
    configure_dialog_fn: Optional[Callable[[], None]] = None,
):
    """
    Render Linear-style platform header, telemetry status bar, and hardware alerts.
    """
    if cuda_available and "cuda" in selected_device:
        dev_label = f"CUDA: {gpu_name}"
        dev_badge_class = "live-badge-emerald"
        dev_dot_color = "var(--emerald-text)"
    else:
        dev_label = f"HOST CPU: {cpu_name}"
        dev_badge_class = "live-badge-amber"
        dev_dot_color = "var(--amber-text)"

    p_count = active_specs.get("p_count", "0.8B") if active_specs else "0.8B"
    engine_display = selected_model_name
    if p_count and p_count not in selected_model_name and f"({p_count})" not in selected_model_name:
        import re
        if not re.search(r'\(\s*\d+(\.\d+)?[BMK]?\s*\)', selected_model_name, re.IGNORECASE):
            engine_display = f"{selected_model_name} ({p_count})"

    # Header Top Bar & Title (Compact, space-efficient hierarchy)
    st.markdown(
        f"""
        <div class="linear-header">
            <div class="linear-title-row">
                <div>
                    <h1 class="linear-title">Hallucination Detector</h1>
                    <p class="linear-subtitle">
                        White-box generation dynamics, stochastic self-consistency probing, and cross-encoder agreement without length shortcut bias.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simplified Engine & Runtime Section: Live Status Chips (Row 1) & Config Details (Row 2)
    st_col1, st_col2 = st.columns([4.4, 0.95], gap="small", vertical_alignment="center")

    with st_col1:
        st.markdown(
            f"""
            <div class="telemetry-panel">
                <!-- Primary Row: Live Engine & Hardware Runtime -->
                <div class="telemetry-runtime-row">
                    <div class="telemetry-runtime-chips">
                        <span class="telemetry-kicker">LIVE RUNTIME</span>
                        <span class="live-badge live-badge-emerald" title="Active causal model engine">
                            <span class="pulse-dot"></span>
                            {engine_display}
                        </span>
                        <span class="live-badge {dev_badge_class}" title="Compute execution hardware">
                            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:{dev_dot_color}; margin-right:2px;"></span>
                            {dev_label}
                        </span>
                        <span class="live-badge live-badge-emerald live-badge-status-pill">
                            READY
                        </span>
                    </div>
                </div>
                <!-- Secondary Row: Static Configuration & Calibration Values -->
                <div class="telemetry-config-row">
                    <div class="telemetry-config-chips">
                        <span class="telemetry-kicker-secondary">SYSTEM CONFIG</span>
                        <span class="config-chip" title="Feature extraction taxonomy dimension">
                            TAXONOMY: 19-FEATURE CORE
                        </span>
                        <span class="config-chip" title="Active decision thresholds">
                            RISK THRESHOLDS: &lt;{threshold_low}% / &ge;{threshold_high}%
                        </span>
                        <span class="config-chip" title="Holdout calibration benchmark size">
                            CALIBRATION: N=4,000 HOLDOUT
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st_col2:
        if configure_dialog_fn:
            if st.button("Configure", key="btn_cfg_engine", icon=":material/tune:", help="Switch model, compute device, or adjust risk thresholds", use_container_width=True):
                configure_dialog_fn()

    if not cuda_available:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; background: var(--amber-bg);
                        border: 1px solid var(--amber-border); border-left: 3px solid var(--amber-text);
                        border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; font-size: 12.5px; color: var(--text-primary);">
                <div style="font-family: var(--font-mono); font-size: 10px; padding: 2px 7px; border-radius: 4px; background: var(--amber-border); color: var(--amber-text); font-weight: 700;">
                    CPU RUNTIME
                </div>
                <div style="font-size: 12px; color: var(--text-secondary);">
                    Executing on Host CPU ({cpu_display}). For high-speed GPU throughput, initialize with <code style="color: var(--amber-text);">.\\run_app.bat</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_infrastructure_strip(
    selected_model_name: str,
    active_specs: Dict[str, Any],
    dev_color: str,
    dev_short_label: str,
    threshold_low: int,
    threshold_high: int,
    configure_dialog_fn: Callable[[], None],
):
    """
    Legacy wrapper - infrastructure status is now seamlessly integrated into render_header.
    Retained for modularity.
    """
    pass
