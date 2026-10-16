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

    # Header Top Bar & Title
    st.markdown(
        f"""
        <div class="linear-header">
            <div class="linear-breadcrumbs">
                <span>VERITAS</span> / <span>NEURAL OBSERVABILITY</span> / <span class="active">HALLUCINATION-DEFENSE v2.4</span>
            </div>
            <div class="linear-title-row">
                <div>
                    <h1 class="linear-title">Neural Hallucination Observability</h1>
                    <p class="linear-subtitle" style="margin-top: 4px; margin-bottom: 0;">
                        White-box generation dynamics, stochastic self-consistency probing, and cross-encoder agreement without length shortcut bias.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Consolidated Infrastructure & Telemetry Status Strip (Semantically Grouped)
    st_col1, st_col2 = st.columns([4.2, 1.2], gap="small")

    with st_col1:
        st.markdown(
            f"""
            <div class="status-bar-container" style="margin-bottom: 14px;">
                <div class="status-group">
                    <span class="status-group-label">ENGINE &amp; RUNTIME:</span>
                    <span class="live-badge live-badge-emerald">
                        <span class="pulse-dot"></span>
                        {engine_display}
                    </span>
                    <span class="live-badge {dev_badge_class}">
                        <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:{dev_dot_color}; margin-right:2px;"></span>
                        {dev_label}
                    </span>
                </div>
                <div class="status-divider"></div>
                <div class="status-group">
                    <span class="status-group-label">CONFIG:</span>
                    <span class="live-badge live-badge-primary">
                        19-FEATURE CORE
                    </span>
                    <span class="live-badge live-badge-primary">
                        &lt;{threshold_low}% / &ge;{threshold_high}%
                    </span>
                </div>
                <div class="status-divider"></div>
                <div class="status-group">
                    <span class="status-group-label">CALIBRATION:</span>
                    <span class="live-badge live-badge-zinc">
                        N=4,000 HOLDOUT
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st_col2:
        if configure_dialog_fn:
            if st.button("Configure Engine", width="stretch", help="Switch model, compute device, or adjust risk thresholds"):
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
