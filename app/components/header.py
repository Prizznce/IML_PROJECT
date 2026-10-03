"""
Header and Infrastructure Strip Components.
"""

from typing import Callable, Dict, Any
import streamlit as st


def render_header(
    selected_model_name: str,
    cuda_available: bool,
    selected_device: str,
    gpu_name: str,
    cpu_name: str,
    cpu_display: str,
):
    """
    Render Linear-style platform header, telemetry status bar, and hardware alerts.
    """
    if cuda_available and "cuda" in selected_device:
        dev_label = f"CUDA: {gpu_name}"
        dev_badge_class = "live-badge-emerald"
    else:
        dev_label = f"HOST CPU: {cpu_name}"
        dev_badge_class = "live-badge-zinc"

    st.markdown(
        f"""
        <div class="linear-header">
            <div class="linear-breadcrumbs">
                <span>VERITAS ENTERPRISE</span> / <span>OBSERVABILITY PLATFORM</span> / <span class="active">HALLUCINATION-ENGINE v2.4-PROD</span>
            </div>
            <div class="linear-title-row">
                <h1 class="linear-title">Catching an LLM Lying</h1>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="live-badge live-badge-emerald" style="font-size: 11px;">
                        <span class="pulse-dot"></span> RUNTIME READY
                    </span>
                    <span class="live-badge live-badge-zinc" style="font-size: 11px;">
                        CALIBRATED DETECTOR
                    </span>
                </div>
            </div>
            <p class="linear-subtitle">
                Enterprise hallucination observability engine computing white-box logit dynamics, stochastic self-consistency probing, and bidirectional NLI agreement without length shortcut bias.
            </p>
            <div class="status-bar-container">
                <span class="live-badge live-badge-emerald">
                    <span class="pulse-dot"></span>
                    ACTIVE ENGINE: {selected_model_name}
                </span>
                <span class="live-badge live-badge-primary">
                    19-FEATURE UNIVERSAL CORE
                </span>
                <span class="live-badge live-badge-cyan">
                    XGBOOST + CALIBRATED LOGISTIC REGRESSION
                </span>
                <span class="live-badge {dev_badge_class}">
                    {dev_label}
                </span>
                <span class="live-badge live-badge-zinc">
                    BENCHMARK CERTIFIED (N=4,000 MULTI-TASK)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not cuda_available:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 14px; background: rgba(245, 158, 11, 0.08);
                        border: 1px solid rgba(245, 158, 11, 0.28); border-left: 3px solid #F59E0B;
                        border-radius: 6px; padding: 12px 16px; margin-bottom: 20px; font-size: 12.5px; color: var(--text-primary);">
                <div style="font-family: var(--font-mono); font-size: 10.5px; padding: 3px 8px; border-radius: 4px; background: rgba(245, 158, 11, 0.2); color: #FBBF24; font-weight: 700; white-space: nowrap;">
                    HARDWARE NOTICE
                </div>
                <div>
                    <strong style="color: #FBBF24;">Runtime executing on Host CPU ({cpu_display}).</strong>
                    A CUDA-compatible GPU accelerator was not detected in this Python environment. For production-speed throughput,
                    initialize the service with the dedicated virtual environment:
                    <code style="background: rgba(0,0,0,0.3); color: #FBBF24; padding: 2px 6px; border-radius: 4px; margin: 0 4px;">.\\run_app.bat</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="linear-notice">
            <div style="font-family: var(--font-mono); font-size: 10.5px; padding: 3px 8px; border-radius: 4px; background: rgba(94, 106, 210, 0.2); color: #828EE8; font-weight: 700; white-space: nowrap;">
                CALIBRATION STANDARD
            </div>
            <div>
                <strong>Production Verification & Quality Standard:</strong> Probability scores represent calibrated posterior hallucination risks,
                empirically certified against a standardized multi-domain holdout evaluation suite (4,000 instances balanced 50% faithful / 50% hallucinated across HaluEval, TruthfulQA, and FEVER).
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
    Render high-density infrastructure telemetry strip and modal trigger.
    """
    st.markdown('<div class="section-kicker">INFRASTRUCTURE & RUNTIME ENGINE</div>', unsafe_allow_html=True)
    infra_col1, infra_col2 = st.columns([3.6, 1.4], gap="medium")

    with infra_col1:
        st.markdown(
            f"""
            <div class="infra-strip-card">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                    <div>
                        <div style="font-size: 10px; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">ACTIVE GENERATIVE MODEL</div>
                        <div style="font-size: 14px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 6px; margin-top: 1px;">
                            <span>{selected_model_name}</span>
                            <span style="font-family: var(--font-mono); font-size: 10.5px; background: rgba(99, 102, 241, 0.2); color: #A5B4FC; padding: 2px 6px; border-radius: 4px; font-weight: 500;">{active_specs['p_count']}</span>
                        </div>
                    </div>
                    <div style="width: 1px; height: 32px; background: var(--border-subtle);"></div>
                    <div>
                        <div style="font-size: 10px; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">COMPUTE BACKEND</div>
                        <div style="font-size: 13px; font-weight: 600; color: {dev_color}; display: flex; align-items: center; gap: 5px; margin-top: 1px;">
                            <span>●</span> {dev_short_label}
                        </div>
                    </div>
                    <div style="width: 1px; height: 32px; background: var(--border-subtle);"></div>
                    <div>
                        <div style="font-size: 10px; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">DECISION THRESHOLDS</div>
                        <div style="font-size: 12.5px; font-family: var(--font-mono); color: var(--text-secondary); margin-top: 1px;">
                            Low &lt; <span style="color: #34D399; font-weight: 600;">{threshold_low}%</span> · High &ge; <span style="color: #FB7185; font-weight: 600;">{threshold_high}%</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with infra_col2:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Configure Engine & Hardware", use_container_width=True, help="Open modal popup to switch model, select GPU/CPU, and adjust decision risk cutoffs"):
            configure_dialog_fn()

    st.markdown("<div style='margin-bottom: 22px;'></div>", unsafe_allow_html=True)
