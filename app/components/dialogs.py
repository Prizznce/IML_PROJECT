"""
Modal Dialogs for Infrastructure & Model Engine Configuration.
"""

import streamlit as st
from app.config import MODEL_OPTIONS, get_model_specs


@st.dialog("Infrastructure & Model Engine Configuration", width="large")
def configure_infrastructure_dialog(cuda_available: bool, gpu_display: str, cpu_display: str):
    """
    Modal popup dialog allowing users to switch models, choose execution hardware,
    calibrate decision thresholds, and inspect feature taxonomy without UI clutter.
    """
    st.markdown(
        """
        <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px; line-height: 1.5;">
            Configure causal generative models, compute execution devices, and calibrate risk thresholds for real-time hallucination detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_model, tab_hardware, tab_thresholds, tab_taxonomy = st.tabs([
        "Generative Model",
        "Compute Hardware",
        "Risk Thresholds",
        "Feature Taxonomy",
    ])

    with tab_model:
        st.markdown('<div class="section-kicker">MODEL SELECTION</div>', unsafe_allow_html=True)
        cur_model = st.session_state.get("selected_model_name", "Qwen 3.5 (0.8B)")
        model_keys = list(MODEL_OPTIONS.keys())
        default_idx = model_keys.index(cur_model) if cur_model in model_keys else 0
        new_model = st.selectbox(
            "Active Generative Model",
            options=model_keys,
            index=default_idx,
            help="Select which causal generative LLM produces responses and token probability dynamics.",
            key="dialog_model_select",
        )

        modal_specs = get_model_specs(new_model)
        m_slug = MODEL_OPTIONS[new_model].split("/")[-1]

        st.markdown(
            f"""
            <div class="spec-box" style="margin-top: 12px;">
                <div class="spec-row">
                    <span class="spec-label">Model Repository Slug</span>
                    <span class="spec-val"><code>{m_slug}</code></span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Architecture</span>
                    <span class="spec-val">{modal_specs['arch']}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Active Parameters</span>
                    <span class="spec-val">{modal_specs['p_count']}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Context Window</span>
                    <span class="spec-val">{modal_specs['ctx_len']}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Holdout ROC-AUC</span>
                    <span class="spec-val" style="color: var(--emerald-text); font-weight:600;">{modal_specs['roc_val']}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Holdout PR-AUC</span>
                    <span class="spec-val">{modal_specs['pr_val']}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Holdout Accuracy</span>
                    <span class="spec-val">{modal_specs['acc_val']}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">ECE Calibration Error</span>
                    <span class="spec-val">{modal_specs['ece_val']}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Engine Deployment State</span>
                    <span class="spec-val" style="color: var(--emerald-text);">● Ready (Optimized)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_hardware:
        st.markdown('<div class="section-kicker">ACCELERATOR BACKEND</div>', unsafe_allow_html=True)
        if cuda_available:
            hw_options = [f"GPU: {gpu_display} (CUDA:0)", f"Host CPU: {cpu_display} (Fallback)"]
            cur_hw_choice = 0 if "GPU" in st.session_state.get("hardware_choice", "GPU") else 1
            new_hw = st.radio(
                "Hardware Acceleration",
                options=hw_options,
                index=cur_hw_choice,
                help="CUDA GPU acceleration executes in float16/bfloat16 precision on dedicated NVIDIA Tensor Cores.",
                key="dialog_hw_select",
            )
            st.markdown(
                f"""
                <div style="font-size: 12px; color: var(--emerald-text); background: var(--emerald-bg);
                            border: 1px solid var(--emerald-border); border-radius: 6px; padding: 10px 12px; margin-top: 12px;">
                    ● <strong>GPU Accelerated:</strong> {gpu_display} (CUDA:0) ready for low-latency batch tensor ops.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.radio(
                "Hardware Acceleration",
                options=[f"Host CPU: {cpu_display}"],
                index=0,
                disabled=True,
                key="dialog_hw_select_cpu",
            )
            st.markdown(
                f"""
                <div style="font-size: 12px; color: var(--amber-text); background: var(--amber-bg);
                            border: 1px solid var(--amber-border); border-radius: 6px; padding: 10px 12px; margin-top: 12px; line-height: 1.45;">
                    <div style="font-weight: 600; margin-bottom: 2px;">HOST CPU ACTIVE</div>
                    <strong>Detected CPU:</strong> {cpu_display}<br>
                    CUDA GPU acceleration was not detected in this runtime. Initialize via <code>.\\run_app.bat</code> for GPU mode.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_thresholds:
        st.markdown('<div class="section-kicker">RISK BOUNDARIES & CLASSIFICATION CUTOFFS</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 12.5px; color: var(--text-secondary); margin-bottom: 12px;">
                Calibrate the posterior risk thresholds that classify responses into Faithful, Indeterminate, or High Hallucination Risk tiers.
            </div>
            """,
            unsafe_allow_html=True,
        )
        new_t_low = st.slider(
            "Low Risk Cutoff (%)",
            min_value=15,
            max_value=50,
            value=int(st.session_state.get("threshold_low", 35)),
            step=5,
            help="Responses with predicted risk below this value are classified as Low Risk / Faithful.",
            key="dialog_t_low",
        )
        new_t_high = st.slider(
            "High Risk Cutoff (%)",
            min_value=50,
            max_value=85,
            value=int(st.session_state.get("threshold_high", 65)),
            step=5,
            help="Responses with predicted risk equal to or above this value are flagged as High Hallucination Risk.",
            key="dialog_t_high",
        )

        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 14px; font-size: 11.5px;">
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="color: #34D399; font-weight: 700; margin-bottom: 3px;">FAITHFUL</div>
                    <div style="color: var(--text-secondary); font-family: var(--font-mono);">&lt; {new_t_low}%</div>
                </div>
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="color: #FBBF24; font-weight: 700; margin-bottom: 3px;">INDETERMINATE</div>
                    <div style="color: var(--text-secondary); font-family: var(--font-mono);">{new_t_low}% – {new_t_high}%</div>
                </div>
                <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="color: #FB7185; font-weight: 700; margin-bottom: 3px;">HIGH RISK</div>
                    <div style="color: var(--text-secondary); font-family: var(--font-mono);">&ge; {new_t_high}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_taxonomy:
        st.markdown('<div class="section-kicker">FEATURE TAXONOMY SPECIFICATION</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 12.5px; line-height: 1.8; color: var(--text-secondary); margin-top: 4px;">
                <div>• <strong style="color: var(--text-primary);">11 Internal Signals:</strong> Minimum / Average / Cumulative Log-Probabilities, Token Entropy, Margin Dynamics, Logit Variance.</div>
                <div>• <strong style="color: var(--text-primary);">5 Self-Consistency Probes:</strong> Stochastic cross-sample agreement across k=5 candidate generations (Temperature T=0.7).</div>
                <div>• <strong style="color: var(--text-primary);">3 NLI Agreement Passes:</strong> Cross-Encoder natural language inference verifying premise entailment and bidirectional contradiction.</div>
                <div style="margin-top: 12px; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); background: var(--badge-zinc-bg); padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border-subtle);">
                    ANTI-CONFOUNDER SHIELD:<br>
                    <code>num_tokens</code> strictly isolated during calibration to eliminate length shortcut bias.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 16px 0 12px 0;'>", unsafe_allow_html=True)
    col_save, col_close = st.columns([1, 1])
    with col_save:
        if st.button("Apply & Update Runtime ↵", type="primary", use_container_width=True):
            st.session_state["selected_model_name"] = new_model
            if cuda_available and "dialog_hw_select" in st.session_state:
                st.session_state["hardware_choice"] = st.session_state["dialog_hw_select"]
            st.session_state["threshold_low"] = new_t_low
            st.session_state["threshold_high"] = new_t_high
            st.rerun()
    with col_close:
        if st.button("Close", use_container_width=True):
            st.rerun()
