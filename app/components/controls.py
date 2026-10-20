"""
Input Controls, Preset Queries, and Execution Action Bar.
"""

from typing import Tuple
import streamlit as st


def render_input_controls() -> Tuple[bool, str, str, bool]:
    """
    Render streamlined Prompt Verification Studio:
    - Quick preset buttons (pure text, no emojis or symbols)
    - Primary prompt text input with robust session state binding
    - Collapsible reference context (evidence grounding)
    - Enterprise action bar (Run button + Engine Settings popover + Export format)

    Returns:
        Tuple of (analyze_clicked, prompt_text, context_text, is_evidence_mode).
    """
    if "user_prompt" not in st.session_state:
        st.session_state["user_prompt"] = ""
    if "user_context" not in st.session_state:
        st.session_state["user_context"] = ""
    if "main_prompt_input_field" not in st.session_state:
        st.session_state["main_prompt_input_field"] = st.session_state["user_prompt"]
    if "main_context_input_field" not in st.session_state:
        st.session_state["main_context_input_field"] = st.session_state["user_context"]
    if "inference_result" not in st.session_state:
        st.session_state["inference_result"] = None

    if "selected_preset_key" not in st.session_state:
        st.session_state["selected_preset_key"] = None

    def _apply_preset(prompt_val: str, context_val: str = "", preset_key: str = None):
        st.session_state["user_prompt"] = prompt_val
        st.session_state["main_prompt_input_field"] = prompt_val
        st.session_state["user_context"] = context_val
        st.session_state["main_context_input_field"] = context_val
        st.session_state["inference_result"] = None
        st.session_state["selected_preset_key"] = preset_key

    # 1. Preset Buttons Bar (Immediately above prompt field, smaller buttons, active selected state)
    st.markdown(
        """
        <div class="preset-section-header">
            <span class="preset-section-label">PRESET BENCHMARK QUERIES</span>
            <span class="preset-section-hint">Select a reference query to probe model calibration</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2, p_col3, p_col4 = st.columns(4, gap="small")
    active_key = st.session_state.get("selected_preset_key")

    with p_col1:
        is_sel = (active_key == "moon")
        if st.button("Moon Landing", key="preset_moon", help="Factual baseline (Apollo 11)", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_preset("Who was the first person to walk on the Moon?", "", "moon")
            st.rerun()

    with p_col2:
        is_sel = (active_key == "paris")
        if st.button("Paris Capital", key="preset_paris", help="Factoid query", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_preset("What is the capital of France?", "", "paris")
            st.rerun()

    with p_col3:
        is_sel = (active_key == "mars")
        if st.button("Mars Human", key="preset_mars", help="Adversarial trap query (elevated risk)", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_preset("Who was the first human to walk on Mars?", "", "mars")
            st.rerun()

    with p_col4:
        is_sel = (active_key == "crispr")
        if st.button("CRISPR-Cas9", key="preset_crispr", help="Grounded retrieval with reference passage", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_preset(
                "How does Cas9 endonuclease induce targeted double-strand breaks in DNA?",
                "CRISPR-Cas9 is a bacterial adaptive immune system adapted for genome editing. The Cas9 protein complexes with a single guide RNA (sgRNA) containing a 20-nucleotide target sequence adjacent to a protospacer adjacent motif (PAM) like 5'-NGG. Upon target hybridization, the HNH and RuvC nuclease domains cleave complementary and non-complementary DNA strands respectively, introducing double-strand breaks (DSBs).",
                "crispr",
            )
            st.rerun()

    # 2. Main Prompt Text Area with Label & Reset inline on the same row
    lbl_col, rst_col = st.columns([4.4, 0.8], gap="small", vertical_alignment="center")
    with lbl_col:
        st.markdown(
            '<div class="prompt-input-label-row">'
            '<label class="prompt-input-label">Input Prompt / Query</label>'
            '<span class="prompt-input-badge">CORE INTERACTION</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with rst_col:
        if st.button("Reset", key="prompt_reset_btn", help="Clear prompt, context, and current results", use_container_width=True, icon=":material/refresh:"):
            _apply_preset("", "", None)
            st.rerun()

    prompt_text = st.text_area(
        "Input Prompt / Query",
        placeholder="Enter a prompt, factual question, or trap (e.g. Who was the first person to walk on the Moon?)...",
        height=85,
        label_visibility="collapsed",
        key="main_prompt_input_field",
    )

    # Concise Example Prompt below field
    st.markdown(
        """
        <div class="prompt-example-hint">
            <span class="prompt-example-prefix">💡 Example:</span>
            <span class="prompt-example-text">"Who was the first person to walk on the Moon?" or "Explain how CRISPR-Cas9 introduces targeted double-strand breaks."</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Collapsible Reference Context
    has_context = bool(st.session_state.get("main_context_input_field", "").strip())
    expander_title = (
        "Reference Context & Grounding Evidence (Active)"
        if has_context
        else "Reference Context (Optional Evidence Grounding)"
    )

    with st.expander(expander_title, expanded=has_context):
        st.markdown(
            """
            <div style="font-size: 12px; color: var(--cyan-text); margin-bottom: 8px;">
                <strong>Evidence Grounding Engine:</strong> Providing a reference passage triggers Phase 19 Groundedness Analysis
                (query-evidence similarity, response-evidence similarity, and retrieval agreement via <code>all-MiniLM-L6-v2</code>).
            </div>
            <label style="display: block; font-size: 12.5px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px;">
                Grounding Evidence / Reference Passage
            </label>
            """,
            unsafe_allow_html=True,
        )
        context_text = st.text_area(
            "Optional Reference Context",
            placeholder="Supporting evidence, background passage, or reference document for groundedness evaluation...",
            height=85,
            label_visibility="collapsed",
            key="main_context_input_field",
        )

    # Synchronize session state values
    st.session_state["user_prompt"] = prompt_text
    st.session_state["user_context"] = context_text
    is_evidence_mode = bool(context_text.strip())

    # 4. Consolidate Bottom Controls:
    # A. Compact Settings Row (Engine Tuning + Output Format side-by-side)
    cur_tokens = int(st.session_state.get("pref_max_tokens", 128))
    cur_temp = float(st.session_state.get("pref_temperature", 0.0))
    label_str = f"Engine Tuning: {cur_tokens} tokens · T={cur_temp:.1f}"

    st.markdown(
        """
        <div class="execution-settings-bar">
            <span class="settings-bar-title">EXECUTION SETTINGS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pref, col_export = st.columns([1, 1], gap="small", vertical_alignment="center")

    with col_pref:
        with st.popover(label_str, use_container_width=True, icon=":material/tune:"):
            st.markdown(
                '<div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.04em;">GENERATION HYPERPARAMETERS</div>',
                unsafe_allow_html=True,
            )
            st.slider(
                "Max New Tokens",
                min_value=32,
                max_value=512,
                value=cur_tokens,
                step=32,
                help="Caps generation length. Limits memory consumption and reduces inference latency.",
                key="pref_max_tokens",
            )
            st.slider(
                "Sampling Temperature",
                min_value=0.0,
                max_value=1.0,
                value=cur_temp,
                step=0.1,
                help="Decoding stochasticity. T=0.0 is deterministic greedy decoding for reproducible evaluation.",
                key="pref_temperature",
            )

    with col_export:
        cur_fmt = st.session_state.get("pref_export_format", "JSON (AST & Logits)")
        opts = ["JSON (AST & Logits)", "CSV (Tabular Features)", "Parquet"]
        fmt_idx = opts.index(cur_fmt) if cur_fmt in opts else 0
        st.selectbox(
            "Output Format",
            options=opts,
            index=fmt_idx,
            help="Serialization format for telemetry audits (JSON, CSV, or Parquet).",
            key="pref_export_format",
            label_visibility="collapsed",
        )

    # B. Prominent Primary Action Row (Full prominent row, immediate visibility without scrolling)
    st.markdown('<div class="primary-action-container">', unsafe_allow_html=True)
    analyze_clicked = st.button(
        "Run Verification Pipeline",
        type="primary",
        icon=":material/rocket_launch:",
        use_container_width=True,
        key="btn_run_verification",
        help="Execute white-box generation, stochastic self-consistency probing, and XGBoost hallucination verification",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    return analyze_clicked, prompt_text, context_text, is_evidence_mode
