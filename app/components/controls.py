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

    def _apply_preset(prompt_val: str, context_val: str = ""):
        st.session_state["user_prompt"] = prompt_val
        st.session_state["main_prompt_input_field"] = prompt_val
        st.session_state["user_context"] = context_val
        st.session_state["main_context_input_field"] = context_val
        st.session_state["inference_result"] = None

    st.markdown('<div class="section-kicker">PROMPT STUDIO</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Query Verification & Diagnostic Presets</div>', unsafe_allow_html=True)

    # 1. Preset Buttons Bar (Clean enterprise text, no symbols)
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 12px; font-family: var(--font-mono); color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                PRESET BENCHMARK QUERIES
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        if st.button("Moon Landing", key="preset_moon", help="Factual baseline (Apollo 11)", width="stretch"):
            _apply_preset("Who was the first person to walk on the Moon?")
            st.rerun()

    with p_col2:
        if st.button("Paris Capital", key="preset_paris", help="Factoid query", width="stretch"):
            _apply_preset("What is the capital of France?")
            st.rerun()

    with p_col3:
        if st.button("Mars Human", key="preset_mars", help="Adversarial trap query (elevated risk)", width="stretch"):
            _apply_preset("Who was the first human to walk on Mars?")
            st.rerun()

    with p_col4:
        if st.button("CRISPR-Cas9", key="preset_crispr", help="Grounded retrieval with reference passage", width="stretch"):
            _apply_preset(
                "How does Cas9 endonuclease induce targeted double-strand breaks in DNA?",
                "CRISPR-Cas9 is a bacterial adaptive immune system adapted for genome editing. The Cas9 protein complexes with a single guide RNA (sgRNA) containing a 20-nucleotide target sequence adjacent to a protospacer adjacent motif (PAM) like 5'-NGG. Upon target hybridization, the HNH and RuvC nuclease domains cleave complementary and non-complementary DNA strands respectively, introducing double-strand breaks (DSBs).",
            )
            st.rerun()

    # 2. Main Prompt Text Area with Visible Label and Action Reset
    lbl_col, rst_col = st.columns([4.2, 0.8], gap="small")
    with lbl_col:
        st.markdown(
            '<label style="display: block; font-size: 13px; font-weight: 600; color: var(--text-primary); margin-top: 10px; margin-bottom: 2px;">'
            'Input Prompt / Query'
            '</label>',
            unsafe_allow_html=True,
        )
    with rst_col:
        st.markdown('<div style="margin-top: 4px;"></div>', unsafe_allow_html=True)
        if st.button("Reset", key="prompt_reset_btn", help="Clear prompt, context, and current results", width="stretch"):
            _apply_preset("", "")
            st.rerun()

    prompt_text = st.text_area(
        "Input Prompt / Query",
        value=st.session_state["main_prompt_input_field"],
        placeholder="Enter a prompt, factual question, or trap (e.g. Who was the first person to walk on the Moon?)...",
        height=85,
        label_visibility="collapsed",
        key="main_prompt_input_field",
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
            value=st.session_state["main_context_input_field"],
            placeholder="Supporting evidence, background passage, or reference document for groundedness evaluation...",
            height=85,
            label_visibility="collapsed",
            key="main_context_input_field",
        )

    # Synchronize session state values
    st.session_state["user_prompt"] = prompt_text
    st.session_state["user_context"] = context_text
    is_evidence_mode = bool(context_text.strip())

    # 4. Action Bar (With explicit section labels for each action)
    col_run, col_pref, col_export = st.columns([2.4, 1.3, 1.3], gap="small")

    with col_run:
        st.markdown(
            '<div style="font-family: var(--font-mono); font-size: 11px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 4px;">'
            'Action'
            '</div>',
            unsafe_allow_html=True,
        )
        analyze_clicked = st.button("Run Verification Pipeline", type="primary", width="stretch")

    with col_pref:
        cur_tokens = int(st.session_state.get("pref_max_tokens", 128))
        cur_temp = float(st.session_state.get("pref_temperature", 0.0))
        label_str = f"Settings ({cur_tokens}t, T={cur_temp:.1f})"
        st.markdown(
            '<div style="font-family: var(--font-mono); font-size: 11px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 4px;">'
            'Engine Tuning'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.popover(label_str, width="stretch"):
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
        st.markdown(
            '<div style="font-family: var(--font-mono); font-size: 11px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 4px;">'
            'Output Format'
            '</div>',
            unsafe_allow_html=True,
        )
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

    return analyze_clicked, prompt_text, context_text, is_evidence_mode
