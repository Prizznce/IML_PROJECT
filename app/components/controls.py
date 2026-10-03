"""
Input Controls, Preset Queries, and Execution Action Bar.
"""

from typing import Tuple
import streamlit as st


def render_input_controls() -> Tuple[bool, str, str, bool]:
    """
    Render detection mode selector, benchmarked presets, prompt & context inputs,
    and action bar with Run Pipeline button, Engine Preferences popover dropdown,
    and Export Format selector.

    Returns:
        Tuple of (analyze_clicked, prompt_text, context_text, is_evidence_mode).
    """
    if "user_prompt" not in st.session_state:
        st.session_state["user_prompt"] = ""
    if "user_context" not in st.session_state:
        st.session_state["user_context"] = ""
    if "inference_result" not in st.session_state:
        st.session_state["inference_result"] = None

    st.markdown('<div class="section-kicker">CONTROL PANEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Query Configuration & Benchmarked Presets</div>', unsafe_allow_html=True)

    # Mode Selector
    mode_selection = st.radio(
        "Detection Mode",
        options=[
            "Internal Signals (Universal Core — 19 Features)",
            "Check Against Evidence (Phase 19 Retrieval Variant)",
        ],
        index=0,
        horizontal=True,
        help="Universal Core evaluates internal generation logits, self-consistency, and NLI. Evidence mode additionally inspects reference passage groundedness.",
        label_visibility="collapsed",
    )

    is_evidence_mode = "Check Against Evidence" in mode_selection

    if is_evidence_mode:
        st.markdown(
            """
            <div style="font-size: 12.5px; color: var(--cyan-text); background: var(--cyan-bg); border: 1px solid var(--cyan-border);
                        border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; line-height: 1.5;">
                <strong>Evidence Groundedness Engine Active:</strong> When reference context is provided below, the pipeline computes
                query-evidence similarity, response-evidence similarity, and retrieval agreement using <code>all-MiniLM-L6-v2</code>.
                <em>(Phase 19 benchmark achieved ROC-AUC 0.9184 on 2,500 HaluEval+FEVER instances).</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Curated Preset Cards
    st.markdown("<div style='font-size: 11.5px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>PRE-INDEXED PRODUCTION TEST PROFILES:</div>", unsafe_allow_html=True)

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)

    with p_col1:
        if st.button("Moon Landing\n[FACTUAL BASELINE]", use_container_width=True):
            st.session_state["user_prompt"] = "Who was the first person to walk on the Moon?"
            st.session_state["user_context"] = ""

    with p_col2:
        if st.button("Paris Capital\n[FACTOID QUERY]", use_container_width=True):
            st.session_state["user_prompt"] = "What is the capital of France?"
            st.session_state["user_context"] = ""

    with p_col3:
        if st.button("Mars Human\n[ADVERSARIAL TRAP]", use_container_width=True):
            st.session_state["user_prompt"] = "Who was the first human to walk on Mars?"
            st.session_state["user_context"] = ""

    with p_col4:
        if st.button("CRISPR-Cas9\n[GROUNDED RETRIEVAL]", use_container_width=True):
            st.session_state["user_prompt"] = "How does Cas9 endonuclease induce targeted double-strand breaks in DNA?"
            st.session_state["user_context"] = "CRISPR-Cas9 is a bacterial adaptive immune system adapted for genome editing. The Cas9 protein complexes with a single guide RNA (sgRNA) containing a 20-nucleotide target sequence adjacent to a protospacer adjacent motif (PAM) like 5'-NGG. Upon target hybridization, the HNH and RuvC nuclease domains cleave complementary and non-complementary DNA strands respectively, introducing double-strand breaks (DSBs)."

    with p_col5:
        if st.button("Reset Workspace\n[CLEAR INPUT]", use_container_width=True):
            st.session_state["user_prompt"] = ""
            st.session_state["user_context"] = ""
            st.session_state["inference_result"] = None

    # Input text areas
    prompt_text = st.text_area(
        "Prompt",
        value=st.session_state["user_prompt"],
        placeholder="Enter a factual question, proposition, or hallucination trap (e.g. Who was the first person to walk on the Moon?)...",
        height=95,
        label_visibility="collapsed",
    )

    context_text = st.text_area(
        "Optional Reference Context",
        value=st.session_state["user_context"],
        placeholder="Optional supporting evidence, background passage, or reference document for groundedness evaluation...",
        height=80,
        label_visibility="collapsed",
    )

    # Action Bar: Run Pipeline + Engine Preferences Popover + Export Format Selector
    col_run, col_pref, col_export = st.columns([2.2, 1.8, 1.4], gap="small")

    with col_run:
        analyze_clicked = st.button("Run Verification Pipeline ↵", type="primary", use_container_width=True)

    with col_pref:
        cur_tokens = int(st.session_state.get("pref_max_tokens", 128))
        cur_temp = float(st.session_state.get("pref_temperature", 0.0))
        label_str = f"Engine Settings ({cur_tokens} tok, T={cur_temp:.1f}) ▾"
        with st.popover(label_str, use_container_width=True):
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
            "Export Format",
            options=opts,
            index=fmt_idx,
            help="Serialization format for telemetry audits (JSON, CSV, or Parquet).",
            key="pref_export_format",
            label_visibility="collapsed",
        )

    return analyze_clicked, prompt_text, context_text, is_evidence_mode
