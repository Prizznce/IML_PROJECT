"""
Benchmark Mode Component: Multi-Model Neural Hallucination Evaluation & Unified Comparison.

Allows operators to select multiple LLMs (Qwen 3.5, Llama 3.2, Gemma 3, etc.)
and run them simultaneously against the same input question.
Collects each model's answer, executes the full 15-stage hallucination detection pipeline,
and presents the results in a unified, side-by-side comparison interface.
"""

import html
import io
import json
import math
import textwrap
import logging
import time

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.config import (
    MODEL_OPTIONS,
    get_cached_models,
    get_model_specs,
)
from app.inference import (
    UNIVERSAL_CORE_FEATURES,
    run_live_inference,
)
from app.session import (
    to_json_serializable,
    calculate_confidence_score,
    record_benchmark_activity,
)


def render_benchmark_mode(
    cuda_available: bool,
    selected_device: str,
    gpu_name: str,
    cpu_name: str,
    cpu_display: str,
    threshold_low: int = 35,
    threshold_high: int = 65,
):
    """
    Render Benchmark Mode: Multi-Model Comparison Studio and Unified Observability Dashboard.
    """
    # --------------------------------------------------------------------------
    # 1. Header & Live Telemetry Strip
    # --------------------------------------------------------------------------
    if cuda_available and "cuda" in selected_device:
        dev_label = f"CUDA: {gpu_name}"
        dev_badge_class = "live-badge-emerald"
        dev_dot_color = "var(--emerald-text)"
    else:
        dev_label = f"HOST CPU: {cpu_name}"
        dev_badge_class = "live-badge-amber"
        dev_dot_color = "var(--amber-text)"

    st.markdown(
        """
        <div class="linear-header">
            <div class="linear-title-row">
                <div>
                    <h1 class="linear-title">Benchmark Mode · Multi-Model Comparison</h1>
                    <p class="linear-subtitle">
                        Simultaneously evaluate multiple LLMs against identical prompts. Intercepts generation outputs,
                        executes 15-stage neural consistency pipelines, and renders side-by-side risk telemetry.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Telemetry Panel Strip
    total_models_available = len(MODEL_OPTIONS)
    st.markdown(
        f"""
        <div class="telemetry-panel" style="margin-bottom: 20px;">
            <div class="telemetry-runtime-row">
                <div class="telemetry-runtime-chips">
                    <span class="telemetry-kicker">BENCHMARK RUNTIME</span>
                    <span class="live-badge live-badge-primary" title="Multi-model comparison mode active">
                        <span class="pulse-dot"></span>
                        MULTI-MODEL SUITE
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
            <div class="telemetry-config-row">
                <div class="telemetry-config-chips">
                    <span class="telemetry-kicker-secondary">SYSTEM CONFIG</span>
                    <span class="config-chip" title="Total model engines available in registry">
                        MODEL POOL: {total_models_available} ENGINES
                    </span>
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

    # --------------------------------------------------------------------------
    # 2. Session State Initialization for Benchmark Mode
    # --------------------------------------------------------------------------
    if "bm_user_prompt" not in st.session_state:
        st.session_state["bm_user_prompt"] = ""
    if "bm_user_context" not in st.session_state:
        st.session_state["bm_user_context"] = ""
    if "bm_prompt_input_field" not in st.session_state:
        st.session_state["bm_prompt_input_field"] = st.session_state["bm_user_prompt"]
    if "bm_context_input_field" not in st.session_state:
        st.session_state["bm_context_input_field"] = st.session_state["bm_user_context"]
    if "bm_results" not in st.session_state:
        st.session_state["bm_results"] = None
    if "bm_preset_key" not in st.session_state:
        st.session_state["bm_preset_key"] = None
    if "bm_selected_models" not in st.session_state:
        st.session_state["bm_selected_models"] = list(MODEL_OPTIONS.keys())

    # --------------------------------------------------------------------------
    # 3. Model Selection & Input Controls Studio
    # --------------------------------------------------------------------------
    def _apply_bm_preset(prompt_val: str, context_val: str = "", preset_key: str = None):
        st.session_state["bm_user_prompt"] = prompt_val
        st.session_state["bm_prompt_input_field"] = prompt_val
        st.session_state["bm_user_context"] = context_val
        st.session_state["bm_context_input_field"] = context_val
        st.session_state["bm_results"] = None
        st.session_state["bm_preset_key"] = preset_key

    # A. Model Selection Multi-Select
    st.markdown(
        """
        <div class="preset-section-header">
            <span class="preset-section-label">SELECT ENGINES FOR BENCHMARK COMPARISON</span>
            <span class="preset-section-hint">Choose 2 or more LLM models to test simultaneously against the query</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_models = st.multiselect(
        "Select LLM Engines to Benchmark",
        options=list(MODEL_OPTIONS.keys()),
        default=st.session_state.get("bm_selected_models", list(MODEL_OPTIONS.keys())),
        help="Select which causal LLMs to run simultaneously for side-by-side hallucination verification.",
        key="bm_selected_models_input",
    )
    st.session_state["bm_selected_models"] = selected_models

    # Specs row for selected models
    if selected_models:
        spec_chips = []
        for m_name in selected_models:
            specs = get_model_specs(m_name)
            p_cnt = specs.get("p_count", "")
            arch = specs.get("arch", "").split("(")[0].strip()
            spec_chips.append(
                f'<span class="config-chip" style="color: var(--text-primary); border-color: var(--border-medium);">'
                f'<strong>{m_name}</strong>: {p_cnt} · {arch}'
                f'</span>'
            )
        st.markdown(
            f'<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;">{"".join(spec_chips)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please select at least one LLM model from the dropdown above to execute benchmarks.")

    # B. Preset Benchmark Queries
    st.markdown(
        """
        <div class="preset-section-header">
            <span class="preset-section-label">PRESET BENCHMARK QUERIES</span>
            <span class="preset-section-hint">Evaluate models against calibrated factual, trap, and retrieval queries</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2, p_col3, p_col4 = st.columns(4, gap="small")
    active_bm_preset = st.session_state.get("bm_preset_key")

    with p_col1:
        is_sel = (active_bm_preset == "moon")
        if st.button("Moon Landing", key="bm_preset_moon", help="Factual baseline query (Apollo 11)", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_bm_preset("Who was the first person to walk on the Moon?", "", "moon")
            st.rerun()

    with p_col2:
        is_sel = (active_bm_preset == "paris")
        if st.button("Paris Capital", key="bm_preset_paris", help="Factoid query", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_bm_preset("What is the capital of France?", "", "paris")
            st.rerun()

    with p_col3:
        is_sel = (active_bm_preset == "mars")
        if st.button("Mars Human", key="bm_preset_mars", help="Adversarial trap query (elevated risk test)", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_bm_preset("Who was the first human to walk on Mars?", "", "mars")
            st.rerun()

    with p_col4:
        is_sel = (active_bm_preset == "crispr")
        if st.button("CRISPR-Cas9", key="bm_preset_crispr", help="Grounded retrieval with reference passage", use_container_width=True, type="primary" if is_sel else "secondary"):
            _apply_bm_preset(
                "How does Cas9 endonuclease induce targeted double-strand breaks in DNA?",
                "CRISPR-Cas9 is a bacterial adaptive immune system adapted for genome editing. The Cas9 protein complexes with a single guide RNA (sgRNA) containing a 20-nucleotide target sequence adjacent to a protospacer adjacent motif (PAM) like 5'-NGG. Upon target hybridization, the HNH and RuvC nuclease domains cleave complementary and non-complementary DNA strands respectively, introducing double-strand breaks (DSBs).",
                "crispr",
            )
            st.rerun()

    # C. Main Prompt Text Input
    lbl_col, rst_col = st.columns([4.4, 0.8], gap="small", vertical_alignment="center")
    with lbl_col:
        st.markdown(
            '<div class="prompt-input-label-row">'
            '<label class="prompt-input-label">Benchmark Input Prompt / Question</label>'
            '<span class="prompt-input-badge">SHARED INPUT ACROSS ALL MODELS</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with rst_col:
        if st.button("Reset", key="bm_prompt_reset_btn", help="Clear prompt, context, and current benchmark results", use_container_width=True, icon=":material/refresh:"):
            _apply_bm_preset("", "", None)
            st.rerun()

    prompt_text = st.text_area(
        "Benchmark Input Prompt / Question",
        placeholder="Enter a prompt or question to benchmark across all selected models (e.g. Who was the first person to walk on the Moon?)...",
        height=85,
        label_visibility="collapsed",
        key="bm_prompt_input_field",
    )

    st.markdown(
        """
        <div class="prompt-example-hint">
            <span class="prompt-example-prefix">💡 Tip:</span>
            <span class="prompt-example-text">Trap prompts like "Who was the first human on Mars?" test model resistance to sycophantic confabulation across architectures.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # D. Collapsible Reference Context
    has_context = bool(st.session_state.get("bm_context_input_field", "").strip())
    expander_title = (
        "Reference Context & Grounding Evidence (Active)"
        if has_context
        else "Reference Context (Optional Grounding Evidence)"
    )

    with st.expander(expander_title, expanded=has_context):
        st.markdown(
            """
            <div style="font-size: 12px; color: var(--cyan-text); margin-bottom: 8px;">
                <strong>Evidence Grounding Engine:</strong> Evaluates factual groundedness and retrieval agreement
                of all generated answers against this reference text using <code>all-MiniLM-L6-v2</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        context_text = st.text_area(
            "Optional Reference Context",
            placeholder="Supporting evidence, background passage, or reference document for multi-model groundedness comparison...",
            height=85,
            label_visibility="collapsed",
            key="bm_context_input_field",
        )

    # Synchronize state
    st.session_state["bm_user_prompt"] = prompt_text
    st.session_state["bm_user_context"] = context_text
    is_evidence_mode = bool(context_text.strip())

    # E. Execution Settings Bar
    cur_tokens = int(st.session_state.get("bm_pref_max_tokens", 128))
    cur_temp = float(st.session_state.get("bm_pref_temperature", 0.0))
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
                '<div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.04em;">BENCHMARK HYPERPARAMETERS</div>',
                unsafe_allow_html=True,
            )
            st.slider(
                "Max New Tokens",
                min_value=32,
                max_value=512,
                value=cur_tokens,
                step=32,
                help="Caps generation length per model to control memory consumption and latency.",
                key="bm_pref_max_tokens",
            )
            st.slider(
                "Sampling Temperature",
                min_value=0.0,
                max_value=1.0,
                value=cur_temp,
                step=0.1,
                help="T=0.0 enforces greedy decoding for deterministic, reproducible benchmarks across all models.",
                key="bm_pref_temperature",
            )

    with col_export:
        cur_fmt = st.session_state.get("bm_pref_export_format", "JSON (Complete Telemetry)")
        opts = ["JSON (Complete Telemetry)", "CSV (Comparative Matrix)"]
        fmt_idx = opts.index(cur_fmt) if cur_fmt in opts else 0
        st.selectbox(
            "Output Format",
            options=opts,
            index=fmt_idx,
            help="Serialization format for exporting comparative benchmark audits.",
            key="bm_pref_export_format",
            label_visibility="collapsed",
        )

    # F. Primary Action Button
    st.markdown('<div class="primary-action-container">', unsafe_allow_html=True)
    run_benchmark_clicked = st.button(
        f"Run Benchmark Suite ({len(selected_models)} Model{'s' if len(selected_models) != 1 else ''})",
        type="primary",
        icon=":material/rocket_launch:",
        use_container_width=True,
        key="btn_run_benchmark_suite",
        disabled=(len(selected_models) == 0),
        help="Simultaneously run generation and 15-stage hallucination detection for each selected model",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. Multi-Model Inference Execution Pipeline
    # --------------------------------------------------------------------------
    if run_benchmark_clicked:
        clean_p = prompt_text.strip()
        clean_c = context_text.strip() if context_text.strip() else None

        if not clean_p:
            st.warning("Please provide a prompt or select a preset benchmark query before executing.")
        elif not selected_models:
            st.warning("Please select at least one LLM model to benchmark.")
        else:
            results_dict: Dict[str, Any] = {}
            status_container = st.status("Executing Multi-Model Benchmark Suite...", expanded=True)

            with status_container:
                for idx, m_name in enumerate(selected_models, 1):
                    m_id = MODEL_OPTIONS[m_name]
                    st.write(f"**[{idx}/{len(selected_models)}]** Loading & running neural pipeline for `{m_name}`...")

                    try:
                        # 1. Fetch cached model weights and classifiers
                        cached = get_cached_models(model_id=m_id, target_device=selected_device)

                        # 2. Run complete 15-stage live inference
                        t_start = time.perf_counter()
                        res = run_live_inference(
                            prompt=clean_p,
                            context=clean_c,
                            device=selected_device,
                            causal_model=cached["causal_model"],
                            tokenizer=cached["tokenizer"],
                            embedding_model=cached["embedding_model"],
                            nli_model=cached["nli_model"],
                            nli_label_indices=cached["nli_label_indices"],
                            lr_pipeline=cached["lr_pipeline"],
                            xgb_model=cached["xgb_model"],
                            is_evidence_mode=is_evidence_mode,
                            max_new_tokens=int(st.session_state.get("bm_pref_max_tokens", 128)),
                            temperature=float(st.session_state.get("bm_pref_temperature", 0.0)),
                        )
                        elapsed_s = time.perf_counter() - t_start

                        results_dict[m_name] = res
                        st.write(f"✓ `{m_name}` completed in {elapsed_s:.2f}s (Risk: {res.get('combined_hallucination_probability', 0.0)*100:.1f}%)")

                    except Exception as err:
                        st.error(f"Inference error on `{m_name}`: {str(err)}")
                        results_dict[m_name] = {"error": str(err)}

                status_container.update(label=f"Benchmark Suite Complete ({len(results_dict)} models evaluated)", state="complete", expanded=False)

            # Record unified benchmark activity into session audit history
            try:
                record_benchmark_activity(
                    prompt=clean_p,
                    benchmark_results=results_dict,
                    context=clean_c,
                    threshold_low=float(threshold_low),
                    threshold_high=float(threshold_high),
                    is_evidence_mode=is_evidence_mode,
                )
            except Exception:
                logger.exception("Failed to record benchmark activity into session audit history")

            st.session_state["bm_results"] = results_dict
            st.session_state["bm_prompt_ran"] = clean_p
            st.session_state["bm_context_ran"] = clean_c

    # --------------------------------------------------------------------------
    # 5. Dashboard Rendering (Idle vs Benchmark Comparison)
    # --------------------------------------------------------------------------
    bm_results = st.session_state.get("bm_results")

    if not bm_results:
        _render_bm_idle_view()
    else:
        _render_bm_comparison_dashboard(
            bm_results=bm_results,
            prompt_ran=st.session_state.get("bm_prompt_ran", prompt_text),
            context_ran=st.session_state.get("bm_context_ran", context_text),
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            is_evidence_mode=is_evidence_mode,
        )


# ==============================================================================
# Helper: Idle State View
# ==============================================================================

def _render_bm_idle_view():
    """
    Render clean system readiness overview when no benchmark has been executed yet.
    """
    st.markdown("<hr style='margin: 24px 0 18px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">BENCHMARK MODE · SYSTEM READY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Multi-Model Hallucination Observability</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ready-overview-card">
            <div style="font-size: 13.5px; color: var(--text-secondary); margin-bottom: 16px; line-height: 1.6;">
                Benchmark Mode extends the VeritasLM detection engine to evaluate multiple LLM architectures side-by-side.
                By feeding identical queries across models and intercepting forward-pass logits, stochastic consistency,
                and cross-encoder agreements, you can directly compare factual reliability, calibration boundaries, and hallucination propensity.
            </div>
            <div class="pillar-grid">
                <div class="pillar-card">
                    <div class="pillar-mono">STEP 01 · SELECTION</div>
                    <div class="pillar-title">Select Model Suite</div>
                    <div class="pillar-desc">
                        Choose two or more models from the supported registry (Qwen 3.5 0.8B, Llama 3.2 1B Instruct, Gemma 3 1B IT).
                    </div>
                    <div class="pillar-tag tag-emerald">MULTI-ARCHITECTURE POOL</div>
                </div>
                <div class="pillar-card">
                    <div class="pillar-mono">STEP 02 · EXECUTION</div>
                    <div class="pillar-title">Shared Pipeline Execution</div>
                    <div class="pillar-desc">
                        Runs deterministic greedy response generation, k=5 self-consistency rollouts, and 20-pass NLI evaluation for each engine.
                    </div>
                    <div class="pillar-tag tag-cyan">15 STAGES PER MODEL</div>
                </div>
                <div class="pillar-card">
                    <div class="pillar-mono">STEP 03 · OBSERVABILITY</div>
                    <div class="pillar-title">Unified Comparison</div>
                    <div class="pillar-desc">
                        Visualizes side-by-side outputs, risk tiers, token-level hesitation heatmaps, and Tree SHAP feature contributions.
                    </div>
                    <div class="pillar-tag tag-amber">SIDE-BY-SIDE VERDICTS</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: var(--badge-primary-bg); border: 1px solid var(--badge-primary-border); border-radius: 8px;
                    padding: 12px 16px; margin-top: 14px; font-size: 12.5px; color: var(--text-secondary); display: flex; align-items: center; gap: 10px;">
            <span>
                <strong>Ready for benchmark:</strong> Select a preset query above (such as Moon Landing or Mars Human)
                and click <strong style="color: var(--brand-primary-light);">Run Benchmark Suite</strong> to execute multi-model verification.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Helper: Benchmark Results & Unified Comparison Dashboard
# ==============================================================================

def _render_bm_comparison_dashboard(
    bm_results: Dict[str, Any],
    prompt_ran: str,
    context_ran: Optional[str],
    threshold_low: int,
    threshold_high: int,
    is_evidence_mode: bool = False,
):
    """
    Render unified multi-model comparison dashboard:
    1. Executive Leaderboard Table & Reliability Ranking
    2. Side-by-Side Model Response Windows
    3. Comparative Visual Metrics (Risk, Uncertainty, Consistency)
    4. Per-Model In-Depth Diagnostic Inspector Tabs
    5. Unified Comparative Audit Export
    """
    valid_results = {k: v for k, v in bm_results.items() if "error" not in v}
    failed_results = {k: v.get("error", "Unknown error") for k, v in bm_results.items() if "error" in v}

    if not valid_results:
        st.error("All benchmark executions encountered errors. Please check console logs or hardware allocation.")
        if failed_results:
            for m_name, err in failed_results.items():
                st.warning(f"**{m_name}** failed: {err}")
        return

    st.markdown("<hr style='margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">BENCHMARK OBSERVABILITY DOSSIER</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Multi-Model Comparative Analysis & Leaderboard</div>', unsafe_allow_html=True)

    if failed_results:
        for m_name, err in failed_results.items():
            st.warning(f"**Benchmark Evaluation Note:** `{m_name}` failed during inference ({err}) and was excluded from comparative rankings.")

    low_cut = float(threshold_low)
    high_cut = float(threshold_high)

    # --------------------------------------------------------------------------
    # 1. Prepare Comparative Metrics Data
    # --------------------------------------------------------------------------
    summary_rows = []
    for m_name, res in valid_results.items():
        xgb_prob = float(res.get("predicted_risk_xgb", res.get("xgb_hallucination_probability", 0.0)) or 0.0)
        lr_prob = float(res.get("predicted_risk_lr", res.get("lr_hallucination_probability", 0.0)) or 0.0)
        comb_prob = float(res.get("predicted_risk_combined", res.get("combined_hallucination_probability", (xgb_prob + lr_prob) / 2.0)) or 0.0)

        xgb_pct = xgb_prob * 100.0
        lr_pct = lr_prob * 100.0
        comb_pct = comb_prob * 100.0
        conf_pct = calculate_confidence_score(comb_prob)

        if comb_pct < low_cut:
            tier_str = "Faithful / Low Risk"
            tier_color = "var(--emerald-text)"
            tier_badge = "live-badge-emerald"
        elif comb_pct >= high_cut:
            tier_str = "High Hallucination Risk"
            tier_color = "var(--rose-text)"
            tier_badge = "live-badge-rose"
        else:
            tier_str = "Uncertain / Borderline"
            tier_color = "var(--amber-text)"
            tier_badge = "live-badge-amber"

        int_sig = res.get("internal_signals", {})
        mean_ent = float(int_sig.get("mean_entropy", 0.0) or 0.0)
        num_toks = int(int_sig.get("num_tokens", len(res.get("token_details", []))) or 0)
        ppl = float(int_sig.get("perplexity", 1.0) or 1.0)

        sc_sig = res.get("self_consistency_features", {})
        mean_sim = float(sc_sig.get("mean_pairwise_similarity", 0.0) or 0.0)

        nli_sig = res.get("nli_scores", res.get("nli_features", {}))
        nli_dis = float(nli_sig.get("nli_disagreement", 0.0) or 0.0)

        tot_lat = float(res.get("total_latency_ms", 0.0) or 0.0)

        summary_rows.append({
            "model_name": m_name,
            "comb_pct": comb_pct,
            "conf_pct": conf_pct,
            "xgb_pct": xgb_pct,
            "lr_pct": lr_pct,
            "tier_str": tier_str,
            "tier_color": tier_color,
            "tier_badge": tier_badge,
            "mean_entropy": mean_ent,
            "perplexity": ppl,
            "num_tokens": num_toks,
            "mean_similarity": mean_sim,
            "nli_disagreement": nli_dis,
            "latency_ms": tot_lat,
            "raw_res": res,
        })

    # Sort by lowest risk (Most Faithful model first)
    summary_rows.sort(key=lambda x: x["comb_pct"])

    # --------------------------------------------------------------------------
    # 2. Executive Leaderboard Cards Row
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-kicker" style="margin-top: 10px;">RELIABILITY RANKING (LOWEST RISK FIRST)</div>', unsafe_allow_html=True)

    n_models = len(summary_rows)
    cols = st.columns(min(n_models, 3), gap="medium")

    for idx, row in enumerate(summary_rows[:3]):
        with cols[idx]:
            rank_label = f"#{idx + 1} MOST FAITHFUL" if idx == 0 else f"#{idx + 1} RANKED"
            rank_color = "var(--emerald-text)" if idx == 0 else ("var(--brand-primary)" if idx == 1 else "var(--text-secondary)")
            st.markdown(
                f"""
                <div class="hud-card" style="border-top: 3.5px solid {row['tier_color']}; min-height: 140px;">
                    <div class="hud-label-row">
                        <span class="hud-label" style="font-weight: 700; color: {rank_color};">{rank_label}</span>
                        <span class="live-badge {row['tier_badge']}" style="font-size: 10px; padding: 1.5px 7px;">
                            {row['tier_str'].split('/')[0].strip()}
                        </span>
                    </div>
                    <div style="font-size: 16px; font-weight: 700; color: var(--text-primary); margin: 4px 0 2px 0;">
                        {row['model_name']}
                    </div>
                    <div class="hud-value-row" style="margin: 2px 0;">
                        <span class="hud-value" style="font-size: 26px; color: {row['tier_color']};">{row['comb_pct']:.1f}%</span>
                        <span style="font-size: 12px; color: var(--text-muted); margin-left: 4px;">risk</span>
                        <span class="live-badge live-badge-primary" style="font-size: 10px; padding: 1.5px 6px; margin-left: 8px;">
                            {row['conf_pct']:.1f}% CONF
                        </span>
                    </div>
                    <div class="hud-desc" style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                        <span>XGB: {row['xgb_pct']:.1f}% · LR: {row['lr_pct']:.1f}%</span>
                        <span>{row['latency_ms']/1000.0:.2f}s</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. Side-by-Side Model Response Windows
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-kicker" style="margin-top: 14px;">SIDE-BY-SIDE GENERATION OUTPUTS</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 12.5px; color: var(--text-muted); margin-bottom: 10px;">Input Query: <em>"{html.escape(prompt_ran)}"</em></div>', unsafe_allow_html=True)

    resp_cols = st.columns(len(summary_rows), gap="medium")
    for idx, row in enumerate(summary_rows):
        res = row["raw_res"]
        resp_text = html.escape(res.get("generated_response", ""))
        with resp_cols[idx]:
            st.markdown(
                f"""
                <div class="terminal-window" style="min-height: 220px; display: flex; flex-direction: column;">
                    <div class="terminal-topbar" style="display: flex; justify-content: space-between; align-items: center;">
                        <div class="terminal-title" style="font-size: 11px;">
                            {row['model_name'].upper()}
                        </div>
                        <span class="live-badge {row['tier_badge']}" style="font-size: 9.5px; padding: 1px 6px;">
                            {row['comb_pct']:.1f}% RISK
                        </span>
                    </div>
                    <div class="terminal-body" style="flex: 1; font-size: 13px; line-height: 1.55;">
                        "{resp_text}"
                    </div>
                    <div style="background: var(--bg-surface-elevated); padding: 6px 12px; border-top: 1px solid var(--border-subtle);
                                font-family: var(--font-mono); font-size: 10.5px; color: var(--text-muted); display: flex; justify-content: space-between;">
                        <span>Length: {row['num_tokens']} tokens</span>
                        <span>Entropy: {row['mean_entropy']:.3f} nats</span>
                        <span>{row['latency_ms']/1000.0:.2f}s</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. Comparative Metrics Breakdown Table
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-kicker">CROSS-MODEL COMPARATIVE METRICS</div>', unsafe_allow_html=True)

    comp_table = []
    for r in summary_rows:
        comp_table.append({
            "Engine": r["model_name"],
            "Risk Verdict": r["tier_str"],
            "Combined Risk (%)": f"{r['comb_pct']:.2f}%",
            "Confidence (%)": f"{r['conf_pct']:.1f}%",
            "XGBoost Risk (%)": f"{r['xgb_pct']:.2f}%",
            "Logistic Reg (%)": f"{r['lr_pct']:.2f}%",
            "Mean Entropy (nats)": f"{r['mean_entropy']:.3f}",
            "Perplexity": f"{r['perplexity']:.2f}",
            "SC Agreement (sim)": f"{r['mean_similarity']:.3f}",
            "NLI Disagree": f"{r['nli_disagreement']:.3f}",
            "Output Tokens": r["num_tokens"],
            "Latency (s)": f"{r['latency_ms']/1000.0:.2f}s",
        })

    df_comp = pd.DataFrame(comp_table)
    st.dataframe(df_comp, width="stretch", hide_index=True)

    # --------------------------------------------------------------------------
    # 5. Visual Comparative Analysis (Risk & Uncertainty Distribution)
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">MULTI-MODEL COMPARATIVE CHARTS</div>', unsafe_allow_html=True)

    chart_c1, chart_c2 = st.columns(2, gap="medium")

    with chart_c1:
        st.markdown("**Hallucination Posterior Risk (%) by Model Engine**")
        chart_data_risk = pd.DataFrame({
            "Model": [r["model_name"] for r in summary_rows],
            "Combined Risk (%)": [r["comb_pct"] for r in summary_rows],
            "XGBoost Risk (%)": [r["xgb_pct"] for r in summary_rows],
            "Logistic Reg (%)": [r["lr_pct"] for r in summary_rows],
        }).set_index("Model")
        st.bar_chart(chart_data_risk, color=["#E11D48", "#2563EB", "#38BDF8"])

    with chart_c2:
        st.markdown("**Predictive Uncertainty (Mean Token Entropy in nats)**")
        chart_data_ent = pd.DataFrame({
            "Model": [r["model_name"] for r in summary_rows],
            "Mean Entropy (nats)": [r["mean_entropy"] for r in summary_rows],
            "Perplexity / 10": [min(r["perplexity"] / 10.0, 10.0) for r in summary_rows],
        }).set_index("Model")
        st.bar_chart(chart_data_ent, color=["#F59E0B", "#10B981"])

    # --------------------------------------------------------------------------
    # 6. Deep-Dive Model Diagnostic Tabs
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">DETAILED ARCHITECTURAL INSPECTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Deep-Dive Diagnostic Telemetry per Model</div>', unsafe_allow_html=True)

    model_tab_names = [f"Inspect {r['model_name']}" for r in summary_rows]
    model_tabs = st.tabs(model_tab_names)

    for idx, r in enumerate(summary_rows):
        res = r["raw_res"]
        with model_tabs[idx]:
            sub_tabs = st.tabs([
                "Token Uncertainty Heatmap",
                "Tree SHAP Feature Attributions",
                "19-Feature Vector",
                "Latency Profiler",
                "Self-Consistency Rollouts",
            ])

            # Sub-Tab 1: Token Heatmap
            with sub_tabs[0]:
                token_details = res.get("token_details", [])
                if token_details:
                    total_tokens = len(token_details)
                    high_c = sum(1 for t in token_details if t.get("band") == "High confidence")
                    med_c = sum(1 for t in token_details if t.get("band") == "Medium confidence")
                    low_c = sum(1 for t in token_details if t.get("band") == "Low confidence")

                    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
                    st_c1.metric("Total Tokens", f"{total_tokens}")
                    st_c2.metric("High Confidence (P≥65%)", f"{high_c}", f"{(high_c/total_tokens*100):.1f}%")
                    st_c3.metric("Nominal Confidence", f"{med_c}", f"{(med_c/total_tokens*100):.1f}%")
                    st_c4.metric("Low Confidence / Sub-Threshold", f"{low_c}", f"{(low_c/total_tokens*100):.1f}%")

                    heatmap_spans = []
                    for t in token_details:
                        t_token = html.escape(t["token"])
                        t_prob = t["probability"]
                        t_ent = t["entropy"]
                        t_rank = t["rank"]
                        t_band = t["band"]
                        t_step = t["step"]

                        if t_band == "High confidence":
                            bg = "var(--emerald-bg)"
                            border_col = "var(--emerald-border)"
                            txt_col = "var(--emerald-text)"
                        elif t_band == "Low confidence":
                            bg = "var(--rose-bg)"
                            border_col = "var(--rose-border)"
                            txt_col = "var(--rose-text)"
                        else:
                            bg = "var(--amber-bg)"
                            border_col = "var(--amber-border)"
                            txt_col = "var(--amber-text)"

                        tooltip = f"Step {t_step}: '{t_token}' | Prob: {t_prob*100:.1f}% | Entropy: {t_ent:.3f} | Rank: {t_rank} | {t_band}"
                        span_html = (
                            f'<span class="heatmap-token-chip" title="{html.escape(tooltip)}" style="background: {bg}; border-bottom: 2px solid {border_col}; '
                            f'color: {txt_col}; font-weight: 600;">{t_token}</span>'
                        )
                        heatmap_spans.append(span_html)

                    box_html = f'<div class="heatmap-container" style="margin-top: 10px;">\n{"".join(heatmap_spans)}\n</div>'
                    st.markdown(box_html, unsafe_allow_html=True)
                else:
                    st.info("No token details recorded.")

            # Sub-Tab 2: Feature Attributions
            with sub_tabs[1]:
                feature_contribs = res.get("feature_contributions", [])
                if feature_contribs:
                    st.markdown("<p style='font-size: 12.5px; color: var(--text-secondary);'>Top local Tree SHAP log-odds drivers of risk for this model:</p>", unsafe_allow_html=True)
                    max_mag = max([c["magnitude"] for c in feature_contribs]) if feature_contribs else 1.0
                    if max_mag == 0.0:
                        max_mag = 1.0

                    for c in feature_contribs:
                        disp_name = c["display_name"]
                        feat_name = c["feature"]
                        val = c["value"]
                        contrib = c["contribution"]
                        direction = c["direction"]
                        mag = c["magnitude"]
                        pct_bar = min(100.0, max(8.0, (mag / max_mag) * 100.0))

                        if direction == "increases risk":
                            badge_html = '<span class="attr-badge attr-badge-rose">INFLATES RISK</span>'
                            fill_color = "linear-gradient(90deg, #E11D48, #FB7185)"
                        else:
                            badge_html = '<span class="attr-badge attr-badge-emerald">ATTENUATES RISK</span>'
                            fill_color = "linear-gradient(90deg, #10B981, #34D399)"

                        card_html = textwrap.dedent(f"""
                        <div class="attr-row">
                            <div class="attr-top">
                                <div>
                                    <span class="attr-name">{disp_name}</span>
                                    <span class="attr-id">({feat_name})</span>
                                </div>
                                {badge_html}
                            </div>
                            <div class="attr-metrics">
                                <span>Measured Value: <strong>{val:.4f}</strong></span>
                                <span>Impact: <strong>{contrib:+.4f}</strong> log-odds</span>
                            </div>
                            <div class="attr-track">
                                <div style="background: {fill_color}; width: {pct_bar:.1f}%; height: 100%; border-radius: 2px;"></div>
                            </div>
                        </div>
                        """).strip()
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.info("No feature attributions available.")

            # Sub-Tab 3: 19-Feature Vector
            with sub_tabs[2]:
                f_vec = res.get("feature_vector", {})
                vec_data = []
                for f_idx, feat in enumerate(UNIVERSAL_CORE_FEATURES, 1):
                    vec_data.append({
                        "#": f_idx,
                        "Feature": feat,
                        "Value": f"{f_vec.get(feat, 0.0):.6f}",
                    })
                st.dataframe(pd.DataFrame(vec_data), width="stretch", hide_index=True)

            # Sub-Tab 4: Latency Profiler
            with sub_tabs[3]:
                t_stages = res.get("stage_latencies_ms", res.get("timing_ms", {}))
                t_total = res.get("total_latency_ms", sum(t_stages.values()) if t_stages else 0.0)
                st.markdown(f"Total Pipeline Latency: **{t_total/1000.0:.2f}s** ({t_total:.1f} ms across 15 stages)")

                if t_stages:
                    t_rows = []
                    for s_k, s_v in t_stages.items():
                        pct = (s_v / t_total * 100.0) if t_total > 0 else 0.0
                        t_rows.append({
                            "Pipeline Stage": s_k,
                            "Latency (ms)": f"{s_v:.2f}",
                            "Share (%)": f"{pct:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)

            # Sub-Tab 5: Self-Consistency Rollouts
            with sub_tabs[4]:
                sc_resps = res.get("consistency_responses", [])
                st.markdown("<div style='font-size: 11.5px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;'>SAMPLED CANDIDATE ROLLOUTS (k=5 STOCHASTIC PROBES AT T=0.7):</div>", unsafe_allow_html=True)
                for s_i, s_txt in enumerate(sc_resps, 1):
                    st.markdown(
                        f"""
                        <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px;
                                    padding: 7px 12px; margin-bottom: 5px; font-size: 12.5px;">
                            <span style="font-family: var(--font-mono); color: var(--brand-indigo-light); font-weight: 600; margin-right: 8px;">PROBE #{s_i}:</span>
                            <span style="color: var(--text-primary);">{html.escape(s_txt)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # --------------------------------------------------------------------------
    # 7. Unified Benchmark Audit Export
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">BENCHMARK AUDIT EXPORT</div>', unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2, gap="medium")

    with exp_col1:
        # Full JSON telemetry export
        export_payload = {
            "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt": prompt_ran,
            "context": context_ran,
            "models_evaluated": list(valid_results.keys()),
            "ranking": [
                {
                    "rank": i + 1,
                    "model_name": r["model_name"],
                    "combined_risk_pct": r["comb_pct"],
                    "confidence_pct": r["conf_pct"],
                    "verdict": r["tier_str"],
                }
                for i, r in enumerate(summary_rows)
            ],
            "results_by_model": valid_results,
        }
        json_str = json.dumps(to_json_serializable(export_payload), indent=2)
        st.download_button(
            label="Download Complete Benchmark Telemetry (JSON)",
            data=json_str,
            file_name=f"benchmark_audit_{int(time.time())}.json",
            mime="application/json",
            icon=":material/download:",
            use_container_width=True,
        )

    with exp_col2:
        # CSV comparison matrix export
        csv_buffer = io.StringIO()
        df_comp.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Comparison Matrix (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"benchmark_comparison_{int(time.time())}.csv",
            mime="text/csv",
            icon=":material/table_view:",
            use_container_width=True,
        )
