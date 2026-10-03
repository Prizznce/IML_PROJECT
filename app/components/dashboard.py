"""
Primary Observability Telemetry & Neural Risk HUD Dashboard Component.
"""

import html
import io
import json
import textwrap
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

from app.inference import UNIVERSAL_CORE_FEATURES


def render_dashboard(
    res: Optional[Dict[str, Any]],
    selected_model_name: str,
    active_model_id: str,
    threshold_low: int,
    threshold_high: int,
    is_evidence_mode: bool = False,
):
    """
    Render Generation Output terminal, 4-card HUD telemetry, and 5-tab deep inspectors:
    1. Token-Level Uncertainty & Heatmap
    2. Feature Attribution Engine & Logistic Waterfall
    3. Context Retrieval & Groundedness Inspector
    4. Latency Waterfall & Micro-Profiler
    5. Raw 19-Feature Vector & Multi-Format Telemetry Export
    """
    if res is None:
        return

    st.markdown("<hr style='margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">OBSERVABILITY TELEMETRY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Generation Output & Neural Risk HUD</div>', unsafe_allow_html=True)

    # 1. Railway-style Terminal Output Window
    resp_text = html.escape(res["generated_response"])
    st.markdown(
        f"""
        <div class="terminal-window">
            <div class="terminal-topbar">
                <div class="terminal-dots">
                    <span class="terminal-dot dot-red"></span>
                    <span class="terminal-dot dot-yellow"></span>
                    <span class="terminal-dot dot-green"></span>
                </div>
                <div class="terminal-title">PRIMARY GENERATION · {selected_model_name.upper()} (T=0 GREEDY)</div>
                <div style="font-family: var(--font-mono); font-size: 11px; color: var(--emerald-text);">STATUS: COMPLETED</div>
            </div>
            <div class="terminal-body">
                "{resp_text}"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Risk Metrics & Dynamic HUD
    xgb_prob = float(res.get("predicted_risk_xgb", res.get("xgb_hallucination_probability", 0.0)) or 0.0)
    lr_prob = float(res.get("predicted_risk_lr", res.get("lr_hallucination_probability", 0.0)) or 0.0)
    xgb_pct = xgb_prob * 100.0
    lr_pct = lr_prob * 100.0
    
    # Dynamic classification based on user-adjusted or default thresholds
    low_cut = float(threshold_low)
    high_cut = float(threshold_high)
    
    # Card 1 (XGBoost) dynamic states
    if xgb_pct < low_cut:
        current_tier = "Low Risk (Faithful)"
        accent_color = "#10B981"
        tier_bg = "rgba(16, 185, 129, 0.08)"
        tier_border = "rgba(16, 185, 129, 0.32)"
        tier_badge = "live-badge-emerald"
        tier_symbol = "●"
        verdict_title = "FAITHFUL / LOW RISK"
        verdict_desc = f"Posterior risk is safely below {low_cut:.0f}%. Internal log-probabilities and consistency probes demonstrate strong stability."
    elif xgb_pct >= high_cut:
        current_tier = "High Hallucination Risk"
        accent_color = "#FB7185"
        tier_bg = "rgba(244, 63, 94, 0.08)"
        tier_border = "rgba(244, 63, 94, 0.35)"
        tier_badge = "live-badge-rose"
        tier_symbol = "▲"
        verdict_title = "ELEVATED RISK OF HALLUCINATION"
        verdict_desc = f"Posterior risk exceeds {high_cut:.0f}%. Internal token rank entropy or NLI cross-encoder detected significant uncertainty/contradiction."
    else:
        current_tier = "Moderate Risk (Uncertain)"
        accent_color = "#FBBF24"
        tier_bg = "rgba(245, 158, 11, 0.08)"
        tier_border = "rgba(245, 158, 11, 0.32)"
        tier_badge = "live-badge-amber"
        tier_symbol = "◆"
        verdict_title = "BORDERLINE UNCERTAINTY"
        verdict_desc = f"Risk falls in the intermediate band ({low_cut:.0f}%–{high_cut:.0f}%). Stochastic probes show partial semantic drift."

    delta_diff = lr_pct - xgb_pct
    delta_str = f"{delta_diff:+.1f}% vs XGB"

    int_signals = res.get("internal_signals", {})
    mean_ent = int_signals.get("mean_entropy", 0.0)
    seq_ppl = int_signals.get("log_perplexity", 0.0)
    min_lp = int_signals.get("min_log_prob", 0.0)

    # 2. Native Linear/Railway Telemetry Metric Cards with Expanded Verdict Box
    m_col1, m_col2, m_col3, m_col4 = st.columns([1.0, 1.0, 1.55, 1.0])

    with m_col1:
        st.metric(
            label="XGBoost Risk Score",
            value=f"{xgb_pct:.1f}%",
            help="Primary non-linear gradient-boosted tree ensemble risk score (Phase 14 Universal Core).",
        )

    with m_col2:
        st.metric(
            label="Logistic Regression Baseline",
            value=f"{lr_pct:.1f}%",
            delta=delta_str,
            delta_color="inverse" if delta_diff > 0 else "normal",
            help="Standardized linear decision baseline probability (Phase 14 Universal Core).",
        )

    with m_col3:
        st.markdown(
            f"""
            <div style="background: {tier_bg}; border: 1px solid {tier_border}; border-radius: 8px; padding: 12px 14px; min-height: 104px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="font-family: var(--font-mono); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">
                    DECISION VERDICT
                </div>
                <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {accent_color}; box-shadow: 0 0 10px {accent_color}; flex-shrink: 0;"></span>
                    <span style="font-size: 18px; font-weight: 700; color: {accent_color}; letter-spacing: -0.015em; line-height: 1.25; white-space: normal; word-break: break-word;">
                        {current_tier}
                    </span>
                </div>
                <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-secondary); line-height: 1.35;">
                    {verdict_title} (Band: &lt;{low_cut:.0f}% / ≥{high_cut:.0f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.metric(
            label="Predictive Uncertainty",
            value=f"{mean_ent:.3f} nats",
            delta=f"Perplexity: {seq_ppl:.2f}",
            delta_color="off",
            help="Mean predictive Shannon entropy across sequence tokens and log sequence perplexity.",
        )

    # Prominent Verdict Rationale Banner
    st.markdown(
        f"""
        <div style="background: {tier_bg}; border: 1px solid {tier_border}; border-left: 4px solid {accent_color};
                    border-radius: 6px; padding: 12px 16px; margin: 12px 0 16px 0; font-size: 13px; line-height: 1.55;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <strong style="color: {accent_color}; font-size: 13.5px;">{verdict_title}</strong>
                <span class="live-badge {tier_badge}" style="font-size: 10px; padding: 1px 6px;">{xgb_pct:.1f}% POSTERIOR RISK</span>
            </div>
            <div style="color: var(--text-secondary);">
                {verdict_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"*Cutoffs configured: Low Risk < {low_cut:.0f}%, High Risk ≥ {high_cut:.0f}%. "
        "Probability values represent model-estimated posterior hallucination risks on holdout splits.*"
    )

    # ==========================================================================
    # 7. Token-by-Token Confidence Heatmap (Inspector View)
    # ==========================================================================
    st.markdown("<hr style='margin: 26px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">STEP-BY-STEP DYNAMICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Token-by-Token Generation Confidence Heatmap</div>', unsafe_allow_html=True)
    st.write(
        "Interactive token highlighting visualizes model uncertainty during the forward generation pass. "
        "Green tokens denote high generation likelihood and low predictive entropy; yellow tokens indicate "
        "intermediate confidence; red tokens signal elevated predictive uncertainty or depressed likelihood."
    )

    token_details = res.get("token_details", [])
    if token_details:
        # Quick token statistics
        total_tokens = len(token_details)
        high_c = sum(1 for t in token_details if t.get("band") == "High confidence")
        med_c = sum(1 for t in token_details if t.get("band") == "Medium confidence")
        low_c = sum(1 for t in token_details if t.get("band") == "Low confidence")

        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            st.metric("Total Generated Tokens", f"{total_tokens}")
        with stat_c2:
            st.metric("High Confidence Tokens", f"{high_c}", f"{(high_c/total_tokens*100):.1f}%")
        with stat_c3:
            st.metric("Medium Confidence Tokens", f"{med_c}", f"{(med_c/total_tokens*100):.1f}%")
        with stat_c4:
            st.metric("Suspicious / Low Conf Tokens", f"{low_c}", f"{(low_c/total_tokens*100):.1f}%")

        # Heatmap Filter Option
        isolate_suspicious = st.checkbox("Highlight sub-threshold tokens only (P < 30% or elevated entropy)", value=False)

        heatmap_spans = []
        for t in token_details:
            t_token = html.escape(t["token"])
            t_prob = t["probability"]
            t_ent = t["entropy"]
            t_rank = t["rank"]
            t_band = t["band"]
            t_step = t["step"]

            if t_band == "High confidence":
                bg = "rgba(46, 204, 113, 0.18)"
                border_col = "#2ecc71"
                txt_col = "#2ecc71"
                opacity = "0.4" if isolate_suspicious else "1.0"
            elif t_band == "Low confidence":
                bg = "rgba(231, 76, 60, 0.28)"
                border_col = "#e74c3c"
                txt_col = "#ff7675"
                opacity = "1.0"
            else:
                bg = "rgba(241, 196, 15, 0.20)"
                border_col = "#f1c40f"
                txt_col = "#f1c40f"
                opacity = "0.4" if isolate_suspicious else "1.0"

            tooltip = f"Step {t_step}: '{t_token}' | Prob: {t_prob*100:.1f}% | Entropy: {t_ent:.3f} | Rank: {t_rank} | {t_band}"
            span_html = (
                f'<span class="heatmap-token-chip" title="{html.escape(tooltip)}" style="background: {bg}; border-bottom: 2px solid {border_col}; '
                f'color: {txt_col}; opacity: {opacity};">{t_token}</span>'
            )
            heatmap_spans.append(span_html)

        rendered_heatmap = "".join(heatmap_spans)
        box_html = f'<div class="heatmap-container">\n{rendered_heatmap}\n</div>'
        st.markdown(box_html, unsafe_allow_html=True)

        legend_html = """
        <div style="display: flex; gap: 18px; font-size: 12.5px; margin-bottom: 12px; flex-wrap: wrap; font-family: var(--font-sans);">
            <span><span style="color: #2ecc71; font-weight: bold;">● High Confidence</span> (P ≥ 65%, low entropy)</span>
            <span><span style="color: #f1c40f; font-weight: bold;">● Nominal Confidence</span> (30% ≤ P &lt; 65%)</span>
            <span><span style="color: #ff7675; font-weight: bold;">● Sub-Threshold / High Risk</span> (P &lt; 30% or elevated entropy)</span>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        with st.expander("Inspect Token-by-Token Logit Dynamics Table", expanded=False):
            t_rows = []
            for t in token_details:
                t_rows.append({
                    "Step": t["step"],
                    "Token": repr(t["token"]),
                    "Probability": f"{t['probability']*100:.2f}%",
                    "Log-Prob": f"{t['log_prob']:.4f}",
                    "Entropy (nats)": f"{t['entropy']:.4f}",
                    "Rank": t["rank"],
                    "Confidence Band": t["band"],
                })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

    # ==========================================================================
    # 8. Feature Contributions / Tree SHAP Attribution
    # ==========================================================================
    st.markdown("<hr style='margin: 26px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why Was This Response Flagged? (Tree SHAP Attributions)</div>', unsafe_allow_html=True)

    feature_contribs = res.get("feature_contributions", [])
    if feature_contribs:
        c_filter = st.radio(
            "Filter Attributions",
            options=["All Significant Drivers", "Risk Inflators (+)", "Risk Attenuators (-)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        filtered_contribs = []
        for c in feature_contribs:
            if c_filter == "Risk Inflators (+)" and c["direction"] != "increases risk":
                continue
            if c_filter == "Risk Attenuators (-)" and c["direction"] == "increases risk":
                continue
            filtered_contribs.append(c)

        max_mag = max([c["magnitude"] for c in feature_contribs]) if feature_contribs else 1.0
        if max_mag == 0.0:
            max_mag = 1.0

        if not filtered_contribs:
            st.info(f"No features match filter: {c_filter}")
        else:
            for c in filtered_contribs:
                disp_name = c["display_name"]
                feat_name = c["feature"]
                val = c["value"]
                contrib = c["contribution"]
                direction = c["direction"]
                mag = c["magnitude"]
                pct_bar = min(100.0, max(8.0, (mag / max_mag) * 100.0))

                if direction == "increases risk":
                    badge_html = '<span class="attr-badge attr-badge-rose">▲ INFLATES RISK</span>'
                    fill_color = "linear-gradient(90deg, #E11D48, #FB7185)"
                else:
                    badge_html = '<span class="attr-badge attr-badge-emerald">▼ ATTENUATES RISK</span>'
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
                        <span>Measured Feature Value: <strong>{val:.4f}</strong></span>
                        <span>TreeSHAP Impact: <strong>{contrib:+.4f}</strong> log-odds</span>
                    </div>
                    <div class="attr-track">
                        <div style="background: {fill_color}; width: {pct_bar:.1f}%; height: 100%; border-radius: 2px;"></div>
                    </div>
                </div>
                """).strip()
                st.markdown(card_html, unsafe_allow_html=True)

        st.caption(
            "Values represent local Tree SHAP log-odds contributions from the fitted XGBoost detector for this specific generation. "
            "Attributions explain classifier decisions and do not claim causal physical intervention."
        )

    # ==========================================================================
    # 9. Evidence Groundedness Section
    # ==========================================================================
    evidence_data = res.get("evidence_analysis", {})
    if is_evidence_mode or (evidence_data and evidence_data.get("has_reference_context")):
        st.markdown("<hr style='margin: 26px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">GROUNDEDNESS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Reference Evidence Alignment & Retrieval Margin</div>', unsafe_allow_html=True)

        if evidence_data.get("has_reference_context"):
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                st.metric("Query-Evidence Similarity", f"{evidence_data.get('top1_evidence_similarity', 0.0):.4f}")
            with e_col2:
                st.metric("Response-Evidence Similarity", f"{evidence_data.get('response_top1_similarity', 0.0):.4f}")
            with e_col3:
                st.metric("Retrieval Agreement Score", f"{evidence_data.get('retrieval_agreement', 0.0):.4f}")

            ev_cards = evidence_data.get("retrieved_evidence", [])
            if ev_cards:
                st.markdown("**Ranked Reference Passages:**")
                for card in ev_cards:
                    st.markdown(
                        f"""
                        <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 7px;
                                    padding: 12px 16px; margin-bottom: 8px; font-size: 13px;">
                            <div style="font-family: var(--font-mono); font-size: 11.5px; color: var(--brand-indigo-light); margin-bottom: 6px;">
                                RANK #{card['rank']} · QUERY SIM: {card['query_similarity']:.4f} · RESPONSE SIM: {card['response_similarity']:.4f}
                            </div>
                            <div style="color: var(--text-secondary); line-height: 1.55;">{card['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                """
                <div class="linear-notice">
                    <div>ℹ️</div>
                    <div>
                        <strong>No Reference Context Provided:</strong> Enter a reference passage above to compute live groundedness.
                        In the offline Phase 19 retrieval benchmark, multi-signal evidence retrieval achieved <strong>ROC-AUC 0.9184</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # 10. Detailed Signal Breakdowns
    # ==========================================================================
    st.markdown("<hr style='margin: 26px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">FEATURE CHANNELS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Signal Vector Inspection</div>', unsafe_allow_html=True)

    tab_internal, tab_sc, tab_nli, tab_vec = st.tabs([
        "Internal Signals (11)",
        "Self-Consistency (5)",
        "NLI Agreement (3)",
        "Canonical 19-Vector",
    ])

    with tab_internal:
        st.markdown(
            "<p style='font-size: 13px; color: var(--text-secondary);'>"
            "White-box generation metrics extracted during an unconstrained greedy forward pass over output logits."
            "</p>",
            unsafe_allow_html=True,
        )
        int_sig = res["internal_signals"]
        internal_table_data = [
            {"Signal": "min_log_prob", "Value": f"{int_sig['min_log_prob']:.4f}", "Interpretation": "Minimum token log-probability (bottleneck token)"},
            {"Signal": "mean_log_prob", "Value": f"{int_sig['mean_log_prob']:.4f}", "Interpretation": "Mean token log-probability across sequence"},
            {"Signal": "mean_token_prob", "Value": f"{int_sig['mean_token_prob']:.4f}", "Interpretation": "Mean token probability"},
            {"Signal": "token_prob_std", "Value": f"{int_sig['token_prob_std']:.4f}", "Interpretation": "Standard deviation of token probabilities"},
            {"Signal": "mean_entropy", "Value": f"{int_sig['mean_entropy']:.4f}", "Interpretation": "Mean predictive Shannon entropy (nats)"},
            {"Signal": "max_entropy", "Value": f"{int_sig['max_entropy']:.4f}", "Interpretation": "Peak predictive entropy in sequence"},
            {"Signal": "entropy_std", "Value": f"{int_sig['entropy_std']:.4f}", "Interpretation": "Standard deviation of predictive entropy"},
            {"Signal": "log_perplexity", "Value": f"{int_sig['log_perplexity']:.4f}", "Interpretation": "Log-transformed sequence perplexity"},
            {"Signal": "log_mean_token_rank", "Value": f"{int_sig['log_mean_token_rank']:.4f}", "Interpretation": "log(1 + mean token rank)"},
            {"Signal": "log_max_token_rank", "Value": f"{int_sig['log_max_token_rank']:.4f}", "Interpretation": "log(1 + max token rank)"},
            {"Signal": "log_rank_std", "Value": f"{int_sig['log_rank_std']:.4f}", "Interpretation": "log(1 + token rank std)"},
        ]
        st.dataframe(pd.DataFrame(internal_table_data), use_container_width=True, hide_index=True)

    with tab_sc:
        st.markdown(
            "<p style='font-size: 13px; color: var(--text-secondary);'>"
            "Five stochastic generations (k=5, T=0.7, top_p=0.9) probed for semantic stability using <code>all-MiniLM-L6-v2</code>."
            "</p>",
            unsafe_allow_html=True,
        )
        sc_sig = res.get("self_consistency_features", {})
        sc_table_data = [
            {"Signal": "exact_match_agreement", "Value": f"{sc_sig.get('exact_match_agreement', 0.0):.4f}", "Interpretation": "Fraction of candidate pairs that match identically"},
            {"Signal": "mean_pairwise_similarity", "Value": f"{sc_sig.get('mean_pairwise_similarity', 0.0):.4f}", "Interpretation": "Mean embedding cosine similarity among candidate outputs"},
            {"Signal": "min_pairwise_similarity", "Value": f"{sc_sig.get('min_pairwise_similarity', 0.0):.4f}", "Interpretation": "Minimum pairwise cosine similarity (worst-case divergence)"},
            {"Signal": "max_pairwise_similarity", "Value": f"{sc_sig.get('max_pairwise_similarity', 0.0):.4f}", "Interpretation": "Maximum pairwise cosine similarity"},
            {"Signal": "pairwise_similarity_std", "Value": f"{sc_sig.get('pairwise_similarity_std', 0.0):.4f}", "Interpretation": "Standard deviation of pairwise cosine similarities"},
        ]
        st.dataframe(pd.DataFrame(sc_table_data), use_container_width=True, hide_index=True)

        st.markdown("<div style='font-size: 12px; font-weight: 600; color: var(--text-muted); margin-top: 14px; margin-bottom: 6px;'>SAMPLED CANDIDATE RESPONSES (k=5 STOCHASTIC PROBES):</div>", unsafe_allow_html=True)
        for i, resp in enumerate(res.get("consistency_responses", []), 1):
            st.markdown(
                f"""
                <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px;
                            padding: 8px 14px; margin-bottom: 6px; font-size: 13px; font-family: var(--font-sans);">
                    <span style="font-family: var(--font-mono); color: var(--brand-indigo-light); font-weight: 600; margin-right: 8px;">SAMPLE #{i}:</span>
                    <span style="color: #E2E8F0;">{html.escape(resp)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_nli:
        st.markdown(
            "<p style='font-size: 13px; color: var(--text-secondary);'>"
            "Natural Language Inference cross-encoder evaluation (<code>nli-MiniLM2-L6-H768</code>) over 10 pairs / 20 directional passes."
            "</p>",
            unsafe_allow_html=True,
        )
        nli_sig = res.get("nli_scores", res.get("nli_features", {}))
        nli_table_data = [
            {"Signal": "mean_pairwise_entailment", "Value": f"{nli_sig.get('mean_pairwise_entailment', 0.0):.4f}", "Interpretation": "Average cross-encoder entailment probability across candidate pairs"},
            {"Signal": "mean_pairwise_contradiction", "Value": f"{nli_sig.get('mean_pairwise_contradiction', 0.0):.4f}", "Interpretation": "Average cross-encoder contradiction probability across candidate pairs"},
            {"Signal": "nli_disagreement", "Value": f"{nli_sig.get('nli_disagreement', 0.0):.4f}", "Interpretation": "Composite NLI disagreement [contradiction + 0.5 * (1 - entailment)]"},
        ]
        st.dataframe(pd.DataFrame(nli_table_data), use_container_width=True, hide_index=True)

    with tab_vec:
        st.markdown(
            "<p style='font-size: 13px; color: var(--text-secondary);'>"
            "Exact tabular vector passed to the fitted Phase 14 XGBoost and Logistic Regression estimators."
            "</p>",
            unsafe_allow_html=True,
        )
        f_vec = res["feature_vector"]
        vector_rows = []
        for idx, feat in enumerate(UNIVERSAL_CORE_FEATURES, 1):
            vector_rows.append({
                "#": idx,
                "Feature Canonical Name": feat,
                "Standardized Value": f"{f_vec[feat]:.6f}",
            })
        st.dataframe(pd.DataFrame(vector_rows), use_container_width=True, hide_index=True)

    # ==========================================================================
    # 11. Pipeline Profiling & Latency Breakdown (Railway execution waterfall)
    # ==========================================================================
    st.markdown("<hr style='margin: 26px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">INSTRUMENTATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Execution Latency & Stage Profiling Waterfall</div>', unsafe_allow_html=True)

    timing_data = res.get("stage_latencies_ms", res.get("timing_ms", {}))
    total_elapsed = res.get("total_latency_ms", sum(timing_data.values()) if timing_data else 0.0)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: var(--bg-surface);
                    border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px 18px; margin-bottom: 12px;">
            <div style="font-size: 13.5px; color: var(--text-secondary);">
                Total Live Pipeline Latency: <strong style="color: #FFFFFF; font-family: var(--font-mono);">{total_elapsed / 1000.0:.2f} s</strong>
                ({total_elapsed:.1f} ms across 15 execution stages)
            </div>
            <span class="live-badge live-badge-emerald">
                <span class="pulse-dot"></span> PROFILED
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if timing_data:
        # Visual Waterfall Execution Bar
        color_palette = ["#5E6AD2", "#38BDF8", "#34D399", "#FBBF24", "#FB7185", "#A855F7", "#EC4899", "#14B8A6"]
        segments_html = []
        for i, (s_name, s_ms) in enumerate(timing_data.items()):
            pct = (s_ms / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            color = color_palette[i % len(color_palette)]
            segments_html.append(
                f'<div class="waterfall-segment" style="width: {pct:.2f}%; background: {color};" title="{html.escape(s_name)}: {s_ms:.1f}ms ({pct:.1f}%)"></div>'
            )
        st.markdown(f'<div class="waterfall-track">{"".join(segments_html)}</div>', unsafe_allow_html=True)

        timing_rows = []
        for s_name, s_ms in timing_data.items():
            pct = (s_ms / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            timing_rows.append({
                "Pipeline Execution Stage": s_name,
                "Latency (ms)": f"{s_ms:.2f}",
                "Share (%)": f"{pct:.1f}%",
            })
        st.dataframe(pd.DataFrame(timing_rows), use_container_width=True, hide_index=True)

    # ==========================================================================
    # 12. Export & Audit Log Tools (New Feature!)
    # ==========================================================================
    st.markdown("<hr style='margin: 26px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">REPORTING & EXPORT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Observability Audit Artifacts</div>', unsafe_allow_html=True)

    export_col1, export_col2 = st.columns(2)

    # Prepare complete JSON dump
    audit_payload = {
        "model_slug": active_model_id,
        "prompt": st.session_state["user_prompt"],
        "context": st.session_state["user_context"],
        "generated_response": res["generated_response"],
        "risk_verdict": {
            "xgb_probability": xgb_prob,
            "lr_probability": lr_prob,
            "tier": current_tier,
            "low_cutoff": low_cut,
            "high_cutoff": high_cut,
        },
        "feature_vector": res["feature_vector"],
        "stage_latencies_ms": timing_data,
        "total_latency_ms": total_elapsed,
    }
    audit_json = json.dumps(audit_payload, indent=2)

    export_fmt = st.session_state.get("pref_export_format", "JSON (AST & Logits)")

    with export_col1:
        if "CSV" in export_fmt:
            export_df = pd.DataFrame([{
                "prompt": st.session_state["user_prompt"],
                "generated_response": res["generated_response"],
                "xgb_probability": xgb_prob,
                "lr_probability": lr_prob,
                "risk_tier": current_tier,
                "model_id": active_model_id,
                **res["feature_vector"]
            }])
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Export Telemetry Audit (CSV)",
                data=csv_data,
                file_name=f"hallucination_features_{int(total_elapsed)}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        elif "Parquet" in export_fmt:
            import io
            export_df = pd.DataFrame([{
                "prompt": st.session_state["user_prompt"],
                "generated_response": res["generated_response"],
                "xgb_probability": xgb_prob,
                "lr_probability": lr_prob,
                "risk_tier": current_tier,
                "model_id": active_model_id,
                **res["feature_vector"]
            }])
            pq_buf = io.BytesIO()
            export_df.to_parquet(pq_buf, index=False)
            st.download_button(
                label="Export Telemetry Audit (Parquet)",
                data=pq_buf.getvalue(),
                file_name=f"hallucination_telemetry_{int(total_elapsed)}.parquet",
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.download_button(
                label="Export Verification Audit (JSON)",
                data=audit_json,
                file_name=f"hallucination_audit_{int(total_elapsed)}.json",
                mime="application/json",
                use_container_width=True,
            )

    with export_col2:
        markdown_summary = textwrap.dedent(f"""
        ### Hallucination Observability & Risk Report
        - **Model**: `{active_model_id}`
        - **Verdict**: {current_tier}
        - **XGBoost Risk**: {xgb_pct:.1f}%
        - **Logistic Regression Risk**: {lr_pct:.1f}%
        - **Prompt**: "{st.session_state['user_prompt']}"
        - **Generation**: "{res['generated_response']}"
        - **Total Pipeline Latency**: {total_elapsed:.1f} ms
        """).strip()

        with st.expander("View Shareable Telemetry Summary (Markdown)", expanded=False):
            st.code(markdown_summary, language="markdown")


# ==============================================================================
