"""
Primary Observability Telemetry & Neural Risk HUD Dashboard Component.
Organized so that important summary items appear first, while deep diagnostic
inspectors are cleanly organized into tabs when a query is executed.
"""

import html
import io
import json
import math
import textwrap
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

from app.inference import UNIVERSAL_CORE_FEATURES
from app.session import calculate_confidence_score


def render_idle_overview():
    """
    Render clean, professional system readiness dashboard when no query has been run yet.
    Highlights the three primary defense pillars and calibration proof without visual clutter.
    """
    st.markdown("<hr style='margin: 24px 0 18px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">SYSTEM STATUS · READY FOR ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Hallucination Defense Architecture Overview</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ready-overview-card">
            <div style="font-size: 13.5px; color: var(--text-secondary); margin-bottom: 16px; line-height: 1.6;">
                Veritas intercepts generative LLM outputs and subjects each response to a 15-stage neural verification pipeline.
                Three complementary defense channels compute a unified 19-dimensional signal vector calibrated against 4,000 multi-domain benchmarks.
            </div>
            <div class="pillar-grid">
                <div class="pillar-card">
                    <div class="pillar-mono">PILLAR 01 · 11 FEATURES</div>
                    <div class="pillar-title">White-Box Logit Probes</div>
                    <div class="pillar-desc">
                        Captures internal token-level Shannon entropy, sequence log-perplexity, minimum log-probability bottleneck, and token rank variance directly from output distributions.
                    </div>
                    <div class="pillar-tag tag-emerald">PROBING LOGITS & HESITATION</div>
                </div>
                <div class="pillar-card">
                    <div class="pillar-mono">PILLAR 02 · 5 FEATURES</div>
                    <div class="pillar-title">Stochastic Self-Consistency</div>
                    <div class="pillar-desc">
                        Samples k=5 stochastic candidate rollouts at T=0.7 and evaluates semantic drift and embedding cosine consensus using <code>all-MiniLM-L6-v2</code>.
                    </div>
                    <div class="pillar-tag tag-cyan">MEASURING DRIFT & STABILITY</div>
                </div>
                <div class="pillar-card">
                    <div class="pillar-mono">PILLAR 03 · 3 FEATURES</div>
                    <div class="pillar-title">Cross-Encoder NLI Probes</div>
                    <div class="pillar-desc">
                        Computes pairwise bidirectional contradiction and entailment probabilities via <code>nli-MiniLM2-L6-H768</code> across candidate generation pairs.
                    </div>
                    <div class="pillar-tag tag-amber">CATCHING MUTUAL CONTRADICTIONS</div>
                </div>
            </div>
            <div class="ready-footer-bar">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--emerald-text); box-shadow: 0 0 8px var(--emerald-glow);"></span>
                    <span style="font-size: 12px; font-weight: 600; color: var(--text-primary);">DUAL DECISION CORE:</span>
                    <span style="font-size: 12px; color: var(--text-secondary);">Calibrated XGBoost + Logistic Regression</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-family: var(--font-mono); font-size: 11px; background: var(--badge-primary-bg); color: var(--badge-primary-text); border: 1px solid var(--badge-primary-border); padding: 2px 7px; border-radius: 4px; font-weight: 600;">0% LENGTH BIAS</span>
                    <span style="font-size: 12px; color: var(--text-secondary);">Generation length strictly quarantined from decision paths</span>
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
                <strong>Ready for analysis:</strong> Select a preset query above (such as Moon Landing or Mars Human)
                or enter a custom prompt and click <strong style="color: var(--brand-primary-light);">Run Verification Pipeline</strong> to execute the 15-stage neural verification pipeline.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(
    res: Optional[Dict[str, Any]],
    selected_model_name: str,
    active_model_id: str,
    threshold_low: int,
    threshold_high: int,
    is_evidence_mode: bool = False,
):
    """
    Render Observability Telemetry & Neural Risk HUD Dashboard.
    When idle, shows system readiness. When executed, puts critical summary
    (Generation Output & Primary Verdict HUD) first, followed by organized tabbed inspectors.
    """
    if res is None:
        render_idle_overview()
        return

    st.markdown("<hr style='margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">OBSERVABILITY TELEMETRY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Generation Output & Neural Risk HUD</div>', unsafe_allow_html=True)

    # 1. Primary Generation Output Window (Linear/Railway Style)
    resp_text = html.escape(res["generated_response"])
    cur_temp = float(st.session_state.get("pref_temperature", 0.0))
    temp_str = "T=0 GREEDY" if cur_temp == 0.0 else f"T={cur_temp:.1f} SAMPLING"
    st.markdown(
        f"""
        <div class="terminal-window">
            <div class="terminal-topbar">
                <div class="terminal-title">PRIMARY GENERATION · {selected_model_name.upper()} ({temp_str})</div>
                <div style="font-family: var(--font-mono); font-size: 11px; color: var(--emerald-text); font-weight: 600;">STATUS: COMPLETED</div>
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
    combined_prob = float(
        res.get(
            "predicted_risk_combined",
            res.get("combined_hallucination_probability", (xgb_prob + lr_prob) / 2.0),
        ) or 0.0
    )
    xgb_pct = xgb_prob * 100.0
    lr_pct = lr_prob * 100.0
    combined_pct = combined_prob * 100.0
    conf_pct = calculate_confidence_score(combined_prob)

    low_cut = float(threshold_low)
    high_cut = float(threshold_high)

    if combined_pct < low_cut:
        current_tier = "Low Risk (Faithful)"
        accent_color = "var(--emerald-text)"
        tier_bg = "var(--emerald-bg)"
        tier_border = "var(--emerald-border)"
        tier_badge = "live-badge-emerald"
        verdict_title = "FAITHFUL / LOW RISK"
        verdict_desc = (
            f"Combined posterior risk ({combined_pct:.1f}%) is safely below {low_cut:.0f}%. "
            f"Dual decision core and consistency probes demonstrate high factual stability."
        )
    elif combined_pct >= high_cut:
        current_tier = "High Hallucination Risk"
        accent_color = "var(--rose-text)"
        tier_bg = "var(--rose-bg)"
        tier_border = "var(--rose-border)"
        tier_badge = "live-badge-rose"
        verdict_title = "ELEVATED RISK OF HALLUCINATION"
        verdict_desc = (
            f"Combined posterior risk ({combined_pct:.1f}%) exceeds {high_cut:.0f}%. "
            f"Dual decision core, internal token entropy, or cross-encoder probes detected significant uncertainty."
        )
    else:
        current_tier = "Moderate Risk (Uncertain)"
        accent_color = "var(--amber-text)"
        tier_bg = "var(--amber-bg)"
        tier_border = "var(--amber-border)"
        tier_badge = "live-badge-amber"
        verdict_title = "BORDERLINE UNCERTAINTY"
        verdict_desc = (
            f"Combined posterior risk ({combined_pct:.1f}%) falls in the intermediate band ({low_cut:.0f}%–{high_cut:.0f}%). "
            f"Stochastic probes show partial semantic divergence."
        )

    delta_diff = lr_pct - xgb_pct
    delta_str = f"{delta_diff:+.1f}% vs XGB"

    int_signals = res.get("internal_signals", {})
    mean_ent = float(int_signals.get("mean_entropy", 0.0) or 0.0)
    raw_ppl = int_signals.get("perplexity")
    if raw_ppl is None:
        log_ppl_val = int_signals.get("log_perplexity")
        if log_ppl_val is not None:
            raw_ppl = max(float(math.exp(float(log_ppl_val))), 1.0)
        else:
            raw_ppl = 1.0
    else:
        raw_ppl = max(float(raw_ppl), 1.0)

    timing_data = res.get("stage_latencies_ms", res.get("timing_ms", {}))
    total_elapsed = res.get("total_latency_ms", sum(timing_data.values()) if timing_data else 0.0)

    # 3. Core Telemetry HUD (Tier 1: Combined Decision Core | Tier 2: Model & Logit Signals)
    # Tier 1: Primary Decision Verdict & Combined Posterior Risk
    h_col1, h_col2 = st.columns([1.1, 1.4], gap="medium")

    with h_col1:
        st.markdown(
            f"""
            <div class="hud-card" style="border-left: 3.5px solid {accent_color}; min-height: 110px;">
                <div class="hud-label-row">
                    <span class="hud-label">COMBINED POSTERIOR RISK</span>
                    <span class="live-badge {tier_badge}" style="font-size: 10px; padding: 1.5px 7px;">DUAL CORE</span>
                </div>
                <div class="hud-value-row" style="margin: 4px 0;">
                    <span class="hud-value" style="font-size: 32px; color: {accent_color};">{combined_pct:.1f}%</span>
                    <span class="live-badge live-badge-primary" style="font-size: 11px; padding: 2px 8px; margin-left: 6px;">
                        {conf_pct:.1f}% CONFIDENCE
                    </span>
                </div>
                <div class="hud-desc" style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                    <span>Calibrated Ensemble (XGBoost + LR)</span>
                    <span>{temp_str}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h_col2:
        st.markdown(
            f"""
            <div class="hud-card" style="background: {tier_bg}; border: 1px solid {tier_border}; border-left: 4px solid {accent_color}; min-height: 110px;">
                <div class="hud-label-row">
                    <span class="hud-label" style="color: var(--text-muted);">DECISION VERDICT</span>
                    <span class="live-badge {tier_badge}" style="font-size: 10px; padding: 1.5px 7px;">{verdict_title}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {accent_color}; box-shadow: 0 0 10px {accent_color}; flex-shrink: 0;"></span>
                    <span style="font-size: 20px; font-weight: 700; color: {accent_color}; letter-spacing: -0.015em;">
                        {current_tier}
                    </span>
                </div>
                <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-secondary); line-height: 1.35;">
                    Active Band: &lt;{low_cut:.0f}% (Faithful) · {low_cut:.0f}%–{high_cut:.0f}% (Uncertain) · &ge;{high_cut:.0f}% (High Risk)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Tier 2: Component Model & Predictive Uncertainty Telemetry (3 Wide Columns)
    s_col1, s_col2, s_col3 = st.columns(3, gap="medium")

    with s_col1:
        st.markdown(
            f"""
            <div class="hud-card" style="min-height: 98px;">
                <div class="hud-label-row">
                    <span class="hud-label">XGBOOST RISK SCORE</span>
                    <span class="hud-tag">TREE SHAP</span>
                </div>
                <div class="hud-value-row">
                    <span class="hud-value">{xgb_pct:.1f}%</span>
                </div>
                <div class="hud-desc" style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
                    Phase 14 Universal Core (19 Features)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s_col2:
        st.markdown(
            f"""
            <div class="hud-card" style="min-height: 98px;">
                <div class="hud-label-row">
                    <span class="hud-label">LOGISTIC REGRESSION</span>
                    <span class="hud-tag">BASELINE</span>
                </div>
                <div class="hud-value-row">
                    <span class="hud-value">{lr_pct:.1f}%</span>
                    <span class="hud-delta" style="color: var(--text-secondary); margin-left: 6px;">{delta_str}</span>
                </div>
                <div class="hud-desc" style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
                    Standardized Linear Baseline vs XGBoost
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s_col3:
        st.markdown(
            f"""
            <div class="hud-card" style="min-height: 98px;">
                <div class="hud-label-row">
                    <span class="hud-label">PREDICTIVE UNCERTAINTY</span>
                    <span class="hud-tag">LOGITS</span>
                </div>
                <div class="hud-value-row">
                    <span class="hud-value">{mean_ent:.3f} <span style="font-size: 14px; font-weight: 500; color: var(--text-muted);">nats</span></span>
                </div>
                <div class="hud-desc" style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
                    Perplexity: <strong>{raw_ppl:.2f}</strong> (exp of NLL)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Decision Rationale Callout Strip
    st.markdown(
        f"""
        <div style="background: {tier_bg}; border: 1px solid {tier_border}; border-left: 4px solid {accent_color};
                    border-radius: 6px; padding: 10px 14px; margin: 10px 0 18px 0; font-size: 12.5px; line-height: 1.5;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px; flex-wrap: wrap;">
                <strong style="color: {accent_color}; font-size: 13px;">{verdict_title}</strong>
                <span class="live-badge {tier_badge}" style="font-size: 10px; padding: 1.5px 7px;">{combined_pct:.1f}% COMBINED POSTERIOR RISK</span>
                <span class="live-badge live-badge-zinc" style="font-size: 10px; padding: 1.5px 7px;">XGB: {xgb_pct:.1f}%</span>
                <span class="live-badge live-badge-zinc" style="font-size: 10px; padding: 1.5px 7px;">LR: {lr_pct:.1f}%</span>
                <span class="live-badge live-badge-primary" style="font-size: 10px; padding: 1.5px 7px;">CONFIDENCE: {conf_pct:.1f}%</span>
                <span style="font-family: var(--font-mono); font-size: 10.5px; color: var(--text-muted); margin-left: auto;">PIPELINE: {total_elapsed/1000.0:.2f}s (15 STAGES)</span>
            </div>
            <div style="color: var(--text-secondary);">
                {verdict_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ==========================================================================
    # 5. Deep-Dive Diagnostic Telemetry Tabs (Organized & Intuitive)
    # ==========================================================================
    evidence_data = res.get("evidence_analysis", {})
    has_grounding = is_evidence_mode or (evidence_data and evidence_data.get("has_reference_context"))

    tab_titles = [
        "Token Uncertainty & Heatmap",
        "Feature Attribution (Tree SHAP)",
        "Signal Channels (19 Features)",
        "Execution Profiler",
    ]
    if has_grounding:
        tab_titles.append("Evidence Grounding")
    tab_titles.append("Audit Export")

    tabs = st.tabs(tab_titles)

    # --- TAB 1: Token Uncertainty & Heatmap ---
    with tabs[0]:
        st.markdown(
            """
            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px;">
                Interactive token highlighting visualizes generation likelihood during the forward pass.
                Tokens with high likelihood and low predictive entropy appear in green; intermediate confidence in yellow;
                and low confidence or elevated entropy in red.
            </div>
            """,
            unsafe_allow_html=True,
        )

        token_details = res.get("token_details", [])
        if token_details:
            total_tokens = len(token_details)
            high_c = sum(1 for t in token_details if t.get("band") == "High confidence")
            med_c = sum(1 for t in token_details if t.get("band") == "Medium confidence")
            low_c = sum(1 for t in token_details if t.get("band") == "Low confidence")

            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            with stat_c1:
                st.metric("Total Tokens", f"{total_tokens}")
            with stat_c2:
                st.metric("High Confidence", f"{high_c}", f"{(high_c/total_tokens*100):.1f}%", delta_color="off")
            with stat_c3:
                st.metric("Medium Confidence", f"{med_c}", f"{(med_c/total_tokens*100):.1f}%", delta_color="off")
            with stat_c4:
                st.metric("Suspicious / Low Conf", f"{low_c}", f"{(low_c/total_tokens*100):.1f}%", delta_color="off")

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
                    bg = "var(--emerald-bg)"
                    border_col = "var(--emerald-border)"
                    txt_col = "var(--emerald-text)"
                    opacity = "0.4" if isolate_suspicious else "1.0"
                elif t_band == "Low confidence":
                    bg = "var(--rose-bg)"
                    border_col = "var(--rose-border)"
                    txt_col = "var(--rose-text)"
                    opacity = "1.0"
                else:
                    bg = "var(--amber-bg)"
                    border_col = "var(--amber-border)"
                    txt_col = "var(--amber-text)"
                    opacity = "0.4" if isolate_suspicious else "1.0"

                tooltip = f"Step {t_step}: '{t_token}' | Prob: {t_prob*100:.1f}% | Entropy: {t_ent:.3f} | Rank: {t_rank} | {t_band}"
                span_html = (
                    f'<span class="heatmap-token-chip" title="{html.escape(tooltip)}" style="background: {bg}; border-bottom: 2px solid {border_col}; '
                    f'color: {txt_col}; opacity: {opacity}; font-weight: 600;">{t_token}</span>'
                )
                heatmap_spans.append(span_html)

            box_html = f'<div class="heatmap-container">\n{"".join(heatmap_spans)}\n</div>'
            st.markdown(box_html, unsafe_allow_html=True)

            st.markdown(
                """
                <div style="display: flex; gap: 18px; font-size: 12px; margin-bottom: 12px; flex-wrap: wrap;">
                    <span><span style="color: var(--emerald-text); font-weight: bold;">● High Confidence</span> (P ≥ 65%)</span>
                    <span><span style="color: var(--amber-text); font-weight: bold;">● Nominal Confidence</span> (30% ≤ P &lt; 65%)</span>
                    <span><span style="color: var(--rose-text); font-weight: bold;">● Sub-Threshold / High Risk</span> (P &lt; 30% or elevated entropy)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
                st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)
        else:
            st.info("No token details recorded for this run.")

    # --- TAB 2: Feature Attribution (Tree SHAP) ---
    with tabs[1]:
        st.markdown(
            """
            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px;">
                Tree SHAP decomposes the XGBoost risk prediction into exact local log-odds contributions
                from each measured feature for this specific generation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        feature_contribs = res.get("feature_contributions", [])
        if feature_contribs:
            c_filter = st.radio(
                "Filter Attributions",
                options=["All Significant Drivers", "Risk Inflators", "Risk Attenuators"],
                horizontal=True,
                label_visibility="collapsed",
                key="tab_shap_filter",
            )

            filtered_contribs = []
            for c in feature_contribs:
                if c_filter == "Risk Inflators" and c["direction"] != "increases risk":
                    continue
                if c_filter == "Risk Attenuators" and c["direction"] == "increases risk":
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
                            <span>TreeSHAP Impact: <strong>{contrib:+.4f}</strong> log-odds</span>
                        </div>
                        <div class="attr-track">
                            <div style="background: {fill_color}; width: {pct_bar:.1f}%; height: 100%; border-radius: 2px;"></div>
                        </div>
                    </div>
                    """).strip()
                    st.markdown(card_html, unsafe_allow_html=True)

            st.caption(
                "*Attributions reflect local Tree SHAP log-odds contributions from the fitted XGBoost model for this query.*"
            )
        else:
            st.info("No feature contributions available.")

    # --- TAB 3: Signal Channels (19 Universal Core) ---
    with tabs[2]:
        tab_internal, tab_sc, tab_nli, tab_vec = st.tabs([
            "Internal Signals (11)",
            "Self-Consistency (5)",
            "NLI Agreement (3)",
            "Canonical 19-Vector",
        ])

        with tab_internal:
            st.markdown(
                "<p style='font-size: 12.5px; color: var(--text-secondary);'>"
                "White-box generation metrics extracted during the greedy forward pass over output logits."
                "</p>",
                unsafe_allow_html=True,
            )
            int_sig = res.get("internal_signals", {})
            internal_table_data = [
                {"Signal": "min_log_prob", "Value": f"{int_sig.get('min_log_prob', 0.0):.4f}", "Interpretation": "Minimum token log-probability (bottleneck token)"},
                {"Signal": "mean_log_prob", "Value": f"{int_sig.get('mean_log_prob', 0.0):.4f}", "Interpretation": "Mean token log-probability across sequence"},
                {"Signal": "mean_token_prob", "Value": f"{int_sig.get('mean_token_prob', 0.0):.4f}", "Interpretation": "Mean token probability"},
                {"Signal": "token_prob_std", "Value": f"{int_sig.get('token_prob_std', 0.0):.4f}", "Interpretation": "Standard deviation of token probabilities"},
                {"Signal": "mean_entropy", "Value": f"{int_sig.get('mean_entropy', 0.0):.4f}", "Interpretation": "Mean predictive Shannon entropy (nats)"},
                {"Signal": "max_entropy", "Value": f"{int_sig.get('max_entropy', 0.0):.4f}", "Interpretation": "Peak predictive entropy in sequence"},
                {"Signal": "entropy_std", "Value": f"{int_sig.get('entropy_std', 0.0):.4f}", "Interpretation": "Standard deviation of predictive entropy"},
                {"Signal": "log_perplexity", "Value": f"{int_sig.get('log_perplexity', 0.0):.4f}", "Interpretation": "Log-transformed sequence perplexity"},
                {"Signal": "log_mean_token_rank", "Value": f"{int_sig.get('log_mean_token_rank', 0.0):.4f}", "Interpretation": "log(1 + mean token rank)"},
                {"Signal": "log_max_token_rank", "Value": f"{int_sig.get('log_max_token_rank', 0.0):.4f}", "Interpretation": "log(1 + max token rank)"},
                {"Signal": "log_rank_std", "Value": f"{int_sig.get('log_rank_std', 0.0):.4f}", "Interpretation": "log(1 + token rank std)"},
            ]
            st.dataframe(pd.DataFrame(internal_table_data), width="stretch", hide_index=True)

        with tab_sc:
            st.markdown(
                "<p style='font-size: 12.5px; color: var(--text-secondary);'>"
                "Five stochastic generations (k=5, T=0.7, top_p=0.9) probed for semantic stability using <code>all-MiniLM-L6-v2</code>."
                "</p>",
                unsafe_allow_html=True,
            )
            sc_sig = res.get("self_consistency_features", {})
            sc_table_data = [
                {"Signal": "exact_match_agreement", "Value": f"{sc_sig.get('exact_match_agreement', 0.0):.4f}", "Interpretation": "Fraction of candidate pairs matching identically"},
                {"Signal": "mean_pairwise_similarity", "Value": f"{sc_sig.get('mean_pairwise_similarity', 0.0):.4f}", "Interpretation": "Mean embedding cosine similarity among candidate outputs"},
                {"Signal": "min_pairwise_similarity", "Value": f"{sc_sig.get('min_pairwise_similarity', 0.0):.4f}", "Interpretation": "Minimum pairwise cosine similarity (worst-case divergence)"},
                {"Signal": "max_pairwise_similarity", "Value": f"{sc_sig.get('max_pairwise_similarity', 0.0):.4f}", "Interpretation": "Maximum pairwise cosine similarity"},
                {"Signal": "pairwise_similarity_std", "Value": f"{sc_sig.get('pairwise_similarity_std', 0.0):.4f}", "Interpretation": "Standard deviation of pairwise cosine similarities"},
            ]
            st.dataframe(pd.DataFrame(sc_table_data), width="stretch", hide_index=True)

            st.markdown("<div style='font-size: 11.5px; font-weight: 600; color: var(--text-muted); margin-top: 12px; margin-bottom: 6px;'>SAMPLED CANDIDATE RESPONSES (k=5 STOCHASTIC PROBES):</div>", unsafe_allow_html=True)
            for i, resp in enumerate(res.get("consistency_responses", []), 1):
                st.markdown(
                    f"""
                    <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px;
                                padding: 7px 12px; margin-bottom: 5px; font-size: 12.5px;">
                        <span style="font-family: var(--font-mono); color: var(--brand-indigo-light); font-weight: 600; margin-right: 8px;">SAMPLE #{i}:</span>
                        <span style="color: var(--text-primary);">{html.escape(resp)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_nli:
            st.markdown(
                "<p style='font-size: 12.5px; color: var(--text-secondary);'>"
                "Natural Language Inference cross-encoder evaluation (<code>nli-MiniLM2-L6-H768</code>) over 10 pairs / 20 directional passes."
                "</p>",
                unsafe_allow_html=True,
            )
            nli_sig = res.get("nli_scores", res.get("nli_features", {}))
            nli_table_data = [
                {"Signal": "mean_pairwise_entailment", "Value": f"{nli_sig.get('mean_pairwise_entailment', 0.0):.4f}", "Interpretation": "Average cross-encoder entailment probability"},
                {"Signal": "mean_pairwise_contradiction", "Value": f"{nli_sig.get('mean_pairwise_contradiction', 0.0):.4f}", "Interpretation": "Average cross-encoder contradiction probability"},
                {"Signal": "nli_disagreement", "Value": f"{nli_sig.get('nli_disagreement', 0.0):.4f}", "Interpretation": "Composite NLI disagreement [contradiction + 0.5 * (1 - entailment)]"},
            ]
            st.dataframe(pd.DataFrame(nli_table_data), width="stretch", hide_index=True)

        with tab_vec:
            st.markdown(
                "<p style='font-size: 12.5px; color: var(--text-secondary);'>"
                "Exact tabular vector passed to the fitted Phase 14 XGBoost and Logistic Regression estimators."
                "</p>",
                unsafe_allow_html=True,
            )
            f_vec = res.get("feature_vector", {})
            vector_rows = []
            for idx, feat in enumerate(UNIVERSAL_CORE_FEATURES, 1):
                vector_rows.append({
                    "#": idx,
                    "Feature Canonical Name": feat,
                    "Standardized Value": f"{f_vec.get(feat, 0.0):.6f}",
                })
            st.dataframe(pd.DataFrame(vector_rows), width="stretch", hide_index=True)

    # --- TAB 4: Execution Profiler ---
    with tabs[3]:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: var(--bg-surface);
                        border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px 16px; margin-bottom: 10px;">
                <div style="font-size: 13px; color: var(--text-secondary);">
                    Total Pipeline Latency: <strong style="color: var(--text-primary); font-family: var(--font-mono);">{total_elapsed / 1000.0:.2f} s</strong>
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
            color_palette = ["#2563EB", "#38BDF8", "#34D399", "#FBBF24", "#FB7185", "#0284C7", "#64748B", "#14B8A6"]
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
                    "Pipeline Stage": s_name,
                    "Latency (ms)": f"{s_ms:.2f}",
                    "Share (%)": f"{pct:.1f}%",
                })
            st.dataframe(pd.DataFrame(timing_rows), width="stretch", hide_index=True)

    # --- TAB 5: Evidence Grounding (if applicable) ---
    curr_tab_idx = 4
    if has_grounding:
        with tabs[curr_tab_idx]:
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
                                        padding: 10px 14px; margin-bottom: 8px; font-size: 12.5px;">
                                <div style="font-family: var(--font-mono); font-size: 11px; color: var(--brand-indigo-light); margin-bottom: 4px;">
                                    RANK #{card['rank']} · QUERY SIM: {card['query_similarity']:.4f} · RESPONSE SIM: {card['response_similarity']:.4f}
                                </div>
                                <div style="color: var(--text-secondary); line-height: 1.5;">{card['text']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    """
                    <div class="linear-notice">
                        <div style="font-family: var(--font-mono); font-size: 10px; padding: 2px 6px; border-radius: 4px; background: var(--cyan-bg); color: var(--cyan-text); border: 1px solid var(--cyan-border); font-weight: 700;">NOTICE</div>
                        <div>
                            <strong>No Reference Context Provided:</strong> Enter a reference passage in the Prompt Studio to compute live groundedness.
                            In offline Phase 19 retrieval benchmarks, reference indexing elevated detection to <strong>ROC-AUC 0.9184</strong>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        curr_tab_idx += 1

    # --- TAB: Audit Export ---
    with tabs[curr_tab_idx]:
        st.markdown(
            """
            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px;">
                Download machine-readable telemetry artifacts for compliance, offline debugging, or research logs.
            </div>
            """,
            unsafe_allow_html=True,
        )

        export_col1, export_col2 = st.columns(2)
        audit_payload = {
            "model_slug": active_model_id,
            "prompt": st.session_state.get("user_prompt", ""),
            "context": st.session_state.get("user_context", ""),
            "generated_response": res["generated_response"],
            "risk_verdict": {
                "combined_probability": combined_prob,
                "combined_risk_pct": combined_pct,
                "confidence_score": conf_pct,
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
        user_prompt_val = st.session_state.get("user_prompt", "")

        with export_col1:
            if "CSV" in export_fmt:
                export_df = pd.DataFrame([{
                    "prompt": user_prompt_val,
                    "generated_response": res["generated_response"],
                    "combined_probability": combined_prob,
                    "combined_risk_pct": combined_pct,
                    "confidence_score": conf_pct,
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
                    width="stretch",
                )
            elif "Parquet" in export_fmt:
                export_df = pd.DataFrame([{
                    "prompt": user_prompt_val,
                    "generated_response": res["generated_response"],
                    "combined_probability": combined_prob,
                    "combined_risk_pct": combined_pct,
                    "confidence_score": conf_pct,
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
                    width="stretch",
                )
            else:
                st.download_button(
                    label="Export Verification Audit (JSON)",
                    data=audit_json,
                    file_name=f"hallucination_audit_{int(total_elapsed)}.json",
                    mime="application/json",
                    width="stretch",
                )

        with export_col2:
            markdown_summary = textwrap.dedent(f"""
            ### Hallucination Observability & Risk Report
            - **Model**: `{active_model_id}`
            - **Verdict**: {current_tier}
            - **Combined Risk**: {combined_pct:.1f}%
            - **Calibrated Confidence**: {conf_pct:.1f}%
            - **XGBoost Risk**: {xgb_pct:.1f}%
            - **Logistic Regression Risk**: {lr_pct:.1f}%
            - **Prompt**: "{user_prompt_val}"
            - **Generation**: "{res['generated_response']}"
            - **Total Pipeline Latency**: {total_elapsed:.1f} ms
            """).strip()

            with st.expander("View Shareable Telemetry Summary (Markdown)", expanded=False):
                st.code(markdown_summary, language="markdown")
