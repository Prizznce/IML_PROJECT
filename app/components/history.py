"""
History Component: Complete Session Detection Audit & Observability Dossier.

Displays all detection records recorded during the active application session:
- Unique Session ID & lifecycle telemetry
- Filterable & searchable audit table/cards
- Detailed input prompt, context, LLM response, risk and confidence scores
- In-memory session containment (automatically cleared when session terminates)
- Actions to reload queries into Detector, delete individual records, or export logs.
"""

import html
from typing import Any, Dict, List
import streamlit as st

from app.session import (
    get_session_id,
    get_session_history,
    clear_session_history,
    reset_session,
    delete_history_record,
    get_session_analytics,
    export_history_to_json,
    export_history_to_csv,
)


def render_history_view():
    """
    Render the master History view for the current active session.
    """
    session_id = get_session_id()
    history = get_session_history()
    analytics = get_session_analytics()

    # 1. Page Header & Session Badge
    st.markdown(
        f"""
        <div class="linear-header">
            <div class="linear-title-row">
                <div>
                    <h1 class="linear-title">Session Audit History</h1>
                    <p class="linear-subtitle" style="margin-top: 4px; margin-bottom: 0;">
                        Live chronological telemetry and hallucination verdicts recorded in current session.
                    </p>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="history-session-chip" title="Active in-memory session ID">
                        <span class="sidebar-pulse-dot"></span>
                        SESSION: <strong>{session_id}</strong>
                    </span>
                    <span class="about-version-chip">{len(history)} AUDITS</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Executive Session Telemetry KPI Strip
    st.markdown(
        f"""
        <div class="history-kpi-grid">
            <div class="history-kpi-card">
                <div class="history-kpi-label">TOTAL AUDITS</div>
                <div class="history-kpi-value">{analytics['total_checks']}</div>
                <div class="history-kpi-sub">Current session checks</div>
            </div>
            <div class="history-kpi-card">
                <div class="history-kpi-label">FAITHFUL VERDICTS</div>
                <div class="history-kpi-value text-emerald">{analytics['faithful_count']} <span class="history-kpi-pct">({analytics['faithful_pct']}%)</span></div>
                <div class="history-kpi-sub">Low hallucination risk</div>
            </div>
            <div class="history-kpi-card">
                <div class="history-kpi-label">HIGH RISK VERDICTS</div>
                <div class="history-kpi-value text-rose">{analytics['high_risk_count']} <span class="history-kpi-pct">({analytics['high_risk_pct']}%)</span></div>
                <div class="history-kpi-sub">Hallucination detected</div>
            </div>
            <div class="history-kpi-card">
                <div class="history-kpi-label">AVG CONFIDENCE</div>
                <div class="history-kpi-value text-primary">{analytics['avg_confidence_pct']}%</div>
                <div class="history-kpi-sub">Classifier certainty</div>
            </div>
            <div class="history-kpi-card">
                <div class="history-kpi-label">AVG LATENCY</div>
                <div class="history-kpi-value">{analytics['avg_latency_ms']} ms</div>
                <div class="history-kpi-sub">15-stage pipeline</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If no records exist in this session yet
    if not history:
        _render_empty_session_state(session_id)
        return

    # 3. Action & Filter Bar
    # 3. Action & Filter Bar
    col_search, col_filter, col_sort = st.columns([2.5, 1.5, 1.5])
    with col_search:
        search_query = st.text_input(
            "Search queries",
            placeholder="Search prompt, context, or response text...",
            label_visibility="collapsed",
            key="hist_search_input",
        ).strip().lower()

    with col_filter:
        tier_filter = st.selectbox(
            "Filter Tier",
            options=["All Tiers", "Faithful", "High Risk", "Indeterminate"],
            label_visibility="collapsed",
            key="hist_tier_filter",
        )

    with col_sort:
        sort_choice = st.selectbox(
            "Sort By",
            options=["Newest First", "Oldest First", "Highest Risk", "Lowest Risk", "Highest Confidence"],
            label_visibility="collapsed",
            key="hist_sort_choice",
        )

    # Session Management & Export Controls Strip
    col_tools_left, col_tools_right = st.columns([2, 3])
    with col_tools_left:
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            json_data = export_history_to_json()
            st.download_button(
                "Export JSON",
                data=json_data,
                file_name=f"veritas_audit_{session_id}.json",
                mime="application/json",
                icon=":material/download:",
                use_container_width=True,
                key="btn_export_hist_json",
            )
        with sub_c2:
            csv_data = export_history_to_csv()
            st.download_button(
                "Export CSV",
                data=csv_data,
                file_name=f"veritas_audit_{session_id}.csv",
                mime="text/csv",
                icon=":material/table_view:",
                use_container_width=True,
                key="btn_export_hist_csv",
            )

    with col_tools_right:
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            if st.button(
                "Clear Session History",
                icon=":material/delete_sweep:",
                use_container_width=True,
                key="btn_clear_hist_main",
                help="Discard all recorded detections for current session",
            ):
                clear_session_history()
                st.toast("Current session audit history cleared.", icon="🗑️")
                st.rerun()

        with act_c2:
            if st.button(
                "Reset Session (New ID)",
                icon=":material/restart_alt:",
                use_container_width=True,
                key="btn_reset_session_main",
                help="Generate a brand new unique session ID and start with fresh empty history",
            ):
                new_id = reset_session()
                st.toast(f"Started new session {new_id}", icon="✨")
                st.rerun()

    # 4. Filter & Sort Records
    filtered_records = list(history)

    # Keyword search
    if search_query:
        filtered_records = [
            r for r in filtered_records
            if search_query in str(r.get("input", {}).get("prompt", r.get("full_prompt", ""))).lower()
            or search_query in str(r.get("input", {}).get("context", r.get("full_context", ""))).lower()
            or search_query in str(r.get("result", {}).get("generated_response", "")).lower()
            or search_query in str(r.get("model", "")).lower()
            or any(
                search_query in str(m.get("model_name", "")).lower()
                or search_query in str(m.get("generated_response", "")).lower()
                for m in r.get("model_summaries", [])
            )
        ]

    # Tier filter
    if tier_filter != "All Tiers":
        filtered_records = [
            r for r in filtered_records
            if r.get("tier", "").lower() == tier_filter.lower()
        ]

    # Sorting
    if sort_choice == "Newest First":
        pass  # already newest first
    elif sort_choice == "Oldest First":
        filtered_records = list(reversed(filtered_records))
    elif sort_choice == "Highest Risk":
        filtered_records.sort(key=lambda x: float(x.get("risk_pct", 0.0)), reverse=True)
    elif sort_choice == "Lowest Risk":
        filtered_records.sort(key=lambda x: float(x.get("risk_pct", 0.0)))
    elif sort_choice == "Highest Confidence":
        filtered_records.sort(key=lambda x: float(x.get("confidence_score", 0.0)), reverse=True)

    # 5. Render Records
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px; margin-bottom: 8px;">
            <div class="sidebar-section-label" style="margin: 0;">
                SHOWING {len(filtered_records)} OF {len(history)} RECORDED DETECTIONS
            </div>
            <div style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">
                SESSION LIFECYCLE: IN-MEMORY ONLY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not filtered_records:
        st.info("No detection records match your active search or filter criteria.")
        return

    for idx, rec in enumerate(filtered_records):
        _render_history_record_card(rec, idx)


def _render_history_record_card(rec: Dict[str, Any], idx: int):
    """
    Render a minimalist history record showing strictly the Question and Model,
    with a View Details toggle button that opens the full card details inline for that specific record.
    """
    is_bm = rec.get("is_benchmark") or rec.get("record_type") == "benchmark"
    rec_id = rec.get("id", f"REC-{idx+1:03d}")

    input_data = rec.get("input", {})
    prompt_text = rec.get("full_prompt", input_data.get("prompt", rec.get("prompt", "")))

    if is_bm:
        models = rec.get("models", [])
        best_model = rec.get("best_model", "N/A")
        best_risk = float(rec.get("best_model_risk", 0.0))
        model_name = f"Benchmark Suite ({len(models)} Models) • Best: {best_model} ({best_risk:.1f}%)"
        tier = "Benchmark"
        tier_badge_class = "live-badge-primary"
    else:
        model_name = rec.get("model", "Qwen 3.5 (0.8B)")
        tier = rec.get("tier", "Indeterminate")
        risk_pct = float(rec.get("risk_pct", 0.0))
        tier_badge_class = (
            "badge-faithful" if tier == "Faithful"
            else ("badge-high-risk" if tier == "High Risk" else "badge-indeterminate")
        )

    safe_prompt = html.escape(str(prompt_text))
    safe_model = html.escape(str(model_name))
    safe_rec_id = html.escape(str(rec_id))
    safe_tier = html.escape(str(tier))

    detail_state_key = f"hist_show_detail_{rec_id}"
    is_open = bool(st.session_state.get(detail_state_key, False))

    col_card_body, col_card_action = st.columns([5.5, 1.5], vertical_alignment="center")

    with col_card_body:
        st.markdown(
            f"""
            <div class="history-minimal-card {'bm-mode' if is_bm else ''}">
                <div class="history-minimal-row">
                    <span class="history-minimal-badge history-minimal-badge-q">QUESTION</span>
                    <span class="history-minimal-text">{safe_prompt}</span>
                </div>
                <div class="history-minimal-row" style="margin-top: 8px;">
                    <span class="history-minimal-badge history-minimal-badge-m">MODEL</span>
                    <span class="history-minimal-model-val">{safe_model}</span>
                    <span class="history-tier-badge {tier_badge_class}" style="font-size: 10px; padding: 2px 7px; margin-left: 6px;">{safe_tier.upper()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_card_action:
        btn_label = "Hide Details" if is_open else "View Details"
        btn_icon = ":material/expand_less:" if is_open else ":material/expand_more:"
        if st.button(
            btn_label,
            key=f"btn_toggle_detail_{rec_id}_{idx}",
            icon=btn_icon,
            type="primary" if is_open else "secondary",
            use_container_width=True,
            help="Open or close complete telemetry, neural signals, and full responses inline",
        ):
            st.session_state[detail_state_key] = not is_open
            st.rerun()

    # If toggled open, render the full card details inline for this specific record
    if is_open:
        with st.container():
            st.markdown(
                """
                <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-medium); border-radius: 8px; padding: 14px 16px; margin: 4px 0 10px 0;">
                """,
                unsafe_allow_html=True,
            )
            _render_full_card_content(rec, idx, key_prefix=f"inline_{rec_id}_{idx}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)


def _render_full_card_content(rec: Dict[str, Any], idx: int, key_prefix: str = "full"):
    """Render the complete view card content for either a single detection or benchmark record."""
    if rec.get("is_benchmark") or rec.get("record_type") == "benchmark":
        _render_full_benchmark_card(rec, idx, key_prefix=key_prefix)
    else:
        _render_full_detection_card(rec, idx, key_prefix=key_prefix)


def _render_full_benchmark_card(rec: Dict[str, Any], idx: int, key_prefix: str = "bm"):
    """Render a unified multi-model benchmark evaluation history card."""
    rec_id = rec.get("id", f"BM-{idx+1:03d}")
    time_str = rec.get("timestamp", rec.get("time", "N/A"))
    models = rec.get("models", [])
    model_summaries = rec.get("model_summaries", [])
    best_model = rec.get("best_model", "N/A")
    best_risk = float(rec.get("best_model_risk", 0.0))
    avg_risk = float(rec.get("risk_pct", 0.0))
    avg_conf = float(rec.get("confidence_score", 75.0))
    telemetry = rec.get("telemetry", {})
    tot_latency = float(telemetry.get("latency_ms", 0.0))
    tot_tokens = int(telemetry.get("tokens", 0))

    prompt_text = rec.get("full_prompt", rec.get("input", {}).get("prompt", ""))
    context_text = rec.get("full_context", rec.get("input", {}).get("context", ""))

    safe_rec_id = html.escape(str(rec_id))
    safe_time = html.escape(str(time_str))
    safe_prompt = html.escape(prompt_text)
    safe_best = html.escape(str(best_model))

    # Build model pill chips
    model_chips_html = []
    for m in model_summaries:
        m_name = html.escape(m.get("model_name", ""))
        m_risk = float(m.get("combined_risk_pct", 0.0))
        m_tier = m.get("tier", "Indeterminate")
        if m_tier == "Faithful":
            badge_cls = "live-badge-emerald"
            border_c = "var(--emerald-border)"
        elif m_tier == "High Risk":
            badge_cls = "live-badge-rose"
            border_c = "var(--rose-border)"
        else:
            badge_cls = "live-badge-amber"
            border_c = "var(--amber-border)"

        chip = (
            f'<div style="background: var(--bg-surface-elevated); border: 1px solid {border_c}; '
            f'border-radius: 6px; padding: 6px 10px; display: flex; align-items: center; justify-content: space-between; gap: 8px; flex: 1; min-width: 170px;">'
            f'<div style="font-size: 12px; font-weight: 600; color: var(--text-primary);">{m_name}</div>'
            f'<span class="live-badge {badge_cls}" style="font-size: 10px; padding: 1px 6px;">{m_risk:.1f}% RISK</span>'
            f'</div>'
        )
        model_chips_html.append(chip)

    models_row_html = "".join(model_chips_html)

    card_html = (
        f'<div class="history-record-card" style="border-left: 4px solid var(--brand-primary); background: var(--bg-surface); margin-bottom: 12px;">'
        f'<div class="history-record-header">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<span class="history-record-id">{safe_rec_id}</span>'
        f'<span class="live-badge live-badge-primary" style="font-size: 10px; padding: 2px 8px; font-weight: 700;">◈ BENCHMARK SUITE</span>'
        f'<span class="history-model-tag" style="background: var(--badge-zinc-bg);">{len(models)} MODELS COMPARED</span>'
        f'<span class="live-badge live-badge-emerald" style="font-size: 10px; padding: 2px 7px;">★ BEST: {safe_best} ({best_risk:.1f}%)</span>'
        f'</div>'
        f'<div class="history-record-meta">'
        f'<span title="Timestamp">🕒 {safe_time}</span>'
        f'<span>⚡ {tot_latency:.0f} ms</span>'
        f'<span>🔤 {tot_tokens} tokens</span>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px 0;">'
        f'{models_row_html}'
        f'</div>'
        f'<div class="history-prompt-preview">'
        f'<div class="history-prompt-label">BENCHMARK INPUT QUERY (SHARED ACROSS ALL MODELS):</div>'
        f'<div class="history-prompt-content">{safe_prompt}</div>'
        f'</div>'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown(f"**Shared Query Prompt:**")
    st.code(prompt_text, language="markdown")
    if context_text:
        st.markdown(f"**Reference Context Passage:**")
        st.code(context_text, language="markdown")

    st.markdown(f"**Side-by-Side Model Answers & Neural Telemetry:**")
    m_tab_names = [m.get("model_name", f"Model {i+1}") for i, m in enumerate(model_summaries)]
    if m_tab_names:
        m_tabs = st.tabs(m_tab_names)
        for m_idx, m_info in enumerate(model_summaries):
            with m_tabs[m_idx]:
                m_resp = m_info.get("generated_response", "")
                col_m_text, col_m_diag = st.columns([1.3, 1.0])
                with col_m_text:
                    st.markdown(f"**Generated Answer ({m_info.get('model_name')}):**")
                    if m_resp:
                        st.code(m_resp, language="markdown")
                    else:
                        st.caption("No response text recorded.")
                with col_m_diag:
                    st.markdown(f"**Neural Observability Metrics:**")
                    st.markdown(
                        f"""
                        - **Decision Verdict:** `{m_info.get('tier')}`
                        - **Combined Risk:** `{m_info.get('combined_risk_pct'):.1f}%`
                        - **XGBoost Risk:** `{m_info.get('xgb_risk_pct'):.1f}%`
                        - **Logistic Reg Risk:** `{m_info.get('lr_risk_pct'):.1f}%`
                        - **Confidence Score:** `{m_info.get('confidence_pct'):.1f}%`
                        - **Mean Entropy:** `{m_info.get('mean_entropy', 0.0):.3f} nats`
                        - **Perplexity:** `{m_info.get('perplexity', 1.0):.2f}`
                        - **Tokens:** `{m_info.get('num_tokens', 0)}`
                        - **Latency:** `{m_info.get('latency_ms', 0.0):.1f} ms`
                        """
                    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    col_b1, col_b2, col_b3 = st.columns([1.5, 1.5, 1.0])

    with col_b1:
        if st.button(
            "Load into Benchmark Mode",
            key=f"{key_prefix}_load_bm_{rec_id}",
            icon=":material/leaderboard:",
            use_container_width=True,
            help="Load this benchmark query and results back into Benchmark Mode",
        ):
            st.session_state["bm_user_prompt"] = prompt_text
            st.session_state["bm_prompt_input_field"] = prompt_text
            st.session_state["bm_user_context"] = context_text
            st.session_state["bm_context_input_field"] = context_text
            st.session_state["bm_results"] = rec.get("benchmark_results")
            st.session_state["bm_prompt_ran"] = prompt_text
            st.session_state["bm_context_ran"] = context_text
            st.session_state["active_nav"] = "Benchmark Mode"
            st.rerun()

    with col_b2:
        if st.button(
            "Load Query into Detector",
            key=f"{key_prefix}_load_det_{rec_id}",
            icon=":material/radar:",
            use_container_width=True,
            help="Load this prompt into single-engine Detector",
        ):
            st.session_state["user_prompt"] = prompt_text
            st.session_state["main_prompt_input_field"] = prompt_text
            st.session_state["user_context"] = context_text
            st.session_state["main_context_input_field"] = context_text
            st.session_state["active_nav"] = "Detector"
            st.rerun()

    with col_b3:
        if st.button(
            "Delete Record",
            key=f"{key_prefix}_del_bm_{rec_id}",
            icon=":material/delete:",
            use_container_width=True,
        ):
            delete_history_record(rec_id)
            st.toast(f"Benchmark record {rec_id} deleted", icon="🗑️")
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


def _render_full_detection_card(rec: Dict[str, Any], idx: int, key_prefix: str = "det"):
    """Render a single high-fidelity detection audit card with all score bars and diagnostics."""
    rec_id = rec.get("id", f"REC-{idx+1:03d}")
    tier = rec.get("tier", "Indeterminate")
    risk_pct = float(rec.get("risk_pct", 0.0))
    conf_score = float(rec.get("confidence_score", 75.0))
    time_str = rec.get("timestamp", rec.get("time", "N/A"))
    model_name = rec.get("model", "Qwen 3.5 (0.8B)")
    telemetry = rec.get("telemetry", {})
    latency = telemetry.get("latency_ms", 0.0)
    tokens = telemetry.get("tokens", 0)
    perplexity = telemetry.get("perplexity", 0.0)

    input_data = rec.get("input", {})
    prompt_text = input_data.get("prompt", rec.get("full_prompt", rec.get("prompt", "")))
    context_text = input_data.get("context", rec.get("full_context", ""))
    result_data = rec.get("result", {})
    gen_text = result_data.get("generated_response", "")

    tier_badge_class = (
        "badge-faithful" if tier == "Faithful"
        else ("badge-high-risk" if tier == "High Risk" else "badge-indeterminate")
    )

    xgb_risk_pct = float(rec.get("xgb_risk_pct", rec.get("result", {}).get("xgb_risk_pct", rec.get("risk_pct", 0.0))))
    lr_risk_pct = float(rec.get("lr_risk_pct", rec.get("result", {}).get("lr_risk_pct", 0.0)))
    combined_risk_pct = float(rec.get("combined_risk_pct", rec.get("result", {}).get("combined_risk_pct", round((xgb_risk_pct + lr_risk_pct) / 2.0, 1))))
    conf_score = float(rec.get("confidence_score", 75.0))

    safe_prompt = html.escape(prompt_text)
    safe_rec_id = html.escape(str(rec_id))
    safe_model = html.escape(str(model_name))
    safe_time = html.escape(str(time_str))

    comb_bar_w = min(100.0, max(4.0, combined_risk_pct))
    xgb_bar_w = min(100.0, max(4.0, xgb_risk_pct))
    lr_bar_w = min(100.0, max(4.0, lr_risk_pct))
    conf_bar_w = min(100.0, max(4.0, conf_score))

    risk_color_cls = f"risk-fill-{tier.lower().replace(' ', '-')}"

    card_html = (
        f'<div class="history-record-card" style="margin-bottom: 12px;">'
        f'<div class="history-record-header">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span class="history-record-id">{safe_rec_id}</span>'
        f'<span class="history-tier-badge {tier_badge_class}">{tier.upper()}</span>'
        f'<span class="history-model-tag">{safe_model}</span>'
        f'</div>'
        f'<div class="history-record-meta">'
        f'<span title="Timestamp">🕒 {safe_time}</span>'
        f'<span>⚡ {latency} ms</span>'
        f'<span>🔤 {tokens} tokens</span>'
        f'</div>'
        f'</div>'
        f'<div class="history-score-bar-row">'
        f'<div class="history-score-item">'
        f'<span class="history-score-label">COMBINED RISK:</span>'
        f'<strong class="history-score-val">{combined_risk_pct}%</strong>'
        f'<div class="history-mini-bar-track">'
        f'<div class="history-mini-bar-fill {risk_color_cls}" style="width:{comb_bar_w}%;"></div>'
        f'</div>'
        f'</div>'
        f'<div class="history-score-item">'
        f'<span class="history-score-label">XGB RISK:</span>'
        f'<strong class="history-score-val">{xgb_risk_pct}%</strong>'
        f'<div class="history-mini-bar-track">'
        f'<div class="history-mini-bar-fill {risk_color_cls}" style="width:{xgb_bar_w}%;"></div>'
        f'</div>'
        f'</div>'
        f'<div class="history-score-item">'
        f'<span class="history-score-label">LR RISK:</span>'
        f'<strong class="history-score-val">{lr_risk_pct}%</strong>'
        f'<div class="history-mini-bar-track">'
        f'<div class="history-mini-bar-fill {risk_color_cls}" style="width:{lr_bar_w}%;"></div>'
        f'</div>'
        f'</div>'
        f'<div class="history-score-item">'
        f'<span class="history-score-label">CONFIDENCE:</span>'
        f'<strong class="history-score-val">{conf_score}%</strong>'
        f'<div class="history-mini-bar-track">'
        f'<div class="history-mini-bar-fill conf-fill" style="width:{conf_bar_w}%;"></div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown("**User Query Prompt:**")
        st.code(prompt_text, language="markdown")
        if context_text:
            st.markdown("**Reference Context Passage:**")
            st.code(context_text, language="markdown")
        else:
            st.caption("No reference context provided (Open-ended query mode).")

    with col_out:
        st.markdown("**Generated Model Response:**")
        if gen_text:
            st.code(gen_text, language="markdown")
        else:
            st.caption("No response string recorded.")

        st.markdown("**Neural Signal Diagnostics:**")
        st.markdown(
            f"""
            - **Combined Risk:** `{combined_risk_pct}%`
            - **XGBoost Risk:** `{xgb_risk_pct}%`
            - **Logistic Regression Risk:** `{lr_risk_pct}%`
            - **Confidence Score:** `{conf_score}%`
            - **Perplexity:** `{perplexity:.2f}`
            - **Latency:** `{latency:.1f} ms`
            - **Output Tokens:** `{tokens}`
            """
        )

    col_btn_load, col_btn_del = st.columns([1.5, 1])
    with col_btn_load:
        if st.button(
            "Load Query into Detector",
            key=f"{key_prefix}_load_{rec_id}",
            icon=":material/upload:",
            use_container_width=True,
        ):
            st.session_state["user_prompt"] = prompt_text
            st.session_state["main_prompt_input_field"] = prompt_text
            st.session_state["user_context"] = context_text
            st.session_state["main_context_input_field"] = context_text
            st.session_state["active_nav"] = "Detector"
            st.rerun()

    with col_btn_del:
        if st.button(
            "Delete Record",
            key=f"{key_prefix}_del_{rec_id}",
            icon=":material/delete:",
            use_container_width=True,
        ):
            delete_history_record(rec_id)
            st.toast(f"Record {rec_id} deleted", icon="🗑️")
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


def _render_empty_session_state(session_id: str):
    """Render a modern Linear-style empty state when no detections have been performed."""
    st.markdown(
        f"""
        <div class="history-empty-container">
            <div class="history-empty-icon">◈</div>
            <h3 class="history-empty-title">Session {session_id} Initialized</h3>
            <p class="history-empty-desc">
                No hallucination detection activity has been recorded in this session yet.<br>
                Every prompt evaluated in the Detector will automatically append its input,
                verdict, confidence score, and neural telemetry here.
            </p>
            <div class="history-empty-badge-row">
                <span class="history-empty-pill">⚡ Real-time Logging</span>
                <span class="history-empty-pill">🔒 In-Memory Only</span>
                <span class="history-empty-pill">🧹 Auto-clears on Session Termination</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_empty_spacer_l, col_empty_btn, col_empty_spacer_r = st.columns([2, 1.5, 2])
    with col_empty_btn:
        if st.button(
            "Open Detector",
            type="primary",
            icon=":material/radar:",
            use_container_width=True,
            key="btn_empty_go_detector",
        ):
            st.session_state["active_nav"] = "Detector"
            st.rerun()
