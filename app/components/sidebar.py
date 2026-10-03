"""
Sidebar Component (Identity & Access, History, Theme Selection).
Enforces zero-scroll, compact fixed viewport design.
"""

import streamlit as st


def render_sidebar():
    """
    Render fixed-height sidebar containing:
    1. Identity & Access Profile
    2. Inference History selector & actions
    3. Theme toggle (Dark Mode / Light Mode)
    """
    with st.sidebar:
        # 1. Identity & Account Profile (Compact)
        st.markdown('<div class="section-kicker" style="margin-bottom: 4px;">IDENTITY & ACCESS</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-user-card" style="padding: 8px 10px; margin-bottom: 6px;">
                <div class="sidebar-avatar" style="width: 28px; height: 28px; font-size: 11px;">AR</div>
                <div style="overflow: hidden; flex: 1;">
                    <div class="sidebar-user-name" style="font-size: 12px;">Aditya R.</div>
                    <div class="sidebar-user-org" style="font-size: 10px;">Veritas Enterprise</div>
                </div>
                <span style="font-size: 9.5px; color: var(--emerald-text); background: var(--emerald-bg); border: 1px solid var(--emerald-border); border-radius: 4px; padding: 2px 5px; font-weight: 600;">ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin: 8px 0 !important;'>", unsafe_allow_html=True)

        # 2. Session Audit History (Dropdown selector: no vertical overflow)
        st.markdown('<div class="section-kicker" style="margin-bottom: 4px;">INFERENCE HISTORY</div>', unsafe_allow_html=True)
        history_items = st.session_state.get("history", [])
        if history_items:
            hist_labels = [
                f"{h.get('time', '')} [{h.get('risk_pct', 0)}%] {h.get('prompt', '')[:18]}"
                for h in history_items[:8]
            ]
            sel_hist_idx = st.selectbox(
                "Recent Audits",
                options=list(range(len(hist_labels))),
                format_func=lambda i: hist_labels[i],
                label_visibility="collapsed",
                key="sidebar_hist_select",
            )
            col_load, col_clear = st.columns([1.3, 1])
            with col_load:
                if st.button("Load Query", key="btn_load_hist_compact", use_container_width=True):
                    st.session_state["user_prompt"] = history_items[sel_hist_idx].get("full_prompt", history_items[sel_hist_idx]["prompt"])
                    st.session_state["user_context"] = history_items[sel_hist_idx].get("full_context", "")
                    st.rerun()
            with col_clear:
                if st.button("Clear", key="btn_clear_hist_compact", use_container_width=True):
                    st.session_state["history"] = []
                    st.rerun()
        else:
            st.markdown('<div style="font-size: 11px; color: var(--text-muted); font-style: italic; padding: 4px 0;">No query audits recorded yet.</div>', unsafe_allow_html=True)

        st.markdown("<hr style='margin: 8px 0 !important;'>", unsafe_allow_html=True)

        # 3. Appearance: Only Two Themes (Dark Mode and Light Mode)
        st.markdown('<div class="section-kicker" style="margin-bottom: 4px;">THEME</div>', unsafe_allow_html=True)

        def _on_theme_radio_changed():
            st.session_state["theme_choice"] = st.session_state["theme_radio_selector"]

        st.radio(
            "Theme",
            options=["Dark Mode", "Light Mode"],
            index=0 if st.session_state.get("theme_choice", "Dark Mode") == "Dark Mode" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="theme_radio_selector",
            on_change=_on_theme_radio_changed,
        )
