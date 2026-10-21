"""
Sidebar Component: 6-Item Navigation System & Developer Telemetry.
The sidebar strictly contains exactly these six items in this exact order:
1. Detector
2. Benchmark Mode
3. System Health
4. History
5. About
6. Profile (at the very bottom, visually separated)
"""

from typing import List, Tuple
import streamlit as st

NAV_ITEMS: List[Tuple[str, str, str]] = [
    ("Detector", "detector", ":material/radar:"),
    ("Benchmark Mode", "benchmark_mode", ":material/leaderboard:"),
    ("System Health", "system_health", ":material/monitor_heart:"),
    ("History", "history", ":material/history:"),
    ("About", "about", ":material/info:"),
    ("Profile", "profile", ":material/account_circle:"),
]


def render_sidebar():
    """
    Render reconstructed 6-item navigation sidebar with clean typography,
    consistent icons, and bottom-anchored profile navigation.
    """
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "Detector"

    current_nav = st.session_state["active_nav"]

    with st.sidebar:
        # Brand Identity Header with Live Status Badge (Clicking toggles navigation bar)
        st.markdown(
            """
            <div class="sidebar-brand-header" id="sidebarBrandHeader" title="Click logo to toggle navigation bar">
                <div class="sidebar-brand-logo"></div>
                <div class="sidebar-brand-text">
                    <div class="sidebar-brand-title">VeritasLM</div>
                    <div class="sidebar-brand-subtitle">HALLUCINATION DEFENSE</div>
                </div>
                <div class="sidebar-status-pill">
                    <span class="sidebar-pulse-dot"></span> LIVE
                </div>
            </div>
            <div class="sidebar-section-label">NAVIGATION</div>
            """,
            unsafe_allow_html=True,
        )

        # Micro-client script: Bind click on brand header to trigger collapse/expand smoothly
        if hasattr(st, "html"):
            st.html(
                """
                <script>
                (function() {
                    try {
                        function setupBrandToggle() {
                            const brandHeader = document.getElementById('sidebarBrandHeader');
                            if (brandHeader && !brandHeader.dataset.collapseBound) {
                                brandHeader.dataset.collapseBound = 'true';
                                brandHeader.addEventListener('click', function(e) {
                                    e.preventDefault();
                                    const collapseBtn = document.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                                                        document.querySelector('[data-testid="stSidebarCollapseButton"]');
                                    if (collapseBtn) {
                                        collapseBtn.click();
                                    }
                                });
                            }
                        }
                        setupBrandToggle();
                        if (!window._veritaslmSidebarObserver) {
                            window._veritaslmSidebarObserver = new MutationObserver(setupBrandToggle);
                            window._veritaslmSidebarObserver.observe(document.body, { childList: true, subtree: true });
                        }
                    } catch (e) {}
                })();
                </script>
                """,
                unsafe_allow_javascript=True,
            )

        # Main Navigation Group (Detector, Benchmark Mode, System Health, History, About)
        for label, slug, icon in NAV_ITEMS[:-1]:
            is_active = (current_nav == label)
            if st.button(
                label,
                key=f"sidebar_nav_{slug}",
                icon=icon,
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if st.session_state.get("active_nav") != label:
                    st.session_state["active_nav"] = label
                    st.rerun()



        # Visual Separator & Flexible Spacer to push Appearance and Profile to the bottom
        st.markdown(
            """
            <div class="sidebar-nav-spacer"></div>
            """,
            unsafe_allow_html=True,
        )



        # Profile / Operator Section
        st.markdown(
            """
            <div class="sidebar-profile-section">
                <div class="sidebar-section-label" style="margin-bottom: 3px;">ACCOUNT</div>
                <div class="sidebar-operator-card">
                    <div class="sidebar-operator-avatar">OP</div>
                    <div class="sidebar-operator-info">
                        <div class="sidebar-operator-name">Operator</div>
                        <div class="sidebar-operator-role">Neural Defense Lab</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Profile Navigation Item (At the bottom, visually anchored)
        label, slug, icon = NAV_ITEMS[-1]
        is_active = (current_nav == label)
        if st.button(
            label,
            key=f"sidebar_nav_{slug}",
            icon=icon,
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if st.session_state.get("active_nav") != label:
                st.session_state["active_nav"] = label
                st.rerun()

        # Version stamp
        st.markdown(
            """
            <div class="sidebar-version-badge">
                VeritasLM · v2.4.0 Applied ML
            </div>
            """,
            unsafe_allow_html=True,
        )
