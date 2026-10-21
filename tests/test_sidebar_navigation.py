"""
Unit tests validating sidebar navigation structure, ordering, icons,
and session routing state for the Hallucination Detector platform.
"""

from unittest.mock import patch, MagicMock
import pytest
from app.components.sidebar import render_sidebar, NAV_ITEMS


def test_sidebar_items_count_and_exact_order():
    """Verify that NAV_ITEMS contains exactly the specified items in exact order."""
    expected_order = [
        "Detector",
        "Benchmark Mode",
        "System Health",
        "History",
        "About",
        "Profile",
    ]
    actual_labels = [item[0] for item in NAV_ITEMS]
    assert len(NAV_ITEMS) == 6, f"Expected exactly 6 navigation items, got {len(NAV_ITEMS)}"
    assert actual_labels == expected_order, f"Order mismatch: expected {expected_order}, got {actual_labels}"


def test_profile_is_at_bottom():
    """Verify Profile is the last item."""
    assert NAV_ITEMS[-1][0] == "Profile"
    assert NAV_ITEMS[5][0] == "Profile"


def test_unique_slugs_and_valid_icons():
    """Verify slugs are unique and each item has a clean icon specified."""
    slugs = [item[1] for item in NAV_ITEMS]
    assert len(slugs) == len(set(slugs)), "Slugs must be strictly unique"
    for label, slug, icon in NAV_ITEMS:
        assert icon.startswith(":material/"), f"Item {label} icon should be a Material symbol: {icon}"
        assert icon.endswith(":")


def test_render_sidebar_initializes_active_nav():
    """Verify render_sidebar sets default active_nav to 'Detector' if unset."""
    mock_session = {}
    with patch("streamlit.session_state", mock_session), \
         patch("streamlit.sidebar"), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", return_value=False):
        render_sidebar()
        assert mock_session.get("active_nav") == "Detector"


def test_render_sidebar_preserves_active_nav():
    """Verify render_sidebar preserves existing active_nav state."""
    mock_session = {"active_nav": "Benchmark Mode"}
    with patch("streamlit.session_state", mock_session), \
         patch("streamlit.sidebar"), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", return_value=False):
        render_sidebar()
        assert mock_session.get("active_nav") == "Benchmark Mode"


def test_sidebar_button_click_updates_active_nav():
    """Verify clicking a navigation item updates active_nav in session state."""
    mock_session = {"active_nav": "Detector"}
    # Simulate clicking 'About'
    def mock_button(label, **kwargs):
        return label == "About"

    with patch("streamlit.session_state", mock_session), \
         patch("streamlit.sidebar"), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", side_effect=mock_button), \
         patch("streamlit.rerun") as mock_rerun:
        render_sidebar()
        assert mock_session.get("active_nav") == "About"
        mock_rerun.assert_called_once()


def test_benchmark_proof_accepts_expanded_arg():
    """Verify render_benchmark_proof can be called with expanded=True."""
    from app.components.benchmarks import render_benchmark_proof
    with patch("streamlit.markdown"), \
         patch("streamlit.expander") as mock_expander, \
         patch("streamlit.tabs", return_value=[MagicMock() for _ in range(5)]), \
         patch("streamlit.dataframe"):
        render_benchmark_proof(expanded=True)
        mock_expander.assert_any_call(
            "Validation Results (4,000-Sample Holdout Audit)", expanded=True
        )


def test_sidebar_does_not_contain_engine_or_device_telemetry():
    """Verify render_sidebar does not render engine or device telemetry cards."""
    rendered_markdown = []
    def mock_md(content, **kwargs):
        rendered_markdown.append(content)

    mock_session = {"active_nav": "Detector"}
    with patch("streamlit.session_state", mock_session), \
         patch("streamlit.sidebar"), \
         patch("streamlit.markdown", side_effect=mock_md), \
         patch("streamlit.button", return_value=False):
        render_sidebar()

    all_content = " ".join(rendered_markdown)
    assert "sidebar-telemetry-card" not in all_content
    assert "ENGINE" not in all_content
    assert "DEVICE" not in all_content


def test_sidebar_brand_header_has_interactive_toggle_attributes():
    """Verify render_sidebar provides id and title attributes for logo toggle interactivity."""
    rendered_markdown = []
    def mock_md(content, **kwargs):
        rendered_markdown.append(content)

    mock_session = {"active_nav": "Detector"}
    with patch("streamlit.session_state", mock_session), \
         patch("streamlit.sidebar"), \
         patch("streamlit.markdown", side_effect=mock_md), \
         patch("streamlit.button", return_value=False):
        render_sidebar()

    all_content = " ".join(rendered_markdown)
    assert 'id="sidebarBrandHeader"' in all_content
    assert "toggle navigation" in all_content.lower()


def test_custom_styles_includes_navigation_logo_toggle():
    """Verify custom styles include rules replacing arrows with the project logo."""
    from app.styles import apply_custom_styles
    rendered_styles = []
    def mock_md(content, **kwargs):
        rendered_styles.append(content)

    with patch("streamlit.markdown", side_effect=mock_md):
        apply_custom_styles("Dark Mode")

    css = " ".join(rendered_styles)
    assert "stExpandSidebarButton" in css
    assert "stSidebarCollapseButton" in css
    assert "data:image/png;base64," in css
    assert "VeritasLM" in css


