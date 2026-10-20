"""
Unit tests validating About & Platform Dossier component structure,
sub-tab configurations, and empirical benchmark integration.
"""

from unittest.mock import patch, MagicMock
import pytest
from app.components.about import render_about_view


def test_render_about_view_renders_without_exception():
    """Verify render_about_view executes cleanly across Streamlit mocks."""
    mock_tabs = [MagicMock() for _ in range(5)]
    def mock_columns(spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(count)]

    with patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.tabs", return_value=mock_tabs) as mock_tabs_call, \
         patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.columns", side_effect=mock_columns), \
         patch("app.components.about.render_validation_results") as mock_val, \
         patch("app.components.about.render_verification_pipeline") as mock_pipe:
        render_about_view()
        mock_tabs_call.assert_called_once_with([
            "15-Stage Architecture",
            "19 Canonical Features",
            "Validation Results",
            "Datasets & Methodology",
            "Technical Stack & Citations",
        ])
        mock_pipe.assert_called_once_with(expanded=True)
        mock_val.assert_called_once_with(expanded=True)


def test_verification_pipeline_no_expander_box_by_default():
    """Verify render_verification_pipeline renders without st.expander box by default."""
    from app.components.benchmarks import render_verification_pipeline

    with patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.expander") as mock_expander:
        render_verification_pipeline()
        mock_expander.assert_not_called()


def test_validation_results_no_expander_box_by_default():
    """Verify render_validation_results renders without st.expander box by default."""
    from app.components.benchmarks import render_validation_results

    mock_tabs = [MagicMock() for _ in range(5)]
    with patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.expander") as mock_expander, \
         patch("streamlit.tabs", return_value=mock_tabs), \
         patch("streamlit.dataframe") as mock_df:
        render_validation_results()
        mock_expander.assert_not_called()

