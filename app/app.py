"""
Catching an LLM Lying: Real-Time Hallucination Detection Engine.
UI Architecture inspired by Linear and Railway design systems:
- Obsidian & zinc surfaces (#08090C, #0E1017, #151822) with subtle 1px borders
- Inter & JetBrains Mono typography with tabular numerals
- Railway-style telemetry inspector, terminal output, and token heatmap
- Linear-style feature attribution bars and status badges
- Clean segmented controls and zero AI-slop visual clutter
"""

import html
from pathlib import Path
import sys
import textwrap

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import streamlit as st
import torch

from app.inference import (
    UNIVERSAL_CORE_FEATURES,
    load_inference_models,
    run_live_inference,
)


@st.cache_resource(show_spinner="Initializing neural pipelines and cached weights...")
def get_cached_models(model_id: str = "Qwen/Qwen3.5-0.8B"):
    """
    Load and persist models in Streamlit's resource cache across sessions and reruns.
    Supports Qwen3.5-0.8B, Llama 3.2 1B Instruct, and Gemma 3 1B IT.
    """
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return load_inference_models(
        device=target_device,
        model_id=model_id,
    )


# ==============================================================================
# 1. Page Configuration & Linear/Railway Custom Design System
# ==============================================================================

st.set_page_config(
    page_title="Catching an LLM Lying · Hallucination Observability",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System (Linear & Railway Aesthetic)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* --- Root Theme Tokens --- */
:root {
    --bg-canvas: #08090C;
    --bg-surface: #0E1017;
    --bg-surface-elevated: #151822;
    --bg-surface-hover: #1C202C;
    --bg-surface-active: #222736;
    
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-medium: rgba(255, 255, 255, 0.12);
    --border-strong: rgba(255, 255, 255, 0.18);
    
    --brand-indigo: #5E6AD2;
    --brand-indigo-light: #7E89E8;
    --brand-indigo-glow: rgba(94, 106, 210, 0.3);
    
    --emerald-text: #34D399;
    --emerald-bg: rgba(16, 185, 129, 0.08);
    --emerald-border: rgba(16, 185, 129, 0.24);
    --emerald-glow: rgba(16, 185, 129, 0.25);
    
    --amber-text: #FBBF24;
    --amber-bg: rgba(245, 158, 11, 0.08);
    --amber-border: rgba(245, 158, 11, 0.24);
    
    --rose-text: #FB7185;
    --rose-bg: rgba(244, 63, 94, 0.09);
    --rose-border: rgba(244, 63, 94, 0.28);
    
    --cyan-text: #38BDF8;
    --cyan-bg: rgba(56, 189, 248, 0.08);
    --cyan-border: rgba(56, 189, 248, 0.24);
    
    --text-primary: #F4F4F6;
    --text-secondary: #A1A1AA;
    --text-muted: #71717A;
    --text-dim: #52525B;
    
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* --- Base Canvas Overhauls --- */
html, body, [class*="css"], .stApp {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-canvas) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.012em;
}

/* Remove default Streamlit top header padding */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px !important;
}

/* --- Scrollbars --- */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-canvas);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.14);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.25);
}

/* --- Sidebar (Railway Project Inspector Style) --- */
[data-testid="stSidebar"] {
    background-color: #0B0D13 !important;
    border-right: 1px solid var(--border-subtle) !important;
    padding-top: 1.5rem !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 1.2rem 0 !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #12151F !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 7px !important;
    color: var(--text-primary) !important;
    font-size: 13.5px !important;
}

/* --- Typography Utilities --- */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
}

/* --- Linear / Railway Command Header Component --- */
.linear-header {
    margin-bottom: 22px;
}

.linear-breadcrumbs {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.linear-breadcrumbs span.active {
    color: var(--brand-indigo-light);
    font-weight: 600;
}

.linear-title-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 8px;
}

.linear-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: #FFFFFF;
    margin: 0;
}

.linear-subtitle {
    font-size: 14.5px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0 0 16px 0;
}

/* Status Bar & Telemetry Badges */
.status-bar-container {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    margin-bottom: 18px;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 9px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
}

.live-badge-primary {
    background: rgba(94, 106, 210, 0.12);
    color: #8C98F5;
    border-color: rgba(94, 106, 210, 0.3);
}

.live-badge-emerald {
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border-color: var(--emerald-border);
}

.live-badge-cyan {
    background: var(--cyan-bg);
    color: var(--cyan-text);
    border-color: var(--cyan-border);
}

.live-badge-zinc {
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-secondary);
    border-color: var(--border-subtle);
}

.pulse-dot {
    width: 6.5px;
    height: 6.5px;
    border-radius: 50%;
    background-color: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse-ring {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* Linear Notice Banner */
.linear-notice {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: rgba(14, 16, 23, 0.7);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--brand-indigo);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 20px;
    font-size: 12.5px;
    line-height: 1.55;
    color: var(--text-secondary);
}

.linear-notice strong {
    color: var(--text-primary);
}

/* --- Section Dividers & Headers --- */
.section-kicker {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.section-title {
    font-size: 17px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    margin-bottom: 12px;
}

/* --- Inputs & Interactive Fields --- */
.stTextArea textarea {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    transition: all 0.18s ease !important;
    padding: 10px 12px !important;
}

.stTextArea textarea:focus {
    border-color: var(--brand-indigo) !important;
    box-shadow: 0 0 0 1px var(--brand-indigo), 0 0 14px -2px var(--brand-indigo-glow) !important;
    background-color: #11131B !important;
}

.stTextArea textarea::placeholder {
    color: var(--text-dim) !important;
}

/* Radio Buttons (Linear Segmented Control) */
.stRadio div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    background: var(--bg-surface);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
}

.stRadio div[role="radiogroup"] > label {
    background: transparent !important;
    padding: 6px 14px !important;
    border-radius: 6px !important;
    border: 1px solid transparent !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

.stRadio div[role="radiogroup"] > label:hover {
    color: #FFFFFF !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

/* --- Buttons --- */
/* Primary Action Button (Linear Electric CTA) */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #5E6AD2 0%, #4D59C2 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    letter-spacing: 0.01em !important;
    padding: 8px 20px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.25), 0 0 14px -2px var(--brand-indigo-glow) !important;
    transition: all 0.16s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(180deg, #6B77DE 0%, #5562CE 100%) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.35), 0 0 22px 0px rgba(94, 106, 210, 0.5) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"]:active, .stButton > button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(0px);
}

/* Secondary Button (Dark Sleek Chip) */
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 12.5px !important;
    padding: 7px 12px !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--bg-surface-elevated) !important;
    color: #FFFFFF !important;
    border-color: var(--border-medium) !important;
    transform: translateY(-1px);
}

/* --- Railway Terminal Container for Output --- */
.terminal-window {
    background: #0B0D13;
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
}

.terminal-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #11141D;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border-subtle);
}

.terminal-dots {
    display: flex;
    gap: 6px;
}

.terminal-dot {
    width: 8.5px;
    height: 8.5px;
    border-radius: 50%;
}
.dot-red { background: #FF5F56; }
.dot-yellow { background: #FFBD2E; }
.dot-green { background: #27C93F; }

.terminal-title {
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}

.terminal-body {
    padding: 16px 18px;
    font-size: 14.5px;
    line-height: 1.65;
    color: #ECECF1;
}

/* --- Linear Telemetry / Risk Metrics Grid --- */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 22px;
}

@media (max-width: 800px) {
    .telemetry-grid {
        grid-template-columns: 1fr;
    }
}

.telemetry-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}

.telemetry-card:hover {
    border-color: var(--border-medium);
}

/* --- Linear Telemetry Cards for Streamlit Metrics --- */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    transition: border-color 0.2s ease, transform 0.15s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-medium) !important;
    transform: translateY(-1px);
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.03em !important;
    font-variant-numeric: tabular-nums !important;
}

.telemetry-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.telemetry-label {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}

.telemetry-tag {
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-secondary);
}

.telemetry-value-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 8px;
}

.telemetry-value {
    font-family: var(--font-mono);
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #FFFFFF;
    font-variant-numeric: tabular-nums;
}

.telemetry-subtext {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.4;
}

/* Telemetry Gauge Bar */
.gauge-track {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 2px;
    margin-top: 12px;
    overflow: hidden;
    position: relative;
}

.gauge-fill-emerald {
    height: 100%;
    background: linear-gradient(90deg, #10B981, #34D399);
    border-radius: 2px;
}

.gauge-fill-amber {
    height: 100%;
    background: linear-gradient(90deg, #F59E0B, #FBBF24);
    border-radius: 2px;
}

.gauge-fill-rose {
    height: 100%;
    background: linear-gradient(90deg, #E11D48, #FB7185);
    border-radius: 2px;
}

/* Status Pill in Card 3 */
.tier-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-top: 4px;
}

.tier-emerald {
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
}

.tier-amber {
    background: var(--amber-bg);
    color: var(--amber-text);
    border: 1px solid var(--amber-border);
}

.tier-rose {
    background: var(--rose-bg);
    color: var(--rose-text);
    border: 1px solid var(--rose-border);
}

/* --- Token Heatmap Inspector --- */
.heatmap-card {
    background: #0B0D13;
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    line-height: 2.3;
}

.heatmap-legend {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 8px;
    font-family: var(--font-sans);
}

.heatmap-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.legend-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
}
.dot-emerald { background: #34D399; box-shadow: 0 0 6px rgba(52, 211, 153, 0.4); }
.dot-amber { background: #FBBF24; box-shadow: 0 0 6px rgba(251, 191, 36, 0.4); }
.dot-rose { background: #FB7185; box-shadow: 0 0 6px rgba(251, 113, 133, 0.4); }

/* --- Feature Attribution Impact Rows (Linear Style) --- */
.attr-row {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 7px;
    transition: border-color 0.15s ease;
}

.attr-row:hover {
    border-color: var(--border-medium);
}

.attr-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.attr-name {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
}

.attr-id {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-muted);
    margin-left: 6px;
}

.attr-badge {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 4px;
}

.attr-badge-rose {
    background: var(--rose-bg);
    color: var(--rose-text);
    border: 1px solid var(--rose-border);
}

.attr-badge-emerald {
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
}

.attr-metrics {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    margin-bottom: 6px;
}

.attr-track {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 2px;
    overflow: hidden;
}

/* --- Tabs (Linear Segmented Tabs) --- */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    gap: 4px !important;
    padding-bottom: 0px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--brand-indigo) !important;
}

/* --- Expanders (Linear Drawer View) --- */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    overflow: hidden;
    margin-bottom: 12px;
}

[data-testid="stExpander"] summary {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    padding: 10px 14px !important;
    background: var(--bg-surface) !important;
    transition: all 0.15s ease !important;
}

[data-testid="stExpander"] summary:hover {
    color: #FFFFFF !important;
    background: var(--bg-surface-elevated) !important;
}

[data-testid="stExpander"] div[role="region"] {
    padding: 14px 16px !important;
    background: #090B10 !important;
    border-top: 1px solid var(--border-subtle) !important;
}

/* --- DataFrames --- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 7px !important;
    overflow: hidden;
}

/* --- Sidebar Telemetry Card --- */
.spec-box {
    background: #11141D;
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    padding: 12px;
    margin-top: 10px;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 12px;
}

.spec-row:last-child {
    border-bottom: none;
}

.spec-label {
    color: var(--text-muted);
}

.spec-val {
    font-family: var(--font-mono);
    color: var(--text-primary);
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. Sidebar Navigation & Model Specifications
# ==============================================================================

with st.sidebar:
    st.markdown('<div class="section-kicker">INFRASTRUCTURE</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size: 17px; margin-top: 0; margin-bottom: 12px;">Active Model Engine</h3>', unsafe_allow_html=True)

    model_options = {
        "Qwen 3.5 (0.8B)": "Qwen/Qwen3.5-0.8B",
        "Llama 3.2 (1B Instruct)": "meta-llama/Llama-3.2-1B-Instruct",
        "Gemma 3 (1B IT)": "google/gemma-3-1b-it",
    }

    selected_model_name = st.selectbox(
        "Generative LLM",
        list(model_options.keys()),
        index=0,
        help="Select which causal generative LLM produces responses and generation signals.",
        label_visibility="collapsed",
    )
    active_model_id = model_options[selected_model_name]

    # Model specifications card
    if "Llama" in selected_model_name:
        p_count = "1.23 B"
        roc_val = "0.7450 (XGB) / 0.6929 (LR)"
        pr_val = "0.7492 (XGB) / 0.6681 (LR)"
        acc_val = "66.88% (XGB)"
        ece_val = "0.0392"
    elif "Gemma" in selected_model_name:
        p_count = "1.00 B"
        roc_val = "0.7913 (XGB) / 0.7092 (LR)"
        pr_val = "0.7914 (XGB) / 0.7054 (LR)"
        acc_val = "71.50% (XGB)"
        ece_val = "0.0410"
    else:
        p_count = "0.80 B"
        roc_val = "0.7688 (XGB) / 0.7079 (LR)"
        pr_val = "0.7795 (XGB) / 0.7124 (LR)"
        acc_val = "67.88% (XGB)"
        ece_val = "0.0341"

    st.markdown(
        textwrap.dedent(f"""
        <div class="spec-box">
            <div class="spec-row">
                <span class="spec-label">Model Slug</span>
                <span class="spec-val"><code>{active_model_id.split('/')[-1]}</code></span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Parameters</span>
                <span class="spec-val">{p_count}</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Holdout ROC-AUC</span>
                <span class="spec-val">{roc_val}</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Holdout PR-AUC</span>
                <span class="spec-val">{pr_val}</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Holdout Accuracy</span>
                <span class="spec-val">{acc_val}</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">ECE Calibration</span>
                <span class="spec-val">{ece_val}</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Engine State</span>
                <span class="spec-val" style="color: var(--emerald-text);">● Ready (Optimized)</span>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">FEATURE TAXONOMY</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 12.5px; line-height: 1.8; color: var(--text-secondary); margin-top: 4px;">
            <div>• <strong style="color: #ECECF1;">11 Internal Signals</strong> (Logits & Entropy)</div>
            <div>• <strong style="color: #ECECF1;">5 Self-Consistency Probes</strong> (k=5 Sampling)</div>
            <div>• <strong style="color: #ECECF1;">3 NLI Agreement Passes</strong> (Cross-Encoder)</div>
            <div style="margin-top: 6px; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
                LENGTH ISOLATED · NO SHORTCUTS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. Main Header & Observability Bar
# ==============================================================================

dev_label = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "HOST: CPU"

st.markdown(
    f"""
    <div class="linear-header">
        <div class="linear-breadcrumbs">
            <span>IML_PROJECT</span> / <span>OBSERVABILITY</span> / <span class="active">HALLUCINATION-DETECTOR</span>
        </div>
        <div class="linear-title-row">
            <h1 class="linear-title">Catching an LLM Lying</h1>
        </div>
        <p class="linear-subtitle">
            Lightweight real-time hallucination detection from internal generation signals and multi-pass consistency dynamics.
        </p>
        <div class="status-bar-container">
            <span class="live-badge live-badge-emerald">
                <span class="pulse-dot"></span>
                ACTIVE: {selected_model_name}
            </span>
            <span class="live-badge live-badge-primary">
                19-FEATURE UNIVERSAL CORE
            </span>
            <span class="live-badge live-badge-cyan">
                XGBOOST + LOGISTIC REGRESSION
            </span>
            <span class="live-badge live-badge-zinc">
                {dev_label}
            </span>
            <span class="live-badge live-badge-zinc">
                HOLD-OUT N=800 BENCHMARK
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="linear-notice">
        <div>ℹ️</div>
        <div>
            <strong>Academic Research Protocol:</strong> Probability scores represent model-estimated posterior hallucination risk,
            not an axiomatic factual guarantee. All benchmark evaluations reported were conducted on a standardized, stratified
            holdout test partition (4,000 instances balanced 50% faithful / 50% hallucinated across HaluEval, TruthfulQA, and FEVER).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 4. Interactive Input & Mode Controls
# ==============================================================================

if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""
if "user_context" not in st.session_state:
    st.session_state["user_context"] = ""
if "inference_result" not in st.session_state:
    st.session_state["inference_result"] = None

st.markdown('<div class="section-kicker">CONTROL PANEL</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Query Configuration & Presets</div>', unsafe_allow_html=True)

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
        <div style="font-size: 12px; color: var(--cyan-text); background: var(--cyan-bg); border: 1px solid var(--cyan-border);
                    border-radius: 6px; padding: 8px 12px; margin-bottom: 14px;">
            <strong>Evidence Groundedness Active:</strong> When reference context is provided below, the pipeline computes
            query-evidence similarity, response-evidence similarity, and retrieval agreement using <code>all-MiniLM-L6-v2</code>.
            <em>(Phase 19 benchmark achieved ROC-AUC 0.9184 on 2,500 HaluEval+FEVER instances).</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Quick Preset Buttons (Linear style action chips)
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    if st.button("Moon Landing · Factual QA", use_container_width=True):
        st.session_state["user_prompt"] = "Who was the first person to walk on the Moon?"
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

with p_col2:
    if st.button("Capital of France · Factoid", use_container_width=True):
        st.session_state["user_prompt"] = "What is the capital of France?"
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

with p_col3:
    if st.button("Walk on Mars · Stress Test", use_container_width=True):
        st.session_state["user_prompt"] = "Who was the first human to walk on Mars?"
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

with p_col4:
    if st.button("Clear Input · Reset", use_container_width=True):
        st.session_state["user_prompt"] = ""
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

# Input text areas
prompt_text = st.text_area(
    "Prompt",
    value=st.session_state["user_prompt"],
    placeholder="Enter a factual question or proposition (e.g. Who was the first person to walk on the Moon?)...",
    height=95,
    label_visibility="collapsed",
)

context_text = st.text_area(
    "Optional Reference Context",
    value=st.session_state["user_context"],
    placeholder="Optional supporting evidence, background passage, or reference document...",
    height=75,
    label_visibility="collapsed",
)

btn_c1, btn_c2 = st.columns([1.5, 4])
with btn_c1:
    analyze_clicked = st.button("Run Detection Pipeline ↵", type="primary", use_container_width=True)


# ==============================================================================
# 5. Live Pipeline Execution
# ==============================================================================

if analyze_clicked:
    clean_p = prompt_text.strip()
    if not clean_p:
        st.warning("Please provide a prompt or select a sample preset before executing.")
    else:
        st.session_state["user_prompt"] = clean_p
        st.session_state["user_context"] = context_text.strip()

        with st.spinner("Executing 15-stage neural inference pipeline..."):
            try:
                cached_models = get_cached_models(model_id=active_model_id)
                result = run_live_inference(
                    prompt=clean_p,
                    context=context_text.strip() if context_text.strip() else None,
                    device=cached_models.get("device"),
                    causal_model=cached_models["causal_model"],
                    tokenizer=cached_models["tokenizer"],
                    embedding_model=cached_models["embedding_model"],
                    nli_model=cached_models["nli_model"],
                    nli_label_indices=cached_models["nli_label_indices"],
                    lr_pipeline=cached_models["lr_pipeline"],
                    xgb_model=cached_models["xgb_model"],
                    is_evidence_mode=is_evidence_mode,
                )
                st.session_state["inference_result"] = result
            except Exception as exc:
                st.error(f"Inference execution failed: {str(exc)}")
                st.session_state["inference_result"] = None


# ==============================================================================
# 6. Primary Results & Telemetry Dashboard
# ==============================================================================

res = st.session_state.get("inference_result")

if res is not None:
    st.markdown("<hr style='margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">OBSERVABILITY TELEMETRY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Generation Output & Risk Telemetry</div>', unsafe_allow_html=True)

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

    # 2. Risk Metrics Card Layout (Native 3-Column Telemetry)
    m_col1, m_col2, m_col3 = st.columns(3)

    xgb_prob = res.get("predicted_risk_xgb", res.get("xgb_hallucination_probability", 0.0))
    lr_prob = res.get("predicted_risk_lr", res.get("lr_hallucination_probability", 0.0))
    xgb_pct = xgb_prob * 100.0
    lr_pct = lr_prob * 100.0
    risk_label = res["risk_level"]

    with m_col1:
        st.metric(
            label="XGBoost Hallucination Risk",
            value=f"{xgb_pct:.1f}%",
            help="Primary non-linear tree ensemble risk score (Phase 14 Universal Core).",
        )

    with m_col2:
        st.metric(
            label="Logistic Regression Risk",
            value=f"{lr_pct:.1f}%",
            help="Linear baseline model risk score (Phase 14 Universal Core).",
        )

    with m_col3:
        st.metric(
            label="Risk Level Band",
            value=risk_label,
            help="UI visualization category: Low (<35%), Moderate (35–65%), High (≥65%).",
        )

    st.caption(
        "*Note: Probability values represent model-estimated posterior hallucination risks. "
        "The risk level bands (Low Risk / Moderate Risk / High Hallucination Risk) are a UI visualization convention "
        "and do not constitute a validated clinical or safety threshold.*"
    )

    # ==========================================================================
    # 7. Token-by-Token Confidence Heatmap (Inspector View)
    # ==========================================================================
    st.markdown("<hr style='margin: 24px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">STEP-BY-STEP DYNAMICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Token-by-Token Generation Confidence Heatmap</div>', unsafe_allow_html=True)
    st.write(
        "Inline token highlighting visualizes model uncertainty during the forward generation pass. "
        "Green tokens denote high generation likelihood and low predictive entropy; yellow tokens indicate "
        "intermediate confidence; red tokens signal elevated predictive uncertainty or depressed likelihood."
    )

    token_details = res.get("token_details", [])
    if token_details:
        heatmap_spans = []
        for t in token_details:
            t_token = html.escape(t["token"])
            t_prob = t["probability"]
            t_ent = t["entropy"]
            t_rank = t["rank"]
            t_band = t["band"]
            t_step = t["step"]

            if t_band == "High confidence":
                bg = "rgba(46, 204, 113, 0.22)"
                border_col = "#2ecc71"
                txt_col = "#2ecc71"
            elif t_band == "Low confidence":
                bg = "rgba(231, 76, 60, 0.25)"
                border_col = "#e74c3c"
                txt_col = "#e74c3c"
            else:
                bg = "rgba(241, 196, 15, 0.22)"
                border_col = "#f1c40f"
                txt_col = "#f1c40f"

            tooltip = f"Step {t_step}: '{t_token}' | Prob: {t_prob:.4f} | Entropy: {t_ent:.3f} | Rank: {t_rank} | {t_band}"
            span_html = (
                f'<span title="{html.escape(tooltip)}" style="background: {bg}; border-bottom: 2px solid {border_col}; '
                f'color: {txt_col}; padding: 2px 4px; margin: 1px 0px; border-radius: 3px; font-family: \'JetBrains Mono\', monospace; '
                f'font-size: 14.5px; font-weight: 500; display: inline-block; white-space: pre-wrap;">{t_token}</span>'
            )
            heatmap_spans.append(span_html)

        rendered_heatmap = "".join(heatmap_spans)
        box_html = (
            f'<div style="padding: 14px 18px; border-radius: 8px; border: 1px solid var(--border-subtle); '
            f'background: #0B0D13; line-height: 2.2; margin-bottom: 10px;">\n{rendered_heatmap}\n</div>'
        )
        st.markdown(box_html, unsafe_allow_html=True)

        legend_html = """<div style="display: flex; gap: 18px; font-size: 13px; margin-bottom: 10px; flex-wrap: wrap; font-family: var(--font-sans);">
    <span><span style="color: #2ecc71; font-weight: bold;">🟩 High confidence</span> (P ≥ 65%, low entropy)</span>
    <span><span style="color: #f1c40f; font-weight: bold;">🟨 Medium confidence</span> (30% ≤ P &lt; 65%)</span>
    <span><span style="color: #e74c3c; font-weight: bold;">🟥 Low confidence</span> (P &lt; 30% or elevated entropy)</span>
</div>"""
        st.markdown(legend_html, unsafe_allow_html=True)
        st.caption("*Token color represents generation confidence/uncertainty during token production, not factual correctness.*")

        with st.expander("🔍 Inspect Token-by-Token Logit Dynamics Table", expanded=False):
            t_rows = []
            for t in token_details:
                t_rows.append({
                    "Step": t["step"],
                    "Token": repr(t["token"]),
                    "Probability": f"{t['probability']:.4f}",
                    "Log-Prob": f"{t['log_prob']:.4f}",
                    "Entropy": f"{t['entropy']:.4f}",
                    "Rank": t["rank"],
                    "Confidence Band": t["band"],
                })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

    # ==========================================================================
    # 8. Feature Contributions / Tree SHAP Attribution
    # ==========================================================================
    st.markdown("<hr style='margin: 24px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why Was This Response Flagged? (Tree SHAP Attributions)</div>', unsafe_allow_html=True)

    feature_contribs = res.get("feature_contributions", [])
    if feature_contribs:
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
            pct_bar = min(100.0, max(6.0, (mag / max_mag) * 100.0))

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
                    <span>Measured: <strong>{val:.4f}</strong></span>
                    <span>SHAP Impact: <strong>{contrib:+.4f}</strong> log-odds</span>
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
        st.markdown("<hr style='margin: 24px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">GROUNDEDNESS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Reference Evidence Alignment</div>', unsafe_allow_html=True)

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
                        <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px;
                                    padding: 10px 14px; margin-bottom: 8px; font-size: 13px;">
                            <div style="font-family: var(--font-mono); font-size: 11.5px; color: var(--brand-indigo-light); margin-bottom: 4px;">
                                RANK {card['rank']} · QUERY SIM: {card['query_similarity']:.4f} · RESPONSE SIM: {card['response_similarity']:.4f}
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
    st.markdown("<hr style='margin: 24px 0 20px 0;'>", unsafe_allow_html=True)
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
            {"Signal": "min_pairwise_similarity", "Value": f"{sc_sig.get('min_pairwise_similarity', 0.0):.4f}", "Interpretation": "Minimum pairwise cosine similarity"},
            {"Signal": "max_pairwise_similarity", "Value": f"{sc_sig.get('max_pairwise_similarity', 0.0):.4f}", "Interpretation": "Maximum pairwise cosine similarity"},
            {"Signal": "pairwise_similarity_std", "Value": f"{sc_sig.get('pairwise_similarity_std', 0.0):.4f}", "Interpretation": "Standard deviation of pairwise cosine similarities"},
        ]
        st.dataframe(pd.DataFrame(sc_table_data), use_container_width=True, hide_index=True)

        st.markdown("<div style='font-size: 12px; font-weight: 600; color: var(--text-muted); margin-top: 12px;'>SAMPLED RESPONSES (k=5):</div>", unsafe_allow_html=True)
        for i, resp in enumerate(res.get("consistency_responses", []), 1):
            st.markdown(
                f"""
                <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px;
                            padding: 6px 12px; margin-bottom: 5px; font-size: 12.5px; font-family: var(--font-mono);">
                    <span style="color: var(--brand-indigo-light);">#{i}:</span> {resp}
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
    # 11. Pipeline Profiling & Latency Breakdown (Railway deploy-step style)
    # ==========================================================================
    st.markdown("<hr style='margin: 24px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">INSTRUMENTATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Execution Latency & Stage Profiling</div>', unsafe_allow_html=True)

    timing_data = res.get("stage_latencies_ms", res.get("timing_ms", {}))
    total_elapsed = res.get("total_latency_ms", sum(timing_data.values()) if timing_data else 0.0)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: var(--bg-surface);
                    border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
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
        timing_rows = []
        for s_name, s_ms in timing_data.items():
            pct = (s_ms / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            timing_rows.append({
                "Pipeline Execution Stage": s_name,
                "Latency (ms)": f"{s_ms:.2f}",
                "Share (%)": f"{pct:.1f}%",
            })
        st.dataframe(pd.DataFrame(timing_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# 12. Research Benchmark Results (Linear Documentation Style)
# ==============================================================================

st.markdown("<hr style='margin: 32px 0 20px 0;'>", unsafe_allow_html=True)

with st.expander("Authoritative Research Benchmarks (Offline Holdout Split)", expanded=False):
    st.markdown(
        """
        <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 14px;">
            Results evaluated on the standardized, fixed stratified 80/20 holdout test set (3,200 train / 800 test,
            balanced 50% faithful / 50% hallucinated across HaluEval, TruthfulQA, and FEVER).
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_models, tab_ablation, tab_generalization, tab_calibration, tab_retrieval = st.tabs([
        "Multi-Model Benchmarks",
        "Feature-Family Ablation",
        "Cross-Dataset Generalization",
        "Probability Calibration",
        "Retrieval Augmentation",
    ])

    with tab_models:
        multi_model_data = [
            {"LLM Architecture": "Google Gemma 3 (1B IT)", "Classifier": "XGBoost", "Accuracy": "71.50%", "F1 Score": "0.7220", "ROC-AUC": "0.7913", "PR-AUC": "0.7914", "Brier Score": "0.1854", "ECE": "0.0410"},
            {"LLM Architecture": "Google Gemma 3 (1B IT)", "Classifier": "Logistic Regression", "Accuracy": "64.75%", "F1 Score": "0.6667", "ROC-AUC": "0.7092", "PR-AUC": "0.7054", "Brier Score": "0.2140", "ECE": "0.0521"},
            {"LLM Architecture": "Qwen 3.5 (0.8B)", "Classifier": "XGBoost", "Accuracy": "67.88%", "F1 Score": "0.6751", "ROC-AUC": "0.7688", "PR-AUC": "0.7795", "Brier Score": "0.1891", "ECE": "0.0341"},
            {"LLM Architecture": "Qwen 3.5 (0.8B)", "Classifier": "Logistic Regression", "Accuracy": "62.88%", "F1 Score": "0.6356", "ROC-AUC": "0.7079", "PR-AUC": "0.7124", "Brier Score": "0.2151", "ECE": "0.0525"},
            {"LLM Architecture": "Qwen 3.5 (0.8B) [Baseline]", "Classifier": "XGBoost (Phase 6 Internal-Only)", "Accuracy": "67.00%", "F1 Score": "0.6887", "ROC-AUC": "0.7411", "PR-AUC": "0.7322", "Brier Score": "0.1999", "ECE": "0.0228"},
            {"LLM Architecture": "Qwen 3.5 (0.8B) [Baseline]", "Classifier": "Logistic Regression (Phase 6 Internal-Only)", "Accuracy": "62.12%", "F1 Score": "0.6363", "ROC-AUC": "0.6930", "PR-AUC": "0.6618", "Brier Score": "0.2177", "ECE": "0.0649"},
            {"LLM Architecture": "Meta Llama 3.2 (1B Instruct)", "Classifier": "XGBoost", "Accuracy": "66.88%", "F1 Score": "0.6834", "ROC-AUC": "0.7450", "PR-AUC": "0.7492", "Brier Score": "0.1991", "ECE": "0.0392"},
            {"LLM Architecture": "Meta Llama 3.2 (1B Instruct)", "Classifier": "Logistic Regression", "Accuracy": "65.12%", "F1 Score": "0.6610", "ROC-AUC": "0.6929", "PR-AUC": "0.6681", "Brier Score": "0.2211", "ECE": "0.0319"},
        ]
        st.dataframe(pd.DataFrame(multi_model_data), use_container_width=True, hide_index=True)

        st.markdown("<div style='font-size: 12px; font-weight: 600; color: var(--text-muted); margin-top: 14px;'>TASK DOMAIN BREAKDOWN (Qwen 3.5 Universal Core, N=800):</div>", unsafe_allow_html=True)
        subgroup_data = [
            {"Domain / Benchmark": "HaluEval (N=300)", "Task Formulation": "Dialogue, QA, Summarization (with context)", "XGB Accuracy": "86.33%", "XGB ROC-AUC": "0.9433", "LR Accuracy": "84.00%", "LR ROC-AUC": "0.9311"},
            {"Domain / Benchmark": "TruthfulQA (N=300)", "Task Formulation": "Closed-book Human Misconceptions", "XGB Accuracy": "59.33%", "XGB ROC-AUC": "0.6066", "LR Accuracy": "51.33%", "LR ROC-AUC": "0.5210"},
            {"Domain / Benchmark": "FEVER (N=200)", "Task Formulation": "Wikipedia Factoid Verification", "XGB Accuracy": "53.00%", "XGB ROC-AUC": "0.5441", "LR Accuracy": "48.50%", "LR ROC-AUC": "0.4870"},
        ]
        st.dataframe(pd.DataFrame(subgroup_data), use_container_width=True, hide_index=True)

    with tab_ablation:
        ablation_data = [
            {"Configuration": "Internal Signals Only", "Features": 11, "Model": "Logistic Regression", "Accuracy": "62.12%", "F1": "0.6363", "ROC-AUC": "0.6930", "PR-AUC": "0.6618", "Brier": "0.2177", "ECE": "0.0649"},
            {"Configuration": "Internal Signals Only", "Features": 11, "Model": "XGBoost", "Accuracy": "66.25%", "F1": "0.6793", "ROC-AUC": "0.7397", "PR-AUC": "0.7329", "Brier": "0.1993", "ECE": "0.0228"},
            {"Configuration": "Self-Consistency Only", "Features": 5, "Model": "Logistic Regression", "Accuracy": "49.00%", "F1": "0.4545", "ROC-AUC": "0.4845", "PR-AUC": "0.4863", "Brier": "0.2515", "ECE": "0.0300"},
            {"Configuration": "Self-Consistency Only", "Features": 5, "Model": "XGBoost", "Accuracy": "51.00%", "F1": "0.5410", "ROC-AUC": "0.4896", "PR-AUC": "0.4870", "Brier": "0.2567", "ECE": "0.0450"},
            {"Configuration": "NLI Agreement Only", "Features": 3, "Model": "Logistic Regression", "Accuracy": "50.38%", "F1": "0.5031", "ROC-AUC": "0.5050", "PR-AUC": "0.5025", "Brier": "0.2503", "ECE": "0.0088"},
            {"Configuration": "NLI Agreement Only", "Features": 3, "Model": "XGBoost", "Accuracy": "47.13%", "F1": "0.4885", "ROC-AUC": "0.4912", "PR-AUC": "0.5005", "Brier": "0.2543", "ECE": "0.0731"},
            {"Configuration": "Internal + Self-Consistency", "Features": 16, "Model": "XGBoost", "Accuracy": "65.12%", "F1": "0.6610", "ROC-AUC": "0.7563", "PR-AUC": "0.7675", "Brier": "0.1918", "ECE": "0.0450"},
            {"Configuration": "Internal + NLI Agreement", "Features": 14, "Model": "XGBoost", "Accuracy": "67.00%", "F1": "0.6615", "ROC-AUC": "0.7659", "PR-AUC": "0.7697", "Brier": "0.1909", "ECE": "0.0257"},
            {"Configuration": "Self-Consistency + NLI", "Features": 8, "Model": "XGBoost", "Accuracy": "50.62%", "F1": "0.5153", "ROC-AUC": "0.5053", "PR-AUC": "0.5128", "Brier": "0.2541", "ECE": "0.0522"},
            {"Configuration": "All Three / Universal Core", "Features": 19, "Model": "Logistic Regression", "Accuracy": "62.88%", "F1": "0.6356", "ROC-AUC": "0.7079", "PR-AUC": "0.7124", "Brier": "0.2151", "ECE": "0.0525"},
            {"Configuration": "All Three / Universal Core", "Features": 19, "Model": "XGBoost", "Accuracy": "67.88%", "F1": "0.6751", "ROC-AUC": "0.7688", "PR-AUC": "0.7795", "Brier": "0.1891", "ECE": "0.0341"},
        ]
        st.dataframe(pd.DataFrame(ablation_data), use_container_width=True, hide_index=True)

    with tab_generalization:
        lodo_data = [
            {"Held-Out Benchmark (Test)": "FEVER (N=1,000)", "Training Benchmarks": "HaluEval + TruthfulQA (N=3,000)", "Features": "Universal Core (19)", "XGB Accuracy": "50.60%", "XGB ROC-AUC": "0.5167", "LR Accuracy": "49.60%", "LR ROC-AUC": "0.4875"},
            {"Held-Out Benchmark (Test)": "FEVER (N=1,000)", "Training Benchmarks": "HaluEval + TruthfulQA (N=3,000)", "Features": "Internal Only (11)", "XGB Accuracy": "48.30%", "XGB ROC-AUC": "0.4903", "LR Accuracy": "50.60%", "LR ROC-AUC": "0.4868"},
            {"Held-Out Benchmark (Test)": "TruthfulQA (N=1,500)", "Training Benchmarks": "HaluEval + FEVER (N=2,500)", "Features": "Universal Core (19)", "XGB Accuracy": "48.53%", "XGB ROC-AUC": "0.4608", "LR Accuracy": "47.53%", "LR ROC-AUC": "0.4575"},
            {"Held-Out Benchmark (Test)": "TruthfulQA (N=1,500)", "Training Benchmarks": "HaluEval + FEVER (N=2,500)", "Features": "Internal Only (11)", "XGB Accuracy": "47.93%", "XGB ROC-AUC": "0.4539", "LR Accuracy": "46.60%", "LR ROC-AUC": "0.4439"},
            {"Held-Out Benchmark (Test)": "HaluEval (N=1,500)", "Training Benchmarks": "TruthfulQA + FEVER (N=2,500)", "Features": "Universal Core (19)", "XGB Accuracy": "43.20%", "XGB ROC-AUC": "0.4368", "LR Accuracy": "44.93%", "LR ROC-AUC": "0.2583"},
            {"Held-Out Benchmark (Test)": "HaluEval (N=1,500)", "Training Benchmarks": "TruthfulQA + FEVER (N=2,500)", "Features": "Internal Only (11)", "XGB Accuracy": "39.80%", "XGB ROC-AUC": "0.3599", "LR Accuracy": "21.40%", "LR ROC-AUC": "0.1601"},
        ]
        st.dataframe(pd.DataFrame(lodo_data), use_container_width=True, hide_index=True)

    with tab_calibration:
        cal_data = [
            {"Model": "XGBoost", "Calibration Method": "Uncalibrated", "Accuracy": "64.88%", "F1 Score": "0.6492", "ROC-AUC": "0.7638", "PR-AUC": "0.7787", "Brier Score": "0.1900", "ECE": "0.0712", "MCE": "0.1888"},
            {"Model": "XGBoost", "Calibration Method": "Sigmoid (Platt Scaling)", "Accuracy": "65.50%", "F1 Score": "0.6452", "ROC-AUC": "0.7638", "PR-AUC": "0.7787", "Brier Score": "0.1935", "ECE": "0.0825", "MCE": "0.1210"},
            {"Model": "XGBoost", "Calibration Method": "Isotonic Regression", "Accuracy": "66.75%", "F1 Score": "0.6472", "ROC-AUC": "0.7641", "PR-AUC": "0.7557", "Brier Score": "0.1890", "ECE": "0.0244", "MCE": "0.1256"},
            {"Model": "Logistic Regression", "Calibration Method": "Uncalibrated", "Accuracy": "63.38%", "F1 Score": "0.6387", "ROC-AUC": "0.7069", "PR-AUC": "0.7096", "Brier Score": "0.2155", "ECE": "0.0444", "MCE": "0.1248"},
            {"Model": "Logistic Regression", "Calibration Method": "Sigmoid (Platt Scaling)", "Accuracy": "63.00%", "F1 Score": "0.6373", "ROC-AUC": "0.7069", "PR-AUC": "0.7096", "Brier Score": "0.2165", "ECE": "0.0547", "MCE": "0.1114"},
            {"Model": "Logistic Regression", "Calibration Method": "Isotonic Regression", "Accuracy": "62.25%", "F1 Score": "0.6505", "ROC-AUC": "0.6971", "PR-AUC": "0.6751", "Brier Score": "0.2151", "ECE": "0.0341", "MCE": "0.2473"},
        ]
        st.dataframe(pd.DataFrame(cal_data), use_container_width=True, hide_index=True)

    with tab_retrieval:
        retrieval_data = [
            {"Condition": "Condition A (Internal-Only)", "Features": 11, "Model": "Logistic Regression", "Accuracy": "74.60%", "F1": "0.7581", "ROC-AUC": "0.8463", "PR-AUC": "0.8486", "Brier": "0.1597", "ECE": "0.0494"},
            {"Condition": "Condition A (Internal-Only)", "Features": 11, "Model": "XGBoost", "Accuracy": "78.80%", "F1": "0.8000", "ROC-AUC": "0.8978", "PR-AUC": "0.8982", "Brier": "0.1264", "ECE": "0.0323"},
            {"Condition": "Condition B (Universal Core)", "Features": 19, "Model": "Logistic Regression", "Accuracy": "75.80%", "F1": "0.7660", "ROC-AUC": "0.8582", "PR-AUC": "0.8657", "Brier": "0.1538", "ECE": "0.0738"},
            {"Condition": "Condition B (Universal Core)", "Features": 19, "Model": "XGBoost", "Accuracy": "78.60%", "F1": "0.7864", "ROC-AUC": "0.9011", "PR-AUC": "0.9035", "Brier": "0.1246", "ECE": "0.0482"},
            {"Condition": "Condition C (Retrieval-Only)", "Features": 6, "Model": "Logistic Regression", "Accuracy": "65.00%", "F1": "0.6824", "ROC-AUC": "0.6990", "PR-AUC": "0.6543", "Brier": "0.2187", "ECE": "0.0350"},
            {"Condition": "Condition C (Retrieval-Only)", "Features": 6, "Model": "XGBoost", "Accuracy": "68.00%", "F1": "0.7193", "ROC-AUC": "0.7491", "PR-AUC": "0.7293", "Brier": "0.2053", "ECE": "0.0523"},
            {"Condition": "Condition D (Internal + Retrieval)", "Features": 17, "Model": "Logistic Regression", "Accuracy": "76.60%", "F1": "0.7754", "ROC-AUC": "0.8637", "PR-AUC": "0.8675", "Brier": "0.1505", "ECE": "0.0380"},
            {"Condition": "Condition D (Internal + Retrieval)", "Features": 17, "Model": "XGBoost", "Accuracy": "80.60%", "F1": "0.8159", "ROC-AUC": "0.9184", "PR-AUC": "0.9214", "Brier": "0.1180", "ECE": "0.0593"},
            {"Condition": "Condition E (Universal Core + Retrieval)", "Features": 25, "Model": "Logistic Regression", "Accuracy": "76.20%", "F1": "0.7680", "ROC-AUC": "0.8706", "PR-AUC": "0.8823", "Brier": "0.1459", "ECE": "0.0606"},
            {"Condition": "Condition E (Universal Core + Retrieval)", "Features": 25, "Model": "XGBoost", "Accuracy": "81.20%", "F1": "0.8178", "ROC-AUC": "0.9169", "PR-AUC": "0.9201", "Brier": "0.1185", "ECE": "0.0518"},
        ]
        st.dataframe(pd.DataFrame(retrieval_data), use_container_width=True, hide_index=True)


# ==============================================================================
# 13. System Architecture & Methodology (Documentation Drawer)
# ==============================================================================

with st.expander("System Pipeline Architecture & 15 Staged Steps", expanded=False):
    st.markdown("""
### Multi-Model End-to-End System Architecture

```
                       User Question + Optional Context
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       Multi-Model Causal Generative Engine       │
             │  (Qwen 3.5 0.8B / Llama 3.2 1B / Gemma 3 1B IT)  │
             │       Greedy Decoding (T=0, bfloat16/float16)    │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                         Primary Generated Response
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐         ┌──────────────────┐
│  Stage 3 & 4:    │        │  Stage 5, 6 & 7: │         │  Stage 8 & 9:    │
│  11 Internal     │        │  5 Consistency   │         │  3 NLI Agreement │
│  Generation      │        │  Behavioral      │         │  Cross-Encoder   │
│  Signals         │        │  Probes (k=5)    │         │  Passes          │
│                  │        │                  │         │                  │
│ • min/mean log-P │        │ • k=5 stochastic │         │ • nli-MiniLM2-L6 │
│ • mean prob, std │        │   generations    │         │ • 20 directional │
│ • Shannon entropy│        │ • all-MiniLM-L6  │         │   passes (k(k-1))│
│ • log perplexity │        │ • pairwise cos-  │         │ • entailment,    │
│ • rank dispersion│        │   similarity     │         │   contradiction, │
│   (mean/max/std) │        │ • exact match    │         │   disagreement   │
└────────┬─────────┘        └────────┬─────────┘         └────────┬─────────┘
         │                           │                            │
         └───────────────────────────┼────────────────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │            Stage 10:              │
                   │  19 Canonical Universal Features  │
                   │    (Strict Length Isolation)      │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │          Stage 11 & 12:           │
                   │  Supervised Risk Classification   │
                   │ • Tuned XGBoost (Primary Non-Lin) │
                   │ • Calibrated Logistic Regression  │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Hallucination Risk Score       │
                   │   [0.0 - 1.0 Posterior Risk]      │
                   └─────────────────┬─────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│   Stage 13:      │        │   Stage 14:      │        │   Stage 15:      │
│  TreeSHAP Feature│        │  Evidence        │        │  Token Heatmap & │
│  Attributions    │        │  Groundedness    │        │  Latency Profile │
│  ("Why flagged?")│        │  Margin (Opt.)   │        │  (P(y_t), H_t)   │
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

#### The 15 Staged Pipeline Execution Steps
| Stage | Name | Description |
| :---: | :--- | :--- |
| **1** | **Model Loading & Cache Resolution** | Loads or resolves the selected causal LM, tokenizer, embedding model, NLI cross-encoder, and trained classifiers from memory cache. |
| **2** | **Primary Response Generation** | Deterministic generation ($T=0$, greedy decoding) using model-specific prompt templates. |
| **3** | **Token & Logit Extraction** | Single forward pass computing per-token output logits, vocab softmax distribution, and loss tensor. |
| **4** | **Internal Signal Computation** | Calculates 11 white-box generation uncertainty metrics (log-probs, predictive entropy, perplexity, and token rank dispersion). |
| **5** | **Self-Consistency Sampling** | Generates $k=5$ stochastic responses ($T=0.7, \\text{top\\_}p=0.9, \\text{max\\_tokens}=128$) to probe the model's semantic stability. |
| **6** | **Embedding Model Verification** | Confirms `sentence-transformers/all-MiniLM-L6-v2` dense embedding model availability. |
| **7** | **Consistency Similarity Computation** | Encodes candidate responses into dense vectors and computes pairwise cosine similarity and exact match score. |
| **8** | **NLI Model Verification** | Confirms `cross-encoder/nli-MiniLM2-L6-H768` cross-encoder availability and output label mappings. |
| **9** | **Pairwise Bidirectional NLI Inference** | Executes $k(k-1) = 20$ directional hypothesis-premise inference passes across the 10 response pairs. |
| **10** | **Universal Feature Vector Assembly** | Compiles the canonical 19-dimensional feature vector, strictly verifying that sequence length confounders are excluded. |
| **11** | **XGBoost Prediction** | Computes the primary non-linear hallucination risk probability using the fitted gradient-boosted decision trees. |
| **12** | **Logistic Regression Prediction** | Computes the linear baseline risk probability using the standardized Logistic Regression pipeline. |
| **13** | **Explainability & Attribution** | Calculates TreeSHAP log-odds feature contributions to explain why the classifiers flagged or cleared the response. |
| **14** | **Evidence Groundedness Analysis** | *(Optional)* Chunks reference context and computes top-1 cosine similarity, evidence margin, and claim agreement. |
| **15** | **Final UI Result Assembly** | Formats per-token probability and entropy tooltips, determines categorical risk tier, and tabulates millisecond stage latencies. |

#### Strict Anti-Confounder Safeguards (Feature Isolation Policy)
- **Sequence Length Isolation**: `num_tokens` is strictly excluded from all training and inference feature sets. As proven in the Phase 7 ablation study, raw token count creates an artificial length shortcut where longer responses are trivially flagged, degrading scientific validity.
- **Metadata Exclusion**: Prompt text, dataset origins, raw IDs, and benchmark labels are completely withheld from the classifier.
- **Hardware Acceleration**: Models run in bfloat16/float16 with PyTorch CUDA tensor execution on GPU with automatic CPU fallback.
    """)
