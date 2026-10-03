"""
Linear / Railway Design Tokens, Custom CSS & Theme Definitions.
"""
import streamlit as st

THEME_VARS_LIGHT = """
    --bg-canvas: #F8FAFC;
    --bg-surface: #FFFFFF;
    --bg-surface-elevated: #F1F5F9;
    --bg-surface-hover: #E2E8F0;
    --bg-surface-active: #CBD5E1;
    --border-subtle: rgba(15, 23, 42, 0.08);
    --border-medium: rgba(15, 23, 42, 0.16);
    --border-strong: rgba(15, 23, 42, 0.28);
    --brand-indigo: #4F46E5;
    --brand-indigo-light: #6366F1;
    --brand-indigo-glow: rgba(79, 70, 229, 0.2);
    --emerald-text: #059669;
    --emerald-bg: rgba(16, 185, 129, 0.12);
    --emerald-border: rgba(16, 185, 129, 0.35);
    --emerald-glow: rgba(16, 185, 129, 0.2);
    --amber-text: #D97706;
    --amber-bg: rgba(245, 158, 11, 0.12);
    --amber-border: rgba(245, 158, 11, 0.35);
    --rose-text: #E11D48;
    --rose-bg: rgba(244, 63, 94, 0.12);
    --rose-border: rgba(244, 63, 94, 0.35);
    --cyan-text: #0284C7;
    --cyan-bg: rgba(56, 189, 248, 0.12);
    --cyan-border: rgba(56, 189, 248, 0.35);
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --text-dim: #94A3B8;
    --sidebar-bg: #FFFFFF;
    --sidebar-border: rgba(15, 23, 42, 0.1);
    --title-gradient: linear-gradient(135deg, #0F172A 20%, #4338CA 100%);
    --header-color: #0F172A;
    --card-bg: #FFFFFF;
    --badge-zinc-bg: rgba(15, 23, 42, 0.06);
    --badge-zinc-color: #334155;
    --select-bg: #FFFFFF;
    --select-text: #0F172A;
    --select-border: rgba(15, 23, 42, 0.18);
"""

THEME_VARS_DARK = """
    --bg-canvas: #07080B;
    --bg-surface: #0E1017;
    --bg-surface-elevated: #151823;
    --bg-surface-hover: #1D2130;
    --bg-surface-active: #24293C;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-medium: rgba(255, 255, 255, 0.14);
    --border-strong: rgba(255, 255, 255, 0.22);
    --brand-indigo: #5E6AD2;
    --brand-indigo-light: #828EE8;
    --brand-indigo-glow: rgba(94, 106, 210, 0.35);
    --emerald-text: #34D399;
    --emerald-bg: rgba(16, 185, 129, 0.09);
    --emerald-border: rgba(16, 185, 129, 0.26);
    --emerald-glow: rgba(16, 185, 129, 0.25);
    --amber-text: #FBBF24;
    --amber-bg: rgba(245, 158, 11, 0.09);
    --amber-border: rgba(245, 158, 11, 0.26);
    --rose-text: #FB7185;
    --rose-bg: rgba(244, 63, 94, 0.10);
    --rose-border: rgba(244, 63, 94, 0.30);
    --cyan-text: #38BDF8;
    --cyan-bg: rgba(56, 189, 248, 0.08);
    --cyan-border: rgba(56, 189, 248, 0.24);
    --text-primary: #F4F4F6;
    --text-secondary: #A1A1AA;
    --text-muted: #71717A;
    --text-dim: #52525B;
    --sidebar-bg: #0A0C12;
    --sidebar-border: rgba(255, 255, 255, 0.08);
    --title-gradient: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 100%);
    --header-color: #FFFFFF;
    --card-bg: #11141E;
    --badge-zinc-bg: rgba(255, 255, 255, 0.04);
    --badge-zinc-color: var(--text-secondary);
    --select-bg: #121520;
    --select-text: var(--text-primary);
    --select-border: var(--border-medium);
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* --- Root Theme Tokens --- */
:root {
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

header[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stWidgetLabel"] p, label, .stSlider label {
    color: var(--text-primary) !important;
}

/* Container Spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1320px !important;
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
    background: rgba(128, 128, 128, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(128, 128, 128, 0.35);
}

/* --- Sidebar Styling (Fixed Viewport, No Scrolling) --- */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: hidden !important;
    overflow-x: hidden !important;
}

[data-testid="stSidebar"] > div:first-child {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: hidden !important;
    overflow-x: hidden !important;
    padding-top: 1rem !important;
    padding-bottom: 0.8rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 0.65rem 0 !important;
}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--select-bg) !important;
    border: 1px solid var(--select-border) !important;
    border-radius: 6px !important;
    color: var(--select-text) !important;
    font-size: 12.5px !important;
}

[data-testid="stSidebar"] .stSlider {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-bottom: -6px !important;
}

/* --- Dialog / Modal Overhauls --- */
div[role="dialog"] {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-medium) !important;
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
    font-size: 11.5px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
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
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.038em;
    background: var(--title-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.linear-subtitle {
    font-size: 14.5px;
    color: var(--text-secondary);
    line-height: 1.55;
    margin: 0 0 16px 0;
    max-width: 900px;
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
    font-size: 11px;
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
    background: var(--badge-zinc-bg);
    color: var(--badge-zinc-color);
    border-color: var(--border-subtle);
}

.live-badge-amber {
    background: var(--amber-bg);
    color: var(--amber-text);
    border-color: var(--amber-border);
}

.live-badge-rose {
    background: var(--rose-bg);
    color: var(--rose-text);
    border-color: var(--rose-border);
}

/* --- Metric Card Anti-Truncation Overhauls --- */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    min-height: 104px !important;
    transition: all 0.15s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-medium) !important;
    background: var(--bg-surface-elevated) !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-sans) !important;
    font-size: 21px !important;
    font-weight: 700 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: normal !important;
    line-height: 1.25 !important;
    color: var(--text-primary) !important;
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
    background: var(--bg-surface);
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
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--header-color);
    margin-bottom: 12px;
}

/* --- Inputs & Interactive Fields --- */
.stTextArea textarea {
    background-color: var(--select-bg) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    transition: all 0.18s ease !important;
    padding: 12px 14px !important;
}

.stTextArea textarea:focus {
    border-color: var(--brand-indigo) !important;
    box-shadow: 0 0 0 1px var(--brand-indigo), 0 0 16px -2px var(--brand-indigo-glow) !important;
    background-color: var(--bg-surface-elevated) !important;
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
    color: var(--text-primary) !important;
    background: var(--bg-surface-hover) !important;
}

/* --- Buttons --- */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #5E6AD2 0%, #4854B8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    padding: 9px 22px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    transition: all 0.16s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(180deg, #6B77DE 0%, #525FC5 100%) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
    transform: translateY(-1px);
}

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
    color: var(--text-primary) !important;
    border-color: var(--border-medium) !important;
    transform: translateY(-1px);
}

/* --- Railway Terminal Container for Output --- */
.terminal-window {
    background: #090B10;
    border: 1px solid var(--border-subtle);
    border-radius: 9px;
    overflow: hidden;
    margin-bottom: 22px;
    box-shadow: 0 6px 24px -4px rgba(0, 0, 0, 0.5);
}

.terminal-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #10131B;
    padding: 9px 15px;
    border-bottom: 1px solid var(--border-subtle);
}

.terminal-dots {
    display: flex;
    gap: 6px;
}

.terminal-dot {
    width: 9px;
    height: 9px;
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
    padding: 16px 20px;
    font-size: 14.5px;
    line-height: 1.65;
    color: #ECECF1;
    font-family: var(--font-sans);
}

/* --- Linear / Railway Telemetry Cards for Streamlit Metrics --- */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 9px !important;
    padding: 16px 18px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-medium) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45) !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    font-variant-numeric: tabular-nums !important;
}

[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* --- Telemetry Cards Grid (Modern High-Tech HUD) --- */
.telemetry-hud-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 22px;
}

@media (max-width: 1024px) {
    .telemetry-hud-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 640px) {
    .telemetry-hud-grid {
        grid-template-columns: 1fr;
    }
}

.hud-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}

.hud-card:hover {
    border-color: var(--border-medium);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}

.hud-card-accent {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
}

.hud-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.hud-label {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}

.hud-tag {
    font-family: var(--font-mono);
    font-size: 10.5px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-secondary);
}

.hud-value-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 8px;
}

.hud-value {
    font-family: var(--font-mono);
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
}

.hud-delta {
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 600;
}

.hud-desc {
    font-size: 11.5px;
    color: var(--text-muted);
    line-height: 1.45;
}

/* Custom Telemetry Gauge Bar */
.gauge-track {
    width: 100%;
    height: 5px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 3px;
    margin-top: 10px;
    overflow: hidden;
    position: relative;
}

.gauge-fill-emerald {
    height: 100%;
    background: linear-gradient(90deg, #059669, #34D399);
    border-radius: 3px;
}

.gauge-fill-amber {
    height: 100%;
    background: linear-gradient(90deg, #D97706, #FBBF24);
    border-radius: 3px;
}

.gauge-fill-rose {
    height: 100%;
    background: linear-gradient(90deg, #E11D48, #FB7185);
    border-radius: 3px;
}

/* Status Pill in Card 3 */
.tier-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-top: 4px;
}

.tier-emerald {
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
    box-shadow: 0 0 12px var(--emerald-glow);
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
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.25);
}

/* --- Token Heatmap Inspector --- */
.heatmap-container {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 9px;
    padding: 16px 20px;
    margin-bottom: 12px;
    line-height: 2.3;
}

.heatmap-token-chip {
    padding: 3px 6px;
    margin: 2px 2px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 500;
    display: inline-block;
    white-space: pre-wrap;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
    cursor: default;
}

.heatmap-token-chip:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}

/* --- Feature Attribution Impact Rows (Linear Style) --- */
.attr-row {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    padding: 11px 15px;
    margin-bottom: 8px;
    transition: all 0.15s ease;
}

.attr-row:hover {
    border-color: var(--border-medium);
    background: var(--bg-surface-elevated);
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
    font-size: 11px;
    color: var(--text-muted);
    margin-left: 6px;
}

.attr-badge {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    padding: 2px 8px;
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

/* --- Waterfall Execution Latency Profiler --- */
.waterfall-track {
    display: flex;
    width: 100%;
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.05);
    margin: 12px 0 18px 0;
}

.waterfall-segment {
    height: 100%;
    transition: width 0.3s ease;
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
    color: var(--brand-indigo) !important;
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
    color: var(--text-primary) !important;
    background: var(--bg-surface-elevated) !important;
}

[data-testid="stExpander"] div[role="region"] {
    padding: 14px 16px !important;
    background: var(--bg-surface-elevated) !important;
    color: var(--text-primary) !important;
    border-top: 1px solid var(--border-subtle) !important;
}

/* --- DataFrames --- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 7px !important;
    overflow: hidden;
}

/* --- Sidebar Spec Box --- */
.spec-box {
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px;
    margin-top: 10px;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-subtle);
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

/* --- Preset Card Chips --- */
.preset-badge {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 2px 5px;
    border-radius: 3px;
    letter-spacing: 0.04em;
    font-weight: 600;
}
.preset-factual { background: rgba(16, 185, 129, 0.15); color: #34D399; }
.preset-trap { background: rgba(244, 63, 94, 0.15); color: #FB7185; }
.preset-evidence { background: rgba(56, 189, 248, 0.15); color: #38BDF8; }

/* --- Infrastructure Strip on Main Canvas --- */
.infra-strip-card {
    background: var(--card-bg);
    border: 1px solid var(--border-medium);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

/* --- Sidebar Identity & User Card --- */
.sidebar-user-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.sidebar-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.sidebar-user-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
}
.sidebar-user-org {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.2;
}

/* --- History Cards --- */
.history-entry-box {
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 12px;
}
.history-entry-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
    font-size: 10.5px;
    color: var(--text-muted);
    font-family: var(--font-mono);
}
.history-prompt-snippet {
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 11.5px;
    margin-bottom: 6px;
}
</style>
"""

def apply_custom_styles(theme_choice: str = "Dark Mode"):
    """
    Inject dynamic CSS variables and unified Linear/Railway stylesheet.
    """
    theme_vars = THEME_VARS_LIGHT if theme_choice == "Light Mode" else THEME_VARS_DARK
    st.markdown(f"<style>:root {{\n{theme_vars}\n}}</style>\n" + CUSTOM_CSS, unsafe_allow_html=True)
