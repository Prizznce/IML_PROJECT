
import streamlit as st

THEME_VARS = """
    --bg-canvas: #07080B;
    --bg-surface: #0E1017;
    --bg-surface-elevated: #151823;
    --bg-surface-hover: #1D2130;
    --bg-surface-active: #24293C;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-medium: rgba(255, 255, 255, 0.14);
    --border-strong: rgba(255, 255, 255, 0.22);
    --brand-primary: #3B82F6;
    --brand-primary-light: #60A5FA;
    --brand-primary-glow: rgba(59, 130, 246, 0.35);
    --brand-indigo: var(--brand-primary);
    --brand-indigo-light: var(--brand-primary-light);
    --brand-indigo-glow: var(--brand-primary-glow);
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
    --text-secondary: #CBD5E1;
    --text-muted: #8E8EA0;
    --text-dim: #64748B;
    --sidebar-bg: #0A0C12;
    --sidebar-border: rgba(255, 255, 255, 0.08);
    --title-gradient: linear-gradient(135deg, #FFFFFF 30%, #93C5FD 100%);
    --header-color: #FFFFFF;
    --card-bg: #11141E;
    --badge-zinc-bg: rgba(255, 255, 255, 0.04);
    --badge-zinc-color: var(--text-secondary);
    --select-bg: #121520;
    --select-text: var(--text-primary);
    --select-border: var(--border-medium);
    --terminal-bg: #090B10;
    --terminal-topbar: #10131B;
    --terminal-text: #ECECF1;
    --terminal-title: var(--text-secondary);
    --track-bg: rgba(255, 255, 255, 0.06);
    --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    --card-shadow-hover: 0 6px 20px rgba(0, 0, 0, 0.45);
    --badge-primary-text: #93C5FD;
    --badge-primary-bg: rgba(59, 130, 246, 0.12);
    --badge-primary-border: rgba(59, 130, 246, 0.30);
"""

THEME_VARS_DARK = THEME_VARS


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

/* --- Reconstructed Modern Sidebar Navigation System --- */
/* --- Reconstructed Modern Sidebar Navigation System --- */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child,
div[data-testid="stSidebarContent"] {
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 10px 14px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

div[data-testid="stSidebarUserContent"] {
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    width: 100% !important;
}

[data-testid="stSidebar"]::-webkit-scrollbar,
[data-testid="stSidebar"] *::-webkit-scrollbar,
div[data-testid="stSidebarContent"]::-webkit-scrollbar,
div[data-testid="stSidebarUserContent"]::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* Empty View Container for Unimplemented Backend Pages */
.empty-view-container {
    min-height: calc(100vh - 180px);
    width: 100%;
}

/* Sidebar Brand Header (Interactive Toggle - Logo stays at exact top-left 14px, 16px) */
.sidebar-brand-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 8px 0;
    margin: 0 0 6px 0;
    border-bottom: 1px solid var(--border-subtle);
    cursor: pointer;
    user-select: none;
    border-radius: 8px;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    z-index: 1;
}

.sidebar-brand-header:hover {
    background-color: var(--bg-surface-hover);
}

.sidebar-brand-header:hover .sidebar-brand-logo {
    transform: scale(1.08) rotate(-4deg);
    box-shadow: 0 4px 14px var(--brand-primary-glow);
}

/* Shared Project Logo Emblem Metrics (Optical Precision) */
.sidebar-brand-logo,
[data-testid="stExpandSidebarButton"]::before,
button[data-testid="stExpandSidebarButton"]::before,
[data-testid="stSidebarCollapsedControl"] button::before,
[data-testid="collapsedControl"] button::before {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    border-radius: 7px !important;
    background: linear-gradient(135deg, var(--brand-primary) 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-family: var(--font-sans), 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    text-align: center !important;
    box-shadow: 0 2px 8px var(--brand-primary-glow) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-sizing: border-box !important;
    padding: 0 !important;
    margin: 0 !important;
    letter-spacing: 0 !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ==============================================================================
   VERITAS Navigation Bar Toggle: Project Logo Replacement System
   - Initial (collapsed): ONLY the logo ◈ appears at top-left (14px, 16px), NO name.
   - Click to expand: Logo stays at the EXACT SAME POSITION (14px, 16px),
     and to the RIGHT of it, the name "VERITAS" & status appears.
   - Click to collapse: Sidebar closes, name disappears, logo stays in same place.
   ============================================================================== */

/* 1. Hide default Streamlit chevron/arrow icons and internal button contents */
[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] span.material-symbols-rounded,
[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] span.material-symbols-rounded,
[data-testid="stSidebarCollapsedControl"] span[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] span.material-symbols-rounded,
[data-testid="collapsedControl"] span[data-testid="stIconMaterial"],
[data-testid="collapsedControl"] svg,
[data-testid="stExpandSidebarButton"] > *,
button[data-testid="stExpandSidebarButton"] > *,
[data-testid="stSidebarCollapsedControl"] button > * {
    display: none !important;
    visibility: hidden !important;
    font-size: 0 !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    line-height: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}

/* 2. Top-Left Toggle (Collapsed State): Project Logo alone at exact position (14px, 16px) */
[data-testid="stExpandSidebarButton"],
button[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    position: fixed !important;
    top: 14px !important;
    left: 16px !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    border-radius: 7px !important;
    cursor: pointer !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box !important;
    outline: none !important;
    overflow: hidden !important;
    opacity: 1 !important;
    visibility: visible !important;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stExpandSidebarButton"]:hover,
button[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {
    transform: scale(1.08) !important;
}

/* Emblem character definition for collapsed button */
[data-testid="stExpandSidebarButton"]::before,
button[data-testid="stExpandSidebarButton"]::before,
[data-testid="stSidebarCollapsedControl"] button::before,
[data-testid="collapsedControl"] button::before {
    content: "◈" !important;
}

/* NO name with logo initially (when collapsed) */
[data-testid="stExpandSidebarButton"]::after,
button[data-testid="stExpandSidebarButton"]::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="collapsedControl"] button::after {
    display: none !important;
    content: "" !important;
    width: 0 !important;
    height: 0 !important;
}

/* 3. Navigation Toggle in Sidebar (Expanded State): Overlay collapse button over logo at (14px, 16px) */
[data-testid="stSidebarHeader"] {
    position: absolute !important;
    top: 14px !important;
    left: 16px !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    margin: 0 !important;
    z-index: 10 !important;
    background: transparent !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer !important;
    opacity: 0 !important;
}

/* 4. Streamlit Header Transparency & Layout Alignment */
header[data-testid="stHeader"],
.stAppHeader {
    background: transparent !important;
}

/* 5. Clean up any empty helper elements */
div[data-testid="stHtml"]:empty,
div:has(> div[data-testid="stHtml"]:empty) {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.sidebar-brand-text {
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.sidebar-brand-title {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    line-height: 1.2;
}

.sidebar-brand-subtitle {
    font-size: 10.5px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
    line-height: 1.2;
}

.sidebar-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    padding: 2.5px 8px;
    border-radius: 999px;
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
    margin-left: auto;
    flex-shrink: 0;
}

.sidebar-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--emerald-text);
    box-shadow: 0 0 6px var(--emerald-glow);
    animation: sidebarPulse 2s infinite ease-in-out;
    display: inline-block;
}

@keyframes sidebarPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* Sidebar Runtime Telemetry Card */
.sidebar-telemetry-card {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 8px 11px;
    margin-bottom: 12px;
    font-family: var(--font-mono);
}

.sidebar-telemetry-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11.5px;
}

.sidebar-telemetry-kicker {
    color: var(--text-muted);
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 0.06em;
}

.sidebar-telemetry-val {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 11px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
    max-width: 140px;
}

.sidebar-telemetry-device {
    font-size: 10.5px;
    font-weight: 600;
}

.sidebar-section-label {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0 4px;
    margin-bottom: 4px;
}

/* Navigation Action Buttons (Increased Size for Enhanced Readability & Touch) */
[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin-bottom: 3px !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    text-align: left !important;
    padding: 7px 12px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    border: 1px solid transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: none !important;
    min-height: 38px !important;
    height: 38px !important;
    line-height: 1.25 !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background-color: var(--bg-surface-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-subtle) !important;
    transform: translateX(2px) !important;
}

/* Active Navigation Button */
[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"],
[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, var(--badge-primary-bg) 0%, transparent 100%) !important;
    background-color: var(--badge-primary-bg) !important;
    color: var(--brand-primary-light) !important;
    font-weight: 600 !important;
    border-left: 3px solid var(--brand-primary) !important;
    border-top: 1px solid var(--badge-primary-border) !important;
    border-right: 1px solid transparent !important;
    border-bottom: 1px solid var(--badge-primary-border) !important;
    box-shadow: inset 1px 0 6px rgba(59, 130, 246, 0.15) !important;
}

/* Icon Uniformity */
[data-testid="stSidebar"] div[data-testid="stButton"] > button span[data-testid="stIconMaterial"],
[data-testid="stSidebar"] div[data-testid="stButton"] > button .material-symbols-rounded {
    font-size: 18px !important;
    width: 20px !important;
    height: 20px !important;
    line-height: 20px !important;
    margin-right: 9px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: inherit !important;
    flex-shrink: 0 !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button [data-testid="stMarkdownContainer"] {
    flex: 1 !important;
    text-align: left !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] > button [data-testid="stMarkdownContainer"] p {
    font-size: 13.5px !important;
    font-weight: 500 !important;
    margin: 0 !important;
    text-align: left !important;
    color: inherit !important;
}

/* Spacer & Profile at bottom */
.sidebar-nav-spacer {
    flex: 1 1 auto !important;
    min-height: 8px !important;
}



.sidebar-profile-section {
    margin-top: auto !important;
    padding-top: 8px !important;
    border-top: 1px solid var(--border-subtle) !important;
}

.sidebar-operator-card {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 6px 9px;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    margin-bottom: 5px;
}

.sidebar-operator-avatar {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-primary) 0%, #6366F1 100%);
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.sidebar-operator-info {
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-operator-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.25;
}

.sidebar-operator-role {
    font-size: 10px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    line-height: 1.25;
}

.sidebar-version-badge {
    text-align: center;
    font-size: 10.5px;
    font-family: var(--font-mono);
    color: var(--text-muted);
    padding: 6px 2px 2px 2px;
    letter-spacing: 0.03em;
}

/* --- About & Empirical Dossier Component Styles --- */
.about-version-chip {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
}

.about-badge-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
}

.about-hero-card {
    background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-surface-elevated) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    position: relative;
    overflow: hidden;
}

.about-hero-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--brand-primary) 0%, #8B5CF6 50%, var(--emerald-text) 100%);
}

.about-hero-tag {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--brand-primary-light);
    text-transform: uppercase;
    margin-bottom: 8px;
}

.about-hero-headline {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 10px 0;
    line-height: 1.3;
    letter-spacing: -0.02em;
}

.about-hero-lead {
    font-size: 13.5px;
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0 0 18px 0;
}

.about-metric-summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px;
}

.about-metric-summary-card {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    padding: 10px 13px;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.about-metric-summary-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-strong);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}

.about-metric-kicker {
    font-family: var(--font-mono);
    font-size: 9.5px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.06em;
}

.about-metric-value {
    font-size: 19px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2px 0;
}

.about-metric-unit {
    font-size: 11px;
    font-weight: 500;
    color: var(--brand-primary-light);
}

.about-metric-detail {
    font-size: 11px;
    color: var(--text-muted);
}

.about-tab-header {
    margin-bottom: 14px;
}

.about-tab-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 4px 0;
}

.about-tab-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.5;
}

.about-diagram-container {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 14px 0;
    overflow-x: auto;
}

.about-ascii-art {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    color: var(--text-primary) !important;
    line-height: 1.35 !important;
    margin: 0 !important;
}

.about-feature-category-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px 18px;
    height: 100%;
    display: flex;
    flex-direction: column;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.about-feature-category-card:hover {
    border-color: var(--border-strong);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
}

.about-category-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
    width: fit-content;
}

.about-category-badge.brand-blue {
    background: var(--badge-primary-bg);
    color: var(--brand-primary-light);
    border: 1px solid var(--badge-primary-border);
}

.about-category-badge.brand-emerald {
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
}

.about-category-badge.brand-amber {
    background: var(--amber-bg);
    color: var(--amber-text);
    border: 1px solid var(--amber-border);
}

.about-category-badge.brand-cyan {
    background: var(--cyan-bg);
    color: var(--cyan-text);
    border: 1px solid var(--cyan-border);
}

.about-category-badge.brand-purple {
    background: rgba(168, 85, 247, 0.12);
    color: #C084FC;
    border: 1px solid rgba(168, 85, 247, 0.3);
}

.about-category-subtitle {
    font-size: 11.5px;
    color: var(--text-muted);
    margin-bottom: 12px;
}

.about-feature-list, .about-stack-list {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.6;
}

.about-feature-list li, .about-stack-list li {
    position: relative;
    padding-left: 14px;
    margin-bottom: 7px;
}

.about-feature-list li::before, .about-stack-list li::before {
    content: "•";
    position: absolute;
    left: 2px;
    color: var(--brand-primary-light);
    font-size: 14px;
    line-height: 1.2;
}

.about-feature-list code, .about-stack-list code {
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 1.5px 5px;
    border-radius: 4px;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
}

.about-callout-box {
    background: var(--bg-surface-elevated);
    border-left: 3px solid var(--brand-primary);
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 12px 16px;
}

.about-callout-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
}

.about-citation-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px 20px;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.about-citation-box:hover {
    border-color: var(--border-strong);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
}

.about-bibtex {
    font-family: var(--font-mono) !important;
    font-size: 11.5px !important;
    color: var(--text-secondary) !important;
    background: var(--bg-surface-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 12px 14px !important;
    margin: 10px 0 0 0 !important;
    white-space: pre-wrap;
    line-height: 1.5 !important;
}

/* Empty View Container for Unimplemented Sections */
.empty-view-container {
    background: var(--bg-surface);
    border: 1px dashed var(--border-subtle);
    border-radius: 8px;
    min-height: 260px;
    margin-top: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
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

/* --- Linear / Railway Command Header Component (Compact Hierarchy) --- */
.linear-header {
    margin-bottom: 14px;
}

.linear-breadcrumbs {
    display: none !important;
}

.linear-title-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 4px;
}

.linear-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: var(--title-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.25;
}

.linear-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.45;
    margin: 4px 0 0 0;
    max-width: 720px;
}

/* --- Simplified Engine & Runtime Telemetry Panel --- */
.telemetry-panel {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    margin-bottom: 12px;
}

.telemetry-runtime-row,
.telemetry-config-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
}

.telemetry-runtime-chips,
.telemetry-config-chips {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
}

.telemetry-kicker {
    font-family: var(--font-mono);
    font-size: 9.5px;
    font-weight: 700;
    color: var(--emerald-text);
    letter-spacing: 0.06em;
    margin-right: 2px;
}

.telemetry-kicker-secondary {
    font-family: var(--font-mono);
    font-size: 9.5px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-right: 2px;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2.5px 8px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
}

.live-badge-status-pill {
    font-size: 9.5px;
    padding: 1.5px 6px;
    font-weight: 700;
    letter-spacing: 0.05em;
}

/* Secondary static configuration chips */
.config-chip {
    display: inline-flex;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 2px 7px;
}

.live-badge-primary {
    background: var(--badge-primary-bg);
    color: var(--badge-primary-text);
    border-color: var(--badge-primary-border);
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

/* Compact Engine Configure Button */
div:has(> button[data-testid*="btn_cfg_engine"]) button,
button[data-testid*="btn_cfg_engine"] {
    font-size: 11.5px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 3px 10px !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-medium) !important;
    transition: all 0.15s ease !important;
}

div:has(> button[data-testid*="btn_cfg_engine"]) button:hover,
button[data-testid*="btn_cfg_engine"]:hover {
    border-color: var(--brand-primary) !important;
    color: var(--brand-primary) !important;
    background: var(--bg-surface-elevated) !important;
}

/* --- Primary Query Area & Presets Focus Styles --- */
.preset-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}

.preset-section-label {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.preset-section-hint {
    font-size: 11px;
    color: var(--text-dim);
}

/* Smaller Preset Benchmark Query Buttons */
div[data-testid*="preset_"] button {
    min-height: 28px !important;
    height: 28px !important;
    padding: 2px 8px !important;
    font-size: 11.5px !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}

div[data-testid*="preset_"] button:hover {
    border-color: var(--brand-primary) !important;
    transform: translateY(-1px) !important;
}

/* Prompt input label row with inline badge and Reset */
.prompt-input-label-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    margin-bottom: 2px;
}

.prompt-input-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

.prompt-input-badge {
    font-family: var(--font-mono);
    font-size: 9.5px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    background: var(--badge-primary-bg);
    color: var(--brand-primary-light);
    letter-spacing: 0.04em;
}

/* Compact Reset Button */
div:has(> button[data-testid*="prompt_reset_btn"]) button,
button[data-testid*="prompt_reset_btn"] {
    min-height: 28px !important;
    height: 28px !important;
    padding: 2px 8px !important;
    font-size: 11.5px !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
    margin-top: 6px !important;
}

/* Subtle Focus Border and Glow on the Primary Prompt Textarea */
div[data-testid="stTextArea"] textarea {
    border-radius: 8px !important;
    border: 1px solid var(--border-medium) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 13.5px !important;
    line-height: 1.5 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    transition: border-color 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.2s ease !important;
}

div[data-testid="stTextArea"] textarea:hover {
    border-color: var(--border-strong) !important;
}

div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextArea"] textarea:focus-visible {
    border-color: var(--brand-primary) !important;
    box-shadow: 0 0 0 3px var(--brand-primary-glow), 0 2px 10px rgba(37, 99, 235, 0.12) !important;
    background: var(--bg-surface) !important;
    outline: none !important;
}

/* Concise example prompt below field */
.prompt-example-hint {
    display: flex;
    align-items: baseline;
    gap: 5px;
    margin-top: 5px;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-muted);
}

.prompt-example-prefix {
    font-weight: 600;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 11px;
}

.prompt-example-text {
    font-style: italic;
    color: var(--text-muted);
    font-size: 11.5px;
}

/* --- Consolidated Bottom Controls --- */
.execution-settings-bar {
    display: flex;
    align-items: center;
    margin-top: 8px;
    margin-bottom: 4px;
}

.settings-bar-title {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.primary-action-container {
    margin-top: 8px;
    margin-bottom: 4px;
}

/* Prominent Primary Run Verification Pipeline Action Button */
div:has(> button[data-testid*="btn_run_verification"]) button,
button[data-testid*="btn_run_verification"] {
    min-height: 44px !important;
    height: 44px !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 10px var(--brand-primary-glow) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div:has(> button[data-testid*="btn_run_verification"]) button:hover,
button[data-testid*="btn_run_verification"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px var(--brand-primary-glow) !important;
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
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
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
    background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
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
    background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%) !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
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
    background: var(--terminal-bg);
    border: 1px solid var(--border-medium);
    border-radius: 9px;
    overflow: hidden;
    margin-bottom: 22px;
    box-shadow: var(--card-shadow);
}

.terminal-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--terminal-topbar);
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
    font-size: 12px;
    font-weight: 600;
    color: var(--terminal-title);
    letter-spacing: 0.04em;
}

.terminal-body {
    padding: 16px 20px;
    font-size: 14.5px;
    line-height: 1.65;
    color: var(--terminal-text);
    font-family: var(--font-sans);
}

/* --- Linear / Railway Telemetry Cards for Streamlit Metrics --- */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 9px !important;
    padding: 16px 18px !important;
    box-shadow: var(--card-shadow) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-medium) !important;
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] label {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-muted) !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    font-family: var(--font-mono) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    font-variant-numeric: tabular-nums !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] > div {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
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
    box-shadow: var(--card-shadow);
}

.hud-card:hover {
    border-color: var(--border-medium);
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover);
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
    background: var(--badge-zinc-bg);
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
    background: var(--track-bg);
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
    background: var(--track-bg);
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
    background: var(--track-bg);
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

/* --- Expanders (Refined Disclosure View - Distinct from Form Controls) --- */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    box-shadow: var(--card-shadow) !important;
    overflow: hidden;
    margin-bottom: 14px;
}

[data-testid="stExpander"] summary {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    padding: 12px 14px !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--text-primary) !important;
    background: var(--bg-surface-hover) !important;
}

[data-testid="stExpander"] summary svg {
    color: var(--brand-primary) !important;
}

[data-testid="stExpander"] div[role="region"] {
    padding: 14px 8px !important;
    background: transparent !important;
    color: var(--text-primary) !important;
    border-top: 1px dashed var(--border-subtle) !important;
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
.preset-factual { background: var(--emerald-bg); color: var(--emerald-text); border: 1px solid var(--emerald-border); }
.preset-trap { background: var(--rose-bg); color: var(--rose-text); border: 1px solid var(--rose-border); }
.preset-evidence { background: var(--cyan-bg); color: var(--cyan-text); border: 1px solid var(--cyan-border); }

/* --- Infrastructure Strip on Main Canvas --- */
.infra-strip-card {
    background: var(--card-bg);
    border: 1px solid var(--border-medium);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    box-shadow: var(--card-shadow);
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
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
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
}

/* --- Readiness Dashboard & Defense Pillar Grid --- */
.ready-overview-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 20px 22px;
    box-shadow: var(--card-shadow);
    transition: all 0.2s ease;
}

.pillar-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}

@media (max-width: 900px) {
    .pillar-grid {
        grid-template-columns: 1fr;
    }
}

.pillar-card {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.pillar-card:hover {
    border-color: var(--brand-primary);
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover);
}

.pillar-mono {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.pillar-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}

.pillar-desc {
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 14px;
}

.pillar-tag {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 4px;
    display: inline-block;
    width: fit-content;
}

.tag-emerald {
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
}

.tag-cyan {
    background: var(--cyan-bg);
    color: var(--cyan-text);
    border: 1px solid var(--cyan-border);
}

.tag-amber {
    background: var(--amber-bg);
    color: var(--amber-text);
    border: 1px solid var(--amber-border);
}

.ready-footer-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    padding-top: 14px;
    border-top: 1px solid var(--border-subtle);
}

/* --- Popover & Dropdown Menu Theming --- */
div[data-testid="stPopover"] > button {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 12.5px !important;
    box-shadow: var(--card-shadow) !important;
}

div[data-testid="stPopover"] > button:hover {
    background: var(--bg-surface-elevated) !important;
    border-color: var(--border-strong) !important;
    color: var(--text-primary) !important;
}

[data-testid="stPopoverBody"], div[data-baseweb="popover"], div[data-baseweb="popover"] > div {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 8px !important;
    box-shadow: var(--card-shadow-hover) !important;
}

ul[data-baseweb="menu"] {
    background-color: var(--select-bg) !important;
    color: var(--select-text) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 8px !important;
    box-shadow: var(--card-shadow-hover) !important;
}

li[data-baseweb="menu-item"] {
    background-color: var(--select-bg) !important;
    color: var(--select-text) !important;
}

li[data-baseweb="menu-item"]:hover {
    background-color: var(--bg-surface-hover) !important;
    color: var(--text-primary) !important;
}

div[data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
div[data-testid="stSlider"] label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
}

div[data-testid="stModal"], div[role="dialog"] {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-medium) !important;
    box-shadow: var(--card-shadow-hover) !important;
}

code {
    background: var(--badge-zinc-bg);
    color: var(--brand-indigo-light);
    border-radius: 4px;
    padding: 2px 5px;
    font-family: var(--font-mono);
}

pre {
    background: var(--bg-surface-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

/* ==============================================================================
   VERITAS History Component & Observability Audit Dossier Styles
   ============================================================================== */

/* History Page Components */
.history-session-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
}

.history-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.history-kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: var(--card-shadow);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.history-kpi-card:hover {
    border-color: var(--border-medium);
    transform: translateY(-1px);
}

.history-kpi-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.history-kpi-value {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 4px;
}

.history-kpi-pct {
    font-size: 13px;
    font-weight: 600;
    opacity: 0.85;
}

.history-kpi-sub {
    font-size: 11px;
    color: var(--text-dim);
}

.history-record-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: var(--card-shadow);
    margin-bottom: 6px;
    transition: all 0.2s ease;
}

.history-record-card:hover {
    border-color: var(--border-medium);
    box-shadow: var(--card-shadow-hover);
}

.history-record-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}

.history-record-id {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    background: var(--bg-surface-elevated);
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid var(--border-subtle);
}

.history-tier-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.03em;
}

.badge-faithful {
    background: var(--emerald-bg);
    color: var(--emerald-text);
    border: 1px solid var(--emerald-border);
}

.badge-high-risk {
    background: var(--rose-bg);
    color: var(--rose-text);
    border: 1px solid var(--rose-border);
}

.badge-indeterminate {
    background: var(--amber-bg);
    color: var(--amber-text);
    border: 1px solid var(--amber-border);
}

.history-model-tag {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-surface-elevated);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid var(--border-subtle);
}

.history-record-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 11px;
    color: var(--text-dim);
    font-family: var(--font-mono);
}

.history-score-bar-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 10px;
    padding: 8px 12px;
    background: var(--bg-surface-elevated);
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
}

.history-score-item {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 180px;
}

.history-score-label {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}

.history-score-val {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-primary);
}

.history-mini-bar-track {
    flex: 1;
    height: 6px;
    background: var(--track-bg);
    border-radius: 3px;
    overflow: hidden;
}

.history-mini-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}

.risk-fill-faithful {
    background: var(--emerald-text);
}

.risk-fill-high-risk {
    background: var(--rose-text);
}

.risk-fill-indeterminate {
    background: var(--amber-text);
}

.conf-fill {
    background: var(--brand-primary);
}

.history-prompt-preview {
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.45;
}

.history-prompt-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    margin-bottom: 2px;
}

.history-prompt-content {
    color: var(--text-primary);
    font-weight: 500;
}

/* Empty State Container */
.history-empty-container {
    background: var(--bg-surface);
    border: 1px dashed var(--border-medium);
    border-radius: 12px;
    padding: 40px 24px;
    text-align: center;
    margin: 20px 0 24px 0;
}

.history-empty-icon {
    width: 48px;
    height: 48px;
    margin: 0 auto 14px auto;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--brand-primary) 0%, #1D4ED8 100%);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 700;
    box-shadow: 0 4px 16px var(--brand-primary-glow);
}

.history-empty-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.history-empty-desc {
    font-size: 13px;
    color: var(--text-secondary);
    max-width: 520px;
    margin: 0 auto 16px auto;
    line-height: 1.5;
}

.history-empty-badge-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
}

.history-empty-pill {
    font-size: 11.5px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 6px;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}

/* =============================================================================
   SYSTEM HEALTH & NEURAL PIPELINE TELEMETRY
   ============================================================================= */

.health-badge-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 4px 11px;
    border-radius: 20px;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
}

.health-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
    margin-bottom: 22px;
}

.health-kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.health-kpi-label {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 6px;
}

.health-kpi-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}

.health-kpi-sub {
    font-size: 11.5px;
    color: var(--text-secondary);
}

.health-section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-subtle);
}

.health-section-title {
    font-size: 14.5px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.01em;
}

.health-section-step {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    color: var(--brand-primary);
    background: var(--brand-primary-glow);
    padding: 2px 7px;
    border-radius: 4px;
}

.health-section-tag {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    text-transform: uppercase;
}

.health-detail-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    height: 100%;
    margin-bottom: 12px;
}

.health-detail-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}

.health-detail-title {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
}

.health-pill-active {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--emerald-text);
    background: var(--emerald-bg);
    border: 1px solid var(--emerald-border);
    padding: 2px 8px;
    border-radius: 4px;
}

.health-pill-neutral {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    padding: 2px 8px;
    border-radius: 4px;
}

.health-kv-list {
    display: flex;
    flex-direction: column;
    gap: 9px;
}

.health-kv-list.compact {
    gap: 7px;
}

.health-kv-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
}

.health-kv-key {
    color: var(--text-secondary);
    font-weight: 500;
}

.health-kv-val {
    font-family: var(--font-mono);
    color: var(--text-primary);
    font-weight: 600;
    text-align: right;
}

.health-model-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    display: flex;
    flex-direction: column;
}

.health-model-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.health-model-badge {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--brand-primary);
    text-transform: uppercase;
}

.health-model-status-online {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--emerald-text);
    background: var(--emerald-bg);
    border: 1px solid var(--emerald-border);
    padding: 2px 7px;
    border-radius: 4px;
}

.health-model-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 3px;
}

.health-model-repo {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 12px;
}

.health-model-divider {
    height: 1px;
    background: var(--border-subtle);
    margin-bottom: 12px;
}

.health-subtext {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 0 0 12px 0;
    line-height: 1.4;
}

.health-dist-bar-container {
    display: flex;
    height: 24px;
    border-radius: 6px;
    overflow: hidden;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    margin-bottom: 14px;
}

.health-dist-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    color: #FFFFFF;
    transition: width 0.3s ease;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}

.health-dist-seg.seg-emerald {
    background: #10B981;
}

.health-dist-seg.seg-amber {
    background: #F59E0B;
}

.health-dist-seg.seg-rose {
    background: #F43F5E;
}

.health-dist-legend {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.health-legend-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
}

.health-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

.health-legend-dot.dot-emerald {
    background: #10B981;
}

.health-legend-dot.dot-amber {
    background: #F59E0B;
}

.health-legend-dot.dot-rose {
    background: #F43F5E;
}

.health-legend-label {
    color: var(--text-secondary);
    flex-grow: 1;
}

.health-legend-val {
    font-family: var(--font-mono);
    color: var(--text-primary);
}

.health-diagnostic-banner {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--emerald-text);
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 14px;
}

.health-diag-title {
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
    color: var(--emerald-text);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    letter-spacing: 0.04em;
}

.health-diag-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
}

.health-diag-card {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 10px 12px;
}

.health-diag-status {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    margin-bottom: 2px;
}

.health-diag-name {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 2px;
}

.health-diag-sub {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.3;
}
</style>
"""

def apply_custom_styles(*args, **kwargs):
    """
    Inject dark mode theme variables and unified Linear/Railway stylesheet.
    """
    st.markdown(f"<style>:root {{\n{THEME_VARS}\n}}</style>\n" + CUSTOM_CSS, unsafe_allow_html=True)
