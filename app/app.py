"""
Streamlit Live Demonstration Application for LLM Hallucination Detection.

Title: Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination
Detection from Internal Generation Signals

This presentation layer interfaces with app/inference.py to execute the complete
19-feature Universal Core Detector on user-supplied queries or benchmark presets.
Provides:
  - Token-by-token generation uncertainty heatmap with interactive tooltips.
  - Model-native explainability ("Why was this response flagged?").
  - Detection mode selection (Universal Core vs. Evidence Groundedness).
  - Detailed signal breakdowns and research benchmark comparisons.
"""

import html
import sys
from pathlib import Path

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


@st.cache_resource(show_spinner="Loading detection models into memory (one-time initialization)...")
def get_cached_models(model_id: str = "Qwen/Qwen3.5-0.8B"):
    """
    Load and persist models in Streamlit's resource cache across sessions and reruns.
    Supports Qwen3.5-0.8B and Llama 3.2 1B Instruct.
    """
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return load_inference_models(
        device=target_device,
        model_id=model_id,
    )

# ==============================================================================
# 1. Page Configuration and Header
# ==============================================================================

st.set_page_config(
    page_title="Catching an LLM Lying | Real-Time Hallucination Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar model selection
with st.sidebar:
    st.markdown("### ⚙️ Generative Model")
    model_options = {
        "Qwen 3.5 (0.8B)": "Qwen/Qwen3.5-0.8B",
        "Llama 3.2 (1B Instruct)": "meta-llama/Llama-3.2-1B-Instruct",
        "Gemma 3 (1B IT)": "google/gemma-3-1b-it",
    }
    selected_model_name = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0,
        help="Select which causal LLM produces responses and generation signals.",
    )
    active_model_id = model_options[selected_model_name]

    st.markdown("---")
    st.markdown("### 🧠 Active Model Status")
    if "Llama" in selected_model_name:
        st.markdown(
            """
            - **Model**: `Llama 3.2 (1B Instruct)`
            - **Parameters**: 1.23 Billion
            - **Detectors**: Retrained on 4,000 samples
            - **ROC-AUC**: 0.7450 (XGB) / 0.6929 (LR)
            - **PR-AUC**: 0.7492 (XGB) / 0.6681 (LR)
            - **Accuracy**: 66.88% (XGB) / 65.12% (LR)
            - **Status**: 🟢 Ready (GPU/CPU)
            """
        )
    elif "Gemma" in selected_model_name:
        st.markdown(
            """
            - **Model**: `Gemma 3 (1B IT)`
            - **Parameters**: 1.0 Billion
            - **Detectors**: Retrained on 4,000 samples
            - **ROC-AUC**: 0.7913 (XGB) / 0.7092 (LR)
            - **PR-AUC**: 0.7914 (XGB) / 0.7054 (LR)
            - **Accuracy**: 71.50% (XGB) / 64.75% (LR)
            - **Status**: 🟢 Ready (GPU/CPU)
            """
        )
    else:
        st.markdown(
            """
            - **Model**: `Qwen 3.5 (0.8B)`
            - **Parameters**: 0.8 Billion
            - **Detectors**: Retrained on 4,000 samples
            - **ROC-AUC**: 0.7688 (XGB) / 0.7079 (LR)
            - **PR-AUC**: 0.7795 (XGB) / 0.7124 (LR)
            - **Accuracy**: 67.88% (XGB) / 62.88% (LR)
            - **Status**: 🟢 Ready (GPU/CPU)
            """
        )

st.title("Catching an LLM Lying")
st.subheader("Lightweight Hallucination Detection from Internal Generation Signals")

# Badges
dev_name = f"🚀 Compute: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "🖥️ Compute: CPU"
badge_html = f"""
<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
  <span style="background: rgba(59, 130, 246, 0.15); color: #3b82f6; padding: 3px 10px; border-radius: 12px; font-size: 13px; font-weight: 500; border: 1px solid rgba(59, 130, 246, 0.3);">
    🟢 Model: {selected_model_name}
  </span>
  <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 3px 10px; border-radius: 12px; font-size: 13px; font-weight: 500; border: 1px solid rgba(16, 185, 129, 0.3);">
    🔬 Universal Core: 19 Features
  </span>
  <span style="background: rgba(139, 92, 246, 0.15); color: #8b5cf6; padding: 3px 10px; border-radius: 12px; font-size: 13px; font-weight: 500; border: 1px solid rgba(139, 92, 246, 0.3);">
    ⚡ Classifiers: XGBoost + Logistic Regression
  </span>
  <span style="background: rgba(14, 165, 233, 0.15); color: #0284c7; padding: 3px 10px; border-radius: 12px; font-size: 13px; font-weight: 500; border: 1px solid rgba(14, 165, 233, 0.3);">
    {dev_name}
  </span>
  <span style="background: rgba(245, 158, 11, 0.15); color: #f59e0b; padding: 3px 10px; border-radius: 12px; font-size: 13px; font-weight: 500; border: 1px solid rgba(245, 158, 11, 0.3);">
    🏛️ Multi-Model Academic Research Demo
  </span>
</div>
"""
st.markdown(badge_html, unsafe_allow_html=True)

st.info(
    "**Research Demonstration**: Probability values represent model-estimated hallucination risk, "
    "not a guarantee of factual correctness. All benchmark evaluations reported in the academic "
    "manuscript were conducted offline across 4,000 curated benchmark instances."
)

# ==============================================================================
# 2. Session State and Preset Management
# ==============================================================================

if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""
if "user_context" not in st.session_state:
    st.session_state["user_context"] = ""
if "inference_result" not in st.session_state:
    st.session_state["inference_result"] = None

st.markdown("### 1. Input Query & Detection Mode")

# Detection Mode Radio
mode_selection = st.radio(
    "**Detection Mode**",
    options=[
        "Internal Signals (Universal Core — 19 Features)",
        "Check Against Evidence (Phase 19 Retrieval Variant)",
    ],
    index=0,
    horizontal=True,
    help="Universal Core evaluates internal generation, self-consistency, and NLI. Evidence mode additionally inspects reference passage groundedness.",
)

is_evidence_mode = "Check Against Evidence" in mode_selection

if is_evidence_mode:
    st.caption(
        "ℹ️ **Evidence Mode Active**: When an optional reference context is provided below, the system "
        "calculates live evidence groundedness (query-evidence similarity, response-evidence similarity, "
        "and retrieval agreement) using `all-MiniLM-L6-v2`. *(Note: The offline Phase 19 retrieval experiment "
        "achieved ROC-AUC 0.9184 on the 2,500-instance HaluEval+FEVER evaluated subset; open-web search is not run locally.)*"
    )

# Preset selection buttons
st.markdown("**Sample Presets** *(Click to populate the prompt field)*:")
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    if st.button("Preset 1: Moon Landing\n\n*(✓ Factual question)*", use_container_width=True):
        st.session_state["user_prompt"] = "Who was the first person to walk on the Moon?"
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

with p_col2:
    if st.button("Preset 2: Capital of France\n\n*(✓ Simple factual question)*", use_container_width=True):
        st.session_state["user_prompt"] = "What is the capital of France?"
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

with p_col3:
    if st.button("Preset 3: Mars Walking\n\n*(⚠ False-premise stress test)*", use_container_width=True):
        st.session_state["user_prompt"] = "Who was the first human to walk on Mars?"
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

with p_col4:
    if st.button("Clear Input\n\n*(Reset prompt & results)*", use_container_width=True):
        st.session_state["user_prompt"] = ""
        st.session_state["user_context"] = ""
        st.session_state["inference_result"] = None

# Input fields
prompt_text = st.text_area(
    "Question / Claim Prompt",
    value=st.session_state["user_prompt"],
    placeholder="e.g., Who was the first person to walk on the Moon?",
    height=100,
    help="Enter the question or proposition to be evaluated by Qwen3.5-0.8B.",
)

context_text = st.text_area(
    "Reference Context (Optional)",
    value=st.session_state["user_context"],
    placeholder="Optional supporting evidence, background passage, or reference document...",
    height=80,
    help="Optional background passage for document-grounded question answering and evidence checking.",
)

btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    analyze_clicked = st.button("Generate & Analyze", type="primary", use_container_width=True)

# ==============================================================================
# 3. Live Inference Execution
# ==============================================================================

if analyze_clicked:
    clean_p = prompt_text.strip()
    if not clean_p:
        st.warning("Please enter a question or prompt before running analysis.")
    else:
        st.session_state["user_prompt"] = clean_p
        st.session_state["user_context"] = context_text.strip()

        with st.spinner("Generating response and analyzing hallucination signals..."):
            try:
                cached_models = get_cached_models(
                    model_id=active_model_id,
                )
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
                st.error(f"Inference pipeline error: {str(exc)}")
                st.session_state["inference_result"] = None

# ==============================================================================
# 4. Results Display
# ==============================================================================

res = st.session_state.get("inference_result")

if res is not None:
    st.markdown("---")
    st.markdown("### 2. Primary Generation & Hallucination Risk Assessment")

    # Primary generated response
    st.markdown(f"**Generated Response ({selected_model_name}):**")
    st.info(f"\"{res['generated_response']}\"")

    # Risk Metrics Card Layout
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
    # FEATURE 1: Token-by-Token Heatmap
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 3. Token-by-Token Generation Confidence Heatmap")
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
                txt_col = "#27ae60"
            elif t_band == "Low confidence":
                bg = "rgba(231, 76, 60, 0.25)"
                border_col = "#e74c3c"
                txt_col = "#c0392b"
            else:
                bg = "rgba(241, 196, 15, 0.22)"
                border_col = "#f1c40f"
                txt_col = "#d4ac0d"

            tooltip = f"Step {t_step}: '{t_token}' | Prob: {t_prob:.4f} | Entropy: {t_ent:.3f} | Rank: {t_rank} | {t_band}"
            span_html = (
                f'<span title="{html.escape(tooltip)}" style="background: {bg}; border-bottom: 2px solid {border_col}; '
                f'color: {txt_col}; padding: 2px 4px; margin: 1px 0px; border-radius: 3px; font-family: monospace; '
                f'font-size: 15px; font-weight: 500; display: inline-block; white-space: pre-wrap;">{t_token}</span>'
            )
            heatmap_spans.append(span_html)

        rendered_heatmap = "".join(heatmap_spans)
        box_html = f"""
        <div style="padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.25);
                    background: rgba(128, 128, 128, 0.05); line-height: 2.2; margin-bottom: 10px;">
            {rendered_heatmap}
        </div>
        """
        st.markdown(box_html, unsafe_allow_html=True)

        # Heatmap legend
        legend_html = """
        <div style="display: flex; gap: 18px; font-size: 13px; margin-bottom: 10px; flex-wrap: wrap;">
            <span><span style="color: #27ae60; font-weight: bold;">🟩 High confidence</span> (P ≥ 65%, low entropy)</span>
            <span><span style="color: #d4ac0d; font-weight: bold;">🟨 Medium confidence</span> (30% ≤ P &lt; 65%)</span>
            <span><span style="color: #c0392b; font-weight: bold;">🟥 Low confidence</span> (P &lt; 30% or elevated entropy)</span>
        </div>
        """
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
    # FEATURE 2: Why Was It Flagged? (Explainability)
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 4. Why Was This Response Flagged? (Model Feature Contributions)")
    st.write(
        "Model-level Tree SHAP feature contributions indicating which generation uncertainty "
        "signals drove the classifier toward or away from flagging hallucination risk."
    )

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
            pct_bar = min(100.0, max(5.0, (mag / max_mag) * 100.0))

            if direction == "increases risk":
                badge_bg = "rgba(231, 76, 60, 0.18)"
                badge_color = "#e74c3c"
                bar_color = "#e74c3c"
                dir_label = "▲ INCREASES RISK"
            else:
                badge_bg = "rgba(46, 204, 113, 0.18)"
                badge_color = "#2ecc71"
                bar_color = "#2ecc71"
                dir_label = "▼ DECREASES RISK"

            card_html = f"""
            <div style="background: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.2);
                        border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: 600; font-size: 14px;">{disp_name} <code style="font-size: 12px; color: #888;">({feat_name})</code></span>
                    <span style="font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: {badge_bg}; color: {badge_color};">
                        {dir_label}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
                    <span>Measured Value: <strong>{val:.4f}</strong></span>
                    <span>Contribution: <strong>{contrib:+.4f}</strong> log-odds</span>
                </div>
                <div style="background: rgba(128, 128, 128, 0.2); height: 6px; border-radius: 3px; overflow: hidden;">
                    <div style="background: {bar_color}; width: {pct_bar:.1f}%; height: 100%; border-radius: 3px;"></div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        st.caption(
            "*Disclaimer: These values are model-level feature contributions (Tree SHAP log-odds values from the primary "
            "XGBoost classifier) for this specific prediction, not causal explanations.*"
        )

    # ==========================================================================
    # FEATURE 3: Evidence Groundedness Section
    # ==========================================================================
    evidence_data = res.get("evidence_analysis", {})
    if is_evidence_mode or (evidence_data and evidence_data.get("has_reference_context")):
        st.markdown("---")
        st.markdown("### 5. Evidence Groundedness Analysis")

        if evidence_data.get("has_reference_context"):
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                st.metric(
                    label="Query-Evidence Similarity",
                    value=f"{evidence_data.get('top1_evidence_similarity', 0.0):.4f}",
                    help="Cosine similarity between user query and rank-1 reference context chunk.",
                )
            with e_col2:
                st.metric(
                    label="Response-Evidence Similarity",
                    value=f"{evidence_data.get('response_top1_similarity', 0.0):.4f}",
                    help="Cosine similarity between generated response and rank-1 reference context chunk.",
                )
            with e_col3:
                st.metric(
                    label="Retrieval Agreement Score",
                    value=f"{evidence_data.get('retrieval_agreement', 0.0):.4f}",
                    help="Dual groundedness index: max(0, q_sim) * max(0, r_sim).",
                )

            ev_cards = evidence_data.get("retrieved_evidence", [])
            if ev_cards:
                st.markdown("**Ranked Reference Context Passages:**")
                for card in ev_cards:
                    st.markdown(
                        f"- **Rank {card['rank']}** *(Query Sim: {card['query_similarity']:.4f} | Response Sim: {card['response_similarity']:.4f})*:\n"
                        f"  > {card['text']}"
                    )
        else:
            st.info(
                "**No Reference Context Provided**: To evaluate live groundedness against reference text, "
                "enter a reference passage in the input text area above. "
                "In the offline research evaluation (Phase 19), retrieval-augmented classification achieved "
                "**ROC-AUC 0.9184** across 2,500 curated HaluEval and FEVER instances."
            )

    # ==========================================================================
    # Detailed Signals Breakdown
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 6. Detailed Signal Breakdown")

    tab_internal, tab_sc, tab_nli, tab_vec = st.tabs([
        "Internal Signals (11)",
        "Self-Consistency (5)",
        "NLI Agreement (3)",
        "19-Feature Vector",
    ])

    # --------------------------------------------------------------------------
    # Tab 1: Internal Signals
    # --------------------------------------------------------------------------
    with tab_internal:
        st.markdown("#### White-Box Internal Generation Uncertainty Signals")
        st.write(
            "Extracted directly from raw model logits during an unadulterated forward pass over "
            "the generated sequence. Tokens produced with high uncertainty exhibit elevated predictive entropy "
            "and depressed log-likelihoods."
        )

        int_sig = res["internal_signals"]
        internal_table_data = [
            {"Signal": "min_log_prob", "Value": f"{int_sig['min_log_prob']:.4f}", "Description": "Minimum token log-probability (bottleneck token)"},
            {"Signal": "mean_log_prob", "Value": f"{int_sig['mean_log_prob']:.4f}", "Description": "Mean token log-probability across sequence"},
            {"Signal": "mean_token_prob", "Value": f"{int_sig['mean_token_prob']:.4f}", "Description": "Mean token probability"},
            {"Signal": "token_prob_std", "Value": f"{int_sig['token_prob_std']:.4f}", "Description": "Standard deviation of token probabilities"},
            {"Signal": "mean_entropy", "Value": f"{int_sig['mean_entropy']:.4f}", "Description": "Mean predictive Shannon entropy (nats)"},
            {"Signal": "max_entropy", "Value": f"{int_sig['max_entropy']:.4f}", "Description": "Peak predictive entropy in sequence"},
            {"Signal": "entropy_std", "Value": f"{int_sig['entropy_std']:.4f}", "Description": "Standard deviation of predictive entropy"},
            {"Signal": "log_perplexity", "Value": f"{int_sig['log_perplexity']:.4f}", "Description": "Log-transformed sequence perplexity"},
            {"Signal": "log_mean_token_rank", "Value": f"{int_sig['log_mean_token_rank']:.4f}", "Description": "log(1 + mean token rank)"},
            {"Signal": "log_max_token_rank", "Value": f"{int_sig['log_max_token_rank']:.4f}", "Description": "log(1 + max token rank)"},
            {"Signal": "log_rank_std", "Value": f"{int_sig['log_rank_std']:.4f}", "Description": "log(1 + token rank std)"},
        ]
        st.dataframe(pd.DataFrame(internal_table_data), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # Tab 2: Self-Consistency
    # --------------------------------------------------------------------------
    with tab_sc:
        st.markdown("#### Stochastic Behavioral Self-Consistency Probing")
        st.write(
            "Five stochastic responses ($k=5, T=0.7, \\text{top\\_}p=0.9$) are compared to measure behavioral "
            "consistency. Disagreement among candidate outputs indicates epistemic uncertainty."
        )

        sc_sig = res.get("self_consistency_features", {})
        sc_table_data = [
            {"Signal": "exact_match_agreement", "Value": f"{sc_sig.get('exact_match_agreement', 0.0):.4f}", "Description": "Fraction of candidate pairs that match identically"},
            {"Signal": "mean_pairwise_similarity", "Value": f"{sc_sig.get('mean_pairwise_similarity', 0.0):.4f}", "Description": "Mean embedding cosine similarity among candidate outputs"},
            {"Signal": "min_pairwise_similarity", "Value": f"{sc_sig.get('min_pairwise_similarity', 0.0):.4f}", "Description": "Minimum pairwise cosine similarity"},
            {"Signal": "max_pairwise_similarity", "Value": f"{sc_sig.get('max_pairwise_similarity', 0.0):.4f}", "Description": "Maximum pairwise cosine similarity"},
            {"Signal": "pairwise_similarity_std", "Value": f"{sc_sig.get('pairwise_similarity_std', 0.0):.4f}", "Description": "Standard deviation of pairwise cosine similarities"},
        ]
        st.dataframe(pd.DataFrame(sc_table_data), use_container_width=True, hide_index=True)

        st.markdown("**Sampled Stochastic Responses ($k=5$):**")
        for i, resp in enumerate(res.get("consistency_responses", []), 1):
            st.text(f"Candidate {i}: {resp}")

    # --------------------------------------------------------------------------
    # Tab 3: NLI Agreement
    # --------------------------------------------------------------------------
    with tab_nli:
        st.markdown("#### Natural Language Inference (NLI) Agreement")
        st.write(
            "NLI measures bidirectional entailment and contradiction relationships among generated candidates "
            "using `cross-encoder/nli-MiniLM2-L6-H768` (10 pairs / 20 forward evaluations). "
            "Elevated contradiction rates signal semantic divergence."
        )

        nli_sig = res.get("nli_scores", res.get("nli_features", {}))
        nli_table_data = [
            {"Signal": "mean_pairwise_entailment", "Value": f"{nli_sig.get('mean_pairwise_entailment', 0.0):.4f}", "Description": "Average cross-encoder entailment probability across candidate pairs"},
            {"Signal": "mean_pairwise_contradiction", "Value": f"{nli_sig.get('mean_pairwise_contradiction', 0.0):.4f}", "Description": "Average cross-encoder contradiction probability across candidate pairs"},
            {"Signal": "nli_disagreement", "Value": f"{nli_sig.get('nli_disagreement', 0.0):.4f}", "Description": "Composite NLI disagreement index [contradiction + 0.5 * (1 - entailment)]"},
        ]
        st.dataframe(pd.DataFrame(nli_table_data), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # Tab 4: 19-Feature Vector
    # --------------------------------------------------------------------------
    with tab_vec:
        st.markdown("#### Exact 19-Feature Classifier Vector (Canonical Order)")
        st.write(
            "The exact tabular feature vector provided to the Phase 14 Logistic Regression and XGBoost classifiers. "
            "Non-signal metadata and sequence length (`num_tokens`) are strictly excluded."
        )

        f_vec = res["feature_vector"]
        vector_rows = []
        for idx, feat in enumerate(UNIVERSAL_CORE_FEATURES, 1):
            vector_rows.append({
                "#": idx,
                "Feature Name": feat,
                "Value": f"{f_vec[feat]:.6f}",
            })
        st.dataframe(pd.DataFrame(vector_rows), use_container_width=True, hide_index=True)

    # ==========================================================================
    # Pipeline Latency & Stage Profiling (Instrumentation)
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 7. Execution Latency & Pipeline Profiling")
    timing_data = res.get("stage_latencies_ms", res.get("timing_ms", {}))
    total_elapsed = res.get("total_latency_ms", sum(timing_data.values()) if timing_data else 0.0)

    st.write(
        f"Live pipeline execution completed in **{total_elapsed / 1000.0:.2f} s** ({total_elapsed:.1f} ms). "
        "Per-stage latency breakdown across all 15 pipeline stages:"
    )

    if timing_data:
        timing_rows = []
        for s_name, s_ms in timing_data.items():
            pct = (s_ms / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            timing_rows.append({
                "Stage": s_name,
                "Latency (ms)": f"{s_ms:.2f}",
                "Share (%)": f"{pct:.1f}%",
            })
        st.dataframe(pd.DataFrame(timing_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# 5. Offline Research Benchmark Results (Reference)
# ==============================================================================

st.markdown("---")
with st.expander("📊 Research Benchmark Results (Offline Holdout Evaluation)", expanded=False):
    st.write(
        "**Authoritative empirical benchmark results** evaluated on the standardized, fixed stratified 80/20 holdout test set "
        "($3,200$ train / $800$ test, balanced 50% faithful / 50% hallucinated across HaluEval, TruthfulQA, and FEVER). "
        "The live query above is evaluated independently in real-time."
    )

    tab_models, tab_ablation, tab_generalization, tab_calibration, tab_retrieval = st.tabs([
        "🏆 Multi-Model Benchmarks",
        "🔬 Feature-Family Ablation",
        "🌐 Cross-Dataset (LODO)",
        "🎯 Calibration Analysis",
        "📚 Retrieval Augmentation",
    ])

    with tab_models:
        st.markdown("#### Primary Multi-Model Evaluation (Universal Core 19-Feature Detector)")
        st.write(
            "Identical 19-feature Universal Core Detector retrained and evaluated across three distinct causal LLM architectures "
            "on the 800-instance holdout test partition:"
        )

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

        st.caption(
            f"Currently active generative model in sidebar: **{selected_model_name}**. "
            "All models use identical 19 canonical features and split protocol."
        )

        st.markdown("##### Task Domain Subgroup Breakdown (Qwen 3.5 Universal Core, N=800)")
        subgroup_data = [
            {"Domain / Benchmark": "HaluEval (N=300)", "Task Type": "Dialogue, QA, Summarization (with context)", "XGB Accuracy": "86.33%", "XGB ROC-AUC": "0.9433", "LR Accuracy": "84.00%", "LR ROC-AUC": "0.9311"},
            {"Domain / Benchmark": "TruthfulQA (N=300)", "Task Type": "Closed-book Human Misconceptions", "XGB Accuracy": "59.33%", "XGB ROC-AUC": "0.6066", "LR Accuracy": "51.33%", "LR ROC-AUC": "0.5210"},
            {"Domain / Benchmark": "FEVER (N=200)", "Task Type": "Wikipedia Factoid Verification (no open web)", "XGB Accuracy": "53.00%", "XGB ROC-AUC": "0.5441", "LR Accuracy": "48.50%", "LR ROC-AUC": "0.4870"},
        ]
        st.dataframe(pd.DataFrame(subgroup_data), use_container_width=True, hide_index=True)

    with tab_ablation:
        st.markdown("#### Feature-Family Ablation Study (Phase 15 Holdout Test Split)")
        st.write(
            "Systematic isolation of the three primary signal families on the fixed 800-instance holdout test set:"
        )

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

        st.markdown(
            "**Key Findings**:\n"
            "1. **Internal Signals Anchor Detection**: Standalone internal token log-probabilities and entropy provide the essential backbone (ROC-AUC `0.7397` for XGBoost).\n"
            "2. **Black-Box Probes Need Grounding**: Standalone self-consistency (`0.4896`) and NLI (`0.4912`) fail to beat random guessing without internal certainty signals.\n"
            "3. **Multi-Modal Synergistic Gain**: Fusing internal signals with behavioral consistency and NLI delivers the peak ROC-AUC (`0.7688`), raising PR-AUC by $+0.0466$."
        )

    with tab_generalization:
        st.markdown("#### Out-of-Domain Generalization (Phase 16 Leave-One-Dataset-Out / LODO)")
        st.write(
            "Rigorous generalization test: Models are trained exclusively on two benchmark datasets and tested on the 100% unseen third dataset:"
        )

        lodo_data = [
            {"Held-Out Benchmark (Test)", "Training Benchmarks", "Detector Features", "XGBoost Accuracy", "XGBoost ROC-AUC", "LR Accuracy", "LR ROC-AUC"},
            {"FEVER (N=1,000)", "HaluEval + TruthfulQA (N=3,000)", "Universal Core (19)", "50.60%", "0.5167", "49.60%", "0.4875"},
            {"FEVER (N=1,000)", "HaluEval + TruthfulQA (N=3,000)", "Internal Only (11)", "48.30%", "0.4903", "50.60%", "0.4868"},
            {"TruthfulQA (N=1,500)", "HaluEval + FEVER (N=2,500)", "Universal Core (19)", "48.53%", "0.4608", "47.53%", "0.4575"},
            {"TruthfulQA (N=1,500)", "HaluEval + FEVER (N=2,500)", "Internal Only (11)", "47.93%", "0.4539", "46.60%", "0.4439"},
            {"HaluEval (N=1,500)", "TruthfulQA + FEVER (N=2,500)", "Universal Core (19)", "43.20%", "0.4368", "44.93%", "0.2583"},
            {"HaluEval (N=1,500)", "TruthfulQA + FEVER (N=2,500)", "Internal Only (11)", "39.80%", "0.3599", "21.40%", "0.1601"},
        ]
        st.dataframe(pd.DataFrame(lodo_data), use_container_width=True, hide_index=True)

        st.markdown(
            "**Takeaways**:\n"
            "- Substantial domain shift exists across diverse hallucination formulations (contextual QA vs factoid verification vs human misconceptions).\n"
            "- Multi-signal integration buffers against out-of-domain degradation (improving XGBoost ROC-AUC from `0.3599` to `0.4368` on HaluEval transfer).\n"
            "- Training across heterogeneous multi-benchmark corpora is strictly necessary for robust generalization."
        )

    with tab_calibration:
        st.markdown("#### Probability Calibration Analysis (Phase 17 Nested Split)")
        st.write(
            "Evaluated via leak-free 3-way nested partition ($2,560$ train / $640$ calibration / $800$ holdout test):"
        )

        cal_data = [
            {"Model", "Calibration Method", "Accuracy", "F1 Score", "ROC-AUC", "PR-AUC", "Brier Score", "ECE (Calibration Error)", "MCE"},
            {"XGBoost", "Uncalibrated", "64.88%", "0.6492", "0.7638", "0.7787", "0.1900", "0.0712", "0.1888"},
            {"XGBoost", "Sigmoid (Platt Scaling)", "65.50%", "0.6452", "0.7638", "0.7787", "0.1935", "0.0825", "0.1210"},
            {"XGBoost", "Isotonic Regression", "66.75%", "0.6472", "0.7641", "0.7557", "0.1890", "0.0244", "0.1256"},
            {"Logistic Regression", "Uncalibrated", "63.38%", "0.6387", "0.7069", "0.7096", "0.2155", "0.0444", "0.1248"},
            {"Logistic Regression", "Sigmoid (Platt Scaling)", "63.00%", "0.6373", "0.7069", "0.7096", "0.2165", "0.0547", "0.1114"},
            {"Logistic Regression", "Isotonic Regression", "62.25%", "0.6505", "0.6971", "0.6751", "0.2151", "0.0341", "0.2473"},
        ]
        st.dataframe(pd.DataFrame(cal_data), use_container_width=True, hide_index=True)

        st.markdown(
            "**Key Findings**:\n"
            "- **Isotonic Regression Minimizes ECE**: Cuts Expected Calibration Error (ECE) for XGBoost from `0.0712` down to **`0.0244`** (a **65.7% reduction**).\n"
            "- **Rank Invariance**: Platt scaling preserves exact rank ordering, maintaining identical ROC-AUC and PR-AUC.\n"
            "- Calibrated probabilities provide trustworthy hallucination confidence scores suitable for decision-critical human review."
        )

    with tab_retrieval:
        st.markdown("#### Phase 19 Retrieval-Augmented Standalone Experiment")
        st.write(
            "Evaluated strictly on the $N=2,500$ population where reference evidence passages exist (HaluEval: 1,500, FEVER: 1,000) "
            "using a dedicated 500-instance holdout split ($2,000$ train / $500$ test, balanced 250/250):"
        )

        retrieval_data = [
            {"Experimental Condition", "Features", "Model", "Accuracy", "F1 Score", "ROC-AUC", "PR-AUC", "Brier Score", "ECE"},
            {"Condition A (Internal-Only)", 11, "Logistic Regression", "74.60%", "0.7581", "0.8463", "0.8486", "0.1597", "0.0494"},
            {"Condition A (Internal-Only)", 11, "XGBoost", "78.80%", "0.8000", "0.8978", "0.8982", "0.1264", "0.0323"},
            {"Condition B (Universal Core)", 19, "Logistic Regression", "75.80%", "0.7660", "0.8582", "0.8657", "0.1538", "0.0738"},
            {"Condition B (Universal Core)", 19, "XGBoost", "78.60%", "0.7864", "0.9011", "0.9035", "0.1246", "0.0482"},
            {"Condition C (Retrieval-Only)", 6, "Logistic Regression", "65.00%", "0.6824", "0.6990", "0.6543", "0.2187", "0.0350"},
            {"Condition C (Retrieval-Only)", 6, "XGBoost", "68.00%", "0.7193", "0.7491", "0.7293", "0.2053", "0.0523"},
            {"Condition D (Internal + Retrieval)", 17, "Logistic Regression", "76.60%", "0.7754", "0.8637", "0.8675", "0.1505", "0.0380"},
            {"Condition D (Internal + Retrieval)", 17, "XGBoost", "80.60%", "0.8159", "0.9184", "0.9214", "0.1180", "0.0593"},
            {"Condition E (Universal Core + Retrieval)", 25, "Logistic Regression", "76.20%", "0.7680", "0.8706", "0.8823", "0.1459", "0.0606"},
            {"Condition E (Universal Core + Retrieval)", 25, "XGBoost", "81.20%", "0.8178", "0.9169", "0.9201", "0.1185", "0.0518"},
        ]
        st.dataframe(pd.DataFrame(retrieval_data), use_container_width=True, hide_index=True)

        st.caption(
            "Note: TruthfulQA (N=1,500) was excluded because its reference context consists of external web URLs rather than local text passages. "
            "When evidence passages are available, adding retrieval agreement elevates ROC-AUC past 0.918."
        )

# ==============================================================================
# 6. Technical Details & Pipeline Architecture
# ==============================================================================

with st.expander("⚙️ How the Detector Works (System Pipeline)", expanded=False):
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
Every live inference run in this application executes 15 distinct, profiled pipeline stages in `app/inference.py`:

| Stage | Name | Description |
| :---: | :--- | :--- |
| **1** | **Model Loading & Cache Resolution** | Loads or resolves the selected causal LM, tokenizer, embedding model, NLI cross-encoder, and trained classifiers from memory cache. |
| **2** | **Primary Response Generation** | Deterministic generation ($T=0$, greedy decoding) using model-specific prompt templates and thinking-token suppression where applicable. |
| **3** | **Token & Logit Extraction** | Single forward pass computing per-token output logits, vocab softmax distribution, and loss tensor. |
| **4** | **Internal Signal Computation** | Calculates 11 white-box generation uncertainty metrics (log-probs, predictive entropy, perplexity, and token rank dispersion). |
| **5** | **Self-Consistency Sampling** | Generates $k=5$ stochastic responses ($T=0.7, \\text{top\\_}p=0.9, \\text{max\\_tokens}=128$) to probe the model's semantic stability. |
| **6** | **Embedding Model Verification** | Confirms `sentence-transformers/all-MiniLM-L6-v2` dense embedding model availability. |
| **7** | **Consistency Similarity Computation** | Encodes all candidate responses into dense vectors and computes the full pairwise cosine similarity matrix and exact-match score. |
| **8** | **NLI Model Verification** | Confirms `cross-encoder/nli-MiniLM2-L6-H768` cross-encoder availability and output label mappings. |
| **9** | **Pairwise Bidirectional NLI Inference** | Executes $k(k-1) = 20$ directional hypothesis-premise inference passes across the 10 response pairs. |
| **10** | **Universal Feature Vector Assembly** | Compiles the canonical 19-dimensional feature vector, strictly verifying that sequence length confounders are excluded. |
| **11** | **XGBoost Prediction** | Computes the primary non-linear hallucination risk probability using the fitted gradient-boosted decision trees. |
| **12** | **Logistic Regression Prediction** | Computes the linear baseline risk probability using the standardized Logistic Regression pipeline. |
| **13** | **Explainability & Attribution** | Calculates TreeSHAP log-odds feature contributions to explain why the classifiers flagged or cleared the response. |
| **14** | **Evidence Groundedness Analysis** | *(Optional)* Chunks reference context and computes top-1 cosine similarity, evidence margin, and claim agreement. |
| **15** | **Final UI Result Assembly** | Formats per-token probability and entropy tooltips, determines categorical risk tier, and tabulates millisecond stage latencies. |

#### The 19 Canonical Universal Features
The 19 features are partitioned into three orthogonal signal families:

1. **Internal Generation Dynamics (11 features)**:
   - `min_log_prob`: Minimum log-probability across all generated tokens (worst-case token confidence).
   - `mean_log_prob`: Length-normalized average log-probability $\\frac{1}{T}\\sum \\ln P(y_t)$.
   - `mean_token_prob`: Average token probability $\\frac{1}{T}\\sum P(y_t)$.
   - `token_prob_std`: Standard deviation of token probabilities across the sequence.
   - `mean_entropy`: Average Shannon entropy $\\frac{1}{T}\\sum H(P_t)$ measuring vocabulary distribution flatness.
   - `max_entropy`: Peak token entropy $\\max_t H(P_t)$ identifying localized hallucination pivot points.
   - `entropy_std`: Volatility/dispersion of predictive entropy across the generation.
   - `log_perplexity`: Logarithm of perplexity $\\ln(\\text{PPL})$.
   - `log_mean_token_rank`: Log of average rank of sampled tokens in the unconstrained vocabulary distribution.
   - `log_max_token_rank`: Log of maximum rank of any sampled token (captures tail-distribution sampling).
   - `log_rank_std`: Dispersion of token ranks across the sequence.

2. **Behavioral Self-Consistency (5 features, $k=5$)**:
   - `exact_match_agreement`: Fraction of stochastic generations that exactly match the primary response.
   - `mean_pairwise_similarity`: Mean cosine similarity across all 10 candidate response embedding pairs (`all-MiniLM-L6-v2`).
   - `min_pairwise_similarity`: Minimum pairwise similarity (lowest semantic consensus between any two candidates).
   - `max_pairwise_similarity`: Maximum pairwise similarity.
   - `pairwise_similarity_std`: Variance of semantic consensus across stochastic generations.

3. **Natural Language Inference Agreement (3 features, 20 directional passes)**:
   - `mean_pairwise_entailment`: Average entailment probability assigned by `nli-MiniLM2-L6-H768` across all pairs.
   - `mean_pairwise_contradiction`: Average contradiction probability across candidate pairs.
   - `nli_disagreement`: Mean symmetric disagreement rate $1 - \\text{Entailment} + \\text{Contradiction}$.

#### Strict Anti-Confounder Safeguards (Feature Isolation Policy)
- **Sequence Length Isolation**: `num_tokens` is strictly excluded from all training and inference feature sets. As proven in the Phase 7 ablation study, raw token count creates an artificial length shortcut where longer responses are trivially flagged, degrading scientific validity.
- **Metadata Exclusion**: Prompt text, dataset origins, raw IDs, and benchmark labels are completely withheld from the classifier.
- **Hardware Acceleration**: Models run in bfloat16/float16 with PyTorch CUDA tensor execution on RTX 3050 GPU, with automatic CPU fallback.
    """)

