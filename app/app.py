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
            - **Status**: 🟢 Ready (RTX 3050 GPU)
            """
        )
    elif "Gemma" in selected_model_name:
        st.markdown(
            """
            - **Model**: `Gemma 3 (1B IT)`
            - **Parameters**: 1.0 Billion
            - **Detectors**: Dedicated Gemma 3 Classifiers
            - **Status**: 🟢 Ready (RTX 3050 GPU)
            """
        )
    else:
        st.markdown(
            """
            - **Model**: `Qwen 3.5 (0.8B)`
            - **Parameters**: 0.8 Billion
            - **Detectors**: Baseline on 4,000 samples
            - **ROC-AUC**: 0.7688 (XGB) / 0.7079 (LR)
            - **Status**: 🟢 Ready (RTX 3050 GPU)
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
        "**Offline benchmark results — 800-example holdout test set** (Phase 14 Universal Core Detector). "
        "The current single live query is evaluated independently and is not part of this benchmark."
    )

    benchmark_data = [
        {"Model": "XGBoost (Phase 14 Universal Core)", "Accuracy": "0.6788", "F1 Score": "0.6751", "ROC-AUC": "0.7688", "PR-AUC": "0.7795", "Brier Score": "0.1891", "ECE": "0.0341"},
        {"Model": "Logistic Regression (Phase 14 Universal Core)", "Accuracy": "0.6288", "F1 Score": "0.6356", "ROC-AUC": "0.7079", "PR-AUC": "0.7124", "Brier Score": "0.2151", "ECE": "0.0525"},
        {"Model": "XGBoost Baseline (Phase 6 Internal-Only)", "Accuracy": "0.6700", "F1 Score": "0.6887", "ROC-AUC": "0.7411", "PR-AUC": "0.7322", "Brier Score": "0.1999", "ECE": "0.0228"},
        {"Model": "Logistic Regression Baseline (Phase 6 Internal-Only)", "Accuracy": "0.6212", "F1 Score": "0.6363", "ROC-AUC": "0.6930", "PR-AUC": "0.6618", "Brier Score": "0.2177", "ECE": "0.0649"},
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

    st.markdown(
        "**Phase 19 Retrieval-Augmented Variant (Offline Evaluation)**:\n"
        "- Evaluated on N=2,500 subset (HaluEval + FEVER) with reference passages/pointers.\n"
        "- Condition E (Universal Core + Retrieval) achieved **ROC-AUC: 0.9184**, PR-AUC: 0.9192, Accuracy: 0.8160, F1: 0.8258.\n"
        "- *Note: This is an offline benchmark experiment on curated corpora, not a live-query accuracy claim.*"
    )

# ==============================================================================
# 6. Technical Details & Pipeline Architecture
# ==============================================================================

with st.expander("⚙️ How the Detector Works (System Pipeline)", expanded=False):
    st.markdown("""
```
User Question
      │
      ▼
Qwen3.5-0.8B (Primary Deterministic Generation)
      │
      ▼
Generated Response
      │
      ├────────────────────────┬────────────────────────┐
      ▼                        ▼                        ▼
11 Internal Signals      k=5 Stochastic Probes    NLI Cross-Encoder
(Logits, Probs, Entropy, (all-MiniLM-L6-v2,       (nli-MiniLM2-L6-H768,
 Token Rank Dispersion)   Pairwise Similarity)     20 Directional Passes)
      │                        │                        │
      └────────────────────────┼────────────────────────┘
                               │
                               ▼
                19 Canonical Universal Features
                               │
                               ▼
                 XGBoost & Logistic Regression
                               │
                               ▼
                   Hallucination Risk Score
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
Token Confidence Heatmap               Tree SHAP Explainability
(P(y_t), H_t per token)                ("Why was it flagged?")
```

#### Core Experimental Configurations:
- **Base Generative LM**: `Qwen/Qwen3.5-0.8B` (unquantized bfloat16, greedy decoding).
- **Self-Consistency**: $k = 5$ stochastic candidate responses ($T = 0.7, \\text{top\\_}p = 0.9, \\text{max\\_tokens} = 128$).
- **Sentence Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`.
- **NLI Cross-Encoder**: `cross-encoder/nli-MiniLM2-L6-H768`.
- **Primary Supervised Classifier**: `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=0.1)`.
- **Linear Baseline Classifier**: `StandardScaler + LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`.
- **Feature Isolation Policy**: Sequence length (`num_tokens`), prompt text, and dataset metadata are strictly excluded from feature space.
    """)
