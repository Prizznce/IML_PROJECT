"""
Streamlit Live Demonstration Application for LLM Hallucination Detection.

Title: Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination
Detection from Internal Generation Signals

This presentation layer calls app/inference.py to execute the complete 19-feature
Universal Core Detector on user-supplied queries or benchmark presets.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import streamlit as st

from app.inference import (
    UNIVERSAL_CORE_FEATURES,
    run_live_inference,
)

# ==============================================================================
# 1. Page Configuration and Header
# ==============================================================================

st.set_page_config(
    page_title="Catching an LLM Lying | Real-Time Hallucination Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Catching an LLM Lying")
st.subheader("Lightweight Hallucination Detection from Internal Generation Signals")

st.info(
    "**Research Demonstration**: Probability represents model-estimated hallucination risk, "
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

st.markdown("### 1. Input Query & Optional Context")

# Preset selection buttons
st.markdown("**Sample Presets** *(Click to populate the prompt field)*:")
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    if st.button("Preset 1: Moon Landing", use_container_width=True):
        st.session_state["user_prompt"] = "Who was the first person to walk on the Moon?"
        st.session_state["user_context"] = ""

with p_col2:
    if st.button("Preset 2: Capital of France", use_container_width=True):
        st.session_state["user_prompt"] = "What is the capital of France?"
        st.session_state["user_context"] = ""

with p_col3:
    if st.button("Preset 3: Mars Exploration", use_container_width=True):
        st.session_state["user_prompt"] = "Who was the first human to walk on Mars?"
        st.session_state["user_context"] = ""

with p_col4:
    if st.button("Clear Input", use_container_width=True):
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
    help="Optional background passage for document-grounded question answering.",
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
                result = run_live_inference(
                    prompt=clean_p,
                    context=context_text.strip() if context_text.strip() else None,
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
    st.markdown("**Generated Response (Qwen3.5-0.8B):**")
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

    st.markdown("---")
    st.markdown("### 3. Detailed Signal Breakdown")

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

        if "tokens" in int_sig and int_sig["tokens"]:
            with st.expander("Inspect Token-Level Logit Dynamics", expanded=False):
                token_rows = []
                for t in int_sig["tokens"]:
                    token_rows.append({
                        "Step": t["step"],
                        "Token": repr(t["token_text"]),
                        "Probability": f"{t['probability']:.4f}",
                        "Log-Prob": f"{t['log_probability']:.4f}",
                        "Entropy": f"{t['entropy']:.4f}",
                        "Rank": t["rank"],
                    })
                st.dataframe(pd.DataFrame(token_rows), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # Tab 2: Self-Consistency
    # --------------------------------------------------------------------------
    with tab_sc:
        st.markdown("#### Stochastic Behavioral Self-Consistency Probing")
        st.write(
            "Five stochastic responses ($k=5, T=0.7, \\text{top\\_}p=0.9$) are compared to measure behavioral "
            "consistency. Disagreement among candidate outputs indicates epistemic uncertainty."
        )

        sc_sig = res["self_consistency_features"]
        sc_table_data = [
            {"Signal": "exact_match_agreement", "Value": f"{sc_sig['exact_match_agreement']:.4f}", "Description": "Fraction of candidate pairs that match identically"},
            {"Signal": "mean_pairwise_similarity", "Value": f"{sc_sig['mean_pairwise_similarity']:.4f}", "Description": "Mean embedding cosine similarity among candidate outputs"},
            {"Signal": "min_pairwise_similarity", "Value": f"{sc_sig['min_pairwise_similarity']:.4f}", "Description": "Minimum pairwise cosine similarity"},
            {"Signal": "max_pairwise_similarity", "Value": f"{sc_sig['max_pairwise_similarity']:.4f}", "Description": "Maximum pairwise cosine similarity"},
            {"Signal": "pairwise_similarity_std", "Value": f"{sc_sig['pairwise_similarity_std']:.4f}", "Description": "Standard deviation of pairwise cosine similarities"},
        ]
        st.dataframe(pd.DataFrame(sc_table_data), use_container_width=True, hide_index=True)

        st.markdown("**Sampled Stochastic Responses ($k=5$):**")
        for i, resp in enumerate(res["consistency_responses"], 1):
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
            {"Signal": "mean_pairwise_entailment", "Value": f"{nli_sig['mean_pairwise_entailment']:.4f}", "Description": "Average cross-encoder entailment probability across candidate pairs"},
            {"Signal": "mean_pairwise_contradiction", "Value": f"{nli_sig['mean_pairwise_contradiction']:.4f}", "Description": "Average cross-encoder contradiction probability across candidate pairs"},
            {"Signal": "nli_disagreement", "Value": f"{nli_sig['nli_disagreement']:.4f}", "Description": "Composite NLI disagreement index [contradiction + 0.5 * (1 - entailment)]"},
        ]
        st.dataframe(pd.DataFrame(nli_table_data), use_container_width=True, hide_index=True)

        with st.expander("Candidate Responses Evaluated", expanded=False):
            for i, resp in enumerate(res["consistency_responses"], 1):
                st.text(f"Candidate {i}: {resp}")

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
```

#### Core Experimental Configurations:
- **Base Generative LM**: `Qwen/Qwen3.5-0.8B` (unquantized bfloat16).
- **Self-Consistency**: $k = 5$ stochastic candidate responses ($T = 0.7, \\text{top\\_}p = 0.9, \\text{max\\_tokens} = 128$).
- **Sentence Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`.
- **NLI Cross-Encoder**: `cross-encoder/nli-MiniLM2-L6-H768`.
- **Primary Supervised Classifier**: `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=0.1)`.
- **Linear Baseline Classifier**: `StandardScaler + LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`.
    """)
