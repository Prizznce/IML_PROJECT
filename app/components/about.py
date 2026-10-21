"""
About & Platform Dossier Component:
Comprehensive Applied ML Research Documentation, System Architecture,
The 19 Canonical Features, Empirical Benchmarks, and Datasets.
"""

import pandas as pd
import streamlit as st

from app.components.benchmarks import render_validation_results, render_verification_pipeline


def render_about_view():
    """
    Render the comprehensive About & Platform Dossier page, incorporating:
    - Scientific premise & executive summary
    - Key empirical headline metrics
    - 15-stage neural verification architecture & flow diagram
    - 19 canonical universal feature specifications & formulations
    - Full holdout benchmark matrices, ablation studies, and calibration proofs
    - Curated dataset portfolio & methodology
    - Hardware acceleration & tech stack specifications
    """
    # Linear Header
    st.markdown(
        """
        <div class="linear-header">
            <div class="linear-title-row">
                <div>
                    <h1 class="linear-title">About VeritasLM</h1>
                    <p class="linear-subtitle" style="margin-top: 4px; margin-bottom: 0;">
                        White-box epistemic uncertainty estimation, multi-signal hallucination triangulation, and real-time neural defense.
                    </p>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="about-version-chip">v2.4 Core</span>
                    <span class="about-badge-status"><span class="sidebar-pulse-dot"></span> OPERATIONAL</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Executive Hero Banner
    st.markdown(
        """
        <div class="about-hero-card">
            <div class="about-hero-tag">APPLIED MACHINE LEARNING RESEARCH & OBSERVABILITY PLATFORM</div>
            <h2 class="about-hero-headline">Catching an LLM Lying from Internal Generation Dynamics</h2>
            <p class="about-hero-lead">
                Most commercial hallucination mitigations treat the Large Language Model as an opaque black box—relying on
                expensive multi-agent consensus debates, recursive self-prompting loops, or high-latency search retrievals.
                <strong>VeritasLM</strong> introduces the <strong>Universal Core Detector</strong>: an ultra-lightweight,
                interpretable supervised classifier that evaluates hallucination risk in real-time from the model's own
                internal token logits, Shannon entropy, token rank dispersion, and cross-encoder consensus.
            </p>
            <div class="about-metric-summary-grid">
                <div class="about-metric-summary-card">
                    <div class="about-metric-kicker">CROSS-MODEL PEAK</div>
                    <div class="about-metric-value">0.7913 <span class="about-metric-unit">ROC-AUC</span></div>
                    <div class="about-metric-detail">Google Gemma 3 (1B IT) Holdout</div>
                </div>
                <div class="about-metric-summary-card">
                    <div class="about-metric-kicker">DUAL-CORE RETRIEVAL</div>
                    <div class="about-metric-value">0.9184 <span class="about-metric-unit">ROC-AUC</span></div>
                    <div class="about-metric-detail">Internal + Semantic Grounding</div>
                </div>
                <div class="about-metric-summary-card">
                    <div class="about-metric-kicker">FEATURE PURITY</div>
                    <div class="about-metric-value">0% <span class="about-metric-unit">Length Bias</span></div>
                    <div class="about-metric-detail">Strict Anti-Shortcut Isolation</div>
                </div>
                <div class="about-metric-summary-card">
                    <div class="about-metric-kicker">CALIBRATION ERROR</div>
                    <div class="about-metric-value">0.0244 <span class="about-metric-unit">ECE</span></div>
                    <div class="about-metric-detail">-65.7% via Isotonic Scaling</div>
                </div>
                <div class="about-metric-summary-card">
                    <div class="about-metric-kicker">FEATURE VECTOR</div>
                    <div class="about-metric-value">19 <span class="about-metric-unit">Signals</span></div>
                    <div class="about-metric-detail">3 Orthogonal Signal Families</div>
                </div>
                <div class="about-metric-summary-card">
                    <div class="about-metric-kicker">BENCHMARK AUDIT</div>
                    <div class="about-metric-value">4,000 <span class="about-metric-unit">Samples</span></div>
                    <div class="about-metric-detail">HaluEval · TruthfulQA · FEVER</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Tabbed Interface
    tab_arch, tab_features, tab_validation, tab_datasets, tab_stack = st.tabs([
        "15-Stage Architecture",
        "19 Canonical Features",
        "Validation Results",
        "Datasets & Methodology",
        "Technical Stack & Citations",
    ])

    # ==========================================================================
    # TAB 1: 15-STAGE ARCHITECTURE (Verification Pipeline from Benchmark Mode)
    # ==========================================================================
    with tab_arch:
        render_verification_pipeline(expanded=True)

    # ==========================================================================
    # TAB 2: 19 CANONICAL FEATURES
    # ==========================================================================
    with tab_features:
        st.markdown(
            """
            <div class="about-tab-header">
                <h3 class="about-tab-title">The 19 Canonical Universal Features</h3>
                <p class="about-tab-desc">
                    The Universal Core feature space is partitioned into three orthogonal signal families.
                    Multi-signal triangulation empirically demonstrates that while standalone behavioral signals collapse,
                    their fusion produces a robust, generalizable detector.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown(
                """
                <div class="about-feature-category-card">
                    <div class="about-category-badge brand-blue">11 INTERNAL SIGNALS</div>
                    <div class="about-category-subtitle">Extracted from Token Logits & Distributions</div>
                    <ul class="about-feature-list">
                        <li><code>min_log_prob</code>: Minimum token log-probability &min;<sub>t</sub> ln P(y<sub>t</sub>) (worst-case bottleneck).</li>
                        <li><code>mean_log_prob</code>: Length-normalized average log-probability <sup>1</sup>/<sub>T</sub> &sum; ln P(y<sub>t</sub>).</li>
                        <li><code>mean_token_prob</code>: Average token probability across output sequence.</li>
                        <li><code>token_prob_std</code>: Standard deviation of token probabilities.</li>
                        <li><code>mean_entropy</code>: Average Shannon entropy <sup>1</sup>/<sub>T</sub> &sum; H(P<sub>t</sub>) (distribution flatness).</li>
                        <li><code>max_entropy</code>: Maximum token entropy identifying localized pivot uncertainty.</li>
                        <li><code>entropy_std</code>: Volatility of predictive entropy across the generation.</li>
                        <li><code>log_perplexity</code>: Natural log of sequence perplexity ln(PPL).</li>
                        <li><code>log_mean_token_rank</code>: Natural log ln(1 + rank) of average token rank in vocab softmax.</li>
                        <li><code>log_max_token_rank</code>: Natural log ln(1 + rank) of maximum token rank (detects rare-token sampling).</li>
                        <li><code>log_rank_std</code>: Natural log ln(1 + std) of token rank dispersion across generation.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_f2:
            st.markdown(
                """
                <div class="about-feature-category-card">
                    <div class="about-category-badge brand-emerald">5 CONSISTENCY PROBES</div>
                    <div class="about-category-subtitle">Derived from k=5 Stochastic Samples (T=0.7)</div>
                    <ul class="about-feature-list">
                        <li><code>exact_match_agreement</code>: Fraction of stochastic completions matching the primary output exactly.</li>
                        <li><code>mean_pairwise_similarity</code>: Mean cosine similarity across all 10 response pairs (via <code>all-MiniLM-L6-v2</code>).</li>
                        <li><code>min_pairwise_similarity</code>: Minimum pairwise cosine similarity (lowest consensus outlier).</li>
                        <li><code>max_pairwise_similarity</code>: Maximum pairwise cosine similarity across candidates.</li>
                        <li><code>pairwise_similarity_std</code>: Dispersion of pairwise semantic embeddings.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_f3:
            st.markdown(
                """
                <div class="about-feature-category-card">
                    <div class="about-category-badge brand-amber">3 NLI AGREEMENT PROBES</div>
                    <div class="about-category-subtitle">k(k-1)=20 Directional Cross-Encoder Passes</div>
                    <ul class="about-feature-list">
                        <li><code>mean_pairwise_entailment</code>: Directional entailment probability across candidate pairs via <code>nli-MiniLM2-L6-H768</code>.</li>
                        <li><code>mean_pairwise_contradiction</code>: Directional contradiction probability across candidate pairs.</li>
                        <li><code>nli_disagreement</code>: Directional disagreement rate: Contradiction + 0.5 &times; Neutral.</li>
                    </ul>
                    <div style="margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border-subtle);">
                        <div class="about-category-badge brand-cyan">6 RETRIEVAL SIGNALS (OPT.)</div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                            Active when reference passages are supplied: query/response top-1 &amp; top-3 dense cosine similarities, evidence margin, and retrieval agreement.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # TAB 3: VALIDATION RESULTS (Relocated from Benchmark Mode)
    # ==========================================================================
    with tab_validation:
        render_validation_results(expanded=True)

    # ==========================================================================
    # TAB 4: DATASETS & METHODOLOGY
    # ==========================================================================
    with tab_datasets:
        st.markdown(
            """
            <div class="about-tab-header">
                <h3 class="about-tab-title">Curated Benchmark Portfolio (N = 4,000)</h3>
                <p class="about-tab-desc">
                    To guarantee rigorous generalization across distinct hallucination failure modes, empirical experiments
                    span three complementary task formulations balanced exactly 50% faithful and 50% hallucinated.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        dataset_overview = [
            {
                "Benchmark Dataset": "HaluEval",
                "Sample Count": "1,500 instances",
                "Task Formulation": "Dialogue, QA & Summarization",
                "Grounding Context": "Source reference passages provided in-context",
                "Target Failure Mode": "Contextual unfaithfulness & hallucinated entities",
            },
            {
                "Benchmark Dataset": "TruthfulQA",
                "Sample Count": "1,500 instances",
                "Task Formulation": "Human Misconceptions & Falsehoods",
                "Grounding Context": "Closed-book adversarial question answering",
                "Target Failure Mode": "Mimicry of human superstitions & falsehoods",
            },
            {
                "Benchmark Dataset": "FEVER",
                "Sample Count": "1,000 instances",
                "Task Formulation": "Wikipedia Factoid Verification",
                "Grounding Context": "Claim statements evaluated without web access",
                "Target Failure Mode": "Fact-checking failure & epistemic overconfidence",
            },
        ]
        st.dataframe(pd.DataFrame(dataset_overview), width="stretch", hide_index=True)

        st.markdown(
            """
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px;">
                <div class="about-feature-category-card">
                    <div class="about-category-badge brand-blue">PARTITIONING PROTOCOL</div>
                    <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6; margin-top: 8px;">
                        The 4,000-instance dataset is partitioned into a fixed, stratified <strong>80/20 train/test split</strong>
                        (3,200 training / 800 holdout test instances, <code>random_state=42</code>), preserving exact class
                        and dataset distribution balance.
                    </div>
                </div>
                <div class="about-feature-category-card">
                    <div class="about-category-badge brand-emerald">3-WAY CALIBRATION PARTITION</div>
                    <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6; margin-top: 8px;">
                        To prevent calibration data leakage, probability calibration is tuned on a nested 3-way split:
                        <strong>2,560 base training</strong> / <strong>640 calibration tuning</strong> / <strong>800 holdout test</strong>.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================================================
    # TAB 5: TECHNICAL STACK & CITATIONS
    # ==========================================================================
    with tab_stack:
        st.markdown(
            """
            <div class="about-tab-header">
                <h3 class="about-tab-title">Technical Infrastructure, Runtime & References</h3>
                <p class="about-tab-desc">
                    Engineered for high-throughput enterprise observability with PyTorch, Hugging Face, XGBoost, and Streamlit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown(
                """
                <div class="about-feature-category-card">
                    <div class="about-category-badge brand-blue">SOFTWARE ECOSYSTEM</div>
                    <ul class="about-stack-list">
                        <li><strong>Deep Learning Runtime:</strong> PyTorch 2.x with CUDA / MPS / CPU tensor acceleration</li>
                        <li><strong>Generative LLMs:</strong> Hugging Face Transformers (Qwen 3.5, Gemma 3, Llama 3.2)</li>
                        <li><strong>Semantic Embeddings:</strong> sentence-transformers (all-MiniLM-L6-v2)</li>
                        <li><strong>Cross-Encoder NLI:</strong> cross-encoder/nli-MiniLM2-L6-H768</li>
                        <li><strong>Supervised Classifiers:</strong> XGBoost 2.0+ (Tuned gradient boosted trees) & Scikit-learn (Calibrated Logistic Regression)</li>
                        <li><strong>Explainability Engine:</strong> TreeSHAP log-odds feature attribution</li>
                        <li><strong>Observability UI:</strong> Streamlit with Linear/Railway developer design system</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_s2:
            st.markdown(
                """
                <div class="about-feature-category-card">
                    <div class="about-category-badge brand-emerald">HARDWARE ACCELERATION</div>
                    <ul class="about-stack-list">
                        <li><strong>Accelerator Support:</strong> NVIDIA CUDA GPU acceleration (bfloat16/float16) with automated host CPU fallback</li>
                        <li><strong>Host Engine:</strong> Multi-threaded Host CPU execution with automatic tensor casting</li>
                        <li><strong>Memory Management:</strong> Streamlit persistent resource caching (<code>@st.cache_resource</code>) preventing redundant weight reloading</li>
                        <li><strong>Open Source License:</strong> MIT Open Source License</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="about-citation-box" style="margin-top: 18px;">
                <div class="about-category-badge brand-purple">ACADEMIC CITATION</div>
                <pre class="about-bibtex">
@software{veritaslm_hallucination_defense_2026,
  author = {Aditya et al.},
  title = {VeritasLM: Catching an LLM Lying from Internal Generation Signals},
  year = {2026},
  publisher = {GitHub},
  journal = {Applied Machine Learning Research Repository},
  howpublished = {\\url{https://github.com/Prizznce/IML_PROJECT}}
}
                </pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
