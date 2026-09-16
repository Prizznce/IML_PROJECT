# Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals

## Project Overview

This project investigates whether internal Large Language Model (LLM) generation signals can be leveraged to detect hallucination risk in real-time without relying exclusively on post-hoc factual verification. Rather than treating an LLM as a black box and relying solely on external fact-checkers, this approach analyzes internal generation dynamics—such as token probability distributions, entropy variations, self-consistency across sampled paths, and retrieval agreement—to reliably and efficiently flag hallucinated or unsupported outputs.

## Research Question

> **"Can internal generation signals, self-consistency, and evidence-retrieval agreement be combined into a lightweight and interpretable classifier that estimates hallucination risk?"**

## Objectives

- **Investigate Generation Uncertainty Signals**: Analyze token-level probabilities, predictive entropy, perplexity, and related statistical dispersion measures.
- **Measure Self-Consistency**: Quantify semantic similarity and consistency across multiple stochastic generation paths.
- **Investigate Evidence & Retrieval Agreement**: Assess alignment between generated content and retrieved evidence using Natural Language Inference (NLI) and ranking metrics.
- **Build Classical ML Baselines**: Train and evaluate lightweight, interpretable baselines including Logistic Regression and XGBoost.
- **Build a Combined Hallucination-Risk Classifier**: Integrate probability/entropy metrics, self-consistency indicators, and retrieval agreement into an ensemble or calibrated classifier.
- **Evaluate Probability Calibration**: Measure and optimize calibration using Expected Calibration Error (ECE) to ensure risk scores reflect true probability of hallucination.
- **Perform Ablation and Error Analysis**: Systematically evaluate individual feature groups and analyze false positives/negatives to understand model behavior.

## Datasets

The project will evaluate performance across multiple benchmark datasets:

- **HaluEval** (Primary): Benchmark designed for evaluating hallucination detection across QA, dialogue, and summarization tasks.
- **TruthfulQA** (Additional Evaluation / Generalization): Probes whether models mimic human falsehoods and conspiracy theories.
- **FEVER** (Additional Evaluation / Generalization): Fact extraction and verification benchmark for assessing claim-evidence consistency.

*(Note: Datasets are planned and will be inspected and downloaded in subsequent phases.)*

## Planned ML Models

- **Logistic Regression**: Interpretable linear baseline with explicit feature coefficient inspection.
- **XGBoost**: Gradient-boosted decision trees for non-linear feature interaction and high tabular performance.
- **Small Transformer-based Model** *(Optional / Exploratory)*: Lightweight sequence/tabular encoder if computationally feasible on local hardware.

## Planned Features

- **Token Probability Statistics**: Mean token probability, minimum token probability, probability variance, tail-token probabilities.
- **Entropy Statistics**: Predictive entropy per token, maximum sequence entropy, entropy spikes.
- **Perplexity**: Generation perplexity and length-normalized sequence uncertainty.
- **Self-Consistency**: Semantic agreement across temperature-sampled generations.
- **Semantic Similarity**: Sentence-embedding cosine similarity among candidate answers.
- **NLI Agreement**: Natural Language Inference entailment/contradiction probabilities between outputs.
- **Retrieval Agreement**: BM25/dense retrieval alignment scores and claim-passage support metrics.

## Evaluation

The framework evaluates detection performance and calibration using:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Ranking & Discrimination**: Area Under the ROC Curve (AUROC), Precision@K, Recall@K, Mean Reciprocal Rank (MRR)
- **Calibration**: Expected Calibration Error (ECE), reliability diagrams
- **Diagnostic Studies**: Systematic ablation studies, cross-dataset generalization evaluation, and qualitative error analysis

## High-Level Pipeline

```
┌─────────────────┐
│ LLM Generation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Internal Signals│ (Token probs, logits, entropy)
└────────┬────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│Feature Extraction│ ──► │ Self-Consistency │ ──► │ Evidence Retrieval  │
└────────┬─────────┘     └──────────────────┘     └──────────┬──────────┘
         │                                                   │
         └───────────────────────┬───────────────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Feature Combination  │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │Lightweight Classifier │ (Logistic Reg / XGBoost)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │Hallucination Risk     │ + Interpretable
                     │Score [0, 1]           │   Explanation
                     └───────────────────────┘
```

## Project Structure

```
Catching-an-LLM-Lying/
├── data/
│   ├── raw/                # Original, immutable benchmark datasets
│   ├── processed/          # Cleaned, tokenized, and preprocessed datasets
│   └── external/           # External knowledge corpora or retrieval indexes
├── models/                 # Serialized model weights, checkpoints, and scalers
├── src/                    # Source code package
│   ├── data/               # Data loaders, downloaders, and cleaners
│   ├── generation/         # LLM interaction, prompt formatting, and generation
│   ├── signals/            # Extraction of internal logits, probabilities, and entropy
│   ├── consistency/        # Multi-sample generation and consistency scoring
│   ├── retrieval/          # BM25/dense evidence retrieval and indexing
│   ├── features/           # Feature vector assembly and tabular transformation
│   ├── training/           # Model training loops, cross-validation, hyperparam search
│   ├── evaluation/         # Metrics (AUROC, ECE, F1), calibration, and ablation tools
│   └── utils/              # Logging, I/O helpers, and common utilities
├── experiments/
│   ├── baselines/          # Baseline experiment runs and logs
│   ├── ablation/           # Feature ablation configurations and results
│   └── cross_dataset/      # Cross-dataset generalization tests
├── docs/                   # Environment and project documentation
├── notebooks/              # Jupyter notebooks for EDA and exploratory experiments
├── app/                    # Interactive demo interface (Streamlit)
├── tests/                  # Unit and integration test suites
├── configs/                # Configuration files (YAML/JSON)
├── logs/                   # Execution and experiment logs
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
└── main.py                 # Project entry point and environment verification
```

## Environment Setup

The project runs within a dedicated Python virtual environment to ensure dependency isolation and reproducibility.

### 1. Virtual Environment
The environment is located in `.venv/` at the project root.

- **PowerShell Activation**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Command Prompt Activation**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

### 2. Python Dependencies
Core dependencies are specified in `requirements.txt` and include:
- Tabular ML & Stats: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`
- NLP & Transformers: `transformers`, `sentence-transformers`, `datasets`, `evaluate`, `nltk`
- Retrieval: `rank-bm25`, `faiss-cpu`
- Exploration & UI: `jupyter`, `streamlit`, `matplotlib`, `seaborn`

### 3. PyTorch & Hardware Acceleration Support
- **PyTorch Build**: PyTorch CPU build (`torch-2.14.0+cpu`) is configured for this environment.
- **Hardware Mode**: Operating in verified CPU mode (Intel Iris Xe Graphics detected; no discrete NVIDIA GPU available).
- The framework is designed to remain lightweight and fully functional on standard CPU hardware.

## Development Status

**Phase 2 — Environment setup completed**
- Phase 1: Project structure and foundation initialized
- Phase 2: Virtual environment created, core ML dependencies installed, test suites and documentation configured

