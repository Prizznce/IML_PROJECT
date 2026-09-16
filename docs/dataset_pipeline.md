# Phase 3: Dataset Pipeline Documentation

## 1. Overview and Purpose
The goal of the dataset pipeline for **"Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals"** is to standardize heterogeneous hallucination benchmarks into a clean, unified schema while avoiding data contamination, class collapse, or unnecessary network overhead.

The canonical schema standardizes all benchmark data into 6 core fields:
```text
┌────────────────────────────────────────────────────────┐
│                   CANONICAL SCHEMA                     │
├──────────────────┬─────────────────────────────────────┤
│ id               │ Unique string identifier            │
│ prompt           │ Input query or question             │
│ response         │ Candidate LLM response / claim      │
│ context          │ Reference passage / evidence text   │
│ label            │ 0 = Faithful, 1 = Hallucinated      │
│ source_dataset   │ Dataset identifier                  │
└──────────────────┴─────────────────────────────────────┘
```

---

## 2. Datasets Used and Sources

### A. HaluEval (Primary Benchmark)
- **Hugging Face Hub**: [`flowaicom/HaluEval`](https://huggingface.co/datasets/flowaicom/HaluEval)
- **Role in Project**: Primary training and evaluation benchmark for question answering and dialogue hallucination detection.
- **Original Fields**: `id`, `passage`, `question`, `answer`, `label`, `source_ds`, `score`.
- **Label Mapping**:
  - `FAIL` (or `score == 0`) $\rightarrow$ **`1`** (Hallucinated / unsupported).
  - `PASS` (or `score == 1`) $\rightarrow$ **`0`** (Faithful / grounded).
  - *(Note: We explicitly avoid assuming the string `"Hallucinated"` as the label, which was a bug in prior naive implementations that caused 100% false mappings).*

### B. TruthfulQA (Generalization & Mimicry Benchmark)
- **Hugging Face Hub**: [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa), configuration: `"generation"`.
- **Role in Project**: Evaluates model susceptibility to human falsehoods, common myths, and conspiracy theories.
- **Original Fields**: `question`, `best_answer`, `correct_answers`, `incorrect_answers`, `source`, `category`.
- **Label Mapping**:
  - Truthful candidates (`best_answer`, `correct_answers`) $\rightarrow$ **`0`** (Faithful / truthful).
  - False / hallucinated candidates (`incorrect_answers`) $\rightarrow$ **`1`** (Hallucinated / untruthful).

### C. FEVER (Fact Extraction and VERification)
- **Source**: FEVER shared task dev split (`https://fever.ai/download/fever/shared_task_dev.jsonl`).
- **Role in Project**: Out-of-domain cross-dataset generalization benchmark.
- **Original Fields**: `id`, `claim`, `evidence`, `label`.
- **Label Mapping**:
  - `SUPPORTS` $\rightarrow$ **`0`** (Supported by Wikipedia evidence).
  - `REFUTES` $\rightarrow$ **`1`** (Refuted by Wikipedia evidence / false claim).
  - `NOT ENOUGH INFO` $\rightarrow$ Handled explicitly (see Section 6).

---

## 3. Why Prompt and Context Must Be Retained
Previous naive implementations stripped prompts and passages, keeping only `{"text": ..., "is_hallucinated": ...}`. This severely cripples hallucination detection for several fundamental reasons:

1. **Conditional Probability & Uncertainty Signals**: Token probabilities, perplexity, and predictive entropy cannot be computed in a vacuum—they must be conditioned on the input prompt ($P(\text{response} \mid \text{prompt})$).
2. **Self-Consistency**: Measuring consistency requires repeatedly prompting the LLM with the *same prompt* to evaluate semantic clustering among sampled outputs.
3. **Retrieval & Evidence Agreement**: Determining whether a response is supported requires computing Natural Language Inference (NLI) and BM25/dense retrieval alignment between the candidate response and the reference `context`/`passage`.

---

## 4. Why TruthfulQA Cannot Simply Use `best_answer` as Label 0
In the preliminary script (`1.py`), TruthfulQA was mapped by selecting only `best_answer` and hardcoding `is_hallucinated = 0`. This caused two critical failure modes:
1. **Single-Class Collapse**: If 100% of samples are labeled `0`, no supervised classifier (Logistic Regression, XGBoost) can be trained, and metrics like AUROC, Precision, Recall, and F1 become undefined.
2. **Discarding Hallucinations**: TruthfulQA's primary value is its curated list of `incorrect_answers`—the very falsehoods models tend to hallucinate. Our pipeline unrolls both correct answers (class 0) and incorrect answers (class 1) paired with their parent question.

---

## 5. Sampling and Class Balancing
To ensure rapid local iteration and avoid memory exhaustion on lightweight student hardware:
- A configurable sample target (e.g. 1,000–1,500 rows per dataset) is specified in `configs/data.yaml`.
- The `balance_and_sample()` helper partitions data by label and samples an equal number of class 0 and class 1 instances ($N/2$ each).
- The random state is fixed to `seed: 42` for exact reproducibility.

---

## 6. Handling of FEVER `NOT ENOUGH INFO`
In FEVER, claims labeled `NOT ENOUGH INFO` cannot be verified using the Wikipedia reference corpus.
- **Our Decision**: In binary classification, `NOT ENOUGH INFO` claims are **excluded by default** (`include_not_enough_info: false`).
- **Rationale**: A lack of documentation in a specific corpus is epistemically different from a fabricated contradiction (`REFUTES`). Conflating the two injects noisy labels into the classifier. If unsupported-claim detection is specifically tested, the setting can be enabled via `configs/data.yaml`.

---

## 7. Local Storage & Caching Structure
```text
Catching-an-LLM-Lying/
├── configs/
│   └── data.yaml                   # Central configuration
├── data/
│   ├── raw/                        # Downloaded raw JSONL/Parquet caches
│   └── processed/                  # Standardized Parquet files
│       ├── halueval_processed.parquet
│       ├── truthfulqa_processed.parquet
│       ├── fever_processed.parquet
│       └── combined_processed.parquet
```
All processed datasets are saved in Apache Parquet format, offering columnar compression, fast disk I/O, and strict schema preservation.

---

## 8. Execution Command
To execute the download and preprocessing pipeline:
```bash
python src/data/load_datasets.py --config configs/data.yaml --download
```
