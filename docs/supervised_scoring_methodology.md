# Supervised Signal Scoring Methodology & Dual-Path Architecture

## 1. Executive Summary: The Dual-Path Signal Architecture

This document formalizes the methodology for extracting internal language model signals (probabilities, predictive entropy, perplexity, and token ranks) for hallucination detection across **two distinct operational paths**:

```
Path A: Supervised Feature Extraction (Training & Evaluation Data)
=================================================================
[Dataset Row: Prompt + Context]  +  [Original Labeled Response]  ---> Forward Pass Scoring ---> [Internal Signal Features] + [Exact Benchmark Label]
                                                                                               (NO Generation, Exact Ground-Truth Preserved)

Path B: Real-Time Generation & Inference (Demo & Production System)
==================================================================
[User / Benchmark Prompt]  ---> Qwen Generation (Direct Answer)  ---> Forward Pass Scoring ---> [Inference Signals] ---> [Trained Classifier] ---> Prediction
```

---

## 2. Theoretical Grounding: Why the Benchmark Label Must Correspond to the Scored Response

### The Label Mismatch Trap
Standard hallucination benchmarks (such as **HaluEval**, **TruthfulQA**, and **FEVER**) provide triples:
$$(x_i, c_i, y_i)$$
where:
- $x_i$: The prompt / question
- $c_i$: The context passage or evidence (if applicable)
- $y_i$: A **specific, curated response candidate**
- $L_i \in \{0, 1\}$: The ground-truth label strictly evaluating candidate $y_i$ ($0 = \text{Faithful / Supported}$, $1 = \text{Hallucinated / Refuted}$).

If a pipeline instead generates an entirely **new response** $y'_i \sim P_{\text{LM}}(\cdot \mid x_i, c_i)$ and naively attaches the original label $L_i$ to $y'_i$, a severe **label mismatch** occurs:
1. **False Positive Labeling**: If example $i$ had a hallucinated benchmark response ($L_i = 1$), but Qwen generates a completely factual and correct answer $y'_i$, the classifier would be trained to penalize high-confidence factual generation.
2. **False Negative Labeling**: If example $i$ had a faithful benchmark response ($L_i = 0$), but Qwen hallucinates or fabricates details in $y'_i$, the classifier would be trained to treat hallucinated tokens as ground-truth faithful generation.

### The Correct Supervised Methodology
To construct an uncorrupted supervised dataset for training tabular classifiers (XGBoost, Random Forest, Logistic Regression, MLPs):
- The model is prompted with the conditioning prefix $(x_i, c_i)$.
- The **original benchmark response $y_i$** is supplied as the continuation sequence.
- An unadulterated forward pass evaluates the internal probability distribution and entropy that the model *assigns* to $y_i$.
- The ground-truth label $L_i$ remains 100% faithful to $y_i$.

---

## 3. Path A: Supervised Benchmark-Response Scoring (`score_labeled_responses.py`)

### Operational Workflow
1. **Input**: Canonical processed benchmarks from `data/processed/` (`halueval`, `truthfulqa`, `fever`, or `combined`).
2. **Conditioning Formatting**:
   - Prompts are formatted through the official ChatML template with `enable_thinking=False` and a direct factual directive:
     - **HaluEval**: `Passage: {context}\n\nQuestion: {prompt}`
     - **TruthfulQA**: `Question: {prompt}`
     - **FEVER**: `Verify whether the following claim is supported by factual evidence.\nClaim: `
   - Assistant prefix terminates cleanly: `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
3. **Continuation Scoring**:
   - The labeled string $y_i$ is encoded as the continuation.
   - `extract_raw_sequence_signals` performs a single forward pass over $[P_i, y_i]$.
   - Response logits are isolated via causal shifting: $\text{logits}[P_i - 1 : P_i + R_i - 1]$.
4. **Calculated Feature Space**:
   - Token-level probability statistics: `mean_token_prob`, `min_token_prob`, `token_prob_std`
   - Log-probability statistics: `mean_log_prob`, `min_log_prob`
   - Predictive Shannon entropy: `mean_entropy`, `max_entropy`, `entropy_std`
   - Sequence perplexity: $\exp(-\text{mean\_log\_prob})$
   - Vocabulary rank dispersion: `mean_token_rank`, `max_token_rank`, `min_token_rank`, `rank_std`
   - Metadata: `num_tokens`, `forward_time_s`
5. **Preserved Attributes**:
   - Canonical `id`, `source_dataset`, `label`, `prompt`, `context`, `response`.

### High-Throughput Efficiency & Checkpointing
- **Latency**: Because no autoregressive generation search or token-by-token sampling is needed, a single forward pass requires only **~105 ms** on the NVIDIA RTX 3050 Laptop GPU (compared to ~2,480 ms for generation).
- **Total Compute**: Scoring all 4,000 examples in `combined_processed.parquet` takes approximately **7 minutes**.
- **Interrupt-Resume Tolerance**: Progress is saved every `--checkpoint-interval` (default 100). If the process is halted, re-running with `--resume` detects the partial parquet checkpoint, skips already scored IDs, and completes without data loss.

---

## 4. Path B: Real-Time Generated-Response Scoring (`generate_signals.py`)

Path B is reserved for the live application, interactive Streamlit demo, and evaluation of free-form model outputs:
1. **Generation**: Qwen generates a fresh answer $y'$ using greedy decoding (`do_sample=False`, `max_new_tokens=128`) with `enable_thinking=False`.
2. **Signal Extraction**: The generated response $y'$ is scored to compute the exact same feature vector.
3. **Inference**: The trained supervised classifier receives the feature vector and outputs $\hat{P}(\text{Hallucination} \mid y')$.

---

## 5. Summary of Schema Comparison

| Column | Supervised Dataset (`score_labeled_responses.py`) | Generation Dataset (`generate_signals.py`) |
| :--- | :--- | :--- |
| `id` | Canonical benchmark sample ID | Canonical benchmark sample ID |
| `source_dataset` | `halueval` / `truthfulqa` / `fever` | `halueval` / `truthfulqa` / `fever` |
| `label` | **True ground truth of scored response** ($0/1$) | Original benchmark label (informative only) |
| `response` | **Original benchmark response being scored** | Original benchmark response |
| `generated_response` | *N/A (No new generation)* | Newly generated answer from Qwen |
| `num_tokens` | Token count of benchmark response | Token count of generated answer |
| Signal Columns | Extracted over benchmark response tokens | Extracted over generated answer tokens |
| Intended Use | **Supervised model training and validation** | **End-to-end demo and inference validation** |
