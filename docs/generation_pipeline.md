# Phase 5: Generation and Internal Signal Extraction Pipeline

## 1. Executive Overview
The **Generation and Internal Signal Extraction Pipeline** executes causal inference with language models and captures white-box token-level generation dynamics before sampling discretization. In this phase, we operationalize **Qwen/Qwen3.5-0.8B** on local GPU hardware (NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM) to generate responses across benchmark queries and compute calibrated uncertainty signals for real-time hallucination detection.

---

## 2. Model & Hardware Configuration

- **Target Model**: `Qwen/Qwen3.5-0.8B` (AutoModelForCausalLM)
  - Parameter count: ~800 Million
  - Architecture: Hybrid Linear Attention / State-Space Model + Transformer
  - Hugging Face Hub: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **Precision / Dtype**: `torch.bfloat16`
  - Unquantized native weights preserving unclipped token logits
  - Native Ampere bfloat16 hardware execution
- **Execution Device**: `cuda:0` (NVIDIA GeForce RTX 3050 Laptop GPU)
- **VRAM Budget & Footprint**:
  - Total VRAM: 6,144 MB (6.0 GB)
  - Model Base Footprint: ~1,500 MB
  - Active Inference Headroom: >4,000 MB available for batching and context expansion

---

## 3. Input Dataset & Conditioning

- **Source File**: `data/processed/combined_processed.parquet` (Phase 3 standardized canonical benchmark)
- **Deterministic Sampling**:
  - Sample size: Exactly 30 examples for pilot verification
  - Random seed: `42` (`pandas.DataFrame.sample(n=30, random_state=42)`)
  - Cross-dataset composition:
    - `halueval`: 15 examples (balanced 0/1)
    - `fever`: 9 examples (balanced 0/1)
    - `truthfulqa`: 6 examples (balanced 0/1)
- **Prompt Conditioning**:
  - **HaluEval**: Passes reference passage context along with question prompt (`Passage: {context}\n\nQuestion: {prompt}\nAnswer:`).
  - **FEVER**: Pairs claim verification instruction with claim proposition (`{prompt}\nClaim: {claim}\nAnswer:`).
  - **TruthfulQA**: Passes question prompt directly (`Question: {prompt}\nAnswer:`).

---

## 4. Generation Settings

To isolate intrinsic uncertainty from stochastic sampling noise, generation is configured deterministically:
- `do_sample`: `False` (Greedy argmax decoding)
- `max_new_tokens`: `64`
- `pad_token_id`: `tokenizer.eos_token_id` (or `tokenizer.pad_token_id`)

---

## 5. Raw-Logit Signal Extraction Method

Internal generation signals are extracted by executing an unadulterated forward pass over the complete sequence:
$$S = [x_0, x_1, \dots, x_{P-1}, y_0, y_1, \dots, y_{R-1}]$$
where $P$ is the prompt token length and $R$ is the generated response token length.

### Causal Shifting & Isolation
Due to autoregressive causal masking:
- Logit vector at sequence index $t$ predicts the probability distribution over token at index $t+1$.
- Specifically, the first generated token $y_0$ is predicted by logits at index $P - 1$.
- The response token logits $\mathbf{z}_0, \dots, \mathbf{z}_{R-1}$ reside strictly in slice $[P-1 : P+R-1]$.
- Prompt tokens $[0 : P-1]$ are strictly isolated from response metrics to prevent prompt length or style from biasing uncertainty estimation.

### Extracted Signals
1. **Raw Token Probability**:
   $$P(y_t \mid x, y_{<t}) = \text{Softmax}(\mathbf{z}_t)_{y_t} = \frac{\exp(z_{t, y_t})}{\sum_{v=1}^{V} \exp(z_{t, v})}$$
2. **Token Log Probability**:
   $$\log P(y_t \mid x, y_{<t})$$
3. **Predictive Shannon Entropy**:
   $$H_t = -\sum_{v=1}^{V} P(v) \log P(v)$$
   *Computed with stable handling for $\lim_{p \to 0} p \log p = 0$ to prevent numerical NaN/Inf.*
4. **Token Rank**:
   $$\text{Rank}(y_t) = 1 + \sum_{v \neq y_t} \mathbb{I}(z_{t, v} > z_{t, y_t})$$
5. **Sequence Dispersion Statistics**:
   - `mean_token_prob`, `min_token_prob`, `token_prob_std`
   - `mean_log_prob`, `min_log_prob`
   - `mean_entropy`, `max_entropy`, `entropy_std`
   - `perplexity`: $\exp(-\frac{1}{R} \sum_{t=0}^{R-1} \log P(y_t))$

---

## 6. Output Schema

The output dataset is saved to:
- `experiments/baselines/pilot_generation_signals.parquet`
- `experiments/baselines/pilot_generation_signals.csv`

| Column | Type | Description |
|---|---|---|
| `id` | `str` | Canonical benchmark sample identifier |
| `source_dataset` | `str` | Benchmark origin (`halueval`, `truthfulqa`, `fever`) |
| `label` | `int64` | Benchmark ground-truth label (0 = Faithful, 1 = Hallucinated) |
| `prompt` | `str` | Original prompt query |
| `context` | `str` | Original reference passage / evidence annotation |
| `response` | `str` | Original candidate answer / claim |
| `original_response` | `str` | Original candidate answer / claim |
| `model_input` | `str` | Full prompt fed into model during generation |
| `generated_response` | `str` | Generated response string from Qwen |
| `num_generated_tokens` | `int64` | Count of generated tokens evaluated |
| `mean_token_prob` | `float64` | Mean raw probability across generated tokens |
| `min_token_prob` | `float64` | Minimum single-token probability in sequence |
| `mean_log_prob` | `float64` | Mean log probability across tokens |
| `min_log_prob` | `float64` | Minimum log probability in sequence |
| `mean_entropy` | `float64` | Mean predictive Shannon entropy |
| `max_entropy` | `float64` | Peak predictive entropy in sequence |
| `entropy_std` | `float64` | Standard deviation of token entropy values |
| `token_prob_std` | `float64` | Standard deviation of token probabilities |
| `perplexity` | `float64` | Exponentiated negative mean log probability |

---

## 7. Quality & Integrity Assertions

The validation module (`validate_pilot_signals`) executes deterministic integrity assertions:
1. Exact row count matches sample size (30 in v1, 10 in v2).
2. Unique ID count equals row count (no duplication).
3. Generated responses are non-empty and non-whitespace.
4. No `NaN`, `inf`, or `-inf` across all 10 signal metrics.
5. Perplexity values are strictly positive ($> 0$).
6. Probability metrics are strictly bounded in $[0.0, 1.0]$.
7. Predictive entropy metrics are non-negative ($\ge 0.0$).
8. Labels remain binary $\{0, 1\}$.
9. Source dataset names belong to valid canonical set (`halueval`, `truthfulqa`, `fever`).
10. (v2 Protocol) Zero `<think>` or `</think>` tags present in extracted responses.

---

## 8. Protocol Evolution: Pilot v1 vs. Pilot v2

### Pilot v1 Observations & Bottlenecks
During the initial 30-example pilot:
- Qwen3.5-0.8B frequently generated internal reasoning tokens (`<think>\nThinking Process: ...`).
- 29 out of 30 examples reached the 64-token generation limit (`max_new_tokens=64`) before completing or even starting the actual answer.
- **Root Cause**: Qwen3.5 is trained with ChatML structure (`<|im_start|>user`, `<|im_start|>assistant`). When presented with raw plain-text completion prompts (`Passage: ... Question: ... Answer:`), the model's base distribution activates its chain-of-thought reasoning behavior.

### Pilot v2 Protocol Enhancements
To produce clean, direct answers for hallucination classification:
1. **ChatML Formatting with Suppressed Thinking**:
   - Prompts are formatted through `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`.
   - In Qwen3.5's official template, `enable_thinking=False` pre-fills `<think>\n\n</think>\n\n` into the prompt prefix, closing the thinking block before generation starts.
2. **Concise Direct Directive**:
   - System instruction: `"Answer the question directly, factually, and concisely. Do not provide preamble or internal thinking."`
3. **Calibrated Token Budget**:
   - Expanded `max_new_tokens` from 64 to 128. Analysis of the benchmark reveals canonical answers average 10.4 tokens with a 99th percentile of 34 tokens. A budget of 128 tokens allows 90%+ of answers to reach natural EOS completion (`<|im_end|>`) while preventing compute explosion for large-scale generation.
4. **Signal Extraction Isolation**:
   - Signal extraction operates strictly on the direct generated answer tokens, ensuring uncertainty metrics measure hallucination risk rather than scratchpad formulation.
