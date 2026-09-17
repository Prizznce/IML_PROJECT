# Internal Generation Signals Documentation: Raw-Logit Extraction

## 1. Overview and Core Philosophy

In our research project, **"Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination Detection from Internal Generation Signals"**, we treat the Large Language Model not as an uninspectable black box, but as a probabilistic generative system whose internal uncertainty dynamics reveal whether an output is factually grounded or fabricated.

When an LLM hallucinates, its internal state transitions often exhibit statistical anomalies:
- Increased dispersion across the vocabulary (high predictive entropy).
- Sharp drops in conditional probability on entity tokens, relations, or factual claims.
- Elevated tail-token probabilities and degraded sequence perplexity.

The module [`src/signals/token_signals.py`](file:///c:/Aditya_workspace/IML_Project/IML_PROJECT/src/signals/token_signals.py) provides a reusable, mathematically verified engine for extracting these internal signals directly from the model's raw forward-pass logits.

---

## 2. Raw Forward-Pass Logits vs. Generation-Time `output_scores`

A central methodological distinction in our framework is between **raw unconstrained logits** and **generation-time sampling scores**:

```
                 RAW FORWARD PASS                             GENERATION-TIME ROLLOUT
    ┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
    │ Input: Prompt + Response Full Sequence │       │ Input: Autoregressive prompt decoding  │
    │ Logits: z_t directly from lm_head      │       │ Scores: output_scores per step         │
    │ Temperature: 1.0 (Unscaled)            │       │ Temperature: Scaled (e.g. 0.7)         │
    │ Vocabulary: Unconstrained full vocab   │       │ Filters: Top-k, Top-p, Min-p applied   │
    │ Applicable to: Generated text & Ground │       │ Applicable to: Newly generated tokens  │
    │ truth benchmark candidates             │       │ only                                   │
    └────────────────────────────────────────┘       └────────────────────────────────────────┘
```

### Why Raw Forward-Pass Logits Are Essential:
1. **Unadulterated Uncertainty**: Generation-time sampling often applies temperature division ($z / T$), top-$k$, top-$p$ (nucleus) filtering, or repetition penalties. These transformations artificially distort probability distributions and compress predictive entropy.
2. **Universal Applicability**: A raw forward pass can be executed over *any* arbitrary text—including pre-existing benchmark responses from HaluEval, TruthfulQA, and FEVER—enabling standardized, symmetric feature extraction for both faithful and hallucinated samples without re-generating them.
3. **Computational Efficiency**: A complete sequence of length $N$ is evaluated in a **single parallel forward pass**, avoiding slow autoregressive step-by-step token generation.

---

## 3. Causal Token Alignment Mechanics

In a causal autoregressive language model (decoder-only architecture), the self-attention mask ensures position $t$ can only attend to positions $\le t$. As a consequence:
$$\mathbf{z}_t = \text{Model}(x_0, x_1, \dots, x_t) \quad \implies \quad \mathbf{z}_t \text{ predicts token } x_{t+1}$$

### Sequence Partitioning and Index Shifting
Given an input prompt $x = [x_0, \dots, x_{P-1}]$ of length $P$, and a candidate response $y = [y_0, \dots, y_{R-1}]$ of length $R$:

```
Full Sequence: [ x_0,  x_1, ..., x_{P-1},  y_0,  y_1, ..., y_{R-1} ]  (Length N = P + R)
Index:            0,    1, ...,   P-1,     P,   P+1, ...,   N-1

Model Logits:  [ z_0,  z_1, ..., z_{P-1},  z_P, z_{P+1}, ..., z_{N-2}, z_{N-1} ]
                  │     │           │       │     │              │        │
Predicts:        x_1   x_2         y_0     y_1   y_2           y_{R-1}   (Next token)
```

1. **First Response Token ($y_0$)**: Predicted by the logit at index **$P - 1$** (the final token of the prompt).
2. **Intermediate Token ($y_t$)**: Predicted by the logit at index **$P - 1 + t$**.
3. **Prompt Exclusion**: Logits at indices $0 \dots P - 2$ predict prompt tokens $x_1 \dots x_{P-1}$ and are strictly discarded. Prompt tokens never contaminate response uncertainty statistics.

---

## 4. Signal Definitions and Mathematical Formulations

For each response step $t \in \{0, \dots, R-1\}$ with logit vector $\mathbf{z}_t \in \mathbb{R}^{V}$ and target token ID $y_t$:

### A. Token-Level Signals

| Signal | Notation | Mathematical Formula | Range | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Log Probability** | $\log P(y_t)$ | $\log \left( \frac{\exp(z_{t, y_t})}{\sum_{v \in V} \exp(z_{t, v})} \right) = \text{log\_softmax}(\mathbf{z}_t)_{y_t}$ | $(-\infty, 0]$ | Log-likelihood of generating $y_t$ given context. |
| **Raw Probability** | $P(y_t)$ | $\frac{\exp(z_{t, y_t})}{\sum_{v \in V} \exp(z_{t, v})} = \text{softmax}(\mathbf{z}_t)_{y_t}$ | $[0, 1]$ | Conditional confidence assigned to token $y_t$. |
| **Predictive Entropy** | $H_t$ | $-\sum_{v \in V} P(v) \log P(v)$ | $[0, \ln \|V\|]$ | Information entropy over entire vocabulary at step $t$. |
| **Token Rank** | $\text{Rank}(y_t)$ | $1 + \sum_{v \in V} \mathbb{I}(z_{t, v} > z_{t, y_t})$ | $\{1, \dots, \|V\|\}$ | Position of $y_t$ in greedy ordering (1 = argmax). |

### B. Sequence-Level Aggregated Statistics

Let $\mathcal{Y} = \{y_0, \dots, y_{R-1}\}$ denote the generated response tokens:

1. **Mean Token Probability**:
   $$\bar{P} = \frac{1}{R} \sum_{t=0}^{R-1} P(y_t \mid x, y_{<t})$$

2. **Minimum Token Probability**:
   $$P_{\min} = \min_{t \in \{0, \dots, R-1\}} P(y_t \mid x, y_{<t})$$
   *Crucial indicator: Hallucinations frequently introduce a single low-probability fabricated entity even when surrounding filler text has high probability.*

3. **Mean Log Probability**:
   $$\overline{\log P} = \frac{1}{R} \sum_{t=0}^{R-1} \log P(y_t \mid x, y_{<t})$$

4. **Minimum Log Probability**:
   $$\log P_{\min} = \min_{t \in \{0, \dots, R-1\}} \log P(y_t \mid x, y_{<t})$$

5. **Mean Predictive Entropy**:
   $$\bar{H} = \frac{1}{R} \sum_{t=0}^{R-1} H_t$$

6. **Maximum Predictive Entropy**:
   $$H_{\max} = \max_{t \in \{0, \dots, R-1\}} H_t$$
   *Signals acute decision uncertainty or factual branching points.*

7. **Entropy Standard Deviation**:
   $$\sigma_H = \sqrt{\frac{1}{R - 1} \sum_{t=0}^{R-1} (H_t - \bar{H})^2}$$

8. **Token Probability Standard Deviation**:
   $$\sigma_P = \sqrt{\frac{1}{R - 1} \sum_{t=0}^{R-1} (P(y_t) - \bar{P})^2}$$

9. **Sequence Perplexity**:
   $$\text{PPL} = \exp\left( -\frac{1}{R} \sum_{t=0}^{R-1} \log P(y_t \mid x, y_{<t}) \right) = \exp(-\overline{\log P})$$

---

## 5. Numerical Stability Guarantees

In modern LLMs with massive vocabularies ($V = 248,320$ in Qwen3.5), thousands of tokens receive extreme negative logits (e.g., $z_v \le -100$ or $-\infty$ for masked tokens). Naive implementations encounter fatal IEEE 754 floating-point pitfalls:
1. **$0 \times (-\infty) \rightarrow \text{NaN}$**: For impossible tokens, $\text{softmax}(z) = 0.0$ while $\text{log\_softmax}(z) = -\infty$. Evaluating $p \log p$ naively results in `NaN`.  
   **Our Solution**: We apply the information-theoretic limit $\lim_{p \to 0^+} p \log p = 0$:
   ```python
   safe_p_log_p = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
   entropy = -safe_p_log_p.sum().item()
   ```
2. **Precision Loss in Half-Precision**: Summing 248,320 bfloat16 numbers accumulates roundoff error.  
   **Our Solution**: Logits are explicitly cast to `float32` before computing softmax, log-softmax, and entropy sums.
3. **Perplexity Overflow**: If an impossible token has near-zero probability, $-\overline{\log P} \to \infty$. We clamp the exponent $\min(-\overline{\log P}, 100.0)$ to prevent Python `OverflowError`.

---

## 6. Special Tokens Policy

- **Prompt BOS Tokens**: Handled during tokenization (`add_special_tokens=True`) so context conditioning is identical to pretraining.
- **Response EOS Tokens**: When evaluating responses, special tokens can be optionally excluded using `exclude_special_tokens=True` (default: included). If trailing `<|im_end|>` or `</s>` tokens are present, they typically exhibit $P \approx 1.0$ and $H \approx 0.0$. Excluding them ensures the statistics reflect only semantic generated content.

---

## 7. Relationship to Hallucination Detection

| Feature | Behavior on Faithful Output | Behavior on Hallucination | Rationale |
| :--- | :--- | :--- | :--- |
| **`min_token_prob`** | High ($\ge 0.5$) | Very Low ($\le 0.05$) | Fabricated named entities, dates, or numbers diverge from training priors. |
| **`max_entropy`** | Moderate | Very High | When the model lacks grounded memory, logits spread flatly across plausible names/facts. |
| **`perplexity`** | Low ($\approx 1.0 - 2.5$) | Elevated ($\ge 5.0$) | Overall sequence surprise is significantly higher for hallucinated sentences. |
| **`entropy_std`** | Stable | Spiky | Fluctuates violently between certain function words and uncertain factual claims. |
