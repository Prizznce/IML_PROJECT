"""
Live Inference Engine for Real-Time Hallucination Detection.

Orchestrates the complete 19-feature Universal Core Detector for a single user query:
  1. Primary response generation via Qwen/Qwen3.5-0.8B (deterministic / greedy).
  2. Internal generation signal extraction (11 token probability/entropy/rank features).
  3. Behavioral self-consistency probing (k=5 stochastic responses, 5 features).
  4. Pairwise Natural Language Inference agreement (10 pairs / 20 passes, 3 features).
  5. Assembly of the canonical 19-feature vector (UNIVERSAL_CORE_FEATURES).
  6. Supervised risk scoring via Phase 14 Logistic Regression and XGBoost classifiers.

Preserves exact Phase 14 feature definitions, hyperparameters, and split protocols.
"""

import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from src.consistency.nli_agreement import (
    aggregate_example_nli_signals,
    extract_pairwise_nli_predictions,
    get_nli_label_indices,
    load_nli_model,
)
from src.consistency.self_consistency import (
    extract_self_consistency_signals,
    generate_stochastic_responses,
    load_embedding_model,
)
from src.features.build_universal_features import (
    FORBIDDEN_FEATURE_COLUMNS,
    INTERNAL_SIGNAL_FEATURES,
    NLI_AGREEMENT_FEATURES,
    SELF_CONSISTENCY_FEATURES,
    UNIVERSAL_CORE_FEATURES,
)
from src.generation.generate_signals import (
    format_model_input,
    load_model_and_tokenizer,
)
from src.retrieval.evidence_retrieval import (
    calculate_evidence_margin,
    calculate_retrieval_agreement,
    compute_cosine_similarity,
    rank_top_k,
)
from src.signals.token_signals import (
    SequenceSignals,
    TokenSignal,
    extract_raw_sequence_signals,
)
from src.training.combined_models import (
    build_logistic_regression_pipeline,
    build_xgboost_model,
    prepare_combined_feature_dataframe,
    split_dataset,
)

logger = logging.getLogger("app.inference")

# Global in-memory cache for models
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}


# ==============================================================================
# 1. Model Loading & Classifier Initialization
# ==============================================================================

def get_trained_classifiers(
    data_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    model_id: str = "Qwen/Qwen3.5-0.8B",
) -> Tuple[Any, Any]:
    """
    Load or fit the Logistic Regression and XGBoost classifiers for a specific model.

    Checks:
      1. Pre-serialized joblib model artifacts in models/{slug}/
      2. In-memory cache
      3. Fitting on data_path / get_universal_features_path(model_id)
      4. Graceful fallback to Universal Baseline classifiers if new model has not been trained yet.

    Parameters:
        data_path: Path to universal features parquet table (optional; auto-resolved if None).
        force_reload: If True, forces re-fitting from training parquet.
        model_id: Model repository ID for which classifiers were trained.

    Returns:
        Tuple of (fitted_lr_pipeline, fitted_xgb_model).
    """
    global _GLOBAL_MODEL_CACHE
    from src.utils.model_registry import get_model_slug, get_saved_model_dir, get_universal_features_path

    slug = get_model_slug(model_id)
    cache_key_lr = f"lr_pipeline_{slug}"
    cache_key_xgb = f"xgb_model_{slug}"

    if not force_reload and cache_key_lr in _GLOBAL_MODEL_CACHE and cache_key_xgb in _GLOBAL_MODEL_CACHE:
        return _GLOBAL_MODEL_CACHE[cache_key_lr], _GLOBAL_MODEL_CACHE[cache_key_xgb]

    # Check for pre-serialized model artifacts on disk
    model_dir = get_saved_model_dir(model_id)
    lr_saved = model_dir / "lr_pipeline.joblib"
    xgb_saved = model_dir / "xgb_model.joblib"
    if not force_reload and lr_saved.exists() and xgb_saved.exists():
        try:
            import joblib
            logger.info(f"Loading serialized classifiers from '{model_dir}' for '{model_id}'...")
            lr_pipe = joblib.load(lr_saved)
            xgb_mdl = joblib.load(xgb_saved)
            _GLOBAL_MODEL_CACHE[cache_key_lr] = lr_pipe
            _GLOBAL_MODEL_CACHE[cache_key_xgb] = xgb_mdl
            return lr_pipe, xgb_mdl
        except Exception as e:
            logger.warning(f"Could not load pre-saved classifiers from {model_dir}: {e}")

    # Determine dataset path
    resolved_path = Path(data_path) if data_path is not None else get_universal_features_path(model_id)

    if not resolved_path.exists():
        if model_id != "Qwen/Qwen3.5-0.8B":
            logger.warning(
                f"Universal features for '{model_id}' not found at '{resolved_path}'. "
                "Falling back to baseline classifiers until retrained."
            )
            return get_trained_classifiers(model_id="Qwen/Qwen3.5-0.8B", force_reload=force_reload)
        raise FileNotFoundError(f"Universal feature dataset not found at '{resolved_path}'")

    df = pd.read_parquet(resolved_path)
    X, y, meta = prepare_combined_feature_dataframe(df)

    # Use stratified split (80/20)
    X_train, _, y_train, _, _, _ = split_dataset(
        X, y, meta, test_size=0.2, random_state=42
    )

    logger.info(f"Fitting Logistic Regression pipeline on {len(X_train)} training rows for '{model_id}'...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=42)
    lr_pipeline.fit(X_train, y_train)

    logger.info(f"Fitting XGBoost model on {len(X_train)} training rows for '{model_id}'...")
    xgb_model = build_xgboost_model(random_state=42)
    xgb_model.fit(X_train, y_train)

    # Serialize to model directory for instant subsequent loading
    try:
        import joblib
        joblib.dump(lr_pipeline, lr_saved)
        joblib.dump(xgb_model, xgb_saved)
        logger.info(f"Saved fitted classifiers to '{model_dir}'")
    except Exception as e:
        logger.warning(f"Could not cache fitted models to disk: {e}")

    _GLOBAL_MODEL_CACHE[cache_key_lr] = lr_pipeline
    _GLOBAL_MODEL_CACHE[cache_key_xgb] = xgb_model

    # Backward compatibility keys for Qwen
    if "qwen" in slug:
        _GLOBAL_MODEL_CACHE["lr_pipeline"] = lr_pipeline
        _GLOBAL_MODEL_CACHE["xgb_model"] = xgb_model

    return lr_pipeline, xgb_model


def load_inference_models(
    device: Optional[str] = None,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    nli_model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load generative LLM, sentence embedding model, NLI model, and trained tabular classifiers.

    Parameters:
        device: Target execution device ('cuda:0' or 'cpu'). Defaults to cuda:0 when CUDA is available.
        model_id: Causal LM repository ID.
        embedding_model_name: SentenceTransformer repository ID.
        nli_model_name: CrossEncoder NLI repository ID.
        token: Hugging Face authentication token for gated repositories (e.g. Llama 3.2).

    Returns:
        Dict containing loaded model instances and metadata.
    """
    global _GLOBAL_MODEL_CACHE
    from src.utils.model_registry import get_hf_token, get_model_slug
    auth_token = get_hf_token(token)
    slug = get_model_slug(model_id)

    if device is None:
        target = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        target = str(device)
    actual_device = target if torch.cuda.is_available() and "cuda" in target else "cpu"

    causal_key = f"causal_model_{slug}"
    tok_key = f"tokenizer_{slug}"

    if causal_key not in _GLOBAL_MODEL_CACHE:
        logger.info(f"Loading causal model '{model_id}' on {actual_device}...")
        causal_model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=actual_device, token=auth_token)
        _GLOBAL_MODEL_CACHE[causal_key] = causal_model
        _GLOBAL_MODEL_CACHE[tok_key] = tokenizer
        _GLOBAL_MODEL_CACHE["causal_model"] = causal_model
        _GLOBAL_MODEL_CACHE["tokenizer"] = tokenizer
    else:
        _GLOBAL_MODEL_CACHE["causal_model"] = _GLOBAL_MODEL_CACHE[causal_key]
        _GLOBAL_MODEL_CACHE["tokenizer"] = _GLOBAL_MODEL_CACHE[tok_key]
        try:
            curr_dev = str(next(_GLOBAL_MODEL_CACHE["causal_model"].parameters()).device)
            if ("cuda" in actual_device and "cuda" not in curr_dev) or ("cpu" in actual_device and "cuda" in curr_dev):
                logger.info(f"Relocating cached causal model from {curr_dev} to {actual_device}...")
                _GLOBAL_MODEL_CACHE["causal_model"] = _GLOBAL_MODEL_CACHE["causal_model"].to(actual_device)
        except Exception:
            pass

    if "embedding_model" not in _GLOBAL_MODEL_CACHE:
        logger.info(f"Loading embedding model '{embedding_model_name}' on {actual_device}...")
        embedding_model = load_embedding_model(model_name=embedding_model_name, device=actual_device)
        _GLOBAL_MODEL_CACHE["embedding_model"] = embedding_model
    else:
        try:
            if hasattr(_GLOBAL_MODEL_CACHE["embedding_model"], "to"):
                _GLOBAL_MODEL_CACHE["embedding_model"] = _GLOBAL_MODEL_CACHE["embedding_model"].to(actual_device)
        except Exception:
            pass

    if "nli_model" not in _GLOBAL_MODEL_CACHE:
        logger.info(f"Loading NLI model '{nli_model_name}' on {actual_device}...")
        nli_res = load_nli_model(model_name=nli_model_name, device=actual_device)
        if isinstance(nli_res, tuple):
            nli_model, label_indices = nli_res
        else:
            nli_model = nli_res
            label_indices = get_nli_label_indices(nli_model)
        _GLOBAL_MODEL_CACHE["nli_model"] = nli_model
        _GLOBAL_MODEL_CACHE["nli_label_indices"] = label_indices
    else:
        try:
            curr_nli_dev = str(getattr(_GLOBAL_MODEL_CACHE["nli_model"], "device", ""))
            if ("cuda" in actual_device and "cuda" not in curr_nli_dev) or ("cpu" in actual_device and "cuda" in curr_nli_dev):
                logger.info(f"Reloading NLI model on {actual_device}...")
                nli_res = load_nli_model(model_name=nli_model_name, device=actual_device)
                if isinstance(nli_res, tuple):
                    _GLOBAL_MODEL_CACHE["nli_model"], _GLOBAL_MODEL_CACHE["nli_label_indices"] = nli_res
                else:
                    _GLOBAL_MODEL_CACHE["nli_model"] = nli_res
                    _GLOBAL_MODEL_CACHE["nli_label_indices"] = get_nli_label_indices(nli_res)
        except Exception:
            pass

    lr_pipeline, xgb_model = get_trained_classifiers(model_id=model_id)

    return {
        "causal_model": _GLOBAL_MODEL_CACHE["causal_model"],
        "tokenizer": _GLOBAL_MODEL_CACHE["tokenizer"],
        "embedding_model": _GLOBAL_MODEL_CACHE["embedding_model"],
        "nli_model": _GLOBAL_MODEL_CACHE["nli_model"],
        "nli_label_indices": _GLOBAL_MODEL_CACHE["nli_label_indices"],
        "lr_pipeline": lr_pipeline,
        "xgb_model": xgb_model,
        "device": actual_device,
        "model_id": model_id,
        "model_slug": slug,
    }



# ==============================================================================
# 2. Generation and Feature Extraction Components
# ==============================================================================

def generate_primary_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    context: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 128,
    system_instruction: str = "Answer the question directly, factually, and concisely. Do not provide preamble or internal thinking.",
) -> Tuple[str, str]:
    """
    Generate the primary evaluated response deterministically using greedy decoding.

    Parameters:
        model: Loaded CausalLM.
        tokenizer: Loaded AutoTokenizer.
        prompt: User question or instruction.
        context: Optional background text.
        device: Execution device (optional; defaults to model device).
        max_new_tokens: Maximum tokens to generate.
        system_instruction: Factual instruction prompt.

    Returns:
        Tuple of (clean_response_text, formatted_model_input).
    """
    model_input = format_model_input(
        prompt=prompt,
        context=context or "",
        source_dataset="",
        use_chat_template=True,
        tokenizer=tokenizer,
        system_instruction=system_instruction,
    )

    # Determine target device consistent with model parameters
    target_dev = device
    try:
        model_param_dev = next(model.parameters()).device
        target_dev = model_param_dev
    except (StopIteration, AttributeError):
        if target_dev is None:
            target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    enc = tokenizer(model_input, return_tensors="pt")
    input_ids = enc.input_ids.to(target_dev)
    attention_mask = enc.attention_mask.to(target_dev) if hasattr(enc, "attention_mask") and enc.attention_mask is not None else None
    input_len = input_ids.shape[1]

    gen_kwargs = {
        "input_ids": input_ids,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        gen_out = model.generate(**gen_kwargs)

    gen_tokens = gen_out[0][input_len:]
    raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return raw_response, model_input


def extract_internal_signals(
    model: Any,
    tokenizer: Any,
    model_input: str,
    response: str,
    device: Optional[str] = None,
) -> Tuple[Dict[str, float], SequenceSignals]:
    """
    Extract the 11 white-box internal generation signals from an unadulterated forward pass.

    Parameters:
        model: CausalLM instance.
        tokenizer: AutoTokenizer instance.
        model_input: Full conditioning prompt.
        response: Generated response string.
        device: Compute device (optional; dynamically inferred from model).

    Returns:
        Tuple of (11-feature dict, raw SequenceSignals dataclass).
    """
    target_dev = device
    try:
        model_param_dev = next(model.parameters()).device
        target_dev = model_param_dev
    except (StopIteration, AttributeError):
        if target_dev is None:
            target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    signals: SequenceSignals = extract_raw_sequence_signals(
        model=model,
        tokenizer=tokenizer,
        prompt=model_input,
        response=response,
        device=target_dev,
    )
    t1 = time.perf_counter()

    # Rank aggregation
    ranks = [t.rank for t in signals.tokens] if signals.tokens else []
    mean_token_rank = float(sum(ranks) / len(ranks)) if ranks else 1.0
    max_token_rank = int(max(ranks)) if ranks else 1
    rank_std = float(statistics.stdev(ranks)) if len(ranks) > 1 else 0.0

    # Numerical transforms matching Phase 13/14
    log_ppl = float(np.log(max(signals.perplexity, 1e-12)))
    log_mean_rank = float(np.log1p(max(mean_token_rank, 0.0)))
    log_max_rank = float(np.log1p(max(float(max_token_rank), 0.0)))
    log_r_std = float(np.log1p(max(rank_std, 0.0)))

    features = {
        "min_log_prob": float(signals.min_log_prob),
        "mean_log_prob": float(signals.mean_log_prob),
        "mean_token_prob": float(signals.mean_token_prob),
        "token_prob_std": float(signals.token_prob_std),
        "mean_entropy": float(signals.mean_entropy),
        "max_entropy": float(signals.max_entropy),
        "entropy_std": float(signals.entropy_std),
        "log_perplexity": log_ppl,
        "log_mean_token_rank": log_mean_rank,
        "log_max_token_rank": log_max_rank,
        "log_rank_std": log_r_std,
    }
    t2 = time.perf_counter()

    try:
        signals._t_logit_extraction_ms = (t1 - t0) * 1000.0
        signals._t_signal_computation_ms = (t2 - t1) * 1000.0
    except Exception:
        pass

    return features, signals


def run_self_consistency_probe(
    model: Any,
    tokenizer: Any,
    embedding_model: Any,
    prompt: str,
    context: Optional[str] = None,
    device: Optional[str] = None,
    num_generations: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Generate k=5 stochastic responses and compute 5 self-consistency features.

    Parameters:
        model: CausalLM.
        tokenizer: AutoTokenizer.
        embedding_model: SentenceTransformer.
        prompt: User question.
        context: Optional reference context.
        device: Compute device (optional; aligns with model device).
        num_generations: Number of stochastic paths (default: 5).
        temperature: Sampling temperature (default: 0.7).
        top_p: Nucleus threshold (default: 0.9).
        max_new_tokens: Token cap.

    Returns:
        Tuple of (list_of_responses, 5-feature dict).
    """
    # Align target device with causal model parameters
    target_dev = device
    try:
        model_param_dev = str(next(model.parameters()).device)
        target_dev = model_param_dev
    except (StopIteration, AttributeError):
        if target_dev is None:
            target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    responses, _, _ = generate_stochastic_responses(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        context=context or "",
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        device=target_dev,
    )
    t1 = time.perf_counter()


    emb_dev = device
    try:
        if hasattr(embedding_model, "device"):
            emb_dev = str(embedding_model.device)
    except Exception:
        pass

    sc_metrics = extract_self_consistency_signals(
        responses=responses,
        embedding_model=embedding_model,
        device=emb_dev,
    )
    t2 = time.perf_counter()

    features = {
        "exact_match_agreement": float(sc_metrics["exact_match_agreement"]),
        "mean_pairwise_similarity": float(sc_metrics["mean_pairwise_similarity"]),
        "min_pairwise_similarity": float(sc_metrics["min_pairwise_similarity"]),
        "max_pairwise_similarity": float(sc_metrics["max_pairwise_similarity"]),
        "pairwise_similarity_std": float(sc_metrics["pairwise_similarity_std"]),
    }
    features["_sc_generation_ms"] = (t1 - t0) * 1000.0
    features["_sc_similarity_ms"] = (t2 - t1) * 1000.0

    return responses, features



def run_nli_probe(
    nli_model: Any,
    responses: List[str],
    label_indices: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Compute pairwise NLI agreement across candidate responses (10 pairs, 20 evaluations).

    Parameters:
        nli_model: CrossEncoder instance or tuple of (CrossEncoder, label_indices).
        responses: List of candidate response strings (k=5).
        label_indices: Dict mapping 'entailment', 'contradiction', 'neutral' to class indices.

    Returns:
        Dict of 3 canonical NLI features.
    """
    if isinstance(nli_model, (tuple, list)):
        if len(nli_model) == 2 and isinstance(nli_model[1], dict):
            if label_indices is None:
                label_indices = nli_model[1]
            nli_model = nli_model[0]
        else:
            nli_model = nli_model[0]

    if label_indices is None:
        label_indices = get_nli_label_indices(nli_model)

    pair_records = extract_pairwise_nli_predictions(
        responses=responses,
        model=nli_model,
        label_indices=label_indices,
    )

    agg = aggregate_example_nli_signals(pair_records, num_generations=len(responses))

    return {
        "mean_pairwise_entailment": float(agg["mean_pairwise_entailment"]),
        "mean_pairwise_contradiction": float(agg["mean_pairwise_contradiction"]),
        "nli_disagreement": float(agg["nli_disagreement"]),
    }


# ==============================================================================
# 3. Feature Assembly and Risk Scoring
# ==============================================================================

def assemble_feature_vector(
    internal_features: Dict[str, float],
    sc_features: Dict[str, float],
    nli_features: Dict[str, float],
) -> Dict[str, float]:
    """
    Assemble and validate the exact 19-feature vector for the Universal Core Detector.

    Enforces canonical ordering matching UNIVERSAL_CORE_FEATURES and validates
    absence of NaNs, Infs, and forbidden columns.

    Parameters:
        internal_features: 11 internal signal features.
        sc_features: 5 self-consistency features.
        nli_features: 3 NLI agreement features.

    Returns:
        Dict mapping feature name to float in exact canonical order.
    """
    combined: Dict[str, float] = {}
    combined.update(internal_features)
    combined.update(sc_features)
    combined.update(nli_features)

    # Check for forbidden columns
    forbidden_present = set(combined.keys()).intersection(FORBIDDEN_FEATURE_COLUMNS)
    if forbidden_present:
        raise ValueError(f"Forbidden columns detected in feature vector: {forbidden_present}")

    ordered_vector: Dict[str, float] = {}
    for feat in UNIVERSAL_CORE_FEATURES:
        if feat not in combined:
            raise KeyError(f"Missing required universal feature: '{feat}'")
        val = float(combined[feat])
        if not np.isfinite(val):
            raise ValueError(f"Non-finite value detected for feature '{feat}': {val}")
        ordered_vector[feat] = val

    if len(ordered_vector) != 19:
        raise ValueError(f"Expected exactly 19 features, got {len(ordered_vector)}")

    return ordered_vector


def format_risk_level(probability: float) -> str:
    """
    Map estimated hallucination probability to a user-facing visualization band.

    NOTE: This is a presentation convention for the demonstration interface,
    not a clinically or operationally validated safety threshold.
    """
    p = float(probability)
    if p < 0.35:
        return "Low Risk"
    elif p < 0.65:
        return "Moderate Risk"
    else:
        return "High Hallucination Risk"


def get_token_confidence_band(
    probability: float,
    entropy: Optional[float] = None,
    high_threshold: float = 0.65,
    low_threshold: float = 0.30,
) -> str:
    """
    Categorize token generation uncertainty into deterministic presentation bands.

    Parameters:
        probability: Token likelihood P(y_t | x, y_<t) in [0.0, 1.0].
        entropy: Predictive Shannon entropy at generation step (optional).
        high_threshold: Minimum probability for 'High confidence' (default: 0.65).
        low_threshold: Maximum probability for 'Low confidence' (default: 0.30).

    Returns:
        One of 'High confidence', 'Medium confidence', or 'Low confidence'.
    """
    p = float(probability)
    if p >= high_threshold and (entropy is None or entropy < 1.5):
        return "High confidence"
    elif p < low_threshold or (entropy is not None and entropy > 2.5):
        return "Low confidence"
    else:
        return "Medium confidence"


def format_token_details(
    tokens: Sequence[Any],
    high_threshold: float = 0.65,
    low_threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """
    Convert raw sequence token signals into sanitized presentation records with confidence bands.

    Parameters:
        tokens: Sequence of TokenSignal objects or token dictionaries.
        high_threshold: Probability threshold for high confidence.
        low_threshold: Probability threshold for low confidence.

    Returns:
        Structured list of token detail dictionaries.
    """
    details: List[Dict[str, Any]] = []
    for idx, t in enumerate(tokens):
        if hasattr(t, "to_dict"):
            d = t.to_dict()
        elif isinstance(t, dict):
            d = t
        else:
            d = {
                "step": getattr(t, "step", idx + 1),
                "token_id": getattr(t, "token_id", 0),
                "token_text": getattr(t, "token_text", str(t)),
                "probability": getattr(t, "probability", 0.0),
                "log_probability": getattr(t, "log_probability", 0.0),
                "entropy": getattr(t, "entropy", 0.0),
                "rank": getattr(t, "rank", 1),
            }

        prob = float(d.get("probability", 0.0))
        entropy = float(d.get("entropy", 0.0)) if "entropy" in d else None
        band = get_token_confidence_band(prob, entropy, high_threshold, low_threshold)

        details.append({
            "step": int(d.get("step", idx + 1)),
            "token": str(d.get("token_text", "")),
            "probability": prob,
            "log_prob": float(d.get("log_probability", d.get("log_prob", 0.0))),
            "entropy": float(entropy if entropy is not None else 0.0),
            "rank": int(d.get("rank", 1)),
            "band": band,
        })
    return details


def compute_feature_contributions(
    feature_vector: Dict[str, float],
    xgb_model: Any,
    lr_pipeline: Optional[Any] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Compute mathematically authentic model-level feature contributions for the prediction.

    Attempts native Tree SHAP contributions via XGBoost booster.predict(pred_contribs=True).
    If unavailable, falls back to Logistic Regression standardized coefficient contributions,
    or transparent signal-profile deviations.

    Parameters:
        feature_vector: Exact 19-feature dictionary matching UNIVERSAL_CORE_FEATURES.
        xgb_model: Fitted XGBoost model.
        lr_pipeline: Optional fitted Logistic Regression pipeline.
        top_k: Number of top contributing features to return (default: 6).

    Returns:
        Sorted list of top contribution dictionaries.
    """
    import xgboost as xgb

    feature_display_names = {
        "min_log_prob": "Minimum Token Log-Probability (Bottleneck)",
        "mean_log_prob": "Mean Token Log-Probability",
        "mean_token_prob": "Mean Token Probability",
        "token_prob_std": "Token Probability Dispersion (Std)",
        "mean_entropy": "Mean Predictive Entropy",
        "max_entropy": "Peak Predictive Entropy",
        "entropy_std": "Predictive Entropy Dispersion",
        "log_perplexity": "Log Sequence Perplexity",
        "log_mean_token_rank": "Log Mean Token Rank",
        "log_max_token_rank": "Log Max Token Rank",
        "log_rank_std": "Log Token Rank Dispersion",
        "exact_match_agreement": "Exact Match Agreement (k=5)",
        "mean_pairwise_similarity": "Mean Semantic Embedding Similarity",
        "min_pairwise_similarity": "Min Pairwise Similarity (Divergence Floor)",
        "max_pairwise_similarity": "Max Pairwise Similarity",
        "pairwise_similarity_std": "Semantic Similarity Dispersion",
        "mean_pairwise_entailment": "NLI Entailment Agreement",
        "mean_pairwise_contradiction": "NLI Contradiction Frequency",
        "nli_disagreement": "Composite NLI Disagreement Index",
    }

    X = pd.DataFrame([[feature_vector[f] for f in UNIVERSAL_CORE_FEATURES]], columns=UNIVERSAL_CORE_FEATURES)

    contributions_dict: Dict[str, float] = {}
    method_used = "Tree SHAP (XGBoost native pred_contribs)"

    try:
        if hasattr(xgb_model, "get_booster"):
            dmat = xgb.DMatrix(X)
            raw_contribs = xgb_model.get_booster().predict(dmat, pred_contribs=True)[0]
            for idx, feat in enumerate(UNIVERSAL_CORE_FEATURES):
                contributions_dict[feat] = float(raw_contribs[idx])
        else:
            raise AttributeError("xgb_model has no get_booster method")
    except Exception as exc:
        logger.warning(f"Native XGBoost SHAP contribution calculation failed: {exc}. Trying LR fallback.")
        if lr_pipeline is not None and hasattr(lr_pipeline, "named_steps"):
            try:
                scaler = lr_pipeline.named_steps.get("scaler")
                clf = lr_pipeline.named_steps.get("classifier")
                if scaler is not None and clf is not None:
                    z = (X.values - scaler.mean_) / scaler.scale_
                    lr_contribs = (z * clf.coef_[0])[0]
                    for idx, feat in enumerate(UNIVERSAL_CORE_FEATURES):
                        contributions_dict[feat] = float(lr_contribs[idx])
                    method_used = "Standardized Coefficient Contribution (Logistic Regression)"
            except Exception as e:
                logger.warning(f"LR contribution fallback failed: {e}. Falling back to Signal Profile.")

        if not contributions_dict:
            method_used = "Signal Profile (Normalized deviation)"
            for feat in UNIVERSAL_CORE_FEATURES:
                contributions_dict[feat] = float(feature_vector[feat])

    records = []
    for feat in UNIVERSAL_CORE_FEATURES:
        contrib = contributions_dict.get(feat, 0.0)
        val = float(feature_vector[feat])
        direction = "increases risk" if contrib > 0 else "decreases risk"
        records.append({
            "feature": feat,
            "display_name": feature_display_names.get(feat, feat),
            "value": val,
            "contribution": contrib,
            "magnitude": abs(contrib),
            "direction": direction,
            "method": method_used,
        })

    records.sort(key=lambda r: r["magnitude"], reverse=True)
    return records[:top_k]


def compute_reference_evidence_agreement(
    query_text: str,
    response_text: str,
    context_text: Optional[str] = None,
    embedding_model: Optional[Any] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Compute live evidence agreement metrics against user-supplied reference context.

    If reference context is provided, splits into passage chunks and calculates
    cosine similarity against query and response using all-MiniLM-L6-v2, and computes
    dual groundedness via calculate_retrieval_agreement().

    If context is omitted, returns clean informative status noting the offline benchmark setup.
    """
    clean_ctx = str(context_text).strip() if context_text is not None and str(context_text).strip() else None

    if not clean_ctx:
        return {
            "has_reference_context": False,
            "top1_evidence_similarity": 0.0,
            "response_top1_similarity": 0.0,
            "evidence_margin": 0.0,
            "retrieval_agreement": 0.0,
            "retrieved_evidence": [],
            "offline_benchmark_note": (
                "Phase 19 retrieval experiment evaluated 2,500 benchmark instances with curated evidence "
                "(HaluEval reference passages and FEVER Wikipedia pointers), achieving ROC-AUC 0.9184 on the holdout split. "
                "For live queries in this demo, enter a reference context above to evaluate groundedness against evidence."
            ),
        }

    # Split context into paragraphs or sentence chunks
    raw_chunks = [c.strip() for c in clean_ctx.split("\n") if c.strip()]
    if not raw_chunks:
        raw_chunks = [clean_ctx]

    seen = set()
    chunks: List[str] = []
    for c in raw_chunks:
        if c not in seen:
            seen.add(c)
            chunks.append(c)

    if embedding_model is None:
        actual_device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        embedding_model = load_embedding_model(device=actual_device)

    # Encode query, response, and chunks
    all_texts = [query_text, response_text] + chunks
    embeddings = embedding_model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = embeddings[0]
    r_emb = embeddings[1]
    c_embs = embeddings[2:]

    q_sims = compute_cosine_similarity(q_emb, c_embs)
    top_indices, top_q_sims = rank_top_k(q_sims, k=min(3, len(chunks)))

    top1_q = float(top_q_sims[0]) if top_q_sims else 0.0
    margin = calculate_evidence_margin(top_q_sims)

    retrieved_cards: List[Dict[str, Any]] = []
    r_sims_top: List[float] = []

    for rank, c_idx in enumerate(top_indices, 1):
        chunk_emb = c_embs[c_idx : c_idx + 1]
        r_sim = float(compute_cosine_similarity(r_emb, chunk_emb)[0])
        r_sims_top.append(r_sim)
        retrieved_cards.append({
            "rank": rank,
            "text": chunks[c_idx],
            "query_similarity": float(top_q_sims[rank - 1]),
            "response_similarity": r_sim,
        })

    top1_r = r_sims_top[0] if r_sims_top else 0.0
    agreement = calculate_retrieval_agreement(top1_q, top1_r)

    return {
        "has_reference_context": True,
        "top1_evidence_similarity": top1_q,
        "response_top1_similarity": top1_r,
        "evidence_margin": margin,
        "retrieval_agreement": agreement,
        "retrieved_evidence": retrieved_cards,
        "offline_benchmark_note": (
            "Phase 19 retrieval-augmented variant achieved holdout ROC-AUC 0.9184 on the 2,500-instance "
            "HaluEval+FEVER benchmark. Live evidence groundedness is computed above using all-MiniLM-L6-v2."
        ),
    }


def predict_hallucination_risk(
    feature_vector: Dict[str, float],
    lr_pipeline: Optional[Any] = None,
    xgb_model: Optional[Any] = None,
) -> Tuple[float, float, str]:
    """
    Score the 19-feature vector using the Phase 14 Logistic Regression and XGBoost classifiers.

    Parameters:
        feature_vector: Exact 19-feature dictionary.
        lr_pipeline: Fitted Logistic Regression pipeline, or tuple of (lr_pipeline, xgb_model).
        xgb_model: Fitted XGBoost model (optional if unpacked from lr_pipeline).

    Returns:
        Tuple of (xgb_probability, lr_probability, risk_level).
    """
    # Unpack if a tuple of classifiers was passed as the second argument
    if isinstance(lr_pipeline, (tuple, list)):
        if len(lr_pipeline) == 2:
            lr_pipeline, xgb_model = lr_pipeline[0], lr_pipeline[1]
        else:
            raise ValueError(f"Expected 2-tuple (lr_pipeline, xgb_model), got {len(lr_pipeline)} items.")

    # Load classifiers from cache/disk if not provided
    if lr_pipeline is None or xgb_model is None:
        cached_lr, cached_xgb = get_trained_classifiers()
        if lr_pipeline is None and xgb_model is None:
            lr_pipeline, xgb_model = cached_lr, cached_xgb
        elif lr_pipeline is None:
            lr_pipeline = cached_lr
        elif xgb_model is None:
            # If a single classifier was passed in lr_pipeline, determine its type
            if isinstance(lr_pipeline, XGBClassifier) or type(lr_pipeline).__name__ == "XGBClassifier":
                xgb_model = lr_pipeline
                lr_pipeline = cached_lr
            else:
                xgb_model = cached_xgb

    # Create single-row DataFrame with exact canonical feature order
    X = pd.DataFrame([[feature_vector[f] for f in UNIVERSAL_CORE_FEATURES]], columns=UNIVERSAL_CORE_FEATURES)

    # Compute LR probability
    if hasattr(lr_pipeline, "predict_proba"):
        lr_prob = float(lr_pipeline.predict_proba(X)[0, 1])
    elif hasattr(lr_pipeline, "predict"):
        lr_prob = float(lr_pipeline.predict(X)[0])
    else:
        raise AttributeError(f"Classifier {type(lr_pipeline)} has neither 'predict_proba' nor 'predict'.")

    # Compute XGB probability
    if hasattr(xgb_model, "predict_proba"):
        xgb_prob = float(xgb_model.predict_proba(X)[0, 1])
    elif hasattr(xgb_model, "predict"):
        xgb_prob = float(xgb_model.predict(X)[0])
    else:
        raise AttributeError(f"Classifier {type(xgb_model)} has neither 'predict_proba' nor 'predict'.")

    # Clamp to [0, 1] range defensively
    lr_prob = max(0.0, min(1.0, lr_prob))
    xgb_prob = max(0.0, min(1.0, xgb_prob))

    risk_level = format_risk_level(xgb_prob)

    return xgb_prob, lr_prob, risk_level


# ==============================================================================
# 4. Master Inference Orchestrator
# ==============================================================================

def run_live_inference(
    prompt: str,
    context: Optional[str] = None,
    device: Optional[str] = None,
    causal_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
    nli_model: Optional[Any] = None,
    nli_label_indices: Optional[Dict[str, int]] = None,
    lr_pipeline: Optional[Any] = None,
    xgb_model: Optional[Any] = None,
    is_evidence_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Execute full end-to-end hallucination risk estimation for a single prompt.
    Includes comprehensive 15-stage timing instrumentation and selective evidence execution.

    Parameters:
        prompt: User query (must be non-empty string).
        context: Optional reference context passage.
        device: Target execution device.
        causal_model, tokenizer, embedding_model, nli_model, nli_label_indices,
        lr_pipeline, xgb_model: Optional pre-loaded models to bypass repeated loading.
        is_evidence_mode: Explicit flag controlling whether evidence analysis runs. If None,
                          runs if clean context is present (for backwards test compatibility).

    Returns:
        Structured result dictionary containing generated responses, signal breakdown,
        the 19-feature vector, predicted hallucination risks, and stage_latencies_ms.
    """
    clean_prompt = str(prompt).strip()
    if not clean_prompt:
        raise ValueError("Prompt must be a non-empty string.")

    clean_context = str(context).strip() if context is not None and str(context).strip() else None

    # Defensive unpacking if models passed as tuples
    if isinstance(lr_pipeline, (tuple, list)) and len(lr_pipeline) == 2:
        lr_pipeline, xgb_model = lr_pipeline[0], lr_pipeline[1]

    if isinstance(nli_model, (tuple, list)):
        if len(nli_model) == 2 and isinstance(nli_model[1], dict):
            if nli_label_indices is None:
                nli_label_indices = nli_model[1]
            nli_model = nli_model[0]
        else:
            nli_model = nli_model[0]

    t_stages: Dict[str, float] = {}

    # Stage 1: model loading (loading or resolving from memory cache)
    t0_s1 = time.perf_counter()
    if (
        causal_model is None
        or tokenizer is None
        or embedding_model is None
        or nli_model is None
        or lr_pipeline is None
        or xgb_model is None
    ):
        models = load_inference_models(device=device)
        causal_model = causal_model or models["causal_model"]
        tokenizer = tokenizer or models["tokenizer"]
        embedding_model = embedding_model or models["embedding_model"]
        nli_model = nli_model or models["nli_model"]
        nli_label_indices = nli_label_indices or models["nli_label_indices"]
        lr_pipeline = lr_pipeline or models["lr_pipeline"]
        xgb_model = xgb_model or models["xgb_model"]
        actual_device = models["device"]
    else:
        # Pre-loaded models were passed in. Detect actual device from causal_model parameters if available
        if causal_model is not None:
            try:
                actual_device = str(next(causal_model.parameters()).device)
            except (StopIteration, AttributeError):
                actual_device = device if (device is not None and torch.cuda.is_available() and "cuda" in device) else ("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            actual_device = device if (device is not None and torch.cuda.is_available() and "cuda" in device) else ("cuda:0" if torch.cuda.is_available() else "cpu")

        if nli_label_indices is None:
            nli_label_indices = get_nli_label_indices(nli_model)
    t_stages["1. model loading"] = (time.perf_counter() - t0_s1) * 1000.0

    # Stage 2: response generation
    t0_s2 = time.perf_counter()
    primary_response, model_input = generate_primary_response(
        model=causal_model,
        tokenizer=tokenizer,
        prompt=clean_prompt,
        context=clean_context,
        device=actual_device,
    )
    t_stages["2. response generation"] = (time.perf_counter() - t0_s2) * 1000.0

    # Stage 3: token/logit extraction & Stage 4: internal signal computation
    t0_s3_4 = time.perf_counter()
    internal_features, seq_signals = extract_internal_signals(
        model=causal_model,
        tokenizer=tokenizer,
        model_input=model_input,
        response=primary_response,
        device=actual_device,
    )
    t_int_total = (time.perf_counter() - t0_s3_4) * 1000.0
    t_logit = getattr(seq_signals, "_t_logit_extraction_ms", None)
    t_sig = getattr(seq_signals, "_t_signal_computation_ms", None)
    t_stages["3. token/logit extraction"] = t_logit if t_logit is not None else t_int_total * 0.95
    t_stages["4. internal signal computation"] = t_sig if t_sig is not None else t_int_total * 0.05

    # Stage 5: self-consistency generation, Stage 6: embedding model loading, Stage 7: self-consistency similarity
    t0_s5_7 = time.perf_counter()
    sc_responses, sc_features = run_self_consistency_probe(
        model=causal_model,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        prompt=clean_prompt,
        context=clean_context,
        device=actual_device,
    )
    t_sc_total = (time.perf_counter() - t0_s5_7) * 1000.0
    sc_gen_ms = sc_features.pop("_sc_generation_ms", None)
    sc_sim_ms = sc_features.pop("_sc_similarity_ms", None)
    t_stages["5. self-consistency generation"] = sc_gen_ms if sc_gen_ms is not None else t_sc_total * 0.99
    t_stages["6. embedding model loading"] = 0.0  # Already loaded / verified in Stage 1
    t_stages["7. self-consistency similarity"] = sc_sim_ms if sc_sim_ms is not None else t_sc_total * 0.01

    # Stage 8: NLI model loading & Stage 9: NLI inference
    t_stages["8. NLI model loading"] = 0.0  # Already loaded / verified in Stage 1
    t0_s9 = time.perf_counter()
    nli_features = run_nli_probe(
        nli_model=nli_model,
        responses=sc_responses,
        label_indices=nli_label_indices,
    )
    t_stages["9. NLI inference"] = (time.perf_counter() - t0_s9) * 1000.0

    # Stage 10: feature vector assembly
    t0_s10 = time.perf_counter()
    feature_vector = assemble_feature_vector(
        internal_features=internal_features,
        sc_features=sc_features,
        nli_features=nli_features,
    )
    t_stages["10. feature vector assembly"] = (time.perf_counter() - t0_s10) * 1000.0

    # Stage 11: XGBoost prediction & Stage 12: Logistic Regression prediction
    X_single = pd.DataFrame([[feature_vector[f] for f in UNIVERSAL_CORE_FEATURES]], columns=UNIVERSAL_CORE_FEATURES)

    # 11. XGBoost prediction
    t0_s11 = time.perf_counter()
    if hasattr(xgb_model, "predict_proba"):
        xgb_prob = float(xgb_model.predict_proba(X_single)[0, 1])
    elif hasattr(xgb_model, "predict"):
        xgb_prob = float(xgb_model.predict(X_single)[0])
    else:
        raise AttributeError(f"Classifier {type(xgb_model)} has neither 'predict_proba' nor 'predict'.")
    xgb_prob = max(0.0, min(1.0, xgb_prob))
    t_stages["11. XGBoost prediction"] = (time.perf_counter() - t0_s11) * 1000.0

    # 12. Logistic Regression prediction
    t0_s12 = time.perf_counter()
    if hasattr(lr_pipeline, "predict_proba"):
        lr_prob = float(lr_pipeline.predict_proba(X_single)[0, 1])
    elif hasattr(lr_pipeline, "predict"):
        lr_prob = float(lr_pipeline.predict(X_single)[0])
    else:
        raise AttributeError(f"Classifier {type(lr_pipeline)} has neither 'predict_proba' nor 'predict'.")
    lr_prob = max(0.0, min(1.0, lr_prob))
    risk_level = format_risk_level(xgb_prob)
    t_stages["12. Logistic Regression prediction"] = (time.perf_counter() - t0_s12) * 1000.0

    # Stage 13: SHAP/XGBoost feature contributions
    t0_s13 = time.perf_counter()
    feature_contributions = compute_feature_contributions(
        feature_vector=feature_vector,
        xgb_model=xgb_model,
        lr_pipeline=lr_pipeline,
        top_k=6,
    )
    t_stages["13. SHAP/XGBoost feature contributions"] = (time.perf_counter() - t0_s13) * 1000.0

    # Stage 14: evidence analysis
    t0_s14 = time.perf_counter()
    if is_evidence_mode is None:
        should_run_evidence = clean_context is not None
    else:
        should_run_evidence = bool(is_evidence_mode) and (clean_context is not None)

    if should_run_evidence:
        evidence_analysis = compute_reference_evidence_agreement(
            query_text=clean_prompt,
            response_text=primary_response,
            context_text=clean_context,
            embedding_model=embedding_model,
            device=actual_device,
        )
    else:
        evidence_analysis = {
            "has_reference_context": False,
            "top1_evidence_similarity": 0.0,
            "response_top1_similarity": 0.0,
            "evidence_margin": 0.0,
            "retrieval_agreement": 0.0,
            "retrieved_evidence": [],
            "offline_benchmark_note": (
                "Phase 19 retrieval experiment evaluated 2,500 benchmark instances with curated evidence "
                "(HaluEval reference passages and FEVER Wikipedia pointers), achieving ROC-AUC 0.9184 on the holdout split. "
                "For live queries in this demo, enter a reference context above and select Evidence Mode to evaluate groundedness."
            ),
        }
    t_stages["14. evidence analysis"] = (time.perf_counter() - t0_s14) * 1000.0

    # Stage 15: final result assembly
    t0_s15 = time.perf_counter()
    token_details = format_token_details(seq_signals.tokens)
    t_stages["15. final result assembly"] = (time.perf_counter() - t0_s15) * 1000.0

    total_latency_ms = sum(t_stages.values())

    # Print / Display elapsed milliseconds for every stage
    print("=" * 65)
    print("LIVE INFERENCE STAGE LATENCIES (ms):")
    for s_name, s_ms in t_stages.items():
        print(f"  {s_name:<40}: {s_ms:8.2f} ms")
    print("-" * 65)
    print(f"  {'TOTAL PIPELINE LATENCY':<40}: {total_latency_ms:8.2f} ms ({total_latency_ms / 1000.0:.2f} s)")
    print("=" * 65)

    logger.info("Live inference stage latencies: %s (Total: %.2f ms)", t_stages, total_latency_ms)

    return {
        "prompt": clean_prompt,
        "context": clean_context,
        "generated_response": primary_response,
        "internal_signals": {
            **internal_features,
            "num_tokens": seq_signals.num_tokens,
            "perplexity": seq_signals.perplexity,
            "tokens": [t.to_dict() for t in seq_signals.tokens],
        },
        "token_details": token_details,
        "feature_contributions": feature_contributions,
        "evidence_analysis": evidence_analysis,
        "consistency_responses": sc_responses,
        "self_consistency_features": sc_features,
        "nli_features": nli_features,
        "nli_scores": nli_features,
        "feature_vector": feature_vector,
        "predicted_risk_xgb": xgb_prob,
        "predicted_risk_lr": lr_prob,
        "xgb_hallucination_probability": xgb_prob,
        "lr_hallucination_probability": lr_prob,
        "risk_level": risk_level,
        "stage_latencies_ms": t_stages,
        "timing_ms": t_stages,
        "total_latency_ms": total_latency_ms,
    }

