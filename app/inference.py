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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd
import torch

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
from src.signals.token_signals import (
    SequenceSignals,
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
    data_path: Union[str, Path] = "experiments/baselines/combined/universal_features.parquet",
    force_reload: bool = False,
) -> Tuple[Any, Any]:
    """
    Load and fit the Phase 14 Logistic Regression and XGBoost classifiers strictly
    on the 3,200 training examples using the exact Phase 14 stratified split (random_state=42).

    Parameters:
        data_path: Path to universal features parquet table.
        force_reload: If True, forces re-fitting from training parquet.

    Returns:
        Tuple of (fitted_lr_pipeline, fitted_xgb_model).
    """
    global _GLOBAL_MODEL_CACHE
    if (
        not force_reload
        and "lr_pipeline" in _GLOBAL_MODEL_CACHE
        and "xgb_model" in _GLOBAL_MODEL_CACHE
    ):
        return _GLOBAL_MODEL_CACHE["lr_pipeline"], _GLOBAL_MODEL_CACHE["xgb_model"]

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Universal feature dataset not found at '{path}'")

    df = pd.read_parquet(path)
    X, y, meta = prepare_combined_feature_dataframe(df)

    # Reproduce exact Phase 14 stratified split (3,200 train / 800 test)
    X_train, _, y_train, _, _, _ = split_dataset(
        X, y, meta, test_size=0.2, random_state=42
    )

    if len(X_train) != 3200:
        raise ValueError(f"Expected 3,200 training samples, but got {len(X_train)}")

    logger.info("Fitting Phase 14 Logistic Regression pipeline on 3,200 training rows...")
    lr_pipeline = build_logistic_regression_pipeline(random_state=42)
    lr_pipeline.fit(X_train, y_train)

    logger.info("Fitting Phase 14 XGBoost model on 3,200 training rows...")
    xgb_model = build_xgboost_model(random_state=42)
    xgb_model.fit(X_train, y_train)

    _GLOBAL_MODEL_CACHE["lr_pipeline"] = lr_pipeline
    _GLOBAL_MODEL_CACHE["xgb_model"] = xgb_model

    return lr_pipeline, xgb_model


def load_inference_models(
    device: str = "cuda:0",
    model_id: str = "Qwen/Qwen3.5-0.8B",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    nli_model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
) -> Dict[str, Any]:
    """
    Load generative LLM, sentence embedding model, NLI model, and trained tabular classifiers.

    Parameters:
        device: Target execution device ('cuda:0' or 'cpu').
        model_id: Causal LM repository ID.
        embedding_model_name: SentenceTransformer repository ID.
        nli_model_name: CrossEncoder NLI repository ID.

    Returns:
        Dict containing loaded model instances and metadata.
    """
    global _GLOBAL_MODEL_CACHE
    actual_device = device if torch.cuda.is_available() and "cuda" in device else "cpu"

    if "causal_model" not in _GLOBAL_MODEL_CACHE:
        logger.info(f"Loading causal model '{model_id}' on {actual_device}...")
        causal_model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=actual_device)
        _GLOBAL_MODEL_CACHE["causal_model"] = causal_model
        _GLOBAL_MODEL_CACHE["tokenizer"] = tokenizer

    if "embedding_model" not in _GLOBAL_MODEL_CACHE:
        logger.info(f"Loading embedding model '{embedding_model_name}' on {actual_device}...")
        embedding_model = load_embedding_model(model_name=embedding_model_name, device=actual_device)
        _GLOBAL_MODEL_CACHE["embedding_model"] = embedding_model

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

    lr_pipeline, xgb_model = get_trained_classifiers()

    return {
        "causal_model": _GLOBAL_MODEL_CACHE["causal_model"],
        "tokenizer": _GLOBAL_MODEL_CACHE["tokenizer"],
        "embedding_model": _GLOBAL_MODEL_CACHE["embedding_model"],
        "nli_model": _GLOBAL_MODEL_CACHE["nli_model"],
        "nli_label_indices": _GLOBAL_MODEL_CACHE["nli_label_indices"],
        "lr_pipeline": lr_pipeline,
        "xgb_model": xgb_model,
        "device": actual_device,
    }


# ==============================================================================
# 2. Generation and Feature Extraction Components
# ==============================================================================

def generate_primary_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    context: Optional[str] = None,
    device: str = "cuda:0",
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
        device: Execution device.
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

    enc = tokenizer(model_input, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

    gen_tokens = gen_out[0][input_len:]
    raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return raw_response, model_input


def extract_internal_signals(
    model: Any,
    tokenizer: Any,
    model_input: str,
    response: str,
    device: str = "cuda:0",
) -> Tuple[Dict[str, float], SequenceSignals]:
    """
    Extract the 11 white-box internal generation signals from an unadulterated forward pass.

    Parameters:
        model: CausalLM instance.
        tokenizer: AutoTokenizer instance.
        model_input: Full conditioning prompt.
        response: Generated response string.
        device: Compute device.

    Returns:
        Tuple of (11-feature dict, raw SequenceSignals dataclass).
    """
    signals: SequenceSignals = extract_raw_sequence_signals(
        model=model,
        tokenizer=tokenizer,
        prompt=model_input,
        response=response,
        device=device,
    )

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

    return features, signals


def run_self_consistency_probe(
    model: Any,
    tokenizer: Any,
    embedding_model: Any,
    prompt: str,
    context: Optional[str] = None,
    device: str = "cuda:0",
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
        device: Compute device.
        num_generations: Number of stochastic paths (default: 5).
        temperature: Sampling temperature (default: 0.7).
        top_p: Nucleus threshold (default: 0.9).
        max_new_tokens: Token cap.

    Returns:
        Tuple of (list_of_responses, 5-feature dict).
    """
    responses, _, _ = generate_stochastic_responses(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        context=context or "",
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    sc_metrics = extract_self_consistency_signals(
        responses=responses,
        embedding_model=embedding_model,
        device=device,
    )

    features = {
        "exact_match_agreement": float(sc_metrics["exact_match_agreement"]),
        "mean_pairwise_similarity": float(sc_metrics["mean_pairwise_similarity"]),
        "min_pairwise_similarity": float(sc_metrics["min_pairwise_similarity"]),
        "max_pairwise_similarity": float(sc_metrics["max_pairwise_similarity"]),
        "pairwise_similarity_std": float(sc_metrics["pairwise_similarity_std"]),
    }

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
    device: str = "cuda:0",
    causal_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
    nli_model: Optional[Any] = None,
    nli_label_indices: Optional[Dict[str, int]] = None,
    lr_pipeline: Optional[Any] = None,
    xgb_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute full end-to-end hallucination risk estimation for a single prompt.

    Parameters:
        prompt: User query (must be non-empty string).
        context: Optional reference context passage.
        device: Target execution device.
        causal_model, tokenizer, embedding_model, nli_model, nli_label_indices,
        lr_pipeline, xgb_model: Optional pre-loaded models to bypass repeated loading.

    Returns:
        Structured result dictionary containing generated responses, signal breakdown,
        the 19-feature vector, and predicted hallucination risks.
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

    # Resolve models: use passed instances or load via cache
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
        actual_device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        if nli_label_indices is None:
            nli_label_indices = get_nli_label_indices(nli_model)

    # 1. Primary generation
    primary_response, model_input = generate_primary_response(
        model=causal_model,
        tokenizer=tokenizer,
        prompt=clean_prompt,
        context=clean_context,
        device=actual_device,
    )

    # 2. Internal generation signals
    internal_features, seq_signals = extract_internal_signals(
        model=causal_model,
        tokenizer=tokenizer,
        model_input=model_input,
        response=primary_response,
        device=actual_device,
    )

    # 3. Behavioral self-consistency (k=5)
    sc_responses, sc_features = run_self_consistency_probe(
        model=causal_model,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        prompt=clean_prompt,
        context=clean_context,
        device=actual_device,
    )

    # 4. Pairwise NLI agreement (10 pairs / 20 passes)
    nli_features = run_nli_probe(
        nli_model=nli_model,
        responses=sc_responses,
        label_indices=nli_label_indices,
    )

    # 5. Assemble canonical 19-feature vector
    feature_vector = assemble_feature_vector(
        internal_features=internal_features,
        sc_features=sc_features,
        nli_features=nli_features,
    )

    # 6. Classifier risk prediction
    xgb_prob, lr_prob, risk_level = predict_hallucination_risk(
        feature_vector=feature_vector,
        lr_pipeline=lr_pipeline,
        xgb_model=xgb_model,
    )

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
    }
