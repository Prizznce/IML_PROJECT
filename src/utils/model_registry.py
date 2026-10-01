"""
Model registry and path resolution utilities for multi-model hallucination detection.

Supports seamless coexistence of:
  - Qwen/Qwen3.5-0.8B (primary baseline)
  - meta-llama/Llama-3.2-1B-Instruct / meta-llama/Llama-3.2-1B (new model)
  - Any standard Hugging Face causal LM
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# Root repository directory
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Known model configurations
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3.5-0.8B": {
        "slug": "qwen3_5_0_8b",
        "display_name": "Qwen 3.5 (0.8B)",
        "supports_thinking_suppression": True,
        "is_instruct": True,
        "default_universal_features": REPO_ROOT / "experiments" / "baselines" / "combined" / "universal_features.parquet",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "slug": "llama_3_2_1b_instruct",
        "display_name": "Llama 3.2 (1B Instruct)",
        "supports_thinking_suppression": False,
        "is_instruct": True,
        "default_universal_features": REPO_ROOT / "experiments" / "models" / "llama_3_2_1b_instruct" / "combined" / "universal_features.parquet",
    },
    "meta-llama/Llama-3.2-1B": {
        "slug": "llama_3_2_1b",
        "display_name": "Llama 3.2 (1B Base)",
        "supports_thinking_suppression": False,
        "is_instruct": False,
        "default_universal_features": REPO_ROOT / "experiments" / "models" / "llama_3_2_1b" / "combined" / "universal_features.parquet",
    },
    "google/gemma-3-1b-it": {
        "slug": "gemma_3_1b_it",
        "display_name": "Gemma 3 (1B IT)",
        "supports_thinking_suppression": False,
        "is_instruct": True,
        "default_universal_features": REPO_ROOT / "experiments" / "models" / "gemma_3_1b_it" / "combined" / "universal_features.parquet",
    },
}

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def get_model_slug(model_id: str) -> str:
    """
    Generate a clean filesystem-safe slug for any Hugging Face model repository ID.

    Examples:
        'Qwen/Qwen3.5-0.8B' -> 'qwen3_5_0_8b'
        'meta-llama/Llama-3.2-1B-Instruct' -> 'llama_3_2_1b_instruct'
    """
    if model_id in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_id]["slug"]
    
    # Generic slugification
    clean = model_id.split("/")[-1].lower()
    clean = re.sub(r"[^a-z0-9_]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "custom_model"


def get_model_experiment_dir(model_id: str) -> Path:
    """
    Resolve the dedicated experiment directory for a given model.
    For Qwen3.5-0.8B, points to experiments/models/qwen3_5_0_8b while keeping
    experiments/baselines preserved as reference.
    """
    slug = get_model_slug(model_id)
    exp_dir = REPO_ROOT / "experiments" / "models" / slug
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_saved_model_dir(model_id: str) -> Path:
    """
    Resolve directory for serialized trained classifiers (joblib / json).
    """
    slug = get_model_slug(model_id)
    save_dir = REPO_ROOT / "models" / slug
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_universal_features_path(model_id: str) -> Path:
    """
    Resolve the path to universal_features.parquet for a specific model.
    Falls back to experiments/baselines/combined/universal_features.parquet for Qwen
    if experiments/models/qwen3_5_0_8b/... does not exist.
    """
    slug = get_model_slug(model_id)
    model_path = REPO_ROOT / "experiments" / "models" / slug / "combined" / "universal_features.parquet"
    if model_path.exists():
        return model_path

    # Check for legacy Qwen path
    if "qwen" in slug:
        legacy_path = REPO_ROOT / "experiments" / "baselines" / "combined" / "universal_features.parquet"
        if legacy_path.exists():
            return legacy_path

    return model_path


def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Retrieve Hugging Face access token with fallback order:
      1. explicit_token argument
      2. HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variables
      3. .env file in project root
      4. huggingface_hub cache token
    """
    if explicit_token:
        return explicit_token

    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    # Try .env
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                    token_val = v.strip().strip("'\"")
                    if token_val:
                        return token_val
        except Exception:
            pass

    # Try huggingface_hub default token
    try:
        from huggingface_hub import get_token
        hub_token = get_token()
        if hub_token:
            return hub_token
    except Exception:
        pass

    return None
