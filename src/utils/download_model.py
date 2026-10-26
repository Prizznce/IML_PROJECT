"""
Unified Model Downloader for VeritasLM.

Provides robust, resume-capable downloading for all supported causal LLMs:
- Qwen/Qwen3.5-0.8B
- meta-llama/Llama-3.2-1B-Instruct
- google/gemma-3-1b-it
- Any configured or standard Hugging Face causal LM repository.

Uses huggingface_hub.snapshot_download() to fetch all required model artifacts:
- Model weights (single or sharded safetensors)
- Model architecture configuration (config.json, generation_config.json)
- Tokenizers (tokenizer.json, tokenizer_config.json, vocab, merges, special tokens)
- Weight indexes (model.safetensors.index.json where applicable)

Preserves offline execution defaults by isolating Hub network access during download
and cleanly restoring the offline environment upon completion.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.model_registry import (
    DEFAULT_MODEL_ID,
    SUPPORTED_MODELS,
    get_hf_token,
    is_model_cached_locally,
)

logger = logging.getLogger("download_model")

CONFIGURED_MODELS: List[str] = [
    "Qwen/Qwen3.5-0.8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-3-1b-it",
]

MODEL_ALIASES: dict[str, str] = {
    "qwen": "Qwen/Qwen3.5-0.8B",
    "qwen3.5": "Qwen/Qwen3.5-0.8B",
    "qwen-0.8b": "Qwen/Qwen3.5-0.8B",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-base": "meta-llama/Llama-3.2-1B",
    "gemma": "google/gemma-3-1b-it",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma3": "google/gemma-3-1b-it",
}


def resolve_model_id(model_name: str, allow_custom: bool = False) -> str:
    """
    Resolve a user-provided model name or convenience alias to its canonical Hugging Face repository ID.

    Parameters:
        model_name: Input model name, alias ('qwen', 'llama', 'gemma'), or full repo ID.
        allow_custom: Whether to permit arbitrary Hugging Face repo IDs (e.g. 'org/model').

    Returns:
        Canonical Hugging Face repository ID.

    Raises:
        ValueError: If model_name is unrecognized or invalid.
    """
    if not model_name or not str(model_name).strip():
        raise ValueError("Model identifier must not be empty.")

    clean_name = str(model_name).strip()
    lower_name = clean_name.lower()

    if lower_name in MODEL_ALIASES:
        return MODEL_ALIASES[lower_name]

    for configured in CONFIGURED_MODELS:
        if clean_name == configured or lower_name == configured.lower():
            return configured

    # Check if registered in SUPPORTED_MODELS dictionary
    if clean_name in SUPPORTED_MODELS:
        return clean_name

    # Validate standard Hugging Face org/model pattern
    if re.match(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$", clean_name):
        if allow_custom:
            return clean_name
        raise ValueError(
            f"Model '{clean_name}' is not in VeritasLM configured models: {', '.join(CONFIGURED_MODELS)}. "
            f"Pass --allow-custom to download arbitrary Hugging Face models."
        )

    raise ValueError(
        f"Invalid model identifier '{clean_name}'. Expected one of {CONFIGURED_MODELS} "
        f"or convenience aliases: {list(MODEL_ALIASES.keys())}."
    )


def is_model_fully_cached(model_id: str, cache_dir: Optional[str | Path] = None) -> bool:
    """
    Verify whether all required artifacts (weights and config) exist locally on disk.

    Parameters:
        model_id: Hugging Face model repository ID.
        cache_dir: Optional custom Hugging Face hub cache directory.

    Returns:
        True if the model is verified to exist locally.
    """
    if cache_dir is not None:
        p = Path(cache_dir)
        repo_folder = "models--" + model_id.replace("/", "--")
        target_dir = p / repo_folder
        if target_dir.exists():
            snaps = target_dir / "snapshots"
            if snaps.exists() and any(snaps.iterdir()):
                return True
        return False

    return is_model_cached_locally(model_id)


def download_model(
    model: str,
    token: Optional[str] = None,
    force: bool = False,
    cache_dir: Optional[str | Path] = None,
    allow_custom: bool = False,
    local_files_only: bool = False,
) -> Optional[Path]:
    """
    Download a complete model repository snapshot (weights, configs, tokenizers) using huggingface_hub.

    Parameters:
        model: Model repository ID or alias ('qwen', 'llama', 'gemma').
        token: Optional Hugging Face authorization token (falls back to get_hf_token()).
        force: If True, re-download even if already cached locally.
        cache_dir: Optional custom cache directory.
        allow_custom: If True, allows unlisted Hugging Face repository IDs.
        local_files_only: If True, only verifies local presence without network access.

    Returns:
        Path to the local snapshot directory on success, or None on failure.
    """
    try:
        canonical_id = resolve_model_id(model, allow_custom=allow_custom)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return None

    # Check local cache first to avoid redundant downloads
    if not force and is_model_fully_cached(canonical_id, cache_dir=cache_dir):
        print(f"Model '{canonical_id}' is already cached locally.", flush=True)
        try:
            from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
            hub_root = Path(cache_dir) if cache_dir else Path(HUGGINGFACE_HUB_CACHE)
            repo_dir = hub_root / f"models--{canonical_id.replace('/', '--')}" / "snapshots"
            if repo_dir.exists():
                latest_snap = next(repo_dir.iterdir(), repo_dir)
                print(f"Snapshot directory: {latest_snap}", flush=True)
                return latest_snap
        except Exception:
            pass
        return Path(cache_dir or "default_cache")

    auth_token = get_hf_token(token)
    if not auth_token and ("llama" in canonical_id.lower() or "gemma" in canonical_id.lower()):
        print(
            f"Warning: Model '{canonical_id}' is a gated Hugging Face repository.\n"
            "If download fails with 401/403, set HF_TOKEN in your environment or pass --token.",
            flush=True,
        )

    # Temporarily isolate online execution mode for snapshot_download
    import huggingface_hub
    import huggingface_hub.constants as _C
    import huggingface_hub.errors as _E

    _orig_c_offline = getattr(_C, "HF_HUB_OFFLINE", False)
    _orig_env_hub = os.environ.get("HF_HUB_OFFLINE")
    _orig_env_tf = os.environ.get("TRANSFORMERS_OFFLINE")

    try:
        # Enable network connectivity for this specific operation
        _C.HF_HUB_OFFLINE = False
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

        print(f"Downloading snapshot for '{canonical_id}' via huggingface_hub...", flush=True)
        snapshot_path_str = huggingface_hub.snapshot_download(
            repo_id=canonical_id,
            token=auth_token,
            force_download=force,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        )
        snapshot_path = Path(snapshot_path_str)
        print(f"Successfully downloaded and verified '{canonical_id}' at:\n  {snapshot_path}", flush=True)
        return snapshot_path

    except _E.GatedRepoError as e:
        print(
            f"Error: Access denied for gated model '{canonical_id}'.\n"
            f"Please verify your Hugging Face access token (HF_TOKEN or --token) and accept model license terms at:\n"
            f"  https://huggingface.co/{canonical_id}\n"
            f"Details: {e}",
            file=sys.stderr,
            flush=True,
        )
        return None

    except _E.RepositoryNotFoundError as e:
        print(
            f"Error: Repository '{canonical_id}' was not found on Hugging Face Hub.\n"
            f"Details: {e}",
            file=sys.stderr,
            flush=True,
        )
        return None

    except _E.LocalEntryNotFoundError as e:
        print(
            f"Error: Local checkpoint for '{canonical_id}' not found and network lookup was disabled.\n"
            f"Details: {e}",
            file=sys.stderr,
            flush=True,
        )
        return None

    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "403" in err_msg or "gated" in err_msg.lower():
            print(
                f"Error: Authentication / authorization failed for '{canonical_id}'.\n"
                f"Please ensure HF_TOKEN is configured with access permissions to https://huggingface.co/{canonical_id}.\n"
                f"Details: {e}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"Error downloading model '{canonical_id}': {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
        return None

    finally:
        # Strictly restore original offline state to keep detector offline by default
        _C.HF_HUB_OFFLINE = _orig_c_offline
        if _orig_env_hub is not None:
            os.environ["HF_HUB_OFFLINE"] = _orig_env_hub
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        if _orig_env_tf is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = _orig_env_tf
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for unified model downloader."""
    parser = argparse.ArgumentParser(
        description="Unified Model Downloader for VeritasLM (Qwen, Llama, Gemma, and standard Hugging Face models)."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model repository ID or convenience alias: 'qwen', 'llama', 'gemma', or full HF ID.",
    )
    parser.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        default=None,
        help="Explicit model identifier or alias (alternative to positional argument).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (overrides HF_TOKEN env var).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fresh download even if complete model is already cached locally.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom destination directory for Hugging Face hub cache.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check whether the model is cached locally without initiating download.",
    )
    parser.add_argument(
        "--allow-custom",
        action="store_true",
        help="Allow downloading arbitrary Hugging Face repository IDs beyond configured VeritasLM models.",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[List[str]] = None) -> None:
    """CLI entry point for unified model downloader."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    args = parse_args(cli_args)

    target_model = args.model or args.model_id
    if not target_model:
        print("Error: No model specified. Please specify a model ID or alias (e.g. 'qwen', 'llama', 'gemma').", file=sys.stderr)
        print("Configured models:\n  " + "\n  ".join(CONFIGURED_MODELS), file=sys.stderr)
        sys.exit(1)

    try:
        canonical_id = resolve_model_id(target_model, allow_custom=args.allow_custom)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.check_only:
        cached = is_model_fully_cached(canonical_id, cache_dir=args.cache_dir)
        print(f"Model '{canonical_id}' cached locally: {cached}")
        sys.exit(0 if cached else 1)

    snapshot_path = download_model(
        model=canonical_id,
        token=args.token,
        force=args.force,
        cache_dir=args.cache_dir,
        allow_custom=args.allow_custom,
    )

    if snapshot_path is None:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
