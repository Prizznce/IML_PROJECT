"""
Unit tests for the Unified Model Downloader (src/utils/download_model.py).

Validates:
1. Canonical model ID resolution and convenience aliases (qwen, llama, gemma).
2. Strict rejection of invalid/unsupported model identifiers.
3. Cache-check bypassing to prevent redundant network requests.
4. Correct invocation of huggingface_hub.snapshot_download for missing or forced downloads.
5. Actionable error reporting on gated repository authentication failures (401/403/GatedRepoError).
6. Nonzero exit status on failure in CLI invocation.
7. Safe isolation and restoration of offline execution environment settings.
8. Preservation of offline inference defaults.
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.utils.download_model import (
    CONFIGURED_MODELS,
    MODEL_ALIASES,
    download_model,
    main,
    parse_args,
    resolve_model_id,
)
from src.utils.model_registry import is_offline_mode


def test_resolve_model_id_configured_models():
    """Verify exact resolution of all primary configured model IDs."""
    for model_id in CONFIGURED_MODELS:
        resolved = resolve_model_id(model_id)
        assert resolved == model_id


def test_resolve_model_id_convenience_aliases():
    """Verify alias mapping for qwen, llama, and gemma variants."""
    assert resolve_model_id("qwen") == "Qwen/Qwen3.5-0.8B"
    assert resolve_model_id("qwen3.5") == "Qwen/Qwen3.5-0.8B"
    assert resolve_model_id("QWEN") == "Qwen/Qwen3.5-0.8B"
    assert resolve_model_id("llama") == "meta-llama/Llama-3.2-1B-Instruct"
    assert resolve_model_id("llama-instruct") == "meta-llama/Llama-3.2-1B-Instruct"
    assert resolve_model_id("llama-base") == "meta-llama/Llama-3.2-1B"
    assert resolve_model_id("gemma") == "google/gemma-3-1b-it"
    assert resolve_model_id("gemma3") == "google/gemma-3-1b-it"


def test_resolve_model_id_invalid_rejection():
    """Verify invalid or unconfigured model identifiers are rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_model_id("")

    with pytest.raises(ValueError, match="Invalid model identifier"):
        resolve_model_id("random-unsupported-string")

    with pytest.raises(ValueError, match="not in VeritasLM configured models"):
        resolve_model_id("other-org/other-model", allow_custom=False)


def test_resolve_model_id_custom_allowed():
    """Verify arbitrary valid org/model IDs are permitted when allow_custom=True."""
    custom_id = "mistralai/Mistral-7B-v0.1"
    resolved = resolve_model_id(custom_id, allow_custom=True)
    assert resolved == custom_id


def test_cached_model_does_not_trigger_network_download():
    """Verify that an already-cached model returns immediately without downloading."""
    with patch("src.utils.download_model.is_model_fully_cached", return_value=True), \
         patch("huggingface_hub.snapshot_download") as mock_snap:
        path = download_model("qwen", force=False)
        assert path is not None
        assert mock_snap.call_count == 0


def test_force_flag_triggers_download_even_if_cached():
    """Verify that --force flag forces invocation of snapshot_download."""
    with patch("src.utils.download_model.is_model_fully_cached", return_value=True), \
         patch("huggingface_hub.snapshot_download", return_value="/mock/snapshot/qwen") as mock_snap:
        path = download_model("qwen", force=True)
        assert path == Path("/mock/snapshot/qwen")
        assert mock_snap.call_count == 1
        assert mock_snap.call_args[1]["force_download"] is True
        assert mock_snap.call_args[1]["repo_id"] == "Qwen/Qwen3.5-0.8B"


def test_missing_model_invokes_snapshot_download():
    """Verify that uncached models trigger snapshot_download with canonical repo ID and token."""
    with patch("src.utils.download_model.is_model_fully_cached", return_value=False), \
         patch("src.utils.download_model.get_hf_token", return_value="mock_token_123"), \
         patch("huggingface_hub.snapshot_download", return_value="/mock/snapshot/gemma") as mock_snap:
        path = download_model("gemma", force=False)
        assert path == Path("/mock/snapshot/gemma")
        assert mock_snap.call_count == 1
        kwargs = mock_snap.call_args[1]
        assert kwargs["repo_id"] == "google/gemma-3-1b-it"
        assert kwargs["token"] == "mock_token_123"


def test_gated_model_authentication_error_is_actionable(capsys):
    """Verify clear error messaging when a gated model fails authentication."""
    import huggingface_hub.errors as E
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    gated_err = E.GatedRepoError("Gated repo access denied", response=mock_resp)
    with patch("src.utils.download_model.is_model_fully_cached", return_value=False), \
         patch("huggingface_hub.snapshot_download", side_effect=gated_err):
        path = download_model("llama", force=False)
        assert path is None
        captured = capsys.readouterr()
        assert "Access denied for gated model" in captured.err
        assert "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct" in captured.err
        assert "HF_TOKEN" in captured.err


def test_repository_not_found_error_is_actionable(capsys):
    """Verify clear error messaging when repository is not found."""
    import huggingface_hub.errors as E
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    repo_err = E.RepositoryNotFoundError("Repo not found", response=mock_resp)
    with patch("src.utils.download_model.is_model_fully_cached", return_value=False), \
         patch("huggingface_hub.snapshot_download", side_effect=repo_err):
        path = download_model("custom/model", allow_custom=True)
        assert path is None
        captured = capsys.readouterr()
        assert "Repository 'custom/model' was not found" in captured.err


def test_offline_environment_isolated_and_restored():
    """Verify download_model temporarily enables network access and restores offline defaults on completion."""
    import huggingface_hub.constants as C
    # Setup initial offline state
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    C.HF_HUB_OFFLINE = True

    observed_offline_during_download = []

    def mock_snapshot(*args, **kwargs):
        observed_offline_during_download.append(C.HF_HUB_OFFLINE)
        observed_offline_during_download.append(os.environ.get("HF_HUB_OFFLINE"))
        return "/mock/path"

    with patch("src.utils.download_model.is_model_fully_cached", return_value=False), \
         patch("huggingface_hub.snapshot_download", side_effect=mock_snapshot):
        path = download_model("qwen", force=True)
        assert path == Path("/mock/path")

    # Inside snapshot_download, offline flags were turned off
    assert observed_offline_during_download == [False, "0"]

    # Outside download_model, original offline state was restored
    assert C.HF_HUB_OFFLINE is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert is_offline_mode() is True


def test_cli_help_flag():
    """Verify CLI --help exits with code 0 without initiating network calls."""
    with patch("sys.argv", ["download_model.py", "--help"]), \
         pytest.raises(SystemExit) as cm:
        main()
    assert cm.value.code == 0


def test_cli_no_args_exits_nonzero(capsys):
    """Verify CLI with no model argument prints usage and exits with code 1."""
    with patch("sys.argv", ["download_model.py"]), \
         pytest.raises(SystemExit) as cm:
        main()
    assert cm.value.code == 1
    captured = capsys.readouterr()
    assert "Error: No model specified" in captured.err


def test_cli_invalid_model_exits_nonzero(capsys):
    """Verify CLI with invalid model exits with code 1."""
    with patch("sys.argv", ["download_model.py", "unsupported-model-xyz"]), \
         pytest.raises(SystemExit) as cm:
        main()
    assert cm.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Invalid model identifier" in captured.err


def test_cli_successful_cached_model_exits_zero():
    """Verify CLI with already cached model exits with code 0."""
    with patch("src.utils.download_model.is_model_fully_cached", return_value=True), \
         patch("sys.argv", ["download_model.py", "qwen"]), \
         pytest.raises(SystemExit) as cm:
        main()
    assert cm.value.code == 0


def test_cli_check_only_flag():
    """Verify --check-only exits 0 if cached and 1 if not cached."""
    with patch("src.utils.download_model.is_model_fully_cached", return_value=True), \
         patch("sys.argv", ["download_model.py", "llama", "--check-only"]), \
         pytest.raises(SystemExit) as cm:
        main()
    assert cm.value.code == 0

    with patch("src.utils.download_model.is_model_fully_cached", return_value=False), \
         patch("sys.argv", ["download_model.py", "llama", "--check-only"]), \
         pytest.raises(SystemExit) as cm:
        main()
    assert cm.value.code == 1


def test_interrupted_download_error_handling(capsys):
    """Verify that network interruptions (e.g. ConnectionResetError) are caught, reported, and return None."""
    with patch("src.utils.download_model.is_model_fully_cached", return_value=False), \
         patch("huggingface_hub.snapshot_download", side_effect=ConnectionResetError("Connection reset by peer")):
        path = download_model("gemma", force=False)
        assert path is None
        captured = capsys.readouterr()
        assert "Error downloading model 'google/gemma-3-1b-it'" in captured.err
        assert "ConnectionResetError" in captured.err


def test_cli_interrupted_download_exits_nonzero(capsys):
    """Verify that CLI invocation with an interrupted download cleanly exits with code 1."""
    with patch("src.utils.download_model.is_model_fully_cached", return_value=False), \
         patch("huggingface_hub.snapshot_download", side_effect=ConnectionResetError("Connection reset mid-stream")), \
         patch("sys.argv", ["download_model.py", "gemma"]), \
         pytest.raises(SystemExit) as cm:
        main()
    assert cm.value.code == 1
    captured = capsys.readouterr()
    assert "Error downloading model 'google/gemma-3-1b-it'" in captured.err

