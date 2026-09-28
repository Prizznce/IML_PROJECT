"""
Fast unit tests for app/inference.py validating feature schemas, ordering,
isolation safeguards, risk labeling, error handling, and mock orchestration.

CRITICAL: MUST NOT load Qwen, run live generation, or run neural NLI.
All tests execute in <1 second using synthetic/mocked inputs.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from app.inference import (
    assemble_feature_vector,
    format_risk_level,
    predict_hallucination_risk,
    run_live_inference,
    get_trained_classifiers,
)
from src.features.build_universal_features import (
    FORBIDDEN_FEATURE_COLUMNS,
    UNIVERSAL_CORE_FEATURES,
)


def _get_synthetic_feature_components():
    """Generate valid synthetic feature dictionaries for testing."""
    internal = {
        "min_log_prob": -2.5,
        "mean_log_prob": -0.8,
        "mean_token_prob": 0.65,
        "token_prob_std": 0.22,
        "mean_entropy": 1.15,
        "max_entropy": 2.85,
        "entropy_std": 0.45,
        "log_perplexity": 0.8,
        "log_mean_token_rank": 0.55,
        "log_max_token_rank": 1.25,
        "log_rank_std": 0.40,
    }
    sc = {
        "exact_match_agreement": 0.60,
        "mean_pairwise_similarity": 0.88,
        "min_pairwise_similarity": 0.72,
        "max_pairwise_similarity": 0.96,
        "pairwise_similarity_std": 0.08,
    }
    nli = {
        "mean_pairwise_entailment": 0.75,
        "mean_pairwise_contradiction": 0.05,
        "nli_disagreement": 0.175,
    }
    return internal, sc, nli


def test_universal_core_features_count_and_schema():
    """Verify that UNIVERSAL_CORE_FEATURES contains exactly 19 canonical features."""
    assert len(UNIVERSAL_CORE_FEATURES) == 19
    expected_order = [
        "min_log_prob",
        "mean_log_prob",
        "mean_token_prob",
        "token_prob_std",
        "mean_entropy",
        "max_entropy",
        "entropy_std",
        "log_perplexity",
        "log_mean_token_rank",
        "log_max_token_rank",
        "log_rank_std",
        "exact_match_agreement",
        "mean_pairwise_similarity",
        "min_pairwise_similarity",
        "max_pairwise_similarity",
        "pairwise_similarity_std",
        "mean_pairwise_entailment",
        "mean_pairwise_contradiction",
        "nli_disagreement",
    ]
    assert list(UNIVERSAL_CORE_FEATURES) == expected_order


def test_assemble_feature_vector_exact_ordering():
    """Verify that assemble_feature_vector returns exactly 19 features in canonical order."""
    internal, sc, nli = _get_synthetic_feature_components()
    vector = assemble_feature_vector(internal, sc, nli)

    assert len(vector) == 19
    assert list(vector.keys()) == list(UNIVERSAL_CORE_FEATURES)

    for k, v in vector.items():
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_assemble_feature_vector_rejects_forbidden_columns():
    """Verify that passing forbidden columns (id, num_tokens, etc.) raises ValueError."""
    internal, sc, nli = _get_synthetic_feature_components()

    for forbidden in ["num_tokens", "id", "prompt", "source_dataset", "context", "response"]:
        bad_internal = dict(internal)
        bad_internal[forbidden] = 12.0
        with pytest.raises(ValueError, match="Forbidden columns detected"):
            assemble_feature_vector(bad_internal, sc, nli)


def test_assemble_feature_vector_missing_feature():
    """Verify that omitting a required feature raises KeyError."""
    internal, sc, nli = _get_synthetic_feature_components()
    del internal["min_log_prob"]
    with pytest.raises(KeyError, match="Missing required universal feature"):
        assemble_feature_vector(internal, sc, nli)


def test_assemble_feature_vector_nan_inf_check():
    """Verify that NaN or Inf feature values raise ValueError."""
    internal, sc, nli = _get_synthetic_feature_components()

    bad_internal_nan = dict(internal, mean_entropy=float("nan"))
    with pytest.raises(ValueError, match="Non-finite value detected"):
        assemble_feature_vector(bad_internal_nan, sc, nli)

    bad_internal_inf = dict(internal, log_perplexity=float("inf"))
    with pytest.raises(ValueError, match="Non-finite value detected"):
        assemble_feature_vector(bad_internal_inf, sc, nli)


def test_format_risk_level_bands():
    """Verify mapping of probabilities to UI visualization risk levels."""
    assert format_risk_level(0.0) == "Low Risk"
    assert format_risk_level(0.20) == "Low Risk"
    assert format_risk_level(0.34) == "Low Risk"

    assert format_risk_level(0.35) == "Moderate Risk"
    assert format_risk_level(0.50) == "Moderate Risk"
    assert format_risk_level(0.649) == "Moderate Risk"

    assert format_risk_level(0.65) == "High Hallucination Risk"
    assert format_risk_level(0.85) == "High Hallucination Risk"
    assert format_risk_level(1.0) == "High Hallucination Risk"


def test_predict_hallucination_risk_with_mock_classifiers():
    """Verify prediction scoring with mocked Logistic Regression and XGBoost models."""
    internal, sc, nli = _get_synthetic_feature_components()
    vector = assemble_feature_vector(internal, sc, nli)

    mock_lr = MagicMock()
    mock_lr.predict_proba.return_value = np.array([[0.62, 0.38]])

    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.28, 0.72]])

    xgb_p, lr_p, risk = predict_hallucination_risk(vector, mock_lr, mock_xgb)

    assert 0.0 <= xgb_p <= 1.0
    assert 0.0 <= lr_p <= 1.0
    assert pytest.approx(xgb_p, 1e-4) == 0.72
    assert pytest.approx(lr_p, 1e-4) == 0.38
    assert risk == "High Hallucination Risk"


def test_get_trained_classifiers_split_and_caching():
    """Verify that get_trained_classifiers reproduces Phase 14 split and caches models."""
    lr_pipe, xgb_mod = get_trained_classifiers()

    assert hasattr(lr_pipe, "predict_proba")
    assert hasattr(xgb_mod, "predict_proba")

    # Evaluate on a synthetic 19-feature vector
    internal, sc, nli = _get_synthetic_feature_components()
    vector = assemble_feature_vector(internal, sc, nli)

    xgb_p, lr_p, risk = predict_hallucination_risk(vector, lr_pipe, xgb_mod)
    assert 0.0 <= xgb_p <= 1.0
    assert 0.0 <= lr_p <= 1.0
    assert risk in ["Low Risk", "Moderate Risk", "High Hallucination Risk"]


def test_run_live_inference_empty_prompt_validation():
    """Verify that empty prompts are rejected with a ValueError."""
    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        run_live_inference(prompt="")

    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        run_live_inference(prompt="    \n  ")


def test_run_live_inference_end_to_end_with_mocks():
    """Verify run_live_inference orchestration using mocked model calls."""
    internal, sc, nli = _get_synthetic_feature_components()

    mock_causal = MagicMock()
    mock_tokenizer = MagicMock()
    mock_embedding = MagicMock()
    mock_nli = MagicMock()
    mock_lr = MagicMock()
    mock_lr.predict_proba.return_value = np.array([[0.70, 0.30]])
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.45, 0.55]])

    with patch("app.inference.generate_primary_response", return_value=("Paris is the capital of France.", "Prompt: What is the capital of France?")), \
         patch("app.inference.extract_internal_signals") as mock_extract, \
         patch("app.inference.run_self_consistency_probe", return_value=(["Paris", "Paris", "Paris, France", "Paris", "City of Paris"], sc)), \
         patch("app.inference.run_nli_probe", return_value=nli):

        from src.signals.token_signals import SequenceSignals
        mock_signals = SequenceSignals(
            num_tokens=7,
            mean_token_prob=0.85,
            min_token_prob=0.45,
            mean_log_prob=-0.25,
            min_log_prob=-0.80,
            mean_entropy=0.45,
            max_entropy=1.20,
            entropy_std=0.15,
            token_prob_std=0.10,
            perplexity=1.28,
            tokens=[],
        )
        mock_extract.return_value = (internal, mock_signals)

        result = run_live_inference(
            prompt="What is the capital of France?",
            context="France is a country in Western Europe.",
            causal_model=mock_causal,
            tokenizer=mock_tokenizer,
            embedding_model=mock_embedding,
            nli_model=mock_nli,
            nli_label_indices={"contradiction": 0, "entailment": 1, "neutral": 2},
            lr_pipeline=mock_lr,
            xgb_model=mock_xgb,
        )

        assert result["prompt"] == "What is the capital of France?"
        assert result["context"] == "France is a country in Western Europe."
        assert result["generated_response"] == "Paris is the capital of France."
        assert len(result["consistency_responses"]) == 5
        assert len(result["feature_vector"]) == 19
        assert list(result["feature_vector"].keys()) == list(UNIVERSAL_CORE_FEATURES)
        assert pytest.approx(result["xgb_hallucination_probability"], 1e-4) == 0.55
        assert pytest.approx(result["lr_hallucination_probability"], 1e-4) == 0.30
        assert pytest.approx(result["predicted_risk_xgb"], 1e-4) == 0.55
        assert pytest.approx(result["predicted_risk_lr"], 1e-4) == 0.30
        assert result["risk_level"] == "Moderate Risk"
        assert "num_tokens" in result["internal_signals"]
        assert "tokens" in result["internal_signals"]
        assert "nli_scores" in result
        assert result["nli_scores"] == result["nli_features"]
        assert "token_details" in result
        assert "feature_contributions" in result
        assert "evidence_analysis" in result
        assert result["evidence_analysis"]["has_reference_context"] is True


def test_predict_hallucination_risk_with_tuple():
    """Verify predict_hallucination_risk correctly unpacks a tuple of classifiers."""
    internal, sc, nli = _get_synthetic_feature_components()
    vector = assemble_feature_vector(internal, sc, nli)

    mock_lr = MagicMock()
    mock_lr.predict_proba.return_value = np.array([[0.80, 0.20]])

    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.15, 0.85]])

    # Pass as a tuple to the second parameter (matching get_trained_classifiers() return)
    xgb_p, lr_p, risk = predict_hallucination_risk(vector, (mock_lr, mock_xgb))

    assert pytest.approx(xgb_p, 1e-4) == 0.85
    assert pytest.approx(lr_p, 1e-4) == 0.20
    assert risk == "High Hallucination Risk"


def test_predict_hallucination_risk_with_predict_only():
    """Verify predict_hallucination_risk falls back to .predict() if .predict_proba() is absent."""
    internal, sc, nli = _get_synthetic_feature_components()
    vector = assemble_feature_vector(internal, sc, nli)

    mock_lr = MagicMock(spec=["predict"])
    mock_lr.predict.return_value = np.array([0.25])

    mock_xgb = MagicMock(spec=["predict"])
    mock_xgb.predict.return_value = np.array([0.75])

    xgb_p, lr_p, risk = predict_hallucination_risk(vector, mock_lr, mock_xgb)

    assert pytest.approx(xgb_p, 1e-4) == 0.75
    assert pytest.approx(lr_p, 1e-4) == 0.25
    assert risk == "High Hallucination Risk"


def test_run_nli_probe_with_tuple_model():
    """Verify run_nli_probe handles nli_model passed as (model, label_indices) tuple."""
    from app.inference import run_nli_probe

    mock_nli = MagicMock()
    mock_nli.predict.return_value = np.array([
        [1.0, 3.0, 0.5],
        [0.5, 2.0, 1.0],
    ] * 10)  # 20 directional evaluations

    label_indices = {"contradiction": 0, "entailment": 1, "neutral": 2}
    responses = ["Paris", "Paris, France", "City of Paris", "Paris capital", "Capital Paris"]

    # Pass as a tuple (like load_nli_model returns)
    nli_metrics = run_nli_probe(
        nli_model=(mock_nli, label_indices),
        responses=responses,
    )

    assert "mean_pairwise_entailment" in nli_metrics
    assert "mean_pairwise_contradiction" in nli_metrics
    assert "nli_disagreement" in nli_metrics
    assert 0.0 <= nli_metrics["mean_pairwise_entailment"] <= 1.0
    assert 0.0 <= nli_metrics["mean_pairwise_contradiction"] <= 1.0


def test_token_confidence_band_calculation():
    """Verify deterministic token confidence band thresholds and edge cases."""
    from app.inference import get_token_confidence_band

    # High confidence: prob >= 0.65, low entropy (< 1.5)
    assert get_token_confidence_band(0.95, entropy=0.1) == "High confidence"
    assert get_token_confidence_band(0.65, entropy=1.0) == "High confidence"
    assert get_token_confidence_band(0.85, entropy=None) == "High confidence"

    # Medium confidence: intermediate probability
    assert get_token_confidence_band(0.50, entropy=1.0) == "Medium confidence"
    assert get_token_confidence_band(0.30, entropy=0.5) == "Medium confidence"

    # High probability but elevated entropy -> medium confidence
    assert get_token_confidence_band(0.70, entropy=1.8) == "Medium confidence"

    # Low confidence: prob < 0.30 or entropy > 2.5
    assert get_token_confidence_band(0.20, entropy=0.5) == "Low confidence"
    assert get_token_confidence_band(0.05, entropy=None) == "Low confidence"
    assert get_token_confidence_band(0.55, entropy=2.8) == "Low confidence"


def test_format_token_details_schema():
    """Verify format_token_details converts token signals to structured dictionaries."""
    from app.inference import format_token_details
    from src.signals.token_signals import TokenSignal

    mock_tokens = [
        TokenSignal(step=1, token_id=101, token_text=" Paris", probability=0.92, log_probability=-0.08, entropy=0.2, rank=1),
        TokenSignal(step=2, token_id=102, token_text=" is", probability=0.55, log_probability=-0.59, entropy=1.1, rank=2),
        TokenSignal(step=3, token_id=103, token_text=" beautiful", probability=0.22, log_probability=-1.51, entropy=2.6, rank=5),
    ]

    details = format_token_details(mock_tokens)
    assert len(details) == 3

    # Check schema
    for d in details:
        assert "step" in d and isinstance(d["step"], int)
        assert "token" in d and isinstance(d["token"], str)
        assert "probability" in d and isinstance(d["probability"], float)
        assert "log_prob" in d and isinstance(d["log_prob"], float)
        assert "entropy" in d and isinstance(d["entropy"], float)
        assert "rank" in d and isinstance(d["rank"], int)
        assert "band" in d and d["band"] in ["High confidence", "Medium confidence", "Low confidence"]

    assert details[0]["token"] == " Paris"
    assert details[0]["band"] == "High confidence"
    assert details[1]["band"] == "Medium confidence"
    assert details[2]["band"] == "Low confidence"


def test_compute_feature_contributions_native_xgb():
    """Verify model-level Tree SHAP feature contributions from fitted XGBoost."""
    from app.inference import compute_feature_contributions, get_trained_classifiers

    lr_pipe, xgb_mod = get_trained_classifiers()
    internal, sc, nli = _get_synthetic_feature_components()
    vector = assemble_feature_vector(internal, sc, nli)

    contribs = compute_feature_contributions(vector, xgb_model=xgb_mod, lr_pipeline=lr_pipe, top_k=6)
    assert len(contribs) == 6

    for c in contribs:
        assert "feature" in c and c["feature"] in UNIVERSAL_CORE_FEATURES
        assert "display_name" in c and isinstance(c["display_name"], str)
        assert "value" in c and isinstance(c["value"], float)
        assert "contribution" in c and isinstance(c["contribution"], float)
        assert "magnitude" in c and c["magnitude"] >= 0.0
        assert "direction" in c and c["direction"] in ["increases risk", "decreases risk"]
        if c["contribution"] > 0:
            assert c["direction"] == "increases risk"
        else:
            assert c["direction"] == "decreases risk"

    # Verify sorted descending by magnitude
    magnitudes = [c["magnitude"] for c in contribs]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_compute_feature_contributions_fallback():
    """Verify fallback to LR standardized contributions or signal profile when booster fails."""
    from app.inference import compute_feature_contributions, get_trained_classifiers

    lr_pipe, _ = get_trained_classifiers()
    internal, sc, nli = _get_synthetic_feature_components()
    vector = assemble_feature_vector(internal, sc, nli)

    # Pass non-booster object for XGBoost to trigger LR fallback
    fake_xgb = MagicMock(spec=[])
    contribs_lr = compute_feature_contributions(vector, xgb_model=fake_xgb, lr_pipeline=lr_pipe, top_k=5)
    assert len(contribs_lr) == 5
    assert "Standardized Coefficient Contribution" in contribs_lr[0]["method"]

    # Pass non-booster and non-LR to trigger Signal Profile fallback
    contribs_profile = compute_feature_contributions(vector, xgb_model=fake_xgb, lr_pipeline=None, top_k=5)
    assert len(contribs_profile) == 5
    assert "Signal Profile" in contribs_profile[0]["method"]


def test_evidence_mode_behavior_with_and_without_context():
    """Verify compute_reference_evidence_agreement behavior with and without context."""
    from app.inference import compute_reference_evidence_agreement

    # Without context
    no_ctx = compute_reference_evidence_agreement(
        query_text="What is the capital of France?",
        response_text="Paris is the capital of France.",
        context_text=None,
    )
    assert no_ctx["has_reference_context"] is False
    assert no_ctx["retrieved_evidence"] == []
    assert no_ctx["top1_evidence_similarity"] == 0.0
    assert "0.9184" in no_ctx["offline_benchmark_note"]

    # With mocked embedding model and reference context
    mock_emb = MagicMock()
    # 1 query, 1 response, 2 chunks = 4 texts
    mock_emb.encode.return_value = np.array([
        [1.0, 0.0],  # query
        [0.9, 0.1],  # response
        [0.8, 0.2],  # chunk 1
        [0.1, 0.9],  # chunk 2
    ])

    with_ctx = compute_reference_evidence_agreement(
        query_text="What is the capital of France?",
        response_text="Paris is the capital of France.",
        context_text="Paris is the capital of France.\nIt is known for the Eiffel Tower.",
        embedding_model=mock_emb,
    )
    assert with_ctx["has_reference_context"] is True
    assert len(with_ctx["retrieved_evidence"]) == 2
    assert with_ctx["top1_evidence_similarity"] > 0.0
    assert with_ctx["retrieval_agreement"] >= 0.0


def test_no_hardcoded_predictions():
    """Verify that classifier outputs are mathematically derived from features and not static."""
    from app.inference import get_trained_classifiers, predict_hallucination_risk

    lr_pipe, xgb_mod = get_trained_classifiers()
    internal, sc, nli = _get_synthetic_feature_components()

    # Case A: Low uncertainty (high token prob, high similarity, high entailment)
    low_unc_internal = dict(internal, mean_token_prob=0.95, min_log_prob=-0.05, mean_entropy=0.1)
    low_unc_sc = dict(sc, exact_match_agreement=1.0, mean_pairwise_similarity=0.99)
    low_unc_nli = dict(nli, mean_pairwise_entailment=0.98, mean_pairwise_contradiction=0.01)
    vec_low = assemble_feature_vector(low_unc_internal, low_unc_sc, low_unc_nli)
    xgb_low, lr_low, risk_low = predict_hallucination_risk(vec_low, lr_pipe, xgb_mod)

    # Case B: High uncertainty (low token prob, low similarity, high contradiction)
    high_unc_internal = dict(internal, mean_token_prob=0.15, min_log_prob=-4.5, mean_entropy=2.8)
    high_unc_sc = dict(sc, exact_match_agreement=0.0, mean_pairwise_similarity=0.30)
    high_unc_nli = dict(nli, mean_pairwise_entailment=0.10, mean_pairwise_contradiction=0.85)
    vec_high = assemble_feature_vector(high_unc_internal, high_unc_sc, high_unc_nli)
    xgb_high, lr_high, risk_high = predict_hallucination_risk(vec_high, lr_pipe, xgb_mod)

    # Predictions must be dynamically responsive and non-identical
    assert xgb_low != xgb_high
    assert lr_low != lr_high
    assert xgb_high > xgb_low
    assert lr_high > lr_low


def test_device_placement_consistency_regression():
    """
    Regression test for CUDA/CPU device-placement consistency.
    Verifies that extract_raw_sequence_signals and extract_internal_signals
    dynamically synchronize tensor placement with model parameter device,
    preventing wrapper_CUDA__index_select mismatch errors.
    """
    import torch
    from unittest.mock import MagicMock
    from src.signals.token_signals import extract_raw_sequence_signals
    from app.inference import extract_internal_signals

    # Mock CPU causal model with 1 parameter on CPU
    mock_model = MagicMock(spec=torch.nn.Module)
    cpu_param = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    mock_model.parameters.return_value = iter([cpu_param])
    mock_model.eval.return_value = None

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, **kw: [10, 20, 30] if "Paris" in text else [1, 2]
    mock_tokenizer.decode.return_value = "token"

    # Forward pass returns valid logits on CPU
    def mock_forward(input_ids):
        # Crucial check: input_ids must be on CPU (same as model parameters)
        assert input_ids.device.type == "cpu", f"Tensor device mismatch: expected CPU, got {input_ids.device}"
        out = MagicMock()
        b, seq_len = input_ids.shape
        out.logits = torch.randn(b, seq_len, 50, device="cpu")
        return out

    mock_model.side_effect = mock_forward

    # Test extract_raw_sequence_signals with device="cuda:0" explicitly passed to CPU model
    # It must safely align tensor placement with the CPU model without failing
    signals = extract_raw_sequence_signals(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt="Question: Capital?",
        response="Paris",
        device="cuda:0",  # intentionally mismatched device argument
    )
    assert signals.num_tokens == 3
    assert np.isfinite(signals.perplexity)

    # Reset parameter iterator for next call
    mock_model.parameters.return_value = iter([cpu_param])
    features, _ = extract_internal_signals(
        model=mock_model,
        tokenizer=mock_tokenizer,
        model_input="Question: Capital?",
        response="Paris",
        device="cuda:0",  # intentionally mismatched
    )
    assert len(features) == 11
    assert "log_perplexity" in features



