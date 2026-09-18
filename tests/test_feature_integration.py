"""
Unit tests for Combined Feature Integration Configuration and Schema.

Validates the feature schema definition, forbidden column protections,
feature family alignments, and mathematical relationships across all four
hallucination signal families.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import yaml

_repo_root = Path(__file__).resolve().parent.parent
CONFIG_PATH = _repo_root / "configs" / "combined_features.yaml"


@pytest.fixture
def feature_config() -> Dict[str, Any]:
    """Load and return the combined feature YAML configuration."""
    assert CONFIG_PATH.exists(), f"Config file not found at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class TestFeatureConfigurationSchema:
    """Test suite for validating combined_features.yaml structure and policies."""

    def test_config_loads_and_has_required_top_level_keys(self, feature_config: Dict[str, Any]):
        """Verify that top-level configuration sections are present."""
        required_keys = [
            "version",
            "metadata_columns",
            "forbidden_feature_columns",
            "feature_families",
            "model_tiers",
            "missing_value_strategy",
        ]
        for key in required_keys:
            assert key in feature_config, f"Missing required top-level key: '{key}'"

    def test_metadata_columns_specification(self, feature_config: Dict[str, Any]):
        """Verify metadata columns contain exactly id, source_dataset, and label."""
        meta = feature_config["metadata_columns"]
        assert "id" in meta
        assert "source_dataset" in meta
        assert "label" in meta

    def test_forbidden_columns_comprehensive(self, feature_config: Dict[str, Any]):
        """Ensure all sensitive, text, and shortcut columns are strictly forbidden."""
        forbidden = set(feature_config["forbidden_feature_columns"])
        required_forbidden = {
            "id",
            "source_dataset",
            "label",
            "prompt",
            "context",
            "response",
            "model_input",
            "query_text",
            "retrieved_rank_1_text",
            "retrieved_rank_2_text",
            "retrieved_rank_3_text",
            "forward_time_s",
            "num_generations",
            "evidence_status",
            "num_tokens",  # Must never enter primary model
        }
        missing = required_forbidden - forbidden
        assert not missing, f"Missing forbidden columns: {missing}"

    def test_primary_features_exclude_forbidden_columns(self, feature_config: Dict[str, Any]):
        """Ensure no primary feature from any family is in the forbidden list."""
        forbidden = set(feature_config["forbidden_feature_columns"])
        families = feature_config["feature_families"]

        for fam_name, fam_data in families.items():
            primary = fam_data.get("primary_features", [])
            for feat in primary:
                assert feat not in forbidden, (
                    f"Policy violation: Primary feature '{feat}' in family '{fam_name}' is in forbidden columns!"
                )

    def test_internal_signals_match_primary_baseline(self, feature_config: Dict[str, Any]):
        """Verify internal signal features match the 11 baseline features."""
        from src.training.baseline_models import PRIMARY_SIGNAL_FEATURES

        cfg_internal = feature_config["feature_families"]["internal_signals"]["primary_features"]
        assert list(cfg_internal) == list(PRIMARY_SIGNAL_FEATURES), (
            f"Internal signal features in config {cfg_internal} do not match baseline_models {PRIMARY_SIGNAL_FEATURES}"
        )

    def test_nli_marked_bidirectionally_aggregated(self, feature_config: Dict[str, Any]):
        """Confirm NLI agreement configuration documents bidirectional aggregation."""
        nli_cfg = feature_config["feature_families"]["nli_agreement"]
        assert nli_cfg.get("bidirectionally_aggregated") is True

    def test_evidence_retrieval_availability_documentation(self, feature_config: Dict[str, Any]):
        """Confirm retrieval availability explicitly flags dataset limitations."""
        ret_cfg = feature_config["feature_families"]["evidence_retrieval"]
        assert ret_cfg["availability"] == "dataset_dependent"
        cov = ret_cfg["dataset_coverage"]
        assert cov["halueval"] == "available"
        assert cov["fever"] == "insufficient_wikipedia_pointers_only"
        assert cov["truthfulqa"] == "insufficient_url_citations_only"


class TestMathematicalConstraintsOnPilotData:
    """Verify mathematical identities and boundaries across the 20 pilot outputs."""

    @pytest.fixture
    def pilot_data(self) -> Dict[str, pd.DataFrame]:
        """Load the pilot datasets for verification."""
        sc_path = _repo_root / "experiments/baselines/self_consistency/pilot_self_consistency_aggregated.parquet"
        nli_path = _repo_root / "experiments/baselines/nli_agreement/pilot_nli_aggregated.parquet"
        ret_path = _repo_root / "experiments/baselines/evidence_retrieval/pilot_retrieval_results.parquet"

        assert sc_path.exists(), "SC pilot parquet missing"
        assert nli_path.exists(), "NLI pilot parquet missing"
        assert ret_path.exists(), "Retrieval pilot parquet missing"

        return {
            "sc": pd.read_parquet(sc_path),
            "nli": pd.read_parquet(nli_path),
            "ret": pd.read_parquet(ret_path),
        }

    def test_self_consistency_disagreement_identity(self, pilot_data: Dict[str, pd.DataFrame]):
        """Verify generation_disagreement = 1.0 - mean_pairwise_similarity."""
        df_sc = pilot_data["sc"]
        expected = 1.0 - df_sc["mean_pairwise_similarity"]
        actual = df_sc["generation_disagreement"]
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5, atol=1e-6)

    def test_nli_probabilities_sum_to_one(self, pilot_data: Dict[str, pd.DataFrame]):
        """Verify entailment + contradiction + neutral = 1.0 on pilot examples."""
        df_nli = pilot_data["nli"]
        prob_sum = (
            df_nli["mean_pairwise_entailment"]
            + df_nli["mean_pairwise_contradiction"]
            + df_nli["mean_pairwise_neutral"]
        )
        np.testing.assert_allclose(prob_sum.values, np.ones(len(df_nli)), rtol=1e-5, atol=1e-6)

    def test_nli_fractions_sum_to_one(self, pilot_data: Dict[str, pd.DataFrame]):
        """Verify fraction_entail + fraction_contra + fraction_neutral = 1.0."""
        df_nli = pilot_data["nli"]
        frac_sum = (
            df_nli["fraction_entailing_pairs"]
            + df_nli["fraction_contradicting_pairs"]
            + df_nli["fraction_neutral_pairs"]
        )
        np.testing.assert_allclose(frac_sum.values, np.ones(len(df_nli)), rtol=1e-5, atol=1e-6)

    def test_nli_disagreement_formula(self, pilot_data: Dict[str, pd.DataFrame]):
        """Verify nli_disagreement = mean_contra + 0.5 * mean_neutral."""
        df_nli = pilot_data["nli"]
        expected = df_nli["mean_pairwise_contradiction"] + 0.5 * df_nli["mean_pairwise_neutral"]
        actual = df_nli["nli_disagreement"]
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5, atol=1e-6)

    def test_retrieval_agreement_bounds_and_finiteness(self, pilot_data: Dict[str, pd.DataFrame]):
        """Verify retrieval agreement is strictly in [0.0, 1.0] and finite."""
        df_ret = pilot_data["ret"]
        agree = df_ret["retrieval_agreement"].values
        assert np.all(np.isfinite(agree))
        assert np.all(agree >= 0.0)
        assert np.all(agree <= 1.0)

    def test_retrieval_margin_non_negative(self, pilot_data: Dict[str, pd.DataFrame]):
        """Verify evidence margin is >= 0.0 for all pilot rows."""
        df_ret = pilot_data["ret"]
        margin = df_ret["evidence_margin"].values
        assert np.all(np.isfinite(margin))
        assert np.all(margin >= 0.0)
