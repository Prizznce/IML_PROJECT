"""
Unit tests for Universal Combined Feature Table Builder.

Verifies:
- Feature set definitions and count invariants (exactly 19 primary features, 3 metadata)
- Log transformation math matching baseline_models.py
- Join and assembly mechanics strictly on exact ID
- Automated validation checks (shape, uniqueness, NaN/Inf, bounds, variance, collinearity)
- Strict exclusion of forbidden columns (num_tokens, text payloads, collinear ablation features)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_universal_features import (
    METADATA_COLUMNS,
    INTERNAL_SIGNAL_FEATURES,
    SELF_CONSISTENCY_FEATURES,
    NLI_AGREEMENT_FEATURES,
    UNIVERSAL_CORE_FEATURES,
    FORBIDDEN_FEATURE_COLUMNS,
    compute_internal_log_transforms,
    build_universal_feature_dataframe,
    validate_universal_feature_dataframe,
    build_and_save_universal_features,
)


class TestUniversalFeatureDefinitions(unittest.TestCase):
    """Test feature definitions and schema invariants."""

    def test_feature_counts_and_disjointness(self):
        self.assertEqual(len(INTERNAL_SIGNAL_FEATURES), 11)
        self.assertEqual(len(SELF_CONSISTENCY_FEATURES), 5)
        self.assertEqual(len(NLI_AGREEMENT_FEATURES), 3)
        self.assertEqual(len(UNIVERSAL_CORE_FEATURES), 19)
        self.assertEqual(len(METADATA_COLUMNS), 3)

        # Ensure no overlap between feature groups
        set_internal = set(INTERNAL_SIGNAL_FEATURES)
        set_sc = set(SELF_CONSISTENCY_FEATURES)
        set_nli = set(NLI_AGREEMENT_FEATURES)
        self.assertEqual(len(set_internal.intersection(set_sc)), 0)
        self.assertEqual(len(set_internal.intersection(set_nli)), 0)
        self.assertEqual(len(set_sc.intersection(set_nli)), 0)

    def test_forbidden_columns_policy(self):
        set_features = set(UNIVERSAL_CORE_FEATURES)
        # Verify forbidden columns are strictly absent
        overlap = set_features.intersection(FORBIDDEN_FEATURE_COLUMNS)
        self.assertEqual(len(overlap), 0, f"Forbidden columns present in features: {overlap}")
        self.assertNotIn("num_tokens", set_features)
        self.assertNotIn("unique_response_ratio", set_features)
        self.assertNotIn("majority_response_fraction", set_features)
        self.assertNotIn("generation_disagreement", set_features)
        self.assertNotIn("mean_pairwise_neutral", set_features)

    def test_internal_log_transforms(self):
        mock_df = pd.DataFrame({
            "perplexity": [10.0, 1.0, 0.0],
            "mean_token_rank": [5.0, 0.0, 2.0],
            "max_token_rank": [20.0, 1.0, 3.0],
            "rank_std": [2.0, 0.0, 1.0],
        })
        transformed = compute_internal_log_transforms(mock_df)

        self.assertAlmostEqual(transformed["log_perplexity"].iloc[0], np.log(10.0))
        self.assertAlmostEqual(transformed["log_perplexity"].iloc[1], 0.0)
        # Perplexity 0.0 clamped to 1e-12
        self.assertAlmostEqual(transformed["log_perplexity"].iloc[2], np.log(1e-12))

        self.assertAlmostEqual(transformed["log_mean_token_rank"].iloc[0], np.log1p(5.0))
        self.assertAlmostEqual(transformed["log_max_token_rank"].iloc[0], np.log1p(20.0))
        self.assertAlmostEqual(transformed["log_rank_std"].iloc[0], np.log1p(2.0))


class TestUniversalFeatureAssembly(unittest.TestCase):
    """Test table building, merging, and validation assertions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

        n = 5
        ids = [f"ex_{i+1:03d}" for i in range(n)]

        self.mock_internal = pd.DataFrame({
            "id": ids,
            "source_dataset": ["halueval", "halueval", "truthfulqa", "fever", "fever"],
            "label": [0, 1, 0, 1, 0],
            "prompt": ["Q"] * n,
            "context": ["C"] * n,
            "response": ["R"] * n,
            "model_input": ["M"] * n,
            "forward_time_s": [0.1] * n,
            "num_tokens": [10] * n,
            "min_log_prob": [-3.5, -2.1, -4.0, -1.8, -2.9],
            "mean_log_prob": [-0.5, -0.3, -0.7, -0.2, -0.4],
            "mean_token_prob": [0.65, 0.82, 0.58, 0.91, 0.73],
            "token_prob_std": [0.15, 0.10, 0.22, 0.08, 0.14],
            "mean_entropy": [0.55, 0.42, 0.88, 0.25, 0.49],
            "max_entropy": [1.45, 1.25, 1.92, 0.95, 1.33],
            "entropy_std": [0.21, 0.14, 0.27, 0.09, 0.17],
            "perplexity": [3.2, 4.1, 2.3, 5.2, 3.7],
            "mean_token_rank": [1.2, 0.7, 1.6, 0.4, 1.1],
            "max_token_rank": [8.2, 4.9, 10.1, 2.8, 7.3],
            "min_token_rank": [0, 0, 0, 0, 0],
            "rank_std": [1.15, 0.85, 1.45, 0.55, 1.05],
        })

        self.mock_sc = pd.DataFrame({
            "id": ids,
            "source_dataset": ["halueval", "halueval", "truthfulqa", "fever", "fever"],
            "original_label": [0, 1, 0, 1, 0],
            "num_generations": [5] * n,
            "exact_match_agreement": [0.8, 0.4, 0.0, 0.2, 0.6],
            "unique_response_ratio": [0.4] * n,
            "majority_response_fraction": [0.6] * n,
            "mean_pairwise_similarity": [0.85, 0.70, 0.65, 0.80, 0.75],
            "min_pairwise_similarity": [0.75, 0.50, 0.40, 0.60, 0.55],
            "max_pairwise_similarity": [0.95, 0.90, 0.85, 0.92, 0.90],
            "pairwise_similarity_std": [0.06, 0.12, 0.14, 0.09, 0.11],
            "generation_disagreement": [0.15] * n,
        })

        self.mock_nli = pd.DataFrame({
            "id": ids,
            "source_dataset": ["halueval", "halueval", "truthfulqa", "fever", "fever"],
            "original_label": [0, 1, 0, 1, 0],
            "num_generations": [5] * n,
            "mean_pairwise_entailment": [0.60, 0.30, 0.10, 0.20, 0.45],
            "mean_pairwise_contradiction": [0.15, 0.35, 0.40, 0.30, 0.20],
            "mean_pairwise_neutral": [0.25, 0.35, 0.50, 0.50, 0.35],
            "fraction_entailing_pairs": [0.6] * n,
            "fraction_contradicting_pairs": [0.2] * n,
            "fraction_neutral_pairs": [0.2] * n,
            "nli_disagreement": [0.15 + 0.5 * 0.25, 0.35 + 0.5 * 0.35, 0.40 + 0.5 * 0.50, 0.30 + 0.5 * 0.50, 0.20 + 0.5 * 0.35],
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_universal_feature_dataframe_success(self):
        df_universal = build_universal_feature_dataframe(
            df_internal=self.mock_internal,
            df_self_consistency=self.mock_sc,
            df_nli=self.mock_nli,
        )

        self.assertEqual(len(df_universal), 5)
        self.assertEqual(len(df_universal.columns), 22)  # 3 metadata + 19 features
        self.assertEqual(df_universal["id"].tolist(), ["ex_001", "ex_002", "ex_003", "ex_004", "ex_005"])

        # Validate with assertion suite
        summary = validate_universal_feature_dataframe(
            df=df_universal,
            df_orig=self.mock_internal,
            expected_count=5,
        )
        self.assertEqual(summary["feature_count"], 19)
        self.assertEqual(summary["missingness_rate"], 0.0)

    def test_id_mismatch_raises_error(self):
        bad_sc = self.mock_sc.copy()
        bad_sc.loc[0, "id"] = "different_id"

        with self.assertRaises(ValueError):
            build_universal_feature_dataframe(
                df_internal=self.mock_internal,
                df_self_consistency=bad_sc,
                df_nli=self.mock_nli,
            )

    def test_validation_catches_nan(self):
        df_universal = build_universal_feature_dataframe(
            df_internal=self.mock_internal,
            df_self_consistency=self.mock_sc,
            df_nli=self.mock_nli,
        )
        df_universal.loc[0, "mean_token_prob"] = np.nan

        with self.assertRaises(AssertionError):
            validate_universal_feature_dataframe(df_universal, expected_count=5)

    def test_validation_catches_forbidden_column(self):
        df_universal = build_universal_feature_dataframe(
            df_internal=self.mock_internal,
            df_self_consistency=self.mock_sc,
            df_nli=self.mock_nli,
        )
        df_universal["num_tokens"] = [10] * 5

        with self.assertRaises(AssertionError):
            validate_universal_feature_dataframe(df_universal, expected_count=5)


if __name__ == "__main__":
    unittest.main()
