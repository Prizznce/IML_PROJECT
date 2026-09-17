"""
Unit tests for probability calibration analysis module.

Tests binning partitioning, ECE, MCE, Brier score, empty bin handling,
and reliability diagram generation on synthetic data without GPU requirements.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.calibration import (
    compute_calibration_data,
    plot_reliability_diagram,
)


class TestCalibration(unittest.TestCase):
    """Test suite for probability calibration metrics and reliability binning."""

    def test_perfect_calibration(self):
        """Verify that perfectly calibrated probabilities yield near-zero ECE and MCE."""
        # 100 samples in bin [0.0, 0.1) with mean 0.05 and 5 positives -> acc=0.05
        # 100 samples in bin [0.9, 1.0] with mean 0.95 and 95 positives -> acc=0.95
        y_prob = np.array([0.05] * 100 + [0.95] * 100)
        y_true = np.array([0] * 95 + [1] * 5 + [0] * 5 + [1] * 95)

        res = compute_calibration_data(y_true, y_prob, n_bins=10)

        self.assertAlmostEqual(res["ece"], 0.0, places=4)
        self.assertAlmostEqual(res["mce"], 0.0, places=4)
        self.assertEqual(res["total_samples"], 200)

    def test_complete_miscalibration(self):
        """Verify that inverted probabilities yield ECE = 1.0 and MCE = 1.0."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.array([1.0] * 50 + [0.0] * 50)

        res = compute_calibration_data(y_true, y_prob, n_bins=10)

        self.assertAlmostEqual(res["ece"], 1.0, places=4)
        self.assertAlmostEqual(res["mce"], 1.0, places=4)

    def test_bin_counts_sum_to_total_samples(self):
        """Verify that sum of sample counts across all bins equals dataset size."""
        np.random.seed(42)
        n = 800
        y_true = np.random.randint(0, 2, n)
        y_prob = np.random.uniform(0.0, 1.0, n)

        res = compute_calibration_data(y_true, y_prob, n_bins=10)

        bin_counts = [b["sample_count"] for b in res["bins"]]
        self.assertEqual(sum(bin_counts), n)
        self.assertEqual(len(res["bins"]), 10)
        self.assertEqual(res["total_samples"], n)

        # Check bin boundaries
        self.assertEqual(res["bins"][0]["bin_lower"], 0.0)
        self.assertEqual(res["bins"][-1]["bin_upper"], 1.0)

    def test_empty_bins_handling(self):
        """Verify that empty bins are handled gracefully without NaN gaps."""
        # All probabilities within [0.4, 0.5)
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.42, 0.45, 0.48, 0.49])

        res = compute_calibration_data(y_true, y_prob, n_bins=10)

        self.assertEqual(res["non_empty_bins"], 1)
        # Empty bins should have sample_count = 0, mean_predicted_probability = None
        for b in res["bins"]:
            if b["bin_index"] == 4:
                self.assertEqual(b["sample_count"], 4)
                self.assertIsNotNone(b["mean_predicted_probability"])
            else:
                self.assertEqual(b["sample_count"], 0)
                self.assertIsNone(b["mean_predicted_probability"])
                self.assertEqual(b["absolute_calibration_gap"], 0.0)

        self.assertGreater(res["ece"], 0.0)
        self.assertGreater(res["mce"], 0.0)

    def test_brier_score_matches_sklearn(self):
        """Verify that Brier score matches scikit-learn standard implementation."""
        np.random.seed(123)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0.0, 1.0, 100)

        res = compute_calibration_data(y_true, y_prob, n_bins=10)
        expected_brier = brier_score_loss(y_true, y_prob)

        self.assertAlmostEqual(res["brier_score"], expected_brier, places=5)

    def test_reliability_diagram_plotting(self):
        """Verify that plot_reliability_diagram executes and produces a valid image file."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0.0, 1.0, 200)

        calib_data = compute_calibration_data(y_true, y_prob, n_bins=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_img = Path(tmpdir) / "test_reliability.png"
            plot_reliability_diagram(calib_data, "Test Model", out_img)

            self.assertTrue(out_img.exists())
            self.assertGreater(out_img.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
