"""
Unit tests for Phase 3 Dataset Pipeline.

Verifies schema adherence, record parsers, balanced sampling,
and ensures no single-class collapse occurs for HaluEval or TruthfulQA.
All tests use mock in-memory data to avoid network overhead.
"""
import sys
import unittest
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import (
    CanonicalExample,
    CANONICAL_COLUMNS,
    LABEL_FAITHFUL,
    LABEL_HALLUCINATED,
    validate_canonical_dataframe,
    create_empty_canonical_df,
)
from src.data.load_datasets import (
    parse_halueval_record,
    parse_truthfulqa_record,
    parse_fever_record,
    balance_and_sample,
    load_config,
)


class TestCanonicalSchema(unittest.TestCase):
    def test_canonical_example_creation(self):
        ex = CanonicalExample(
            id="test_1",
            prompt="What is the capital of France?",
            response="Paris",
            context="France is in Europe.",
            label=0,
            source_dataset="test_source"
        )
        self.assertEqual(ex.id, "test_1")
        self.assertEqual(ex.prompt, "What is the capital of France?")
        self.assertEqual(ex.response, "Paris")
        self.assertEqual(ex.context, "France is in Europe.")
        self.assertEqual(ex.label, 0)
        self.assertEqual(ex.source_dataset, "test_source")

    def test_canonical_example_invalid_label(self):
        with self.assertRaises(ValueError):
            CanonicalExample(
                id="test_2",
                prompt="Prompt",
                response="Response",
                label=99,  # Invalid label
                source_dataset="test"
            )

    def test_validate_canonical_dataframe_success(self):
        df = pd.DataFrame([{
            "id": "1",
            "prompt": "Q?",
            "response": "A.",
            "context": "C.",
            "label": 0,
            "source_dataset": "demo"
        }])
        self.assertTrue(validate_canonical_dataframe(df))

    def test_validate_canonical_dataframe_missing_column(self):
        df = pd.DataFrame([{
            "id": "1",
            "prompt": "Q?",
            # missing response
            "context": "C.",
            "label": 0,
            "source_dataset": "demo"
        }])
        with self.assertRaises(ValueError):
            validate_canonical_dataframe(df)

    def test_validate_canonical_dataframe_empty_prompt(self):
        df = pd.DataFrame([{
            "id": "1",
            "prompt": "   ",  # Whitespace only
            "response": "A.",
            "context": "",
            "label": 0,
            "source_dataset": "demo"
        }])
        with self.assertRaises(ValueError):
            validate_canonical_dataframe(df)


class TestDatasetParsers(unittest.TestCase):
    def test_halueval_parser_fail_label(self):
        mock_row = {
            "id": "halu_001",
            "question": "Did Neil Armstrong land on Mars?",
            "answer": "Yes, Neil Armstrong walked on Mars in 1969.",
            "passage": "Neil Armstrong was an American astronaut who landed on the Moon.",
            "label": "FAIL",
            "score": 0,
            "source_ds": "qa"
        }
        canonical = parse_halueval_record(mock_row, record_idx=1)
        self.assertEqual(canonical.id, "halu_001")
        self.assertEqual(canonical.prompt, "Did Neil Armstrong land on Mars?")
        self.assertEqual(canonical.response, "Yes, Neil Armstrong walked on Mars in 1969.")
        self.assertEqual(canonical.label, LABEL_HALLUCINATED)
        self.assertEqual(canonical.source_dataset, "halueval")

    def test_halueval_parser_pass_label(self):
        mock_row = {
            "id": "halu_002",
            "question": "Did Neil Armstrong land on the Moon?",
            "answer": "Yes, Neil Armstrong landed on the Moon in 1969.",
            "passage": "Neil Armstrong was an astronaut who walked on the Moon.",
            "label": "PASS",
            "score": 1,
            "source_ds": "qa"
        }
        canonical = parse_halueval_record(mock_row, record_idx=2)
        self.assertEqual(canonical.label, LABEL_FAITHFUL)
        self.assertEqual(canonical.source_dataset, "halueval")

    def test_halueval_no_single_class_collapse(self):
        batch = [
            {"id": "1", "question": "Q1", "answer": "A1", "label": "FAIL", "score": 0},
            {"id": "2", "question": "Q2", "answer": "A2", "label": "PASS", "score": 1},
        ]
        parsed = [parse_halueval_record(r, i).to_dict() for i, r in enumerate(batch)]
        df = pd.DataFrame(parsed)
        labels = set(df["label"].unique())
        self.assertEqual(labels, {0, 1}, "HaluEval must produce both classes 0 and 1.")

    def test_truthfulqa_parser_candidate_extraction(self):
        mock_row = {
            "question": "What happens if you eat watermelon seeds?",
            "best_answer": "Nothing happens; the seeds pass through your digestive system.",
            "correct_answers": [
                "Nothing happens.",
                "You simply digest or pass them."
            ],
            "incorrect_answers": [
                "A watermelon will grow inside your stomach.",
                "You will get appendicitis immediately."
            ],
            "source": "https://en.wikipedia.org/wiki/Watermelon"
        }
        examples = parse_truthfulqa_record(mock_row, record_idx=0)

        # Ensure candidates were produced
        self.assertGreater(len(examples), 1)

        labels = [ex.label for ex in examples]
        self.assertIn(LABEL_FAITHFUL, labels, "Must have faithful (0) candidates.")
        self.assertIn(LABEL_HALLUCINATED, labels, "Must have hallucinated (1) candidates.")

        # Ensure prompt is preserved across all candidates
        for ex in examples:
            self.assertEqual(ex.prompt, "What happens if you eat watermelon seeds?")
            self.assertEqual(ex.source_dataset, "truthfulqa")
            self.assertGreater(len(ex.response), 0)

    def test_truthfulqa_no_single_class_collapse(self):
        mock_row = {
            "question": "Can you sneeze with your eyes open?",
            "best_answer": "Yes, some people can sneeze with their eyes open.",
            "correct_answers": ["Yes, it is possible."],
            "incorrect_answers": ["No, your eyeballs will pop out."],
            "source": "https://medicalnews.com"
        }
        examples = parse_truthfulqa_record(mock_row, record_idx=0)
        df = pd.DataFrame([e.to_dict() for e in examples])
        labels = set(df["label"].unique())
        self.assertEqual(labels, {0, 1}, "TruthfulQA must contain both label 0 and 1.")

    def test_fever_parser(self):
        mock_supports = {
            "id": 101,
            "claim": "The Earth orbits the Sun.",
            "label": "SUPPORTS",
            "evidence": [[1, 2, "Earth", 4]]
        }
        ex_supports = parse_fever_record(mock_supports)
        self.assertIsNotNone(ex_supports)
        self.assertEqual(ex_supports.label, LABEL_FAITHFUL)
        self.assertEqual(ex_supports.source_dataset, "fever")

        mock_refutes = {
            "id": 102,
            "claim": "The Moon is made of green cheese.",
            "label": "REFUTES",
            "evidence": [[1, 2, "Moon", 8]]
        }
        ex_refutes = parse_fever_record(mock_refutes)
        self.assertIsNotNone(ex_refutes)
        self.assertEqual(ex_refutes.label, LABEL_HALLUCINATED)

        mock_nei = {
            "id": 103,
            "claim": "Aliens exist on Mars.",
            "label": "NOT ENOUGH INFO",
            "evidence": []
        }
        # By default, NEI is discarded to maintain clean binary supervision
        ex_nei_default = parse_fever_record(mock_nei, include_nei=False)
        self.assertIsNone(ex_nei_default)

        # When explicitly enabled
        ex_nei_included = parse_fever_record(mock_nei, include_nei=True)
        self.assertIsNotNone(ex_nei_included)
        self.assertEqual(ex_nei_included.label, LABEL_HALLUCINATED)


class TestBalancedSampling(unittest.TestCase):
    def test_balance_and_sample(self):
        # Create imbalanced synthetic dataset
        records = []
        for i in range(100):
            records.append({
                "id": f"ex_{i}",
                "prompt": f"Prompt {i}",
                "response": f"Response {i}",
                "context": "",
                "label": 0 if i < 90 else 1,  # 90 vs 10
                "source_dataset": "synthetic"
            })
        df = pd.DataFrame(records)
        balanced = balance_and_sample(df, target_size=16, seed=42)

        count_0 = (balanced["label"] == 0).sum()
        count_1 = (balanced["label"] == 1).sum()

        self.assertEqual(count_0, count_1)
        self.assertEqual(count_0 + count_1, 16)
        self.assertTrue(validate_canonical_dataframe(balanced))


class TestDataConfiguration(unittest.TestCase):
    def test_data_yaml_exists_and_valid(self):
        config_path = Path("configs/data.yaml")
        self.assertTrue(config_path.exists(), "configs/data.yaml must exist.")
        cfg = load_config(str(config_path))
        self.assertIn("seed", cfg)
        self.assertIn("paths", cfg)
        self.assertIn("datasets", cfg)
        self.assertIn("halueval", cfg["datasets"])
        self.assertIn("truthfulqa", cfg["datasets"])
        self.assertIn("fever", cfg["datasets"])


if __name__ == "__main__":
    unittest.main()
