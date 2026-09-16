"""
Canonical schema definition and validation for hallucination detection dataset pipeline.

All datasets (HaluEval, TruthfulQA, FEVER, etc.) are standardized to this schema
before feature extraction and model training.
"""
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import pandas as pd

# Canonical label definitions
LABEL_FAITHFUL = 0        # Faithful, grounded, supported, truthful
LABEL_HALLUCINATED = 1    # Hallucinated, unsupported, refuted, fabricated

CANONICAL_COLUMNS = [
    "id",
    "prompt",
    "response",
    "context",
    "label",
    "source_dataset"
]


@dataclass
class CanonicalExample:
    """Represents a single standardized example in the hallucination detection pipeline."""
    id: str
    prompt: str
    response: str
    context: str = ""
    label: int = LABEL_FAITHFUL
    source_dataset: str = ""

    def __post_init__(self):
        self.id = str(self.id)
        self.prompt = str(self.prompt).strip()
        self.response = str(self.response).strip()
        self.context = str(self.context).strip() if self.context else ""
        self.label = int(self.label)
        self.source_dataset = str(self.source_dataset).strip()

        if self.label not in (LABEL_FAITHFUL, LABEL_HALLUCINATED):
            raise ValueError(
                f"Invalid label {self.label}. Label must be {LABEL_FAITHFUL} (faithful) "
                f"or {LABEL_HALLUCINATED} (hallucinated)."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_empty_canonical_df() -> pd.DataFrame:
    """Create an empty DataFrame with the canonical schema and dtypes."""
    return pd.DataFrame({
        "id": pd.Series(dtype="str"),
        "prompt": pd.Series(dtype="str"),
        "response": pd.Series(dtype="str"),
        "context": pd.Series(dtype="str"),
        "label": pd.Series(dtype="int64"),
        "source_dataset": pd.Series(dtype="str"),
    })


def validate_canonical_dataframe(df: pd.DataFrame, allow_empty: bool = False) -> bool:
    """
    Validate that a DataFrame conforms strictly to the canonical schema.

    Checks:
    - All CANONICAL_COLUMNS are present.
    - No NaN values in prompt, response, label, source_dataset.
    - label values are strictly {0, 1}.
    - prompt and response are non-empty strings.
    """
    # Check column presence
    missing_cols = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required canonical columns: {missing_cols}")

    if df.empty:
        if allow_empty:
            return True
        raise ValueError("DataFrame is empty.")

    # Check for NaN in critical columns
    critical_cols = ["id", "prompt", "response", "label", "source_dataset"]
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Column '{col}' contains {null_count} null/NaN values.")

    # Check label values
    invalid_labels = set(df["label"].unique()) - {LABEL_FAITHFUL, LABEL_HALLUCINATED}
    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {invalid_labels}. Allowed: {{0, 1}}"
        )

    # Check non-empty prompt and response
    empty_prompts = (df["prompt"].astype(str).str.strip() == "").sum()
    if empty_prompts > 0:
        raise ValueError(f"Found {empty_prompts} rows with empty 'prompt'.")

    empty_responses = (df["response"].astype(str).str.strip() == "").sum()
    if empty_responses > 0:
        raise ValueError(f"Found {empty_responses} rows with empty 'response'.")

    return True
