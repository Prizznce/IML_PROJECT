"""
Dataset loading, parsing, balanced sampling, and caching pipeline.

Standardizes HaluEval, TruthfulQA, and FEVER into the canonical schema:
    - id
    - prompt
    - response
    - context
    - label (0 = Faithful/Supported, 1 = Hallucinated/Unsupported)
    - source_dataset

Preserves teammate's reference script (1.py) and provides reproducible,
modular functions suitable for lightweight semester project requirements.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("load_datasets")


# =====================================================================
# INDIVIDUAL PARSERS (Pure Functions for Robust Testing & Processing)
# =====================================================================

def parse_halueval_record(row: Dict[str, Any], record_idx: int = 0) -> CanonicalExample:
    """
    Parse a single row from flowaicom/HaluEval.

    Label mapping:
        - FAIL (score=0): Hallucinated / Unsupported -> 1
        - PASS (score=1): Grounded / Faithful -> 0
    """
    raw_label = str(row.get("label", "")).strip().upper()
    score = row.get("score")

    # Correctly identify hallucination vs faithful
    if raw_label in ("FAIL", "0") or score == 0:
        label = LABEL_HALLUCINATED
    elif raw_label in ("PASS", "1") or score == 1:
        label = LABEL_FAITHFUL
    else:
        # Fallback heuristic if string representation varies
        label = LABEL_HALLUCINATED if "FAIL" in raw_label else LABEL_FAITHFUL

    example_id = str(row.get("id", f"halueval_{record_idx}"))
    prompt = str(row.get("question", "")).strip()
    response = str(row.get("answer", "")).strip()
    context = str(row.get("passage", "")).strip()

    return CanonicalExample(
        id=example_id,
        prompt=prompt,
        response=response,
        context=context,
        label=label,
        source_dataset="halueval"
    )


def parse_truthfulqa_record(row: Dict[str, Any], record_idx: int = 0) -> List[CanonicalExample]:
    """
    Parse a single question from truthfulqa/truthful_qa (generation config).

    Unlike simplistic mappings that only label best_answer as 0 (which causes
    complete class collapse), this extracts:
        - best_answer / correct_answers -> label 0 (Faithful / Truthful)
        - incorrect_answers -> label 1 (Hallucinated / Untruthful)
    Each candidate answer is paired with the prompt question.
    """
    question = str(row.get("question", "")).strip()
    source = str(row.get("source", "")).strip()
    examples: List[CanonicalExample] = []

    # 1. Truthful answers (label = 0)
    seen_truthful = set()
    best_answer = str(row.get("best_answer", "")).strip()
    if best_answer:
        seen_truthful.add(best_answer.lower())
        examples.append(CanonicalExample(
            id=f"truthfulqa_{record_idx}_best",
            prompt=question,
            response=best_answer,
            context=source,
            label=LABEL_FAITHFUL,
            source_dataset="truthfulqa"
        ))

    correct_answers = row.get("correct_answers", [])
    if isinstance(correct_answers, list):
        for c_idx, ans in enumerate(correct_answers):
            ans_clean = str(ans).strip()
            if ans_clean and ans_clean.lower() not in seen_truthful:
                seen_truthful.add(ans_clean.lower())
                examples.append(CanonicalExample(
                    id=f"truthfulqa_{record_idx}_corr_{c_idx}",
                    prompt=question,
                    response=ans_clean,
                    context=source,
                    label=LABEL_FAITHFUL,
                    source_dataset="truthfulqa"
                ))

    # 2. Hallucinated / Untruthful answers (label = 1)
    seen_untruthful = set()
    incorrect_answers = row.get("incorrect_answers", [])
    if isinstance(incorrect_answers, list):
        for inc_idx, ans in enumerate(incorrect_answers):
            ans_clean = str(ans).strip()
            if ans_clean and ans_clean.lower() not in seen_untruthful:
                seen_untruthful.add(ans_clean.lower())
                examples.append(CanonicalExample(
                    id=f"truthfulqa_{record_idx}_inc_{inc_idx}",
                    prompt=question,
                    response=ans_clean,
                    context=source,
                    label=LABEL_HALLUCINATED,
                    source_dataset="truthfulqa"
                ))

    return examples


def parse_fever_record(
    row: Dict[str, Any],
    include_nei: bool = False
) -> Optional[CanonicalExample]:
    """
    Parse a single FEVER record.

    Label mapping:
        - SUPPORTS -> 0 (Supported / Faithful)
        - REFUTES  -> 1 (Refuted / Hallucinated fact)
        - NOT ENOUGH INFO -> Discarded by default in binary classification,
          as lack of documentation is distinct from fabricated contradiction.
    """
    raw_label = str(row.get("label", "")).strip().upper()
    claim = str(row.get("claim", "")).strip()
    evidence = row.get("evidence", [])
    example_id = f"fever_{row.get('id', 'unknown')}"

    if raw_label == "SUPPORTS":
        label = LABEL_FAITHFUL
    elif raw_label == "REFUTES":
        label = LABEL_HALLUCINATED
    elif raw_label == "NOT ENOUGH INFO":
        if include_nei:
            # If user explicitly chooses to include ungrounded/unverified claims
            label = LABEL_HALLUCINATED
        else:
            # Exclude ambiguous NEI to maintain clean binary supervision
            return None
    else:
        return None

    # Context format from evidence annotations
    context_str = json.dumps(evidence) if evidence else ""

    return CanonicalExample(
        id=example_id,
        prompt="Verify whether the following claim is supported by factual evidence.",
        response=claim,
        context=context_str,
        label=label,
        source_dataset="fever"
    )


# =====================================================================
# BALANCED SAMPLING HELPER
# =====================================================================

def balance_and_sample(
    df: pd.DataFrame,
    target_size: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    Sample a balanced subset containing equal numbers of faithful (0)
    and hallucinated (1) instances.
    """
    if df.empty:
        return df

    df_0 = df[df["label"] == LABEL_FAITHFUL]
    df_1 = df[df["label"] == LABEL_HALLUCINATED]

    if df_0.empty or df_1.empty:
        logger.warning(
            "Dataset does not have both classes (class 0: %d, class 1: %d). "
            "Returning uniform random sample.",
            len(df_0), len(df_1)
        )
        sample_n = min(len(df), target_size)
        return df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    half_target = target_size // 2
    n_0 = min(len(df_0), half_target)
    n_1 = min(len(df_1), half_target)

    # Balance evenly
    sample_per_class = min(n_0, n_1)
    sampled_0 = df_0.sample(n=sample_per_class, random_state=seed)
    sampled_1 = df_1.sample(n=sample_per_class, random_state=seed)

    balanced = pd.concat([sampled_0, sampled_1], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =====================================================================
# LOADERS WITH OFFLINE CACHING
# =====================================================================

def load_config(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Load YAML dataset configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_process_halueval(
    sample_size: int = 1500,
    seed: int = 42,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    target_file: str = "halueval_processed.parquet"
) -> pd.DataFrame:
    """
    Load, parse, balance, and cache flowaicom/HaluEval.
    """
    from datasets import load_dataset

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(processed_dir) / target_file

    logger.info("Loading HaluEval from Hugging Face ('flowaicom/HaluEval')...")
    ds = load_dataset("flowaicom/HaluEval", split="test")

    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        canonical = parse_halueval_record(row, record_idx=idx)
        records.append(canonical.to_dict())

    df = pd.DataFrame(records)
    logger.info(
        "HaluEval parsed: %d total rows (Class 0: %d, Class 1: %d)",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum()
    )

    balanced_df = balance_and_sample(df, target_size=sample_size, seed=seed)
    validate_canonical_dataframe(balanced_df)

    balanced_df.to_parquet(out_path, index=False)
    logger.info("Saved %d balanced HaluEval examples to %s", len(balanced_df), out_path)
    return balanced_df


def load_and_process_truthfulqa(
    sample_size: int = 1500,
    seed: int = 42,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    target_file: str = "truthfulqa_processed.parquet"
) -> pd.DataFrame:
    """
    Load, parse, balance, and cache truthfulqa/truthful_qa.
    """
    from datasets import load_dataset

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(processed_dir) / target_file

    logger.info("Loading TruthfulQA from Hugging Face ('truthfulqa/truthful_qa', 'generation')...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        for canonical in parse_truthfulqa_record(row, record_idx=idx):
            records.append(canonical.to_dict())

    df = pd.DataFrame(records)
    logger.info(
        "TruthfulQA parsed: %d candidate rows (Class 0: %d, Class 1: %d)",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum()
    )

    balanced_df = balance_and_sample(df, target_size=sample_size, seed=seed)
    validate_canonical_dataframe(balanced_df)

    balanced_df.to_parquet(out_path, index=False)
    logger.info("Saved %d balanced TruthfulQA examples to %s", len(balanced_df), out_path)
    return balanced_df


def load_and_process_fever(
    sample_size: int = 1000,
    seed: int = 42,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    include_nei: bool = False,
    url: str = "https://fever.ai/download/fever/shared_task_dev.jsonl",
    target_file: str = "fever_processed.parquet"
) -> pd.DataFrame:
    """
    Load, parse, balance, and cache a lightweight subset of FEVER from dev set.
    """
    import urllib.request

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(processed_dir) / target_file
    raw_jsonl_path = Path(raw_dir) / "fever_shared_task_dev.jsonl"

    if not raw_jsonl_path.exists():
        logger.info("Downloading FEVER dev set from %s...", url)
        urllib.request.urlretrieve(url, raw_jsonl_path)
        logger.info("Cached raw FEVER dev set to %s", raw_jsonl_path)

    records: List[Dict[str, Any]] = []
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            canonical = parse_fever_record(row, include_nei=include_nei)
            if canonical is not None:
                records.append(canonical.to_dict())

    df = pd.DataFrame(records)
    logger.info(
        "FEVER parsed: %d candidate rows (Class 0: %d, Class 1: %d)",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum()
    )

    balanced_df = balance_and_sample(df, target_size=sample_size, seed=seed)
    validate_canonical_dataframe(balanced_df)

    balanced_df.to_parquet(out_path, index=False)
    logger.info("Saved %d balanced FEVER examples to %s", len(balanced_df), out_path)
    return balanced_df


def run_pipeline(config_path: str = "configs/data.yaml") -> Dict[str, pd.DataFrame]:
    """Execute the full dataset pipeline according to the configuration file."""
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    paths = cfg.get("paths", {})
    raw_dir = paths.get("raw_dir", "data/raw")
    processed_dir = paths.get("processed_dir", "data/processed")

    results: Dict[str, pd.DataFrame] = {}

    # HaluEval
    h_cfg = cfg["datasets"]["halueval"]
    results["halueval"] = load_and_process_halueval(
        sample_size=h_cfg.get("sample_size", 1500),
        seed=seed,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        target_file=h_cfg.get("target_file", "halueval_processed.parquet")
    )

    # TruthfulQA
    t_cfg = cfg["datasets"]["truthfulqa"]
    results["truthfulqa"] = load_and_process_truthfulqa(
        sample_size=t_cfg.get("sample_size", 1500),
        seed=seed,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        target_file=t_cfg.get("target_file", "truthfulqa_processed.parquet")
    )

    # FEVER
    f_cfg = cfg["datasets"]["fever"]
    results["fever"] = load_and_process_fever(
        sample_size=f_cfg.get("sample_size", 1000),
        seed=seed,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        include_nei=f_cfg.get("include_not_enough_info", False),
        url=f_cfg.get("url", "https://fever.ai/download/fever/shared_task_dev.jsonl"),
        target_file=f_cfg.get("target_file", "fever_processed.parquet")
    )

    # Combined master dataset
    combined = pd.concat(list(results.values()), ignore_index=True)
    combined_file = paths.get("combined_file", "combined_processed.parquet")
    combined_path = Path(processed_dir) / combined_file
    combined.to_parquet(combined_path, index=False)
    logger.info("Combined master dataset saved to %s (%d total rows)", combined_path, len(combined))

    results["combined"] = combined
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Dataset Processing Pipeline")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to YAML config")
    parser.add_argument("--download", action="store_true", help="Execute download and processing")
    args = parser.parse_args()

    if args.download:
        print("Starting dataset download and processing pipeline...")
        run_pipeline(args.config)
        print("Dataset pipeline processing complete.")
    else:
        print("Dry run / validation mode. Use --download to execute pipeline.")
