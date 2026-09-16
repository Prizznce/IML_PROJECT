"""
Catching an LLM Lying: A Lightweight Classifier for Real-Time Hallucination
Detection from Internal Generation Signals

Entry point for project verification and initialization checks.
"""
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print("=" * 60)
    print("Catching an LLM Lying")
    print("A Lightweight Classifier for Real-Time Hallucination Detection")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Project Root  : {project_root}")
    print("Project initialization successful.")
    print("=" * 60)


if __name__ == "__main__":
    main()
