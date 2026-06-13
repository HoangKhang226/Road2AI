"""Evaluate a results file against a hand-labelled local reference file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.evaluation import macro_retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("references", type=Path)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))
    print(json.dumps(macro_retrieval_metrics(predictions, references), indent=2))


if __name__ == "__main__":
    main()
