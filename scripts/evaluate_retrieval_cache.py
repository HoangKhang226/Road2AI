"""Evaluate retrieval cache on questions containing explicit Điều X labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.evaluation import silver_retrieval_recall
from aiguru.paths import DATA_DIR, KNOWLEDGE_DIR


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DATA_DIR / "R2AIStage1DATA.json")
    parser.add_argument("--cache", type=Path, default=KNOWLEDGE_DIR / "retrieval_results.jsonl")
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    print(json.dumps(silver_retrieval_recall(questions, read_jsonl(args.cache)), indent=2))


if __name__ == "__main__":
    main()
