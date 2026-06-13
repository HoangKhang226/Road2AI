"""Validate results.json against the test set and create a flat submission.zip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.paths import DATA_DIR, OUTPUT_DIR
from aiguru.submission import validate_and_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=OUTPUT_DIR / "results.json")
    parser.add_argument("--questions", type=Path, default=DATA_DIR / "R2AIStage1DATA.json")
    parser.add_argument("--zip", type=Path, default=OUTPUT_DIR / "submission.zip")
    args = parser.parse_args()
    output = validate_and_package(args.results, args.questions, args.zip)
    print(f"Submission validated and packaged: {output}")


if __name__ == "__main__":
    main()
