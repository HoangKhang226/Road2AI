"""Run the complete competition pipeline as isolated crash-safe phases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--index-batch-size", type=int, default=512)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--reset-index", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()
    if sys.version_info >= (3, 13):
        raise RuntimeError("Use Python 3.10-3.12 for the full pipeline.")
    python = sys.executable

    if not args.skip_collect:
        run([python, "scripts/run_phase1_collect.py"])
        run([python, "scripts/run_phase1_chunk.py"])
    if not args.skip_index:
        run([python, "scripts/run_phase2_build_bm25.py"])
        build_index = [
            python,
            "scripts/run_phase2_build_qdrant.py",
            "--device",
            args.device,
            "--batch-size",
            str(args.index_batch_size),
        ]
        if args.reset_index:
            build_index.append("--reset")
        run(build_index)
    retrieve = [
        python,
        "scripts/run_phase2_retrieve.py",
        "--embedding-device",
        args.device,
        "--reranker-device",
        args.device,
    ]
    if args.no_reranker:
        retrieve.append("--no-reranker")
    fresh_upstream = not args.skip_collect or not args.skip_index
    if fresh_upstream:
        retrieve.append("--reset")
    run(retrieve)
    generate = [
        python,
        "scripts/run_phase3_4.py",
        "data/R2AIStage1DATA.json",
        "--batch-size",
        str(args.generation_batch_size),
    ]
    if fresh_upstream:
        generate.append("--reset")
    run(generate)
    run([python, "scripts/validate_submission.py"])


if __name__ == "__main__":
    main()
