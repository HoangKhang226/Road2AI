"""Run Phase 3-4 with Qdrant hybrid retrieval and batched Unsloth inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.paths import KNOWLEDGE_DIR, OUTPUT_DIR
from aiguru.phase2.cache import RetrievalCache
from aiguru.phase3.config import GenerationConfig
from aiguru.phase3.generator import UnslothGenerator
from aiguru.phase4.pipeline import run_generation_pipeline
from aiguru.phase4.postprocess import PostProcessConfig, PostProcessor
from aiguru.phase4.streaming import JsonlResultStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Competition test JSON file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "results_partial.jsonl")
    parser.add_argument("--results-json", type=Path, default=OUTPUT_DIR / "results.json")
    parser.add_argument("--model", default=GenerationConfig().model_name)
    parser.add_argument("--batch-size", type=int, default=GenerationConfig().batch_size)
    parser.add_argument("--retrieval-top-k", type=int, default=25)
    parser.add_argument("--safe-threshold", type=float, default=0.3)
    parser.add_argument("--high-conf-threshold", type=float, default=0.5)
    parser.add_argument("--min-articles", type=int, default=3)
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--retrieval-cache",
        type=Path,
        default=KNOWLEDGE_DIR / "retrieval_results.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if args.reset:
        for path in (args.output, args.results_json):
            if path.exists():
                path.unlink()

    retriever = RetrievalCache(args.retrieval_cache)
    missing = {int(question["id"]) for question in questions} - retriever.completed_ids
    if missing:
        raise RuntimeError(
            f"Retrieval cache is missing {len(missing)} question(s). "
            "Run scripts/run_phase2_retrieve.py first."
        )

    # Load Unsloth only after the lightweight cache check, and before any
    # Transformers-based model in this process.
    config = GenerationConfig(model_name=args.model, batch_size=args.batch_size)
    generator = UnslothGenerator.from_pretrained(config)

    store = JsonlResultStore(args.output)
    written = run_generation_pipeline(
        questions=questions,
        retriever=retriever,
        generator=generator,
        postprocessor=PostProcessor(PostProcessConfig(
            safe_threshold=args.safe_threshold,
            high_conf_threshold=args.high_conf_threshold,
            min_high_conf_articles=args.min_articles,
            max_articles=args.max_articles,
            max_fallback_citations=args.max_articles,
        )),
        result_store=store,
        retrieval_top_k=args.retrieval_top_k,
    )
    store.export_json(args.results_json)
    print(f"Phase 3-4 complete: wrote {written} new results to {args.output}")
    print(f"Exported submission data to {args.results_json}")


if __name__ == "__main__":
    main()
