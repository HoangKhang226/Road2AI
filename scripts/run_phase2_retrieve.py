"""Precompute hybrid retrieval + reranking for all competition questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiguru.paths import DATA_DIR, KNOWLEDGE_DIR
from aiguru.phase2.cache import RetrievalCache
from aiguru.phase2.retriever import HybridLegalRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DATA_DIR / "R2AIStage1DATA.json")
    parser.add_argument("--output", type=Path, default=KNOWLEDGE_DIR / "retrieval_results.jsonl")
    parser.add_argument("--collection", default="aiguru_legal")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--fusion-top-k", type=int, default=30)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from aiguru.phase2.reranker import BGEReranker
    from aiguru.phase2.vector_db import VectorDBManager

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.reset and args.output.exists():
        args.output.unlink()

    from aiguru.paths import STORAGE_DIR
    qdrant_meta = STORAGE_DIR / args.collection / "docstore.json"
    if qdrant_meta.exists():
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device=args.embedding_device)
        vector_db = VectorDBManager(embed_model, collection_name=args.collection)
        index = vector_db.get_index()
        vector_retriever = index.as_retriever(similarity_top_k=50)
    else:
        print(f"⚠️ Qdrant index metadata not found at {qdrant_meta}. Falling back to BM25-only retrieval.")
        vector_retriever = None
    reranker = None if args.no_reranker else BGEReranker(
        args.reranker_model,
        device=args.reranker_device,
        max_length=512,
    )
    retriever = HybridLegalRetriever(
        vector_retriever=vector_retriever,
        reranker=reranker,
        bm25_top_k=args.bm25_top_k,
        fusion_top_k=args.fusion_top_k,
    )
    cache = RetrievalCache(args.output)
    completed_ids = cache.completed_ids
    written = 0
    for position, question in enumerate(questions, 1):
        question_id = int(question["id"])
        if question_id in completed_ids:
            continue
        results = retriever.retrieve(str(question["question"]))
        written += int(cache.append(question_id, str(question["question"]), results))
        if position % 25 == 0:
            print(f"Retrieved {position}/{len(questions)} questions")
    print(f"Phase 2 retrieval complete: wrote {written} new rows to {args.output}")


if __name__ == "__main__":
    main()
