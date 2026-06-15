"""End-to-end Phase 3-4 orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from aiguru.phase3.generator import UnslothGenerator, normalize_retrieval_results
from aiguru.phase4.postprocess import PostProcessor
from aiguru.phase4.streaming import JsonlResultStore


def run_generation_pipeline(
    questions: Sequence[Dict[str, Any]],
    retriever: Any,
    generator: UnslothGenerator,
    postprocessor: PostProcessor,
    result_store: JsonlResultStore,
    retrieval_top_k: int = 25,
) -> int:
    """Retrieve, batch-generate, post-process, and persist unanswered questions."""
    completed_ids = result_store.completed_ids
    pending = [question for question in questions if int(question["id"]) not in completed_ids]
    written = 0
    for start in range(0, len(pending), generator.config.batch_size):
        batch = pending[start : start + generator.config.batch_size]
        all_contexts: List[list] = []
        selected_per_question: List[list] = []
        for question in batch:
            if hasattr(retriever, "retrieve_by_id"):
                items = retriever.retrieve_by_id(int(question["id"]), str(question["question"]))
            else:
                items = retriever.retrieve(str(question["question"]))
            chunks = normalize_retrieval_results(items)[:retrieval_top_k]
            selected_chunks = postprocessor.select_relevant_chunks(chunks)
            # LLM sees ALL retrieved chunks for richer context
            all_contexts.append(chunks)
            # But only selected eligible chunks go to relevant_articles
            selected_per_question.append(selected_chunks)
        answers = generator.generate([str(item["question"]) for item in batch], all_contexts)
        for question, answer, chunks in zip(batch, answers, selected_per_question):
            result = postprocessor.build_result(
                question_id=int(question["id"]),
                question=str(question["question"]),
                answer=answer,
                chunks=chunks,
            )
            written += int(result_store.append(result))
        print(f"Generated {start + len(batch)}/{len(pending)} answers...")
    return written
