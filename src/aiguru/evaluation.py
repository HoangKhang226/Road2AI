"""Local macro retrieval metrics for hand-labelled development sets."""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from aiguru.phase1.metadata_schema import normalize_doc_id


def _articles(row: Dict[str, Any]) -> set[str]:
    return {str(value) for value in row.get("relevant_articles", [])}


def macro_retrieval_metrics(
    predictions: Sequence[Dict[str, Any]],
    references: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    reference_map = {int(row["id"]): row for row in references}
    precision_values = []
    recall_values = []
    f2_values = []
    for prediction in predictions:
        expected = _articles(reference_map[int(prediction["id"])])
        actual = _articles(prediction)
        correct = len(actual & expected)
        precision = correct / len(actual) if actual else 0.0
        recall = correct / len(expected) if expected else 0.0
        f2 = (5 * precision * recall / (4 * precision + recall)) if precision or recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f2_values.append(f2)
    count = len(precision_values) or 1
    return {
        "macro_precision": sum(precision_values) / count,
        "macro_recall": sum(recall_values) / count,
        "macro_f2": sum(f2_values) / count,
        "evaluated_queries": len(precision_values),
    }


def silver_retrieval_recall(
    questions: Sequence[Dict[str, Any]],
    cached_rows: Sequence[Dict[str, Any]],
    cutoffs: Sequence[int] = (3, 5, 10),
) -> Dict[str, float]:
    """Measure recall on questions that explicitly mention an article number."""
    cache_map = {int(row["id"]): row for row in cached_rows}
    totals = {cutoff: 0 for cutoff in cutoffs}
    evaluated = 0
    eligible_coverage = 0
    for question in questions:
        match = re.search(r"\bĐiều\s+(\d+[A-Za-z]?)\b", question["question"], flags=re.IGNORECASE)
        if not match or int(question["id"]) not in cache_map:
            continue
        expected_article = f"Điều {match.group(1)}".lower()
        expected_doc = normalize_doc_id(question["question"])
        chunks = cache_map[int(question["id"])]["chunks"]
        evaluated += 1
        if any((chunk.get("metadata") or {}).get("submission_eligible") for chunk in chunks[:10]):
            eligible_coverage += 1
        for cutoff in cutoffs:
            found = False
            for chunk in chunks[:cutoff]:
                metadata = chunk.get("metadata") or {}
                article_matches = str(metadata.get("article_number") or "").lower() == expected_article
                doc_matches = not expected_doc or str(metadata.get("doc_id") or "") == expected_doc
                if article_matches and doc_matches:
                    found = True
                    break
            totals[cutoff] += int(found)
    denominator = evaluated or 1
    result = {f"silver_recall_at_{cutoff}": totals[cutoff] / denominator for cutoff in cutoffs}
    result["submission_eligible_coverage_at_10"] = eligible_coverage / denominator
    result["evaluated_queries"] = evaluated
    return result
