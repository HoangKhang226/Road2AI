"""
AIGuru Legal RAG - Phase 1 collector.

Outputs:
- raw_data/legal_docs_raw.jsonl
- raw_data/precedents_raw.jsonl
- raw_data/collection_report.json
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aiguru.phase1.config import (
    ANLE_DATASET_CANDIDATES,
    COLLECTION_REPORT_FILE,
    DOC_TYPE_PATTERNS,
    PHAPDIEN_DATASET_CANDIDATES,
    RAW_DATA_DIR,
    RAW_LEGAL_DOCS_FILE,
    RAW_PRECEDENTS_FILE,
    SME_KEYWORDS_HIGH,
    SME_KEYWORDS_MEDIUM,
)
from aiguru.phase1.metadata_schema import infer_doc_type, normalize_whitespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ensure_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_hf_dataset(candidate_names: List[str], config_name: Optional[str] = None) -> Tuple[Optional[Any], Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        return None, {
            "status": "datasets_library_unavailable",
            "error": str(exc),
            "candidates": candidate_names,
        }

    attempts = []
    for name in candidate_names:
        try:
            if config_name:
                dataset = load_dataset(name, config_name)
            else:
                dataset = load_dataset(name)
            return dataset, {"status": "success", "dataset_name": name, "config": config_name}
        except Exception as exc:
            attempts.append({"dataset_name": name, "config": config_name, "error": str(exc)[:500]})
    return None, {"status": "all_candidates_failed", "attempts": attempts}


def flatten_dataset(dataset: Any) -> Iterable[Dict[str, Any]]:
    if dataset is None:
        return []
    if hasattr(dataset, "keys"):
        for split_name in dataset.keys():
            split = dataset[split_name]
            for row in split:
                item = dict(row)
                item["_split"] = split_name
                yield item
    else:
        for row in dataset:
            yield dict(row)


def first_non_empty(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in row and normalize_whitespace(row.get(key)):
            return normalize_whitespace(row.get(key))
    return ""


def extract_doc_id(text: str) -> str:
    text = normalize_whitespace(text)
    patterns = [
        r"\b\d{1,3}/\d{4}/[A-ZĐ\-]+(?:-[A-ZĐ]+)?\b",
        r"\b\d{1,3}/\d{4}/QH\d+\b",
        r"\b\d{1,3}/\d{4}/NĐ-CP\b",
        r"\b\d{1,3}/\d{4}/TT-[A-ZĐ]+\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return ""


def score_sme(text: str) -> float:
    lowered = normalize_whitespace(text).lower()
    score = 0.0
    for keyword in SME_KEYWORDS_HIGH:
        if keyword.lower() in lowered:
            score += 2.0
    for keyword in SME_KEYWORDS_MEDIUM:
        if keyword.lower() in lowered:
            score += 1.0
    return score


def normalize_legal_doc(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    # Phapdien 'articles' subset: content_text, article_title, chapter_title
    title = first_non_empty(
        row,
        [
            "article_title",
            "title",
            "doc_title",
            "subject_title",
            "topic_title",
        ],
    )
    raw_text = first_non_empty(
        row,
        [
            "content_text",
            "markdown",
            "text",
            "content",
            "noi_dung",
        ],
    )
    combined = f"{title} {raw_text}"
    doc_id = first_non_empty(row, ["doc_id", "law_id", "so_hieu", "document_id", "article_anchor", "id"])
    if not extract_doc_id(doc_id):
        extracted = extract_doc_id(combined)
        if extracted:
            doc_id = extracted
    doc_type = first_non_empty(row, ["doc_type", "loai_van_ban", "type"])
    if not doc_type:
        doc_type = infer_doc_type(combined, DOC_TYPE_PATTERNS)
    return {
        "doc_id": normalize_whitespace(doc_id),
        "doc_type": normalize_whitespace(doc_type),
        "doc_title": title or normalize_whitespace(row.get("title", "")),
        "source": source,
        "raw_text": raw_text,
        "structure": row,
        "sme_score": score_sme(combined),
        "collected_at": datetime.utcnow().isoformat() + "Z",
    }


def normalize_precedent(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    # Primary: extract from markdown (tmquan anle structure)
    title = first_non_empty(row, ["title", "doc_name", "name", "case_name", "doc_code"])
    raw_text = first_non_empty(row, ["markdown", "text", "content"])
    
    # Fallback: if markdown empty, try extracting from structure_json sentences
    if not raw_text and "structure_json" in row:
        try:
            structure = row["structure_json"]
            if isinstance(structure, str):
                import json
                structure = json.loads(structure)
            if "sentences" in structure:
                sentences = [s.get("text", "") for s in structure["sentences"] if s.get("text")]
                raw_text = " ".join(sentences)
        except Exception:
            pass
    
    doc_id = first_non_empty(row, ["doc_code", "doc_name", "doc_id", "case_id", "precedent_number"])
    
    # Preserve rich metadata for SME scoring and hybrid search
    applied_article = row.get("applied_article_number")
    applied_code = row.get("applied_article_code")
    
    return {
        "doc_id": normalize_whitespace(doc_id),
        "doc_type": "Án lệ",
        "doc_title": title,
        "source": source,
        "raw_text": raw_text,
        "structure": row,
        "applied_article_number": applied_article,
        "applied_article_code": applied_code,
        "sme_score": score_sme(f"{title} {raw_text}"),
        "collected_at": datetime.utcnow().isoformat() + "Z",
    }


def collect() -> Dict[str, Any]:
    ensure_dirs()
    report: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "legal_docs": {},
        "precedents": {},
    }

    phapdien_dataset, phapdien_status = load_hf_dataset(PHAPDIEN_DATASET_CANDIDATES, config_name="articles")
    legal_docs = [normalize_legal_doc(row, "phapdien") for row in flatten_dataset(phapdien_dataset)]
    legal_docs = [doc for doc in legal_docs if doc["raw_text"] or doc["doc_title"]]
    legal_count = write_jsonl(RAW_LEGAL_DOCS_FILE, legal_docs)
    report["legal_docs"] = {
        "load_status": phapdien_status,
        "output_file": str(RAW_LEGAL_DOCS_FILE),
        "count": legal_count,
        "missing_doc_id": sum(1 for doc in legal_docs if not doc["doc_id"]),
        "missing_doc_title": sum(1 for doc in legal_docs if not doc["doc_title"]),
        "top_sme_docs": sorted(
            [
                {
                    "doc_id": doc["doc_id"],
                    "doc_type": doc["doc_type"],
                    "doc_title": doc["doc_title"],
                    "sme_score": doc["sme_score"],
                }
                for doc in legal_docs
            ],
            key=lambda x: x["sme_score"],
            reverse=True,
        )[:30],
    }

    anle_dataset, anle_status = load_hf_dataset(ANLE_DATASET_CANDIDATES)
    precedents = [normalize_precedent(row, "anle") for row in flatten_dataset(anle_dataset)]
    precedents = [doc for doc in precedents if doc["raw_text"] or doc["doc_title"]]
    precedent_count = write_jsonl(RAW_PRECEDENTS_FILE, precedents)
    report["precedents"] = {
        "load_status": anle_status,
        "output_file": str(RAW_PRECEDENTS_FILE),
        "count": precedent_count,
        "missing_doc_id": sum(1 for doc in precedents if not doc["doc_id"]),
        "missing_doc_title": sum(1 for doc in precedents if not doc["doc_title"]),
        "top_sme_precedents": sorted(
            [
                {
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["doc_title"],
                    "sme_score": doc["sme_score"],
                }
                for doc in precedents
            ],
            key=lambda x: x["sme_score"],
            reverse=True,
        )[:30],
    }

    COLLECTION_REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    result = collect()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
