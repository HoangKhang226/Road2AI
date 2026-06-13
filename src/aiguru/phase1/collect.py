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
from datetime import datetime, timezone
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
from aiguru.phase1.metadata_schema import (
    extract_article_from_source_note,
    extract_articles_from_source_note,
    extract_doc_ids,
    infer_doc_type,
    normalize_doc_id,
    normalize_whitespace,
)

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


def source_links_text(row: Dict[str, Any]) -> str:
    values = []
    for link in row.get("source_links") or []:
        if isinstance(link, dict):
            values.append(str(link.get("text") or link.get("href") or ""))
        else:
            values.append(str(link))
    return normalize_whitespace(" ".join(values))


def extract_doc_id(text: str) -> str:
    return normalize_doc_id(text)


def source_title(source_note: str, fallback: str) -> str:
    """Build a compact instrument title from a Pháp điển source citation."""
    source_note = normalize_whitespace(source_note)
    if not source_note:
        return fallback
    doc_id = extract_doc_id(source_note)
    if doc_id:
        match = re.search(
            rf"(?:Thông tư liên tịch|Nghị quyết liên tịch|Bộ luật|Luật|"
            rf"Nghị định|Thông tư|Quyết định|Nghị quyết|Pháp lệnh)"
            rf"\s+(?:số\s+)?{re.escape(doc_id)}",
            source_note,
            flags=re.IGNORECASE,
        )
        if match:
            source_note = source_note[match.start():]
    cleaned = re.sub(r"^\(?\s*(?:Điều|Khoản|Điểm)\s+\w+\s+", "", source_note, flags=re.IGNORECASE)
    issue_dates = list(
        re.finditer(r"\s+ngày\s+\d{1,2}/\d{1,2}/\d{4}\b", cleaned, flags=re.IGNORECASE)
    )
    if issue_dates:
        cleaned = cleaned[: issue_dates[0].start()]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ();,.-")
    return cleaned[:500] or fallback


def best_source_title(source_note: str, link_text: str, fallback: str) -> str:
    candidates = {
        source_title(value, fallback)
        for value in (source_note, link_text)
        if normalize_whitespace(value)
    }
    if not candidates:
        return fallback
    return sorted(
        candidates,
        key=lambda value: (
            "http" in value.lower(),
            " ngày " in value.lower(),
            len(value),
        ),
    )[0]


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
    article_title = first_non_empty(
        row,
        [
            "article_title",
            "title",
            "doc_title",
            "subject_title",
            "topic_title",
        ],
    )
    source_note = first_non_empty(row, ["source_note_text", "source_note", "citation"])
    link_text = source_links_text(row)
    if not source_note:
        source_note = link_text
    title = first_non_empty(
        row,
        ["document_title", "document_name", "law_title", "subject_title", "topic_title"],
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
    combined = f"{source_note} {link_text} {title} {article_title} {raw_text}"
    doc_id = extract_doc_id(
        first_non_empty(row, ["doc_id", "law_id", "so_hieu", "document_id", "document_code"])
    )
    if not doc_id:
        doc_id = extract_doc_id(combined)
    doc_type = first_non_empty(row, ["doc_type", "loai_van_ban", "type"])
    if not doc_type:
        doc_type = infer_doc_type(source_note or combined, DOC_TYPE_PATTERNS)
    article_number = first_non_empty(row, ["article_number", "article_num"])
    article_numbers = [article_number] if article_number else []
    submission_article = bool(article_numbers)
    if not article_numbers:
        source_articles = extract_articles_from_source_note(source_note)
        # Multiple source articles are safe only when they map to one instrument.
        article_numbers = source_articles if len(extract_doc_ids(source_note)) == 1 else source_articles[:1]
        article_number = article_numbers[0] if article_numbers else ""
        submission_article = bool(article_numbers)
    if not article_number:
        article_number = extract_article_from_source_note(article_title)
    enriched_text = ". ".join(
        value for value in (article_title, raw_text, f"Căn cứ nguồn: {source_note}" if source_note else "")
        if value
    )
    return {
        "doc_id": normalize_whitespace(doc_id),
        "doc_type": normalize_whitespace(doc_type),
        "doc_title": best_source_title(source_note, link_text, title or article_title),
        "article_number": article_number,
        "article_numbers": article_numbers,
        "submission_article": submission_article,
        "article_title": article_title,
        "source_note": source_note,
        "source": source,
        "raw_text": enriched_text,
        "structure": row,
        "sme_score": score_sme(combined),
        "collected_at": datetime.now(timezone.utc).isoformat(),
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
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def collect() -> Dict[str, Any]:
    ensure_dirs()
    report: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
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
