"""
AIGuru Legal RAG - Phase 1 structural chunking and metadata injection.

Outputs:
- knowledge_store/chunks.jsonl
- knowledge_store/chunk_stats.json
- knowledge_store/metadata_errors.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from aiguru.phase1.config import (
    CHUNK_STATS_FILE,
    CHUNKS_FILE,
    KNOWLEDGE_DIR,
    MAX_ARTICLE_CHARS_BEFORE_PARAGRAPH_SPLIT,
    METADATA_ERRORS_FILE,
    MIN_CHUNK_CHARS,
    RAW_LEGAL_DOCS_FILE,
    RAW_PRECEDENTS_FILE,
)
from aiguru.phase1.metadata_schema import (
    ChunkMetadata,
    KnowledgeChunk,
    build_chunk_id,
    build_formatted_article,
    build_formatted_doc,
    normalize_article_number,
    is_submission_doc_id,
    normalize_whitespace,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARTICLE_PATTERN = re.compile(
    r"(?P<header>(?:^|\n)\s*Điều\s+(?P<num>\d+[a-zA-Z]?)\.?[^\n]*)",
    flags=re.IGNORECASE,
)

PARAGRAPH_PATTERN = re.compile(
    r"(?m)^\s*(?P<num>\d+)\.\s+(?P<body>.*?)(?=^\s*\d+\.\s+|\Z)",
    flags=re.DOTALL,
)


def ensure_dirs() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_error": str(exc), "_line_no": line_no})
    return rows


def split_articles(text: str) -> List[Tuple[str, str]]:
    text = normalize_whitespace(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"\s+(Điều\s+\d+[a-zA-Z]?\.)", r"\n\1", text, flags=re.IGNORECASE)
    matches = list(ARTICLE_PATTERN.finditer(text))
    if not matches:
        return []

    articles: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        article_number = normalize_article_number(match.group("num"))
        article_text = normalize_whitespace(text[start:end])
        if len(article_text) >= MIN_CHUNK_CHARS:
            articles.append((article_number, article_text))
    return articles


def split_long_article(article_text: str) -> List[Tuple[str, str]]:
    if len(article_text) <= MAX_ARTICLE_CHARS_BEFORE_PARAGRAPH_SPLIT:
        return [("", article_text)]
    paragraphs = []
    for match in PARAGRAPH_PATTERN.finditer(article_text):
        paragraph_number = match.group("num")
        body = normalize_whitespace(match.group(0))
        if len(body) >= MIN_CHUNK_CHARS:
            paragraphs.append((paragraph_number, body))
    return paragraphs or [("", article_text)]


def build_chunk(
    doc: Dict[str, Any],
    article_number: str,
    text: str,
    paragraph_number: str = "",
    suffix: str = "",
) -> KnowledgeChunk:
    doc_id = normalize_whitespace(doc.get("doc_id")) or "UNKNOWN_DOC"
    doc_type = normalize_whitespace(doc.get("doc_type")) or "Văn bản"
    doc_title = normalize_whitespace(doc.get("doc_title")) or doc_id
    article_number = normalize_article_number(article_number)
    chunk_id = build_chunk_id(doc_id, article_number, paragraph_number, suffix)
    metadata = ChunkMetadata(
        doc_id=doc_id,
        doc_type=doc_type,
        doc_title=doc_title,
        article_number=article_number,
        formatted_doc=build_formatted_doc(doc_id, doc_type, doc_title),
        formatted_article=build_formatted_article(doc_id, doc_type, doc_title, article_number)
        if article_number
        else "",
        source=normalize_whitespace(doc.get("source")),
        parent_id=doc_id,
        paragraph_number=paragraph_number,
        sme_score=float(doc.get("sme_score") or 0.0),
        source_note=normalize_whitespace(doc.get("source_note")),
        submission_eligible=bool(
            article_number
            and is_submission_doc_id(doc_id)
            and doc.get("submission_article", True)
        ),
    )
    return KnowledgeChunk(chunk_id=chunk_id, text=normalize_whitespace(text), metadata=metadata)


def chunk_legal_doc(doc: Dict[str, Any]) -> List[KnowledgeChunk]:
    raw_text = normalize_whitespace(doc.get("raw_text"))
    if not raw_text:
        return []
    source_articles = [
        normalize_article_number(article)
        for article in (doc.get("article_numbers") or [doc.get("article_number")])
        if normalize_article_number(article)
    ]
    if source_articles:
        return [
            build_chunk(doc, source_article, chunk_text, paragraph_number)
            for source_article in source_articles
            for paragraph_number, chunk_text in split_long_article(raw_text)
        ]
    articles = split_articles(raw_text)
    chunks: List[KnowledgeChunk] = []

    if not articles:
        chunks.append(build_chunk(doc, "", raw_text, suffix="full_doc"))
        return chunks

    for article_number, article_text in articles:
        for paragraph_number, chunk_text in split_long_article(article_text):
            chunks.append(build_chunk(doc, article_number, chunk_text, paragraph_number))
    return chunks


def chunk_precedent(doc: Dict[str, Any]) -> List[KnowledgeChunk]:
    raw_text = normalize_whitespace(doc.get("raw_text"))
    if not raw_text:
        return []
    return [build_chunk(doc, "", raw_text, suffix="precedent")]


def validate_chunk(chunk: KnowledgeChunk) -> List[str]:
    errors = []
    data = chunk.to_dict()
    metadata = data["metadata"]
    if not data["chunk_id"]:
        errors.append("missing_chunk_id")
    if not data["text"]:
        errors.append("missing_text")
    if not metadata.get("doc_id"):
        errors.append("missing_doc_id")
    if not metadata.get("formatted_doc"):
        errors.append("missing_formatted_doc")
    if metadata.get("article_number") and not metadata.get("formatted_article"):
        errors.append("missing_formatted_article")
    return errors


def canonicalize_document_titles(chunks: List[KnowledgeChunk]) -> List[KnowledgeChunk]:
    """Use one stable, frequently observed title for every source document."""
    title_counts: Dict[str, Counter] = defaultdict(Counter)
    for chunk in chunks:
        if chunk.metadata.submission_eligible:
            title_counts[chunk.metadata.doc_id][chunk.metadata.doc_title] += 1

    canonical = {}
    for doc_id, counts in title_counts.items():
        canonical[doc_id] = sorted(
            counts,
            key=lambda title: (-counts[title], len(title), title),
        )[0]

    normalized = []
    for chunk in chunks:
        title = canonical.get(chunk.metadata.doc_id)
        if not title or title == chunk.metadata.doc_title:
            normalized.append(chunk)
            continue
        formatted_doc = build_formatted_doc(
            chunk.metadata.doc_id,
            chunk.metadata.doc_type,
            title,
        )
        metadata = replace(
            chunk.metadata,
            doc_title=title,
            formatted_doc=formatted_doc,
            formatted_article=f"{formatted_doc}|{chunk.metadata.article_number}",
        )
        normalized.append(KnowledgeChunk(chunk.chunk_id, chunk.text, metadata))
    return normalized


def run_chunking() -> Dict[str, Any]:
    ensure_dirs()
    legal_docs = read_jsonl(RAW_LEGAL_DOCS_FILE)
    precedents = read_jsonl(RAW_PRECEDENTS_FILE)

    all_chunks: List[KnowledgeChunk] = []
    metadata_errors: List[Dict[str, Any]] = []

    for doc in legal_docs:
        if "_error" in doc:
            metadata_errors.append(doc)
            continue
        all_chunks.extend(chunk_legal_doc(doc))

    for doc in precedents:
        if "_error" in doc:
            metadata_errors.append(doc)
            continue
        all_chunks.extend(chunk_precedent(doc))

    all_chunks = canonicalize_document_titles(all_chunks)
    chunk_id_counts = Counter(chunk.chunk_id for chunk in all_chunks)
    deduped_chunks: List[KnowledgeChunk] = []
    duplicate_counter: Dict[str, int] = defaultdict(int)
    for chunk in all_chunks:
        if chunk_id_counts[chunk.chunk_id] > 1:
            duplicate_counter[chunk.chunk_id] += 1
            new_id = f"{chunk.chunk_id}_dup_{duplicate_counter[chunk.chunk_id]}"
            chunk = KnowledgeChunk(chunk_id=new_id, text=chunk.text, metadata=chunk.metadata)
        deduped_chunks.append(chunk)

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in deduped_chunks:
            errors = validate_chunk(chunk)
            if errors:
                metadata_errors.append({"chunk_id": chunk.chunk_id, "errors": errors})
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    with METADATA_ERRORS_FILE.open("w", encoding="utf-8") as f:
        for error in metadata_errors:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")

    doc_type_counts = Counter(chunk.metadata.doc_type for chunk in deduped_chunks)
    source_counts = Counter(chunk.metadata.source for chunk in deduped_chunks)
    missing_article_count = sum(1 for chunk in deduped_chunks if not chunk.metadata.article_number)
    eligible_chunks = [chunk for chunk in deduped_chunks if chunk.metadata.submission_eligible]

    stats = {
        "total_legal_docs": len(legal_docs),
        "total_precedents": len(precedents),
        "total_chunks": len(deduped_chunks),
        "doc_type_counts": dict(doc_type_counts),
        "source_counts": dict(source_counts),
        "missing_article_count": missing_article_count,
        "submission_eligible_chunk_count": len(eligible_chunks),
        "unique_submission_doc_count": len({chunk.metadata.formatted_doc for chunk in eligible_chunks}),
        "unique_submission_article_count": len(
            {chunk.metadata.formatted_article for chunk in eligible_chunks}
        ),
        "metadata_error_count": len(metadata_errors),
        "duplicate_chunk_id_count": sum(1 for _, count in chunk_id_counts.items() if count > 1),
        "output_file": str(CHUNKS_FILE),
    }
    CHUNK_STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    result = run_chunking()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
