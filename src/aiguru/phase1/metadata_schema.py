"""
AIGuru Legal RAG - Phase 1 metadata schema helpers.

The goal of this module is provenance safety: every knowledge chunk must carry
preformatted strings that can be copied directly into results.json later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import re
import unicodedata


@dataclass(frozen=True)
class ChunkMetadata:
    doc_id: str
    doc_type: str
    doc_title: str
    article_number: str
    formatted_doc: str
    formatted_article: str
    source: str = ""
    parent_id: str = ""
    paragraph_number: str = ""
    sme_score: float = 0.0


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": asdict(self.metadata),
        }


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_slug(value: str) -> str:
    value = normalize_whitespace(value)
    value = value.replace("/", "_").replace("\\", "_").replace("|", "_")
    value = re.sub(r"[^0-9A-Za-zÀ-ỹ_\-.]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def infer_doc_type(text: str, patterns: Dict[str, list]) -> str:
    lowered = normalize_whitespace(text).lower()
    for doc_type, keys in patterns.items():
        if any(key in lowered for key in keys):
            return doc_type
    return "Văn bản"


def normalize_article_number(value: Any) -> str:
    value = normalize_whitespace(value)
    if not value:
        return ""
    match = re.search(r"(?:Điều|điều)\s*(\d+[a-zA-Z]?)", value)
    if match:
        return f"Điều {match.group(1)}"
    match = re.search(r"^(\d+[a-zA-Z]?)$", value)
    if match:
        return f"Điều {match.group(1)}"
    return value


def build_formatted_doc(doc_id: str, doc_type: str, doc_title: str) -> str:
    doc_id = normalize_whitespace(doc_id)
    doc_type = normalize_whitespace(doc_type)
    doc_title = normalize_whitespace(doc_title)
    title_core = doc_title
    title_lower = title_core.lower()
    if doc_id and doc_id.lower() not in title_lower:
        title_core = f"{doc_id} {title_core}".strip()
    if doc_type and not title_core.lower().startswith(doc_type.lower()):
        title_core = f"{doc_type} {title_core}".strip()
    return f"{doc_id}|{title_core}"


def build_formatted_article(
    doc_id: str,
    doc_type: str,
    doc_title: str,
    article_number: str,
) -> str:
    article_number = normalize_article_number(article_number)
    return f"{build_formatted_doc(doc_id, doc_type, doc_title)}|{article_number}"


def build_chunk_id(
    doc_id: str,
    article_number: str,
    paragraph_number: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    parts = [safe_slug(doc_id)]
    article = normalize_article_number(article_number)
    if article:
        parts.append(safe_slug(article.replace("Điều", "Dieu")))
    if paragraph_number:
        parts.append(safe_slug(f"Khoan_{paragraph_number}"))
    if suffix:
        parts.append(safe_slug(suffix))
    return "_".join(parts)
