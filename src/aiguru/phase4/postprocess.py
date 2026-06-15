"""Three-tier post-processing and competition result assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from aiguru.phase1.metadata_schema import is_submission_doc_id
from aiguru.phase3.generator import RetrievedChunk

ARTICLE_PATTERN = re.compile(r"(?:Điều|Điểu|điều|điểu)\s+(\d+[A-Za-z]?)", re.IGNORECASE)
STANDARD_WARNING = (
    "Cảnh báo giới hạn: Đây là tư vấn sơ bộ từ AI, doanh nghiệp cần đối chiếu "
    "văn bản gốc hoặc tham khảo chuyên gia pháp lý trước khi áp dụng."
)


@dataclass(frozen=True)
class PostProcessConfig:
    safe_threshold: float = 0.3
    high_conf_threshold: float = 0.5
    fallback_threshold: float = 0.0
    max_articles: int = 8
    max_context_chunks: int = 20
    min_high_conf_articles: int = 3
    max_fallback_citations: int = 5


def extract_article_numbers(text: str) -> List[str]:
    seen = set()
    results = []
    for value in ARTICLE_PATTERN.findall(text or ""):
        article = f"Điều {value.upper()}"
        if article not in seen:
            seen.add(article)
            results.append(article)
    return results


def _dedupe(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


class PostProcessor:
    def __init__(self, config: PostProcessConfig | None = None):
        self.config = config or PostProcessConfig()

    def select_relevant_chunks(self, chunks: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        ranked = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
        eligible = [
            chunk for chunk in ranked
            if chunk.metadata.get("formatted_article")
            and is_submission_doc_id(chunk.metadata.get("doc_id"))
            and chunk.metadata.get("submission_eligible", True)
        ]
        unique_eligible = []
        seen_articles = set()
        for chunk in eligible:
            article = str(chunk.metadata.get("formatted_article"))
            if article not in seen_articles:
                seen_articles.add(article)
                unique_eligible.append(chunk)
        eligible = unique_eligible

        selected = [chunk for chunk in eligible if chunk.score >= self.config.high_conf_threshold]
        if len(selected) < self.config.min_high_conf_articles:
            selected = [chunk for chunk in eligible if chunk.score >= self.config.safe_threshold]
        if len(selected) < self.config.min_high_conf_articles:
            selected = eligible[: self.config.min_high_conf_articles]

        return selected[: self.config.max_articles]

    @staticmethod
    def _remove_unsupported_citations(answer: str, unsupported: Sequence[str]) -> str:
        for article in unsupported:
            number = re.escape(article.split()[-1])
            answer = re.sub(
                rf"\b(?:Điều|Điểu|điều|điểu)\s+{number}\b",
                "quy định liên quan",
                answer,
                flags=re.IGNORECASE,
            )
        return answer

    def process_answer(
        self,
        answer: str,
        relevant_chunks: Sequence[RetrievedChunk],
    ) -> tuple[str, List[str]]:
        """Validate answer citations and append a grounded fallback when needed."""
        cited = extract_article_numbers(answer)
        available = {
            str(chunk.metadata.get("article_number")): str(chunk.metadata.get("formatted_article"))
            for chunk in relevant_chunks
            if chunk.metadata.get("article_number") and chunk.metadata.get("formatted_article")
        }
        hallucinated = [article for article in cited if article not in available]
        answer = self._remove_unsupported_citations(answer, hallucinated)
        supported_citations = [article for article in extract_article_numbers(answer) if article in available]

        top_score = max((chunk.score for chunk in relevant_chunks), default=0.0)
        if top_score >= self.config.fallback_threshold:
            references = []
            for chunk in relevant_chunks:
                article = str(chunk.metadata.get("article_number") or "")
                formatted_doc = str(chunk.metadata.get("formatted_doc") or "")
                if article and formatted_doc:
                    doc_name = formatted_doc.split("|", 1)[-1]
                    references.append(f"{article} của {doc_name}")
            references = _dedupe(references)[: self.config.max_fallback_citations]
            if references:
                answer = answer.rstrip() + "\n\nCơ sở pháp lý tham chiếu: " + "; ".join(references) + "."
        if STANDARD_WARNING.lower() not in answer.lower():
            answer = answer.rstrip() + "\n\n" + STANDARD_WARNING
        return answer, hallucinated

    def build_result(
        self,
        question_id: int,
        question: str,
        answer: str,
        chunks: Sequence[RetrievedChunk],
    ) -> Dict[str, Any]:
        selected = self.select_relevant_chunks(chunks)
        if not answer.strip():
            answer = "Hiện tại hệ thống dữ liệu chưa ghi nhận quy định pháp lý cụ thể cho tình huống này."
        answer, _hallucinated = self.process_answer(answer, selected)
        relevant_articles = _dedupe(
            [str(chunk.metadata.get("formatted_article") or "") for chunk in selected]
        )
        relevant_docs = _dedupe(
            [str(chunk.metadata.get("formatted_doc") or "") for chunk in selected]
        )
        return {
            "id": int(question_id),
            "question": question,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        }
