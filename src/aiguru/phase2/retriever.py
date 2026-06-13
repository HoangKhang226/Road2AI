"""Competition-oriented BM25 + dense + RRF + optional cross-encoder retrieval."""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from aiguru.phase2.bm25_indexer_simple import simple_tokenize
from aiguru.phase2.config import (
    BM25_CHUNK_ID_MAP_FILE,
    BM25_CORPUS_FILE,
    BM25_TOP_K,
    CHUNKS_FILE,
    RERANKER_BATCH_SIZE,
    RRF_K,
    RRF_TOP_K,
)


@dataclass
class ScoredChunk:
    node: Dict[str, Any]
    score: float


DOMAIN_EXPANSIONS = {
    ("doanh nghiệp nhỏ và vừa", "dnnvv", "khởi nghiệp sáng tạo"): "Luật hỗ trợ doanh nghiệp nhỏ và vừa nghị định hướng dẫn hỗ trợ DNNVV",
    ("người lao động", "nhân viên", "hợp đồng lao động", "tiền lương"): "Bộ luật Lao động xử phạt vi phạm lao động",
    ("bảo hiểm xã hội", "bhxh", "bảo hiểm thất nghiệp"): "Luật Bảo hiểm xã hội xử phạt chậm đóng bảo hiểm",
    ("thuế", "khai thuế", "nộp thuế", "chậm nộp"): "Luật Quản lý thuế xử phạt vi phạm hành chính về thuế",
    ("hóa đơn", "chứng từ"): "hóa đơn chứng từ xử phạt vi phạm hành chính",
    ("kế toán", "báo cáo tài chính"): "Luật Kế toán chuẩn mực báo cáo tài chính",
    ("nhãn hiệu", "sở hữu trí tuệ", "bản quyền"): "Luật Sở hữu trí tuệ",
    ("hợp đồng", "thương mại", "mua bán"): "Luật Thương mại Bộ luật Dân sự hợp đồng",
    ("xử phạt", "bị phạt", "vi phạm", "khắc phục hậu quả"): "nghị định xử phạt vi phạm hành chính biện pháp khắc phục hậu quả",
    ("đất đai", "mặt bằng sản xuất", "thuê đất"): "Luật Đất đai hỗ trợ mặt bằng sản xuất",
    ("đấu thầu", "nhà thầu"): "Luật Đấu thầu ưu đãi doanh nghiệp nhỏ và vừa",
    ("tín dụng", "bảo lãnh", "vay vốn"): "quỹ bảo lãnh tín dụng hỗ trợ doanh nghiệp nhỏ và vừa",
}


def expand_legal_query(query: str) -> str:
    lowered = query.lower()
    additions = [
        expansion
        for keywords, expansion in DOMAIN_EXPANSIONS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return f"{query} {' '.join(additions)}".strip()


def load_chunk_map(path: Path = CHUNKS_FILE) -> Dict[str, Dict[str, Any]]:
    chunks: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk = json.loads(line)
                chunks[str(chunk["chunk_id"])] = chunk
    return chunks


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class HybridLegalRetriever:
    """Uses persisted BM25 artifacts plus an optional Qdrant/LlamaIndex retriever."""

    def __init__(
        self,
        vector_retriever: Any | None = None,
        reranker: Any | None = None,
        chunks_file: Path = CHUNKS_FILE,
        bm25_file: Path = BM25_CORPUS_FILE,
        bm25_map_file: Path = BM25_CHUNK_ID_MAP_FILE,
        bm25_top_k: int = BM25_TOP_K,
        fusion_top_k: int = RRF_TOP_K,
    ):
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self.bm25_top_k = bm25_top_k
        self.fusion_top_k = fusion_top_k
        self.chunks = load_chunk_map(chunks_file)
        with bm25_file.open("rb") as handle:
            self.bm25 = pickle.load(handle)
        self.bm25_ids = json.loads(bm25_map_file.read_text(encoding="utf-8"))
        if len(self.bm25_ids) != len(self.bm25.doc_len):
            raise ValueError("BM25 chunk ID map does not match corpus size")

    def _bm25_ranking(self, query: str) -> List[str]:
        scores = np.asarray(self.bm25.get_scores(simple_tokenize(query)))
        top_k = min(self.bm25_top_k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [str(self.bm25_ids[index]) for index in indices if scores[index] > 0]

    def _vector_ranking(self, query: str) -> List[str]:
        if self.vector_retriever is None:
            return []
        results = self.vector_retriever.retrieve(query)
        ranking = []
        for item in results:
            node = getattr(item, "node", item)
            if isinstance(node, dict):
                chunk_id = node.get("chunk_id") or node.get("id") or ""
            else:
                chunk_id = getattr(node, "node_id", "") or getattr(node, "id_", "")
            if chunk_id:
                ranking.append(str(chunk_id))
        return ranking

    def _lexical_boost(self, query: str, chunk: Dict[str, Any]) -> float:
        metadata = chunk.get("metadata") or {}
        haystack = " ".join(
            str(metadata.get(key) or "") for key in ("doc_id", "doc_title", "article_number")
        ).lower()
        boost = 0.0
        for token in re.findall(r"\d{1,3}/\d{4}/[\wĐđ-]+|điều\s+\d+[a-z]?", query.lower()):
            if token in haystack:
                boost += 0.02
        boost += min(float(metadata.get("sme_score") or 0.0) / 20.0, 1.0) * 0.03
        if not metadata.get("submission_eligible"):
            boost -= 0.05
        return boost

    def _rerank(self, query: str, chunk_ids: Sequence[str], rrf_scores: Dict[str, float]) -> List[ScoredChunk]:
        chunks = [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]
        if not chunks:
            return []
        max_rrf = max(rrf_scores.values()) or 1.0
        if self.reranker is None:
            return [
                ScoredChunk(chunk, min(1.0, rrf_scores[chunk["chunk_id"]] / max_rrf + self._lexical_boost(query, chunk)))
                for chunk in chunks
            ]
        logits = self.reranker.predict(
            [(query, chunk["text"]) for chunk in chunks],
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
        output = []
        for chunk, logit in zip(chunks, logits):
            rerank_score = sigmoid(float(logit))
            rrf_score = rrf_scores[chunk["chunk_id"]] / max_rrf
            score = 0.8 * rerank_score + 0.2 * rrf_score + self._lexical_boost(query, chunk)
            output.append(ScoredChunk(chunk, min(1.0, score)))
        return output

    def retrieve(self, query: str) -> List[ScoredChunk]:
        expanded_query = expand_legal_query(query)
        rankings = [
            ranking
            for ranking in [self._bm25_ranking(expanded_query), self._vector_ranking(expanded_query)]
            if ranking
        ]
        fused = reciprocal_rank_fusion(rankings)
        ordered = []
        article_counts: Dict[str, int] = {}
        for chunk_id in sorted(fused, key=fused.get, reverse=True):
            chunk = self.chunks.get(chunk_id) or {}
            article = str((chunk.get("metadata") or {}).get("formatted_article") or chunk_id)
            if article_counts.get(article, 0) >= 2:
                continue
            article_counts[article] = article_counts.get(article, 0) + 1
            ordered.append(chunk_id)
            if len(ordered) >= self.fusion_top_k:
                break
        results = self._rerank(query, ordered, fused)
        return sorted(results, key=lambda item: item.score, reverse=True)
