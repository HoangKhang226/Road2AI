"""Crash-safe retrieval cache shared between Phase 2 and Phase 3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from aiguru.phase2.retriever import ScoredChunk


class RetrievalCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = self._load()

    def _load(self) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        if not self.path.exists():
            return rows
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    rows[int(row["id"])] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return rows

    @property
    def completed_ids(self) -> set[int]:
        return set(self._rows)

    def append(self, question_id: int, question: str, chunks: Iterable[ScoredChunk]) -> bool:
        question_id = int(question_id)
        if question_id in self._rows:
            return False
        row = {
            "id": question_id,
            "question": question,
            "chunks": [
                {
                    "chunk_id": chunk.node["chunk_id"],
                    "text": chunk.node["text"],
                    "metadata": chunk.node.get("metadata") or {},
                    "score": chunk.score,
                }
                for chunk in chunks
            ],
        }
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._rows[question_id] = row
        return True

    def retrieve_by_id(self, question_id: int, question: str | None = None) -> List[Dict[str, Any]]:
        row = self._rows[int(question_id)]
        if question is not None and row["question"] != question:
            raise ValueError(f"Retrieval cache question mismatch for id {question_id}")
        return list(row["chunks"])

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        matches = [row for row in self._rows.values() if row["question"] == question]
        if len(matches) != 1:
            raise KeyError("Question is missing or ambiguous in retrieval cache")
        return list(matches[0]["chunks"])
