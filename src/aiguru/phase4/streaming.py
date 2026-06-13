"""Crash-safe JSONL result storage with resume support."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set


class JsonlResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed_ids = self._load_completed_ids()

    def _load_completed_ids(self) -> Set[int]:
        completed: Set[int] = set()
        if not self.path.exists():
            return completed
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    completed.add(int(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return completed

    @property
    def completed_ids(self) -> Set[int]:
        return set(self._completed_ids)

    def append(self, result: Dict[str, Any]) -> bool:
        result_id = int(result["id"])
        if result_id in self._completed_ids:
            return False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._completed_ids.add(result_id)
        return True

    def read_all(self) -> List[Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    rows[int(row["id"])] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return list(rows.values())

    def export_json(self, path: str | Path) -> Path:
        """Atomically export the accumulated JSONL rows as a JSON array."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        rows = sorted(self.read_all(), key=lambda row: int(row["id"]))
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(output_path)
        return output_path
