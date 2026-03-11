"""Lightweight storage utilities for experiment events and experiences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory.experience_pool import ExperienceRecord


class JsonlLogger:
    """Append-only JSONL logger for state/action events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any]) -> None:
        """Write one event as a single JSON object line."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class ExperienceJsonlLogger:
    """Persist experience tuples to JSONL for offline analysis."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperienceRecord) -> None:
        """Write one ``ExperienceRecord`` as a JSONL line."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


class CsvExporter:
    """Placeholder interface for future CSV export extension."""

    def export(self, rows: list[dict[str, Any]], path: str | Path) -> None:
        """Future extension point for lightweight CSV output."""
        raise NotImplementedError("CSV export is not implemented in M5 MVP")
