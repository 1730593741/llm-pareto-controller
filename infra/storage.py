"""Lightweight storage utilities for experiment events (JSONL)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only JSONL logger for state/action events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any]) -> None:
        """Write one event as a single JSON object line."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
