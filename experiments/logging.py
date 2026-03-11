"""Experiment-level logging helpers for M7 artifact organization."""

from __future__ import annotations

import json
from pathlib import Path


def split_event_stream(*, events_path: Path, state_log_path: Path, action_log_path: Path) -> None:
    """Split the unified event stream into dedicated state/action JSONL files."""
    state_log_path.parent.mkdir(parents=True, exist_ok=True)
    action_log_path.parent.mkdir(parents=True, exist_ok=True)

    with events_path.open("r", encoding="utf-8") as src, \
        state_log_path.open("w", encoding="utf-8") as state_f, \
        action_log_path.open("w", encoding="utf-8") as action_f:
        for line in src:
            payload = json.loads(line)
            event_type = payload.get("event")
            if event_type == "state":
                state_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            elif event_type == "action":
                action_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
