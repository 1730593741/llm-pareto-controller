"""Experience pool primitives for M5 memory-enabled closed-loop control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ExperienceRecord:
    """A single transition tuple captured from one control cycle."""

    state: dict[str, Any]
    action: dict[str, Any]
    reward: float
    next_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record for storage backends."""
        return asdict(self)


class ExperiencePool:
    """In-memory sliding window storage for recent control experiences."""

    def __init__(self, max_size: int = 100) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = max_size
        self._records: list[ExperienceRecord] = []

    def append(self, record: ExperienceRecord) -> None:
        """Add one experience and enforce sliding-window trimming."""
        self._records.append(record)
        overflow = len(self._records) - self.max_size
        if overflow > 0:
            self._records = self._records[overflow:]

    def recent(self, n: int) -> list[ExperienceRecord]:
        """Return at most the last ``n`` records in chronological order."""
        if n <= 0:
            return []
        return self._records[-n:]

    def get_recent(self, n: int) -> list[ExperienceRecord]:
        """Alias kept for future caller compatibility."""
        return self.recent(n)

    def __len__(self) -> int:
        return len(self._records)
