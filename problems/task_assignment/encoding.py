"""Encoding utilities for the task-assignment problem."""

from __future__ import annotations

import random
from typing import TypeAlias

Assignment: TypeAlias = list[int]


def random_assignment(n_tasks: int, n_resources: int, rng: random.Random | None = None) -> Assignment:
    """Create a random task-to-resource assignment.

    Each task is assigned to exactly one resource index in ``[0, n_resources)``.

    Args:
        n_tasks: Number of tasks to assign.
        n_resources: Number of available resources.
        rng: Optional random generator for reproducibility.

    Returns:
        A list where position ``i`` is the resource index for task ``i``.

    Raises:
        ValueError: If ``n_tasks`` is negative or ``n_resources`` is not positive.
    """
    if n_tasks < 0:
        raise ValueError("n_tasks must be >= 0")
    if n_resources <= 0:
        raise ValueError("n_resources must be > 0")

    generator = rng or random.Random()
    return [generator.randrange(n_resources) for _ in range(n_tasks)]
