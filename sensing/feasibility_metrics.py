"""Feasibility and constraint-violation metrics for population sensing."""

from __future__ import annotations

from collections.abc import Sequence


def feasible_ratio(violations: Sequence[float], *, tolerance: float = 0.0) -> float:
    """Return the fraction of solutions whose violation is within tolerance."""
    if not violations:
        return 0.0
    feasible_count = sum(1 for value in violations if value <= tolerance)
    return feasible_count / len(violations)


def mean_constraint_violation(violations: Sequence[float]) -> float:
    """Return the arithmetic mean of non-negative constraint violations."""
    if not violations:
        return 0.0
    return sum(max(0.0, value) for value in violations) / len(violations)


def max_constraint_violation(violations: Sequence[float]) -> float:
    """Return the maximum non-negative constraint violation."""
    if not violations:
        return 0.0
    return max(max(0.0, value) for value in violations)
