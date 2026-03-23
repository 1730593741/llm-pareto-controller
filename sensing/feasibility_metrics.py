"""可行性与约束违反指标 用于 种群 感知."""

from __future__ import annotations

from collections.abc import Sequence


def feasible_ratio(violations: Sequence[float], *, tolerance: float = 0.0) -> float:
    """返回 该 fraction 的 solutions whose 违反 为 within tolerance."""
    if not violations:
        return 0.0
    feasible_count = sum(1 for value in violations if value <= tolerance)
    return feasible_count / len(violations)


def mean_constraint_violation(violations: Sequence[float]) -> float:
    """返回 该 arithmetic 均值 的 non-negative 约束 violations."""
    if not violations:
        return 0.0
    return sum(max(0.0, value) for value in violations) / len(violations)


def max_constraint_violation(violations: Sequence[float]) -> float:
    """返回 该 maximum non-negative 约束 违反."""
    if not violations:
        return 0.0
    return max(max(0.0, value) for value in violations)
