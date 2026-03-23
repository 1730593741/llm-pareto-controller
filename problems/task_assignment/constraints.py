"""约束检查 用于 该 任务-分配 问题."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from problems.task_assignment.encoding import Assignment


@dataclass(frozen=True, slots=True)
class ConstraintBreakdown:
    """Decomposed 约束 违反 用于 interpretable 可行性 analysis."""

    capacity: float = 0.0
    compatibility: float = 0.0
    time_window: float = 0.0
    stage_transition: float = 0.0

    @property
    def total(self) -> float:
        """返回 聚合 约束 违反 作为 该 sum 的 所有 components."""
        return self.capacity + self.compatibility + self.time_window + self.stage_transition


def resource_loads(assignment: Assignment, task_loads: Sequence[float], n_resources: int) -> list[float]:
    """Aggregate per-资源 负载 从 一个 任务 分配."""
    loads = [0.0] * n_resources
    for task_idx, resource_idx in enumerate(assignment):
        loads[resource_idx] += float(task_loads[task_idx])
    return loads


def compatibility_violation(assignment: Assignment, compatibility_matrix: Sequence[Sequence[int]] | None) -> float:
    """返回 任务 count 已分配 到 incompatible 资源."""
    if compatibility_matrix is None:
        return 0.0
    violation = 0.0
    for task_idx, resource_idx in enumerate(assignment):
        if int(compatibility_matrix[task_idx][resource_idx]) == 0:
            violation += 1.0
    return violation


def _interval_distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    """返回 non-overlap 距离 的 两个 closed intervals."""
    lhs_start, lhs_end = float(lhs[0]), float(lhs[1])
    rhs_start, rhs_end = float(rhs[0]), float(rhs[1])
    if lhs_end < rhs_start:
        return rhs_start - lhs_end
    if rhs_end < lhs_start:
        return lhs_start - rhs_end
    return 0.0


def time_window_violation(
    assignment: Assignment,
    task_time_windows: Sequence[Sequence[float]] | None,
    resource_time_windows: Sequence[Sequence[float]] | None,
) -> float:
    """返回 summed non-overlap 距离 用于 已分配 任务-资源 time windows."""
    if task_time_windows is None or resource_time_windows is None:
        return 0.0
    return float(
        sum(
            _interval_distance(task_time_windows[task_idx], resource_time_windows[resource_idx])
            for task_idx, resource_idx in enumerate(assignment)
        )
    )


def stage_transition_violation(
    assignment: Assignment,
    resource_stage_levels: Sequence[int] | None,
    stage_transitions: Sequence[Sequence[int]] | None,
) -> float:
    """返回 total stage-order 违反 over precedence-like 任务 transitions."""
    if resource_stage_levels is None or stage_transitions is None:
        return 0.0

    violation = 0.0
    for predecessor_task, successor_task in stage_transitions:
        predecessor_stage = int(resource_stage_levels[assignment[predecessor_task]])
        successor_stage = int(resource_stage_levels[assignment[successor_task]])
        violation += max(0, predecessor_stage - successor_stage)
    return float(violation)


def is_capacity_feasible(assignment: Assignment, task_loads: Sequence[float], capacities: Sequence[float]) -> bool:
    """返回 whether 所有 资源 capacities 为 satisfied."""
    loads = resource_loads(assignment, task_loads, len(capacities))
    return all(load <= float(capacity) for load, capacity in zip(loads, capacities))


def capacity_violation(assignment: Assignment, task_loads: Sequence[float], capacities: Sequence[float]) -> float:
    """计算 summed overload amount 跨 所有 资源."""
    loads = resource_loads(assignment, task_loads, len(capacities))
    return float(sum(max(0.0, load - float(capacity)) for load, capacity in zip(loads, capacities)))


def constraint_breakdown(
    assignment: Assignment,
    *,
    task_loads: Sequence[float],
    capacities: Sequence[float],
    compatibility_matrix: Sequence[Sequence[int]] | None = None,
    task_time_windows: Sequence[Sequence[float]] | None = None,
    resource_time_windows: Sequence[Sequence[float]] | None = None,
    resource_stage_levels: Sequence[int] | None = None,
    stage_transitions: Sequence[Sequence[int]] | None = None,
) -> ConstraintBreakdown:
    """返回 per-约束 违反 components 用于 一个 分配."""
    return ConstraintBreakdown(
        capacity=capacity_violation(assignment, task_loads, capacities),
        compatibility=compatibility_violation(assignment, compatibility_matrix),
        time_window=time_window_violation(assignment, task_time_windows, resource_time_windows),
        stage_transition=stage_transition_violation(assignment, resource_stage_levels, stage_transitions),
    )
