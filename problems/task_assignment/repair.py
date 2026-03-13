"""Simple repair heuristics for constrained task assignments."""

from __future__ import annotations

from typing import Sequence

from problems.task_assignment.constraints import (
    constraint_breakdown,
    resource_loads,
)
from problems.task_assignment.encoding import Assignment


def _task_stage_penalty(
    assignment: Assignment,
    *,
    task_idx: int,
    candidate_resource: int,
    resource_stage_levels: Sequence[int] | None,
    stage_transitions: Sequence[Sequence[int]] | None,
) -> float:
    if resource_stage_levels is None or stage_transitions is None:
        return 0.0

    penalty = 0.0
    candidate_stage = int(resource_stage_levels[candidate_resource])
    for predecessor_task, successor_task in stage_transitions:
        if predecessor_task == task_idx:
            successor_stage = int(resource_stage_levels[assignment[successor_task]])
            penalty += max(0, candidate_stage - successor_stage)
        elif successor_task == task_idx:
            predecessor_stage = int(resource_stage_levels[assignment[predecessor_task]])
            penalty += max(0, predecessor_stage - candidate_stage)
    return float(penalty)


def _best_resource_for_task(
    assignment: Assignment,
    *,
    task_idx: int,
    task_loads: Sequence[float],
    capacities: Sequence[float],
    compatibility_matrix: Sequence[Sequence[int]] | None,
    task_time_windows: Sequence[Sequence[float]] | None,
    resource_time_windows: Sequence[Sequence[float]] | None,
    resource_stage_levels: Sequence[int] | None,
    stage_transitions: Sequence[Sequence[int]] | None,
) -> int:
    current_resource = assignment[task_idx]
    loads = resource_loads(assignment, task_loads, len(capacities))
    task_load = float(task_loads[task_idx])

    best_resource = current_resource
    best_score: tuple[float, float, float, float] | None = None

    for candidate_resource in range(len(capacities)):
        compatibility_penalty = 0.0
        if compatibility_matrix is not None and int(compatibility_matrix[task_idx][candidate_resource]) == 0:
            compatibility_penalty = 1.0

        time_window_penalty = 0.0
        if task_time_windows is not None and resource_time_windows is not None:
            task_window = task_time_windows[task_idx]
            resource_window = resource_time_windows[candidate_resource]
            if float(task_window[1]) < float(resource_window[0]):
                time_window_penalty = float(resource_window[0]) - float(task_window[1])
            elif float(resource_window[1]) < float(task_window[0]):
                time_window_penalty = float(task_window[0]) - float(resource_window[1])

        projected_load = loads[candidate_resource] + task_load
        if candidate_resource == current_resource:
            projected_load = loads[candidate_resource]
        capacity_penalty = max(0.0, projected_load - float(capacities[candidate_resource]))

        stage_penalty = _task_stage_penalty(
            assignment,
            task_idx=task_idx,
            candidate_resource=candidate_resource,
            resource_stage_levels=resource_stage_levels,
            stage_transitions=stage_transitions,
        )

        score = (compatibility_penalty, time_window_penalty, stage_penalty, capacity_penalty)
        if best_score is None or score < best_score:
            best_score = score
            best_resource = candidate_resource

    return best_resource


def repair_overloaded_assignment(
    assignment: Assignment,
    task_loads: Sequence[float],
    capacities: Sequence[float],
    compatibility_matrix: Sequence[Sequence[int]] | None = None,
    task_time_windows: Sequence[Sequence[float]] | None = None,
    resource_time_windows: Sequence[Sequence[float]] | None = None,
    resource_stage_levels: Sequence[int] | None = None,
    stage_transitions: Sequence[Sequence[int]] | None = None,
) -> Assignment:
    """Repair assignment violations with a deterministic greedy pass."""
    repaired = list(assignment)

    max_rounds = max(1, len(repaired) * 4)
    for _ in range(max_rounds):
        before = constraint_breakdown(
            repaired,
            task_loads=task_loads,
            capacities=capacities,
            compatibility_matrix=compatibility_matrix,
            task_time_windows=task_time_windows,
            resource_time_windows=resource_time_windows,
            resource_stage_levels=resource_stage_levels,
            stage_transitions=stage_transitions,
        )

        changed = False
        for task_idx in range(len(repaired)):
            target_resource = _best_resource_for_task(
                repaired,
                task_idx=task_idx,
                task_loads=task_loads,
                capacities=capacities,
                compatibility_matrix=compatibility_matrix,
                task_time_windows=task_time_windows,
                resource_time_windows=resource_time_windows,
                resource_stage_levels=resource_stage_levels,
                stage_transitions=stage_transitions,
            )
            if target_resource != repaired[task_idx]:
                repaired[task_idx] = target_resource
                changed = True

        after = constraint_breakdown(
            repaired,
            task_loads=task_loads,
            capacities=capacities,
            compatibility_matrix=compatibility_matrix,
            task_time_windows=task_time_windows,
            resource_time_windows=resource_time_windows,
            resource_stage_levels=resource_stage_levels,
            stage_transitions=stage_transitions,
        )

        if after.total == 0.0:
            break
        if not changed or after.total >= before.total:
            break

    return repaired
