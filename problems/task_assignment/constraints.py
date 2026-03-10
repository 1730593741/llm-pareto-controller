"""Constraint checks for the task-assignment problem."""

from __future__ import annotations

from typing import Sequence

from problems.task_assignment.encoding import Assignment


def resource_loads(assignment: Assignment, task_loads: Sequence[float], n_resources: int) -> list[float]:
    """Aggregate per-resource load from a task assignment."""
    loads = [0.0] * n_resources
    for task_idx, resource_idx in enumerate(assignment):
        loads[resource_idx] += float(task_loads[task_idx])
    return loads


def is_capacity_feasible(assignment: Assignment, task_loads: Sequence[float], capacities: Sequence[float]) -> bool:
    """Return whether all resource capacities are satisfied."""
    loads = resource_loads(assignment, task_loads, len(capacities))
    return all(load <= float(capacity) for load, capacity in zip(loads, capacities))


def capacity_violation(assignment: Assignment, task_loads: Sequence[float], capacities: Sequence[float]) -> float:
    """Compute summed overload amount across all resources."""
    loads = resource_loads(assignment, task_loads, len(capacities))
    return float(sum(max(0.0, load - float(capacity)) for load, capacity in zip(loads, capacities)))
