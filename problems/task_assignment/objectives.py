"""Objective functions for the task-assignment problem."""

from __future__ import annotations

from typing import Sequence

from problems.task_assignment.encoding import Assignment


def total_cost(assignment: Assignment, cost_matrix: Sequence[Sequence[float]]) -> float:
    """Compute total assignment cost.

    Args:
        assignment: Resource index for each task.
        cost_matrix: 2D matrix where ``cost_matrix[task][resource]`` is cost.

    Returns:
        Sum of selected task-resource costs.
    """
    return float(sum(cost_matrix[task_idx][resource_idx] for task_idx, resource_idx in enumerate(assignment)))


def load_imbalance(assignment: Assignment, task_loads: Sequence[float], n_resources: int) -> float:
    """Compute load imbalance as ``max_resource_load - min_resource_load``.

    Args:
        assignment: Resource index for each task.
        task_loads: Load demand for each task.
        n_resources: Number of resources.

    Returns:
        Difference between max and min aggregate resource loads.
    """
    loads = [0.0] * n_resources
    for task_idx, resource_idx in enumerate(assignment):
        loads[resource_idx] += float(task_loads[task_idx])

    return max(loads) - min(loads) if loads else 0.0


def compute_objectives(
    assignment: Assignment,
    cost_matrix: Sequence[Sequence[float]],
    task_loads: Sequence[float],
    n_resources: int,
) -> tuple[float, float]:
    """Compute the two optimization objectives: total cost and load imbalance."""
    return total_cost(assignment, cost_matrix), load_imbalance(assignment, task_loads, n_resources)
