"""Simple repair heuristics for overloaded task assignments."""

from __future__ import annotations

from typing import Sequence

from problems.task_assignment.constraints import resource_loads
from problems.task_assignment.encoding import Assignment


def repair_overloaded_assignment(
    assignment: Assignment,
    task_loads: Sequence[float],
    capacities: Sequence[float],
) -> Assignment:
    """Repair an assignment by moving tasks away from overloaded resources.

    The strategy is intentionally simple and deterministic:
    - For each overloaded resource, inspect tasks assigned to it in descending load.
    - Move a task to the first resource with enough remaining capacity.
    - Stop when no resource is overloaded or no further move is possible.

    Args:
        assignment: Current task-resource assignment.
        task_loads: Per-task loads.
        capacities: Per-resource capacities.

    Returns:
        A repaired assignment copy. If no feasible move exists for some overload,
        the function returns the best effort partially repaired assignment.
    """
    repaired = list(assignment)
    n_resources = len(capacities)
    loads = resource_loads(repaired, task_loads, n_resources)

    changed = True
    while changed:
        changed = False
        for overloaded_resource, (load, capacity) in enumerate(zip(loads, capacities)):
            overload = load - float(capacity)
            if overload <= 0:
                continue

            assigned_tasks = [i for i, r in enumerate(repaired) if r == overloaded_resource]
            assigned_tasks.sort(key=lambda task_idx: float(task_loads[task_idx]), reverse=True)

            for task_idx in assigned_tasks:
                task_load = float(task_loads[task_idx])
                moved = False
                for candidate_resource in range(n_resources):
                    if candidate_resource == overloaded_resource:
                        continue
                    if loads[candidate_resource] + task_load <= float(capacities[candidate_resource]):
                        repaired[task_idx] = candidate_resource
                        loads[overloaded_resource] -= task_load
                        loads[candidate_resource] += task_load
                        changed = True
                        moved = True
                        break
                if moved and loads[overloaded_resource] <= float(capacity):
                    break

    return repaired
