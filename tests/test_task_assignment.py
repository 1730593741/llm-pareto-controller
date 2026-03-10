"""Basic tests for task-assignment M1 modules."""

from __future__ import annotations

import random

from problems.task_assignment.constraints import capacity_violation, is_capacity_feasible
from problems.task_assignment.encoding import random_assignment
from problems.task_assignment.objectives import compute_objectives, load_imbalance, total_cost
from problems.task_assignment.repair import repair_overloaded_assignment


def test_random_assignment_shape_and_range() -> None:
    """Random initializer should produce valid index values."""
    assignment = random_assignment(n_tasks=8, n_resources=3, rng=random.Random(7))

    assert len(assignment) == 8
    assert all(0 <= resource < 3 for resource in assignment)


def test_objectives_compute_total_cost_and_imbalance() -> None:
    """Objective helpers should return expected values for a small example."""
    assignment = [0, 1, 1]
    cost_matrix = [
        [1.0, 3.0],
        [2.0, 1.0],
        [4.0, 2.0],
    ]
    task_loads = [2.0, 1.0, 3.0]

    assert total_cost(assignment, cost_matrix) == 4.0
    assert load_imbalance(assignment, task_loads, n_resources=2) == 2.0
    assert compute_objectives(assignment, cost_matrix, task_loads, n_resources=2) == (4.0, 2.0)


def test_constraints_feasibility_and_violation() -> None:
    """Constraint helpers should detect overload and violation amount."""
    assignment = [0, 0, 1]
    task_loads = [3.0, 2.0, 1.0]
    capacities = [4.0, 2.0]

    assert not is_capacity_feasible(assignment, task_loads, capacities)
    assert capacity_violation(assignment, task_loads, capacities) == 1.0


def test_repair_reduces_or_eliminates_capacity_violation() -> None:
    """Repair should produce a solution with lower or zero violation when possible."""
    assignment = [0, 0, 0, 1]
    task_loads = [2.0, 1.0, 1.0, 1.0]
    capacities = [3.0, 3.0]

    before = capacity_violation(assignment, task_loads, capacities)
    repaired = repair_overloaded_assignment(assignment, task_loads, capacities)
    after = capacity_violation(repaired, task_loads, capacities)

    assert after < before
    assert is_capacity_feasible(repaired, task_loads, capacities)
