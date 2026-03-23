"""基础测试 用于 任务-分配 M1 modules."""

from __future__ import annotations

import random

from problems.task_assignment.constraints import (
    capacity_violation,
    constraint_breakdown,
    is_capacity_feasible,
)
from problems.task_assignment.encoding import random_assignment
from problems.task_assignment.objectives import compute_objectives, load_imbalance, total_cost
from problems.task_assignment.repair import repair_overloaded_assignment


def test_random_assignment_shape_and_range() -> None:
    """Random initializer 应生成合法的索引值."""
    assignment = random_assignment(n_tasks=8, n_resources=3, rng=random.Random(7))

    assert len(assignment) == 8
    assert all(0 <= resource < 3 for resource in assignment)


def test_objectives_compute_total_cost_and_imbalance() -> None:
    """目标辅助函数 应返回预期值 用于 一个 small example."""
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
    """Constraint 辅助工具 应能检测超载和违反量."""
    assignment = [0, 0, 1]
    task_loads = [3.0, 2.0, 1.0]
    capacities = [4.0, 2.0]

    assert not is_capacity_feasible(assignment, task_loads, capacities)
    assert capacity_violation(assignment, task_loads, capacities) == 1.0


def test_repair_reduces_or_eliminates_capacity_violation() -> None:
    """Repair 在可能时应产出违反更低或为零的解."""
    assignment = [0, 0, 0, 1]
    task_loads = [2.0, 1.0, 1.0, 1.0]
    capacities = [3.0, 3.0]

    before = capacity_violation(assignment, task_loads, capacities)
    repaired = repair_overloaded_assignment(assignment, task_loads, capacities)
    after = capacity_violation(repaired, task_loads, capacities)

    assert after < before
    assert is_capacity_feasible(repaired, task_loads, capacities)


def test_constraint_breakdown_reports_complex_components() -> None:
    assignment = [0, 0, 1]
    task_loads = [1.5, 1.0, 1.0]
    capacities = [2.0, 2.0]
    compatibility_matrix = [[0, 1], [1, 1], [1, 1]]
    task_time_windows = [[0.0, 1.0], [0.0, 2.0], [0.0, 1.0]]
    resource_time_windows = [[2.0, 3.0], [0.0, 2.0]]
    resource_stage_levels = [2, 1]
    stage_transitions = [[0, 2]]

    breakdown = constraint_breakdown(
        assignment,
        task_loads=task_loads,
        capacities=capacities,
        compatibility_matrix=compatibility_matrix,
        task_time_windows=task_time_windows,
        resource_time_windows=resource_time_windows,
        resource_stage_levels=resource_stage_levels,
        stage_transitions=stage_transitions,
    )

    assert breakdown.capacity > 0.0
    assert breakdown.compatibility > 0.0
    assert breakdown.time_window > 0.0
    assert breakdown.stage_transition > 0.0
    assert breakdown.total == breakdown.capacity + breakdown.compatibility + breakdown.time_window + breakdown.stage_transition


def test_repair_handles_complex_constraints() -> None:
    assignment = [0, 0, 0, 0]
    task_loads = [1.0, 1.0, 1.0, 1.0]
    capacities = [1.5, 2.5]
    compatibility_matrix = [[0, 1], [1, 1], [1, 1], [1, 1]]
    task_time_windows = [[0.0, 1.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]
    resource_time_windows = [[2.0, 3.0], [0.0, 3.0]]
    resource_stage_levels = [2, 3]
    stage_transitions = [[0, 1], [0, 2], [0, 3]]

    before = constraint_breakdown(
        assignment,
        task_loads=task_loads,
        capacities=capacities,
        compatibility_matrix=compatibility_matrix,
        task_time_windows=task_time_windows,
        resource_time_windows=resource_time_windows,
        resource_stage_levels=resource_stage_levels,
        stage_transitions=stage_transitions,
    )
    repaired = repair_overloaded_assignment(
        assignment,
        task_loads,
        capacities,
        compatibility_matrix=compatibility_matrix,
        task_time_windows=task_time_windows,
        resource_time_windows=resource_time_windows,
        resource_stage_levels=resource_stage_levels,
        stage_transitions=stage_transitions,
    )
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

    assert after.total < before.total
