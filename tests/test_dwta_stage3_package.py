"""Stage-3 测试 用于 该 DWTA 问题包."""

from __future__ import annotations

import random

from main import ProblemConfig, build_solver
from optimizers.nsga2.solver import NSGA2Config
from problems.dwta.constraints import constraint_breakdown
from problems.dwta.objectives import compute_objectives
from problems.dwta.repair import repair_allocation


def test_dwta_objectives_compute_expected_values() -> None:
    allocation = [[2, 1], [0, 1]]
    remaining_survivability, ammo_consumption = compute_objectives(
        allocation,
        n_weapons=2,
        n_targets=2,
        required_damage=[5.0, 3.0],
        lethality_matrix=[[2.0, 1.0], [1.0, 2.0]],
    )

    assert remaining_survivability == 1.0
    assert ammo_consumption == 4.0


def test_capacity_constraint_violation_detection() -> None:
    breakdown = constraint_breakdown(
        [[3, 2], [0, 1]],
        ammo_capacities=[4, 1],
        compatibility_matrix=[[1, 1], [1, 1]],
        n_targets=2,
    )

    assert breakdown.capacity == 1.0
    assert breakdown.compatibility == 0.0


def test_compatibility_constraint_violation_detection() -> None:
    breakdown = constraint_breakdown(
        [[1, 3], [2, 0]],
        ammo_capacities=[5, 2],
        compatibility_matrix=[[1, 0], [0, 1]],
        n_targets=2,
    )

    assert breakdown.capacity == 0.0
    assert breakdown.compatibility == 5.0


def test_repair_restores_feasible_solution_deterministically() -> None:
    repaired = repair_allocation(
        [[2, 3], [3, 2]],
        ammo_capacities=[3, 2],
        compatibility_matrix=[[1, 1], [1, 0]],
        n_targets=2,
        rng=random.Random(99),
        lethality_matrix=[[2.0, 0.4], [1.5, 0.2]],
        required_damage=[2.0, 1.0],
    )

    breakdown = constraint_breakdown(
        repaired,
        ammo_capacities=[3, 2],
        compatibility_matrix=[[1, 1], [1, 0]],
        n_targets=2,
    )

    assert repaired == [1, 2, 2, 0]
    assert breakdown.total == 0.0


def test_dwta_end_to_end_feasibility_smoke() -> None:
    problem = ProblemConfig.model_validate(
        {
            "problem_type": "dwta",
            "munition_types": [{"id": "m1", "max_range": 12.0, "flight_speed": 3.0, "lethality": 2.0}],
            "weapons": [
                {"id": "w1", "x": 0.0, "y": 0.0, "munition_type_id": "m1", "ammo_capacity": 3},
                {"id": "w2", "x": 2.0, "y": 0.0, "munition_type_id": "m1", "ammo_capacity": 2},
            ],
            "targets": [
                {"id": "t1", "x": 3.0, "y": 0.0, "required_damage": 4.0, "time_window": [0.0, 4.0]},
                {"id": "t2", "x": 7.0, "y": 0.0, "required_damage": 3.0, "time_window": [0.0, 4.0]},
            ],
        }
    )
    solver = build_solver(
        problem,
        NSGA2Config(population_size=10, generations=3, seed=123, mutation_prob=0.3, repair_prob=1.0),
    )

    population = solver.run()

    assert len(population) == 10
    assert all(ind.feasible for ind in population)
    assert all(ind.constraint_violation == 0.0 for ind in population)
