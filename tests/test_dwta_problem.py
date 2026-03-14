"""Tests for the DWTA benchmark package and solver integration."""

from __future__ import annotations

from main import ProblemConfig, build_solver
from optimizers.nsga2.solver import NSGA2Config
from problems.dwta.constraints import constraint_breakdown
from problems.dwta.objectives import compute_objectives
from problems.dwta.precompute import build_precomputed_matrices
from problems.dwta.repair import repair_allocation
from problems.dwta.model import MunitionType, Target, Weapon


def test_dwta_precompute_builds_compatibility_and_lethality() -> None:
    data = build_precomputed_matrices(
        munitions=[MunitionType(id="m1", max_range=5.0, flight_speed=2.5, lethality=2.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=4)],
        targets=[
            Target(id="t1", x=3.0, y=0.0, required_damage=3.0, time_window=(0.0, 5.0)),
            Target(id="t2", x=8.0, y=0.0, required_damage=3.0, time_window=(0.0, 5.0)),
        ],
    )

    assert data.compatibility_matrix == [[1, 0]]
    assert data.lethality_matrix == [[2.0, 2.0]]


def test_dwta_objectives_and_constraints_match_definition() -> None:
    genome = [2, 1, 0, 1]
    objectives = compute_objectives(
        genome,
        n_weapons=2,
        n_targets=2,
        required_damage=[5.0, 3.0],
        lethality_matrix=[[2.0, 1.0], [1.0, 2.0]],
    )
    breakdown = constraint_breakdown(
        genome,
        ammo_capacities=[3, 1],
        compatibility_matrix=[[1, 1], [1, 0]],
        n_targets=2,
    )

    assert objectives == (1.0, 4.0)
    assert breakdown.capacity == 0.0
    assert breakdown.compatibility == 1.0


def test_dwta_repair_enforces_capacity_and_compatibility() -> None:
    repaired = repair_allocation(
        [2, 2, 0, 2],
        ammo_capacities=[3, 1],
        compatibility_matrix=[[1, 1], [1, 0]],
        n_targets=2,
        rng=__import__("random").Random(7),
    )
    breakdown = constraint_breakdown(
        repaired,
        ammo_capacities=[3, 1],
        compatibility_matrix=[[1, 1], [1, 0]],
        n_targets=2,
    )
    assert breakdown.total == 0.0


def test_build_solver_supports_dwta_problem_type() -> None:
    problem = ProblemConfig.model_validate(
        {
            "problem_type": "dwta",
            "munitions": [{"id": "m1", "max_range": 12.0, "flight_speed": 3.0, "lethality": 2.0}],
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
    solver = build_solver(problem, NSGA2Config(population_size=8, generations=2, seed=3))

    final_population = solver.run()

    assert len(final_population) == 8
    assert all(len(ind.genome) == 4 for ind in final_population)
    assert any("compatibility" in ind.constraint_components for ind in final_population)
