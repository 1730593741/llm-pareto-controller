"""Integration test for minimal NSGA-II solver."""

import pytest

from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver


def test_solver_runs_and_returns_annotated_population() -> None:
    cost_matrix = [
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.0],
        [1.0, 1.2],
    ]
    task_loads = [1.0, 1.0, 1.0, 1.0]
    capacities = [3.0, 3.0]

    solver = NSGA2Solver(
        n_tasks=4,
        n_resources=2,
        cost_matrix=cost_matrix,
        task_loads=task_loads,
        capacities=capacities,
        config=NSGA2Config(
            population_size=10,
            generations=5,
            crossover_prob=0.9,
            mutation_prob=0.2,
            seed=42,
        ),
    )

    final_population = solver.run()

    assert len(final_population) == 10
    assert all(len(ind.objectives) == 2 for ind in final_population)
    assert all(ind.rank >= 0 for ind in final_population)
    assert all(ind.constraint_violation >= 0.0 for ind in final_population)


def test_solver_validates_problem_shapes() -> None:
    with pytest.raises(ValueError, match="cost_matrix row count"):
        NSGA2Solver(
            n_tasks=2,
            n_resources=2,
            cost_matrix=[[1.0, 2.0]],
            task_loads=[1.0, 1.0],
            capacities=[2.0, 2.0],
            config=NSGA2Config(),
        )
