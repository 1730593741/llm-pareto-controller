"""Basic tests for minimal NSGA-II components."""

from __future__ import annotations

import random

import pytest

from optimizers.nsga2.operators import crossover, mutate
from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import environmental_select, non_dominated_sort, tournament_select
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver, TaskAssignmentProblemData
from problems.task_assignment.constraints import capacity_violation


def test_operators_keep_assignment_bounds() -> None:
    """Crossover and mutation should keep valid resource indices."""
    rng = random.Random(42)
    p1 = [0, 0, 0, 0]
    p2 = [1, 1, 1, 1]

    c1, c2 = crossover(p1, p2, crossover_prob=1.0, rng=rng)
    m1 = mutate(c1, n_resources=2, mutation_prob=1.0, rng=rng)
    m2 = mutate(c2, n_resources=2, mutation_prob=1.0, rng=rng)

    assert all(g in (0, 1) for g in m1)
    assert all(g in (0, 1) for g in m2)


def test_non_dominated_sort_and_environmental_selection_size() -> None:
    """Selection utilities should produce fronts and fixed-size populations."""
    problem = TaskAssignmentProblemData(
        cost_matrix=[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
        task_loads=[1.0, 1.0, 1.0],
        capacities=[2.0, 2.0],
    )
    solver = NSGA2Solver(problem, NSGA2Config(population_size=6, generations=1, seed=0))
    pop = solver.initialize_population()

    fronts = non_dominated_sort(pop)
    assert len(fronts) >= 1
    next_pop = environmental_select(pop + pop, population_size=6)
    assert len(next_pop) == 6


def test_operator_parameter_validation() -> None:
    """Operators should reject invalid probability/resource inputs."""
    rng = random.Random(0)

    with pytest.raises(ValueError):
        crossover([0, 1], [1, 0], crossover_prob=1.5, rng=rng)

    with pytest.raises(ValueError):
        mutate([0, 1], n_resources=0, mutation_prob=0.2, rng=rng)

    with pytest.raises(ValueError):
        mutate([0, 1], n_resources=2, mutation_prob=-0.1, rng=rng)


def test_selection_edge_cases() -> None:
    """Selection helpers should handle empty/singleton edge cases safely."""
    assert non_dominated_sort([]) == []

    only = Individual(genome=[0], rank=0, crowding_distance=0.0)
    chosen = tournament_select([only], random.Random(1))
    assert chosen is only

    with pytest.raises(ValueError):
        tournament_select([], random.Random(1))

    with pytest.raises(ValueError):
        environmental_select([only], population_size=0)


def test_nsga2_solver_smoke_run_multiple_generations() -> None:
    """Solver should run several generations and return evaluated individuals."""
    problem = TaskAssignmentProblemData(
        cost_matrix=[
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        task_loads=[1.0, 2.0, 1.0, 1.0],
        capacities=[3.0, 3.0, 3.0],
    )
    config = NSGA2Config(population_size=10, generations=4, crossover_prob=0.9, mutation_prob=0.2, seed=123)
    solver = NSGA2Solver(problem, config)

    final_population = solver.run()

    assert len(final_population) == config.population_size
    for individual in final_population:
        assert len(individual.genome) == problem.n_tasks
        assert len(individual.objectives) == 2
        assert individual.constraint_violation == capacity_violation(
            individual.genome,
            problem.task_loads,
            problem.capacities,
        )
