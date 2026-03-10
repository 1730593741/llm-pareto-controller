"""Minimal NSGA-II solver integrated with task-assignment modules."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from optimizers.nsga2.operators import crossover, mutate
from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import assign_crowding_distance, environmental_select, non_dominated_sort, tournament_select
from problems.task_assignment.constraints import capacity_violation, is_capacity_feasible
from problems.task_assignment.encoding import random_assignment
from problems.task_assignment.objectives import compute_objectives
from problems.task_assignment.repair import repair_overloaded_assignment


@dataclass(slots=True)
class TaskAssignmentProblemData:
    """Input data needed to evaluate task-assignment genomes."""

    cost_matrix: Sequence[Sequence[float]]
    task_loads: Sequence[float]
    capacities: Sequence[float]

    @property
    def n_tasks(self) -> int:
        """Return number of tasks."""
        return len(self.task_loads)

    @property
    def n_resources(self) -> int:
        """Return number of resources."""
        return len(self.capacities)


@dataclass(slots=True)
class NSGA2Config:
    """Configuration for minimal NSGA-II execution."""

    population_size: int = 20
    generations: int = 20
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    seed: int | None = None


class NSGA2Solver:
    """Small, extendable NSGA-II implementation for task assignment."""

    def __init__(self, problem: TaskAssignmentProblemData, config: NSGA2Config) -> None:
        """Create solver with problem data and algorithm configuration."""
        if config.population_size <= 0:
            raise ValueError("population_size must be > 0")
        if config.generations < 0:
            raise ValueError("generations must be >= 0")
        if not (0.0 <= config.crossover_prob <= 1.0):
            raise ValueError("crossover_prob must be in [0, 1]")
        if not (0.0 <= config.mutation_prob <= 1.0):
            raise ValueError("mutation_prob must be in [0, 1]")

        if problem.n_resources == 0:
            raise ValueError("problem must contain at least one resource")
        if len(problem.cost_matrix) != problem.n_tasks:
            raise ValueError("cost_matrix rows must match number of tasks")
        if any(len(row) != problem.n_resources for row in problem.cost_matrix):
            raise ValueError("each cost_matrix row must match number of resources")

        self.problem = problem
        self.config = config
        self.rng = random.Random(config.seed)

    def _evaluate_genome(self, genome: list[int]) -> Individual:
        """Build and evaluate one individual from a genome."""
        objectives = compute_objectives(
            genome,
            self.problem.cost_matrix,
            self.problem.task_loads,
            self.problem.n_resources,
        )
        violation = capacity_violation(genome, self.problem.task_loads, self.problem.capacities)
        feasible = is_capacity_feasible(genome, self.problem.task_loads, self.problem.capacities)
        return Individual(genome=genome, objectives=objectives, constraint_violation=violation, feasible=feasible)

    def initialize_population(self) -> list[Individual]:
        """Initialize and evaluate a random population."""
        population: list[Individual] = []
        for _ in range(self.config.population_size):
            genome = random_assignment(self.problem.n_tasks, self.problem.n_resources, self.rng)
            repaired = repair_overloaded_assignment(genome, self.problem.task_loads, self.problem.capacities)
            population.append(self._evaluate_genome(repaired))

        fronts = non_dominated_sort(population)
        for front in fronts:
            assign_crowding_distance(population, front)
        return population

    def _make_offspring(self, population: list[Individual]) -> list[Individual]:
        """Generate and evaluate offspring with tournament selection + variation."""
        offspring: list[Individual] = []
        while len(offspring) < self.config.population_size:
            parent1 = tournament_select(population, self.rng)
            parent2 = tournament_select(population, self.rng)

            child1_genome, child2_genome = crossover(
                parent1.genome,
                parent2.genome,
                self.config.crossover_prob,
                self.rng,
            )

            child1_genome = mutate(child1_genome, self.problem.n_resources, self.config.mutation_prob, self.rng)
            child2_genome = mutate(child2_genome, self.problem.n_resources, self.config.mutation_prob, self.rng)

            child1_genome = repair_overloaded_assignment(
                child1_genome,
                self.problem.task_loads,
                self.problem.capacities,
            )
            child2_genome = repair_overloaded_assignment(
                child2_genome,
                self.problem.task_loads,
                self.problem.capacities,
            )

            offspring.append(self._evaluate_genome(child1_genome))
            if len(offspring) < self.config.population_size:
                offspring.append(self._evaluate_genome(child2_genome))

        return offspring

    def run(self) -> list[Individual]:
        """Run NSGA-II and return the final population."""
        population = self.initialize_population()

        for _ in range(self.config.generations):
            offspring = self._make_offspring(population)
            combined = population + offspring
            population = environmental_select(combined, self.config.population_size)

        return population
