"""Minimal runnable NSGA-II solver for the task-assignment problem."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from controller.operator_space import OperatorCapabilities, OperatorParams
from optimizers.nsga2.operators import mutate_assignment, one_point_crossover
from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import (
    assign_crowding_distance,
    environmental_selection,
    non_dominated_sort,
    parent_selection,
)
from problems.task_assignment.constraints import capacity_violation
from problems.task_assignment.encoding import random_assignment
from problems.task_assignment.objectives import compute_objectives
from problems.task_assignment.repair import repair_overloaded_assignment


@dataclass(slots=True)
class NSGA2Config:
    """Config for minimal NSGA-II loop."""

    population_size: int = 40
    generations: int = 30
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    repair_prob: float = 1.0
    eta_c: float | None = None
    eta_m: float | None = None
    local_search_prob: float | None = None
    seed: int | None = None


class NSGA2Solver:
    """NSGA-II solver with injectable problem arrays for task assignment."""

    def __init__(
        self,
        *,
        n_tasks: int,
        n_resources: int,
        cost_matrix: Sequence[Sequence[float]],
        task_loads: Sequence[float],
        capacities: Sequence[float],
        config: NSGA2Config,
    ) -> None:
        if config.population_size <= 1:
            raise ValueError("population_size must be > 1")
        if config.generations < 0:
            raise ValueError("generations must be >= 0")
        if not 0.0 <= config.crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be in [0, 1]")
        if not 0.0 <= config.mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be in [0, 1]")
        if not 0.0 <= config.repair_prob <= 1.0:
            raise ValueError("repair_prob must be in [0, 1]")
        if n_tasks < 0:
            raise ValueError("n_tasks must be >= 0")
        if n_resources <= 0:
            raise ValueError("n_resources must be > 0")
        if len(cost_matrix) != n_tasks:
            raise ValueError("cost_matrix row count must equal n_tasks")
        if len(task_loads) != n_tasks:
            raise ValueError("task_loads length must equal n_tasks")
        if len(capacities) != n_resources:
            raise ValueError("capacities length must equal n_resources")
        if any(len(row) != n_resources for row in cost_matrix):
            raise ValueError("each cost_matrix row length must equal n_resources")

        self.n_tasks = n_tasks
        self.n_resources = n_resources
        self.cost_matrix = cost_matrix
        self.task_loads = task_loads
        self.capacities = capacities
        self.config = config
        self.rng = random.Random(config.seed)

    def get_operator_capabilities(self) -> OperatorCapabilities:
        """Return runtime-supported parameter dimensions."""
        return OperatorCapabilities(
            supports_eta_c=False,
            supports_eta_m=False,
            supports_repair_prob=True,
            supports_local_search_prob=False,
        )

    def get_operator_params(self) -> OperatorParams:
        """Return current unified parameter snapshot."""
        return OperatorParams(
            mutation_prob=self.config.mutation_prob,
            crossover_prob=self.config.crossover_prob,
            eta_c=self.config.eta_c,
            eta_m=self.config.eta_m,
            repair_prob=self.config.repair_prob,
            local_search_prob=self.config.local_search_prob,
        )

    def set_operator_params(self, params: OperatorParams) -> None:
        """Update runtime parameters with capability-aware application."""
        if not 0.0 <= params.crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be in [0, 1]")
        if not 0.0 <= params.mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be in [0, 1]")
        self.config.mutation_prob = params.mutation_prob
        self.config.crossover_prob = params.crossover_prob

        capabilities = self.get_operator_capabilities()
        if capabilities.supports_repair_prob and params.repair_prob is not None:
            if not 0.0 <= params.repair_prob <= 1.0:
                raise ValueError("repair_prob must be in [0, 1]")
            self.config.repair_prob = params.repair_prob

    def set_operator_probs(self, *, mutation_prob: float, crossover_prob: float) -> None:
        """Backward-compatible API for M4/M5 code paths."""
        self.set_operator_params(
            OperatorParams(
                mutation_prob=mutation_prob,
                crossover_prob=crossover_prob,
                repair_prob=self.config.repair_prob,
            )
        )

    def _evaluate(self, genome: list[int]) -> Individual:
        objectives = compute_objectives(
            assignment=genome,
            cost_matrix=self.cost_matrix,
            task_loads=self.task_loads,
            n_resources=self.n_resources,
        )
        cv = capacity_violation(genome, self.task_loads, self.capacities)
        return Individual(
            genome=genome,
            objectives=objectives,
            constraint_violation=cv,
            feasible=cv <= 0.0,
        )

    def initialize_population(self) -> list[Individual]:
        """Create an initial repaired population for iterative closed-loop use."""
        return [
            self._evaluate(
                repair_overloaded_assignment(
                    random_assignment(self.n_tasks, self.n_resources, self.rng),
                    self.task_loads,
                    self.capacities,
                )
            )
            for _ in range(self.config.population_size)
        ]

    def _annotate_population(self, individuals: list[Individual]) -> None:
        fronts = non_dominated_sort(individuals)
        for front in fronts:
            assign_crowding_distance(front)

    def _make_offspring(self, population: list[Individual]) -> list[Individual]:
        self._annotate_population(population)
        parents = parent_selection(population, self.config.population_size, self.rng)

        offspring: list[Individual] = []
        for idx in range(0, len(parents), 2):
            parent_a = parents[idx]
            parent_b = parents[(idx + 1) % len(parents)]

            child_a, child_b = one_point_crossover(
                parent_a.genome,
                parent_b.genome,
                crossover_prob=self.config.crossover_prob,
                rng=self.rng,
            )
            child_a = mutate_assignment(child_a, self.n_resources, self.config.mutation_prob, self.rng)
            child_b = mutate_assignment(child_b, self.n_resources, self.config.mutation_prob, self.rng)

            repaired_a = child_a
            repaired_b = child_b
            if self.rng.random() < self.config.repair_prob:
                repaired_a = repair_overloaded_assignment(child_a, self.task_loads, self.capacities)
            if self.rng.random() < self.config.repair_prob:
                repaired_b = repair_overloaded_assignment(child_b, self.task_loads, self.capacities)

            offspring.append(self._evaluate(repaired_a))
            if len(offspring) < self.config.population_size:
                offspring.append(self._evaluate(repaired_b))

        return offspring[: self.config.population_size]

    def evolve_one_generation(self, population: list[Individual]) -> list[Individual]:
        """Run a single evolutionary generation and return selected population."""
        offspring = self._make_offspring(population)
        merged = population + offspring
        return environmental_selection(merged, self.config.population_size)

    def run(self) -> list[Individual]:
        """Run NSGA-II and return final population."""
        population = self.initialize_population()

        for _ in range(self.config.generations):
            population = self.evolve_one_generation(population)

        self._annotate_population(population)
        return population
