"""Minimal but extensible NSGA-II solver used across project problem types.

The solver currently supports two families of benchmarks:

- generic task-assignment instances;
- DWTA instances represented by scenario matrices.

The closed-loop controller interacts with this module through a small,
capability-aware operator interface so that online parameter adjustment remains
possible without coupling the controller to problem-specific details.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from controller.operator_space import OperatorCapabilities, OperatorParams
from optimizers.nsga2.operators import (
    matrix_block_crossover,
    mutate_assignment,
    mutate_dwta_allocation,
    one_point_crossover,
)
from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import (
    assign_crowding_distance,
    environmental_selection,
    non_dominated_sort,
    parent_selection,
)
from problems.dwta.constraints import constraint_breakdown as dwta_constraint_breakdown
from problems.dwta.encoding import random_allocation
from problems.dwta.model import DWTABenchmarkData
from problems.dwta.objectives import compute_objectives as compute_dwta_objectives
from problems.dwta.repair import repair_allocation
from problems.task_assignment.constraints import constraint_breakdown
from problems.task_assignment.encoding import random_assignment
from problems.task_assignment.objectives import compute_objectives
from problems.task_assignment.repair import repair_overloaded_assignment


@dataclass(slots=True)
class NSGA2Config:
    """Configuration for the minimal NSGA-II loop."""

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
    """NSGA-II solver for task-assignment and DWTA problems.

    The constructor normalizes the two supported problem families into a single
    runtime object. Downstream code can therefore initialize populations,
    evaluate individuals, and update operator parameters without branching on
    the underlying benchmark type.
    """

    def __init__(
        self,
        *,
        n_tasks: int,
        n_resources: int,
        cost_matrix: Sequence[Sequence[float]],
        task_loads: Sequence[float],
        capacities: Sequence[float],
        task_time_windows: Sequence[Sequence[float]] | None = None,
        resource_time_windows: Sequence[Sequence[float]] | None = None,
        compatibility_matrix: Sequence[Sequence[int]] | None = None,
        resource_stage_levels: Sequence[int] | None = None,
        stage_transitions: Sequence[Sequence[int]] | None = None,
        config: NSGA2Config,
        dwta_data: DWTABenchmarkData | None = None,
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

        self.config = config
        self.rng = random.Random(config.seed)
        self.dwta_data = dwta_data

        # DWTA uses a benchmark object instead of generic task-assignment arrays.
        # We still populate a compatible solver shape so the controller and
        # logging layers can query the solver through a uniform interface.
        if self.dwta_data is not None:
            self.problem_mode = "dwta"
            self.n_tasks = self.dwta_data.n_weapons * self.dwta_data.n_targets
            self.n_resources = 1
            self.cost_matrix = [[0.0] for _ in range(self.n_tasks)]
            self.task_loads = [0.0] * self.n_tasks
            self.capacities = [float(value) for value in self.dwta_data.ammo_capacities]
            self.task_time_windows = None
            self.resource_time_windows = None
            self.compatibility_matrix = None
            self.resource_stage_levels = None
            self.stage_transitions = None
            return

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

        self.problem_mode = "task_assignment"
        self.n_tasks = n_tasks
        self.n_resources = n_resources
        self.cost_matrix = cost_matrix
        self.task_loads = task_loads
        self.capacities = capacities
        self.task_time_windows = task_time_windows
        self.resource_time_windows = resource_time_windows
        self.compatibility_matrix = compatibility_matrix
        self.resource_stage_levels = resource_stage_levels
        self.stage_transitions = stage_transitions

    def get_operator_capabilities(self) -> OperatorCapabilities:
        """Describe which operator dimensions are meaningful for this problem.

        The controller uses this metadata to avoid proposing adjustments that
        would have no effect in the active solver/problem combination.
        """
        if self.dwta_data is not None:
            return OperatorCapabilities(
                supports_eta_c=True,
                supports_eta_m=True,
                supports_repair_prob=True,
                supports_local_search_prob=True,
            )
        return OperatorCapabilities(
            supports_eta_c=False,
            supports_eta_m=False,
            supports_repair_prob=True,
            supports_local_search_prob=False,
        )

    def get_operator_params(self) -> OperatorParams:
        """Return a unified snapshot of the current operator parameters."""
        return OperatorParams(
            mutation_prob=self.config.mutation_prob,
            crossover_prob=self.config.crossover_prob,
            eta_c=self.config.eta_c,
            eta_m=self.config.eta_m,
            repair_prob=self.config.repair_prob,
            local_search_prob=self.config.local_search_prob,
        )

    def set_operator_params(self, params: OperatorParams) -> None:
        """Update runtime parameters using capability-aware application rules."""
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
        if capabilities.supports_eta_c:
            self.config.eta_c = params.eta_c
        if capabilities.supports_eta_m:
            self.config.eta_m = params.eta_m
        if capabilities.supports_local_search_prob and params.local_search_prob is not None:
            if not 0.0 <= params.local_search_prob <= 1.0:
                raise ValueError("local_search_prob must be in [0, 1]")
            self.config.local_search_prob = params.local_search_prob

    def set_operator_probs(self, *, mutation_prob: float, crossover_prob: float) -> None:
        """Backward-compatible wrapper retained for early controller code paths."""
        self.set_operator_params(
            OperatorParams(
                mutation_prob=mutation_prob,
                crossover_prob=crossover_prob,
                repair_prob=self.config.repair_prob,
            )
        )

    def _evaluate(self, genome: list[int]) -> Individual:
        """Evaluate one genome and package the result as an ``Individual``."""
        if self.dwta_data is not None:
            objectives = compute_dwta_objectives(
                genome,
                n_weapons=self.dwta_data.n_weapons,
                n_targets=self.dwta_data.n_targets,
                required_damage=self.dwta_data.required_damage,
                lethality_matrix=self.dwta_data.lethality_matrix,
            )
            breakdown = dwta_constraint_breakdown(
                genome,
                ammo_capacities=self.dwta_data.ammo_capacities,
                compatibility_matrix=self.dwta_data.compatibility_matrix,
                n_targets=self.dwta_data.n_targets,
            )
            cv = breakdown.total
            return Individual(
                genome=genome,
                objectives=objectives,
                constraint_violation=cv,
                feasible=cv <= 0.0,
                constraint_components={
                    "capacity": breakdown.capacity,
                    "compatibility": breakdown.compatibility,
                    "time_window": 0.0,
                    "stage_transition": 0.0,
                },
            )

        objectives = compute_objectives(
            assignment=genome,
            cost_matrix=self.cost_matrix,
            task_loads=self.task_loads,
            n_resources=self.n_resources,
        )
        breakdown = constraint_breakdown(
            genome,
            task_loads=self.task_loads,
            capacities=self.capacities,
            compatibility_matrix=self.compatibility_matrix,
            task_time_windows=self.task_time_windows,
            resource_time_windows=self.resource_time_windows,
            resource_stage_levels=self.resource_stage_levels,
            stage_transitions=self.stage_transitions,
        )
        cv = breakdown.total
        return Individual(
            genome=genome,
            objectives=objectives,
            constraint_violation=cv,
            feasible=cv <= 0.0,
            constraint_components={
                "capacity": breakdown.capacity,
                "compatibility": breakdown.compatibility,
                "time_window": breakdown.time_window,
                "stage_transition": breakdown.stage_transition,
            },
        )

    def initialize_population(self) -> list[Individual]:
        """Create the initial repaired population used by the closed loop."""
        if self.dwta_data is not None:
            return [
                self._evaluate(
                    repair_allocation(
                        random_allocation(
                            self.dwta_data.ammo_capacities,
                            self.dwta_data.compatibility_matrix,
                            self.rng,
                        ),
                        ammo_capacities=self.dwta_data.ammo_capacities,
                        compatibility_matrix=self.dwta_data.compatibility_matrix,
                        n_targets=self.dwta_data.n_targets,
                        rng=self.rng,
                    )
                )
                for _ in range(self.config.population_size)
            ]

        return [
            self._evaluate(
                repair_overloaded_assignment(
                    random_assignment(self.n_tasks, self.n_resources, self.rng),
                    self.task_loads,
                    self.capacities,
                    compatibility_matrix=self.compatibility_matrix,
                    task_time_windows=self.task_time_windows,
                    resource_time_windows=self.resource_time_windows,
                    resource_stage_levels=self.resource_stage_levels,
                    stage_transitions=self.stage_transitions,
                )
            )
            for _ in range(self.config.population_size)
        ]

    def _annotate_population(self, individuals: list[Individual]) -> None:
        """Refresh NSGA-II rank/crowding annotations in-place for a population."""
        fronts = non_dominated_sort(individuals)
        for front in fronts:
            assign_crowding_distance(front)

    def _make_offspring(self, population: list[Individual]) -> list[Individual]:
        """Generate one offspring batch using selection, variation, and repair.

        For DWTA problems, the controller-adjustable operator knobs map to a
        matrix-aware variation process:

        - ``mutation_prob`` controls per-cell mutation trigger rate;
        - ``eta_m`` shapes mutation locality;
        - ``local_search_prob`` enables an additional row-level improvement move;
        - ``eta_c`` biases matrix block crossover behavior;
        - ``repair_prob`` keeps its standard post-variation feasibility role.
        """
        self._annotate_population(population)
        parents = parent_selection(population, self.config.population_size, self.rng)

        dwta_mutation_step = max(1.0, self.config.mutation_prob * 10.0)

        offspring: list[Individual] = []
        for idx in range(0, len(parents), 2):
            parent_a = parents[idx]
            parent_b = parents[(idx + 1) % len(parents)]

            if self.dwta_data is not None:
                child_a, child_b = matrix_block_crossover(
                    parent_a.genome,
                    parent_b.genome,
                    n_weapons=self.dwta_data.n_weapons,
                    n_targets=self.dwta_data.n_targets,
                    crossover_prob=self.config.crossover_prob,
                    rng=self.rng,
                    eta_c=self.config.eta_c,
                )
                child_a = mutate_dwta_allocation(
                    child_a,
                    n_weapons=self.dwta_data.n_weapons,
                    n_targets=self.dwta_data.n_targets,
                    compatibility_matrix=self.dwta_data.compatibility_matrix,
                    mutation_prob=self.config.mutation_prob,
                    rng=self.rng,
                    mutation_step=dwta_mutation_step,
                    eta_m=self.config.eta_m,
                    local_search_prob=self.config.local_search_prob,
                )
                child_b = mutate_dwta_allocation(
                    child_b,
                    n_weapons=self.dwta_data.n_weapons,
                    n_targets=self.dwta_data.n_targets,
                    compatibility_matrix=self.dwta_data.compatibility_matrix,
                    mutation_prob=self.config.mutation_prob,
                    rng=self.rng,
                    mutation_step=dwta_mutation_step,
                    eta_m=self.config.eta_m,
                    local_search_prob=self.config.local_search_prob,
                )
            else:
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
                if self.dwta_data is not None:
                    repaired_a = repair_allocation(
                        child_a,
                        ammo_capacities=self.dwta_data.ammo_capacities,
                        compatibility_matrix=self.dwta_data.compatibility_matrix,
                        n_targets=self.dwta_data.n_targets,
                        rng=self.rng,
                    )
                else:
                    repaired_a = repair_overloaded_assignment(
                        child_a,
                        self.task_loads,
                        self.capacities,
                        compatibility_matrix=self.compatibility_matrix,
                        task_time_windows=self.task_time_windows,
                        resource_time_windows=self.resource_time_windows,
                        resource_stage_levels=self.resource_stage_levels,
                        stage_transitions=self.stage_transitions,
                    )
            if self.rng.random() < self.config.repair_prob:
                if self.dwta_data is not None:
                    repaired_b = repair_allocation(
                        child_b,
                        ammo_capacities=self.dwta_data.ammo_capacities,
                        compatibility_matrix=self.dwta_data.compatibility_matrix,
                        n_targets=self.dwta_data.n_targets,
                        rng=self.rng,
                    )
                else:
                    repaired_b = repair_overloaded_assignment(
                        child_b,
                        self.task_loads,
                        self.capacities,
                        compatibility_matrix=self.compatibility_matrix,
                        task_time_windows=self.task_time_windows,
                        resource_time_windows=self.resource_time_windows,
                        resource_stage_levels=self.resource_stage_levels,
                        stage_transitions=self.stage_transitions,
                    )

            offspring.append(self._evaluate(repaired_a))
            if len(offspring) < self.config.population_size:
                offspring.append(self._evaluate(repaired_b))

        return offspring[: self.config.population_size]

    def evolve_one_generation(self, population: list[Individual]) -> list[Individual]:
        """Advance the population by one NSGA-II generation."""
        offspring = self._make_offspring(population)
        merged = population + offspring
        return environmental_selection(merged, self.config.population_size)

    def run(self) -> list[Individual]:
        """Run the configured number of generations and return the final population."""
        population = self.initialize_population()

        for _ in range(self.config.generations):
            population = self.evolve_one_generation(population)

        self._annotate_population(population)
        return population

    def solve(self) -> list[Individual]:
        """Alias for ``run`` kept for callers that prefer solver-style naming."""
        return self.run()
