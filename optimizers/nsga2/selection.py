"""Selection and ranking utilities for NSGA-II."""

from __future__ import annotations

import math
import random

from optimizers.nsga2.population import Individual


def dominates(left: Individual, right: Individual) -> bool:
    """Return whether ``left`` dominates ``right`` under constrained NSGA-II rules."""
    if left.feasible and not right.feasible:
        return True
    if not left.feasible and right.feasible:
        return False
    if not left.feasible and not right.feasible:
        return left.constraint_violation < right.constraint_violation

    left_not_worse = all(l <= r for l, r in zip(left.objectives, right.objectives))
    left_better_in_any = any(l < r for l, r in zip(left.objectives, right.objectives))
    return left_not_worse and left_better_in_any


def non_dominated_sort(individuals: list[Individual]) -> list[list[int]]:
    """Sort individuals into non-dominated fronts and return fronts as index lists."""
    if not individuals:
        return []

    domination_count = [0] * len(individuals)
    dominates_set: list[list[int]] = [[] for _ in individuals]
    fronts: list[list[int]] = [[]]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals):
            if i == j:
                continue
            if dominates(ind_i, ind_j):
                dominates_set[i].append(j)
            elif dominates(ind_j, ind_i):
                domination_count[i] += 1

        if domination_count[i] == 0:
            ind_i.rank = 0
            fronts[0].append(i)

    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominates_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    individuals[j].rank = current + 1
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current += 1

    return fronts


def assign_crowding_distance(individuals: list[Individual], front: list[int]) -> None:
    """Assign crowding distance values for one front in-place."""
    if not front:
        return
    if len(front) <= 2:
        for idx in front:
            individuals[idx].crowding_distance = math.inf
        return

    for idx in front:
        individuals[idx].crowding_distance = 0.0

    n_objectives = len(individuals[front[0]].objectives)
    for objective_idx in range(n_objectives):
        sorted_front = sorted(front, key=lambda idx: individuals[idx].objectives[objective_idx])
        min_val = individuals[sorted_front[0]].objectives[objective_idx]
        max_val = individuals[sorted_front[-1]].objectives[objective_idx]

        individuals[sorted_front[0]].crowding_distance = math.inf
        individuals[sorted_front[-1]].crowding_distance = math.inf

        if max_val == min_val:
            continue

        for pos in range(1, len(sorted_front) - 1):
            prev_val = individuals[sorted_front[pos - 1]].objectives[objective_idx]
            next_val = individuals[sorted_front[pos + 1]].objectives[objective_idx]
            increment = (next_val - prev_val) / (max_val - min_val)
            if not math.isinf(individuals[sorted_front[pos]].crowding_distance):
                individuals[sorted_front[pos]].crowding_distance += increment


def tournament_select(individuals: list[Individual], rng: random.Random) -> Individual:
    """Binary tournament selection by rank, then crowding distance."""
    if not individuals:
        raise ValueError("Cannot select from an empty population")
    if len(individuals) == 1:
        return individuals[0]

    a, b = rng.sample(individuals, 2)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding_distance > b.crowding_distance:
        return a
    if b.crowding_distance > a.crowding_distance:
        return b
    return a if rng.random() < 0.5 else b


def environmental_select(individuals: list[Individual], population_size: int) -> list[Individual]:
    """Select the next generation from combined parent-offspring individuals."""
    if population_size <= 0:
        raise ValueError("population_size must be > 0")

    fronts = non_dominated_sort(individuals)
    next_population: list[Individual] = []

    for front in fronts:
        assign_crowding_distance(individuals, front)
        if len(next_population) + len(front) <= population_size:
            next_population.extend(individuals[i] for i in front)
        else:
            sorted_front = sorted(
                (individuals[i] for i in front),
                key=lambda ind: ind.crowding_distance,
                reverse=True,
            )
            remaining = population_size - len(next_population)
            next_population.extend(sorted_front[:remaining])
            break

    return next_population
