"""Variation operators for NSGA-II genomes."""

from __future__ import annotations

import random

from problems.task_assignment.encoding import Assignment



def one_point_crossover(
    parent_a: Assignment,
    parent_b: Assignment,
    crossover_prob: float,
    rng: random.Random,
) -> tuple[Assignment, Assignment]:
    """Perform one-point crossover for assignment genomes."""
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")

    if len(parent_a) < 2 or rng.random() >= crossover_prob:
        return list(parent_a), list(parent_b)

    cut = rng.randint(1, len(parent_a) - 1)
    child_a = list(parent_a[:cut]) + list(parent_b[cut:])
    child_b = list(parent_b[:cut]) + list(parent_a[cut:])
    return child_a, child_b



def mutate_assignment(
    genome: Assignment,
    n_resources: int,
    mutation_prob: float,
    rng: random.Random,
) -> Assignment:
    """Mutate assignment by random reset mutation per task gene."""
    if n_resources <= 0:
        raise ValueError("n_resources must be positive")

    mutated = list(genome)
    for idx, resource in enumerate(mutated):
        if resource < 0 or resource >= n_resources:
            raise ValueError("Genome contains invalid resource index")
        if rng.random() < mutation_prob:
            mutated[idx] = rng.randrange(n_resources)

    return mutated


def mutate_bounded_integers(
    genome: list[int],
    upper_bounds: list[int],
    mutation_prob: float,
    rng: random.Random,
) -> list[int]:
    """Mutate integer genes with per-gene inclusive upper bounds."""
    if len(genome) != len(upper_bounds):
        raise ValueError("upper_bounds length must match genome length")

    mutated = list(genome)
    for idx, value in enumerate(mutated):
        upper = int(upper_bounds[idx])
        if upper < 0:
            raise ValueError("upper bounds must be >= 0")
        if value < 0 or value > upper:
            raise ValueError("Genome contains invalid bounded value")
        if rng.random() < mutation_prob:
            mutated[idx] = rng.randint(0, upper)

    return mutated
