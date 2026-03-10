"""Variation operators for assignment genomes used by NSGA-II."""

from __future__ import annotations

import random

from problems.task_assignment.encoding import Assignment


def crossover(
    parent1: Assignment,
    parent2: Assignment,
    crossover_prob: float,
    rng: random.Random,
) -> tuple[Assignment, Assignment]:
    """Apply one-point crossover to two parent assignments."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have equal genome lengths")

    genome_length = len(parent1)
    if genome_length < 2 or rng.random() > crossover_prob:
        return list(parent1), list(parent2)

    point = rng.randint(1, genome_length - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(genome: Assignment, n_resources: int, mutation_prob: float, rng: random.Random) -> Assignment:
    """Apply per-gene random reset mutation."""
    mutated = list(genome)
    for idx in range(len(mutated)):
        if rng.random() < mutation_prob:
            mutated[idx] = rng.randrange(n_resources)
    return mutated
