"""Tests for NSGA-II variation operators."""

import random

from optimizers.nsga2.operators import mutate_assignment, one_point_crossover


def test_one_point_crossover_respects_genome_length_and_values() -> None:
    rng = random.Random(0)
    p1 = [0, 0, 0, 0]
    p2 = [1, 1, 1, 1]

    c1, c2 = one_point_crossover(p1, p2, crossover_prob=1.0, rng=rng)

    assert len(c1) == len(p1)
    assert len(c2) == len(p2)
    assert set(c1).issubset({0, 1})
    assert set(c2).issubset({0, 1})


def test_mutation_keeps_values_in_resource_range() -> None:
    rng = random.Random(1)
    genome = [0, 1, 2, 1, 0]

    mutated = mutate_assignment(genome, n_resources=3, mutation_prob=1.0, rng=rng)

    assert len(mutated) == len(genome)
    assert all(0 <= gene < 3 for gene in mutated)
