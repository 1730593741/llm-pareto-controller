"""Variation operators for NSGA-II genomes."""

from __future__ import annotations

import random

from problems.dwta.encoding import to_genome, to_matrix
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


def matrix_block_crossover(
    parent_a: list[int],
    parent_b: list[int],
    *,
    n_weapons: int,
    n_targets: int,
    crossover_prob: float,
    rng: random.Random,
    eta_c: float | None = None,
) -> tuple[list[int], list[int]]:
    """Perform matrix-aware block crossover for DWTA integer allocations.

    The operator swaps contiguous matrix blocks along either the weapon axis
    (row block) or the target axis (column block). This preserves integer
    values by construction because only existing integer entries are copied.

    ``eta_c`` is mapped to axis preference and kept backward-compatible with
    the existing controller interface:
    - ``eta_c is None``: 50% row-block and 50% column-block crossover.
    - otherwise: ``P(row-block) = clamp(eta_c / 100, 0, 1)``.
      Higher ``eta_c`` therefore biases toward weapon-dimension exchange.
    """
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")
    if len(parent_a) != n_weapons * n_targets:
        raise ValueError("Parent length must equal n_weapons * n_targets")
    if n_weapons <= 0 or n_targets <= 0:
        raise ValueError("n_weapons and n_targets must be positive")

    if rng.random() >= crossover_prob:
        return list(parent_a), list(parent_b)

    matrix_a = to_matrix(parent_a, n_weapons=n_weapons, n_targets=n_targets)
    matrix_b = to_matrix(parent_b, n_weapons=n_weapons, n_targets=n_targets)
    child_a = [row[:] for row in matrix_a]
    child_b = [row[:] for row in matrix_b]

    row_prob = 0.5 if eta_c is None else max(0.0, min(1.0, eta_c / 100.0))
    use_row_block = n_weapons > 1 and (n_targets == 1 or rng.random() < row_prob)

    if use_row_block:
        cut = rng.randint(1, n_weapons - 1)
        child_a[cut:], child_b[cut:] = child_b[cut:], child_a[cut:]
    elif n_targets > 1:
        cut = rng.randint(1, n_targets - 1)
        for weapon_idx in range(n_weapons):
            child_a[weapon_idx][cut:], child_b[weapon_idx][cut:] = (
                child_b[weapon_idx][cut:],
                child_a[weapon_idx][cut:],
            )

    return to_genome(child_a), to_genome(child_b)


def mutate_dwta_allocation(
    genome: list[int],
    *,
    n_weapons: int,
    n_targets: int,
    compatibility_matrix: list[list[int]],
    mutation_prob: float,
    rng: random.Random,
    mutation_step: float,
    eta_m: float | None,
    local_search_prob: float | None,
) -> list[int]:
    """Mutate DWTA integer matrix with compatibility-aware edits.

    Parameter mapping to preserve the unified controller interface:
    - ``mutation_prob``: per-compatible-cell mutation trigger probability.
    - ``mutation_step``: base integer mutation amplitude, usually driven by the
      existing controller step-size schedule that updates ``mutation_prob``.
    - ``eta_m``: controls mutation locality. Higher values produce smaller
      integer steps; lower values allow larger step amplitudes.
    - ``local_search_prob``: probability of using an in-row transfer move
      (shift one or more rounds from target A to B for the same weapon).
      Remaining probability uses integer +/- perturbation.
    """
    if len(genome) != n_weapons * n_targets:
        raise ValueError("Genome length must equal n_weapons * n_targets")
    if len(compatibility_matrix) != n_weapons or any(len(row) != n_targets for row in compatibility_matrix):
        raise ValueError("compatibility_matrix shape must match DWTA dimensions")

    matrix = to_matrix(genome, n_weapons=n_weapons, n_targets=n_targets)
    transfer_prob = 0.0 if local_search_prob is None else max(0.0, min(1.0, local_search_prob))
    eta_locality = 20.0 if eta_m is None else max(0.0, float(eta_m))
    base_step = max(1, int(round(max(1e-9, mutation_step))))
    locality_scale = max(1.0, 1.0 + eta_locality / 10.0)
    delta_max = max(1, int(round(base_step / locality_scale * 4.0)))

    for weapon_idx in range(n_weapons):
        compatible_targets = [idx for idx, flag in enumerate(compatibility_matrix[weapon_idx]) if int(flag) == 1]
        if not compatible_targets:
            continue

        for target_idx in compatible_targets:
            if rng.random() >= mutation_prob:
                continue

            can_transfer = len(compatible_targets) > 1 and matrix[weapon_idx][target_idx] > 0
            if can_transfer and rng.random() < transfer_prob:
                destination_candidates = [idx for idx in compatible_targets if idx != target_idx]
                destination = rng.choice(destination_candidates)
                transferable = matrix[weapon_idx][target_idx]
                delta = min(transferable, rng.randint(1, delta_max))
                matrix[weapon_idx][target_idx] -= delta
                matrix[weapon_idx][destination] += delta
                continue

            delta = rng.randint(1, delta_max)
            if matrix[weapon_idx][target_idx] > 0 and rng.random() < 0.5:
                matrix[weapon_idx][target_idx] = max(0, matrix[weapon_idx][target_idx] - delta)
            else:
                matrix[weapon_idx][target_idx] += delta

    return to_genome(matrix)
