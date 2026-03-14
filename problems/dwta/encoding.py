"""Encoding helpers for DWTA integer shot-allocation genomes."""

from __future__ import annotations

import random

DWTAAllocationGenome = list[int]


def flatten_index(weapon_idx: int, target_idx: int, n_targets: int) -> int:
    """Return flattened vector index for ``X[weapon_idx][target_idx]``."""
    return weapon_idx * n_targets + target_idx


def random_allocation(
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    rng: random.Random | None = None,
) -> DWTAAllocationGenome:
    """Create a random feasible-by-capacity DWTA genome."""
    generator = rng or random.Random()
    n_targets = len(compatibility_matrix[0]) if compatibility_matrix else 0
    genome = [0] * (len(ammo_capacities) * n_targets)

    for weapon_idx, ammo_capacity in enumerate(ammo_capacities):
        remaining = int(ammo_capacity)
        compatible_targets = [idx for idx, flag in enumerate(compatibility_matrix[weapon_idx]) if int(flag) == 1]
        if remaining <= 0 or not compatible_targets:
            continue

        for _ in range(remaining):
            if generator.random() < 0.5:
                continue
            target_idx = generator.choice(compatible_targets)
            genome[flatten_index(weapon_idx, target_idx, n_targets)] += 1

    return genome
