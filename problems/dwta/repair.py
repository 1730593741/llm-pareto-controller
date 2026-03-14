"""Repair heuristics for DWTA integer shot-allocation genomes."""

from __future__ import annotations

import random


def repair_allocation(
    genome: list[int],
    *,
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    n_targets: int,
    rng: random.Random,
) -> list[int]:
    """Repair incompatible or over-capacity DWTA allocations."""
    repaired = [max(0, int(value)) for value in genome]

    for weapon_idx, row in enumerate(compatibility_matrix):
        row_start = weapon_idx * n_targets
        for target_idx, flag in enumerate(row):
            if int(flag) == 0:
                repaired[row_start + target_idx] = 0

        ammo_capacity = int(ammo_capacities[weapon_idx])
        while sum(repaired[row_start : row_start + n_targets]) > ammo_capacity:
            positive_targets = [idx for idx in range(n_targets) if repaired[row_start + idx] > 0]
            if not positive_targets:
                break
            drop_target = rng.choice(positive_targets)
            repaired[row_start + drop_target] -= 1

    return repaired
