"""Repair heuristics for DWTA integer shot-allocation genomes."""

from __future__ import annotations

import random

from problems.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, to_genome, to_matrix


def _removal_priority(
    *,
    target_idx: int,
    inflicted_damage: list[float],
    required_damage: list[float] | None,
    lethality: float,
) -> tuple[int, float, int]:
    """Return deterministic priority key for removing one shot.

    Priority order:
    1) shots on oversaturated targets (higher oversaturation removed first),
    2) lower lethality efficiency removed first,
    3) lower target index for deterministic tie-breaking.
    """
    oversaturation = 0.0
    if required_damage is not None:
        oversaturation = max(0.0, inflicted_damage[target_idx] - float(required_damage[target_idx]))
    oversaturation_bucket = 0 if oversaturation > 0.0 else 1
    return (oversaturation_bucket, lethality, target_idx)


def repair_allocation(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    *,
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    n_targets: int,
    rng: random.Random,
    lethality_matrix: list[list[float]] | None = None,
    required_damage: list[float] | None = None,
) -> list[int]:
    """Repair incompatible or over-capacity DWTA allocations.

    The heuristic is deterministic and does not sample random choices. ``rng`` is
    retained for backward-compatible function signatures used by the solver stack.
    """
    del rng

    n_weapons = len(ammo_capacities)
    matrix = to_matrix(allocation, n_weapons=n_weapons, n_targets=n_targets)

    for weapon_idx, row in enumerate(compatibility_matrix):
        for target_idx, flag in enumerate(row):
            if int(flag) == 0:
                matrix[weapon_idx][target_idx] = 0

    inflicted_damage = [0.0] * n_targets
    for weapon_idx in range(n_weapons):
        for target_idx in range(n_targets):
            lethality = float(lethality_matrix[weapon_idx][target_idx]) if lethality_matrix is not None else 0.0
            inflicted_damage[target_idx] += matrix[weapon_idx][target_idx] * lethality

    for weapon_idx, ammo_capacity in enumerate(ammo_capacities):
        while sum(matrix[weapon_idx]) > int(ammo_capacity):
            candidates = [target_idx for target_idx in range(n_targets) if matrix[weapon_idx][target_idx] > 0]
            if not candidates:
                break

            def priority_key(target_idx: int) -> tuple[int, float, int]:
                lethality = float(lethality_matrix[weapon_idx][target_idx]) if lethality_matrix is not None else 0.0
                return _removal_priority(
                    target_idx=target_idx,
                    inflicted_damage=inflicted_damage,
                    required_damage=required_damage,
                    lethality=lethality,
                )

            drop_target = min(candidates, key=priority_key)
            matrix[weapon_idx][drop_target] -= 1
            if lethality_matrix is not None:
                inflicted_damage[drop_target] -= float(lethality_matrix[weapon_idx][drop_target])

    return to_genome(matrix)
