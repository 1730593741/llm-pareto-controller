"""Objective functions for Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations


def compute_objectives(
    genome: list[int],
    *,
    n_weapons: int,
    n_targets: int,
    required_damage: list[float],
    lethality_matrix: list[list[float]],
) -> tuple[float, float]:
    """Return (remaining_survivability, ammo_consumption)."""
    inflicted_damage = [0.0] * n_targets
    ammo_consumption = 0.0

    for weapon_idx in range(n_weapons):
        for target_idx in range(n_targets):
            shots = int(genome[weapon_idx * n_targets + target_idx])
            inflicted_damage[target_idx] += shots * float(lethality_matrix[weapon_idx][target_idx])
            ammo_consumption += shots

    remaining_survivability = float(
        sum(max(0.0, float(required_damage[target_idx]) - inflicted_damage[target_idx]) for target_idx in range(n_targets))
    )
    return remaining_survivability, ammo_consumption
