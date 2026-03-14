"""Scenario preprocessing for Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

import math

from problems.dwta.model import DWTABenchmarkData, MunitionType, Target, Weapon


def distance(weapon: Weapon, target: Target) -> float:
    """Compute Euclidean distance between one weapon and one target."""
    return math.hypot(weapon.x - target.x, weapon.y - target.y)


def build_scenario_matrices(
    munition_types: list[MunitionType],
    weapons: list[Weapon],
    targets: list[Target],
) -> DWTABenchmarkData:
    """Precompute compatibility and lethality matrices once at setup time.

    Compatibility rule:
    - distance <= max_range
    - flight_time in [target.time_window.start, target.time_window.end]
    """
    munition_by_id = {munition.id: munition for munition in munition_types}

    compatibility_matrix: list[list[int]] = []
    lethality_matrix: list[list[float]] = []

    for weapon in weapons:
        munition = munition_by_id[weapon.munition_type_id]
        compat_row: list[int] = []
        lethality_row: list[float] = []

        for target in targets:
            d = distance(weapon, target)
            in_range = d <= munition.max_range
            flight_time = d / munition.flight_speed if munition.flight_speed > 0 else float("inf")
            t_start, t_end = target.time_window
            in_time_window = float(t_start) <= flight_time <= float(t_end)
            compat_row.append(1 if in_range and in_time_window else 0)
            lethality_row.append(float(munition.lethality))

        compatibility_matrix.append(compat_row)
        lethality_matrix.append(lethality_row)

    return DWTABenchmarkData(
        n_weapons=len(weapons),
        n_targets=len(targets),
        ammo_capacities=[weapon.ammo_capacity for weapon in weapons],
        compatibility_matrix=compatibility_matrix,
        lethality_matrix=lethality_matrix,
        required_damage=[target.required_damage for target in targets],
    )

