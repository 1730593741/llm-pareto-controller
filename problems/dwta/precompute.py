"""Precomputation utilities for DWTA compatibility and lethality matrices."""

from __future__ import annotations

import math

from problems.dwta.model import DWTABenchmarkData, MunitionType, Target, Weapon


def _distance(lhs_x: float, lhs_y: float, rhs_x: float, rhs_y: float) -> float:
    return math.hypot(lhs_x - rhs_x, lhs_y - rhs_y)


def build_precomputed_matrices(
    munitions: list[MunitionType],
    weapons: list[Weapon],
    targets: list[Target],
) -> DWTABenchmarkData:
    """Build DWTA matrices using range, flight-time, and target time-window checks."""
    munition_by_id = {munition.id: munition for munition in munitions}

    compatibility_matrix: list[list[int]] = []
    lethality_matrix: list[list[float]] = []

    for weapon in weapons:
        munition = munition_by_id[weapon.munition_type_id]
        compat_row: list[int] = []
        lethality_row: list[float] = []

        for target in targets:
            distance = _distance(weapon.x, weapon.y, target.x, target.y)
            in_range = distance <= munition.max_range
            flight_time = distance / munition.flight_speed if munition.flight_speed > 0 else float("inf")
            _, window_end = target.time_window
            in_window = flight_time <= float(window_end)
            compatible = 1 if in_range and in_window else 0

            compat_row.append(compatible)
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
