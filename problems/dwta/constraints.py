"""Constraint checks for Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DWTAConstraintBreakdown:
    """DWTA constraint decomposition used by feasibility metrics."""

    capacity: float = 0.0
    compatibility: float = 0.0

    @property
    def total(self) -> float:
        """Return aggregate violation value."""
        return self.capacity + self.compatibility


def capacity_violation(genome: list[int], ammo_capacities: list[int], n_targets: int) -> float:
    """Return total capacity overflow across all weapons."""
    violation = 0.0
    for weapon_idx, ammo_capacity in enumerate(ammo_capacities):
        row_start = weapon_idx * n_targets
        used = sum(int(value) for value in genome[row_start : row_start + n_targets])
        violation += max(0.0, float(used - ammo_capacity))
    return float(violation)


def compatibility_violation(genome: list[int], compatibility_matrix: list[list[int]], n_targets: int) -> float:
    """Return number of incompatible shot assignments."""
    violation = 0.0
    for weapon_idx, row in enumerate(compatibility_matrix):
        row_start = weapon_idx * n_targets
        for target_idx, flag in enumerate(row):
            shots = int(genome[row_start + target_idx])
            if int(flag) == 0 and shots > 0:
                violation += float(shots)
    return float(violation)


def constraint_breakdown(
    genome: list[int],
    *,
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    n_targets: int,
) -> DWTAConstraintBreakdown:
    """Return DWTA violation decomposition."""
    return DWTAConstraintBreakdown(
        capacity=capacity_violation(genome, ammo_capacities, n_targets),
        compatibility=compatibility_violation(genome, compatibility_matrix, n_targets),
    )
