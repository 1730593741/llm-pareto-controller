"""约束检查 用于 Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

from dataclasses import dataclass

from problems.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, to_matrix


@dataclass(frozen=True, slots=True)
class DWTAConstraintBreakdown:
    """DWTA 约束分解 用于 可行性指标."""

    capacity: float = 0.0
    compatibility: float = 0.0

    @property
    def total(self) -> float:
        """返回 聚合 违反 值."""
        return self.capacity + self.compatibility


def capacity_violation(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    ammo_capacities: list[int],
    n_targets: int,
) -> float:
    """返回 total capacity overflow 跨 所有 Weapons."""
    matrix = to_matrix(allocation, n_weapons=len(ammo_capacities), n_targets=n_targets)
    violation = 0.0
    for weapon_idx, ammo_capacity in enumerate(ammo_capacities):
        used = sum(matrix[weapon_idx])
        violation += max(0.0, float(used - ammo_capacity))
    return float(violation)


def compatibility_violation(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    compatibility_matrix: list[list[int]],
    n_targets: int,
) -> float:
    """返回 total incompatible shot assignments."""
    matrix = to_matrix(allocation, n_weapons=len(compatibility_matrix), n_targets=n_targets)
    violation = 0.0
    for weapon_idx, row in enumerate(compatibility_matrix):
        for target_idx, flag in enumerate(row):
            shots = matrix[weapon_idx][target_idx]
            if int(flag) == 0 and shots > 0:
                violation += float(shots)
    return float(violation)


def constraint_breakdown(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    *,
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    n_targets: int,
) -> DWTAConstraintBreakdown:
    """返回 DWTA 违反 decomposition."""
    return DWTAConstraintBreakdown(
        capacity=capacity_violation(allocation, ammo_capacities, n_targets),
        compatibility=compatibility_violation(allocation, compatibility_matrix, n_targets),
    )
