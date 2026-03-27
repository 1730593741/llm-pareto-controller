"""约束检查 用于 Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, to_matrix
from src.dwta.live_cache import DWTALiveCache


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
    matrix = np.array(to_matrix(allocation, n_weapons=len(ammo_capacities), n_targets=n_targets), dtype=float)
    capacities = np.array(ammo_capacities, dtype=float)
    used = np.sum(matrix, axis=1)
    return float(np.sum(np.maximum(0.0, used - capacities)))


def compatibility_violation(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    compatibility_matrix: list[list[int]],
    n_targets: int,
) -> float:
    """返回 total incompatible shot assignments."""
    matrix = np.array(to_matrix(allocation, n_weapons=len(compatibility_matrix), n_targets=n_targets), dtype=float)
    compatibility = np.array(compatibility_matrix, dtype=float)
    incompatible_mask = compatibility <= 0.0
    return float(np.sum(matrix * incompatible_mask))


def constraint_breakdown(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    *,
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    n_targets: int,
    live_cache: DWTALiveCache | None = None,
) -> DWTAConstraintBreakdown:
    """返回 DWTA 违反 decomposition.

    When ``live_cache`` is provided, static matrix arguments are ignored.
    """
    if live_cache is not None:
        snapshot = live_cache.get_snapshot()
        ammo_capacities = snapshot.ammo_capacities.astype(int).tolist()
        compatibility_matrix = snapshot.compatibility_mask.astype(int).tolist()
        n_targets = snapshot.n_targets

    return DWTAConstraintBreakdown(
        capacity=capacity_violation(allocation, ammo_capacities, n_targets),
        compatibility=compatibility_violation(allocation, compatibility_matrix, n_targets),
    )
