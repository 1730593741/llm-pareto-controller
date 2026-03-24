"""目标函数 用于 Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

import numpy as np

from problems.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, to_matrix
from problems.dwta.live_cache import DWTALiveCache


def compute_objectives(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    *,
    n_weapons: int,
    n_targets: int,
    required_damage: list[float],
    lethality_matrix: list[list[float]],
    live_cache: DWTALiveCache | None = None,
) -> tuple[float, float]:
    """返回 (remaining_survivability, ammo_consumption).

    Objective 1 minimizes unmet 必需的 damage 跨 Targets.
    Objective 2 minimizes total fired ammunition.

    When ``live_cache`` is provided, environment-dependent matrices are read
    from cache snapshots; legacy static arguments remain backward-compatible.
    """
    if live_cache is not None:
        snapshot = live_cache.get_snapshot()
        n_weapons = snapshot.n_weapons
        n_targets = snapshot.n_targets
        req = snapshot.required_damage
        lethality = snapshot.lethality_matrix
    else:
        req = np.array(required_damage, dtype=float)
        lethality = np.array(lethality_matrix, dtype=float)

    matrix = np.array(to_matrix(allocation, n_weapons=n_weapons, n_targets=n_targets), dtype=float)
    inflicted_damage = np.sum(matrix * lethality, axis=0)
    ammo_consumption = float(np.sum(matrix))
    remaining_survivability = float(np.sum(np.maximum(0.0, req - inflicted_damage)))
    return remaining_survivability, ammo_consumption
