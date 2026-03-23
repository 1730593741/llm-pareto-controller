"""目标函数 用于 Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

from problems.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, to_matrix


def compute_objectives(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    *,
    n_weapons: int,
    n_targets: int,
    required_damage: list[float],
    lethality_matrix: list[list[float]],
) -> tuple[float, float]:
    """返回 (remaining_survivability, ammo_consumption).
    
        Objective 1 minimizes unmet 必需的 damage 跨 Targets.
        Objective 2 minimizes total fired ammunition.
        """
    matrix = to_matrix(allocation, n_weapons=n_weapons, n_targets=n_targets)

    inflicted_damage = [0.0] * n_targets
    ammo_consumption = 0.0

    for weapon_idx in range(n_weapons):
        for target_idx in range(n_targets):
            shots = matrix[weapon_idx][target_idx]
            inflicted_damage[target_idx] += shots * float(lethality_matrix[weapon_idx][target_idx])
            ammo_consumption += shots

    remaining_survivability = float(
        sum(max(0.0, float(required_damage[target_idx]) - inflicted_damage[target_idx]) for target_idx in range(n_targets))
    )
    return remaining_survivability, float(ammo_consumption)
