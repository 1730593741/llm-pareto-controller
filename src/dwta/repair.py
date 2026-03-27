"""修复启发式 用于 DWTA 整数 shot-allocation genomes."""

from __future__ import annotations

import random

from src.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, to_genome, to_matrix
from src.dwta.live_cache import DWTALiveCache


def _removal_priority(
    *,
    target_idx: int,
    inflicted_damage: list[float],
    required_damage: list[float] | None,
    lethality: float,
) -> tuple[int, float, int]:
    """返回 确定性 priority key 用于 removing 一个 shot.

    Priority order:
    1) shots on oversaturated Targets (higher oversaturation removed first),
    2) lower lethality efficiency removed first,
    3) lower Target 索引 用于 确定性 tie-breaking.
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
    live_cache: DWTALiveCache | None = None,
) -> list[int]:
    """Repair incompatible 或 over-capacity DWTA allocations.

    该 heuristic 为 确定性 与 does 不 sample 随机 choices. ``rng`` 为
    retained 用于 backward-兼容的 function signatures 用于 该 求解器 stack.
    """
    del rng

    if live_cache is not None:
        snapshot = live_cache.get_snapshot()
        ammo_capacities = snapshot.ammo_capacities.astype(int).tolist()
        compatibility_matrix = snapshot.compatibility_mask.astype(int).tolist()
        n_targets = snapshot.n_targets
        lethality_matrix = snapshot.lethality_matrix.astype(float).tolist()
        required_damage = snapshot.required_damage.astype(float).tolist()

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
