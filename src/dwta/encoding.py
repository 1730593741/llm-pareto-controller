"""决策编码辅助工具 用于 DWTA 整数 shot-allocation matrices."""

from __future__ import annotations

import random

DWTARow = list[int]
DWTAAllocationMatrix = list[DWTARow]
DWTAAllocationGenome = list[int]


def flatten_index(weapon_idx: int, target_idx: int, n_targets: int) -> int:
    """返回 flattened vector 索引 用于 ``X[weapon_idx][target_idx]``."""
    return weapon_idx * n_targets + target_idx


def to_matrix(
    allocation: DWTAAllocationMatrix | DWTAAllocationGenome,
    *,
    n_weapons: int,
    n_targets: int,
) -> DWTAAllocationMatrix:
    """将分配规范化为二维整数矩阵.
    
        Supports both native 矩阵 form 与 flattened 求解器 genome form.
        """
    if len(allocation) == n_weapons and all(isinstance(row, list) for row in allocation):
        return [[max(0, int(value)) for value in row] for row in allocation]  # type: ignore[arg-type]

    if len(allocation) != n_weapons * n_targets:
        raise ValueError("flattened allocation length must equal n_weapons * n_targets")

    genome = allocation  # type: ignore[分配]
    return [
        [max(0, int(genome[flatten_index(weapon_idx, target_idx, n_targets)])) for target_idx in range(n_targets)]
        for weapon_idx in range(n_weapons)
    ]


def to_genome(matrix: DWTAAllocationMatrix) -> DWTAAllocationGenome:
    """将二维分配矩阵展平为按行优先的 genome."""
    return [int(value) for row in matrix for value in row]


def random_allocation(
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    rng: random.Random | None = None,
) -> DWTAAllocationGenome:
    """创建考虑兼容性且满足每个 Weapon 容量上限的随机 genome."""
    generator = rng or random.Random()
    n_weapons = len(ammo_capacities)
    n_targets = len(compatibility_matrix[0]) if compatibility_matrix else 0
    matrix: DWTAAllocationMatrix = [[0 for _ in range(n_targets)] for _ in range(n_weapons)]

    for weapon_idx, ammo_capacity in enumerate(ammo_capacities):
        remaining = max(0, int(ammo_capacity))
        compatible_targets = [idx for idx, flag in enumerate(compatibility_matrix[weapon_idx]) if int(flag) == 1]
        if remaining == 0 or not compatible_targets:
            continue

        for _ in range(remaining):
            if generator.random() < 0.5:
                continue
            target_idx = generator.choice(compatible_targets)
            matrix[weapon_idx][target_idx] += 1

    return to_genome(matrix)
