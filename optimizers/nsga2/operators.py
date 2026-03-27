"""变异与交叉算子 用于 NSGA-II genomes."""

from __future__ import annotations

import random

from src.dwta.encoding import to_genome, to_matrix
from problems.task_assignment.encoding import Assignment



def one_point_crossover(
    parent_a: Assignment,
    parent_b: Assignment,
    crossover_prob: float,
    rng: random.Random,
) -> tuple[Assignment, Assignment]:
    """对 分配 genome 执行单点交叉."""
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")

    if len(parent_a) < 2 or rng.random() >= crossover_prob:
        return list(parent_a), list(parent_b)

    cut = rng.randint(1, len(parent_a) - 1)
    child_a = list(parent_a[:cut]) + list(parent_b[cut:])
    child_b = list(parent_b[:cut]) + list(parent_a[cut:])
    return child_a, child_b



def mutate_assignment(
    genome: Assignment,
    n_resources: int,
    mutation_prob: float,
    rng: random.Random,
) -> Assignment:
    """对 分配 执行按任务基因的随机重置变异."""
    if n_resources <= 0:
        raise ValueError("n_resources must be positive")

    mutated = list(genome)
    for idx, resource in enumerate(mutated):
        if resource < 0 or resource >= n_resources:
            raise ValueError("Genome contains invalid resource index")
        if rng.random() < mutation_prob:
            mutated[idx] = rng.randrange(n_resources)

    return mutated


def mutate_bounded_integers(
    genome: list[int],
    upper_bounds: list[int],
    mutation_prob: float,
    rng: random.Random,
) -> list[int]:
    """对整数基因执行逐基因含上界变异."""
    if len(genome) != len(upper_bounds):
        raise ValueError("upper_bounds length must match genome length")

    mutated = list(genome)
    for idx, value in enumerate(mutated):
        upper = int(upper_bounds[idx])
        if upper < 0:
            raise ValueError("upper bounds must be >= 0")
        if value < 0 or value > upper:
            raise ValueError("Genome contains invalid bounded value")
        if rng.random() < mutation_prob:
            mutated[idx] = rng.randint(0, upper)

    return mutated


def matrix_block_crossover(
    parent_a: list[int],
    parent_b: list[int],
    *,
    n_weapons: int,
    n_targets: int,
    crossover_prob: float,
    rng: random.Random,
    eta_c: float | None = None,
) -> tuple[list[int], list[int]]:
    """Perform 矩阵-aware block 交叉 用于 DWTA 整数 allocations.
    
        该 算子 swaps contiguous 矩阵 blocks along either 该 Weapon axis
        (row block) 或 该 Target axis (column block). This preserves 整数
        值 通过 construction because only existing 整数 entries 为 copied.
    
        ``eta_c`` 为 mapped 到 axis preference 与 kept backward-兼容的 并带有
        该 existing 控制器 接口:
        - ``eta_c 为 None``: 50% row-block 与 50% column-block 交叉.
        - otherwise: ``P(row-block) = clamp(eta_c / 100, 0, 1)``.
          Higher ``eta_c`` therefore biases toward Weapon-dimension exchange.
        """
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")
    if len(parent_a) != n_weapons * n_targets:
        raise ValueError("Parent length must equal n_weapons * n_targets")
    if n_weapons <= 0 or n_targets <= 0:
        raise ValueError("n_weapons and n_targets must be positive")

    if rng.random() >= crossover_prob:
        return list(parent_a), list(parent_b)

    matrix_a = to_matrix(parent_a, n_weapons=n_weapons, n_targets=n_targets)
    matrix_b = to_matrix(parent_b, n_weapons=n_weapons, n_targets=n_targets)
    child_a = [row[:] for row in matrix_a]
    child_b = [row[:] for row in matrix_b]

    row_prob = 0.5 if eta_c is None else max(0.0, min(1.0, eta_c / 100.0))
    use_row_block = n_weapons > 1 and (n_targets == 1 or rng.random() < row_prob)

    if use_row_block:
        cut = rng.randint(1, n_weapons - 1)
        child_a[cut:], child_b[cut:] = child_b[cut:], child_a[cut:]
    elif n_targets > 1:
        cut = rng.randint(1, n_targets - 1)
        for weapon_idx in range(n_weapons):
            child_a[weapon_idx][cut:], child_b[weapon_idx][cut:] = (
                child_b[weapon_idx][cut:],
                child_a[weapon_idx][cut:],
            )

    return to_genome(child_a), to_genome(child_b)


def mutate_dwta_allocation(
    genome: list[int],
    *,
    n_weapons: int,
    n_targets: int,
    compatibility_matrix: list[list[int]],
    mutation_prob: float,
    rng: random.Random,
    mutation_step: float,
    eta_m: float | None,
    local_search_prob: float | None,
) -> list[int]:
    """Mutate DWTA 整数 矩阵，并带有 兼容性-aware edits.
    
        Parameter mapping 到 preserve 该 unified 控制器 接口:
        - ``mutation_prob``: per-兼容的-cell 变异 trigger 概率.
        - ``mutation_step``: base 整数 变异 amplitude, usually driven 通过 该
          existing 控制器 step-size schedule 该 updates ``mutation_prob``.
        - ``eta_m``: 控制s 变异 locality. Higher 值 produce smaller
          整数 steps; lower 值 allow larger step amplitudes.
        - ``local_search_prob``: 概率 的 使用 一个 在-row transfer move
          (shift 一个 或 more rounds 从 Target A 到 B 用于 该 same Weapon).
          Remaining 概率 uses 整数 +/- perturbation.
        """
    if len(genome) != n_weapons * n_targets:
        raise ValueError("Genome length must equal n_weapons * n_targets")
    if len(compatibility_matrix) != n_weapons or any(len(row) != n_targets for row in compatibility_matrix):
        raise ValueError("compatibility_matrix shape must match DWTA dimensions")

    matrix = to_matrix(genome, n_weapons=n_weapons, n_targets=n_targets)
    transfer_prob = 0.0 if local_search_prob is None else max(0.0, min(1.0, local_search_prob))
    eta_locality = 20.0 if eta_m is None else max(0.0, float(eta_m))
    base_step = max(1, int(round(max(1e-9, mutation_step))))
    locality_scale = max(1.0, 1.0 + eta_locality / 10.0)
    delta_max = max(1, int(round(base_step / locality_scale * 4.0)))

    for weapon_idx in range(n_weapons):
        compatible_targets = [idx for idx, flag in enumerate(compatibility_matrix[weapon_idx]) if int(flag) == 1]
        if not compatible_targets:
            continue

        for target_idx in compatible_targets:
            if rng.random() >= mutation_prob:
                continue

            can_transfer = len(compatible_targets) > 1 and matrix[weapon_idx][target_idx] > 0
            if can_transfer and rng.random() < transfer_prob:
                destination_candidates = [idx for idx in compatible_targets if idx != target_idx]
                destination = rng.choice(destination_candidates)
                transferable = matrix[weapon_idx][target_idx]
                delta = min(transferable, rng.randint(1, delta_max))
                matrix[weapon_idx][target_idx] -= delta
                matrix[weapon_idx][destination] += delta
                continue

            delta = rng.randint(1, delta_max)
            if matrix[weapon_idx][target_idx] > 0 and rng.random() < 0.5:
                matrix[weapon_idx][target_idx] = max(0, matrix[weapon_idx][target_idx] - delta)
            else:
                matrix[weapon_idx][target_idx] += delta

    return to_genome(matrix)
