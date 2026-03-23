"""Pareto-状态 感知 用于 NSGA-II 种群 snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import dominates
from sensing.feasibility_metrics import feasible_ratio, mean_constraint_violation
from sensing.hypervolume import HypervolumeCalculator, SimplifiedHypervolumeCalculator


@dataclass(slots=True)
class ParetoState:
    """结构化快照 的 当前 优化器 search 状态."""

    generation: int
    hv: float
    delta_hv: float
    feasible_ratio: float
    rank1_ratio: float
    mean_cv: float
    diversity_score: float
    crowding_entropy: float
    d_dec: float
    d_front: float
    stagnation_len: int
    rank1_objectives: list[tuple[float, ...]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化 状态 到 一个 dictionary 用于 日志."""
        return asdict(self)


class ParetoStateSensor:
    """构建 ParetoState 从 当前 种群，并带有 可替换 指标."""

    def __init__(self, hv_calculator: HypervolumeCalculator | None = None) -> None:
        self._hv_calculator = hv_calculator or SimplifiedHypervolumeCalculator()

    def sense(
        self,
        *,
        generation: int,
        population: list[Individual],
        previous_state: ParetoState | None = None,
        reference_point: tuple[float, ...] | None = None,
        stagnation_tolerance: float = 1e-12,
    ) -> ParetoState:
        """生成 ParetoState 从 当前 种群 与 previous 状态."""
        if generation < 0:
            raise ValueError("generation must be >= 0")

        violations = [ind.constraint_violation for ind in population]
        objectives = [ind.objectives for ind in population if ind.objectives]

        current_feasible_ratio = feasible_ratio(violations)
        current_mean_cv = mean_constraint_violation(violations)
        current_rank1_ratio = _compute_rank1_ratio(population)
        current_diversity = _compute_diversity_score(objectives)
        rank1_individuals = _rank1_individuals(population)
        current_crowding_entropy = _compute_crowding_entropy(rank1_individuals)
        current_d_dec = _compute_decision_diversity(population)
        current_d_front = _compute_front_separation(population, rank1_individuals)

        hv_reference = reference_point or _default_reference_point(objectives)
        current_hv = self._hv_calculator.compute(objectives, hv_reference) if objectives else 0.0

        prev_hv = previous_state.hv if previous_state else 0.0
        delta_hv = current_hv - prev_hv

        if previous_state is None:
            stagnation_len = 0
        elif delta_hv <= stagnation_tolerance:
            stagnation_len = previous_state.stagnation_len + 1
        else:
            stagnation_len = 0

        return ParetoState(
            generation=generation,
            hv=current_hv,
            delta_hv=delta_hv,
            feasible_ratio=current_feasible_ratio,
            rank1_ratio=current_rank1_ratio,
            mean_cv=current_mean_cv,
            diversity_score=current_diversity,
            crowding_entropy=current_crowding_entropy,
            d_dec=current_d_dec,
            d_front=current_d_front,
            stagnation_len=stagnation_len,
            rank1_objectives=[tuple(ind.objectives) for ind in rank1_individuals if ind.objectives],
        )


def _compute_rank1_ratio(population: list[Individual]) -> float:
    """计算第一前沿个体比例 在 该 given 种群."""
    if not population:
        return 0.0

    rank1_size = sum(1 for candidate in population if _is_nondominated(candidate, population))
    return rank1_size / len(population)


def _rank1_individuals(population: list[Individual]) -> list[Individual]:
    """返回 所有 非支配 (rank-1) 个体 且不修改 rank."""
    return [candidate for candidate in population if _is_nondominated(candidate, population)]


def _is_nondominated(candidate: Individual, population: list[Individual]) -> bool:
    """返回 True if no other 个体 dominates 该 candidate."""
    for other in population:
        if other is candidate:
            continue
        if dominates(other, candidate):
            return False
    return True


def _compute_diversity_score(objectives: list[tuple[float, ...]]) -> float:
    """计算简单的目标空间多样性分数.
    
        该 MVP 指标为到质心的平均欧氏距离。
        """
    if len(objectives) <= 1:
        return 0.0

    matrix = np.asarray(objectives, dtype=float)
    centroid = matrix.mean(axis=0)
    distances = np.linalg.norm(matrix - centroid, axis=1)
    return float(np.mean(distances))


def _compute_crowding_entropy(rank1_individuals: list[Individual], eps: float = 1e-12) -> float:
    """计算 rank-1 局部邻域距离的熵.
    
        步骤：
        1) For each rank-1 点, 计算 目标-space nearest-neighbor 距离.
        2) Normalize distances 转换为 一个 概率 mass function.
        3) 返回 normalized Shannon entropy 在 [0, 1].
    
        回退规则：
        - ``len(rank1_individuals) < 3`` -> ``0.0`` (结构不足).
        - 所有 nearest-neighbor distances 为 (near) zero -> ``0.0``.
        """
    if len(rank1_individuals) < 3:
        return 0.0

    front = np.asarray([ind.objectives for ind in rank1_individuals], dtype=float)
    if front.ndim != 2 or front.size == 0:
        return 0.0

    distance_matrix = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest = distance_matrix.min(axis=1)
    nearest = np.nan_to_num(nearest, nan=0.0, posinf=0.0, neginf=0.0)

    total = float(np.sum(nearest))
    if total <= eps:
        return 0.0

    probs = np.clip(nearest / total, eps, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)))
    normalizer = float(np.log(len(probs)))
    if normalizer <= eps:
        return 0.0

    return float(np.clip(entropy / normalizer, 0.0, 1.0))


def _compute_decision_diversity(population: list[Individual]) -> float:
    """通过归一化 Hamming 距离均值计算决策空间多样性.
    
        该 分配 encoding 为 discrete (任务 -> 资源 索引), so Euclidean
        距离 为 不 appropriate. We therefore 使用 per-position mismatch ratio.
        """
    if len(population) <= 1:
        return 0.0

    distances: list[float] = []
    for i, lhs in enumerate(population):
        for rhs in population[i + 1 :]:
            distances.append(_normalized_hamming(lhs.genome, rhs.genome))

    if not distances:
        return 0.0
    return float(np.mean(np.asarray(distances, dtype=float)))


def _normalized_hamming(lhs: list[int], rhs: list[int]) -> float:
    """返回带长度不匹配惩罚的归一化 Hamming 距离."""
    max_len = max(len(lhs), len(rhs))
    if max_len == 0:
        return 0.0

    overlap = min(len(lhs), len(rhs))
    mismatches = sum(1 for idx in range(overlap) if lhs[idx] != rhs[idx])
    mismatches += abs(len(lhs) - len(rhs))
    return mismatches / max_len


def _compute_front_separation(
    population: list[Individual],
    rank1_individuals: list[Individual],
    eps: float = 1e-12,
) -> float:
    """衡量 rank-1 与被支配个体之间的分离度.
    
        Multi-目标 replacement 的 单个-目标 ``Dratio``:
        - ``inter``: 均值 nearest 目标-space 距离 从 dominated 点
          到 该 rank-1 set.
        - ``intra``: 均值 pairwise 距离 within 该 rank-1 set.
        - ``d_front = inter / (inter + intra + eps)`` 在 ``[0, 1]``.
    
        解释：
        - high 值: dominated solutions 为 far away while rank-1 前沿 remains
          compact.
        - low 值: dominated 与 rank-1 solutions 为 mixed.
    
        回退规则：
        - no rank-1 (only possible 用于 空 种群): ``0.0``.
        - 所有 个体 为 rank-1: ``1.0``.
        """
    if not rank1_individuals:
        return 0.0

    dominated = [ind for ind in population if not _is_nondominated(ind, population)]
    if not dominated:
        return 1.0

    front = np.asarray([ind.objectives for ind in rank1_individuals], dtype=float)
    dominated_objs = np.asarray([ind.objectives for ind in dominated], dtype=float)

    pairwise_fd = np.linalg.norm(dominated_objs[:, None, :] - front[None, :, :], axis=2)
    inter = float(np.mean(np.min(pairwise_fd, axis=1)))

    if len(rank1_individuals) <= 1:
        intra = 0.0
    else:
        pairwise_ff = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
        upper = pairwise_ff[np.triu_indices(len(rank1_individuals), k=1)]
        intra = float(np.mean(upper)) if upper.size > 0 else 0.0

    score = inter / (inter + intra + eps)
    return float(np.clip(np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))


def _default_reference_point(objectives: list[tuple[float, ...]]) -> tuple[float, ...]:
    """构建 一个 保守的默认 reference 点 从 种群 目标."""
    if not objectives:
        return (1.0, 1.0)

    matrix = np.asarray(objectives, dtype=float)
    maxima = matrix.max(axis=0)
    padding = np.maximum(np.abs(maxima) * 0.1, 1e-6)
    return tuple((maxima + padding).tolist())
