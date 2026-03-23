"""Paper-level 多目标指标 用于 最小化问题.

该模块实现多目标论文中常用的稳健指标:
- IGD
- IGD+
- Spacing
- Spread (Deb 多样性 指标 用于 双目标前沿)

所有指标都假设目标是最小化，并接收如下形式的目标点
``Sequence[tuple[float, ...]]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ObjectivePoint = tuple[float, ...]
_EPS = 1e-12


def unique_points(points: Sequence[ObjectivePoint], *, ndigits: int = 12) -> list[ObjectivePoint]:
    """返回 去重后的点，并保持稳定顺序与数值容差."""
    seen: set[tuple[float, ...]] = set()
    result: list[ObjectivePoint] = []
    for point in points:
        key = tuple(round(float(v), ndigits) for v in point)
        if key in seen:
            continue
        seen.add(key)
        result.append(tuple(float(v) for v in point))
    return result


def to_matrix(points: Sequence[ObjectivePoint]) -> np.ndarray:
    """将点序列转换为通过校验的二维 float 矩阵."""
    if not points:
        return np.empty((0, 0), dtype=float)
    matrix = np.asarray(points, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("points must be a 2D collection")
    return matrix


def igd(obtained_front: Sequence[ObjectivePoint], reference_front: Sequence[ObjectivePoint]) -> float:
    """计算 IGD (Inverted Generational Distance).
    
        IGD = mean_{r 在 R} min_{一个 在 A} ||r - 一个||_2
        where R 为 reference 前沿 与 A 为 obtained 前沿.
    
        返回：
          - ``0.0`` 当 reference 前沿 为 空.
          - ``inf`` 当 obtained 前沿 为 空 but reference 前沿 为 不 空.
        """
    ref = to_matrix(unique_points(reference_front))
    if ref.size == 0:
        return 0.0
    got = to_matrix(unique_points(obtained_front))
    if got.size == 0:
        return float("inf")
    if ref.shape[1] != got.shape[1]:
        raise ValueError("obtained_front and reference_front must have same objective dimension")

    distances = np.linalg.norm(ref[:, None, :] - got[None, :, :], axis=2)
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest))


def igd_plus(obtained_front: Sequence[ObjectivePoint], reference_front: Sequence[ObjectivePoint]) -> float:
    """计算 IGD+.
    
        IGD+ uses 该 modified 距离:
          d+(r, 一个) = || max(一个 - r, 0) ||_2
        用于 最小化问题.
    
        返回：
          - ``0.0`` 当 reference 前沿 为 空.
          - ``inf`` 当 obtained 前沿 为 空 but reference 前沿 为 不 空.
        """
    ref = to_matrix(unique_points(reference_front))
    if ref.size == 0:
        return 0.0
    got = to_matrix(unique_points(obtained_front))
    if got.size == 0:
        return float("inf")
    if ref.shape[1] != got.shape[1]:
        raise ValueError("obtained_front and reference_front must have same objective dimension")

    diff = np.maximum(got[None, :, :] - ref[:, None, :], 0.0)
    distances = np.linalg.norm(diff, axis=2)
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest))


def spacing(front: Sequence[ObjectivePoint]) -> float:
    """计算 Spacing 指标.
    
        For each 点 i, 计算 d_i 作为 nearest-neighbor L1 距离.
        Spacing = sqrt(sum((d_i - 均值(d))^2) / (n - 1)).
    
        返回s 0.0 当 less than 2 点 为 可用.
        """
    matrix = to_matrix(unique_points(front))
    if matrix.shape[0] < 2:
        return 0.0

    pairwise = np.sum(np.abs(matrix[:, None, :] - matrix[None, :, :]), axis=2)
    np.fill_diagonal(pairwise, np.inf)
    d = np.min(pairwise, axis=1)
    if d.shape[0] < 2:
        return 0.0
    return float(np.std(d, ddof=1))


def spread(front: Sequence[ObjectivePoint], reference_front: Sequence[ObjectivePoint]) -> float:
    """计算 Deb's spread (Δ) 用于 双目标前沿.
    
        For 2 目标 (minimization), 点 为 sorted 通过 目标-0.
        Let d_f, d_l be distances 从 obtained extreme 点 到 reference extremes,
        与 d_i be distances between consecutive obtained 点.
    
        Δ = (d_f + d_l + sum(|d_i - 均值(d)|)) / (d_f + d_l + (n-1)*均值(d))
    
        返回：
          - ``0.0`` if both obtained 与 reference 前沿 为 空.
          - ``1.0`` 当 obtained 前沿 has <2 点 but reference 为 non-空.
        """
    got = to_matrix(unique_points(front))
    ref = to_matrix(unique_points(reference_front))
    if ref.size == 0 and got.size == 0:
        return 0.0
    if got.size == 0:
        return 1.0
    if got.shape[1] != 2:
        raise ValueError("spread currently supports exactly 2 objectives")
    if ref.size == 0:
        # no explicit extremes 可用, degrade gracefully
        ref = got.copy()
    if ref.shape[1] != 2:
        raise ValueError("spread currently supports exactly 2 objectives")

    got = got[np.argsort(got[:, 0])]
    ref = ref[np.argsort(ref[:, 0])]

    if got.shape[0] < 2:
        return 1.0

    d_f = float(np.linalg.norm(got[0] - ref[0]))
    d_l = float(np.linalg.norm(got[-1] - ref[-1]))

    consecutive = np.linalg.norm(got[1:] - got[:-1], axis=1)
    mean_d = float(np.mean(consecutive))
    if mean_d <= _EPS:
        return 0.0 if d_f <= _EPS and d_l <= _EPS else 1.0

    numerator = d_f + d_l + float(np.sum(np.abs(consecutive - mean_d)))
    denominator = d_f + d_l + (got.shape[0] - 1) * mean_d
    if denominator <= _EPS:
        return 0.0
    return float(numerator / denominator)
