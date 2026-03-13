"""Paper-level multi-objective metrics for minimization problems.

This module implements robust metrics used in multi-objective papers:
- IGD
- IGD+
- Spacing
- Spread (Deb diversity metric for bi-objective fronts)

All metrics assume minimization objectives and accept objective points as
``Sequence[tuple[float, ...]]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ObjectivePoint = tuple[float, ...]
_EPS = 1e-12


def unique_points(points: Sequence[ObjectivePoint], *, ndigits: int = 12) -> list[ObjectivePoint]:
    """Return de-duplicated points with stable order and numeric tolerance."""
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
    """Convert point sequence to a validated 2D float matrix."""
    if not points:
        return np.empty((0, 0), dtype=float)
    matrix = np.asarray(points, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("points must be a 2D collection")
    return matrix


def igd(obtained_front: Sequence[ObjectivePoint], reference_front: Sequence[ObjectivePoint]) -> float:
    """Compute IGD (Inverted Generational Distance).

    IGD = mean_{r in R} min_{a in A} ||r - a||_2
    where R is reference front and A is obtained front.

    Returns:
      - ``0.0`` when reference front is empty.
      - ``inf`` when obtained front is empty but reference front is not empty.
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
    """Compute IGD+.

    IGD+ uses the modified distance:
      d+(r, a) = || max(a - r, 0) ||_2
    for minimization problems.

    Returns:
      - ``0.0`` when reference front is empty.
      - ``inf`` when obtained front is empty but reference front is not empty.
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
    """Compute Spacing metric.

    For each point i, compute d_i as nearest-neighbor L1 distance.
    Spacing = sqrt(sum((d_i - mean(d))^2) / (n - 1)).

    Returns 0.0 when less than 2 points are available.
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
    """Compute Deb's spread (Δ) for bi-objective fronts.

    For 2 objectives (minimization), points are sorted by objective-0.
    Let d_f, d_l be distances from obtained extreme points to reference extremes,
    and d_i be distances between consecutive obtained points.

    Δ = (d_f + d_l + sum(|d_i - mean(d)|)) / (d_f + d_l + (n-1)*mean(d))

    Returns:
      - ``0.0`` if both obtained and reference fronts are empty.
      - ``1.0`` when obtained front has <2 points but reference is non-empty.
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
        # no explicit extremes available, degrade gracefully
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
