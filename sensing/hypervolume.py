"""Hypervolume utilities for Pareto-state sensing.

This module currently exposes a stable interface with a simplified 2-objective
implementation to keep the MVP lightweight and replaceable.
"""

from __future__ import annotations

from collections.abc import Sequence

ObjectivePoint = tuple[float, ...]


class HypervolumeCalculator:
    """Stable, replaceable hypervolume calculator interface."""

    def compute(self, points: Sequence[ObjectivePoint], reference_point: ObjectivePoint) -> float:
        """Compute hypervolume for a minimization front and reference point."""
        raise NotImplementedError


class SimplifiedHypervolumeCalculator(HypervolumeCalculator):
    """Simplified MVP hypervolume implementation.

    Notes:
      - Supports only two objectives for now.
      - Assumes minimization objectives.
      - Accumulates rectangle slices after filtering dominated points.
    """

    def compute(self, points: Sequence[ObjectivePoint], reference_point: ObjectivePoint) -> float:
        """Compute simplified 2D hypervolume.

        Returns 0.0 for empty inputs. Raises ValueError if dimensions are invalid.
        """
        if not points:
            return 0.0
        if len(reference_point) != 2:
            raise ValueError("Simplified hypervolume currently supports 2 objectives")

        filtered: list[tuple[float, float]] = []
        for point in points:
            if len(point) != 2:
                raise ValueError("Simplified hypervolume currently supports 2 objectives")
            p0, p1 = point
            r0, r1 = reference_point
            if p0 <= r0 and p1 <= r1:
                filtered.append((p0, p1))

        if not filtered:
            return 0.0

        nondominated = _filter_nondominated_2d(filtered)
        sorted_points = sorted(nondominated, key=lambda value: value[0])

        hv = 0.0
        current_y_limit = reference_point[1]
        for x, y in sorted_points:
            if y >= current_y_limit:
                continue
            width = max(0.0, reference_point[0] - x)
            height = current_y_limit - y
            hv += width * height
            current_y_limit = y
        return hv


def compute_hypervolume(points: Sequence[ObjectivePoint], reference_point: ObjectivePoint) -> float:
    """Compute hypervolume via the default simplified calculator."""
    return SimplifiedHypervolumeCalculator().compute(points, reference_point)


def _filter_nondominated_2d(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Filter non-dominated points for two-objective minimization."""
    result: list[tuple[float, float]] = []
    for point in points:
        dominated = False
        for other in points:
            if other != point and other[0] <= point[0] and other[1] <= point[1]:
                dominated = True
                break
        if not dominated:
            result.append(point)
    return result
