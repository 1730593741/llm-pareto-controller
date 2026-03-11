"""Pareto-state sensing for NSGA-II population snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import dominates
from sensing.feasibility_metrics import feasible_ratio, mean_constraint_violation
from sensing.hypervolume import HypervolumeCalculator, SimplifiedHypervolumeCalculator


@dataclass(slots=True)
class ParetoState:
    """Structured snapshot of current optimizer search state."""

    generation: int
    hv: float
    delta_hv: float
    feasible_ratio: float
    rank1_ratio: float
    mean_cv: float
    diversity_score: float
    stagnation_len: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a dictionary for logging."""
        return asdict(self)


class ParetoStateSensor:
    """Build ParetoState from current population with replaceable metrics."""

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
        """Generate a ParetoState from current population and previous state."""
        if generation < 0:
            raise ValueError("generation must be >= 0")

        violations = [ind.constraint_violation for ind in population]
        objectives = [ind.objectives for ind in population if ind.objectives]

        current_feasible_ratio = feasible_ratio(violations)
        current_mean_cv = mean_constraint_violation(violations)
        current_rank1_ratio = _compute_rank1_ratio(population)
        current_diversity = _compute_diversity_score(objectives)

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
            stagnation_len=stagnation_len,
        )


def _compute_rank1_ratio(population: list[Individual]) -> float:
    """Compute ratio of first-front individuals in the given population."""
    if not population:
        return 0.0

    rank1_size = sum(1 for candidate in population if _is_nondominated(candidate, population))
    return rank1_size / len(population)


def _is_nondominated(candidate: Individual, population: list[Individual]) -> bool:
    """Return True if no other individual dominates the candidate."""
    for other in population:
        if other is candidate:
            continue
        if dominates(other, candidate):
            return False
    return True


def _compute_diversity_score(objectives: list[tuple[float, ...]]) -> float:
    """Compute a simple objective-space diversity score.

    This MVP metric is the average Euclidean distance to the centroid.
    """
    if len(objectives) <= 1:
        return 0.0

    matrix = np.asarray(objectives, dtype=float)
    centroid = matrix.mean(axis=0)
    distances = np.linalg.norm(matrix - centroid, axis=1)
    return float(np.mean(distances))


def _default_reference_point(objectives: list[tuple[float, ...]]) -> tuple[float, ...]:
    """Build a conservative default reference point from population objectives."""
    if not objectives:
        return (1.0, 1.0)

    matrix = np.asarray(objectives, dtype=float)
    maxima = matrix.max(axis=0)
    padding = np.maximum(np.abs(maxima) * 0.1, 1e-6)
    return tuple((maxima + padding).tolist())
