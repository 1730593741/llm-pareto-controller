"""Pareto-state sensing for NSGA-II population snapshots."""

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
    """Structured snapshot of current optimizer search state."""

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
    """Compute ratio of first-front individuals in the given population."""
    if not population:
        return 0.0

    rank1_size = sum(1 for candidate in population if _is_nondominated(candidate, population))
    return rank1_size / len(population)


def _rank1_individuals(population: list[Individual]) -> list[Individual]:
    """Return all non-dominated (rank-1) individuals without mutating ranks."""
    return [candidate for candidate in population if _is_nondominated(candidate, population)]


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


def _compute_crowding_entropy(rank1_individuals: list[Individual], eps: float = 1e-12) -> float:
    """Compute entropy of rank-1 local-neighborhood distances.

    Steps:
    1) For each rank-1 point, compute objective-space nearest-neighbor distance.
    2) Normalize distances into a probability mass function.
    3) Return normalized Shannon entropy in [0, 1].

    Fallbacks:
    - ``len(rank1_individuals) < 3`` -> ``0.0`` (insufficient structure).
    - all nearest-neighbor distances are (near) zero -> ``0.0``.
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
    """Compute decision-space diversity via mean normalized Hamming distance.

    The assignment encoding is discrete (task -> resource index), so Euclidean
    distance is not appropriate. We therefore use per-position mismatch ratio.
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
    """Return normalized Hamming distance with length-mismatch penalty."""
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
    """Measure separation between rank-1 and dominated individuals.

    Multi-objective replacement of single-objective ``Dratio``:
    - ``inter``: mean nearest objective-space distance from dominated points
      to the rank-1 set.
    - ``intra``: mean pairwise distance within the rank-1 set.
    - ``d_front = inter / (inter + intra + eps)`` in ``[0, 1]``.

    Interpretation:
    - high value: dominated solutions are far away while rank-1 front remains
      compact.
    - low value: dominated and rank-1 solutions are mixed.

    Fallbacks:
    - no rank-1 (only possible for empty population): ``0.0``.
    - all individuals are rank-1: ``1.0``.
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
    """Build a conservative default reference point from population objectives."""
    if not objectives:
        return (1.0, 1.0)

    matrix = np.asarray(objectives, dtype=float)
    maxima = matrix.max(axis=0)
    padding = np.maximum(np.abs(maxima) * 0.1, 1e-6)
    return tuple((maxima + padding).tolist())
