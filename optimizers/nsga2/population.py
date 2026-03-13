"""Population data structures for the NSGA-II optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from problems.task_assignment.encoding import Assignment


@dataclass(slots=True)
class Individual:
    """A single NSGA-II candidate solution."""

    genome: Assignment
    objectives: tuple[float, ...] = field(default_factory=tuple)
    constraint_violation: float = 0.0
    feasible: bool = True
    constraint_components: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0

    def copy(self) -> "Individual":
        """Return a shallow value copy with independent genome list."""
        return Individual(
            genome=list(self.genome),
            objectives=tuple(self.objectives),
            constraint_violation=self.constraint_violation,
            feasible=self.feasible,
            constraint_components=dict(self.constraint_components),
            rank=self.rank,
            crowding_distance=self.crowding_distance,
        )


@dataclass(slots=True)
class Population:
    """A convenience wrapper over a list of individuals."""

    individuals: list[Individual] = field(default_factory=list)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    def append(self, individual: Individual) -> None:
        """Append one individual to this population."""
        self.individuals.append(individual)

    def extend(self, individuals: list[Individual]) -> None:
        """Extend with multiple individuals."""
        self.individuals.extend(individuals)
