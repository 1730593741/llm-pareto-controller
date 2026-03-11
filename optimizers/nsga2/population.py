"""Population data structures for the NSGA-II optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field

from problems.task_assignment.encoding import Assignment


@dataclass(slots=True)
class Individual:
    """A single solution in the NSGA-II population."""

    genome: Assignment
    objectives: tuple[float, float] = (0.0, 0.0)
    constraint_violation: float = 0.0
    feasible: bool = True
    rank: int = 0
    crowding_distance: float = 0.0


@dataclass(slots=True)
class Population:
    """Container for a list of individuals."""

    individuals: list[Individual] = field(default_factory=list)

    def __len__(self) -> int:
        """Return number of individuals."""
        return len(self.individuals)
