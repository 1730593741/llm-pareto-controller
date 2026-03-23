"""种群数据结构 用于 该 NSGA-II 优化器."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from problems.task_assignment.encoding import Assignment


@dataclass(slots=True)
class Individual:
    """A 单个 NSGA-II 候选解."""

    genome: Assignment
    objectives: tuple[float, ...] = field(default_factory=tuple)
    constraint_violation: float = 0.0
    feasible: bool = True
    constraint_components: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0

    def copy(self) -> "Individual":
        """返回 一个 shallow 值 copy，并带有 independent genome list."""
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
    """A 便捷封装 over 一个 list 的 个体."""

    individuals: list[Individual] = field(default_factory=list)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    def append(self, individual: Individual) -> None:
        """Append 一个 个体 到 this 种群."""
        self.individuals.append(individual)

    def extend(self, individuals: list[Individual]) -> None:
        """Extend，并带有 multiple 个体."""
        self.individuals.extend(individuals)
