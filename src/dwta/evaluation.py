"""DWTA 个体评估封装。"""

from __future__ import annotations

from dataclasses import dataclass

from src.dwta.constraints import DWTAConstraintBreakdown, constraint_breakdown
from src.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix
from src.dwta.live_cache import DWTALiveCache
from src.dwta.objectives import compute_objectives


@dataclass(frozen=True, slots=True)
class DWTAEvaluationResult:
    """DWTA 解评估结果（目标 + 约束分解）。"""

    objectives: tuple[float, float]
    constraint_breakdown: DWTAConstraintBreakdown

    @property
    def constraint_violation(self) -> float:
        return self.constraint_breakdown.total

    @property
    def feasible(self) -> bool:
        return self.constraint_violation <= 0.0


def evaluate_allocation(
    allocation: DWTAAllocationGenome | DWTAAllocationMatrix,
    *,
    n_weapons: int,
    n_targets: int,
    required_damage: list[float],
    lethality_matrix: list[list[float]],
    ammo_capacities: list[int],
    compatibility_matrix: list[list[int]],
    live_cache: DWTALiveCache | None = None,
) -> DWTAEvaluationResult:
    """统一计算 DWTA allocation 的 objectives 与 constraints。"""
    objectives = compute_objectives(
        allocation,
        n_weapons=n_weapons,
        n_targets=n_targets,
        required_damage=required_damage,
        lethality_matrix=lethality_matrix,
        live_cache=live_cache,
    )
    breakdown = constraint_breakdown(
        allocation,
        ammo_capacities=ammo_capacities,
        compatibility_matrix=compatibility_matrix,
        n_targets=n_targets,
        live_cache=live_cache,
    )
    return DWTAEvaluationResult(objectives=objectives, constraint_breakdown=breakdown)
