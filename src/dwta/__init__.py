"""DWTA 领域模块。"""

from src.dwta.constraints import DWTAConstraintBreakdown, constraint_breakdown
from src.dwta.dynamics import DWTALiveCache, DWTALiveSnapshot, build_dynamic_scenario, build_scenario_matrices, distance
from src.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, random_allocation, to_genome, to_matrix
from src.dwta.evaluation import DWTAEvaluationResult, evaluate_allocation
from src.dwta.model import (
    DWTABenchmarkData,
    DWTAEnvironment,
    DWTAScenarioScript,
    DWTAWaveEvent,
    MunitionType,
    Target,
    Weapon,
)
from src.dwta.objectives import compute_objectives
from src.dwta.precompute import build_precomputed_matrices
from src.dwta.repair import repair_allocation

__all__ = [
    "DWTAAllocationGenome",
    "DWTAAllocationMatrix",
    "DWTABenchmarkData",
    "DWTAEnvironment",
    "DWTALiveCache",
    "DWTALiveSnapshot",
    "DWTAScenarioScript",
    "DWTAConstraintBreakdown",
    "DWTAEvaluationResult",
    "DWTAWaveEvent",
    "MunitionType",
    "Target",
    "Weapon",
    "build_dynamic_scenario",
    "build_precomputed_matrices",
    "build_scenario_matrices",
    "compute_objectives",
    "constraint_breakdown",
    "distance",
    "evaluate_allocation",
    "random_allocation",
    "repair_allocation",
    "to_genome",
    "to_matrix",
]
