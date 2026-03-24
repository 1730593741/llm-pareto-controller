"""Dynamic Weapon-Target Assignment (DWTA) 问题包."""

from problems.dwta.constraints import DWTAConstraintBreakdown, constraint_breakdown
from problems.dwta.encoding import DWTAAllocationGenome, DWTAAllocationMatrix, random_allocation, to_genome, to_matrix
from problems.dwta.live_cache import DWTALiveCache, DWTALiveSnapshot
from problems.dwta.model import (
    DWTABenchmarkData,
    DWTAEnvironment,
    DWTAScenarioScript,
    DWTAWaveEvent,
    MunitionType,
    Target,
    Weapon,
)
from problems.dwta.objectives import compute_objectives
from problems.dwta.precompute import build_precomputed_matrices
from problems.dwta.repair import repair_allocation
from problems.dwta.scenario_builder import build_dynamic_scenario, build_scenario_matrices

__all__ = [
    "DWTAAllocationGenome",
    "DWTAAllocationMatrix",
    "DWTABenchmarkData",
    "DWTAEnvironment",
    "DWTALiveCache",
    "DWTALiveSnapshot",
    "DWTAScenarioScript",
    "DWTAConstraintBreakdown",
    "DWTAWaveEvent",
    "MunitionType",
    "Target",
    "Weapon",
    "build_dynamic_scenario",
    "build_precomputed_matrices",
    "build_scenario_matrices",
    "compute_objectives",
    "constraint_breakdown",
    "random_allocation",
    "repair_allocation",
    "to_genome",
    "to_matrix",
]
