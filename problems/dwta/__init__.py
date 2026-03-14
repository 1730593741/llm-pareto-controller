"""Dynamic Weapon-Target Assignment (DWTA) problem package."""

from problems.dwta.constraints import DWTAConstraintBreakdown, constraint_breakdown
from problems.dwta.model import DWTABenchmarkData, MunitionType, Target, Weapon
from problems.dwta.objectives import compute_objectives
from problems.dwta.precompute import build_precomputed_matrices
from problems.dwta.repair import repair_allocation

__all__ = [
    "DWTABenchmarkData",
    "DWTAConstraintBreakdown",
    "MunitionType",
    "Target",
    "Weapon",
    "build_precomputed_matrices",
    "compute_objectives",
    "constraint_breakdown",
    "repair_allocation",
]
