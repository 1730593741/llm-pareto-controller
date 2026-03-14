"""Precomputation utilities for DWTA compatibility and lethality matrices."""

from __future__ import annotations

from problems.dwta.model import DWTABenchmarkData, MunitionType, Target, Weapon
from problems.dwta.scenario_builder import build_scenario_matrices


def build_precomputed_matrices(
    munitions: list[MunitionType],
    weapons: list[Weapon],
    targets: list[Target],
) -> DWTABenchmarkData:
    """Backward-compatible entrypoint for DWTA setup-time preprocessing."""
    return build_scenario_matrices(munitions, weapons, targets)
