"""预计算工具 用于 DWTA 兼容性 与 lethality matrices."""

from __future__ import annotations

from src.dwta.model import DWTABenchmarkData, MunitionType, Target, Weapon
from src.dwta.scenario_builder import build_scenario_matrices


def build_precomputed_matrices(
    munitions: list[MunitionType],
    weapons: list[Weapon],
    targets: list[Target],
) -> DWTABenchmarkData:
    """向后兼容的入口 用于 DWTA setup-time preprocessing."""
    return build_scenario_matrices(munitions, weapons, targets)
