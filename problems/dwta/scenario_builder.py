"""场景预处理 用于 Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

import math

from problems.dwta.model import (
    DWTABenchmarkData,
    DWTAEnvironment,
    DWTAScenarioScript,
    DWTAWaveEvent,
    MunitionType,
    Target,
    Weapon,
)


def distance(weapon: Weapon, target: Target) -> float:
    """计算单个 Weapon 与单个 Target 之间的欧氏距离."""
    return math.hypot(weapon.x - target.x, weapon.y - target.y)


def build_scenario_matrices(
    munition_types: list[MunitionType],
    weapons: list[Weapon],
    targets: list[Target],
) -> DWTABenchmarkData:
    """Precompute 兼容性 与 lethality matrices once at setup time.
    
        Compatibility 规则:
        - 距离 <= max_range
        - flight_time 在 [Target.time_window.start, Target.time_window.end]
        """
    munition_by_id = {munition.id: munition for munition in munition_types}

    compatibility_matrix: list[list[int]] = []
    lethality_matrix: list[list[float]] = []

    for weapon in weapons:
        if weapon.munition_type_id not in munition_by_id:
            raise ValueError(f"weapon '{weapon.id}' references unknown munition_type_id '{weapon.munition_type_id}'")
        munition = munition_by_id[weapon.munition_type_id]
        compat_row: list[int] = []
        lethality_row: list[float] = []

        for target in targets:
            d = distance(weapon, target)
            in_range = d <= munition.max_range
            flight_time = d / munition.flight_speed if munition.flight_speed > 0 else float("inf")
            t_start, t_end = target.time_window
            in_time_window = float(t_start) <= flight_time <= float(t_end)
            compat_row.append(1 if in_range and in_time_window else 0)
            lethality_row.append(float(munition.lethality))

        compatibility_matrix.append(compat_row)
        lethality_matrix.append(lethality_row)

    return DWTABenchmarkData(
        n_weapons=len(weapons),
        n_targets=len(targets),
        ammo_capacities=[weapon.ammo_capacity for weapon in weapons],
        compatibility_matrix=compatibility_matrix,
        lethality_matrix=lethality_matrix,
        required_damage=[target.required_damage for target in targets],
    )


def build_dynamic_scenario(
    munition_types: list[MunitionType],
    weapons: list[Weapon],
    targets: list[Target],
    *,
    scenario_mode: str = "static",
    max_weapons: int | None = None,
    max_targets: int | None = None,
    waves: list[DWTAWaveEvent] | None = None,
) -> DWTAEnvironment:
    """构建 DWTAEnvironment（动态环境对象）并返回初始静态快照.

    当前阶段只负责“可解析、可组装”：
    - 总是先构建一次 base ``DWTABenchmarkData``；
    - 当 ``scenario_mode=scripted_waves`` 时，附加脚本对象；
    - 不在此处修改主求解逻辑或运行期行为。
    """
    base_data = build_scenario_matrices(munition_types, weapons, targets)
    script = DWTAScenarioScript(waves=waves or []) if scenario_mode == "scripted_waves" else None
    return DWTAEnvironment(
        base_data=base_data,
        scenario_mode=scenario_mode,
        max_weapons=max_weapons,
        max_targets=max_targets,
        script=script,
    )
