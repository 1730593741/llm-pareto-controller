"""场景预处理与固定画布构建 用于 Dynamic Weapon-Target Assignment (DWTA)."""

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
    """Precompute compatibility 与 lethality matrices once at setup time."""
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
    """构建 DWTAEnvironment，并在 scripted_waves 模式启用固定画布策略.

    Compatibility strategy:
    - static 模式保留原始规模，不做补齐；
    - scripted_waves 模式按 ``max_weapons/max_targets`` 补齐占位实体；
    - 占位武器 ammo=0、占位目标 required_damage=0，用于 active-mask 控制。
    """
    if scenario_mode == "scripted_waves":
        if max_weapons is None:
            max_weapons = len(weapons)
        if max_targets is None:
            max_targets = len(targets)
        if max_weapons < len(weapons):
            raise ValueError("max_weapons must be >= initial weapon count")
        if max_targets < len(targets):
            raise ValueError("max_targets must be >= initial target count")
        if not munition_types:
            raise ValueError("scripted_waves mode requires at least one munition type")

        default_munition_id = munition_types[0].id
        padded_weapons = list(weapons)
        padded_targets = list(targets)

        for idx in range(len(padded_weapons), max_weapons):
            padded_weapons.append(
                Weapon(
                    id=f"__inactive_weapon_{idx}",
                    x=0.0,
                    y=0.0,
                    munition_type_id=default_munition_id,
                    ammo_capacity=0,
                )
            )

        for idx in range(len(padded_targets), max_targets):
            padded_targets.append(
                Target(
                    id=f"__inactive_target_{idx}",
                    x=0.0,
                    y=0.0,
                    required_damage=0.0,
                    time_window=(0.0, 0.0),
                )
            )

        weapons = padded_weapons
        targets = padded_targets

    base_data = build_scenario_matrices(munition_types, weapons, targets)
    script = DWTAScenarioScript(waves=waves or []) if scenario_mode == "scripted_waves" else None
    return DWTAEnvironment(
        base_data=base_data,
        scenario_mode=scenario_mode,
        max_weapons=max_weapons,
        max_targets=max_targets,
        script=script,
        munitions=list(munition_types),
        weapons=list(weapons),
        targets=list(targets),
    )
