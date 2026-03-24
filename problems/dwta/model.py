"""领域模型与打包类型 用于 Dynamic Weapon-Target Assignment (DWTA).

本模块当前同时承载两类模型：
1) 现有静态求解路径使用的 ``DWTABenchmarkData``（保持向后兼容）；
2) 新增动态场景建模对象（脚本波次 schema），用于后续动态求解扩展。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MunitionType:
    """Munition characteristics shared 通过 mounted Weapons."""

    id: str
    max_range: float
    flight_speed: float
    lethality: float


@dataclass(frozen=True, slots=True)
class Weapon:
    """Weapon platform，并带有 fixed position 与 mounted Munition type."""

    id: str
    x: float
    y: float
    munition_type_id: str
    ammo_capacity: int


@dataclass(frozen=True, slots=True)
class Target:
    """Target definition，并带有 location, 必需的 damage, 与 exposure window."""

    id: str
    x: float
    y: float
    required_damage: float
    time_window: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DWTABenchmarkData:
    """Precomputed DWTA arrays consumed 通过 该 NSGA-II 求解器."""

    n_weapons: int
    n_targets: int
    ammo_capacities: list[int]
    compatibility_matrix: list[list[int]]
    lethality_matrix: list[list[float]]
    required_damage: list[float]


@dataclass(frozen=True, slots=True)
class DWTAWaveEvent:
    """单个波次事件定义（当前仅建模，不驱动主求解逻辑）.

    Attributes:
        wave_id: 波次标识符，便于日志与调试追踪。
        trigger_generation: 建议触发代数（后续阶段由 runner/环境执行器消费）。
        target_damage_scale: 当前波次对目标 required_damage 的统一缩放因子。
        compatibility_override: 可选兼容矩阵覆盖（行=weapon, 列=target）。
        note: 事件说明，便于实验脚本注释。
    """

    wave_id: str
    trigger_generation: int
    target_damage_scale: float = 1.0
    compatibility_override: list[list[int]] | None = None
    note: str | None = None


@dataclass(slots=True)
class DWTAScenarioScript:
    """DWTA 波次脚本容器（用于 scripted_waves 模式）."""

    waves: list[DWTAWaveEvent]


@dataclass(slots=True)
class DWTAEnvironment:
    """DWTA 动态环境对象（可变），为后续动态闭环留接口.

    说明：
    - ``base_data`` 对应当前求解器可直接消费的静态矩阵快照；
    - ``script`` 为可选波次脚本，当前阶段仅完成加载与组装；
    - ``max_weapons/max_targets`` 预留大规模动态场景容量信息。
    """

    base_data: DWTABenchmarkData
    scenario_mode: str = "static"
    max_weapons: int | None = None
    max_targets: int | None = None
    script: DWTAScenarioScript | None = None
