"""领域模型与打包类型 用于 Dynamic Weapon-Target Assignment (DWTA)."""

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
