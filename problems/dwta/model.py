"""Domain models and bundle types for Dynamic Weapon-Target Assignment (DWTA)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MunitionType:
    """Munition characteristics shared by mounted weapons."""

    id: str
    max_range: float
    flight_speed: float
    lethality: float


@dataclass(frozen=True, slots=True)
class Weapon:
    """Weapon platform with fixed position and mounted munition type."""

    id: str
    x: float
    y: float
    munition_type_id: str
    ammo_capacity: int


@dataclass(frozen=True, slots=True)
class Target:
    """Target definition with location, required damage, and exposure window."""

    id: str
    x: float
    y: float
    required_damage: float
    time_window: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DWTABenchmarkData:
    """Precomputed DWTA arrays consumed by the NSGA-II solver."""

    n_weapons: int
    n_targets: int
    ammo_capacities: list[int]
    compatibility_matrix: list[list[int]]
    lethality_matrix: list[list[float]]
    required_damage: list[float]
