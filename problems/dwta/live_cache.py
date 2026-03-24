"""Dynamic cache layer for DWTA environment-dependent matrices.

该模块把“环境依赖但个体无关”的推导量集中缓存，并支持：
- 初次构建（lazy）；
- 手动失效（invalidate）；
- 事件后刷新（refresh）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from problems.dwta.model import DWTABenchmarkData, DWTAEnvironment


@dataclass(frozen=True, slots=True)
class DWTALiveSnapshot:
    """Live cache snapshot used by objective/constraint/repair modules."""

    n_weapons: int
    n_targets: int
    ammo_capacities: np.ndarray
    required_damage: np.ndarray
    distance_matrix: np.ndarray
    flight_time_matrix: np.ndarray
    compatibility_mask: np.ndarray
    lethality_matrix: np.ndarray
    weapon_active_mask: np.ndarray
    target_active_mask: np.ndarray

    def as_benchmark_data(self) -> DWTABenchmarkData:
        """Convert snapshot to legacy benchmark payload for backward compatibility."""
        return DWTABenchmarkData(
            n_weapons=self.n_weapons,
            n_targets=self.n_targets,
            ammo_capacities=self.ammo_capacities.astype(int).tolist(),
            compatibility_matrix=self.compatibility_mask.astype(int).tolist(),
            lethality_matrix=self.lethality_matrix.astype(float).tolist(),
            required_damage=self.required_damage.astype(float).tolist(),
        )


class DWTALiveCache:
    """Cache for DWTA matrices that depend on mutable environment state.

    Notes:
        - ``environment.state_epoch`` 可由外部事件处理器更新；
        - 调用 ``refresh`` 会在 epoch 变化或缓存失效时重建；
        - 也可显式调用 ``invalidate`` 强制下一次 refresh 重算。
    """

    def __init__(self, environment: DWTAEnvironment) -> None:
        self.environment = environment
        self._snapshot: DWTALiveSnapshot | None = None
        self._cached_epoch: int | None = None

    def invalidate(self) -> None:
        """Manually mark current cache as invalid."""
        self._snapshot = None
        self._cached_epoch = None

    def refresh(self, *, force: bool = False) -> DWTALiveSnapshot:
        """Rebuild cache if invalidated or environment epoch changed."""
        if not force and self._snapshot is not None and self._cached_epoch == self.environment.state_epoch:
            return self._snapshot

        snapshot = self._build_snapshot()
        self._snapshot = snapshot
        self._cached_epoch = self.environment.state_epoch
        # Keep the legacy static snapshot in sync for old call sites.
        self.environment.base_data = snapshot.as_benchmark_data()
        return snapshot

    def get_snapshot(self) -> DWTALiveSnapshot:
        """Return an up-to-date cache snapshot (lazy refresh)."""
        return self.refresh(force=False)

    def as_benchmark_data(self) -> DWTABenchmarkData:
        """Compatibility adapter for static DWTA code paths."""
        return self.get_snapshot().as_benchmark_data()

    def _build_snapshot(self) -> DWTALiveSnapshot:
        """Build all environment-dependent matrices via NumPy vectorization."""
        weapons = self.environment.weapons
        targets = self.environment.targets
        munitions = {item.id: item for item in self.environment.munitions}

        n_weapons = len(weapons)
        n_targets = len(targets)

        if n_weapons == 0 or n_targets == 0:
            zeros = np.zeros((n_weapons, n_targets), dtype=float)
            return DWTALiveSnapshot(
                n_weapons=n_weapons,
                n_targets=n_targets,
                ammo_capacities=np.array([weapon.ammo_capacity for weapon in weapons], dtype=int),
                required_damage=np.array([target.required_damage for target in targets], dtype=float),
                distance_matrix=zeros,
                flight_time_matrix=zeros,
                compatibility_mask=zeros.astype(bool),
                lethality_matrix=zeros,
                weapon_active_mask=np.array([weapon.ammo_capacity > 0 for weapon in weapons], dtype=bool),
                target_active_mask=np.array([target.required_damage > 0.0 for target in targets], dtype=bool),
            )

        weapon_xy = np.array([(weapon.x, weapon.y) for weapon in weapons], dtype=float)
        target_xy = np.array([(target.x, target.y) for target in targets], dtype=float)
        diff = weapon_xy[:, np.newaxis, :] - target_xy[np.newaxis, :, :]
        distance_matrix = np.linalg.norm(diff, axis=2)

        weapon_munitions = []
        for weapon in weapons:
            munition = munitions.get(weapon.munition_type_id)
            if munition is None:
                raise ValueError(
                    f"weapon '{weapon.id}' references unknown munition_type_id '{weapon.munition_type_id}'"
                )
            weapon_munitions.append(munition)

        max_range = np.array([munition.max_range for munition in weapon_munitions], dtype=float)[:, np.newaxis]
        speed = np.array([munition.flight_speed for munition in weapon_munitions], dtype=float)[:, np.newaxis]
        lethality = np.array([munition.lethality for munition in weapon_munitions], dtype=float)[:, np.newaxis]

        safe_speed = np.where(speed > 0.0, speed, np.nan)
        flight_time_matrix = np.divide(distance_matrix, safe_speed)
        flight_time_matrix = np.where(np.isnan(flight_time_matrix), np.inf, flight_time_matrix)

        target_windows = np.array([target.time_window for target in targets], dtype=float)
        t_start = target_windows[:, 0][np.newaxis, :]
        t_end = target_windows[:, 1][np.newaxis, :]

        in_range = distance_matrix <= max_range
        in_window = (flight_time_matrix >= t_start) & (flight_time_matrix <= t_end)

        weapon_active = np.array([weapon.ammo_capacity > 0 for weapon in weapons], dtype=bool)
        target_active = np.array([target.required_damage > 0.0 for target in targets], dtype=bool)

        compatibility_mask = (
            in_range
            & in_window
            & weapon_active[:, np.newaxis]
            & target_active[np.newaxis, :]
        )

        lethality_matrix = np.repeat(lethality, n_targets, axis=1)
        lethality_matrix = np.where(compatibility_mask, lethality_matrix, 0.0)

        return DWTALiveSnapshot(
            n_weapons=n_weapons,
            n_targets=n_targets,
            ammo_capacities=np.array([weapon.ammo_capacity for weapon in weapons], dtype=int),
            required_damage=np.array([target.required_damage for target in targets], dtype=float),
            distance_matrix=distance_matrix,
            flight_time_matrix=flight_time_matrix,
            compatibility_mask=compatibility_mask,
            lethality_matrix=lethality_matrix,
            weapon_active_mask=weapon_active,
            target_active_mask=target_active,
        )
