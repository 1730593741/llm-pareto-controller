"""领域模型与运行时事件类型 用于 Dynamic Weapon-Target Assignment (DWTA).

本模块同时承载三层职责：
1) 现有静态求解路径使用的 ``DWTABenchmarkData``（保持向后兼容）；
2) scripted-waves 事件 schema（用于配置与运行时触发）；
3) 可变 ``DWTAEnvironment``（支持在固定画布上应用动态事件）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


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


@dataclass(frozen=True, slots=True)
class DWTAWaveEvent:
    """Scripted runtime event descriptor for dynamic DWTA experiments.

    Notes:
        - ``event_type`` is the primary dispatch key for runtime handlers.
        - ``payload`` stores type-specific parameters in a configuration-driven form.
        - ``target_damage_scale`` and ``compatibility_override`` are retained as
          backward-compatible legacy fields from earlier scripted schema versions.
    """

    wave_id: str
    trigger_generation: int
    event_type: str = "legacy_target_damage_scale"
    payload: dict[str, Any] = field(default_factory=dict)
    target_damage_scale: float | None = None
    compatibility_override: list[list[int]] | None = None
    note: str | None = None


@dataclass(slots=True)
class DWTAScenarioScript:
    """DWTA 波次脚本容器（用于 scripted_waves 模式）."""

    waves: list[DWTAWaveEvent]


@dataclass(slots=True)
class DWTAEnvironment:
    """Mutable DWTA environment with fixed canvas + active-mask semantics.

    Design:
        - Weapon/target list lengths are expected to stay constant during runtime
          when scripted waves are enabled.
        - Events mutate activity/attributes in-place via replacement; they never
          resize the chromosome canvas.
        - ``state_epoch`` is incremented on every applied event so that live-cache
          users can detect stale snapshots and refresh deterministically.
    """

    base_data: DWTABenchmarkData
    scenario_mode: str = "static"
    max_weapons: int | None = None
    max_targets: int | None = None
    script: DWTAScenarioScript | None = None
    munitions: list[MunitionType] = field(default_factory=list)
    weapons: list[Weapon] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)
    state_epoch: int = 0

    def apply_wave_event(self, event: DWTAWaveEvent) -> dict[str, Any]:
        """Apply one scripted event and return a structured update summary."""
        before_epoch = self.state_epoch
        event_type = event.event_type
        result: dict[str, Any]

        if event_type == "activate_targets":
            result = self._handle_activate_targets(event.payload)
        elif event_type == "inject_targets":
            result = self._handle_inject_targets(event.payload)
        elif event_type == "disable_weapons":
            result = self._handle_disable_weapons(event.payload)
        elif event_type == "ammo_delta":
            result = self._handle_ammo_delta(event.payload)
        elif event_type == "target_priority_update":
            result = self._handle_target_priority_update(event.payload)
        elif event_type == "time_window_update":
            result = self._handle_time_window_update(event.payload)
        elif event_type == "legacy_target_damage_scale":
            scale = float(event.target_damage_scale or 1.0)
            result = self._handle_target_priority_update({"priority_scale": scale, "target_ids": None})
            if event.compatibility_override is not None:
                result["compatibility_override_rows"] = len(event.compatibility_override)
        else:
            raise ValueError(f"unsupported DWTA event_type '{event_type}'")

        self.state_epoch += 1
        return {
            "wave_id": event.wave_id,
            "trigger_generation": event.trigger_generation,
            "event_type": event_type,
            "note": event.note,
            "state_epoch_before": before_epoch,
            "state_epoch_after": self.state_epoch,
            "updates": result,
        }

    def _weapon_index(self, weapon_id: str) -> int:
        for idx, weapon in enumerate(self.weapons):
            if weapon.id == weapon_id:
                return idx
        raise ValueError(f"unknown weapon id '{weapon_id}'")

    def _target_index(self, target_id: str) -> int:
        for idx, target in enumerate(self.targets):
            if target.id == target_id:
                return idx
        raise ValueError(f"unknown target id '{target_id}'")

    def _handle_activate_targets(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Activate existing targets by id on the fixed target canvas."""
        activated: list[str] = []
        for item in payload.get("targets", []):
            target_id = str(item["id"])
            idx = self._target_index(target_id)
            current = self.targets[idx]
            required_damage = float(item.get("required_damage", max(current.required_damage, 1.0)))
            time_window_raw = item.get("time_window", current.time_window)
            time_window = (float(time_window_raw[0]), float(time_window_raw[1]))
            self.targets[idx] = replace(current, required_damage=max(0.0, required_damage), time_window=time_window)
            activated.append(target_id)
        return {"activated_target_ids": activated}

    def _handle_inject_targets(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Inject new targets into inactive slots without resizing the canvas."""
        injected: list[str] = []
        for item in payload.get("targets", []):
            target_id = str(item["id"])
            existing_idx = next((i for i, tgt in enumerate(self.targets) if tgt.id == target_id), None)
            if existing_idx is None:
                existing_idx = next((i for i, tgt in enumerate(self.targets) if tgt.required_damage <= 0.0), None)
            if existing_idx is None:
                raise ValueError("inject_targets requires at least one inactive target slot")
            base = self.targets[existing_idx]
            window_raw = item.get("time_window", base.time_window)
            self.targets[existing_idx] = Target(
                id=target_id,
                x=float(item.get("x", base.x)),
                y=float(item.get("y", base.y)),
                required_damage=max(0.0, float(item["required_damage"])),
                time_window=(float(window_raw[0]), float(window_raw[1])),
            )
            injected.append(target_id)
        return {"injected_target_ids": injected}

    def _handle_disable_weapons(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Disable weapons by setting ammo to zero."""
        disabled: list[str] = []
        for weapon_id in payload.get("weapon_ids", []):
            idx = self._weapon_index(str(weapon_id))
            weapon = self.weapons[idx]
            self.weapons[idx] = replace(weapon, ammo_capacity=0)
            disabled.append(weapon.id)
        return {"disabled_weapon_ids": disabled}

    def _handle_ammo_delta(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply signed ammo deltas per weapon while clamping to non-negative."""
        updates: dict[str, int] = {}
        deltas = payload.get("deltas", {})
        for weapon_id, delta in deltas.items():
            idx = self._weapon_index(str(weapon_id))
            weapon = self.weapons[idx]
            new_capacity = max(0, weapon.ammo_capacity + int(delta))
            self.weapons[idx] = replace(weapon, ammo_capacity=new_capacity)
            updates[weapon.id] = new_capacity
        return {"ammo_capacities": updates}

    def _handle_target_priority_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Scale target required damage to emulate changing mission priorities."""
        scale = float(payload.get("priority_scale", 1.0))
        ids = payload.get("target_ids")
        scoped_ids = set(str(item) for item in ids) if ids is not None else None
        updates: dict[str, float] = {}
        for idx, target in enumerate(self.targets):
            if scoped_ids is not None and target.id not in scoped_ids:
                continue
            new_damage = max(0.0, float(target.required_damage) * scale)
            self.targets[idx] = replace(target, required_damage=new_damage)
            updates[target.id] = new_damage
        return {"required_damage": updates, "priority_scale": scale}

    def _handle_time_window_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update target exposure windows by id."""
        updates: dict[str, tuple[float, float]] = {}
        window_updates = payload.get("time_windows", {})
        for target_id, window_raw in window_updates.items():
            idx = self._target_index(str(target_id))
            target = self.targets[idx]
            window = (float(window_raw[0]), float(window_raw[1]))
            self.targets[idx] = replace(target, time_window=window)
            updates[target.id] = window
        return {"time_windows": updates}
