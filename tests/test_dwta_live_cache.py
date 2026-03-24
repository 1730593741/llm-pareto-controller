"""Unit tests for DWTA live cache refresh semantics."""

from __future__ import annotations

from problems.dwta.live_cache import DWTALiveCache
from problems.dwta.model import MunitionType, Target, Weapon
from problems.dwta.scenario_builder import build_dynamic_scenario


def test_live_cache_refresh_after_environment_change() -> None:
    """Cache should rebuild matrices after invalidation and environment update."""
    scenario = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=10.0, flight_speed=2.0, lethality=3.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=3)],
        targets=[Target(id="t1", x=4.0, y=0.0, required_damage=2.0, time_window=(0.0, 3.0))],
        scenario_mode="scripted_waves",
    )
    cache = DWTALiveCache(scenario)

    first = cache.refresh()
    assert first.compatibility_mask.astype(int).tolist() == [[1]]
    assert first.flight_time_matrix.tolist()[0][0] == 2.0

    scenario.targets = [
        Target(
            id="t1",
            x=9.0,
            y=0.0,
            required_damage=2.0,
            time_window=(0.0, 3.0),
        )
    ]
    scenario.state_epoch += 1

    cache.invalidate()
    second = cache.refresh()
    assert second.compatibility_mask.astype(int).tolist() == [[0]]
    assert second.flight_time_matrix.tolist()[0][0] == 4.5
