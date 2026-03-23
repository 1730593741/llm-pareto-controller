"""Checks 必需的 ablation 矩阵 switches 为 存在 与 minimally 格式正确."""

from __future__ import annotations

from experiments.ablations.matrix_runner import _ablation_specs


def test_required_ablation_switches_present() -> None:
    specs = dict(_ablation_specs((1, 3, 5, 10), (5, 20, 50)))

    assert "no_pareto_state_deep_features" in specs
    assert "no_experience_pool" in specs
    assert "binary_state_machine" in specs
    assert "four_state_machine" in specs
    assert "pc_pm_only" in specs
    assert "extended_action_space" in specs

    assert specs["binary_state_machine"]["controller"]["feasible_ratio_low"] < 0
    assert specs["binary_state_machine"]["controller"]["diversity_low"] < 0

    for tau in (1, 3, 5, 10):
        assert f"tau_{tau}" in specs
    for window in (5, 20, 50):
        assert f"memory_window_{window}" in specs
