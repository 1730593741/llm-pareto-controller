"""用于测试 main configuration 兼容性."""

from pathlib import Path

import pytest

from main import load_config, parse_cli_args


def test_load_config_backwards_compatible_without_memory_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
problem:
  n_tasks: 2
  n_resources: 2
  cost_matrix:
    - [1.0, 2.0]
    - [2.0, 1.0]
  task_loads: [1.0, 1.0]
  capacities: [1.5, 1.5]
optimizer:
  population_size: 6
  generations: 2
  crossover_prob: 0.9
  mutation_prob: 0.1
  seed: 7
controller:
  control_interval: 1
  min_mutation_prob: 0.03
  max_mutation_prob: 0.75
  min_crossover_prob: 0.45
  max_crossover_prob: 0.98
  mutation_step: 0.05
  crossover_step: 0.04
  feasible_ratio_low: 0.6
  diversity_low: 0.12
  improvement_threshold: 0.0001
log_path: runs/m4/events.jsonl
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.memory.enabled is True
    assert cfg.memory.memory_window == 100
    assert cfg.memory.reward_alpha == 1.0
    assert cfg.memory.reward_beta == 0.1
    assert cfg.controller_mode.mode == "rule"
    assert cfg.controller_mode.experience_lookback == 5


def test_problem_config_validates_complex_constraint_shapes(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_complex.yaml"
    config_path.write_text(
        """
problem:
  n_tasks: 2
  n_resources: 2
  cost_matrix:
    - [1.0, 2.0]
    - [2.0, 1.0]
  task_loads: [1.0, 1.0]
  capacities: [1.5, 1.5]
  compatibility_matrix:
    - [1, 0, 1]
    - [1, 1, 1]
optimizer:
  population_size: 6
  generations: 2
  crossover_prob: 0.9
  mutation_prob: 0.1
controller:
  control_interval: 1
  min_mutation_prob: 0.03
  max_mutation_prob: 0.75
  min_crossover_prob: 0.45
  max_crossover_prob: 0.98
  mutation_step: 0.05
  crossover_step: 0.04
  feasible_ratio_low: 0.6
  diversity_low: 0.12
  improvement_threshold: 0.0001
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compatibility_matrix column count"):
        load_config(config_path)


def test_parse_cli_args_supports_config_flag_and_positional() -> None:
    assert parse_cli_args(["--config", "experiments/configs/small_complex_smoke.yaml"]) == "experiments/configs/small_complex_smoke.yaml"
    assert parse_cli_args(["experiments/configs/default.yaml"]) == "experiments/configs/default.yaml"


def test_load_dwta_precomputed_config() -> None:
    cfg = load_config("experiments/configs/dwta_medium.yaml")

    assert cfg.problem.problem_type == "dwta"
    assert cfg.problem.precomputed is not None
    assert cfg.problem.precomputed.ammo_capacities == [4, 4, 3, 3]


def test_load_dwta_scripted_config() -> None:
    cfg = load_config("experiments/configs/dwta_scripted_minimal.yaml")

    assert cfg.problem.problem_type == "dwta"
    assert cfg.problem.scenario_mode == "scripted_waves"
    assert cfg.problem.max_weapons == 4
    assert cfg.problem.max_targets == 6
    assert len(cfg.problem.waves) == 2
