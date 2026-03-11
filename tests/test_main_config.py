"""Tests for main configuration compatibility."""

from pathlib import Path

from main import load_config


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
