"""Tests for M7 config modes and runtime outputs."""

from __future__ import annotations

import json
from pathlib import Path

from main import load_config, run_experiment


def test_default_config_runs_and_writes_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
experiment:
  name: test_default
problem:
  n_tasks: 4
  n_resources: 2
  cost_matrix:
    - [1.0, 2.0]
    - [2.0, 1.0]
    - [1.5, 1.1]
    - [1.2, 1.3]
  task_loads: [1.0, 1.0, 1.0, 1.0]
  capacities: [2.0, 2.0]
optimizer:
  population_size: 10
  generations: 3
  crossover_prob: 0.9
  mutation_prob: 0.1
  seed: 13
controller:
  control_interval: 2
  min_mutation_prob: 0.03
  max_mutation_prob: 0.75
  min_crossover_prob: 0.45
  max_crossover_prob: 0.98
  mutation_step: 0.05
  crossover_step: 0.04
  feasible_ratio_low: 0.6
  diversity_low: 0.12
  improvement_threshold: 0.0001
memory:
  enabled: true
  memory_window: 10
logging:
  output_dir: __OUT__
""".replace("__OUT__", str(tmp_path / "logs")),
        encoding="utf-8",
    )

    summary = run_experiment(str(config_path))
    assert summary["final_generation"] == 3

    logs_dir = tmp_path / "logs"
    assert (logs_dir / "events.jsonl").exists()
    assert (logs_dir / "config_snapshot.yaml").exists()
    assert (logs_dir / "summary.json").exists()
    assert (logs_dir / "generation_metrics.jsonl").exists()
    assert (logs_dir / "actions.jsonl").exists()
    assert (logs_dir / "experiences.jsonl").exists()

    payload = json.loads((logs_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["controller_mode"] == "rule"


def test_mode_switch_from_mock_config() -> None:
    config = load_config("experiments/configs/mock_llm.yaml")
    assert config.controller_mode.mode == "mock_llm"
    assert config.logging.output_dir.endswith("mock_llm")
