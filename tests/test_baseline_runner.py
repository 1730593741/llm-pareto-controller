"""Tests for M7 baseline runners."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.baselines.runner import run_baseline_nsga2, run_no_memory_baseline


def test_baseline_nsga2_runner_executes(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        """
experiment:
  name: baseline_test
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
  generations: 2
  crossover_prob: 0.9
  mutation_prob: 0.1
  seed: 7
controller:
  control_interval: 99999
  min_mutation_prob: 0.1
  max_mutation_prob: 0.1
  min_crossover_prob: 0.9
  max_crossover_prob: 0.9
  mutation_step: 0.0
  crossover_step: 0.0
  feasible_ratio_low: 0.0
  diversity_low: 0.0
  improvement_threshold: 1.0
memory:
  enabled: false
logging:
  output_dir: __OUT__
""".replace("__OUT__", str(tmp_path / "baseline_logs")),
        encoding="utf-8",
    )

    summary = run_baseline_nsga2(str(config_path))
    assert summary["final_generation"] == 2
    assert summary["experiences_path"] is None
    assert (tmp_path / "baseline_logs" / "generation_metrics.jsonl").exists()
    assert (tmp_path / "baseline_logs" / "actions.jsonl").exists()


def test_no_memory_baseline_uses_ablation_switch(tmp_path: Path) -> None:
    config_path = tmp_path / "rule.yaml"
    payload = yaml.safe_load(Path("experiments/configs/rule_control.yaml").read_text(encoding="utf-8"))
    payload["logging"]["output_dir"] = str(tmp_path / "no_memory_logs")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    summary = run_no_memory_baseline(str(config_path))
    assert summary["controller_mode"] == "rule"
    assert summary["experiences_path"] is None
