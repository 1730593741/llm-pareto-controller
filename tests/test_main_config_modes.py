"""用于测试 M7 配置 模式 与 运行时 输出."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

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


def test_summary_fields_semantics(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg_summary.yaml"
    config_path.write_text(
        """
experiment:
  name: test_summary
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
  generations: 4
  crossover_prob: 0.9
  mutation_prob: 0.1
  seed: 21
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
""".replace("__OUT__", str(tmp_path / "logs_summary")),
        encoding="utf-8",
    )

    summary = run_experiment(str(config_path))
    summary_payload = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))

    for field in [
        "best_hv",
        "best_generation",
        "hv_auc",
        "mean_hv",
        "num_actions",
        "num_experiences",
        "seed",
        "source_config_path",
        "controller_mode",
        "run_id",
        "config_fingerprint",
        "final_operator_params",
        "final_effective_params",
        "operator_capabilities",
    ]:
        assert field in summary_payload

    assert summary_payload["final_generation"] == 4
    assert summary_payload["best_hv"] >= summary_payload["final_hv"]

    generation_lines = Path(summary_payload["generation_log_path"]).read_text(encoding="utf-8").strip().splitlines()
    final_generation_payload = json.loads(generation_lines[-1])
    assert summary_payload["final_mutation_prob"] == final_generation_payload["mutation_prob"]
    assert summary_payload["final_crossover_prob"] == final_generation_payload["crossover_prob"]

    action_count = len(Path(summary_payload["action_log_path"]).read_text(encoding="utf-8").strip().splitlines())
    assert summary_payload["num_actions"] == action_count

    for unsupported in ["eta_c", "eta_m", "local_search_prob"]:
        assert unsupported not in summary_payload["final_effective_params"]

    experience_count = len(Path(summary_payload["experiences_path"]).read_text(encoding="utf-8").strip().splitlines())
    assert summary_payload["num_experiences"] == experience_count

    with Path(summary_payload["config_snapshot_path"]).open("r", encoding="utf-8") as f:
        snapshot_payload = yaml.safe_load(f)

    assert snapshot_payload["run_id"] == summary_payload["run_id"]
    assert snapshot_payload["config_fingerprint"] == summary_payload["config_fingerprint"]


def test_repeated_run_resets_append_only_logs(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg_repeat.yaml"
    config_path.write_text(
        """
experiment:
  name: test_repeat
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
  generations: 4
  crossover_prob: 0.9
  mutation_prob: 0.1
  seed: 21
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
""".replace("__OUT__", str(tmp_path / "logs_repeat")),
        encoding="utf-8",
    )

    summary1 = run_experiment(str(config_path))
    summary2 = run_experiment(str(config_path))

    assert summary1["num_actions"] == summary2["num_actions"]
    assert summary1["num_experiences"] == summary2["num_experiences"]
