"""Helpers to run matched baseline/rule/mock-LLM experiments with aligned settings."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from main import run_experiment

_MATCHED_CONFIGS: dict[str, str] = {
    "baseline_nsga2": "experiments/configs/baseline_nsga2.yaml",
    "rule_control": "experiments/configs/rule_control.yaml",
    "mock_llm": "experiments/configs/mock_llm.yaml",
}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def run_matched_experiments(
    *,
    output_root: str | Path,
    seed: int,
    generations: int,
) -> dict[str, dict[str, Any]]:
    """Run 3 modes with matched seed/generation/problem config and return their summaries."""
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    base_payload = _load_yaml(_MATCHED_CONFIGS["baseline_nsga2"])
    results: dict[str, dict[str, Any]] = {}

    for mode_name, config_path in _MATCHED_CONFIGS.items():
        payload = _load_yaml(config_path)
        matched_payload = copy.deepcopy(payload)
        matched_payload["problem"] = copy.deepcopy(base_payload["problem"])
        matched_payload.setdefault("optimizer", {})["seed"] = seed
        matched_payload["optimizer"]["generations"] = generations
        matched_payload.setdefault("experiment", {})["seed"] = seed
        matched_payload["experiment"]["name"] = f"matched_{mode_name}"

        log_dir = output_root_path / mode_name
        matched_payload.setdefault("logging", {})["output_dir"] = str(log_dir)

        config_tmp_path = output_root_path / f"{mode_name}.matched.tmp.yaml"
        _dump_yaml(config_tmp_path, matched_payload)
        try:
            results[mode_name] = run_experiment(str(config_tmp_path))
        finally:
            if config_tmp_path.exists():
                config_tmp_path.unlink()

    return results


def run_matched_seed_sweep(
    *,
    output_root: str | Path,
    seeds: list[int],
    generations: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Run matched 3-way comparisons for multiple seeds.

    Artifacts are stored under ``<output_root>/seed_<seed>/<method>/...``.
    """
    root = Path(output_root)
    results: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        results[seed] = run_matched_experiments(
            output_root=root / f"seed_{seed}",
            seed=seed,
            generations=generations,
        )
    return results
