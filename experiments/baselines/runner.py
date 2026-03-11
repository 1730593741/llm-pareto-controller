"""Thin wrappers that run predefined baseline configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.ablations.switches import apply_ablation_switches
from main import run_experiment


def run_baseline_nsga2(config_path: str = "experiments/configs/baseline_nsga2.yaml") -> dict[str, Any]:
    """Run standard NSGA-II baseline (no adaptive control/memory)."""
    return run_experiment(config_path)


def run_rule_control_baseline(config_path: str = "experiments/configs/rule_control.yaml") -> dict[str, Any]:
    """Run rule-based closed-loop baseline."""
    return run_experiment(config_path)


def run_no_memory_baseline(config_path: str = "experiments/configs/rule_control.yaml") -> dict[str, Any]:
    """Run rule baseline with memory disabled as an ablation baseline."""
    target = Path(config_path)
    patched = target.with_name(f"{target.stem}.no_memory.tmp.yaml")
    apply_ablation_switches(config_path=target, output_path=patched, switches={"no_memory": True})
    try:
        return run_experiment(str(patched))
    finally:
        if patched.exists():
            patched.unlink()
