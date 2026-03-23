"""轻量包装器 该 运行 predefined 基线 configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.ablations.switches import apply_ablation_switches
from main import run_experiment


def run_baseline_nsga2(config_path: str = "experiments/configs/baseline_nsga2.yaml") -> dict[str, Any]:
    """运行 standard NSGA-II 基线 (no adaptive 控制/记忆)."""
    return run_experiment(config_path)


def run_rule_control_baseline(config_path: str = "experiments/configs/rule_control.yaml") -> dict[str, Any]:
    """运行 规则-based 闭环 基线."""
    return run_experiment(config_path)


def run_no_memory_baseline(config_path: str = "experiments/configs/rule_control.yaml") -> dict[str, Any]:
    """运行 规则 基线，并带有 记忆 disabled 作为 一个 ablation 基线."""
    target = Path(config_path)
    patched = target.with_name(f"{target.stem}.no_memory.tmp.yaml")
    apply_ablation_switches(config_path=target, output_path=patched, switches={"no_memory": True})
    try:
        return run_experiment(str(patched))
    finally:
        if patched.exists():
            patched.unlink()
