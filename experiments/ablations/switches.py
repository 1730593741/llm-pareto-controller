"""Config-level ablation switches 用于 quick M7 comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def apply_ablation_switches(
    *,
    config_path: str | Path,
    output_path: str | Path,
    switches: dict[str, bool],
) -> None:
    """Apply boolean ablation switches 与 write 一个 derived YAML 配置."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if switches.get("no_memory", False):
        config.setdefault("memory", {})
        config["memory"]["enabled"] = False
        config["memory"]["experience_log_path"] = None
        config.setdefault("controller", {})
        config["controller"]["memory_enabled"] = False
        config["controller"]["experience_log_path"] = None

    if switches.get("no_state_metric_x", False):
        config.setdefault("controller", {})
        config["controller"]["improvement_threshold"] = 1.0
        config.setdefault("controller", {}).setdefault("rule", {})
        config["controller"]["rule"]["improvement_threshold"] = 1.0

    if switches.get("fixed_control_interval", False):
        config.setdefault("controller", {})
        config["controller"]["control_interval"] = 99999
        config.setdefault("controller", {}).setdefault("rule", {})
        config["controller"]["rule"]["control_interval"] = 99999

    if switches.get("no_llm_chain", False):
        config.setdefault("controller_mode", {})
        config["controller_mode"]["mode"] = "rule"
        config.setdefault("controller", {})
        config["controller"]["mode"] = "rule"

    with Path(output_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
