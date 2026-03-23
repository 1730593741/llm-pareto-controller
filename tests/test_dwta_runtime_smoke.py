"""冒烟测试 用于 DWTA 运行时 end-到-end execution."""

from __future__ import annotations

from pathlib import Path

import yaml

from main import run_experiment


def test_dwta_small_smoke_runs_end_to_end(tmp_path: Path) -> None:
    source_path = Path("experiments/configs/dwta_small_smoke.yaml")
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload["logging"]["output_dir"] = str(tmp_path / "dwta_small_smoke")

    config_path = tmp_path / "dwta_small_smoke.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    summary = run_experiment(str(config_path))

    assert summary["benchmark"] == "dwta_small_smoke"
    assert summary["final_generation"] == 6
    assert 0.0 <= summary["final_feasible_ratio"] <= 1.0

    output_dir = Path(payload["logging"]["output_dir"])
    assert output_dir.exists()
    assert Path(summary["generation_log_path"]).exists()
    assert Path(summary["action_log_path"]).exists()
    assert Path(summary["summary_path"]).exists()
