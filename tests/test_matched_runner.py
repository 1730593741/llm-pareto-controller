"""Smoke tests for matched baseline/rule/mock_llm experiment helper."""

from __future__ import annotations

from pathlib import Path

from experiments.baselines.matched_runner import run_matched_experiments


def test_run_matched_experiments_smoke(tmp_path: Path) -> None:
    results = run_matched_experiments(output_root=tmp_path / "matched", seed=33, generations=3)

    assert set(results.keys()) == {"baseline_nsga2", "rule_control", "mock_llm"}
    for mode, summary in results.items():
        assert summary["seed"] == 33
        assert summary["generations"] == 3
        assert summary["controller_mode"] in {"rule", "mock_llm"}
        assert Path(summary["summary_path"]).exists()
        assert str(tmp_path / "matched" / mode) in summary["summary_path"]
