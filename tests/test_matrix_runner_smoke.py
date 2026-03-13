"""Smoke test for preset matrix runner."""

from __future__ import annotations

from pathlib import Path

from experiments.run_matrix import run_matrix


def test_run_matrix_toy_smoke(tmp_path: Path) -> None:
    payload = run_matrix(preset="toy", output_root=tmp_path / "runs", include_ablation=False)

    manifest = Path(payload["manifest_path"])
    assert manifest.exists()
    assert payload["matched"]["benchmarks"] == ["small_complex_smoke"]
    assert payload["matched"]["methods"] == ["baseline_nsga2", "rule_control", "mock_llm"]
