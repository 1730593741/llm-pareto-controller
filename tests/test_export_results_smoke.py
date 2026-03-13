"""Smoke tests for result export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.export_results import collect_rows, export_results


def _write_summary(path: Path, method: str, benchmark: str, seed: int, hv: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "benchmark": benchmark,
        "seed": seed,
        "final_hv": hv,
        "final_igd_plus": 0.2,
        "final_spacing": 0.1,
        "final_spread": 0.3,
        "final_feasible_ratio": 1.0,
        "runtime_s": 0.5,
        "llm_overhead_s": 0.05,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_results_smoke(tmp_path: Path) -> None:
    _write_summary(tmp_path / "matched" / "small_complex" / "seed_1" / "rule_control" / "summary.json", "rule_control", "small_complex", 1, 1.2)
    _write_summary(tmp_path / "matched" / "small_complex" / "seed_2" / "mock_llm" / "summary.json", "mock_llm", "small_complex", 2, 1.3)

    rows = collect_rows(tmp_path)
    assert len(rows) == 2

    outputs = export_results(runs_root=tmp_path, output_dir=tmp_path / "exports")
    for path in outputs.values():
        assert Path(path).exists()
