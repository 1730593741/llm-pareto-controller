"""验证 exported file structure 用于 paper-ready artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.export_results import export_results


def test_export_file_structure(tmp_path: Path) -> None:
    run_dir = tmp_path / "matched" / "small_complex" / "seed_1" / "baseline_nsga2"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": "baseline_nsga2",
                "benchmark": "small_complex",
                "seed": 1,
                "final_hv": 1.0,
                "final_igd_plus": 0.5,
                "final_spacing": 0.2,
                "final_spread": 0.4,
                "final_feasible_ratio": 0.9,
                "runtime_s": 1.2,
                "llm_overhead_s": 0.0,
            }
        ),
        encoding="utf-8",
    )

    outputs = export_results(runs_root=tmp_path, output_dir=tmp_path / "exports")

    raw_csv = Path(outputs["aggregated_runs_csv"])
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader)
    assert first["method"] == "baseline_nsga2"
    assert first["benchmark"] == "small_complex"

    paper_csv = Path(outputs["paper_table_method_csv"])
    assert "hv_mean" in paper_csv.read_text(encoding="utf-8")
