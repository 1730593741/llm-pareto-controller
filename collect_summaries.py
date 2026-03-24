from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("experiments/runs/main_results")
DEFAULT_OUT_DIR = Path("experiments/exports/main_results")


FIELDS = [
    "method",
    "controller_mode",
    "benchmark",
    "seed",
    "source_config_path",
    "run_id",
    "generations",
    "final_generation",
    "final_hv",
    "best_hv",
    "best_generation",
    "hv_auc",
    "mean_hv",
    "final_feasible_ratio",
    "final_rank1_ratio",
    "final_igd",
    "final_igd_plus",
    "final_spacing",
    "final_spread",
    "runtime_s",
    "llm_overhead_s",
    "num_actions",
    "num_experiences",
    "events_path",
    "experiences_path",
    "config_snapshot_path",
    "generation_log_path",
    "action_log_path",
    "summary_path",
]


AGG_METRICS = [
    "final_hv",
    "best_hv",
    "hv_auc",
    "final_feasible_ratio",
    "final_rank1_ratio",
    "final_igd",
    "final_igd_plus",
    "final_spacing",
    "final_spread",
    "runtime_s",
    "llm_overhead_s",
    "num_actions",
    "num_experiences",
]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
        return None
    except (TypeError, ValueError):
        return None


def safe_get(data: dict[str, Any], key: str) -> Any:
    return data.get(key)


def discover_summary_files(runs_root: Path) -> list[Path]:
    return sorted(runs_root.rglob("summary.json"))


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def build_row(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}

    for field in FIELDS:
        if field == "summary_path":
            row[field] = str(summary_path)
        else:
            row[field] = safe_get(summary, field)

    # 兜底：如果 method 缺失，尝试用 experiment.name
    if not row.get("method"):
        experiment = summary.get("experiment", {})
        if isinstance(experiment, dict):
            row["method"] = experiment.get("name")

    # 兜底：如果 seed 缺失，尝试从 experiment.seed 取
    if row.get("seed") is None:
        experiment = summary.get("experiment", {})
        if isinstance(experiment, dict):
            row["seed"] = experiment.get("seed")

    return row


def write_rows_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_metric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    if len(values) == 1:
        return {
            "n": 1,
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
        }

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        method = str(row.get("method") or "unknown")
        grouped.setdefault(method, []).append(row)

    agg_rows: list[dict[str, Any]] = []

    for method, method_rows in sorted(grouped.items()):
        agg_row: dict[str, Any] = {
            "method": method,
            "runs": len(method_rows),
        }

        controller_modes = sorted({str(r.get("controller_mode")) for r in method_rows if r.get("controller_mode") is not None})
        benchmarks = sorted({str(r.get("benchmark")) for r in method_rows if r.get("benchmark") is not None})

        agg_row["controller_modes"] = ",".join(controller_modes)
        agg_row["benchmarks"] = ",".join(benchmarks)

        for metric in AGG_METRICS:
            values = [to_float(r.get(metric)) for r in method_rows]
            numeric_values = [v for v in values if v is not None]
            stats = summarize_metric(numeric_values)

            agg_row[f"{metric}_n"] = stats["n"]
            agg_row[f"{metric}_mean"] = stats["mean"]
            agg_row[f"{metric}_std"] = stats["std"]
            agg_row[f"{metric}_min"] = stats["min"]
            agg_row[f"{metric}_max"] = stats["max"]

        agg_rows.append(agg_row)

    return agg_rows


def write_agg_csv(agg_rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "runs", "controller_modes", "benchmarks"]
    for metric in AGG_METRICS:
        fieldnames.extend(
            [
                f"{metric}_n",
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)


def main() -> int:
    runs_root = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_RUNS_ROOT
    out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_OUT_DIR

    if not runs_root.exists():
        print(f"[ERROR] Runs root does not exist: {runs_root}", file=sys.stderr)
        return 1

    summary_files = discover_summary_files(runs_root)
    if not summary_files:
        print(f"[ERROR] No summary.json files found under: {runs_root}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    bad_files: list[tuple[Path, str]] = []

    for summary_path in summary_files:
        try:
            summary = load_summary(summary_path)
            row = build_row(summary, summary_path)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            bad_files.append((summary_path, str(exc)))

    rows_csv = out_dir / "summary_rows.csv"
    agg_csv = out_dir / "summary_agg.csv"

    write_rows_csv(rows, rows_csv)
    agg_rows = aggregate_rows(rows)
    write_agg_csv(agg_rows, agg_csv)

    print(f"[OK] Found {len(summary_files)} summary files")
    print(f"[OK] Wrote rows CSV: {rows_csv}")
    print(f"[OK] Wrote aggregate CSV: {agg_csv}")

    if bad_files:
        print(f"[WARN] Failed to parse {len(bad_files)} files:", file=sys.stderr)
        for path, err in bad_files:
            print(f"  - {path}: {err}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())