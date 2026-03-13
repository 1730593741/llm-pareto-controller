"""Reference front construction helpers for reproducible metric evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sensing.hypervolume import ObjectivePoint


def dominates(a: ObjectivePoint, b: ObjectivePoint) -> bool:
    """Return True when point ``a`` dominates point ``b`` (minimization)."""
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))


def nondominated(points: list[ObjectivePoint]) -> list[ObjectivePoint]:
    """Filter non-dominated points with stable deterministic order."""
    result: list[ObjectivePoint] = []
    for idx, point in enumerate(points):
        is_dominated = False
        for j, other in enumerate(points):
            if idx == j:
                continue
            if dominates(other, point):
                is_dominated = True
                break
        if not is_dominated:
            result.append(point)
    return result


@dataclass(frozen=True, slots=True)
class ReferenceFront:
    """Reference front and its explicit provenance metadata."""

    points: list[ObjectivePoint]
    source: str
    details: dict[str, Any]


def build_true_reference_front(points: list[ObjectivePoint], *, name: str) -> ReferenceFront:
    """Build reference from known true Pareto front points."""
    return ReferenceFront(
        points=nondominated(points),
        source="true_pareto_front",
        details={"name": name, "num_points": len(points)},
    )


def read_final_front_from_generation_log(path: Path) -> list[ObjectivePoint]:
    """Read final non-dominated front objectives from generation_metrics JSONL.

    This is explicit and reproducible because points are recorded in the run log,
    then declared as sourced from that exact file path.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return []
    final = rows[-1]
    raw_front = final.get("rank1_objectives")
    if not isinstance(raw_front, list):
        return []
    parsed: list[ObjectivePoint] = []
    for point in raw_front:
        if isinstance(point, (list, tuple)):
            parsed.append(tuple(float(v) for v in point))
    return nondominated(parsed)


def build_empirical_reference_front(method_to_generation_logs: dict[str, list[Path]]) -> ReferenceFront:
    """Build empirical reference front from matched runs across compared methods."""
    aggregated: list[ObjectivePoint] = []
    consumed: dict[str, list[str]] = {}
    for method, logs in method_to_generation_logs.items():
        consumed[method] = []
        for log_path in logs:
            consumed[method].append(str(log_path))
            aggregated.extend(read_final_front_from_generation_log(log_path))

    return ReferenceFront(
        points=nondominated(aggregated),
        source="empirical_matched_runs",
        details={
            "methods": sorted(method_to_generation_logs.keys()),
            "consumed_generation_logs": consumed,
            "num_aggregated_points": len(aggregated),
        },
    )


def build_intra_run_reference_front(generation_events: list[dict[str, Any]]) -> ReferenceFront:
    """Build empirical reference from rank-1 fronts across a single run timeline."""
    points: list[ObjectivePoint] = []
    for event in generation_events:
        raw = event.get("rank1_objectives")
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, (list, tuple)):
                points.append(tuple(float(v) for v in item))
    return ReferenceFront(
        points=nondominated(points),
        source="intra_run_union_front",
        details={"num_events": len(generation_events), "num_aggregated_points": len(points)},
    )
