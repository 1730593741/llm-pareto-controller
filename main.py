"""Run configurable closed-loop NSGA-II experiments."""

from __future__ import annotations

import hashlib
import json
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from eval.metrics import igd, igd_plus, spacing, spread
from eval.reference_front import (
    ReferenceFront,
    build_intra_run_reference_front,
    build_true_reference_front,
)

import yaml
from pydantic import BaseModel, Field, model_validator

from controller.closed_loop import (
    ClosedLoopRunner,
    LLMChainController,
    RewardConfig,
    RuleBasedController,
    RuleControllerConfig,
)
from controller.operator_space import OperatorParams
from infra.llm_client import LLMClient, LLMClientConfig
from infra.storage import ExperienceJsonlLogger, JsonlLogger
from llm.actuator import Actuator
from llm.analyst import Analyst
from llm.strategist import Strategist
from memory.experience_pool import ExperiencePool
from experiments.logging import split_event_stream
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from sensing.pareto_state import ParetoStateSensor


class ProblemConfig(BaseModel):
    """Task-assignment problem specification for MVP experiments."""

    n_tasks: int
    n_resources: int
    cost_matrix: list[list[float]]
    task_loads: list[float]
    capacities: list[float]
    task_time_windows: list[list[float]] | None = None
    resource_time_windows: list[list[float]] | None = None
    compatibility_matrix: list[list[int]] | None = None
    resource_stage_levels: list[int] | None = None
    stage_transitions: list[list[int]] | None = None

    @model_validator(mode="after")
    def _validate_shapes(self) -> "ProblemConfig":
        if self.task_time_windows is not None:
            if len(self.task_time_windows) != self.n_tasks:
                raise ValueError("task_time_windows length must equal n_tasks")
            for idx, window in enumerate(self.task_time_windows):
                if len(window) != 2:
                    raise ValueError(f"task_time_windows[{idx}] must contain [start, end]")
                if float(window[0]) > float(window[1]):
                    raise ValueError(f"task_time_windows[{idx}] start must be <= end")

        if self.resource_time_windows is not None:
            if len(self.resource_time_windows) != self.n_resources:
                raise ValueError("resource_time_windows length must equal n_resources")
            for idx, window in enumerate(self.resource_time_windows):
                if len(window) != 2:
                    raise ValueError(f"resource_time_windows[{idx}] must contain [start, end]")
                if float(window[0]) > float(window[1]):
                    raise ValueError(f"resource_time_windows[{idx}] start must be <= end")

        if (self.task_time_windows is None) ^ (self.resource_time_windows is None):
            raise ValueError("task_time_windows and resource_time_windows must be configured together")

        if self.compatibility_matrix is not None:
            if len(self.compatibility_matrix) != self.n_tasks:
                raise ValueError("compatibility_matrix row count must equal n_tasks")
            for row in self.compatibility_matrix:
                if len(row) != self.n_resources:
                    raise ValueError("compatibility_matrix column count must equal n_resources")
                if any(value not in (0, 1) for value in row):
                    raise ValueError("compatibility_matrix values must be 0 or 1")

        if self.resource_stage_levels is not None and len(self.resource_stage_levels) != self.n_resources:
            raise ValueError("resource_stage_levels length must equal n_resources")

        if self.stage_transitions is not None:
            for edge in self.stage_transitions:
                if len(edge) != 2:
                    raise ValueError("stage_transitions entries must be [predecessor_task, successor_task]")
                predecessor, successor = edge
                if not 0 <= predecessor < self.n_tasks or not 0 <= successor < self.n_tasks:
                    raise ValueError("stage_transitions task indices must be in [0, n_tasks)")

        if (self.resource_stage_levels is None) ^ (self.stage_transitions is None):
            raise ValueError("resource_stage_levels and stage_transitions must be configured together")

        return self


class MemoryConfig(BaseModel):
    """M5 memory and reward-related configuration."""

    enabled: bool = True
    memory_window: int = 100
    experience_log_path: str | None = None
    reward_alpha: float = 1.0
    reward_beta: float = 0.1


class ControllerModeConfig(BaseModel):
    """Controller mode switch and mode-specific options."""

    mode: Literal["rule", "mock_llm", "real_llm"] = "rule"
    experience_lookback: int = 5


class LLMRuntimeConfig(BaseModel):
    """Config schema for LLM runtime backend."""

    provider: str = "openai"
    model: str = "gpt-mock"
    timeout_s: float = 10.0
    max_retries: int = 2
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model_env: str = "OPENAI_MODEL"
    base_url: str = "https://api.openai.com/v1/chat/completions"
    fallback_mode: Literal["mock_llm", "hold"] = "mock_llm"


class LoggingConfig(BaseModel):
    """Output layout for reproducible experiment artifacts."""

    output_dir: str = "experiments/logs/default"
    events_file: str = "events.jsonl"
    experiences_file: str = "experiences.jsonl"
    summary_file: str = "summary.json"
    config_snapshot_file: str = "config_snapshot.yaml"
    generation_log_file: str = "generation_metrics.jsonl"
    action_log_file: str = "actions.jsonl"


class ExperimentMetaConfig(BaseModel):
    """Lightweight metadata persisted into summary output."""

    name: str = "default"
    seed: int | None = None
    method: str | None = None
    benchmark: str | None = None


class EvaluationConfig(BaseModel):
    """Evaluation settings for paper-level metrics and reference-front provenance."""

    reference_front_mode: Literal["auto", "true_front_file", "intra_run"] = "auto"
    true_pareto_front_path: str | None = None


class ExperimentConfig(BaseModel):
    """Top-level experiment config parsed from YAML."""

    optimizer: NSGA2Config
    controller: RuleControllerConfig
    controller_mode: ControllerModeConfig = Field(default_factory=ControllerModeConfig)
    llm: LLMRuntimeConfig = Field(default_factory=LLMRuntimeConfig)
    problem: ProblemConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentMetaConfig = Field(default_factory=ExperimentMetaConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    log_path: str | None = None


@dataclass(slots=True)
class RunArtifacts:
    """Resolved file paths for one concrete run."""

    output_dir: Path
    events_path: Path
    experiences_path: Path | None
    summary_path: Path
    config_snapshot_path: Path
    generation_log_path: Path
    action_log_path: Path


@dataclass(slots=True)
class RuntimeBundle:
    """Assembled runtime pieces used by main and baseline wrappers."""

    config: ExperimentConfig
    solver: NSGA2Solver
    runner: ClosedLoopRunner
    artifacts: RunArtifacts


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment config from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


def build_solver(problem: ProblemConfig, optimizer: NSGA2Config) -> NSGA2Solver:
    """Create NSGA-II solver from structured config."""
    return NSGA2Solver(
        n_tasks=problem.n_tasks,
        n_resources=problem.n_resources,
        cost_matrix=problem.cost_matrix,
        task_loads=problem.task_loads,
        capacities=problem.capacities,
        task_time_windows=problem.task_time_windows,
        resource_time_windows=problem.resource_time_windows,
        compatibility_matrix=problem.compatibility_matrix,
        resource_stage_levels=problem.resource_stage_levels,
        stage_transitions=problem.stage_transitions,
        config=optimizer,
    )


def build_controller(config: ExperimentConfig) -> RuleBasedController | LLMChainController:
    """Build rule or LLM controller by mode without changing runner API."""
    mode = config.controller_mode.mode
    if mode == "rule":
        return RuleBasedController(config.controller)

    llm_client = LLMClient(
        LLMClientConfig(
            mode=mode,
            provider=config.llm.provider,
            model=config.llm.model,
            timeout_s=config.llm.timeout_s,
            max_retries=config.llm.max_retries,
            api_key_env=config.llm.api_key_env,
            base_url_env=config.llm.base_url_env,
            model_env=config.llm.model_env,
            base_url=config.llm.base_url,
            fallback_mode=config.llm.fallback_mode,
        )
    )
    analyst = Analyst(llm_client)
    strategist = Strategist(llm_client)
    actuator = Actuator(
        llm_client,
        min_mutation_prob=config.controller.min_mutation_prob,
        max_mutation_prob=config.controller.max_mutation_prob,
        min_crossover_prob=config.controller.min_crossover_prob,
        max_crossover_prob=config.controller.max_crossover_prob,
        min_eta_c=config.controller.min_eta_c,
        max_eta_c=config.controller.max_eta_c,
        min_eta_m=config.controller.min_eta_m,
        max_eta_m=config.controller.max_eta_m,
        min_repair_prob=config.controller.min_repair_prob,
        max_repair_prob=config.controller.max_repair_prob,
        min_local_search_prob=config.controller.min_local_search_prob,
        max_local_search_prob=config.controller.max_local_search_prob,
    )
    return LLMChainController(
        control_interval=config.controller.control_interval,
        experience_lookback=config.controller_mode.experience_lookback,
        analyst=analyst,
        strategist=strategist,
        actuator=actuator,
    )


def resolve_artifacts(config: ExperimentConfig) -> RunArtifacts:
    """Resolve and create output paths for one run."""
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = Path(config.log_path) if config.log_path else output_dir / config.logging.events_file
    experiences_path: Path | None = None
    if config.memory.enabled:
        if config.memory.experience_log_path:
            experiences_path = Path(config.memory.experience_log_path)
        else:
            experiences_path = output_dir / config.logging.experiences_file

    return RunArtifacts(
        output_dir=output_dir,
        events_path=events_path,
        experiences_path=experiences_path,
        summary_path=output_dir / config.logging.summary_file,
        config_snapshot_path=output_dir / config.logging.config_snapshot_file,
        generation_log_path=output_dir / config.logging.generation_log_file,
        action_log_path=output_dir / config.logging.action_log_file,
    )


def build_runtime(config: ExperimentConfig) -> RuntimeBundle:
    """Assemble solver, controller, runner, and output artifacts."""
    artifacts = resolve_artifacts(config)
    solver = build_solver(config.problem, config.optimizer)
    sensor = ParetoStateSensor()
    controller = build_controller(config)
    logger = JsonlLogger(artifacts.events_path)

    experience_pool = ExperiencePool(config.memory.memory_window) if config.memory.enabled else None
    experience_logger = ExperienceJsonlLogger(artifacts.experiences_path) if artifacts.experiences_path else None
    reward_config = RewardConfig(alpha=config.memory.reward_alpha, beta=config.memory.reward_beta)

    runner = ClosedLoopRunner(
        solver=solver,
        sensor=sensor,
        controller=controller,
        logger=logger,
        experience_pool=experience_pool,
        experience_logger=experience_logger,
        reward_config=reward_config,
    )
    return RuntimeBundle(config=config, solver=solver, runner=runner, artifacts=artifacts)




def _config_identity(config: ExperimentConfig, config_path: str | Path) -> tuple[str, str]:
    """Build deterministic fingerprint and run id from config snapshot inputs."""
    material = {
        "source_config_path": str(config_path),
        "resolved_config": config.model_dump(mode="json"),
    }
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    run_id = f"run-{fingerprint[:12]}"
    return fingerprint, run_id


def _write_config_snapshot(path: Path, config: ExperimentConfig, config_path: str | Path, *, config_fingerprint: str, run_id: str) -> None:
    payload = {
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "source_config_path": str(config_path),
        "resolved_config": config.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)






def _reset_run_logs(artifacts: RunArtifacts) -> None:
    """Reset append-only logs so each run keeps isolated metrics/actions counts."""
    for path in [
        artifacts.events_path,
        artifacts.generation_log_path,
        artifacts.action_log_path,
        artifacts.experiences_path,
    ]:
        if path is not None and path.exists():
            path.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _count_control_states(action_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in action_events:
        state = event.get("control_state")
        if not isinstance(state, str):
            continue
        counts[state] = counts.get(state, 0) + 1
    return counts


def _load_true_reference_front(path: Path) -> list[tuple[float, ...]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("true pareto front file must be a JSON list of points")
    points: list[tuple[float, ...]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("each true front point must be a list")
        points.append(tuple(float(v) for v in row))
    return points


def _resolve_reference_front(config: ExperimentConfig, generation_events: list[dict[str, Any]]) -> ReferenceFront:
    mode = config.evaluation.reference_front_mode
    true_path = config.evaluation.true_pareto_front_path
    if mode in {"auto", "true_front_file"} and true_path:
        path = Path(true_path)
        return build_true_reference_front(_load_true_reference_front(path), name=str(path))
    return build_intra_run_reference_front(generation_events)


def run_experiment(config_path: str = "experiments/configs/default.yaml") -> dict[str, Any]:
    """Run one experiment and persist snapshot + summary outputs."""
    config = load_config(config_path)
    runtime = build_runtime(config)
    _reset_run_logs(runtime.artifacts)
    config_fingerprint, run_id = _config_identity(config, config_path)
    _write_config_snapshot(
        runtime.artifacts.config_snapshot_path,
        config,
        config_path,
        config_fingerprint=config_fingerprint,
        run_id=run_id,
    )

    start_ts = time.perf_counter()
    states = runtime.runner.run(generations=config.optimizer.generations)
    runtime_s = time.perf_counter() - start_ts
    split_event_stream(
        events_path=runtime.artifacts.events_path,
        state_log_path=runtime.artifacts.generation_log_path,
        action_log_path=runtime.artifacts.action_log_path,
    )

    final = states[-1]
    best_state = max(states, key=lambda state: state.hv)
    hv_auc = sum(state.hv for state in states) / len(states) if states else 0.0

    generation_events = _read_jsonl(runtime.artifacts.generation_log_path)
    action_events = _read_jsonl(runtime.artifacts.action_log_path)
    final_generation_event = generation_events[-1] if generation_events else {}
    num_experiences = 0
    if runtime.artifacts.experiences_path and runtime.artifacts.experiences_path.exists():
        num_experiences = len(_read_jsonl(runtime.artifacts.experiences_path))

    reference_front = _resolve_reference_front(config, generation_events)
    final_front_raw = final_generation_event.get("rank1_objectives", final.rank1_objectives)
    final_front = [tuple(float(v) for v in point) for point in final_front_raw] if isinstance(final_front_raw, list) else []

    final_igd = igd(final_front, reference_front.points)
    final_igd_plus = igd_plus(final_front, reference_front.points)
    final_spacing = spacing(final_front)
    final_spread = spread(final_front, reference_front.points) if final_front and reference_front.points else 0.0

    llm_overhead_s = sum(float(event.get("decision_runtime_s", 0.0)) for event in action_events)

    summary = {
        "experiment": config.experiment.model_dump(mode="json"),
        "controller_mode": config.controller_mode.mode,
        "method": config.experiment.method or config.experiment.name,
        "benchmark": config.experiment.benchmark or "unknown",
        "seed": config.experiment.seed if config.experiment.seed is not None else config.optimizer.seed,
        "source_config_path": str(config_path),
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "generations": config.optimizer.generations,
        "final_generation": final.generation,
        "final_hv": final.hv,
        "best_hv": best_state.hv,
        "best_generation": best_state.generation,
        "hv_auc": hv_auc,
        "mean_hv": hv_auc,
        "final_feasible_ratio": final.feasible_ratio,
        "final_rank1_ratio": final.rank1_ratio,
        "final_igd": final_igd,
        "final_igd_plus": final_igd_plus,
        "final_spacing": final_spacing,
        "final_spread": final_spread,
        "reference_front": {
            "source": reference_front.source,
            "details": reference_front.details,
            "num_points": len(reference_front.points),
        },
        "final_mutation_prob": final_generation_event.get("mutation_prob", runtime.solver.config.mutation_prob),
        "final_crossover_prob": final_generation_event.get("crossover_prob", runtime.solver.config.crossover_prob),
        "final_operator_params": (
            final_generation_event.get("operator_params")
            or runtime.solver.get_operator_params().to_dict()
        ),
        "final_effective_params": (
            OperatorParams(**(final_generation_event.get("operator_params") or runtime.solver.get_operator_params().to_dict()))
            .active_params(runtime.solver.get_operator_capabilities())
        ),
        "operator_capabilities": runtime.solver.get_operator_capabilities().to_dict(),
        "events_path": str(runtime.artifacts.events_path),
        "experiences_path": str(runtime.artifacts.experiences_path) if runtime.artifacts.experiences_path else None,
        "num_actions": len(action_events),
        "runtime_s": runtime_s,
        "llm_overhead_s": llm_overhead_s,
        "control_state_counts": _count_control_states(action_events),
        "num_experiences": num_experiences,
        "config_snapshot_path": str(runtime.artifacts.config_snapshot_path),
        "generation_log_path": str(runtime.artifacts.generation_log_path),
        "action_log_path": str(runtime.artifacts.action_log_path),
    }
    _write_summary(runtime.artifacts.summary_path, summary)
    summary["summary_path"] = str(runtime.artifacts.summary_path)
    return summary


def main(config_path: str = "experiments/configs/default.yaml") -> None:
    """Run closed loop and print final summary."""
    summary = run_experiment(config_path)
    print(
        "Run complete | "
        f"generation={summary['final_generation']} hv={summary['final_hv']:.4f} "
        f"mutation_prob={summary['final_mutation_prob']:.3f} "
        f"crossover_prob={summary['final_crossover_prob']:.3f} "
        f"log_path={summary['events_path']}"
    )


def parse_cli_args(argv: list[str] | None = None) -> str:
    """Parse CLI args and return resolved config path."""
    parser = argparse.ArgumentParser(description="Run closed-loop NSGA-II experiment")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to YAML experiment config")
    parser.add_argument("--config", dest="config_flag", default=None, help="Path to YAML experiment config")
    args = parser.parse_args(argv)
    return args.config_flag or args.config_path or "experiments/configs/default.yaml"


if __name__ == "__main__":
    main(parse_cli_args())
