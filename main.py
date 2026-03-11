"""Run the M4/M5 rule-based closed-loop NSGA-II workflow."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from controller.closed_loop import ClosedLoopRunner, RewardConfig, RuleBasedController, RuleControllerConfig
from infra.storage import ExperienceJsonlLogger, JsonlLogger
from memory.experience_pool import ExperiencePool
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from sensing.pareto_state import ParetoStateSensor


class ProblemConfig(BaseModel):
    """Task-assignment problem specification for MVP experiments."""

    n_tasks: int
    n_resources: int
    cost_matrix: list[list[float]]
    task_loads: list[float]
    capacities: list[float]


class MemoryConfig(BaseModel):
    """M5 memory and reward-related configuration."""

    enabled: bool = True
    memory_window: int = 100
    experience_log_path: str | None = Field(default="runs/m5/experiences.jsonl")
    reward_alpha: float = 1.0
    reward_beta: float = 0.1


class ExperimentConfig(BaseModel):
    """Top-level experiment config parsed from YAML."""

    optimizer: NSGA2Config
    controller: RuleControllerConfig
    problem: ProblemConfig
    log_path: str = Field(default="runs/m4/events.jsonl")
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


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
        config=optimizer,
    )


def main(config_path: str = "experiments/configs/default.yaml") -> None:
    """Run closed loop and print final summary."""
    config = load_config(config_path)

    solver = build_solver(config.problem, config.optimizer)
    sensor = ParetoStateSensor()
    controller = RuleBasedController(config.controller)
    logger = JsonlLogger(config.log_path)

    experience_pool = ExperiencePool(config.memory.memory_window) if config.memory.enabled else None
    experience_logger = (
        ExperienceJsonlLogger(config.memory.experience_log_path)
        if config.memory.enabled and config.memory.experience_log_path
        else None
    )
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
    states = runner.run(generations=config.optimizer.generations)

    final = states[-1]
    print(
        "Run complete | "
        f"generation={final.generation} hv={final.hv:.4f} "
        f"mutation_prob={solver.config.mutation_prob:.3f} "
        f"crossover_prob={solver.config.crossover_prob:.3f} "
        f"log_path={config.log_path}"
    )


if __name__ == "__main__":
    main()
