"""Run the M4 rule-based closed-loop NSGA-II workflow."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from controller.closed_loop import ClosedLoopRunner, RuleBasedController, RuleControllerConfig
from infra.storage import JsonlLogger
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from sensing.pareto_state import ParetoStateSensor


class ProblemConfig(BaseModel):
    """Task-assignment problem specification for MVP experiments."""

    n_tasks: int
    n_resources: int
    cost_matrix: list[list[float]]
    task_loads: list[float]
    capacities: list[float]


class ExperimentConfig(BaseModel):
    """Top-level experiment config parsed from YAML."""

    optimizer: NSGA2Config
    controller: RuleControllerConfig
    problem: ProblemConfig
    log_path: str = Field(default="runs/m4/events.jsonl")


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
    """Run M4 closed loop and print final summary."""
    config = load_config(config_path)

    solver = build_solver(config.problem, config.optimizer)
    sensor = ParetoStateSensor()
    controller = RuleBasedController(config.controller)
    logger = JsonlLogger(config.log_path)

    runner = ClosedLoopRunner(solver=solver, sensor=sensor, controller=controller, logger=logger)
    states = runner.run(generations=config.optimizer.generations)

    final = states[-1]
    print(
        "M4 run complete | "
        f"generation={final.generation} hv={final.hv:.4f} "
        f"mutation_prob={solver.config.mutation_prob:.3f} "
        f"crossover_prob={solver.config.crossover_prob:.3f} "
        f"log_path={config.log_path}"
    )


if __name__ == "__main__":
    main()
