"""Tests for M4 rule-based closed-loop runner."""

import pytest

from controller.closed_loop import ClosedLoopRunner, RuleBasedController, RuleControllerConfig
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from sensing.pareto_state import ParetoStateSensor


class InMemoryLogger:
    """Collect events for assertions in tests."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def log(self, event: dict[str, object]) -> None:
        self.events.append(event)


def test_closed_loop_runs_and_logs_actions() -> None:
    solver = NSGA2Solver(
        n_tasks=4,
        n_resources=2,
        cost_matrix=[
            [1.0, 2.0],
            [2.0, 1.1],
            [1.5, 1.8],
            [2.2, 1.0],
        ],
        task_loads=[1.0, 1.0, 1.0, 1.0],
        capacities=[2.0, 2.0],
        config=NSGA2Config(
            population_size=12,
            generations=6,
            crossover_prob=0.9,
            mutation_prob=0.1,
            seed=11,
        ),
    )
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    sensor = ParetoStateSensor()
    logger = InMemoryLogger()

    runner = ClosedLoopRunner(solver=solver, sensor=sensor, controller=controller, logger=logger)
    states = runner.run(generations=6)

    assert len(states) == 7
    assert states[-1].generation == 6
    assert any(event["event"] == "action" for event in logger.events)
    assert 0.0 <= solver.config.mutation_prob <= 1.0
    assert 0.0 <= solver.config.crossover_prob <= 1.0


def test_rule_controller_config_validates_control_interval() -> None:
    with pytest.raises(ValueError, match="control_interval"):
        RuleControllerConfig(control_interval=0)
