"""Tests for M4/M5 rule-based closed-loop runner."""

import pytest

from controller.closed_loop import ClosedLoopRunner, RewardConfig, RuleBasedController, RuleControllerConfig
from controller.control_semantics import ControlState
from memory.experience_pool import ExperiencePool
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from sensing.pareto_state import ParetoStateSensor


class InMemoryLogger:
    """Collect events for assertions in tests."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def log(self, event: dict[str, object]) -> None:
        self.events.append(event)


class InMemoryExperienceLogger:
    """Collect experience records for assertions in tests."""

    def __init__(self) -> None:
        self.records: list[object] = []

    def log(self, record: object) -> None:
        self.records.append(record)


def _build_solver() -> NSGA2Solver:
    return NSGA2Solver(
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


def test_closed_loop_runs_and_logs_actions() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    sensor = ParetoStateSensor()
    logger = InMemoryLogger()

    runner = ClosedLoopRunner(solver=solver, sensor=sensor, controller=controller, logger=logger)
    states = runner.run(generations=6)

    assert len(states) == 7
    assert states[-1].generation == 6
    assert any(event["event"] == "action" for event in logger.events)
    action_events = [event for event in logger.events if event["event"] == "action"]
    assert all("control_state" in event for event in action_events)
    assert all("reason_detail" in event for event in action_events)
    state_events = [event for event in logger.events if event["event"] == "state"]
    assert state_events
    latest_state_event = state_events[-1]
    assert "crowding_entropy" in latest_state_event
    assert "d_dec" in latest_state_event
    assert "d_front" in latest_state_event
    assert 0.0 <= solver.config.mutation_prob <= 1.0
    assert 0.0 <= solver.config.crossover_prob <= 1.0


def test_closed_loop_records_experience_chain() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    sensor = ParetoStateSensor()
    experience_pool = ExperiencePool(max_size=10)
    experience_logger = InMemoryExperienceLogger()

    runner = ClosedLoopRunner(
        solver=solver,
        sensor=sensor,
        controller=controller,
        experience_pool=experience_pool,
        experience_logger=experience_logger,
        reward_config=RewardConfig(alpha=1.0, beta=0.1),
    )
    runner.run(generations=6)

    assert len(experience_pool) == 2
    first = experience_pool.recent(1)[0]
    assert set(first.to_dict().keys()) == {"state", "action", "reward", "next_state"}
    assert "crowding_entropy" in first.state
    assert "d_dec" in first.state
    assert "d_front" in first.state
    assert "crowding_entropy" in first.next_state
    assert "d_dec" in first.next_state
    assert "d_front" in first.next_state
    assert len(experience_logger.records) == 2


def test_rule_controller_config_validates_control_interval() -> None:
    with pytest.raises(ValueError, match="control_interval"):
        RuleControllerConfig(control_interval=0)


def test_rule_controller_decide_backwards_compatible_signature() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    population = solver.initialize_population()
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=0, population=population, previous_state=None, reference_point=None)

    action = controller.decide(
        state=state,
        current_mutation=solver.config.mutation_prob,
        current_crossover=solver.config.crossover_prob,
    )

    assert action.generation == 0
    assert 0.0 <= action.mutation_prob <= 1.0
    assert 0.0 <= action.crossover_prob <= 1.0


def test_closed_loop_skips_terminal_generation_action() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=3))
    sensor = ParetoStateSensor()
    logger = InMemoryLogger()

    runner = ClosedLoopRunner(solver=solver, sensor=sensor, controller=controller, logger=logger)
    runner.run(generations=6)

    action_generations = [int(event["generation"]) for event in logger.events if event["event"] == "action"]
    assert action_generations == [3]


def test_rule_controller_maps_to_feasibility_state() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2, feasible_ratio_low=0.95))
    population = solver.initialize_population()
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=0, population=population, previous_state=None, reference_point=None)
    state.feasible_ratio = 0.2

    action = controller.decide(
        state=state,
        current_mutation=solver.config.mutation_prob,
        current_crossover=solver.config.crossover_prob,
    )

    assert action.control_state == ControlState.INCREASE_FEASIBILITY


def test_rule_controller_maps_to_diversity_state() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2, diversity_low=0.95, feasible_ratio_low=0.0))
    population = solver.initialize_population()
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=0, population=population, previous_state=None, reference_point=None)

    action = controller.decide(
        state=state,
        current_mutation=solver.config.mutation_prob,
        current_crossover=solver.config.crossover_prob,
    )

    assert action.control_state == ControlState.INCREASE_DIVERSITY
