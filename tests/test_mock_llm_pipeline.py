"""Integration test for mock_llm closed-loop mode."""

from pathlib import Path

from main import build_controller, build_solver, load_config
from controller.closed_loop import ClosedLoopRunner, LLMChainController
from memory.experience_pool import ExperiencePool
from sensing.pareto_state import ParetoStateSensor


def test_mock_llm_mode_runs_closed_loop(tmp_path: Path) -> None:
    config = load_config("experiments/configs/mock_llm.yaml")
    config.log_path = str(tmp_path / "events.jsonl")
    config.memory.experience_log_path = str(tmp_path / "exp.jsonl")

    solver = build_solver(config.problem, config.optimizer)
    controller = build_controller(config)
    assert isinstance(controller, LLMChainController)

    experience_pool = ExperiencePool(config.memory.memory_window)
    runner = ClosedLoopRunner(
        solver=solver,
        sensor=ParetoStateSensor(),
        controller=controller,
        experience_pool=experience_pool,
    )
    states = runner.run(generations=6)

    assert states[-1].generation == 6
    assert 0.0 <= solver.config.mutation_prob <= 1.0
    assert 0.0 <= solver.config.crossover_prob <= 1.0

    records = experience_pool.recent(10)
    assert records
    assert all("control_state" in record.action for record in records)
