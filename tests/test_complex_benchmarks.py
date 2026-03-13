"""Smoke tests for complex constrained assignment benchmarks."""

from __future__ import annotations

from pathlib import Path

from main import build_solver, load_config, run_experiment
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver


ROOT = Path(__file__).resolve().parents[1]


def test_small_complex_smoke_runs_and_tracks_feasibility() -> None:
    config = load_config(ROOT / "experiments/configs/small_complex_smoke.yaml")
    solver = build_solver(config.problem, config.optimizer)

    population = solver.run()

    assert len(population) == config.optimizer.population_size
    assert any("compatibility" in ind.constraint_components for ind in population)
    assert any("time_window" in ind.constraint_components for ind in population)
    assert any("stage_transition" in ind.constraint_components for ind in population)


def test_repair_probability_changes_feasibility_in_complex_case() -> None:
    kwargs = dict(
        n_tasks=8,
        n_resources=3,
        cost_matrix=[
            [2.0, 1.2, 2.8],
            [1.8, 2.1, 1.1],
            [2.2, 1.5, 1.4],
            [1.7, 2.4, 1.3],
            [2.4, 1.1, 2.2],
            [1.3, 2.0, 1.9],
            [2.1, 1.4, 1.6],
            [1.5, 2.3, 1.2],
        ],
        task_loads=[1.1, 0.9, 1.0, 1.2, 1.0, 0.8, 1.1, 0.9],
        capacities=[2.8, 2.8, 2.8],
        compatibility_matrix=[
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        task_time_windows=[[0, 2], [1, 4], [0, 3], [2, 5], [1, 3], [0, 2], [2, 4], [1, 5]],
        resource_time_windows=[[0, 3], [1, 5], [2, 6]],
        resource_stage_levels=[1, 2, 3],
        stage_transitions=[[0, 3], [1, 4], [2, 6]],
    )

    with_repair = NSGA2Solver(
        **kwargs,
        config=NSGA2Config(population_size=20, generations=5, crossover_prob=0.9, mutation_prob=0.2, repair_prob=1.0, seed=99),
    )
    no_repair = NSGA2Solver(
        **kwargs,
        config=NSGA2Config(population_size=20, generations=5, crossover_prob=0.9, mutation_prob=0.2, repair_prob=0.0, seed=99),
    )

    repaired_population = with_repair.run()
    raw_population = no_repair.run()

    repaired_mean_cv = sum(ind.constraint_violation for ind in repaired_population) / len(repaired_population)
    raw_mean_cv = sum(ind.constraint_violation for ind in raw_population) / len(raw_population)

    assert repaired_mean_cv <= raw_mean_cv


def test_complex_benchmark_configs_load() -> None:
    for config_name in ("small_complex.yaml", "medium_complex.yaml", "hard_complex.yaml"):
        config = load_config(ROOT / f"experiments/configs/{config_name}")
        assert config.problem.compatibility_matrix is not None
        assert config.problem.task_time_windows is not None
        assert config.problem.stage_transitions is not None


def test_toy_config_still_loads() -> None:
    config = load_config(ROOT / "experiments/configs/default.yaml")
    assert config.problem.compatibility_matrix is None
    assert config.problem.task_time_windows is None
    assert config.problem.stage_transitions is None


def test_small_complex_smoke_experiment_logs_feasibility_signals() -> None:
    summary = run_experiment(str(ROOT / "experiments/configs/small_complex_smoke.yaml"))
    generation_log_path = Path(summary["generation_log_path"])
    payload_lines = generation_log_path.read_text(encoding="utf-8").splitlines()

    assert payload_lines
    assert any("\"feasible_ratio\"" in line for line in payload_lines)
    assert any("\"mean_cv\"" in line for line in payload_lines)
