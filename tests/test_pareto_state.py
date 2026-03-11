"""Tests for Pareto-state sensing over NSGA-II populations."""

from optimizers.nsga2.population import Individual
from sensing.pareto_state import ParetoStateSensor


def test_pareto_state_sensor_outputs_expected_fields() -> None:
    population = [
        Individual(genome=[0], objectives=(1.0, 4.0), constraint_violation=0.0, feasible=True),
        Individual(genome=[1], objectives=(2.0, 2.0), constraint_violation=0.0, feasible=True),
        Individual(genome=[2], objectives=(4.0, 1.0), constraint_violation=1.0, feasible=False),
    ]

    sensor = ParetoStateSensor()
    state = sensor.sense(generation=1, population=population, reference_point=(5.0, 5.0))

    assert state.generation == 1
    assert state.hv == 11.0
    assert state.delta_hv == 11.0
    assert state.feasible_ratio == 2 / 3
    assert state.rank1_ratio == 2 / 3
    assert state.mean_cv == 1 / 3
    assert state.diversity_score > 0.0
    assert state.stagnation_len == 0
    assert state.to_dict()["hv"] == 11.0


def test_pareto_state_stagnation_counter_increments() -> None:
    population = [
        Individual(genome=[0], objectives=(2.0, 2.0), constraint_violation=0.0, feasible=True),
        Individual(genome=[1], objectives=(2.5, 2.5), constraint_violation=0.0, feasible=True),
    ]

    sensor = ParetoStateSensor()
    state1 = sensor.sense(generation=1, population=population, reference_point=(5.0, 5.0))
    state2 = sensor.sense(
        generation=2,
        population=population,
        previous_state=state1,
        reference_point=(5.0, 5.0),
    )

    assert state2.delta_hv == 0.0
    assert state2.stagnation_len == 1


def test_pareto_state_sensor_does_not_mutate_population_ranks() -> None:
    population = [
        Individual(genome=[0], objectives=(1.0, 4.0), rank=7, constraint_violation=0.0, feasible=True),
        Individual(genome=[1], objectives=(2.0, 2.0), rank=8, constraint_violation=0.0, feasible=True),
    ]

    sensor = ParetoStateSensor()
    _ = sensor.sense(generation=1, population=population, reference_point=(5.0, 5.0))

    assert [ind.rank for ind in population] == [7, 8]
