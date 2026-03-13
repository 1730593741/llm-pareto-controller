"""Reusable experiment matrix specifications for matched and ablation runs."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MATCHED_METHODS: tuple[str, ...] = (
    "baseline_nsga2",
    "rule_control",
    "mock_llm",
    "real_llm",
)

DEFAULT_MATCHED_BENCHMARKS: tuple[str, ...] = (
    "small_complex",
    "medium_complex",
)

DEFAULT_MATCHED_SEEDS: tuple[int, ...] = (11, 23, 37)


@dataclass(frozen=True, slots=True)
class MatchedMatrix:
    """Matched matrix axes for paper-ready method comparisons."""

    methods: tuple[str, ...] = DEFAULT_MATCHED_METHODS
    benchmarks: tuple[str, ...] = DEFAULT_MATCHED_BENCHMARKS
    seeds: tuple[int, ...] = DEFAULT_MATCHED_SEEDS
    generations: int = 20
    population_size: int = 24


@dataclass(frozen=True, slots=True)
class AblationMatrix:
    """Ablation matrix axes for method component removal/comparison."""

    seeds: tuple[int, ...] = DEFAULT_MATCHED_SEEDS
    benchmarks: tuple[str, ...] = DEFAULT_MATCHED_BENCHMARKS
    generations: int = 20
    population_size: int = 24
    tau_values: tuple[int, ...] = (1, 3, 5, 10)
    memory_windows: tuple[int, ...] = (5, 20, 50)


MATCHED_PRESETS: dict[str, MatchedMatrix] = {
    "toy": MatchedMatrix(
        methods=("baseline_nsga2", "rule_control", "mock_llm"),
        benchmarks=("small_complex_smoke",),
        seeds=(7,),
        generations=4,
        population_size=12,
    ),
    "pilot": MatchedMatrix(
        methods=DEFAULT_MATCHED_METHODS,
        benchmarks=("small_complex",),
        seeds=(11, 23),
        generations=10,
        population_size=20,
    ),
    "paper": MatchedMatrix(),
}

ABLATION_PRESETS: dict[str, AblationMatrix] = {
    "toy": AblationMatrix(
        seeds=(7,),
        benchmarks=("small_complex_smoke",),
        generations=4,
        population_size=12,
        tau_values=(1, 3),
        memory_windows=(5, 20),
    ),
    "pilot": AblationMatrix(
        seeds=(11,),
        benchmarks=("small_complex",),
        generations=10,
        population_size=20,
    ),
    "paper": AblationMatrix(),
}
