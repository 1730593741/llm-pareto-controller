"""Tests for LLM actuator structured output and clipping."""

from __future__ import annotations

from controller.closed_loop import OperatorParams
from controller.control_semantics import ControlState
from infra.llm_client import LLMClient, LLMClientConfig, LLMResponse
from llm.actuator import Actuator
from llm.strategist import StrategyDecision


class OutOfRangeMockClient(LLMClient):
    """Mock client that returns out-of-bound probabilities for clipping tests."""

    def generate_json(self, *, task: str, payload: dict[str, object], prompt_template: str):
        del payload, prompt_template
        if task != "actuator":
            raise ValueError("This test client only supports actuator task")
        return LLMResponse(
            success=True,
            task="actuator",
            content={"mutation_prob": 1.4, "crossover_prob": -0.1, "reason_detail": "extreme"},
        )


class BadFormatMockClient(LLMClient):
    """Mock client that simulates malformed LLM payload for actuator hold-path tests."""

    def generate_json(self, *, task: str, payload: dict[str, object], prompt_template: str):
        del task, payload, prompt_template
        return LLMResponse(success=False, task="actuator", content=None, error="bad_format", mode_used="hold")


def test_actuator_clips_parameters_to_bounds() -> None:
    client = OutOfRangeMockClient(LLMClientConfig(mode="mock_llm"))
    actuator = Actuator(
        client,
        min_mutation_prob=0.05,
        max_mutation_prob=0.7,
        min_crossover_prob=0.4,
        max_crossover_prob=0.95,
    )
    action = actuator.act(
        generation=3,
        strategy=StrategyDecision(control_state=ControlState.INCREASE_DIVERSITY, rationale="test"),
        current_params=OperatorParams(mutation_prob=0.1, crossover_prob=0.9),
    )

    assert action.generation == 3
    assert action.mutation_prob <= 0.7
    assert action.crossover_prob >= 0.4
    assert action.reason == "increase_diversity"
    assert action.reason_detail == "extreme"


def test_actuator_holds_when_llm_response_is_unavailable() -> None:
    client = BadFormatMockClient(LLMClientConfig(mode="mock_llm"))
    actuator = Actuator(
        client,
        min_mutation_prob=0.05,
        max_mutation_prob=0.7,
        min_crossover_prob=0.4,
        max_crossover_prob=0.95,
    )
    params = OperatorParams(mutation_prob=0.2, crossover_prob=0.85)
    action = actuator.act(
        generation=2,
        strategy=StrategyDecision(control_state=ControlState.MAINTAIN_BALANCE, rationale="test"),
        current_params=params,
    )

    assert action.mutation_prob == params.mutation_prob
    assert action.crossover_prob == params.crossover_prob
    assert action.reason == "hold_due_to_llm_error"
    assert action.control_state == ControlState.MAINTAIN_BALANCE
