"""Tests for LLM actuator structured output and clipping."""

from controller.closed_loop import OperatorParams
from infra.llm_client import LLMClient, LLMClientConfig
from llm.actuator import Actuator
from llm.strategist import StrategyDecision


class OutOfRangeMockClient(LLMClient):
    """Mock client that returns out-of-bound probabilities for clipping tests."""

    def generate_json(self, *, task: str, payload: dict[str, object], prompt_template: str) -> dict[str, object]:
        del payload, prompt_template
        if task != "actuator":
            raise ValueError("This test client only supports actuator task")
        return {"mutation_prob": 1.4, "crossover_prob": -0.1, "reason": "extreme"}


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
        strategy=StrategyDecision(strategy="increase_exploration", rationale="test"),
        current_params=OperatorParams(mutation_prob=0.1, crossover_prob=0.9),
    )

    assert action.generation == 3
    assert action.mutation_prob == 0.7
    assert action.crossover_prob == 0.4
    assert action.reason == "extreme"
