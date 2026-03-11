"""Actuator role: convert strategy decisions into bounded control actions."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from controller.closed_loop import ControlAction, OperatorParams
from infra.llm_client import LLMClient
from llm.strategist import StrategyDecision


class ActuatorOutput(BaseModel):
    """Structured LLM output before conversion to ControlAction."""

    mutation_prob: float
    crossover_prob: float
    reason: str


class Actuator:
    """Turn strategy output into safe operator probabilities."""

    def __init__(
        self,
        client: LLMClient,
        *,
        min_mutation_prob: float,
        max_mutation_prob: float,
        min_crossover_prob: float,
        max_crossover_prob: float,
        prompt_path: str = "llm/prompts/actuator.txt",
    ) -> None:
        self.client = client
        self.min_mutation_prob = min_mutation_prob
        self.max_mutation_prob = max_mutation_prob
        self.min_crossover_prob = min_crossover_prob
        self.max_crossover_prob = max_crossover_prob
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def act(self, *, generation: int, strategy: StrategyDecision, current_params: OperatorParams) -> ControlAction:
        payload = {
            "strategy": strategy.model_dump(),
            "current_params": current_params.to_dict(),
        }
        raw = self.client.generate_json(task="actuator", payload=payload, prompt_template=self.prompt_template)
        llm_output = ActuatorOutput.model_validate(raw)
        return ControlAction(
            generation=generation,
            mutation_prob=_clip(llm_output.mutation_prob, self.min_mutation_prob, self.max_mutation_prob),
            crossover_prob=_clip(llm_output.crossover_prob, self.min_crossover_prob, self.max_crossover_prob),
            reason=llm_output.reason,
        )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
