"""Actuator role: convert strategy decisions into bounded control actions."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from controller.closed_loop import ControlAction, OperatorParams
from infra.llm_client import LLMClient
from llm.strategist import StrategyDecision

logger = logging.getLogger(__name__)


class ActuatorOutput(BaseModel):
    """Structured LLM output before conversion to ControlAction."""

    mutation_prob: float = Field()
    crossover_prob: float = Field()
    reason_detail: str


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
        max_step: float = 0.08,
    ) -> None:
        self.client = client
        self.min_mutation_prob = min_mutation_prob
        self.max_mutation_prob = max_mutation_prob
        self.min_crossover_prob = min_crossover_prob
        self.max_crossover_prob = max_crossover_prob
        self.max_step = max_step
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def act(self, *, generation: int, strategy: StrategyDecision, current_params: OperatorParams) -> ControlAction:
        payload = {
            "strategy": strategy.model_dump(mode="json"),
            "current_params": current_params.to_dict(),
            "constraints": {
                "min_mutation_prob": self.min_mutation_prob,
                "max_mutation_prob": self.max_mutation_prob,
                "min_crossover_prob": self.min_crossover_prob,
                "max_crossover_prob": self.max_crossover_prob,
                "max_step": self.max_step,
            },
        }
        response = self.client.generate_json(task="actuator", payload=payload, prompt_template=self.prompt_template)
        if not response.content:
            logger.warning("Actuator hold fallback due to llm error: %s", response.error)
            return ControlAction(
                generation=generation,
                mutation_prob=_clip(current_params.mutation_prob, self.min_mutation_prob, self.max_mutation_prob),
                crossover_prob=_clip(current_params.crossover_prob, self.min_crossover_prob, self.max_crossover_prob),
                reason="hold_due_to_llm_error",
                control_state=strategy.control_state,
                reason_detail="hold_due_to_llm_error",
            )

        llm_output = ActuatorOutput.model_validate(response.content)
        target_mutation = _clip(llm_output.mutation_prob, self.min_mutation_prob, self.max_mutation_prob)
        target_crossover = _clip(llm_output.crossover_prob, self.min_crossover_prob, self.max_crossover_prob)

        mutation = _smooth(current=current_params.mutation_prob, target=target_mutation, max_step=self.max_step)
        crossover = _smooth(current=current_params.crossover_prob, target=target_crossover, max_step=self.max_step)

        reason_detail = llm_output.reason_detail
        if response.error:
            logger.warning("Actuator used fallback mode=%s: %s", response.mode_used, response.error)
            reason_detail = f"{reason_detail};fallback={response.mode_used}"
        return ControlAction(
            generation=generation,
            mutation_prob=mutation,
            crossover_prob=crossover,
            reason=strategy.control_state.value,
            control_state=strategy.control_state,
            reason_detail=reason_detail,
        )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smooth(*, current: float, target: float, max_step: float) -> float:
    delta = target - current
    if abs(delta) <= max_step:
        return target
    if delta > 0:
        return current + max_step
    return current - max_step
