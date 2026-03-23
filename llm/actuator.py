"""Actuator 角色: convert strategy decisions 转换为 bounded 控制 动作."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from controller.closed_loop import ControlAction
from controller.operator_space import OperatorCapabilities, OperatorParams
from infra.llm_client import LLMClient
from llm.strategist import StrategyDecision

logger = logging.getLogger(__name__)


class ActuatorOutput(BaseModel):
    """结构化 LLM 输出 before conversion 到 ControlAction."""

    mutation_prob: float = Field()
    crossover_prob: float = Field()
    eta_c: float | None = None
    eta_m: float | None = None
    repair_prob: float | None = None
    local_search_prob: float | None = None
    reason_detail: str


class Actuator:
    """将策略输出转换为安全的算子概率."""

    def __init__(
        self,
        client: LLMClient,
        *,
        min_mutation_prob: float,
        max_mutation_prob: float,
        min_crossover_prob: float,
        max_crossover_prob: float,
        min_eta_c: float,
        max_eta_c: float,
        min_eta_m: float,
        max_eta_m: float,
        min_repair_prob: float,
        max_repair_prob: float,
        min_local_search_prob: float,
        max_local_search_prob: float,
        prompt_path: str = "llm/prompts/actuator.txt",
        max_step: float = 0.08,
    ) -> None:
        self.client = client
        self.min_mutation_prob = min_mutation_prob
        self.max_mutation_prob = max_mutation_prob
        self.min_crossover_prob = min_crossover_prob
        self.max_crossover_prob = max_crossover_prob
        self.min_eta_c = min_eta_c
        self.max_eta_c = max_eta_c
        self.min_eta_m = min_eta_m
        self.max_eta_m = max_eta_m
        self.min_repair_prob = min_repair_prob
        self.max_repair_prob = max_repair_prob
        self.min_local_search_prob = min_local_search_prob
        self.max_local_search_prob = max_local_search_prob
        self.max_step = max_step
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def act(
        self,
        *,
        generation: int,
        strategy: StrategyDecision,
        current_params: OperatorParams,
        capabilities: OperatorCapabilities,
    ) -> ControlAction:
        payload = {
            "strategy": strategy.model_dump(mode="json"),
            "current_params": current_params.to_dict(),
            "capabilities": capabilities.to_dict(),
            "constraints": {
                "min_mutation_prob": self.min_mutation_prob,
                "max_mutation_prob": self.max_mutation_prob,
                "min_crossover_prob": self.min_crossover_prob,
                "max_crossover_prob": self.max_crossover_prob,
                "min_eta_c": self.min_eta_c,
                "max_eta_c": self.max_eta_c,
                "min_eta_m": self.min_eta_m,
                "max_eta_m": self.max_eta_m,
                "min_repair_prob": self.min_repair_prob,
                "max_repair_prob": self.max_repair_prob,
                "min_local_search_prob": self.min_local_search_prob,
                "max_local_search_prob": self.max_local_search_prob,
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
                requested_params=current_params.to_dict(),
                applied_params=current_params.active_params(capabilities),
                capabilities=capabilities.to_dict(),
                control_state=strategy.control_state,
                reason_detail="hold_due_to_llm_error",
            )

        llm_output = ActuatorOutput.model_validate(response.content)
        target_mutation = _clip(llm_output.mutation_prob, self.min_mutation_prob, self.max_mutation_prob)
        target_crossover = _clip(llm_output.crossover_prob, self.min_crossover_prob, self.max_crossover_prob)

        mutation = _smooth(current=current_params.mutation_prob, target=target_mutation, max_step=self.max_step)
        crossover = _smooth(current=current_params.crossover_prob, target=target_crossover, max_step=self.max_step)
        eta_c = _bounded_param(
            current=current_params.eta_c,
            proposed=llm_output.eta_c,
            low=self.min_eta_c,
            high=self.max_eta_c,
            max_step=self.max_step * 20,
            enabled=capabilities.supports_eta_c,
        )
        eta_m = _bounded_param(
            current=current_params.eta_m,
            proposed=llm_output.eta_m,
            low=self.min_eta_m,
            high=self.max_eta_m,
            max_step=self.max_step * 20,
            enabled=capabilities.supports_eta_m,
        )
        repair_prob = _bounded_param(
            current=current_params.repair_prob,
            proposed=llm_output.repair_prob,
            low=self.min_repair_prob,
            high=self.max_repair_prob,
            max_step=self.max_step,
            enabled=capabilities.supports_repair_prob,
        )
        local_search_prob = _bounded_param(
            current=current_params.local_search_prob,
            proposed=llm_output.local_search_prob,
            low=self.min_local_search_prob,
            high=self.max_local_search_prob,
            max_step=self.max_step,
            enabled=capabilities.supports_local_search_prob,
        )

        reason_detail = llm_output.reason_detail
        if response.error:
            logger.warning("Actuator used fallback mode=%s: %s", response.mode_used, response.error)
            reason_detail = f"{reason_detail};fallback={response.mode_used}"
        requested = OperatorParams(
            mutation_prob=target_mutation,
            crossover_prob=target_crossover,
            eta_c=llm_output.eta_c,
            eta_m=llm_output.eta_m,
            repair_prob=llm_output.repair_prob,
            local_search_prob=llm_output.local_search_prob,
        )
        applied = OperatorParams(
            mutation_prob=mutation,
            crossover_prob=crossover,
            eta_c=eta_c,
            eta_m=eta_m,
            repair_prob=repair_prob,
            local_search_prob=local_search_prob,
        )
        return ControlAction(
            generation=generation,
            mutation_prob=mutation,
            crossover_prob=crossover,
            reason=strategy.control_state.value,
            requested_params=requested.to_dict(),
            applied_params=applied.active_params(capabilities),
            capabilities=capabilities.to_dict(),
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


def _bounded_param(
    *,
    current: float | None,
    proposed: float | None,
    low: float,
    high: float,
    max_step: float,
    enabled: bool,
) -> float | None:
    if not enabled:
        return current
    if current is None:
        current = low
    if proposed is None:
        return current
    target = _clip(proposed, low, high)
    return _smooth(current=current, target=target, max_step=max_step)
