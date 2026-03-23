"""Strategist 角色: map analysis 到 four-状态 控制 意图，并带有 依据."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from controller.control_semantics import ControlState
from infra.llm_client import LLMClient
from llm.analyst import AnalysisResult

logger = logging.getLogger(__name__)


class StrategyDecision(BaseModel):
    """为 actuator 选择的高层控制状态."""

    control_state: ControlState
    rationale: str


class Strategist:
    """将 analyst 输出转换为规范的控制状态决策."""

    def __init__(self, client: LLMClient, prompt_path: str = "llm/prompts/strategist.txt") -> None:
        self.client = client
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def plan(self, diagnosis: AnalysisResult) -> StrategyDecision:
        payload = {"diagnosis": diagnosis.model_dump(mode="json")}
        response = self.client.generate_json(task="strategist", payload=payload, prompt_template=self.prompt_template)
        if not response.content:
            logger.warning("Strategist hold fallback due to llm error: %s", response.error)
            return StrategyDecision(control_state=diagnosis.control_state, rationale="fallback_use_analyst_state")
        decision = StrategyDecision.model_validate(response.content)
        if response.error:
            logger.warning("Strategist used fallback mode=%s: %s", response.mode_used, response.error)
        return decision
