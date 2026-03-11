"""Strategist role: map diagnosis into a compact control strategy set."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from infra.llm_client import LLMClient
from llm.analyst import AnalysisResult

logger = logging.getLogger(__name__)


class StrategyDecision(BaseModel):
    """High-level strategy selected for the actuator."""

    strategy: Literal["increase_exploration", "improve_feasibility", "stabilize_exploitation"]
    rationale: str


class Strategist:
    """Translate analysis into one of a small strategy set."""

    def __init__(self, client: LLMClient, prompt_path: str = "llm/prompts/strategist.txt") -> None:
        self.client = client
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def plan(self, diagnosis: AnalysisResult) -> StrategyDecision:
        payload = {"diagnosis": diagnosis.model_dump()}
        response = self.client.generate_json(task="strategist", payload=payload, prompt_template=self.prompt_template)
        if not response.content:
            logger.warning("Strategist hold fallback due to llm error: %s", response.error)
            return StrategyDecision(strategy="stabilize_exploitation", rationale="fallback_from_llm_error")
        decision = StrategyDecision.model_validate(response.content)
        if response.error:
            logger.warning("Strategist used fallback mode=%s: %s", response.mode_used, response.error)
        return decision
