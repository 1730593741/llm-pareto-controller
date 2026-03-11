"""Strategist role: map diagnosis into a compact control strategy set."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from infra.llm_client import LLMClient
from llm.analyst import AnalysisResult


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
        raw = self.client.generate_json(task="strategist", payload=payload, prompt_template=self.prompt_template)
        return StrategyDecision.model_validate(raw)
