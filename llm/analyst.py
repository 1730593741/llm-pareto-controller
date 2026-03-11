"""Analyst role: summarize Pareto state and recent experiences into diagnostics."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from infra.llm_client import LLMClient
from memory.experience_pool import ExperienceRecord
from sensing.pareto_state import ParetoState


class AnalysisResult(BaseModel):
    """Structured diagnosis consumed by the strategist."""

    stagnating: bool
    feasibility_risk: bool
    diversity_risk: bool
    trend: str
    summary: str


class Analyst:
    """Build diagnosis object from state and recent experiences."""

    def __init__(self, client: LLMClient, prompt_path: str = "llm/prompts/analyst.txt") -> None:
        self.client = client
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def analyze(self, *, state: ParetoState, recent_experiences: list[ExperienceRecord]) -> AnalysisResult:
        payload = {
            "state": state.to_dict(),
            "recent_experiences": [record.to_dict() for record in recent_experiences],
        }
        raw = self.client.generate_json(task="analyst", payload=payload, prompt_template=self.prompt_template)
        return AnalysisResult.model_validate(raw)
