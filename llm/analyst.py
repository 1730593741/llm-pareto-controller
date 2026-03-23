"""Analyst 角色: 总结 Pareto 状态 与 近期经验 转换为 诊断信息."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from controller.control_semantics import ControlState
from infra.llm_client import LLMClient
from memory.experience_pool import ExperienceRecord
from sensing.pareto_state import ParetoState

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Structured diagnosis consumed 通过 该 strategist."""

    control_state: ControlState
    reason: str
    convergence_signal: float
    diversity_signal: float
    feasibility_signal: float


class Analyst:
    """构建 diagnosis object 从 状态 与 近期经验."""

    def __init__(self, client: LLMClient, prompt_path: str = "llm/prompts/analyst.txt") -> None:
        self.client = client
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def analyze(self, *, state: ParetoState, recent_experiences: list[ExperienceRecord]) -> AnalysisResult:
        payload = {
            "state": state.to_dict(),
            "recent_experiences": [record.to_dict() for record in recent_experiences],
        }
        response = self.client.generate_json(task="analyst", payload=payload, prompt_template=self.prompt_template)
        if not response.content:
            logger.warning("Analyst hold fallback due to llm error: %s", response.error)
            return _fallback_analysis(state=state, recent_experiences=recent_experiences)

        result = AnalysisResult.model_validate(response.content)
        if response.error:
            logger.warning("Analyst used fallback mode=%s: %s", response.mode_used, response.error)
        return result


def _fallback_analysis(*, state: ParetoState, recent_experiences: list[ExperienceRecord]) -> AnalysisResult:
    recent_rewards = [float(record.reward) for record in recent_experiences]
    reward_trend = (sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0

    if state.feasible_ratio < 0.6:
        control_state = ControlState.INCREASE_FEASIBILITY
        reason = "fallback_low_feasible_ratio"
    elif state.diversity_score < 0.12:
        control_state = ControlState.INCREASE_DIVERSITY
        reason = "fallback_low_diversity"
    elif state.stagnation_len > 0 or state.delta_hv <= 1e-4 or reward_trend < -1e-6:
        control_state = ControlState.INCREASE_CONVERGENCE
        reason = "fallback_stagnation_or_negative_recent_reward"
    else:
        control_state = ControlState.MAINTAIN_BALANCE
        reason = "fallback_balanced"

    return AnalysisResult(
        control_state=control_state,
        reason=reason,
        convergence_signal=float(max(0.0, min(1.0, state.rank1_ratio))),
        diversity_signal=float(max(0.0, min(1.0, state.diversity_score))),
        feasibility_signal=float(max(0.0, min(1.0, state.feasible_ratio))),
    )
