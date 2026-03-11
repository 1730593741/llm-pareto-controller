"""Unified LLM client abstraction with mock support for M6 controller chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LLMClientConfig:
    """Runtime settings for LLM client modes."""

    mode: str = "mock_llm"
    provider: str = "openai"
    model: str = "gpt-mock"
    timeout_s: float = 10.0
    max_retries: int = 2
    api_key_env: str = "OPENAI_API_KEY"


class LLMClient:
    """Small facade that supports mock and future real-LLM calls."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config

    def generate_json(self, *, task: str, payload: dict[str, Any], prompt_template: str) -> dict[str, Any]:
        """Return structured response for a named chain step."""
        del prompt_template
        if self.config.mode == "mock_llm":
            return self._mock_response(task=task, payload=payload)
        if self.config.mode == "real_llm":
            raise NotImplementedError("real_llm mode is reserved for next milestone")
        raise ValueError(f"Unsupported llm mode: {self.config.mode}")

    def _mock_response(self, *, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = payload.get("state", {})
        if task == "analyst":
            feasible_ratio = float(state.get("feasible_ratio", 1.0))
            stagnation_len = int(state.get("stagnation_len", 0))
            diversity_score = float(state.get("diversity_score", 1.0))
            return {
                "stagnating": stagnation_len > 0,
                "feasibility_risk": feasible_ratio < 0.6,
                "diversity_risk": diversity_score < 0.12,
                "trend": "improving" if float(state.get("delta_hv", 0.0)) > 1e-4 else "flat",
                "summary": "mock diagnosis",
            }

        if task == "strategist":
            diagnosis = payload.get("diagnosis", {})
            if diagnosis.get("feasibility_risk"):
                strategy = "improve_feasibility"
            elif diagnosis.get("diversity_risk") or diagnosis.get("stagnating"):
                strategy = "increase_exploration"
            else:
                strategy = "stabilize_exploitation"
            return {"strategy": strategy, "rationale": "mock strategy"}

        if task == "actuator":
            strategy = payload.get("strategy", {}).get("strategy", "stabilize_exploitation")
            params = payload.get("current_params", {})
            mutation = float(params.get("mutation_prob", 0.1))
            crossover = float(params.get("crossover_prob", 0.9))

            if strategy == "increase_exploration":
                mutation += 0.06
                crossover -= 0.04
            elif strategy == "improve_feasibility":
                mutation -= 0.03
                crossover += 0.04
            else:
                mutation -= 0.02
                crossover += 0.02

            return {
                "mutation_prob": mutation,
                "crossover_prob": crossover,
                "reason": f"mock::{strategy}",
            }

        raise ValueError(f"Unsupported llm task: {task}")
