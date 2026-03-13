"""Unified LLM client with mock/real transport, retry, and safe fallback behavior."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMClientConfig:
    """Runtime settings for LLM client modes."""

    mode: Literal["mock_llm", "real_llm"] = "mock_llm"
    provider: str = "openai"
    model: str = "gpt-mock"
    timeout_s: float = 10.0
    max_retries: int = 2
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model_env: str = "OPENAI_MODEL"
    base_url: str = "https://api.openai.com/v1/chat/completions"
    fallback_mode: Literal["mock_llm", "hold"] = "mock_llm"


@dataclass(slots=True)
class LLMResponse:
    """Unified response passed to upper LLM business roles."""

    success: bool
    task: str
    content: dict[str, Any] | None = None
    error: str | None = None
    mode_used: str = "mock_llm"


class LLMClient:
    """Facade for mock and real LLM calls without leaking HTTP details upstream."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config

    def generate_json(self, *, task: str, payload: dict[str, Any], prompt_template: str) -> LLMResponse:
        """Return a unified structured response for chain tasks."""
        if self.config.mode == "mock_llm":
            return LLMResponse(success=True, task=task, content=self._mock_response(task=task, payload=payload))
        if self.config.mode == "real_llm":
            return self._generate_real(task=task, payload=payload, prompt_template=prompt_template)
        raise ValueError(f"Unsupported llm mode: {self.config.mode}")

    def _generate_real(self, *, task: str, payload: dict[str, Any], prompt_template: str) -> LLMResponse:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            msg = f"Missing API key in env var: {self.config.api_key_env}"
            logger.warning("%s", msg)
            return self._fallback(task=task, payload=payload, error=msg)

        base_url = os.getenv(self.config.base_url_env, self.config.base_url)
        model = os.getenv(self.config.model_env, self.config.model)
        try:
            output = self._call_real_with_retry(
                base_url=base_url,
                api_key=api_key,
                model=model,
                payload=payload,
                prompt_template=prompt_template,
            )
            return LLMResponse(success=True, task=task, content=output, mode_used="real_llm")
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
            msg = f"real_llm task={task} failed: {exc}"
            logger.exception(msg)
            return self._fallback(task=task, payload=payload, error=msg)

    def _fallback(self, *, task: str, payload: dict[str, Any], error: str) -> LLMResponse:
        if self.config.fallback_mode == "mock_llm":
            logger.warning("Fallback to mock_llm for task=%s", task)
            return LLMResponse(
                success=False,
                task=task,
                content=self._mock_response(task=task, payload=payload),
                error=error,
                mode_used="mock_llm",
            )
        return LLMResponse(success=False, task=task, content=None, error=error, mode_used="hold")

    def _call_real_with_retry(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        payload: dict[str, Any],
        prompt_template: str,
    ) -> dict[str, Any]:
        attempts = max(0, int(self.config.max_retries)) + 1
        retrying = Retrying(
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)),
            wait=wait_exponential(multiplier=0.3, min=0.3, max=3),
            stop=stop_after_attempt(attempts),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self._call_real_once(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    payload=payload,
                    prompt_template=prompt_template,
                )
        raise RuntimeError("retrying exhausted unexpectedly")

    def _call_real_once(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        payload: dict[str, Any],
        prompt_template: str,
    ) -> dict[str, Any]:
        max_retries = max(0, int(self.config.max_retries))
        timeout = max(0.1, float(self.config.timeout_s))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                base_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": prompt_template},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                },
            )
            if response.status_code >= 500 and max_retries > 0:
                response.raise_for_status()
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"LLM HTTP {response.status_code}: {response.text[:200]}",
                    request=response.request,
                    response=response,
                )
        body = response.json()
        content_text = body["choices"][0]["message"]["content"]
        return json.loads(_strip_code_fence(content_text))

    def _mock_response(self, *, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = payload.get("state", {})
        if task == "analyst":
            feasible_ratio = float(state.get("feasible_ratio", 1.0))
            stagnation_len = int(state.get("stagnation_len", 0))
            diversity_score = float(state.get("diversity_score", 1.0))
            experiences = payload.get("recent_experiences", [])
            rewards = [float(item.get("reward", 0.0)) for item in experiences if isinstance(item, dict)]
            reward_trend = (sum(rewards) / len(rewards)) if rewards else 0.0
            if feasible_ratio < 0.6:
                control_state = "increase_feasibility"
                reason = "mock_low_feasible_ratio"
            elif diversity_score < 0.12:
                control_state = "increase_diversity"
                reason = "mock_low_diversity"
            elif stagnation_len > 0 or float(state.get("delta_hv", 0.0)) <= 1e-4 or reward_trend < -1e-6:
                control_state = "increase_convergence"
                reason = "mock_stagnation_or_negative_recent_reward"
            else:
                control_state = "maintain_balance"
                reason = "mock_balanced"
            return {
                "control_state": control_state,
                "reason": reason,
                "convergence_signal": float(state.get("rank1_ratio", 0.0)),
                "diversity_signal": diversity_score,
                "feasibility_signal": feasible_ratio,
            }

        if task == "strategist":
            diagnosis = payload.get("diagnosis", {})
            return {
                "control_state": diagnosis.get("control_state", "maintain_balance"),
                "rationale": "mock strategy",
            }

        if task == "actuator":
            strategy = payload.get("strategy", {}).get("control_state", "maintain_balance")
            params = payload.get("current_params", {})
            capabilities = payload.get("capabilities", {})
            mutation = float(params.get("mutation_prob", 0.1))
            crossover = float(params.get("crossover_prob", 0.9))
            repair_prob = params.get("repair_prob")

            if strategy == "increase_diversity":
                mutation += 0.06
                crossover -= 0.04
            elif strategy == "increase_feasibility":
                mutation -= 0.03
                crossover += 0.04
                if capabilities.get("supports_repair_prob"):
                    repair_prob = 1.0 if repair_prob is None else float(repair_prob) + 0.08
            elif strategy == "increase_convergence":
                mutation -= 0.02
                crossover += 0.03
            else:
                mutation += 0.0
                crossover += 0.0

            response = {
                "mutation_prob": mutation,
                "crossover_prob": crossover,
                "reason_detail": f"mock::{strategy}",
            }
            if capabilities.get("supports_repair_prob"):
                response["repair_prob"] = repair_prob
            return response

        raise ValueError(f"Unsupported llm task: {task}")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped
