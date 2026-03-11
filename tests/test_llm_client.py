"""Tests for LLM client mode handling."""

import pytest

from infra.llm_client import LLMClient, LLMClientConfig


def test_llm_client_real_mode_placeholder() -> None:
    client = LLMClient(LLMClientConfig(mode="real_llm"))
    with pytest.raises(NotImplementedError):
        client.generate_json(task="analyst", payload={}, prompt_template="")


def test_llm_client_invalid_mode_raises_value_error() -> None:
    client = LLMClient(LLMClientConfig(mode="invalid_mode"))
    with pytest.raises(ValueError, match="Unsupported llm mode"):
        client.generate_json(task="analyst", payload={}, prompt_template="")
