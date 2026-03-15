"""Consistency checks for DWTA domain terminology across assets."""

from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_dwta_configs_use_canonical_entity_keys() -> None:
    """DWTA configs should consistently use munition_types/weapons/targets keys."""
    for path in [
        "experiments/configs/dwta_small.yaml",
        "experiments/configs/dwta_small_smoke.yaml",
    ]:
        problem = _load_yaml(path)["problem"]
        assert "munition_types" in problem
        assert "weapons" in problem
        assert "targets" in problem


def test_dwta_strategist_prompt_mentions_canonical_entities() -> None:
    """The Strategist prompt should use canonical Weapons/Targets/Munitions terms."""
    prompt_text = Path("llm/prompts/strategist.txt").read_text(encoding="utf-8")
    assert "Weapons, Targets, and Munitions" in prompt_text
