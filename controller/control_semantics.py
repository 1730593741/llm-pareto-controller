"""Shared four-state control semantics for rule and LLM controllers."""

from __future__ import annotations

from enum import Enum


class ControlState(str, Enum):
    """Canonical high-level control intent for closed-loop adjustments."""

    INCREASE_DIVERSITY = "increase_diversity"
    INCREASE_CONVERGENCE = "increase_convergence"
    INCREASE_FEASIBILITY = "increase_feasibility"
    MAINTAIN_BALANCE = "maintain_balance"
