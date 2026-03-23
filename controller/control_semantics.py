"""共享的 four-状态 控制 semantics 用于 规则 与 LLM 控制器."""

from __future__ import annotations

from enum import Enum


class ControlState(str, Enum):
    """规范的 高层 控制 意图 用于 闭环 调整."""

    INCREASE_DIVERSITY = "increase_diversity"
    INCREASE_CONVERGENCE = "increase_convergence"
    INCREASE_FEASIBILITY = "increase_feasibility"
    MAINTAIN_BALANCE = "maintain_balance"
