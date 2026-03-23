"""统一的 算子参数空间 与 能力模型 用于 闭环 控制."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class OperatorCapabilities:
    """运行时能力标记 用于 可控算子/搜索参数."""

    supports_eta_c: bool = False
    supports_eta_m: bool = False
    supports_repair_prob: bool = False
    supports_local_search_prob: bool = False

    def to_dict(self) -> dict[str, bool]:
        """序列化 能力标记 用于 prompt 与日志."""
        return asdict(self)


@dataclass(slots=True)
class OperatorParams:
    """统一的 参数载体 用于 优化器 算子/搜索设置."""

    mutation_prob: float
    crossover_prob: float
    eta_c: float | None = None
    eta_m: float | None = None
    repair_prob: float | None = None
    local_search_prob: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """序列化 所有 已知参数.
        
                不支持的参数应保持为 ``None`` 与 为 guarded 通过 capability
                flags 以避免误导性的有效性表述.
                """

        return asdict(self)

    def active_params(self, capabilities: OperatorCapabilities) -> dict[str, float]:
        """返回 实际受支持且有效的参数."""
        active: dict[str, float] = {
            "mutation_prob": self.mutation_prob,
            "crossover_prob": self.crossover_prob,
        }
        if capabilities.supports_eta_c and self.eta_c is not None:
            active["eta_c"] = self.eta_c
        if capabilities.supports_eta_m and self.eta_m is not None:
            active["eta_m"] = self.eta_m
        if capabilities.supports_repair_prob and self.repair_prob is not None:
            active["repair_prob"] = self.repair_prob
        if capabilities.supports_local_search_prob and self.local_search_prob is not None:
            active["local_search_prob"] = self.local_search_prob
        return active


def merge_supported_params(
    *,
    current: OperatorParams,
    target: OperatorParams,
    capabilities: OperatorCapabilities,
) -> OperatorParams:
    """构建更新后的参数集合，同时保留不受支持的维度."""
    return OperatorParams(
        mutation_prob=target.mutation_prob,
        crossover_prob=target.crossover_prob,
        eta_c=target.eta_c if capabilities.supports_eta_c else current.eta_c,
        eta_m=target.eta_m if capabilities.supports_eta_m else current.eta_m,
        repair_prob=target.repair_prob if capabilities.supports_repair_prob else current.repair_prob,
        local_search_prob=(
            target.local_search_prob if capabilities.supports_local_search_prob else current.local_search_prob
        ),
    )


def to_float_if_present(value: Any) -> float | None:
    """在可选标量存在时将其转换为 float."""
    if value is None:
        return None
    return float(value)

