"""Unified operator-parameter space and capability model for closed-loop control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class OperatorCapabilities:
    """Runtime capability flags for controllable operator/search parameters."""

    supports_eta_c: bool = False
    supports_eta_m: bool = False
    supports_repair_prob: bool = False
    supports_local_search_prob: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Serialize capability flags for prompts and logs."""
        return asdict(self)


@dataclass(slots=True)
class OperatorParams:
    """Unified parameter carrier for optimizer operator/search settings."""

    mutation_prob: float
    crossover_prob: float
    eta_c: float | None = None
    eta_m: float | None = None
    repair_prob: float | None = None
    local_search_prob: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Serialize all known parameters.

        Unsupported parameters should remain ``None`` and are guarded by capability
        flags to avoid misleading claims of effectiveness.
        """

        return asdict(self)

    def active_params(self, capabilities: OperatorCapabilities) -> dict[str, float]:
        """Return only parameters that are actually supported and effective."""
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
    """Build an updated parameter set while preserving unsupported dimensions."""
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
    """Convert optional scalar to float when present."""
    if value is None:
        return None
    return float(value)

