"""Closed-loop controller orchestration with pluggable control policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from memory.experience_pool import ExperiencePool, ExperienceRecord
from optimizers.nsga2.solver import NSGA2Solver
from sensing.pareto_state import ParetoState, ParetoStateSensor


class EventLogger(Protocol):
    """Protocol for storage backends used by the closed-loop runner."""

    def log(self, event: dict[str, Any]) -> None:
        """Persist a single structured event."""


class ExperienceLogger(Protocol):
    """Protocol for optional experience persistence backend."""

    def log(self, record: ExperienceRecord) -> None:
        """Persist a single experience record."""


@dataclass(slots=True)
class RuleControllerConfig:
    """Thresholds and step sizes for the M4 rule policy."""

    control_interval: int = 5
    min_mutation_prob: float = 0.02
    max_mutation_prob: float = 0.8
    min_crossover_prob: float = 0.4
    max_crossover_prob: float = 0.98
    mutation_step: float = 0.05
    crossover_step: float = 0.04
    feasible_ratio_low: float = 0.55
    diversity_low: float = 0.08
    improvement_threshold: float = 1e-3

    def __post_init__(self) -> None:
        """Validate basic controller-rule bounds for safe runtime use."""
        if self.control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        if not 0.0 <= self.min_mutation_prob <= self.max_mutation_prob <= 1.0:
            raise ValueError("mutation probability bounds must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_crossover_prob <= self.max_crossover_prob <= 1.0:
            raise ValueError("crossover probability bounds must satisfy 0 <= min <= max <= 1")


@dataclass(slots=True)
class RewardConfig:
    """Reward weighting used by M5 experience collection."""

    alpha: float = 1.0
    beta: float = 0.1


@dataclass(slots=True)
class OperatorParams:
    """Current optimizer operator probabilities."""

    mutation_prob: float
    crossover_prob: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ControlAction:
    """Action chosen by controller for operator probabilities."""

    generation: int
    mutation_prob: float
    crossover_prob: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize action for logs."""
        return asdict(self)


class ControlPolicy(Protocol):
    """Shared interface for rule and LLM-based controllers."""

    control_interval: int

    def decide(
        self,
        *,
        state: ParetoState,
        recent_experiences: list[ExperienceRecord],
        current_params: OperatorParams,
    ) -> ControlAction:
        """Return the next control action."""


class RuleBasedController:
    """Simple heuristic policy; extension point for M6 LLM controllers."""

    def __init__(self, config: RuleControllerConfig) -> None:
        self.config = config
        self.control_interval = config.control_interval

    def decide(
        self,
        *,
        state: ParetoState,
        recent_experiences: list[ExperienceRecord] | None = None,
        current_params: OperatorParams | None = None,
        current_mutation: float | None = None,
        current_crossover: float | None = None,
    ) -> ControlAction:
        """Decide new operator parameters from sensed Pareto state."""
        del recent_experiences
        if current_params is not None:
            mutation = current_params.mutation_prob
            crossover = current_params.crossover_prob
        elif current_mutation is not None and current_crossover is not None:
            mutation = current_mutation
            crossover = current_crossover
        else:
            msg = "Either current_params or both current_mutation/current_crossover must be provided"
            raise ValueError(msg)
        reasons: list[str] = []

        if state.stagnation_len > 0:
            mutation += self.config.mutation_step
            crossover -= self.config.crossover_step * 0.5
            reasons.append("stagnation_up_mutation")

        if state.feasible_ratio < self.config.feasible_ratio_low:
            mutation -= self.config.mutation_step * 0.5
            crossover += self.config.crossover_step
            reasons.append("low_feasible_safer_search")

        if state.diversity_score < self.config.diversity_low:
            mutation += self.config.mutation_step
            crossover -= self.config.crossover_step
            reasons.append("low_diversity_more_exploration")

        if state.delta_hv > self.config.improvement_threshold and state.stagnation_len == 0:
            mutation -= self.config.mutation_step * 0.5
            crossover += self.config.crossover_step * 0.5
            reasons.append("stable_improvement_reduce_exploration")

        mutation = _clip(mutation, self.config.min_mutation_prob, self.config.max_mutation_prob)
        crossover = _clip(crossover, self.config.min_crossover_prob, self.config.max_crossover_prob)

        reason = ";".join(reasons) if reasons else "hold"
        return ControlAction(
            generation=state.generation,
            mutation_prob=mutation,
            crossover_prob=crossover,
            reason=reason,
        )


class LLMChainController:
    """Composable controller using analyst -> strategist -> actuator chain."""

    def __init__(
        self,
        *,
        control_interval: int,
        experience_lookback: int,
        analyst: Any,
        strategist: Any,
        actuator: Any,
    ) -> None:
        if control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        self.control_interval = control_interval
        self.experience_lookback = max(1, experience_lookback)
        self.analyst = analyst
        self.strategist = strategist
        self.actuator = actuator

    def decide(
        self,
        *,
        state: ParetoState,
        recent_experiences: list[ExperienceRecord],
        current_params: OperatorParams,
    ) -> ControlAction:
        diagnosis = self.analyst.analyze(state=state, recent_experiences=recent_experiences)
        strategy = self.strategist.plan(diagnosis)
        return self.actuator.act(generation=state.generation, strategy=strategy, current_params=current_params)


@dataclass(slots=True)
class _PendingExperience:
    """Internal transition holder until next state becomes available."""

    state: ParetoState
    action: ControlAction


class ClosedLoopRunner:
    """Coordinate optimizer, sensing, control policy, and lightweight logging."""

    def __init__(
        self,
        *,
        solver: NSGA2Solver,
        sensor: ParetoStateSensor,
        controller: ControlPolicy,
        logger: EventLogger | None = None,
        experience_pool: ExperiencePool | None = None,
        experience_logger: ExperienceLogger | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.solver = solver
        self.sensor = sensor
        self.controller = controller
        self.logger = logger
        self.experience_pool = experience_pool
        self.experience_logger = experience_logger
        self.reward_config = reward_config or RewardConfig()

    def run(self, *, generations: int, reference_point: tuple[float, ...] | None = None) -> list[ParetoState]:
        """Execute closed-loop optimization and return sensed states."""
        if generations < 0:
            raise ValueError("generations must be >= 0")
        population = self.solver.initialize_population()
        previous_state: ParetoState | None = None
        states: list[ParetoState] = []
        pending_experience: _PendingExperience | None = None

        initial_state = self.sensor.sense(
            generation=0,
            population=population,
            previous_state=None,
            reference_point=reference_point,
        )
        states.append(initial_state)
        self._log_state(initial_state, self.solver.config.mutation_prob, self.solver.config.crossover_prob)
        previous_state = initial_state

        for generation in range(1, generations + 1):
            population = self.solver.evolve_one_generation(population)
            state = self.sensor.sense(
                generation=generation,
                population=population,
                previous_state=previous_state,
                reference_point=reference_point,
            )
            states.append(state)
            self._log_state(state, self.solver.config.mutation_prob, self.solver.config.crossover_prob)

            if pending_experience is not None:
                self._finalize_experience(pending_experience, state)
                pending_experience = None

            if generation % self.controller.control_interval == 0 and generation < generations:
                recent_experiences = self._recent_experiences()
                action = self.controller.decide(
                    state=state,
                    recent_experiences=recent_experiences,
                    current_params=OperatorParams(
                        mutation_prob=self.solver.config.mutation_prob,
                        crossover_prob=self.solver.config.crossover_prob,
                    ),
                )
                self.solver.set_operator_probs(
                    mutation_prob=action.mutation_prob,
                    crossover_prob=action.crossover_prob,
                )
                self._log_action(action)
                pending_experience = _PendingExperience(state=state, action=action)

            previous_state = state

        return states

    def _recent_experiences(self) -> list[ExperienceRecord]:
        if self.experience_pool is None:
            return []
        lookback = getattr(self.controller, "experience_lookback", 5)
        return self.experience_pool.recent(int(lookback))

    def _log_state(self, state: ParetoState, mutation_prob: float, crossover_prob: float) -> None:
        if self.logger is None:
            return
        self.logger.log(
            {
                "event": "state",
                **state.to_dict(),
                "mutation_prob": mutation_prob,
                "crossover_prob": crossover_prob,
            }
        )

    def _log_action(self, action: ControlAction) -> None:
        if self.logger is None:
            return
        self.logger.log({"event": "action", **action.to_dict()})

    def _finalize_experience(self, pending: _PendingExperience, next_state: ParetoState) -> None:
        if self.experience_pool is None and self.experience_logger is None:
            return

        reward = compute_reward(state=pending.state, next_state=next_state, config=self.reward_config)
        record = ExperienceRecord(
            state=pending.state.to_dict(),
            action=pending.action.to_dict(),
            reward=reward,
            next_state=next_state.to_dict(),
        )
        if self.experience_pool is not None:
            self.experience_pool.append(record)
        if self.experience_logger is not None:
            self.experience_logger.log(record)


def compute_reward(*, state: ParetoState, next_state: ParetoState, config: RewardConfig) -> float:
    """Compute M5 lightweight reward from adjacent states."""
    delta_hv = next_state.hv - state.hv
    delta_feasible_ratio = next_state.feasible_ratio - state.feasible_ratio
    return delta_hv + config.alpha * delta_feasible_ratio - config.beta * next_state.mean_cv


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
