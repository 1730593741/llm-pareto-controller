"""闭环 控制器 orchestration，并带有 pluggable 控制 策略."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Protocol

from controller.control_semantics import ControlState
from controller.operator_space import OperatorCapabilities, OperatorParams
from memory.experience_pool import ExperiencePool, ExperienceRecord
from optimizers.nsga2.solver import NSGA2Solver
from sensing.pareto_state import ParetoState, ParetoStateSensor


class EventLogger(Protocol):
    """协议 用于 存储后端 用于 该 闭环 运行器."""

    def log(self, event: dict[str, Any]) -> None:
        """持久化 一个 单个 结构化 事件."""


class ExperienceLogger(Protocol):
    """协议 用于 可选的 经验持久化后端."""

    def log(self, record: ExperienceRecord) -> None:
        """持久化 一个 单个 experience 记录."""


@dataclass(slots=True)
class RuleControllerConfig:
    """阈值与步长 用于 该 M4 规则 策略."""

    control_interval: int = 5
    min_mutation_prob: float = 0.02
    max_mutation_prob: float = 0.8
    min_crossover_prob: float = 0.4
    max_crossover_prob: float = 0.98
    min_eta_c: float = 5.0
    max_eta_c: float = 40.0
    min_eta_m: float = 5.0
    max_eta_m: float = 80.0
    min_repair_prob: float = 0.0
    max_repair_prob: float = 1.0
    min_local_search_prob: float = 0.0
    max_local_search_prob: float = 1.0
    mutation_step: float = 0.05
    crossover_step: float = 0.04
    eta_c_step: float = 2.0
    eta_m_step: float = 4.0
    repair_step: float = 0.08
    local_search_step: float = 0.05
    feasible_ratio_low: float = 0.55
    diversity_low: float = 0.08
    improvement_threshold: float = 1e-3

    def __post_init__(self) -> None:
        """校验 基础 控制器-规则 bounds 用于 安全 运行时 使用."""
        if self.control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        if not 0.0 <= self.min_mutation_prob <= self.max_mutation_prob <= 1.0:
            raise ValueError("mutation probability bounds must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_crossover_prob <= self.max_crossover_prob <= 1.0:
            raise ValueError("crossover probability bounds must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_repair_prob <= self.max_repair_prob <= 1.0:
            raise ValueError("repair probability bounds must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_local_search_prob <= self.max_local_search_prob <= 1.0:
            raise ValueError("local-search probability bounds must satisfy 0 <= min <= max <= 1")


@dataclass(slots=True)
class RewardConfig:
    """奖励加权 用于 M5 经验采集."""

    alpha: float = 1.0
    beta: float = 0.1


@dataclass(slots=True)
class ControlAction:
    """由 控制器 选择的算子概率动作."""

    generation: int
    mutation_prob: float
    crossover_prob: float
    reason: str
    requested_params: dict[str, float | None] | None = None
    applied_params: dict[str, float] | None = None
    capabilities: dict[str, bool] | None = None
    control_state: ControlState = ControlState.MAINTAIN_BALANCE
    reason_detail: str = ""
    decision_runtime_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化 动作 用于 日志."""
        return asdict(self)


class ControlPolicy(Protocol):
    """共享接口 用于 规则控制器与基于 LLM 的控制器."""

    control_interval: int

    def decide(
        self,
        *,
        state: ParetoState,
        recent_experiences: list[ExperienceRecord],
        current_params: OperatorParams,
        capabilities: OperatorCapabilities,
    ) -> ControlAction:
        """返回 该 下一步 控制 动作."""


class RuleBasedController:
    """简单启发式策略; 扩展点 用于 M6 LLM 控制器."""

    def __init__(self, config: RuleControllerConfig) -> None:
        self.config = config
        self.control_interval = config.control_interval

    def decide(
        self,
        *,
        state: ParetoState,
        recent_experiences: list[ExperienceRecord] | None = None,
        current_params: OperatorParams | None = None,
        capabilities: OperatorCapabilities | None = None,
        current_mutation: float | None = None,
        current_crossover: float | None = None,
    ) -> ControlAction:
        """根据 new 算子 parameters 从 sensed Pareto 状态."""
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
        capabilities = capabilities or OperatorCapabilities()
        control_state = ControlState.MAINTAIN_BALANCE
        reason_details: list[str] = []

        if state.feasible_ratio < self.config.feasible_ratio_low:
            control_state = ControlState.INCREASE_FEASIBILITY
            mutation -= self.config.mutation_step * 0.5
            crossover += self.config.crossover_step
            reason_details.append("feasible_ratio_below_threshold")
        elif state.diversity_score < self.config.diversity_low:
            control_state = ControlState.INCREASE_DIVERSITY
            mutation += self.config.mutation_step
            crossover -= self.config.crossover_step
            reason_details.append("diversity_score_below_threshold")
        elif state.stagnation_len > 0 or state.delta_hv <= self.config.improvement_threshold:
            control_state = ControlState.INCREASE_CONVERGENCE
            mutation -= self.config.mutation_step * 0.5
            crossover += self.config.crossover_step * 0.5
            reason_details.append("stagnation_or_low_hv_progress")
        else:
            control_state = ControlState.MAINTAIN_BALANCE
            reason_details.append("metrics_in_expected_range")

        mutation = _clip(mutation, self.config.min_mutation_prob, self.config.max_mutation_prob)
        crossover = _clip(crossover, self.config.min_crossover_prob, self.config.max_crossover_prob)

        eta_c = current_params.eta_c if current_params else None
        eta_m = current_params.eta_m if current_params else None
        repair_prob = current_params.repair_prob if current_params else None
        local_search_prob = current_params.local_search_prob if current_params else None

        if capabilities.supports_repair_prob and repair_prob is not None:
            if control_state == ControlState.INCREASE_FEASIBILITY:
                repair_prob += self.config.repair_step
                reason_details.append("raise_repair_for_feasibility")
            elif control_state == ControlState.INCREASE_DIVERSITY:
                repair_prob -= self.config.repair_step * 0.5
                reason_details.append("lower_repair_for_exploration")
            repair_prob = _clip(repair_prob, self.config.min_repair_prob, self.config.max_repair_prob)

        reason = control_state.value
        reason_detail = ";".join(reason_details)
        return ControlAction(
            generation=state.generation,
            mutation_prob=mutation,
            crossover_prob=crossover,
            reason=reason,
            requested_params=OperatorParams(
                mutation_prob=mutation,
                crossover_prob=crossover,
                eta_c=eta_c,
                eta_m=eta_m,
                repair_prob=repair_prob,
                local_search_prob=local_search_prob,
            ).to_dict(),
            applied_params=OperatorParams(
                mutation_prob=mutation,
                crossover_prob=crossover,
                eta_c=eta_c,
                eta_m=eta_m,
                repair_prob=repair_prob,
                local_search_prob=local_search_prob,
            ).active_params(capabilities),
            capabilities=capabilities.to_dict(),
            control_state=control_state,
            reason_detail=reason_detail,
        )


class LLMChainController:
    """可组合的 控制器 使用 analyst -> strategist -> actuator 链."""

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
        capabilities: OperatorCapabilities,
    ) -> ControlAction:
        diagnosis = self.analyst.analyze(state=state, recent_experiences=recent_experiences)
        strategy = self.strategist.plan(diagnosis)
        return self.actuator.act(
            generation=state.generation,
            strategy=strategy,
            current_params=current_params,
            capabilities=capabilities,
        )


@dataclass(slots=True)
class _PendingExperience:
    """在下一状态可用前保存内部转移的占位对象."""

    state: ParetoState
    action: ControlAction


class ClosedLoopRunner:
    """协调优化器、感知、控制策略与轻量日志."""

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
        """执行 闭环 优化 与 返回 sensed 状态."""
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
        self._log_state(initial_state, self.solver.get_operator_params())
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
            self._log_state(state, self.solver.get_operator_params())

            if pending_experience is not None:
                self._finalize_experience(pending_experience, state)
                pending_experience = None

            if generation % self.controller.control_interval == 0 and generation < generations:
                recent_experiences = self._recent_experiences()
                action_started = time.perf_counter()
                action = self.controller.decide(
                    state=state,
                    recent_experiences=recent_experiences,
                    current_params=self.solver.get_operator_params(),
                    capabilities=self.solver.get_operator_capabilities(),
                )
                action.decision_runtime_s = time.perf_counter() - action_started
                applied_params = action.applied_params or {}
                self.solver.set_operator_params(
                    OperatorParams(
                        mutation_prob=action.mutation_prob,
                        crossover_prob=action.crossover_prob,
                        eta_c=float(applied_params["eta_c"]) if "eta_c" in applied_params else None,
                        eta_m=float(applied_params["eta_m"]) if "eta_m" in applied_params else None,
                        repair_prob=float(applied_params["repair_prob"]) if "repair_prob" in applied_params else None,
                        local_search_prob=(
                            float(applied_params["local_search_prob"])
                            if "local_search_prob" in applied_params
                            else None
                        ),
                    )
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

    def _log_state(self, state: ParetoState, params: OperatorParams) -> None:
        if self.logger is None:
            return
        self.logger.log(
            {
                "event": "state",
                **state.to_dict(),
                "mutation_prob": params.mutation_prob,
                "crossover_prob": params.crossover_prob,
                "operator_params": params.to_dict(),
                "operator_capabilities": self.solver.get_operator_capabilities().to_dict(),
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
    """计算 M5 轻量 reward 从 adjacent 状态."""
    delta_hv = next_state.hv - state.hv
    delta_feasible_ratio = next_state.feasible_ratio - state.feasible_ratio
    return delta_hv + config.alpha * delta_feasible_ratio - config.beta * next_state.mean_cv


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
