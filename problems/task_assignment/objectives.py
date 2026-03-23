"""目标函数 用于 该 任务-分配 问题."""

from __future__ import annotations

from typing import Sequence

from problems.task_assignment.encoding import Assignment


def total_cost(assignment: Assignment, cost_matrix: Sequence[Sequence[float]]) -> float:
    """计算任务分配总成本.
    
        参数：
            分配: Resource 索引 用于 each 任务.
            cost_matrix: 2D 矩阵 where ``cost_matrix[任务][资源]`` 为 成本.
    
        返回：
            Sum 的 selected 任务-资源 costs.
        """
    return float(sum(cost_matrix[task_idx][resource_idx] for task_idx, resource_idx in enumerate(assignment)))


def load_imbalance(assignment: Assignment, task_loads: Sequence[float], n_resources: int) -> float:
    """计算负载不均衡，定义为 ``max_resource_load - min_resource_load``.
    
        参数：
            分配: Resource 索引 用于 each 任务.
            task_loads: Load demand 用于 each 任务.
            n_resources: Number 的 资源.
    
        返回：
            Difference between max 与 min 聚合 资源 loads.
        """
    loads = [0.0] * n_resources
    for task_idx, resource_idx in enumerate(assignment):
        loads[resource_idx] += float(task_loads[task_idx])

    return max(loads) - min(loads) if loads else 0.0


def compute_objectives(
    assignment: Assignment,
    cost_matrix: Sequence[Sequence[float]],
    task_loads: Sequence[float],
    n_resources: int,
) -> tuple[float, float]:
    """计算两个优化目标: total 成本 与 负载 imbalance."""
    return total_cost(assignment, cost_matrix), load_imbalance(assignment, task_loads, n_resources)
