"""编码工具 用于 该 任务-分配 问题."""

from __future__ import annotations

import random
from typing import TypeAlias

Assignment: TypeAlias = list[int]


def random_assignment(n_tasks: int, n_resources: int, rng: random.Random | None = None) -> Assignment:
    """创建随机的任务到资源分配.
    
        每个任务都会被分配到 ``[0, n_resources)`` 中恰好一个资源索引。
    
        参数：
            n_tasks: Number 的 任务 到 assign.
            n_resources: Number 的 可用 资源.
            rng: Optional 随机 generator 用于 reproducibility.
    
        返回：
            A list where position ``i`` 为 该 资源 索引 用于 任务 ``i``.
    
        异常：
            ValueError: If ``n_tasks`` 为 negative 或 ``n_resources`` 为 不 positive.
        """
    if n_tasks < 0:
        raise ValueError("n_tasks must be >= 0")
    if n_resources <= 0:
        raise ValueError("n_resources must be > 0")

    generator = rng or random.Random()
    return [generator.randrange(n_resources) for _ in range(n_tasks)]
