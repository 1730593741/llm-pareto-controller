# TASKS.md

## 1. 文档用途

本文件用于把 `SPEC.md` 中定义的研究项目拆解成一系列可以交给 Codex 执行的开发任务。

目标不是让 Codex 一次性“写完整个项目”，而是让它按里程碑分阶段构建：

1. 先搭骨架；
2. 再做最小可运行版本（MVP）；
3. 再接闭环控制；
4. 再接经验池与真实 LLM；
5. 最后补实验系统、测试和重构。

---

## 2. 使用原则

使用 Codex 时请遵守以下原则：

### 2.1 一次只做一个里程碑或一个子任务
不要让 Codex 同时做太多模块，否则容易：

- 目录结构混乱；
- 接口不一致；
- 重复代码增多；
- 主循环跑不通。

### 2.2 每次任务都要明确 4 件事
每次给 Codex 下任务时，都应说明：

1. 目标是什么；
2. 修改哪些文件；
3. 不允许做什么；
4. 如何验收。

### 2.3 先跑通，再优化
优先保证：

- 主程序可运行；
- 模块接口清晰；
- 日志可输出；
- 测试能执行。

不要在前期过度追求：

- 高性能；
- 复杂抽象；
- 真实 API 深度集成；
- 大规模实验功能。

### 2.4 不要让 prompt 散落在代码里
LLM 提示词统一放到 `llm/prompts/` 或独立模板模块，不要直接硬写进主业务逻辑。

### 2.5 保持 mock 可运行
即使没有真实 API Key，整个系统也必须能用：

- `rule` 模式；
- `mock_llm` 模式；

完成最小闭环。

---

## 3. 任务总览

本项目按以下里程碑推进：

- `M0`：工程骨架
- `M1`：问题层 MVP
- `M2`：优化器 MVP
- `M3`：状态感知 MVP
- `M4`：规则控制闭环 MVP
- `M5`：经验池 MVP
- `M6`：真实 LLM 接入
- `M7`：实验系统与对照/消融
- `M8`：测试补全与工程化整理

---

## 4. M0：工程骨架

### 4.1 目标
创建项目目录、基础文件、空模块和运行入口，为后续开发提供统一结构。

### 4.2 需要创建/修改的文件

```text
project/
├─ problems/
│  └─ task_assignment/
├─ optimizers/
│  └─ nsga2/
├─ sensing/
├─ llm/
│  └─ prompts/
├─ memory/
├─ controller/
├─ infra/
├─ experiments/
│  ├─ configs/
│  ├─ baselines/
│  ├─ logs/
│  └─ ablations/
├─ tests/
├─ main.py
├─ README.md
├─ SPEC.md
├─ TASKS.md
├─ pyproject.toml
├─ requirements.txt
└─ .gitignore
```

同时创建必要的 `__init__.py`。

### 4.3 要求
1. 每个 Python 文件先写模块级 docstring；
2. 不要急着写复杂逻辑；
3. `main.py` 至少能打印“project bootstrapped”；
4. 建立最基本的 Python 包结构；
5. 在 `README.md` 中写清最基本的运行方式。

### 4.4 验收标准
- 项目目录完整；
- 所有基础模块存在；
- `python main.py` 能运行；
- 不报 import 错误。

### 4.5 给 Codex 的任务指令模板

```text
请根据 SPEC.md 为该项目创建工程骨架。

要求：
1. 创建以下目录与文件：
   - problems/task_assignment
   - optimizers/nsga2
   - sensing
   - llm/prompts
   - memory
   - controller
   - infra
   - experiments/configs
   - experiments/baselines
   - experiments/logs
   - experiments/ablations
   - tests
   - main.py
   - pyproject.toml
   - requirements.txt
   - README.md
   - .gitignore

2. 为所有 Python 模块加入 module-level docstring。
3. 暂时不要实现复杂逻辑。
4. main.py 至少可以运行并打印一条启动信息。
5. 所有代码使用 Python 3.11 风格和类型注解。

请先列出将创建的文件，再给出代码。
```

---

## 5. M1：问题层 MVP

### 5.1 目标
实现一个简化版多目标任务分配问题，为优化器提供可评价的候选解空间。

### 5.2 需要实现的模块

- `problems/task_assignment/encoding.py`
- `problems/task_assignment/objectives.py`
- `problems/task_assignment/constraints.py`
- `problems/task_assignment/repair.py`

### 5.3 最小问题设定建议
先做一个简化问题：

- 有 `n_tasks` 个任务；
- 有 `n_resources` 个资源；
- 每个任务必须被分配给一个资源；
- 每个资源有容量上限；
- 每个任务有成本和负载需求；
- 优化两个目标：
  - 总成本最小；
  - 负载不均衡最小。

### 5.4 具体要求

#### `encoding.py`
- 定义解的表示方式；
- 建议用 `list[int]` 或 `numpy.ndarray`；
- 提供初始化一个合法或近似合法解的函数。

#### `objectives.py`
- 提供目标函数计算；
- 返回二维目标值，如 `[cost, imbalance]`。

#### `constraints.py`
- 提供可行性判断；
- 提供约束违反度计算。

#### `repair.py`
- 当某个资源超容量时，尝试把任务重新分配到其他资源；
- 保证修复函数可重复调用。

### 5.5 验收标准
- 能生成初始解；
- 能计算目标；
- 能检查约束；
- 能修复明显不合法的解；
- 至少有基础单元测试。

### 5.6 给 Codex 的任务指令模板

```text
请实现 problems/task_assignment 下的 MVP 问题层。

要求：
1. encoding.py
   - 定义任务分配解的表示方式
   - 提供随机初始化函数

2. objectives.py
   - 计算两个目标：
     a. 总成本
     b. 负载不均衡

3. constraints.py
   - 检查容量约束
   - 计算约束违反度

4. repair.py
   - 对超容量解执行简单修复

要求：
- 接口清晰
- 使用类型注解
- 写 docstring
- 同时补基础 pytest 测试
- 先做简化实现，保证可运行

请先列出将修改/新增的文件，再给出代码。
```

---

## 6. M2：优化器 MVP

### 6.1 目标
实现一个最小可运行的 NSGA-II，使系统能够进行多代迭代搜索。

### 6.2 需要实现的模块

- `optimizers/nsga2/population.py`
- `optimizers/nsga2/operators.py`
- `optimizers/nsga2/selection.py`
- `optimizers/nsga2/solver.py`

### 6.3 具体要求

#### `population.py`
- 定义 `Individual`；
- 定义 `Population`；
- 个体至少包含：
  - genome
  - objectives
  - constraint_violation
  - feasible
  - rank
  - crowding_distance

#### `operators.py`
- 实现交叉；
- 实现变异；
- 交叉、变异都要兼容任务分配编码；
- 交叉率和变异率必须可传参。

#### `selection.py`
- 实现非支配排序；
- 实现拥挤距离；
- 实现父代选择和环境选择；
- 允许简化实现，但逻辑要正确。

#### `solver.py`
- 初始化种群；
- 评价；
- 父代选择；
- 交叉、变异、修复；
- 合并父子代；
- 完成多代迭代。

### 6.4 验收标准
- 能基于问题实例运行至少 20 代；
- 每代都能输出群体规模和最优信息；
- 主流程不崩溃；
- 可以切换交叉率和变异率。

### 6.5 给 Codex 的任务指令模板

```text
请实现 NSGA-II 的最小可运行版本。

要求：
1. population.py
   - Individual / Population 数据结构
2. operators.py
   - 交叉与变异
3. selection.py
   - 非支配排序
   - 拥挤距离
   - 选择逻辑
4. solver.py
   - 完整多代迭代主流程

约束：
- 先保证逻辑正确和可运行
- 不追求性能最优
- 与 problems/task_assignment 接口打通
- 保留后续控制器可注入的接口
- 添加基本测试

请先列出要修改/新增的文件。
```

---

## 7. M3：状态感知 MVP

### 7.1 目标
把“当前种群状态”提取成结构化的 Pareto-State，为后续控制器提供输入。

### 7.2 需要实现的模块

- `sensing/pareto_state.py`
- `sensing/hypervolume.py`
- `sensing/feasibility_metrics.py`

### 7.3 MVP 期建议先做的状态指标

- `generation`
- `feasible_ratio`
- `rank1_ratio`
- `mean_cv`
- `diversity_score`
- `stagnation_len`
- `hv`（可先做简化版）
- `delta_hv`

### 7.4 具体要求

#### `feasibility_metrics.py`
- 计算可行解比例；
- 计算平均约束违反度；
- 计算最大约束违反度。

#### `hypervolume.py`
- 可以先写一个占位版或简化版 HV 接口；
- 接口保持后续可替换。

#### `pareto_state.py`
- 根据当前种群输出 `ParetoState`；
- 允许依赖上一代状态以计算 `delta_hv` 和 `stagnation_len`。

### 7.5 验收标准
- 每代都能生成一个结构化状态对象；
- 状态可序列化；
- 可被日志系统记录；
- 可以供控制器读取。

### 7.6 给 Codex 的任务指令模板

```text
请实现状态感知模块 sensing/。

要求：
1. feasibility_metrics.py
   - feasible_ratio
   - mean_constraint_violation
   - max_constraint_violation

2. hypervolume.py
   - 定义 HV 计算接口
   - 先做简化实现或占位实现，但接口必须稳定

3. pareto_state.py
   - 输出 ParetoState 数据结构
   - 包含 generation, hv, delta_hv, feasible_ratio,
     rank1_ratio, mean_cv, diversity_score, stagnation_len

要求：
- 代码清晰
- 类型注解完整
- 可与 solver 输出的种群打通
- 补基础测试
```

---

## 8. M4：规则控制闭环 MVP

### 8.1 目标
先不用真实 LLM，而是实现一个规则控制器，把“优化器—状态提取—参数更新”闭环跑通。

### 8.2 需要实现的模块

- `controller/closed_loop.py`
- `main.py`
- 必要时增加一个简单规则控制器模块

### 8.3 控制逻辑建议
每 `k` 代执行一次控制：

- 如果 `stagnation_len` 增长，则提高 `mutation_prob`；
- 如果 `feasible_ratio` 很低，则提高 `repair_strength`；
- 如果 `diversity_score` 太低，则提高探索力度；
- 如果 HV 稳定提升，则略提高利用倾向。

### 8.4 具体要求
- 控制动作至少包括：
  - `mutation_prob`
  - `crossover_prob`
- 动作必须记录到日志中；
- `main.py` 必须能跑完整个流程；
- 没有 API Key 也能运行。

### 8.5 验收标准
- 能看到参数在运行过程中发生变化；
- 每隔若干代更新一次参数；
- 运行日志包含 state 和 action；
- 输出结果目录中有结构化日志文件。

### 8.6 给 Codex 的任务指令模板

```text
请实现规则控制版闭环优化流程。

要求：
1. controller/closed_loop.py
   - 每隔 k 代读取 ParetoState
   - 根据规则输出新的 mutation_prob 和 crossover_prob
   - 将动作应用到下一阶段优化中

2. main.py
   - 加载默认配置
   - 创建问题、优化器、控制器
   - 启动闭环运行
   - 输出每代日志

要求：
- 不依赖真实 LLM
- 系统必须能独立运行
- 动作和状态都要保存
- 保留未来切换到 mock_llm / real_llm 的接口
```

---

## 9. M5：经验池 MVP

### 9.1 目标
引入历史经验记录，使控制器具备最基本的“记忆”能力。

### 9.2 需要实现的模块

- `memory/experience_pool.py`
- `infra/storage.py`

### 9.3 经验记录最小结构
每条经验至少包含：

- `state`
- `action`
- `reward`
- `next_state`

### 9.4 具体要求

#### `experience_pool.py`
- 保存最近 `k` 条经验；
- 支持 append；
- 支持滑动窗口裁剪；
- 支持检索最近若干条经验；
- 后续可扩展相似状态检索。

#### `storage.py`
- 能保存 jsonl/csv；
- 记录每代状态；
- 记录控制动作；
- 记录经验池快照或实验摘要。

### 9.5 奖励建议
初期可用简单奖励：

```text
reward = delta_hv + alpha * delta_feasible_ratio - beta * mean_cv
```

### 9.6 验收标准
- 每次控制后能写入经验；
- 经验池窗口大小受配置控制；
- 可从日志中追踪 state-action-reward 链条。

### 9.7 给 Codex 的任务指令模板

```text
请为项目增加经验池 MVP。

要求：
1. memory/experience_pool.py
   - 定义 ExperienceRecord
   - 支持 append
   - 支持滑动窗口
   - 支持读取最近 n 条经验

2. infra/storage.py
   - 保存逐代日志到 jsonl/csv
   - 保存经验记录

3. controller/closed_loop.py
   - 每次产生控制动作后，计算 reward
   - 将 state/action/reward/next_state 写入经验池

要求：
- 先做轻量存储
- 不接数据库
- 保持 mock 模式可运行
```

---

## 10. M6：真实 LLM 接入

### 10.1 目标
在不破坏 MVP 的前提下，将规则控制器升级为可替换的 LLM 控制器。

### 10.2 需要实现的模块

- `infra/llm_client.py`
- `llm/analyst.py`
- `llm/strategist.py`
- `llm/actuator.py`
- `llm/prompts/` 下的模板文件
- 修改 `controller/closed_loop.py`

### 10.3 设计要求
必须支持三种模式：

1. `rule`
2. `mock_llm`
3. `real_llm`

### 10.4 具体要求

#### `infra/llm_client.py`
- 提供统一接口；
- 支持 mock 响应；
- 支持真实 HTTP 调用；
- 支持超时、重试、异常保护；
- API Key 从环境变量读取。

#### `llm/analyst.py`
输入：
- 当前 `ParetoState`
- 最近经验

输出：
- 局面分析文本或结构化诊断

#### `llm/strategist.py`
输入：
- analyst 输出

输出：
- 高层策略建议，如：
  - increase exploration
  - improve feasibility
  - stabilize exploitation

#### `llm/actuator.py`
输入：
- strategist 输出

输出：
- 结构化 `ControlAction`

#### `llm/prompts/`
保存：
- analyst prompt
- strategist prompt
- actuator prompt
- JSON 输出格式要求

### 10.5 验收标准
- 没有 API Key 时 mock 模式可运行；
- real_llm 模式下接口清晰；
- action 输出有结构化校验；
- prompt 不散落在业务代码中。

### 10.6 给 Codex 的任务指令模板

```text
请为项目接入 LLM 控制层，但保持可替换设计。

要求：
1. infra/llm_client.py
   - 抽象统一 LLM client
   - 支持 rule / mock_llm / real_llm 三种模式
   - API Key 不能写死

2. llm/analyst.py
   - 基于 state 和 recent experiences 生成诊断

3. llm/strategist.py
   - 基于诊断输出高层策略

4. llm/actuator.py
   - 基于策略输出 ControlAction

5. llm/prompts/
   - 提供模板，不要把 prompt 硬编码在业务模块里

6. controller/closed_loop.py
   - 接入上述控制链

要求：
- 优先保证 mock_llm 可运行
- 所有输出都应结构化
- 使用 pydantic 或 dataclass 做校验
```

---

## 11. M7：实验系统、基线与消融

### 11.1 目标
让项目具备研究实验所需的对比和复现能力。

### 11.2 需要实现的模块

- `experiments/configs/`
- `experiments/baselines/`
- `experiments/ablations/`
- 必要的绘图或统计脚本

### 11.3 具体要求

#### `experiments/configs/`
提供默认配置文件，例如：

- `default.yaml`
- `baseline_nsga2.yaml`
- `rule_control.yaml`
- `mock_llm.yaml`
- `real_llm.yaml`

#### `experiments/baselines/`
至少提供：

- 标准 NSGA-II baseline；
- 规则控制 baseline；
- 不带经验池 baseline。

#### `experiments/ablations/`
至少提供以下消融开关：

- `no_memory`
- `no_strategist`
- `no_pareto_state`

#### 可视化建议
至少支持绘制：

- HV 曲线；
- 可行率曲线；
- 参数变化曲线；
- reward 曲线。

### 11.4 验收标准
- 可以通过配置运行不同模式；
- 可以输出不同实验结果；
- 可以做最基本的对照与消融。

### 11.5 给 Codex 的任务指令模板

```text
请为项目增加实验系统。

要求：
1. experiments/configs
   - 添加 default.yaml、baseline 配置、mock_llm 配置等

2. experiments/baselines
   - 标准 NSGA-II
   - 规则控制版
   - 无经验池版

3. experiments/ablations
   - no_memory
   - no_strategist
   - no_pareto_state

4. 增加绘图或统计脚本
   - HV 曲线
   - feasible_ratio 曲线
   - 参数变化曲线

5. main.py
   - 支持通过命令行加载配置和模式

要求：
- 以可复现实验为目标
- 配置优先
- 不要把实验逻辑塞进单个超大文件
```

---

## 12. M8：测试补全与工程化整理

### 12.1 目标
对已有代码做清理、补测试、补文档，使项目进入稳定迭代状态。

### 12.2 需要做的工作

#### 代码整理
- 移除重复逻辑；
- 抽取公共函数；
- 修复循环依赖；
- 清理无用模块；
- 统一命名风格。

#### 测试补全
必须补齐：

- `tests/test_objectives.py`
- `tests/test_constraints.py`
- `tests/test_repair.py`
- `tests/test_selection.py`
- `tests/test_pareto_state.py`
- `tests/test_experience_pool.py`
- `tests/test_closed_loop.py`

#### 文档补全
- `README.md`
- 配置说明
- 运行说明
- 实验说明
- 环境说明

### 12.3 验收标准
- 默认配置可直接运行；
- `pytest` 能执行；
- 项目无明显重复代码；
- 文档能指导他人运行。

### 12.4 给 Codex 的任务指令模板

```text
请对当前项目做一次工程化整理。

要求：
1. 清理重复代码和无用逻辑
2. 修复可能的循环依赖
3. 统一命名和类型注解
4. 补齐 pytest 测试：
   - objectives
   - constraints
   - repair
   - selection
   - pareto_state
   - experience_pool
   - closed_loop
5. 更新 README.md：
   - 安装方式
   - 运行方式
   - 配置说明
   - 项目结构说明

要求：
- 不要重写整个系统
- 尽量小步重构
- 在给出代码前先列出修改清单
```

---

## 13. 推荐执行顺序

Codex 施工时建议严格按下列顺序推进：

1. `M0` 工程骨架
2. `M1` 问题层
3. `M2` NSGA-II
4. `M3` 状态感知
5. `M4` 规则闭环
6. `M5` 经验池
7. `M6` 真实 LLM
8. `M7` 实验系统
9. `M8` 测试与重构

不要跳步骤。  
特别是不要在 `M2` 之前就开始实现真实 LLM 或复杂实验系统。

---

## 14. 建议的每轮协作模板

每次把任务发给 Codex 时，建议用下面这个模板：

```text
当前任务：<填写本轮任务名>

项目背景：
这是一个“LLM 闭环调控 NSGA-II 的多目标任务分配优化系统”研究项目。
请遵守 SPEC.md 和 TASKS.md 的约束。

本轮目标：
<写本轮目标>

需要修改/新增的文件：
- xxx.py
- yyy.py

明确要求：
1. ...
2. ...
3. ...

禁止事项：
1. 不要重写已有模块
2. 不要把多个职责塞进同一个大文件
3. 不要引入未说明的重量级依赖
4. 不要写死 API Key 或路径

交付要求：
1. 先列出修改计划
2. 再给出代码
3. 最后说明如何运行和测试
```

---

## 15. 第一轮到第四轮的建议实际发包顺序

### 第 1 轮
只做 `M0`，搭骨架。

### 第 2 轮
只做 `M1`，问题层。

### 第 3 轮
只做 `M2`，NSGA-II 主流程。

### 第 4 轮
只做 `M3 + M4`，状态提取和规则闭环。

到这里，你应该已经有一个：

- 可运行；
- 可输出日志；
- 参数会变化；
- 不依赖真实 LLM；

的研究原型。

---

## 16. 常见错误提醒

### 错误 1：一开始就要求 Codex 写完整项目
这通常会导致代码表面完整、内部混乱。

### 错误 2：太早引入真实数据库
当前阶段更适合：

- 本地 JSONL
- CSV
- YAML 配置

### 错误 3：优化器和控制器耦合过深
优化器应尽量只暴露参数接口，控制器不要侵入其内部细节。

### 错误 4：把 prompt 直接写进业务代码
后期难以管理和做消融。

### 错误 5：没有测试就快速迭代
后期一改状态指标或动作空间，主流程很容易崩。

---

## 17. 最小完成定义（Definition of Done）

当满足以下条件时，当前阶段可认为“初步完成”：

1. `main.py` 能用默认配置运行；
2. 能完成至少 20 代优化；
3. 能提取 Pareto-State；
4. 能应用规则或 mock 控制动作；
5. 能记录每代状态和动作；
6. 能输出日志文件；
7. 核心模块有基础测试。

---

## 18. 项目进入下一阶段的判断标准

当以下条件成立时，才建议开始真实 LLM 接入：

- MVP 已稳定运行；
- 规则控制闭环已经跑通；
- 状态提取指标已经固定一版；
- 日志与经验池已可用；
- 问题定义与优化器接口不再频繁变化。

---

## 19. 一句话总结

`TASKS.md` 的核心作用是：**把你的研究项目从“一个想法”变成 Codex 可以逐步施工的工程任务单。**

先骨架，后闭环；先规则，后 LLM；先跑通，后论文化。