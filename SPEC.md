\# SPEC.md



\## 1. 项目名称



\*\*LLM-Pareto-Controller\*\*



一个面向研究实验的 Python 项目，用于构建“\*\*LLM 闭环调控 NSGA-II 的多目标任务分配优化系统\*\*”。



---



\## 2. 项目背景



本项目的目标不是让大语言模型直接替代优化算法，而是让 LLM 作为\*\*高层控制器\*\*，在进化优化过程中动态观察搜索状态、分析瓶颈、给出参数调整策略，并驱动优化器持续迭代。



项目聚焦于以下研究思想：



1\. 用 \*\*NSGA-II\*\* 作为基础多目标优化器。

2\. 用 \*\*Pareto-State Sensing\*\* 将种群状态压缩成结构化特征。

3\. 用 \*\*LLM-CoR（Chain of Reasoning）\*\* 充当在线调参控制器。

4\. 用 \*\*Experience Pool\*\* 保存历史状态、动作和反馈，形成闭环学习与检索机制。

5\. 在“多目标任务分配问题”上完成最小可运行系统、基线比较和消融实验。



---



\## 3. 项目总目标



\### 3.1 核心目标



构建一个具备如下能力的研究型工程：



\- 能定义多目标任务分配问题；

\- 能运行一个可复现的 NSGA-II 优化过程；

\- 能提取 Pareto 前沿与搜索过程的状态特征；

\- 能由 LLM 根据当前状态和历史经验输出调参动作；

\- 能将该动作应用到下一轮优化；

\- 能保存实验过程并支持对照实验、消融实验和可视化分析。



\### 3.2 最小可运行目标（MVP）



第一阶段不追求完整论文级系统，而是先实现以下能力：



\- 一个简化任务分配问题；

\- 一个可运行的 NSGA-II；

\- 一个基础状态提取器；

\- 一个可替换的控制器接口；

\- 一个先由规则代替 LLM 的闭环控制流程；

\- 一个可运行的 `main.py`；

\- 一套基础日志与配置系统。



\### 3.3 最终研究目标



在 MVP 稳定后，逐步扩展为：



\- 真实 LLM 接入；

\- 多角色推理链（Strategist / Analyst / Actuator）；

\- 滑动窗口经验池；

\- 更丰富的 Pareto-State 指标；

\- 更完整的基线与消融实验；

\- 更强的可解释性分析与实验记录能力。



---



\## 4. 非目标（当前阶段不做）



为了避免项目过早复杂化，以下内容不作为当前阶段的目标：



1\. 不追求一次性实现所有复杂约束与真实工业场景；

2\. 不要求一开始就接真实数据库；

3\. 不要求一开始就接真实在线 LLM API；

4\. 不优先开发 Web 前端；

5\. 不追求一开始就实现高性能并行化；

6\. 不要求第一版就达到论文投稿级效果。



---



\## 5. 设计原则



\### 5.1 模块化



项目必须严格模块化，问题定义、优化器、感知器、控制器、经验池、基础设施和实验管理必须分层组织，避免脚本式堆叠。



\### 5.2 低耦合，高内聚



\- 问题定义不能和 LLM 调用逻辑耦合；

\- LLM 调用不能直接侵入优化器内部；

\- 存储层不能散落在业务代码中；

\- 各模块通过清晰的数据结构和接口通信。



\### 5.3 先可运行，再扩展



先做 MVP，确保主循环能跑通。  

在此基础上，再逐步引入：



\- 经验池

\- 真实 LLM

\- 多角色提示链

\- 复杂状态特征

\- 完整实验系统



\### 5.4 配置优先



\- 超参数不应硬编码在各模块内部；

\- 实验设置应优先来自配置文件；

\- 所有策略阈值、窗口大小、控制频率等都应可配置。



\### 5.5 可测试



关键模块必须可单元测试，尤其是：



\- 目标函数

\- 约束检查

\- 修复函数

\- 排序与选择逻辑

\- 状态提取

\- 经验池

\- 控制循环



\### 5.6 可替换



以下组件都应具备可替换性：



\- 问题定义

\- 优化器

\- 状态提取器

\- LLM Client

\- 控制策略

\- 存储后端



---



\## 6. 项目目录规范



项目目录约定如下：



```text

project/

├─ problems/

│  └─ task\_assignment/

│     ├─ encoding.py

│     ├─ objectives.py

│     ├─ constraints.py

│     └─ repair.py

├─ optimizers/

│  └─ nsga2/

│     ├─ population.py

│     ├─ operators.py

│     ├─ selection.py

│     └─ solver.py

├─ sensing/

│  ├─ pareto\_state.py

│  ├─ hypervolume.py

│  └─ feasibility\_metrics.py

├─ llm/

│  ├─ strategist.py

│  ├─ analyst.py

│  ├─ actuator.py

│  └─ prompts/

├─ memory/

│  └─ experience\_pool.py

├─ controller/

│  └─ closed\_loop.py

├─ infra/

│  ├─ llm\_client.py

│  ├─ storage.py

│  └─ api\_bridge.py

├─ experiments/

│  ├─ configs/

│  ├─ baselines/

│  ├─ logs/

│  └─ ablations/

└─ main.py

```



---



\## 7. 各目录职责说明



\### 7.1 `problems/`



负责定义具体优化问题，包括：



\- 解的编码方式；

\- 多目标函数；

\- 约束条件；

\- 非法解修复逻辑。



\#### `problems/task\_assignment/encoding.py`

定义候选解的表示形式。  

例如，一个长度为 `n\_tasks` 的整数向量，表示每个任务被分配到哪个资源。



\#### `problems/task\_assignment/objectives.py`

定义多目标评价函数。  

例如：



\- 最小化总成本；

\- 最小化总延迟；

\- 最小化负载不均衡。



\#### `problems/task\_assignment/constraints.py`

定义约束检查与约束违反度。  

例如：



\- 资源容量限制；

\- 时间窗约束；

\- 每个任务必须唯一分配。



\#### `problems/task\_assignment/repair.py`

定义修复函数。  

当交叉或变异导致解不合法时，负责将其修正为合法或更接近合法的解。



---



\### 7.2 `optimizers/`



负责定义优化器本身。



\#### `optimizers/nsga2/population.py`

定义个体与种群结构，包括：



\- 个体的决策变量；

\- 目标值；

\- 约束违反度；

\- rank；

\- crowding distance。



\#### `optimizers/nsga2/operators.py`

定义遗传算子，包括：



\- 交叉；

\- 变异；

\- 针对任务分配编码的定制扰动。



\#### `optimizers/nsga2/selection.py`

定义选择相关逻辑，包括：



\- 非支配排序；

\- 拥挤距离；

\- 锦标赛选择；

\- 环境选择。



\#### `optimizers/nsga2/solver.py`

定义优化器主循环，包括：



\- 初始化；

\- 评价；

\- 父代选择；

\- 交叉变异；

\- 修复；

\- 合并与保留；

\- 代际更新。



---



\### 7.3 `sensing/`



负责状态提取与搜索态势感知。  

目标是把“当前种群”压缩成可供控制器理解的结构化状态。



\#### `sensing/pareto\_state.py`

定义 Pareto-State 提取逻辑。  

可包含如下指标：



\- 当前代数；

\- Rank-1 个体比例；

\- 可行解比例；

\- 超体积（HV）；

\- HV 增量；

\- 平均约束违反度；

\- 多样性指标；

\- 停滞长度。



\#### `sensing/hypervolume.py`

定义 HV 计算接口。  

后续可以替换为更准确或更高性能的实现。



\#### `sensing/feasibility\_metrics.py`

定义可行性指标计算逻辑，例如：



\- feasible\_ratio

\- mean\_constraint\_violation

\- max\_constraint\_violation



---



\### 7.4 `llm/`



负责大模型推理链，不直接求解，而是负责\*\*分析状态并输出控制策略\*\*。



\#### `llm/analyst.py`

根据当前 Pareto-State 和历史经验，分析当前搜索局面。  

例如：



\- 是否陷入停滞；

\- 是否可行率过低；

\- 是否多样性不足；

\- 当前主要瓶颈是什么。



\#### `llm/strategist.py`

根据分析结果给出高层策略。  

例如：



\- 偏探索；

\- 偏利用；

\- 偏可行性修复；

\- 偏多样性恢复。



\#### `llm/actuator.py`

把高层策略转为可执行控制动作。  

例如输出：



\- `mutation\_prob`

\- `crossover\_prob`

\- `repair\_strength`

\- `selection\_pressure`



\#### `llm/prompts/`

保存提示词模板与输出格式约定。  

要求提示词与 Python 业务代码分离。



---



\### 7.5 `memory/`



负责历史经验存储与检索。



\#### `memory/experience\_pool.py`

保存历史记录，包括：



\- state

\- action

\- reward

\- next\_state



支持：



\- 滑动窗口裁剪；

\- 相似状态检索；

\- 历史摘要生成。



---



\### 7.6 `controller/`



负责将优化器、状态感知、经验池与 LLM 串成闭环。



\#### `controller/closed\_loop.py`

主控制逻辑应包括：



1\. 运行若干代优化；

2\. 提取当前状态；

3\. 检索历史经验；

4\. 调用控制器；

5\. 获取动作；

6\. 应用动作到优化器参数；

7\. 写回经验池和日志；

8\. 继续下一轮。



---



\### 7.7 `infra/`



负责基础设施连接。



\#### `infra/llm\_client.py`

封装大模型调用，要求：



\- 支持 mock 模式；

\- 支持真实 API 模式；

\- 屏蔽底层 HTTP 调用细节；

\- 提供统一接口。



\#### `infra/storage.py`

负责存储层抽象。  

支持保存：



\- 配置快照；

\- 每代日志；

\- 状态轨迹；

\- 动作轨迹；

\- 实验汇总。



优先采用轻量方式，如：



\- JSONL

\- CSV

\- 本地文件



后续再扩展到：



\- SQLite

\- MySQL

\- 其他数据库



\#### `infra/api\_bridge.py`

可选的外部桥接层。  

当需要对接 FastAPI、Flask、远程流程平台或其他系统时使用。



---



\### 7.8 `experiments/`



负责实验组织，而不是业务逻辑。



\#### `experiments/configs/`

保存配置文件，如：



\- 问题规模；

\- 种群规模；

\- 代数；

\- 控制周期；

\- LLM 模式；

\- 经验池窗口长度；

\- 随机种子。



\#### `experiments/baselines/`

保存基线实现，如：



\- 标准 NSGA-II；

\- 规则控制版本；

\- 不带经验池版本。



\#### `experiments/logs/`

保存实验日志与输出结果。



\#### `experiments/ablations/`

保存消融实验脚本与配置。



---



\### 7.9 `main.py`

项目总入口，负责：



\- 读取配置；

\- 创建问题实例；

\- 创建优化器；

\- 创建控制器；

\- 启动运行；

\- 输出结果与日志。



---



\## 8. 核心数据流



系统核心数据流应为：



```text

Problem Instance

&nbsp;   ↓

NSGA-II Optimizer

&nbsp;   ↓

Population / Pareto Front

&nbsp;   ↓

Pareto-State Sensing

&nbsp;   ↓

Experience Retrieval

&nbsp;   ↓

LLM Analysis / Strategy / Action

&nbsp;   ↓

Optimizer Parameter Update

&nbsp;   ↓

Next Generations

&nbsp;   ↓

Logging + Experience Writeback

```



这意味着：



\- 优化器负责搜索；

\- 状态感知器负责抽象当前局面；

\- LLM 负责高层控制；

\- 经验池负责记忆；

\- 控制器负责闭环调度；

\- 存储层负责记录实验事实。



---



\## 9. 核心数据模型约定



建议使用 `pydantic` 或 dataclass 定义核心数据结构。



\### 9.1 `Individual`



```python

from dataclasses import dataclass, field

from typing import List



@dataclass

class Individual:

&nbsp;   genome: list\[int]

&nbsp;   objectives: list\[float] = field(default\_factory=list)

&nbsp;   constraint\_violation: float = 0.0

&nbsp;   feasible: bool = True

&nbsp;   rank: int = -1

&nbsp;   crowding\_distance: float = 0.0

```



\### 9.2 `ParetoState`



```python

from dataclasses import dataclass



@dataclass

class ParetoState:

&nbsp;   generation: int

&nbsp;   hv: float

&nbsp;   delta\_hv: float

&nbsp;   feasible\_ratio: float

&nbsp;   rank1\_ratio: float

&nbsp;   mean\_cv: float

&nbsp;   diversity\_score: float

&nbsp;   stagnation\_len: int

```



\### 9.3 `ControlAction`



```python

from dataclasses import dataclass



@dataclass

class ControlAction:

&nbsp;   mutation\_prob: float

&nbsp;   crossover\_prob: float

&nbsp;   repair\_strength: float = 0.5

&nbsp;   selection\_pressure: float = 1.0

```



\### 9.4 `ExperienceRecord`



```python

from dataclasses import dataclass



@dataclass

class ExperienceRecord:

&nbsp;   state: dict

&nbsp;   action: dict

&nbsp;   reward: float

&nbsp;   next\_state: dict

```



---



\## 10. 控制逻辑约定



\### 10.1 控制周期



控制器不一定每一代都调用。  

默认约定为每 `k` 代调用一次，例如每 5 代调用一次。



\### 10.2 动作空间



初期仅控制以下参数：



\- `mutation\_prob`

\- `crossover\_prob`



后续可扩展：



\- `repair\_strength`

\- `selection\_pressure`

\- `offspring\_ratio`

\- `mutation\_step\_size`



\### 10.3 奖励定义



初期奖励可简单定义为：



```text

reward = w1 \* delta\_hv + w2 \* delta\_feasible\_ratio - w3 \* mean\_cv

```



后续可扩展为：



\- 多指标加权；

\- 阶段性奖励；

\- 停滞惩罚；

\- 稳定性奖励。



\### 10.4 控制器替换策略



当前阶段必须支持以下三种控制模式：



1\. `rule`：纯规则控制；

2\. `mock\_llm`：模拟 LLM 输出；

3\. `real\_llm`：真实 LLM API。



这样可以保证即使没有 API Key，系统也能跑通。



---



\## 11. 配置系统要求



所有关键参数应来自配置文件，而不是硬编码在代码里。



建议支持 YAML 配置。



示例：



```yaml

seed: 42



problem:

&nbsp; name: task\_assignment

&nbsp; n\_tasks: 50

&nbsp; n\_resources: 10

&nbsp; capacity\_limit: 100



optimizer:

&nbsp; name: nsga2

&nbsp; population\_size: 100

&nbsp; n\_generations: 100

&nbsp; crossover\_prob: 0.9

&nbsp; mutation\_prob: 0.1



controller:

&nbsp; mode: rule

&nbsp; control\_interval: 5

&nbsp; memory\_window: 20



logging:

&nbsp; output\_dir: experiments/logs

&nbsp; save\_population: false

```



---



\## 12. 日志与输出要求



每次实验至少保存以下内容：



1\. 配置快照；

2\. 每代指标日志；

3\. 控制动作日志；

4\. 最终 Pareto 解集；

5\. 运行摘要。



\### 12.1 每代日志建议字段



\- generation

\- hv

\- delta\_hv

\- feasible\_ratio

\- rank1\_ratio

\- mean\_cv

\- diversity\_score

\- stagnation\_len

\- mutation\_prob

\- crossover\_prob

\- reward



\### 12.2 输出格式建议



优先采用：



\- `jsonl`：适合记录逐代结构化日志；

\- `csv`：适合画图和统计；

\- `yaml/json`：适合保存配置快照。



---



\## 13. 测试要求



必须为下列模块提供测试：



\### 13.1 单元测试



\- `objectives.py`

\- `constraints.py`

\- `repair.py`

\- `selection.py`

\- `pareto\_state.py`

\- `experience\_pool.py`



\### 13.2 集成测试



\- `solver.py` 是否能完成最小代数运行；

\- `closed\_loop.py` 是否能完成一次完整控制循环；

\- `main.py` 是否能基于默认配置直接运行。



\### 13.3 测试原则



\- 优先测试纯函数；

\- 尽量减少对真实外部服务的依赖；

\- LLM 调用必须可 mock；

\- 存储层必须可替换为临时目录。



---



\## 14. 代码风格要求



\### 14.1 语言与版本

\- 使用 Python 3.11。



\### 14.2 风格要求

\- 使用类型注解；

\- 每个模块必须有 module-level docstring；

\- 关键类和函数必须有 docstring；

\- 避免超长函数；

\- 避免脚本式重复逻辑；

\- 禁止把 API Key 写死在代码里；

\- 禁止把 prompt 文本散落在业务逻辑中。



\### 14.3 工具建议

\- `ruff`：代码检查和格式规范；

\- `pytest`：测试；

\- `pydantic`：结构化数据模型；

\- `pyyaml`：配置；

\- `numpy`：数值计算；

\- `matplotlib`：绘图；

\- `httpx`：HTTP 请求；

\- `tenacity`：重试控制。



---



\## 15. 推荐技术栈



\### 15.1 必选

\- Python 3.11

\- `numpy`

\- `pydantic`

\- `pyyaml`

\- `pytest`

\- `matplotlib`



\### 15.2 推荐

\- `httpx`

\- `tenacity`

\- `pandas`

\- `ruff`



\### 15.3 后续可选

\- `pymoo`：用于对照、验证或替代部分优化器实现；

\- `sqlite3`：轻量存储；

\- `fastapi`：如需提供服务接口。



---



\## 16. 开发里程碑



\### M0：工程骨架

\- 创建项目目录

\- 创建基础文件

\- 创建空模块和说明文档



\### M1：问题层 MVP

\- 实现简化任务分配编码

\- 实现 2 个目标

\- 实现基础约束与修复



\### M2：优化器 MVP

\- 实现简化 NSGA-II

\- 能运行若干代

\- 输出基础种群结果



\### M3：状态感知 MVP

\- 提取可行率、rank1\_ratio、停滞长度等指标

\- HV 先允许占位或简化实现



\### M4：规则控制闭环

\- 实现控制周期

\- 用规则替代 LLM 输出参数动作

\- 跑通完整闭环流程



\### M5：经验池

\- 实现状态-动作-反馈记录

\- 支持滑动窗口与简单检索



\### M6：真实 LLM 接入

\- 接入统一 LLMClient

\- 分离 Analyst / Strategist / Actuator

\- 支持 mock / real 双模式



\### M7：实验系统

\- 增加 baseline、ablation、日志、绘图

\- 支持命令行配置运行



\### M8：重构与测试补全

\- 清理重复代码

\- 补齐测试

\- 补齐 README 与实验说明



---



\## 17. Codex 实现要求



当使用 Codex 生成代码时，必须遵守以下规则：



1\. \*\*先搭骨架，再实现逻辑，不要一次性铺满所有复杂功能。\*\*

2\. \*\*先保证 `main.py` 可运行，再做性能和高级特性。\*\*

3\. \*\*每次修改前应说明要新增/修改哪些文件。\*\*

4\. \*\*禁止将多个职责混进单一大文件。\*\*

5\. \*\*控制器、经验池、LLM、优化器之间必须通过明确接口交互。\*\*

6\. \*\*优先写最小可运行实现，不要预先过度抽象。\*\*

7\. \*\*代码生成后应补充运行说明与测试建议。\*\*



---



\## 18. 初始交付标准



第一轮工程必须满足以下最低标准：



\- 项目结构存在；

\- 默认配置存在；

\- `main.py` 能运行；

\- 不依赖真实 API Key；

\- 能输出逐代日志；

\- 至少有一个 baseline 可跑；

\- 至少有一个闭环控制版本可跑；

\- 至少有基础测试样例。



---



\## 19. 成功判定标准



当满足以下条件时，可认为项目进入“可研究迭代”阶段：



1\. 能重复运行并得到稳定输出；

2\. 能切换不同控制模式；

3\. 能保存完整实验日志；

4\. 能进行基线对比；

5\. 能进行消融实验；

6\. 能解释某一阶段为什么调整参数；

7\. 能方便扩展新问题、新优化器和新状态特征。



---



\## 20. 后续文档建议



项目除 `SPEC.md` 外，建议后续再补：



\- `TASKS.md`：开发任务清单

\- `README.md`：运行说明

\- `ARCHITECTURE.md`：系统架构说明

\- `EXPERIMENTS.md`：实验设计说明

\- `PROMPTS.md`：提示词设计说明



---



\## 21. 一句话总结



这是一个用于研究“\*\*LLM 如何作为高层闭环控制器调控 NSGA-II 多目标任务分配优化过程\*\*”的模块化 Python 工程，必须从 MVP 开始，逐步扩展为具备状态感知、经验记忆、LLM 推理、实验对照与可解释分析能力的完整研究系统。

