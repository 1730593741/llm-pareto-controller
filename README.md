# LLM-Pareto-Controller

一个面向研究实验的 Python 项目：在多目标任务分配问题上，以 NSGA-II 进行底层搜索，并通过规则控制器或 LLM 控制链做在线调参，形成闭环优化。

## 项目目标

- 用 NSGA-II 作为可复现的多目标优化基线。
- 用 Pareto state 感知搜索状态（可行率、多样性、改进趋势等）。
- 用 `rule` / `mock_llm` / `real_llm` 控制模式驱动参数自适应。
- 支持经验池、对照实验、消融实验与结果导出。

## 当前能力（MVP → 工程化）

当前仓库可直接运行以下能力：

- 闭环优化主流程（`main.py`）
- 两类问题场景：
  - `task_assignment`（通用任务-资源分配）
  - `dwta`（动态武器-目标分配，支持实体输入或预计算矩阵）
- 三种控制模式：`rule` / `mock_llm` / `real_llm`
- 经验池日志链路：`state -> action -> reward -> next_state`
- 基线与矩阵实验：baseline、matched、ablation
- 指标与导出：HV、IGD、IGD+、Spread、Spacing 及聚合表导出

## 环境安装

> Python 版本：**3.11**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 快速开始

### 1) 默认运行

```bash
python main.py
```

默认会读取：`experiments/configs/default.yaml`。

### 2) 指定配置运行

```bash
python -c "from main import main; main('experiments/configs/rule_control.yaml')"
python -c "from main import main; main('experiments/configs/mock_llm.yaml')"
python -c "from main import main; main('experiments/configs/real_llm.yaml')"
```

### 3) DWTA 场景（建议先 smoke）

```bash
python -c "from main import main; main('experiments/configs/dwta_small_smoke.yaml')"
python -c "from main import main; main('experiments/configs/dwta_small.yaml')"
python -c "from main import main; main('experiments/configs/dwta_medium.yaml')"
```

## Baseline 与实验矩阵

### 1) Baseline Runner

```bash
python -c "from experiments.baselines.runner import run_baseline_nsga2; print(run_baseline_nsga2())"
python -c "from experiments.baselines.runner import run_rule_control_baseline; print(run_rule_control_baseline())"
python -c "from experiments.baselines.runner import run_no_memory_baseline; print(run_no_memory_baseline())"
```

### 2) Matrix Runner

```bash
python -m experiments.run_matrix --preset toy --output-root experiments/runs/toy
python -m experiments.run_matrix --preset pilot --skip-ablation --output-root experiments/runs/pilot_matched
python -m experiments.run_matrix --preset paper --output-root experiments/runs/paper
```

### 3) 结果导出

```bash
python -m experiments.export_results --runs-root experiments/runs/paper --output-dir experiments/exports/paper
```

## 配置结构（`experiments/configs/*.yaml`）

主配置按模块分层：

- `experiment`: 实验名称、seed、方法标识
- `problem`: 问题定义
  - `task_assignment`: `n_tasks/n_resources/cost_matrix/task_loads/capacities`
  - `dwta`:
    - 实体输入：`munition_types + weapons + targets`（运行时自动预计算）
    - 预计算输入：`precomputed`（直接加载矩阵）
- `optimizer`: NSGA-II 参数
- `controller`: 闭环调参阈值、步长、边界
- `controller_mode`: `rule` / `mock_llm` / `real_llm`
- `memory`: 经验池开关与 reward 参数
- `logging`: 输出目录与日志命名
- `llm`: 真实 LLM 连接参数（provider/model/env key 等）

## 运行输出

每次运行会在 `logging.output_dir` 下写出：

- `config_snapshot.yaml`：配置快照（含来源路径）
- `events.jsonl`：统一事件流（state/action）
- `generation_metrics.jsonl`：逐代状态日志
- `actions.jsonl`：控制动作日志
- `experiences.jsonl`：经验日志（memory 启用时）
- `summary.json`：本次运行摘要（最终指标、参数、日志路径）

### 动态 DWTA 指标补充（为何不能只看 `final_hv`）

在 `scripted_waves` / hard-realworld 动态场景中，单一 `final_hv` 容易掩盖过程风险：

- 事件冲击后是否恢复，需要看 `post_event_recovery_generations_mean`；
- 任务压力与资源退化是累积过程，需要看 `cumulative_unmet_damage` 与 `cumulative_resource_consumption`；
- 波次任务执行完整性，需要看 `wave_completion_rate`；
- 控制器是否真正响应事件，需要看 `event_triggered_actions` 与 `event_types_seen`。

因此，`final_hv` 只描述终点质量，动态 DWTA 还必须结合事件与时序指标描述“稳定性 + 韧性 + 完成度”。

## 消融开关

`experiments/ablations/switches.py` 目前支持：

- `no_memory`
- `no_state_metric_x`
- `fixed_control_interval`
- `no_llm_chain`

## Real LLM 说明

`real_llm` 模式使用环境变量读取密钥，默认不在代码中硬编码：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（可选）

建议流程：先跑 `rule` / `mock_llm`，确认闭环稳定后再切换 `real_llm`。

## 测试与质量检查

```bash
pytest -q
ruff check .
```

## 文档索引

- 快速开始与运行说明：`README.md`
- 系统级详细项目说明：`ARCHITECTURE.md`
- 研究规格：`SPEC.md`
- 任务里程碑：`TASKS.md`
- 技术栈说明：`TECH_STACK.md`
- 实验流水线说明：`EXPERIMENTS.md`
