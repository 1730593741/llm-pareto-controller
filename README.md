# LLM-Pareto-Controller

一个面向研究实验的 Python 项目：在多目标任务分配问题上，用 NSGA-II 做底层搜索，用规则/LLM 控制链做闭环调参，并支持实验对照与消融。

## 当前状态（截至 M7）

已完成能力：
- M4：可运行的规则闭环（state/action 日志）。
- M5：可选经验池（state -> action -> reward -> next_state）。
- M6：`rule` / `mock_llm` / `real_llm` 三种控制模式，LLM 链路为 Analyst -> Strategist -> Actuator。
- M7：最小实验系统（配置化运行、baseline、ablation 开关、配置快照与运行摘要输出）。

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 运行方式

### 1) 默认闭环实验

```bash
python main.py
```

默认读取：`experiments/configs/default.yaml`。

### 2) 指定配置运行

```bash
python -c "from main import main; main('experiments/configs/rule_control.yaml')"
python -c "from main import main; main('experiments/configs/mock_llm.yaml')"
python -c "from main import main; main('experiments/configs/real_llm.yaml')"
```

### 3) baseline 运行

```bash
python -c "from experiments.baselines.runner import run_baseline_nsga2; print(run_baseline_nsga2())"
python -c "from experiments.baselines.runner import run_rule_control_baseline; print(run_rule_control_baseline())"
python -c "from experiments.baselines.runner import run_no_memory_baseline; print(run_no_memory_baseline())"
```

## 配置说明（`experiments/configs/*.yaml`）

配置统一按以下分层：
- `problem`: 问题规模与数据（任务数、资源数、成本矩阵、负载、容量）。
- `optimizer`: NSGA-II 参数（种群、代数、交叉/变异概率、随机种子）。
- `controller`: 控制器参数（控制周期、阈值、参数边界、步长）。
- `controller_mode`: 运行模式（`rule` / `mock_llm` / `real_llm`）。
- `memory`: 经验池开关与 reward 参数。
- `logging`: 输出目录与文件命名。
- `llm`: LLM 运行时配置（provider/model/env key 等）。
- `experiment`: 实验元信息（名称、seed）。

## 实验输出

每次运行会在 `logging.output_dir` 下至少生成：
- `config_snapshot.yaml`: 运行配置快照（含来源 config 路径）。
- `events.jsonl`: 统一事件流（state/action）。
- `generation_metrics.jsonl`: 逐代状态日志。
- `actions.jsonl`: 动作日志。
- `experiences.jsonl`（启用 memory 时）: 经验记录。
- `summary.json`: 本次运行摘要（最终 hv、参数、日志路径等）。

## 消融开关

`experiments/ablations/switches.py` 当前支持：
- `no_memory`
- `no_state_metric_x`
- `fixed_control_interval`
- `no_llm_chain`

可用于派生临时配置做对照实验。

## 测试

```bash
pytest -q
```

M7 相关重点测试：
- `tests/test_main_config_modes.py`
- `tests/test_baseline_runner.py`
