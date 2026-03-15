# Experiments Workflow (from 0 to paper matrix)

## 0) 目标与适用范围

这份文档面向「第一次接手本仓库」的同学，给出：

1. 从 0 开始如何把环境跑通；
2. 如何修改问题数据参数（任务数、成本矩阵、容量等）；
3. 如何修改优化与控制参数（NSGA-II + rule/LLM 控制）；
4. 如何执行 smoke / pilot / paper 三层实验；
5. 如何运行 DWTA 场景（实体输入与预计算矩阵输入）；
6. 如何导出汇总结果做对照分析。

---

## 1) 从 0 开始：环境准备

### 1.1 Python 版本

项目推荐 Python 3.11。

### 1.2 安装依赖

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 1.3 最小可运行检查

```bash
python main.py
```

默认会读取 `experiments/configs/default.yaml`，并写出日志到该配置中的 `logging.output_dir`。

---

## 2) 配置文件结构：你要改参数，先看这 8 个块

> DWTA 场景新增说明：`problem.problem_type: dwta` 时，支持两种配置路径：
> - **实体输入**：`munition_types + weapons + targets`，运行时自动预计算 `compatibility_matrix` / `lethality_matrix`；
> - **预计算输入**：`precomputed`，直接使用矩阵，适合复现实验与批量对照。

实验主配置位于 `experiments/configs/*.yaml`，核心字段如下：

- `experiment`: 实验名称、随机种子、方法标识。
- `problem`: 问题规模与数据（任务、资源、成本、负载、约束）。
- `optimizer`: NSGA-II 参数（种群、迭代代数、交叉/变异等）。
- `controller`: 闭环调参边界、步长、触发阈值。
- `controller_mode`: `rule` / `mock_llm` / `real_llm`。
- `memory`: 经验池开关、窗口大小、reward 权重。
- `logging`: 本次 run 的输出目录与日志文件名。
- `llm`: 真实 LLM 连接参数与失败回退策略。

建议做实验时：

1. **不要直接改默认配置**；
2. 从现有配置复制一份（例如 `small_complex.yaml`），命名为 `my_exp_xxx.yaml`；
3. 只改你这次要研究的变量。

---

## 3) 从 0 改“数据参数”：problem 区块怎么改

## 3.1 最小必改项（规模）

在 `problem` 中优先关注：

- `n_tasks`
- `n_resources`
- `cost_matrix`
- `task_loads`
- `capacities`

这些字段的维度必须一致：

- `cost_matrix` 行数 = `n_tasks`
- `cost_matrix` 列数 = `n_resources`
- `len(task_loads)` = `n_tasks`
- `len(capacities)` = `n_resources`

## 3.2 进阶约束项（复杂基准）

如果你使用 `medium_complex` / `hard_complex` 风格数据，还可能设置：

- `compatibility_matrix`：任务-资源可兼容性（0/1）。
- `task_time_windows` 与 `resource_time_windows`：时间窗约束。
- `resource_stage_levels` 与 `stage_transitions`：工序阶段约束。

推荐做法：

- 第一次先不引入这些复杂约束，先在 `small_complex_smoke` 量级验证流程；
- 然后逐步打开每类约束，避免一次改太多导致无法定位问题。

## 3.3 一个典型“改数据”流程

1. 复制基准配置：

```bash
cp experiments/configs/small_complex.yaml experiments/configs/my_data_v1.yaml
```

2. 编辑 `my_data_v1.yaml` 的 `problem` 块（规模 + 数据）；
3. 保持 `optimizer.seed` 与 `experiment.seed` 固定，先做可复现实验；
4. 跑一次单配置：

```bash
python -c "from main import main; main('experiments/configs/my_data_v1.yaml')"
```

5. 检查输出 `summary.json` 与 `generation_metrics.jsonl` 是否合理。

---


## 3.4 DWTA 配置结构（新增）

### A) 实体输入（自动预计算）

```yaml
problem:
  problem_type: dwta
  munition_types: ...
  weapons: ...
  targets: ...
```

运行时会生成：
- `ammo_capacities`（武器弹药上限）
- `compatibility_matrix`（射程+时窗耦合兼容）
- `lethality_matrix`（武器-目标毁伤贡献）
- `required_damage`（目标所需毁伤）

### B) 预计算输入（直接加载）

```yaml
problem:
  problem_type: dwta
  precomputed:
    ammo_capacities: [...]
    compatibility_matrix: ...
    lethality_matrix: ...
    required_damage: [...]
```

当你已离线固定场景矩阵（例如论文复现实验）时，推荐使用 `precomputed`。

## 4) 改“算法参数”：optimizer / controller / memory

## 4.1 NSGA-II 参数（`optimizer`）

关键参数与建议：

- `population_size`: 初期建议 12~40；
- `generations`: smoke 先 4~10，正式实验再 20+；
- `crossover_prob`: 常见 0.8~0.95；
- `mutation_prob`: 常见 0.05~0.2；
- `repair_prob`: 有约束问题建议保持 1.0；
- `seed`: 做对照实验必须固定或显式扫 seed。

## 4.2 闭环控制参数（`controller`）

重点看：

- `control_interval`: 每多少代触发一次调参；
- `min_*/max_*`: 参数边界（防止越界）；
- `*_step`: 每次动作幅度；
- `feasible_ratio_low`, `diversity_low`, `improvement_threshold`: 触发策略阈值。

经验建议：

- 如果动作过于抖动，减小 `*_step`；
- 如果几乎不动作，调低阈值或缩短 `control_interval`；
- 先用 `rule` 模式跑稳，再切 `mock_llm` / `real_llm`。

## 4.3 经验池（`memory`）

- `enabled`: 是否启用经验记录与检索。
- `memory_window`: 历史窗口大小。
- `reward_alpha`, `reward_beta`: reward 计算权重。

建议：先固定 `reward_*`，只单独扫 `memory_window`（例如 5/20/50）。

---

## 5) 三种运行层级：单配置 / 矩阵 / 导出

## 5.1 单配置快速迭代（最常用）

```bash
python -c "from main import main; main('experiments/configs/rule_control.yaml')"
python -c "from main import main; main('experiments/configs/mock_llm.yaml')"
python -c "from main import main; main('experiments/configs/real_llm.yaml')"
```

DWTA smoke / mainline：

```bash
python -c "from main import main; main('experiments/configs/dwta_small_smoke.yaml')"
python -c "from main import main; main('experiments/configs/dwta_small.yaml')"
python -c "from main import main; main('experiments/configs/dwta_medium.yaml')"
```

适用：调试某个参数改动是否生效。

## 5.2 矩阵实验（统一批量）

### A. Toy smoke（快速）

```bash
python -m experiments.run_matrix --preset toy --output-root experiments/runs/toy
```

### B. Pilot matched（方法对照）

```bash
python -m experiments.run_matrix --preset pilot --skip-ablation --output-root experiments/runs/pilot_matched
```

### C. Full paper matrix（matched + ablation）

```bash
python -m experiments.run_matrix --preset paper --output-root experiments/runs/paper
```

## 5.3 导出聚合结果（表格输入）

```bash
python -m experiments.export_results --runs-root experiments/runs/paper --output-dir experiments/exports/paper
```

输出：

- `aggregated_runs.csv`
- `aggregated_runs.json`
- `paper_table_method.csv`
- `paper_table_method_benchmark.csv`

---

## 6) 如何“系统性做实验”：推荐顺序

1. **Smoke 阶段**：先在 `small_complex_smoke` 跑通命令、日志、导出；
2. **数据确认阶段**：修改你的 `problem` 数据，单配置重复跑，确认约束与指标正常；
3. **方法对照阶段**：固定 benchmark 与 seeds，对比 `baseline_nsga2 / rule_control / mock_llm`；
4. **LLM 接入阶段**：确认 `mock_llm` 稳定后再开 `real_llm`；
5. **消融阶段**：围绕单一变量做扫参（`tau`、`memory_window`、control 间隔等）；
6. **汇总阶段**：统一导出并在同一 CSV/JSON 上做统计检验与画图。

---

## 7) real_llm 环境变量与回退策略

`real_llm` 模式读取：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（可选）
- `OPENAI_MODEL`（可选）

`llm.fallback_mode`：

- `mock_llm`（默认）：API 出错自动回退 mock，保证实验不中断；
- `hold`：API 出错时保持当前参数不更新。

---

## 8) 每次实验结束要检查的 6 个文件

在每个 run 目录下至少检查：

1. `config_snapshot.yaml`（确认真实生效参数）；
2. `summary.json`（最终 hv、运行时间、方法标识）；
3. `generation_metrics.jsonl`（逐代指标变化）；
4. `actions.jsonl`（控制动作轨迹）；
5. `events.jsonl`（统一事件流）；
6. `experiences.jsonl`（仅 memory 开启时存在）。

如果结果异常，优先看：

- 配置快照是否与你编辑文件一致；
- seed 是否在不同方法间对齐；
- 输出目录是否被旧实验污染（建议每次用新目录）。


## DWTA Smoke 预期日志片段

典型 `summary.json` 关注字段：
- `benchmark`: `dwta_small_smoke`
- `controller_mode`: `rule` / `mock_llm` / `real_llm`
- `final_hv`, `final_feasible_ratio`, `control_state_counts`

典型 `actions.jsonl` 信号：
- 若 `feasible_ratio` 较低，常见 `control_state=increase_feasibility`，用于抑制弹药超载与时空/射程兼容性冲突。
