# 对比试验与消融实验方案（llm-pareto-controller）

> 面向当前仓库可直接执行的“论文实验版”方案，覆盖：实验目标、分组设计、执行流程、质量控制与结果产出。

## 1. 实验目标与核心问题

### 1.1 对比试验（Matched Comparison）

验证在相同问题、相同随机种子、相同优化预算下，以下方法的性能差异：

- `baseline_nsga2`：无闭环控制（纯 NSGA-II）
- `rule_control`：规则闭环控制
- `mock_llm`：LLM 闭环（模拟）
- `real_llm`：LLM 闭环（真实 API）

重点回答：

1. 闭环控制是否显著优于 baseline？
2. LLM 策略相较规则控制的收益是否稳定？
3. 在不同 DWTA 难度（small/medium）下结论是否一致？

### 1.2 消融实验（Ablation）

验证控制器关键组件的贡献：

- 观测特征（如 `no_pareto_state_deep_features`）
- 经验池记忆（如 `no_experience_pool`、`memory_window_*`）
- 状态机粒度（`binary_state_machine` vs `four_state_machine`）
- 动作空间（`pc_pm_only` vs `extended_action_space`）
- 控制频率（`tau_1/3/5/10`）

重点回答：

1. 哪些模块对 HV/IGD 改善贡献最大？
2. 记忆窗口和控制周期是否存在“过小/过大”拐点？
3. 动作空间扩展是否带来稳定收益或仅个别场景有效？

---

## 2. 实验设计

## 2.1 统一控制变量（保证公平）

- 同一 benchmark 使用同一问题定义。
- 同一 seed 在各方法下严格对齐。
- 每组方法统一 `generations` 与 `population_size`。
- 输出统一落盘（`summary.json` + `*.jsonl` + `config_snapshot.yaml`）。

## 2.2 预设矩阵（推荐执行顺序）

### 阶段 A：toy（链路验证）

- 用途：快速 smoke，验证流程和日志是否完整。
- 预期：分钟级完成，检查脚本、路径、产物一致性。

### 阶段 B：pilot（参数与稳定性预评估）

- 用途：在较小规模下初步看趋势。
- 预期：确认是否进入正式 paper 规模，提前发现异常方差。

### 阶段 C：paper（正式统计）

- 用途：产出论文主表和消融主结论。
- 预期：形成可复现的汇总结果与对齐后处理文件。

---

## 3. 详细实验操作流程（可直接执行）

## 3.1 环境准备

1. 创建并激活虚拟环境。
2. 安装依赖：`pip install -r requirements.txt`。
3. 若跑 `real_llm`，配置：`OPENAI_API_KEY`（以及可选 `OPENAI_BASE_URL`、`OPENAI_MODEL`）。

## 3.2 运行对比试验（matched）

> 建议统一采用模块方式运行，避免导入路径问题。

### Step 1：toy 验证

```bash
python -m experiments.run_matrix --preset toy --output-root experiments/runs
```

### Step 2：pilot 试验

```bash
python -m experiments.run_matrix --preset pilot --output-root experiments/runs
```

### Step 3：paper 正式对比

```bash
python -m experiments.run_matrix --preset paper --output-root experiments/runs --skip-ablation
```

产物重点检查：

- `experiments/runs/matrix_manifest.json`
- `experiments/runs/matched/<benchmark>/seed_<seed>/<method>/summary.json`
- 对应 `events.jsonl / generation_metrics.jsonl / actions.jsonl`

## 3.3 运行消融实验

### Step 4：paper 消融

```bash
python -m experiments.run_matrix --preset paper --output-root experiments/runs
```

或（仅消融，代码入口调用时）按 benchmark × seed 跑 `run_ablation_matrix`。

产物重点检查：

- `experiments/runs/ablations/<benchmark>/seed_<seed>/<ablation_name>/summary.json`
- 消融配置是否覆盖：`tau_*`、`memory_window_*`、状态机/动作空间相关变体。

## 3.4 结果导出与后处理

### Step 5：导出汇总

```bash
python -m experiments.export_results --runs-root experiments/runs --output-dir experiments/exports
```

### Step 6：matched 后处理（重算/聚合）

```bash
python -m experiments.postprocess_matched --runs-root experiments/runs/matched/<benchmark>
```

输出建议用于论文表：

- 方法维度均值、标准差、分位数
- 指标：HV / IGD / IGD+ / Spacing / Spread
- 每个方法在每个 benchmark 的 seeds 完整覆盖率

---

## 4. 统计分析与结论撰写建议

## 4.1 统计口径

- 主报告：均值 ± 标准差（按 seed 聚合）。
- 显著性：建议用非参数检验（如 Mann-Whitney U）+ 效应量（Cliff's delta）。
- 多重比较：对多方法多指标进行 FDR 或 Bonferroni 校正。

## 4.2 结论模板

- 对比实验：
  - “在 `dwta_small` 和 `dwta_medium` 上，`mock_llm/real_llm` 相比 `rule_control` 在 HV 上提升 X%，且方差变化为 Y。”
- 消融实验：
  - “去除经验池后 HV 下降最明显，说明记忆模块是主要收益来源。”
  - “`tau=3~5` 区间表现最稳，`tau=1` 易震荡，`tau=10` 响应滞后。”

---

## 5. 质量控制（必做）

1. **可复现性**：固定 seed，保留 `config_snapshot.yaml`。
2. **完整性**：检查每个方法/benchmark/seed 是否都有 `summary.json`。
3. **一致性**：确保 matched 实验中方法间 generations/population 一致。
4. **真实调用标注**：明确区分 `mock_llm` 与 `real_llm`，避免混写。
5. **异常回放**：对极端差值 run 回看 `events.jsonl` 与 `actions.jsonl`。

---

## 6. 推荐执行日程（示例）

- Day 1：toy + pilot，修复配置/日志问题。
- Day 2：paper matched（可先 `--skip-ablation`）。
- Day 3：paper ablation。
- Day 4：导出、统计检验、绘图与论文主表草稿。

---

## 7. 一键命令清单（复制即用）

```bash
# 0) 快速回归测试
pytest tests/test_dwta_minimum_suite.py tests/test_baseline_runner.py tests/test_mock_llm_pipeline.py

# 1) toy
python -m experiments.run_matrix --preset toy --output-root experiments/runs

# 2) pilot
python -m experiments.run_matrix --preset pilot --output-root experiments/runs

# 3) paper 对比
python -m experiments.run_matrix --preset paper --output-root experiments/runs --skip-ablation

# 4) paper 全量（含消融）
python -m experiments.run_matrix --preset paper --output-root experiments/runs

# 5) 导出
python -m experiments.export_results --runs-root experiments/runs --output-dir experiments/exports

# 6) 后处理（按具体 benchmark 替换）
python -m experiments.postprocess_matched --runs-root experiments/runs/matched/dwta_small
```

