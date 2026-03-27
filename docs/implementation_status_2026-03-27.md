# 实现状态盘点（2026-03-27）

## 结论（TL;DR）

当前仓库已经具备完整的“**方法对比实验** + **消融实验** + **结果导出/后处理**”闭环能力，且有对应自动化测试覆盖关键路径；可以直接用于小规模 smoke、pilot 和 paper 预设矩阵实验。

在工程可用性上，主要短板不在实验框架，而在 **real LLM 外部依赖**（API Key/网络/服务稳定性）以及 **执行入口细节**（建议以 `python -m ...` 方式运行模块）。

---

## 1) 已实现能力盘点

### 1.1 单次实验运行（含 baseline/rule/LLM）

- `main.py` 提供统一实验入口，负责读取 YAML、构建 solver/controller、执行闭环与产出摘要。支持 `controller.mode` 形态覆盖 baseline、rule、mock_llm、real_llm 的统一运行范式。
- README 已给出按配置启动 baseline/rule/mock_llm/real_llm 的标准命令与日志产物说明。

### 1.2 方法对比实验（matched comparisons）

- `experiments/baselines/matched_runner.py` 支持按 **方法 × benchmark × seed** 对齐运行，确保同口径比较。
- 内置方法集合：`baseline_nsga2 / rule_control / mock_llm / real_llm`。
- `experiments/run_matrix.py` 可在 preset 维度一次拉起 matched 矩阵。

### 1.3 消融实验（ablation matrix）

- `experiments/ablations/matrix_runner.py` 已实现消融矩阵自动派生与运行。
- 已覆盖核心开关：
  - 观测相关（如 no_pareto_state_deep_features）
  - 记忆相关（no_experience_pool、memory_window_*）
  - 控制状态机粒度（binary/four_state_machine）
  - 动作空间维度（pc_pm_only、extended_action_space）
  - 控制周期（tau_*）

### 1.4 预设矩阵（toy/pilot/paper）

- `experiments/matrix.py` 给出 matched 与 ablation 的预设轴。
- `experiments/run_matrix.py` 支持 `--preset toy|pilot|paper`，并支持 `--skip-ablation`。

### 1.5 结果导出与论文后处理

- `experiments/export_results.py`：聚合 `summary.json` 到 CSV/JSON 与 paper table 输入。
- `experiments/postprocess_matched.py`：按方法汇总、重算 IGD/IGD+ 等指标，生成 paper 风格 summary。

### 1.6 测试覆盖现状（与实验能力强相关）

- 已有 smoke/规格测试覆盖矩阵 runner 与消融开关完整性。
- real_llm 配置连线与缺 key fallback 行为已有测试。

---

## 2) “能不能做具体对比实验/消融实验？”

## 可以，且路径清晰

### 2.1 做“方法对比实验”

推荐：

1. 先跑 toy（验证链路）
2. 再跑 pilot（验证统计稳定性）
3. 最后跑 paper（正式结果）

命令（推荐模块方式）：

```bash
python -m experiments.run_matrix --preset toy --output-root experiments/runs
python -m experiments.run_matrix --preset pilot --output-root experiments/runs
python -m experiments.run_matrix --preset paper --output-root experiments/runs
```

如果只做 matched 对比，不做消融：

```bash
python -m experiments.run_matrix --preset paper --output-root experiments/runs --skip-ablation
```

### 2.2 做“消融实验”

可直接复用同一 matrix 入口（默认包含 ablation）。
如需细控，可以直接调用 `run_ablation_matrix`（代码层）。

### 2.3 统计与导表

```bash
python -m experiments.export_results --runs-root experiments/runs --output-dir experiments/exports
python -m experiments.postprocess_matched --runs-root experiments/runs/matched/<benchmark>
```

---

## 3) 当前已确认的运行情况（本地验证）

- 相关测试通过：
  - `tests/test_dwta_minimum_suite.py`
  - `tests/test_baseline_runner.py`
  - `tests/test_mock_llm_pipeline.py`
  - `tests/test_matrix_runner_smoke.py`
  - `tests/test_ablation_matrix_specs.py`
  - `tests/test_real_llm_mode_config.py`

- `python -m experiments.run_matrix --preset toy ...` 已成功写出 `matrix_manifest.json`。
- `experiments.export_results` 与 `experiments.postprocess_matched` 已成功产出聚合文件。

---

## 4) 现阶段限制与注意事项

1. `real_llm` 对外部环境有依赖（API key / base_url / 网络可达）。
2. 缺 key 时系统会按配置 fallback（默认可回退 mock），这有利于流程不阻塞，但正式对比时要明确标注是否真实调用。
3. 从仓库根目录直接 `python experiments/run_matrix.py ...` 可能遇到导入路径问题；建议统一使用 `python -m experiments.run_matrix ...`。

---

## 5) 对你下一步实验执行的建议

1. **先锁定实验协议**：方法集、benchmark 集、seeds、代数、种群大小。
2. **先 toy 后 paper**：先保证链路与日志，再扩大规模。
3. **分开记录 mock_llm 与 real_llm**：避免在结果表中混淆。
4. **固定随机种子并保留 config_snapshot**：保证可追溯。
5. **导出后先做完整性检查**：每个方法/seed/benchmark 是否齐全，再做统计显著性分析。

