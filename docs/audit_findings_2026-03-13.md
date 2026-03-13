"""代码审阅纪要：指标严谨性、语义控制落地、实验支撑度（2026-03-13）。"""

# 审阅范围

- `eval/metrics.py`
- `sensing/pareto_state.py`
- `controller/control_semantics.py` + `controller/closed_loop.py`
- `llm/prompts/*.txt` + `llm/{analyst,strategist,actuator}.py`
- `experiments/matrix.py` + `experiments/run_matrix.py` + `experiments/baselines/matched_runner.py` + `experiments/ablations/matrix_runner.py`

---

## 1) 数学指标实现严谨性（IGD / IGD+ / CrowdingEntropy 等）

### 1.1 结论（简版）

- **IGD / IGD+ / Spacing / Spread**：实现思路总体正确，边界条件处理较完善，可用于论文级结果汇总。
- **CrowdingEntropy / d_dec / d_front**：作为“状态感知特征”是可用的，但属于启发式代理量，不宜在论文中表述为“标准指标替代”。
- 目前最需要补强的是：
  1) `Spread(Δ)` 对极值点匹配策略与标准定义的一致性说明；
  2) `CrowdingEntropy` 的定义引用与“与拥挤距离的关系”；
  3) 对异常输入（NaN、维度不一致、全重复点）的更系统单测。

### 1.2 `eval/metrics.py` 逐项审阅

- `igd`：实现了经典定义 `mean_r min_a ||r-a||2`，并对空参考前沿返回 `0`、空获得前沿返回 `inf`，逻辑清晰。
- `igd_plus`：采用 `max(a-r,0)` 的修正距离，方向（最小化）正确。
- `spacing`：使用最近邻 L1 距离并返回样本标准差（`ddof=1`），与常见文献口径一致。
- `spread`：
  - 限定 2 目标、排序后计算 `d_f, d_l` 与相邻距离离散性，符合 Deb 的 Δ 指标框架；
  - 当参考前沿缺失时退化为 `ref = got`，工程上可运行，但论文中应标注“近似策略”，否则易被审稿人质疑“极值点是否真实来自真 PF”。

### 1.3 `sensing/pareto_state.py` 逐项审阅

- `crowding_entropy`：
  - 用 rank-1 点最近邻距离构建分布并做归一化熵，数值稳定（有 `eps`、`nan_to_num`、裁剪）；
  - 但它是“邻域距离熵”，不是 NSGA-II 标准 crowding distance 本体。论文里建议命名为 **NN-distance entropy** 或显式给定义式。
- `d_dec`：
  - 离散任务分配用归一化 Hamming，建模合理；
  - 与连续决策空间方法不可直接比较，应在方法章声明“问题特定”。
- `d_front`：
  - 采用 `inter/(inter+intra)`，在 [0,1] 内且解释直观；
  - 但它不是标准 Pareto gap 指标，建议在附录给出与已有指标的关联讨论。
- `diversity_score`（质心平均距离）可用但受尺度影响；跨 benchmark 比较时应配合目标归一化。

---

## 2) “收敛-分布-可行性”三位一体语义控制的落地情况

### 2.1 当前实现链路

1. **状态语义离散化**：`ControlState` 定义四态：
   - `increase_convergence`
   - `increase_diversity`
   - `increase_feasibility`
   - `maintain_balance`
2. **Analyst**：读取 `ParetoState + recent_experiences`，输出三信号（convergence/diversity/feasibility）+ 单一 `control_state`。
3. **Strategist**：将分析结果映射到统一控制意图（四态之一）。
4. **Actuator**：在参数边界和步长约束下生成 `mutation/crossover/(eta/repair/local_search)` 动作。
5. **ClosedLoopRunner**：按 `control_interval` 触发控制并写入状态-动作-奖励闭环。

### 2.2 落地质量评估

- **优点**
  - 语义层（四态）与执行层（参数空间）分离清晰，便于替换策略或做消融。
  - Prompt 未散落业务代码，集中在 `llm/prompts/`，可维护性好。
  - Actuator 有硬边界 + 平滑步长，能避免 LLM 输出造成参数抖动。

- **当前不足（影响论文说服力）**
  1. 三位一体并未形成显式多目标决策函数（例如加权效用或约束优先级优化），更多是**阈值/规则 + LLM文本决策**。
  2. Analyst 的 fallback 与 mock 逻辑强依赖阈值（`feasible_ratio<0.6`、`diversity<0.12` 等），可能导致“LLM贡献”与“规则贡献”边界不清。
  3. Strategist 基本是透传 control_state（mock 下尤其明显），会被质疑为“多角色链是否必要”。
  4. 三信号虽有输出，但当前动作选择没有可追溯的显式映射矩阵（例如每态对各参数方向/幅度的统一表）。

---

## 3) 对比基线 + 消融矩阵是否足以支撑“4个创新点”

### 3.1 现有矩阵覆盖

- **Matched methods**：`baseline_nsga2`, `rule_control`, `mock_llm`, `real_llm`（同 seed/benchmark/群体规模）。
- **Ablation**：
  - `no_pareto_state_deep_features`
  - `no_experience_pool`
  - `binary_state_machine` vs `four_state_machine`
  - `pc_pm_only` vs `extended_action_space`
  - `tau` 与 `memory_window` 灵敏度

### 3.2 是否“足够支撑 4 创新点”

- **可支撑的部分**（中等强度）
  - 创新点A（闭环优于静态参数）：`baseline_nsga2` vs `rule_control/mock_llm/real_llm`。
  - 创新点B（四态语义优于二态）：`binary_state_machine` vs `four_state_machine`。
  - 创新点C（扩展动作空间价值）：`pc_pm_only` vs `extended_action_space`。
  - 创新点D（经验池贡献）：`no_experience_pool` vs 默认。

- **仍不足的部分**（论文级风险）
  1. 缺少 `no_strategist` / `no_analyst` 等**链路级消融**，难以证明三角色分工的必要性。
  2. `no_pareto_state_deep_features` 目前主要靠阈值改写，尚不是“严格删除特征并保持其它不变”的最小干预实验。
  3. 默认 paper preset 仅 `small_complex + medium_complex`，若声称“复杂任务分配泛化”，建议纳入 `hard_complex`。
  4. 未见显式统计显著性流程（如 Wilcoxon/Cliff's delta），当前更偏工程验证，不足以直接支撑强因果结论。

---

## 4) 建议的最小补强（不重构）

1. **加两组关键消融**：`no_strategist`、`no_analyst`（或 strategist=identity / analyst=rule-only）。
2. **把 deep feature ablation 做成真删特征**：输入端移除 `crowding_entropy/d_dec/d_front`，并保持阈值不变。
3. **paper preset 扩展到 hard_complex**，至少 3 benchmarks。
4. **补统计脚本**：对 IGD+/HV 做 paired test + effect size。
5. **文稿术语收敛**：将 CrowdingEntropy 明确写成“rank-1 最近邻距离熵（自定义）”。

---

## 5) 结论

当前实现已经具备“可运行的论文实验骨架”，但要稳妥支撑“4个创新点”，还需要**链路级消融 + 更严格特征消融 + 更强统计证据**三项补强。
