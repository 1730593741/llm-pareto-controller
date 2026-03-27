# llm-pareto-controller

面向论文实验的 **LLM 闭环 NSGA-II**（聚焦 DWTA）研究代码仓库。

本项目用于可复现实验对比：
- Vanilla NSGA-II（baseline）
- 规则闭环 NSGA-II（rule-based controller）
- LLM 闭环 NSGA-II（mock / real LLM）
- 消融实验（观测、记忆、动作维度开关）

---

## 1. 项目定位

该仓库不是通用优化框架，而是为 DWTA 论文实验服务的显式实现，重点关注：

1. 可重复（配置、seed、日志与结果文件可追溯）
2. 可对比（baseline / rule / llm 同口径输出）
3. 可消融（矩阵化运行与后处理）

---

## 2. 环境与安装

- Python: `>=3.11`
- 依赖管理: `pip`（或你熟悉的虚拟环境工具）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

如果你希望按打包元数据安装，也可以：

```bash
pip install -e .
```

---

## 3. 目录结构（实验视角）

```text
.
├── main.py                     # 单次实验入口（读取 YAML 并运行）
├── experiments/
│   ├── configs/                # 基线、rule、mock_llm、real_llm 与 DWTA 场景配置
│   ├── baselines/              # baseline 与 matched matrix 运行器
│   ├── ablations/              # 消融开关与矩阵运行
│   ├── run_matrix.py           # toy/pilot/paper 预设矩阵入口
│   ├── export_results.py       # 结果导出
│   └── postprocess_matched.py  # 对齐后处理
├── src/dwta/                   # DWTA 问题定义、约束、评估、修复、场景构造
├── optimizers/nsga2/           # NSGA-II 核心（种群、选择、算子、求解）
├── controller/                 # 闭环控制语义、动作空间、runner
├── sensing/                    # 状态感知（HV、可行率等）
├── llm/                        # analyst/strategist/actuator 与 prompts
├── infra/                      # LLM client 与日志存储
├── memory/                     # 经验池
└── tests/                      # 单测与 smoke/regression 用例
```

---

## 4. 快速开始

### 4.1 运行一次默认实验

```bash
python main.py
```

或显式指定配置：

```bash
python main.py experiments/configs/default.yaml
# 或
python main.py --config experiments/configs/default.yaml
```

运行后会在配置中的 `logging.output_dir` 写入：
- `events.jsonl`
- `generation_metrics.jsonl`
- `actions.jsonl`
- `experiences.jsonl`（若启用 memory）
- `summary.json`
- `config_snapshot.yaml`

### 4.2 运行典型方法对比

```bash
python main.py experiments/configs/baseline_nsga2.yaml
python main.py experiments/configs/rule_control.yaml
python main.py experiments/configs/mock_llm.yaml
python main.py experiments/configs/real_llm.yaml
```

> `real_llm.yaml` 需要可用的 API 环境变量（见下文“LLM 配置”）。

---

## 5. 复现实验矩阵

### 5.1 一键运行预设矩阵

```bash
python experiments/run_matrix.py --preset toy --output-root experiments/runs
python experiments/run_matrix.py --preset pilot --output-root experiments/runs
python experiments/run_matrix.py --preset paper --output-root experiments/runs
```

如需跳过消融：

```bash
python experiments/run_matrix.py --preset paper --skip-ablation
```

输出根目录下会生成 `matrix_manifest.json`，记录矩阵参数与结果路径。

### 5.2 多 seed 配置生成

```bash
python make_seed_config.py \
  experiments/configs/mock_llm.yaml \
  42 \
  experiments/runs/mock_llm/seed_42 \
  tmp_seed_configs/mock_llm_seed_42.yaml
```

---

## 6. 配置说明（YAML）

关键字段通常包括：

- `experiment`: 实验名、方法名、benchmark、seed
- `problem` / `dwta`: 场景定义（静态或 scripted waves）
- `optimizer`: NSGA-II 参数（population / generations / crossover / mutation 等）
- `controller`: 控制周期、阈值与动作边界
- `controller_mode`: `none` / `rule` / `llm`（依配置实现）
- `memory`: 经验池开关与窗口
- `llm`: provider/model/timeout/retry/fallback
- `logging`: 输出目录与各类文件名

建议以 `experiments/configs/*.yaml` 为模板复制修改，不要从零手写。

---

## 7. LLM 配置

`real_llm` 模式通常需要以下环境变量（以实际配置为准）：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（可选）
- `OPENAI_MODEL`（可选）

`mock_llm` 模式可用于离线 smoke 与流程验证，不依赖真实网络调用。

---

## 8. 测试

运行全部测试：

```bash
pytest
```

建议至少覆盖以下三类快速回归：

```bash
pytest tests/test_dwta_minimum_suite.py
pytest tests/test_baseline_runner.py
pytest tests/test_mock_llm_pipeline.py
```

---

## 9. 结果与产物

`summary.json` 会包含（示例）：
- 最终/最优 HV、IGD、IGD+、Spacing、Spread
- 最终算子参数与动作统计
- 运行耗时与 LLM 开销
- 事件日志/动作日志/配置快照路径

论文统计通常在 `experiments/runs/` 与导出脚本中完成（见 `experiments/export_results.py`）。

---

## 10. 常见问题（FAQ）

### Q1: baseline 如何关闭闭环控制？
使用 `experiments/configs/baseline_nsga2.yaml`，该配置用于无控制基线。

### Q2: 想做消融实验从哪里入手？
优先看：
- `experiments/ablations/switches.py`
- `experiments/ablations/matrix_runner.py`

### Q3: 如何保证可复现？
- 固定 `experiment.seed` 与 `optimizer.seed`
- 保存 `config_snapshot.yaml`
- 使用统一输出目录并保留 `summary.json` 与各类 `.jsonl` 日志

---

## 11. 许可证

当前仓库未单独提供 LICENSE 文件；如需对外发布或复用，请先补充明确许可证声明。
