# TECH_STACK.md

## 1. 目标

本文件用于约束项目的技术选型，避免在开发过程中频繁换栈、换工具、换依赖，导致工程结构不稳定。

本项目定位为：

- 研究型 Python 工程；
- 面向“LLM 闭环调控 NSGA-II”的多目标优化实验；
- 优先强调可运行、可维护、可测试、可扩展；
- 暂不追求前端、分布式、数据库重度工程化。

---

## 2. 总体选型原则

### 2.1 先轻量，后扩展
第一阶段优先使用：

- 本地文件存储；
- 轻量依赖；
- 清晰目录结构；
- 易于测试的模块设计。

不在一开始引入：

- 重型数据库；
- 消息队列；
- 前后端分离架构；
- 复杂服务编排。

### 2.2 标准 Python 工程优先
项目优先采用标准 Python 生态，确保：

- 本地开发简单；
- Codex 易于生成与维护；
- 后续迁移和复现方便。

### 2.3 配置驱动
尽可能使用配置文件控制实验参数，而不是把参数散落在代码里。

---

## 3. Python 与工程管理

### 3.1 Python 版本
推荐固定使用：

- **Python 3.11**

原因：

- 语法现代；
- 生态稳定；
- 类型注解支持成熟；
- 与大多数科学计算库兼容较好。

### 3.2 包管理与环境
推荐使用：

- `venv` 进行虚拟环境隔离；
- `pip` 安装依赖；
- `pyproject.toml` 管理项目元信息；
- `requirements.txt` 作为简洁依赖清单。

推荐原因：

- 对 Codex 最友好；
- 学习成本低；
- 足够支持当前研究项目；
- 不强依赖额外工具。

### 3.3 推荐基础命令

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

---

## 4. 核心依赖

### 4.1 数值计算
#### `numpy`
用途：

- 数组与向量操作；
- 个体编码处理；
- 目标函数计算；
- 统计指标计算。

结论：**必选**

---

### 4.2 数据模型与校验
#### `pydantic`
用途：

- 定义结构化配置；
- 定义 LLM 输出模型；
- 校验 `ParetoState`、`ControlAction`、`ExperienceRecord` 等数据结构；
- 降低字符串拼接式编程的出错率。

结论：**强烈推荐**

如果你希望更轻量，也可以在局部使用 `dataclasses`，但配置和 LLM 输出建议优先使用 `pydantic`。

---

### 4.3 配置文件
#### `pyyaml`
用途：

- 读取实验配置；
- 区分 baseline / ablation / llm 模式配置；
- 保存配置快照。

结论：**必选**

---

### 4.4 绘图
#### `matplotlib`
用途：

- 绘制 HV 曲线；
- 绘制可行率曲线；
- 绘制参数变化轨迹；
- 绘制 reward 变化。

结论：**必选**

---

### 4.5 HTTP 调用
#### `httpx`
用途：

- 调用 LLM API；
- 调用外部服务；
- 替代 `requests` 作为更现代的 HTTP 客户端。

结论：**推荐**

---

### 4.6 重试机制
#### `tenacity`
用途：

- LLM 调用失败时自动重试；
- 临时网络波动恢复；
- 统一封装 API 稳定性处理。

结论：**推荐**

---

## 5. 测试与质量工具

### 5.1 测试框架
#### `pytest`
用途：

- 单元测试；
- 集成测试；
- 参数化测试；
- 验证主流程是否可运行。

结论：**必选**

---

### 5.2 代码检查
#### `ruff`
用途：

- lint；
- import 排序；
- 风格统一；
- 发现明显代码问题。

结论：**强烈推荐**

---

## 6. 可选依赖

### 6.1 数据分析
#### `pandas`
用途：

- 分析实验日志；
- 读取和汇总 CSV；
- 便于后期画图和做对照表。

结论：**推荐，但不是首批必须**

---

### 6.2 多目标优化参考库
#### `pymoo`
用途：

- 作为多目标优化基线；
- 校验自实现 NSGA-II 的行为；
- 后期进行方法对照实验。

结论：**后期可选，不建议一开始就强依赖**

原因：

- 第一阶段更适合先自己实现一版清晰的 NSGA-II；
- 否则项目容易过度依赖外部框架，削弱你对内部结构的控制。

---

### 6.3 Web 服务
#### `fastapi`
用途：

- 如果未来要把控制器包装成服务；
- 如果需要做远程调用或前端对接；
- 如果要把实验系统暴露为 API。

结论：**当前阶段不必引入**

---

## 7. 存储方案

### 7.1 当前阶段推荐
优先使用：

- `jsonl`
- `csv`
- `yaml`
- 本地目录结构

用途：

- 记录逐代日志；
- 保存实验配置；
- 保存状态与动作轨迹；
- 保存经验池快照。

原因：

- 轻量；
- 易于调试；
- 易于版本管理；
- 适合研究实验。

### 7.2 后续可扩展
如果后期确实需要，再考虑：

- `sqlite3`
- `mysql`
- 远程对象存储

结论：**第一阶段不要上数据库**

---

## 8. LLM 接入策略

### 8.1 第一阶段
必须支持：

- `rule` 模式；
- `mock_llm` 模式；

即使没有真实 API Key，也必须能跑通主流程。

### 8.2 第二阶段
再接入：

- `real_llm` 模式；
- 统一 `LLMClient` 封装；
- 提示词模板文件；
- 输出结构化校验。

### 8.3 约束
禁止：

- 把 API Key 写死在代码里；
- 把 prompt 文本散落在多个业务模块中；
- 让优化器直接依赖 HTTP 调用。

---

## 9. 日志与实验输出格式

### 9.1 推荐输出格式
- `jsonl`：结构化逐代日志
- `csv`：便于统计和画图
- `yaml/json`：保存配置快照

### 9.2 推荐日志字段
每代建议至少记录：

- `generation`
- `hv`
- `delta_hv`
- `feasible_ratio`
- `rank1_ratio`
- `mean_cv`
- `diversity_score`
- `stagnation_len`
- `mutation_prob`
- `crossover_prob`
- `reward`

---

## 10. 推荐项目开发工具链

### 10.1 编辑器
推荐以下任意一种：

- VS Code
- Cursor
- Windsurf

### 10.2 终端
推荐：

- macOS/Linux 终端
- Windows 使用 PowerShell 或 WSL

### 10.3 版本控制
必须使用：

- `git`

建议从项目一开始就初始化 Git 仓库。

---

## 11. 推荐依赖分层

### 11.1 第一批必须安装
```text
numpy
pydantic
pyyaml
pytest
matplotlib
ruff
httpx
tenacity
```

### 11.2 第二批推荐安装
```text
pandas
```

### 11.3 后期按需安装
```text
pymoo
fastapi
```

---

## 12. requirements.txt 建议初稿

```txt
numpy
pydantic
pyyaml
pytest
matplotlib
ruff
httpx
tenacity
pandas
```

---

## 13. pyproject.toml 约束建议

项目后续应使用 `pyproject.toml` 统一管理：

- Python 版本要求；
- 项目元信息；
- ruff 配置；
- pytest 配置；
- 包结构设置。

---

## 14. 不推荐的当前阶段做法

当前阶段不推荐：

1. 一开始就引入 Docker 复杂编排；
2. 一开始就引入 MySQL；
3. 一开始就依赖外部工作流平台；
4. 一开始就把 NSGA-II 完全交给第三方黑盒框架；
5. 一开始就做 Web 前端；
6. 一开始就引入过多异步机制。

---

## 15. 一句话总结

本项目的推荐技术栈是：

- **Python 3.11**
- **venv + pip + pyproject.toml**
- **numpy + pydantic + pyyaml**
- **pytest + ruff**
- **matplotlib + httpx + tenacity**
- **本地 jsonl/csv/yaml 存储**
- **前期 rule/mock_llm，后期再接 real_llm**

目标是：**先把研究型闭环优化系统稳定搭起来，再逐步增强，而不是一开始就过度工程化。**