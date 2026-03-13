# Experiments Workflow (toy -> pilot -> full paper matrix)

## 1) Matrix design

### Matched experiment matrix
- Methods: `baseline_nsga2`, `rule_control`, `mock_llm`, `real_llm`.
- Shared axes: same `seeds`, `generations`, `population_size`, `benchmark`.
- Supported benchmarks: `small_complex_smoke`, `small_complex`, `medium_complex`, `hard_complex`.

### Ablation matrix
- `no_pareto_state_deep_features`
- `no_experience_pool`
- `binary_state_machine` vs `four_state_machine`
- `pc_pm_only` vs `extended_action_space`
- `tau in {1, 3, 5, 10}`
- `memory window in {5, 20, 50}`

## 2) Run commands

### A. Toy smoke (fast)
```bash
python -m experiments.run_matrix --preset toy --output-root experiments/runs/toy
```

### B. Matched pilot
```bash
python -m experiments.run_matrix --preset pilot --skip-ablation --output-root experiments/runs/pilot_matched
```

### C. Full paper matrix (matched + ablation)
```bash
python -m experiments.run_matrix --preset paper --output-root experiments/runs/paper
```

## 3) Export aggregated results (machine-readable + paper-table inputs)

```bash
python -m experiments.export_results --runs-root experiments/runs/paper --output-dir experiments/exports/paper
```

Outputs:
- `aggregated_runs.csv`
- `aggregated_runs.json`
- `paper_table_method.csv`
- `paper_table_method_benchmark.csv`

`aggregated_runs.*` fields include:
- `method`
- `benchmark`
- `seed`
- `hv`
- `igd_plus`
- `spacing`
- `spread`
- `feasible_ratio`
- `runtime`
- `llm_overhead`

## 4) real_llm environment variables and fallback

`real_llm` mode reads:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (optional)

Fallback behavior (configured by `llm.fallback_mode`):
- `mock_llm` (default): API failures degrade to mock responses, so default experiments stay runnable without online API.
- `hold`: if API fails, keep current parameters without LLM updates.

## 5) Recommended order for writing paper experiments

1. **Toy smoke**: validate matrix runner + logs + exporter.
2. **Pilot matched**: validate method ranking and stability on one benchmark.
3. **Full paper matrix**: run all matched + ablation settings, then export final tables.
