# LLM-Pareto-Controller

A research-oriented Python project for **LLM closed-loop control of NSGA-II** in multi-objective task assignment.

Current progress includes:
- M4: runnable rule-based closed loop (`state` + `action` logging)
- M5 MVP: optional experience memory (`state -> action -> reward -> next_state`)
- M6-B: pluggable controller mode (`rule` / `mock_llm` / `real_llm`) with structured Analyst -> Strategist -> Actuator chain and fallback safeguards

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python main.py
```

Example output:

```text
Run complete | generation=20 hv=... mutation_prob=... crossover_prob=... log_path=runs/m4/events.jsonl
```

## Configuration

Default config: `experiments/configs/default.yaml`

- `log_path`: state/action JSONL output
- `memory.enabled`: enable/disable M5 experience collection
- `memory.memory_window`: sliding window size of in-memory experience pool
- `memory.experience_log_path`: optional JSONL output for experiences
- `memory.reward_alpha` / `memory.reward_beta`: reward weights

When `memory.enabled: false`, the system remains in the M4-compatible rule loop path.

Controller mode config:
- `controller_mode.mode`: `rule` (default) / `mock_llm` / `real_llm`
- `controller_mode.experience_lookback`: experience window used by LLM chain

LLM runtime config:
- `llm.api_key_env`: env var name used to load API Key
- `llm.base_url_env`: optional env var name used to override endpoint URL
- `llm.model_env`: optional env var name used to override model name
- `llm.base_url`: default endpoint URL when `base_url_env` is unset
- `llm.fallback_mode`: `mock_llm` or `hold` when real call fails

Mock LLM run example:
```bash
python -c "from main import main; main('experiments/configs/mock_llm.yaml')"
```

Real LLM config example (still safe without key due to fallback):
```bash
python -c "from main import main; main('experiments/configs/real_llm.yaml')"
```

## Testing

```bash
pytest -q
ruff check .
```
