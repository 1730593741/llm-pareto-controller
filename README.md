# LLM-Pareto-Controller

A research-oriented Python project for **LLM closed-loop control of NSGA-II** in multi-objective task assignment.

Current progress includes:
- M4: runnable rule-based closed loop (`state` + `action` logging)
- M5 MVP: optional experience memory (`state -> action -> reward -> next_state`)

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

## Testing

```bash
pytest -q
ruff check .
```
