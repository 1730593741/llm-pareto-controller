# Project Goal

This repository exists only for paper experiments on LLM-closed-loop NSGA-II for the DWTA problem.

The purpose of this codebase is:
1. run reproducible DWTA experiments,
2. compare vanilla NSGA-II, rule-based closed-loop NSGA-II, and LLM closed-loop NSGA-II,
3. support ablation studies and reporting.

## Hard Constraints

- Do not introduce generic multi-problem abstractions.
- Do not add plugin systems, registries, or factories unless they are required by the paper experiments.
- Keep the code explicit and experiment-oriented.
- Preserve reproducibility: configs, seeds, metrics, and output artifacts must remain clear.
- Before deleting code, identify whether it supports baseline, rule, llm, or ablation experiments.
- Prefer small, reviewable changes over large rewrites.

## Architecture Direction

- `src/dwta`: problem definition, dynamics, objectives, constraints, encoding, evaluation
- `src/nsga2`: solver core and adjustable search knobs
- `src/control`: observation extraction, controller actions, rule controller, llm controller
- `src/experiment`: config, runner, logging, metrics, reports

## Controller Design

The controller must not directly manipulate arbitrary solver internals.
The controller should operate through an explicit action space.

Preferred controllable knobs:
- mutation rate
- crossover rate
- immigrant or restart ratio
- repair strength
- diversity bias
- selection pressure

## Experiment Design

The codebase must support:
- baseline: controller = none
- rule-based controller
- llm-based controller
- ablations by removing observations, history, or action dimensions

## Testing

Every refactor should keep:
- one tiny DWTA smoke test,
- one baseline regression case,
- one mock-llm closed-loop smoke test.
