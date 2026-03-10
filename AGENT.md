# AGENTS.md

## Repository rules

1. Before making changes, always read:
   - SPEC.md
   - TASKS.md
   - TECH_STACK.md

2. Follow TASKS.md milestones strictly in order:
   - M0
   - M1
   - M2
   - M3
   - M4
   - M5
   - M6
   - M7
   - M8

3. Do not skip ahead to real LLM integration before MVP is running.

4. Keep the project modular:
   - problems/
   - optimizers/
   - sensing/
   - llm/
   - memory/
   - controller/
   - infra/
   - experiments/

5. Use Python 3.11.

6. Prefer:
   - numpy
   - pydantic
   - pyyaml
   - pytest
   - matplotlib
   - httpx
   - tenacity
   - ruff

7. Do not hardcode API keys.

8. Do not scatter prompts across business code. Put prompt templates under llm/prompts/.

9. Always use type hints and module-level docstrings.

10. For each task:
    - first propose a plan,
    - then list files to create or modify,
    - then implement,
    - then explain how to run and test.

11. Prioritize a minimal runnable implementation over overengineering.

12. Do not introduce databases, Docker, or frontend code in the early milestones unless explicitly requested.