from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    """
    用法:
        python make_seed_config.py <src_yaml> <seed> <output_dir> <dst_yaml>

    示例:
        python make_seed_config.py experiments/configs/mock_llm.yaml 42 experiments/runs/mock_llm/seed_42 tmp_seed_configs/mock_llm_seed_42.yaml
    """
    if len(sys.argv) != 5:
        print(
            "Usage: python make_seed_config.py <src_yaml> <seed> <output_dir> <dst_yaml>",
            file=sys.stderr,
        )
        return 2

    src_yaml = Path(sys.argv[1])
    seed = int(sys.argv[2])
    output_dir = sys.argv[3]
    dst_yaml = Path(sys.argv[4])

    if not src_yaml.exists():
        print(f"[ERROR] Source config not found: {src_yaml}", file=sys.stderr)
        return 1

    with src_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        print("[ERROR] YAML root must be a mapping/object.", file=sys.stderr)
        return 1

    cfg.setdefault("experiment", {})
    if not isinstance(cfg["experiment"], dict):
        print("[ERROR] 'experiment' must be a mapping/object.", file=sys.stderr)
        return 1
    cfg["experiment"]["seed"] = seed

    cfg.setdefault("optimizer", {})
    if not isinstance(cfg["optimizer"], dict):
        print("[ERROR] 'optimizer' must be a mapping/object.", file=sys.stderr)
        return 1
    cfg["optimizer"]["seed"] = seed

    cfg.setdefault("logging", {})
    if not isinstance(cfg["logging"], dict):
        print("[ERROR] 'logging' must be a mapping/object.", file=sys.stderr)
        return 1
    cfg["logging"]["output_dir"] = output_dir

    dst_yaml.parent.mkdir(parents=True, exist_ok=True)

    with dst_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Wrote seed config: {dst_yaml}")
    print(f"     seed={seed}")
    print(f"     output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())