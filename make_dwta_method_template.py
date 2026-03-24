from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    """
    用法:
        python make_dwta_method_template.py <src_yaml> <mode> <memory_enabled:true|false> <dst_yaml>

    示例:
        python make_dwta_method_template.py experiments/configs/dwta_small_smoke.yaml rule false tmp_seed_configs_dwta/baseline_dwta_template.yaml
    """
    if len(sys.argv) != 5:
        print(
            "Usage: python make_dwta_method_template.py <src_yaml> <mode> <memory_enabled:true|false> <dst_yaml>",
            file=sys.stderr,
        )
        return 2

    src_yaml = Path(sys.argv[1])
    mode = sys.argv[2].strip()
    memory_enabled_raw = sys.argv[3].strip().lower()
    dst_yaml = Path(sys.argv[4])

    if not src_yaml.exists():
        print(f"[ERROR] Source config not found: {src_yaml}", file=sys.stderr)
        return 1

    if mode not in {"rule", "mock_llm", "real_llm"}:
        print(f"[ERROR] Unsupported mode: {mode}", file=sys.stderr)
        return 1

    if memory_enabled_raw not in {"true", "false"}:
        print("[ERROR] memory_enabled must be true or false", file=sys.stderr)
        return 1

    memory_enabled = memory_enabled_raw == "true"

    with src_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        print("[ERROR] YAML root must be a mapping/object.", file=sys.stderr)
        return 1

    cfg.setdefault("controller_mode", {})
    if not isinstance(cfg["controller_mode"], dict):
        print("[ERROR] 'controller_mode' must be a mapping/object.", file=sys.stderr)
        return 1
    cfg["controller_mode"]["mode"] = mode

    cfg.setdefault("memory", {})
    if not isinstance(cfg["memory"], dict):
        print("[ERROR] 'memory' must be a mapping/object.", file=sys.stderr)
        return 1
    cfg["memory"]["enabled"] = memory_enabled

    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    with dst_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Wrote DWTA template: {dst_yaml}")
    print(f"     mode={mode}")
    print(f"     memory.enabled={memory_enabled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())