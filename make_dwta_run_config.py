from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    """
    用法:
        python make_dwta_run_config.py <src_yaml> <mode> <memory_enabled:true|false> <seed> <output_dir> <dst_yaml>

    示例:
        python make_dwta_run_config.py experiments/configs/dwta_small_smoke.yaml real_llm true 42 experiments/runs/dwta_main_results/real_llm/seed_42 tmp_seed_configs_dwta/real_llm_dwta_seed_42.yaml
    """
    if len(sys.argv) != 7:
        print(
            "Usage: python make_dwta_run_config.py <src_yaml> <mode> <memory_enabled:true|false> <seed> <output_dir> <dst_yaml>",
            file=sys.stderr,
        )
        return 2

    src_yaml = Path(sys.argv[1])
    mode = sys.argv[2].strip()
    memory_enabled_raw = sys.argv[3].strip().lower()
    seed = int(sys.argv[4])
    output_dir = sys.argv[5]
    dst_yaml = Path(sys.argv[6])

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

    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = seed

    cfg.setdefault("optimizer", {})
    cfg["optimizer"]["seed"] = seed

    cfg.setdefault("controller_mode", {})
    cfg["controller_mode"]["mode"] = mode

    cfg.setdefault("memory", {})
    cfg["memory"]["enabled"] = memory_enabled

    cfg.setdefault("logging", {})
    cfg["logging"]["output_dir"] = output_dir

    # 关键：显式写死 llm 配置，避免任何默认值/旧模板歧义
    cfg["llm"] = {
        "provider": "openai",
        "model": "qwen3-max",
        "timeout_s": 10.0,
        "max_retries": 2,
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "fallback_mode": "mock_llm",
    }

    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    with dst_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Wrote DWTA run config: {dst_yaml}")
    print(f"     mode={mode}")
    print(f"     memory.enabled={memory_enabled}")
    print(f"     seed={seed}")
    print(f"     output_dir={output_dir}")
    print("     llm.api_key_env=OPENAI_API_KEY")
    print("     llm.base_url_env=OPENAI_BASE_URL")
    print("     llm.model_env=OPENAI_MODEL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())