import json
import pandas as pd
from pathlib import Path

# 您的硬核动态实验输出目录
base_dir = Path("experiments/runs/dwta_hard_results")
methods = ["baseline_nsga2", "rule_control", "mock_llm", "real_llm"]
seed = 2024

rows = []

for method in methods:
    # 拼接每个方法 seed 100 的 summary.json 路径
    summary_path = base_dir / method / f"seed_{seed}" / "summary.json"

    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 安全提取指标（如果是静态基线可能没有 dynamic_summary）
            dyn_summary = data.get("dynamic_summary", {})
            num_events = dyn_summary.get("num_events", 0)

            rows.append({
                "Method": method,
                "Final HV": round(data.get("final_hv", 0), 2),
                "Best HV": round(data.get("best_hv", 0), 2),
                "HV-AUC": round(data.get("hv_auc", 0), 2),
                "Feasible Ratio": round(data.get("final_feasible_ratio", 0), 4),
                "Actions": data.get("num_actions", 0),
                "Runtime (s)": round(data.get("runtime_s", 0), 2),
                "LLM Overhead (s)": round(data.get("llm_overhead_s", 0), 2),
                "Events Resisted": num_events
            })
    else:
        print(f"⚠️ 找不到文件: {summary_path}")

# 转换为 DataFrame 并保存
if rows:
    df = pd.DataFrame(rows)
    csv_filename = "dwta_hard_seed100_comparison.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✅ 成功生成对比表格：{csv_filename}\n")
    print(df.to_markdown(index=False))
else:
    print("❌ 没有找到任何数据，请检查目录路径是否正确。")