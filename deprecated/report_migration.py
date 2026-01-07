# ./component/report_migration.py
import argparse
import json
from pathlib import Path
import csv
from collections import Counter


def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, default="data/migration_plan.jsonl",
                        help="迁移计划文件")
    parser.add_argument("--out-csv", type=str, default="reports/migration_report.csv",
                        help="导出 CSV 路径")
    parser.add_argument("--topk", type=int, default=20,
                        help="打印前 topk 映射")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    items = load_jsonl(plan_path)
    if not items:
        print(f"[WARN] {plan_path} 为空")
        return

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(exist_ok=True, parents=True)

    total_pairs = len(items)
    with_tf = sum(1 for r in items if r.get("num_tf_tests", 0) > 0)
    with_pt = sum(1 for r in items if r.get("num_pt_tests", 0) > 0)
    with_both = sum(1 for r in items if (r.get("num_tf_tests", 0) > 0 and r.get("num_pt_tests", 0) > 0))

    total_tf_tests = sum(r.get("num_tf_tests", 0) for r in items)
    total_pt_tests = sum(r.get("num_pt_tests", 0) for r in items)

    # 统计每个 tf_api / pt_api 覆盖的测试数
    tf_api_cnt = Counter()
    pt_api_cnt = Counter()
    for r in items:
        tf_api = r.get("tf_api")
        pt_api = r.get("pt_api")
        n_tf = r.get("num_tf_tests", 0)
        n_pt = r.get("num_pt_tests", 0)
        if tf_api:
            tf_api_cnt[tf_api] += n_tf
        if pt_api:
            pt_api_cnt[pt_api] += n_pt

    # 导出 CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "tf_api", "pt_api", "score",
            "num_tf_tests", "num_pt_tests"
        ])
        for r in items:
            w.writerow([
                r.get("tf_api"),
                r.get("pt_api"),
                r.get("score"),
                r.get("num_tf_tests", 0),
                r.get("num_pt_tests", 0),
            ])

    print("\n==== MIGRATION REPORT SUMMARY ====")
    print(f"总映射对数: {total_pairs}")
    print(f"  - 覆盖 TF 测试的映射数: {with_tf} ({with_tf/total_pairs:.2%})")
    print(f"  - 覆盖 PT 测试的映射数: {with_pt} ({with_pt/total_pairs:.2%})")
    print(f"  - 同时覆盖 TF/PT 测试的映射数: {with_both} ({with_both/total_pairs:.2%})")
    print(f"总可迁移 TF 测试用例数: {total_tf_tests}")
    print(f"总可参考 PT 测试用例数: {total_pt_tests}")
    print(f"CSV 报告已写入: {out_csv}")

    # 打印 top-k 映射（按 TF 测试覆盖数排序）
    print(f"\nTop {args.topk} 映射（按可迁移 TF 测试数量排序）:")
    top_items = sorted(items, key=lambda r: r.get("num_tf_tests", 0), reverse=True)[:args.topk]
    for i, r in enumerate(top_items, 1):
        print(f"{i:2d}. {r.get('tf_api')}  ->  {r.get('pt_api')}  "
              f"| score={r.get('score'):.3f}  "
              f"| TF_tests={r.get('num_tf_tests', 0)}  "
              f"| PT_tests={r.get('num_pt_tests', 0)}")

    # 也可以顺便给你一段“PPT 背景稿”的模板文案：
    print("\n=== 建议汇报话术示例（可复制到 PPT 备注） ===")
    print(
        "本阶段基于组件级 API 映射结果，系统性地梳理了可迁移的单元测试空间。\n"
        f"共识别出 {total_pairs} 条 TensorFlow↔PyTorch API 映射，其中约 {with_tf/total_pairs:.1%} "
        "的映射在 TensorFlow 侧已经存在配套测试用例，为后续自动化迁移提供了直接支撑。"
        f"在这些映射下，累计发现可迁移的 TF 测试用例 {total_tf_tests} 条，"
        f"其中有 {with_both} 条映射在两侧均存在测试实现，可以作为迁移质量评估的对照组。"
    )


if __name__ == "__main__":
    main()
