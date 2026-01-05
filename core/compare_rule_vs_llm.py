from pathlib import Path
import json, csv
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RULE_SAMPLES = DATA / "diff_types_samples.jsonl"     # 规则版逐条样例（前面脚本生成）
LLM_JSONL    = DATA / "diff_types_llm.jsonl"         # 大模型版逐条结果
OUT_SUMMARY  = DATA / "compare_rule_vs_llm_summary.csv"

def read_jsonl(p):
    return [json.loads(l) for l in open(p, "r")]

def primary_label_rule(r):
    # 规则版是按“桶”写样例的，这里从样例的 diff_type 直接当作主标签
    return r.get("diff_type","OTHER")

def primary_label_llm(r):
    # LLM 多选标签时，取一个主标签用于混淆矩阵/一致性统计
    order = ["CROSS_FAMILY_CONFLICT","API_OVERLAP_LOW","ASSERT_OVERLAP_LOW",
             "MANY_TO_ONE_TF","MANY_TO_ONE_PT","TRIVIAL_TEST",
             "IDENTICAL_SEMANTICS","OTHER"]
    labels = r.get("labels",[]) or ["OTHER"]
    for k in order:
        if k in labels:
            return k
    return labels[0]

def pair_key(r):
    return (r.get("tf_file"), r.get("tf_name"), r.get("pt_file"), r.get("pt_name"))

def main():
    rule_rows = read_jsonl(RULE_SAMPLES)
    llm_rows  = read_jsonl(LLM_JSONL)

    # 对齐
    rule_idx = {pair_key(r): r for r in rule_rows}
    llm_idx  = {pair_key(r): r for r in llm_rows}
    keys = set(rule_idx.keys()) & set(llm_idx.keys())
    if not keys:
        print("[WARN] 没有重叠样本，检查输入文件是否对应同一批数据。")
        return

    cm = defaultdict(int)  # 混淆计数： (rule_label, llm_label) -> count
    both_any = 0
    agree = 0
    per_label_agree = Counter()
    per_label_total = Counter()

    for k in keys:
        r = rule_idx[k]; l = llm_idx[k]
        a = primary_label_rule(r)
        b = primary_label_llm(l)
        cm[(a,b)] += 1
        both_any += 1
        per_label_total[a] += 1
        if a == b:
            agree += 1
            per_label_agree[a] += 1

    acc = agree / both_any if both_any else 0.0
    print(f"[INFO] 对齐样本数: {both_any}, 主标签一致率: {acc:.2%}")

    # 输出混淆矩阵为 CSV（长表）
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule_label","llm_label","count"])
        for (a,b),cnt in sorted(cm.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            w.writerow([a,b,cnt])

    print(f"[INFO] 混淆长表写入: {OUT_SUMMARY}")
    print("\n[PER-LABEL ACCURACY]（以规则标签为参照）")
    for lab, tot in per_label_total.items():
        acc_lab = per_label_agree[lab]/tot if tot else 0.0
        print(f"  - {lab:22s}: {acc_lab:.2%}  (n={tot})")

if __name__ == "__main__":
    main()
