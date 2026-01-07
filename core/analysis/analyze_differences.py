import json
import re
import csv
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Set

# ========== 可调参数 ==========
FUSED_PATH = "../data/recall_pairs_fused.jsonl"
TF_PATH    = "../data/tests_tf.mapped.jsonl"
PT_PATH    = "../data/tests_pt.mapped.jsonl"
OUT_SUMMARY_CSV = "../data/diff_types_summary.csv"
OUT_SAMPLES_JSONL = "../data/diff_types_samples.jsonl"

FINAL_SCORE_THRESHOLD = 0.5
API_JACCARD_LOW = 0.5
ASSERT_JACCARD_LOW = 0.5
TRIVIAL_API_MAX = 1           # apis_mapped 数量 <= 1 视为“弱/轻”测试
TRIVIAL_ASSERT_MAX = 0        # asserts 数量 <= 0
SAMPLE_PER_TYPE = 10          # 每类输出多少个样例

# ========== 工具函数 ==========
def load_jsonl(path: str) -> List[dict]:
    return [json.loads(line) for line in open(path, "r")]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0

def tokenize_name(name: str) -> List[str]:
    if not name:
        return []
    # 分词：下划线+驼峰
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").lower().split()
    return [p for p in parts if p]

def family_from_api_id(api_id: str) -> str:
    """
    把 API_ID 归纳为家族：CONV/DENSE/POOL/NORM/OPT/LOSS/ACT/OTHER
    期望传入 apis_mapped 中的元素，如 'API_CONV2D', 'API_DENSE', 'API_RELU' ...
    """
    s = api_id.upper()
    if "CONV" in s:
        return "CONV"
    if "DENSE" in s or "LINEAR" in s or "FC" in s:
        return "DENSE"
    if "POOL" in s or "POOLING" in s:
        return "POOL"
    if "BATCHNORM" in s or "NORM" in s or "NORMALIZATION" in s:
        return "NORM"
    if "OPT_" in s or "OPTIM" in s:
        return "OPT"
    if "LOSS" in s or "CROSSENTROPY" in s or "MSE" in s:
        return "LOSS"
    if "RELU" in s or "SIGMOID" in s or "TANH" in s or "ACT" in s:
        return "ACT"
    return "OTHER"

def dominant_family(apis_mapped: List[str]) -> str:
    if not apis_mapped:
        return "NONE"
    fams = [family_from_api_id(a) for a in apis_mapped]
    return Counter(fams).most_common(1)[0][0]

# ========== 主逻辑 ==========
def main():
    # 读取数据
    pairs_all = load_jsonl(FUSED_PATH)
    tf_tests = load_jsonl(TF_PATH)
    pt_tests = load_jsonl(PT_PATH)

    # 建索引
    tf_idx = {(t.get("file"), t.get("name")): t for t in tf_tests}
    pt_idx = {(t.get("file"), t.get("name")): t for t in pt_tests}

    # 过滤高分对
    pairs = [p for p in pairs_all if p.get("final_score", 0.0) >= FINAL_SCORE_THRESHOLD]
    total = len(pairs)
    print(f"[INFO] 高分对数量: {total} (final_score >= {FINAL_SCORE_THRESHOLD})")

    # 辅助检测：一对多/多对一
    tf_key_counts = Counter([(p["tf_file"], p["tf_name"]) for p in pairs])
    pt_key_counts = Counter([(p["pt_file"], p["pt_name"]) for p in pairs])

    # 分类容器
    buckets: Dict[str, List[dict]] = defaultdict(list)

    for p in pairs:
        tf_t = tf_idx.get((p["tf_file"], p["tf_name"]))
        pt_t = pt_idx.get((p["pt_file"], p["pt_name"]))
        if not tf_t or not pt_t:
            # 元数据缺失，归入 OTHER
            p["diff_reason"] = "MISSING_META"
            buckets["OTHER"].append(p)
            continue

        apis_tf = tf_t.get("apis_mapped", []) or []
        apis_pt = pt_t.get("apis_mapped", []) or []
        asserts_tf = tf_t.get("asserts", []) or []
        asserts_pt = pt_t.get("asserts", []) or []

        api_j = jaccard(apis_tf, apis_pt)
        asr_j = jaccard(asserts_tf, asserts_pt)

        # 主导家族
        fam_tf = dominant_family(apis_tf)
        fam_pt = dominant_family(apis_pt)

        # 名称 token（可作为辅助判读/报告展示）
        name_tf_tokens = tokenize_name(tf_t.get("name", ""))
        name_pt_tokens = tokenize_name(pt_t.get("name", ""))

        # 标注
        labels: Set[str] = set()

        if api_j < API_JACCARD_LOW:
            labels.add("API_OVERLAP_LOW")

        if asr_j < ASSERT_JACCARD_LOW:
            labels.add("ASSERT_OVERLAP_LOW")

        # 家族冲突（都不为 NONE 且不同）
        if fam_tf != "NONE" and fam_pt != "NONE" and fam_tf != fam_pt:
            labels.add("CROSS_FAMILY_CONFLICT")

        # 一对多/多对一
        if tf_key_counts[(p["tf_file"], p["tf_name"])] > 1:
            labels.add("MANY_TO_ONE_TF")
        if pt_key_counts[(p["pt_file"], p["pt_name"])] > 1:
            labels.add("MANY_TO_ONE_PT")

        # “轻/泛”测试
        if (len(apis_tf) <= TRIVIAL_API_MAX and len(asserts_tf) <= TRIVIAL_ASSERT_MAX) or \
           (len(apis_pt) <= TRIVIAL_API_MAX and len(asserts_pt) <= TRIVIAL_ASSERT_MAX) or \
           (tf_t.get("name","").lower() in {"test", "testfn", "test_fn"} or
            pt_t.get("name","").lower() in {"test", "testfn", "test_fn"}):
            labels.add("TRIVIAL_TEST")

        # 如果没有命中任何“差异类”，视为高度一致
        if not labels:
            labels.add("IDENTICAL_SEMANTICS")

        # 将该 pair 复制一份带上分析字段
        enriched = dict(p)
        enriched.update({
            "api_jaccard": api_j,
            "assert_jaccard": asr_j,
            "tf_family": fam_tf,
            "pt_family": fam_pt,
            "tf_api_cnt": len(apis_tf),
            "pt_api_cnt": len(apis_pt),
            "tf_assert_cnt": len(asserts_tf),
            "pt_assert_cnt": len(asserts_pt),
            "tf_name_tokens": name_tf_tokens,
            "pt_name_tokens": name_pt_tokens,
        })

        # 放入所有命中的类别；一个 pair 可能进入多个桶
        for lb in labels:
            buckets[lb].append(enriched)

    # 汇总统计
    total_pairs = len(pairs)
    rows = []
    print("\n[SUMMARY] 差异类型分布：")
    for lb, items in sorted(buckets.items(), key=lambda x: -len(x[1])):
        count = len(items)
        ratio = count / total_pairs if total_pairs > 0 else 0.0
        print(f"  - {lb:22s}: {count:5d}  ({ratio:.2%})")
        rows.append([lb, count, f"{ratio:.2%}"])

    # 写 CSV 汇总
    Path(OUT_SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["diff_type", "count", "ratio"])
        writer.writerows(rows)
    print(f"\n[OUTPUT] 汇总写入: {OUT_SUMMARY_CSV}")

    # 写每类样例（前 SAMPLE_PER_TYPE 条）
    with open(OUT_SAMPLES_JSONL, "w") as f:
        for lb, items in buckets.items():
            for ex in items[:SAMPLE_PER_TYPE]:
                out = dict(ex)
                out["diff_type"] = lb
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[OUTPUT] 样例写入: {OUT_SAMPLES_JSONL}  (每类最多 {SAMPLE_PER_TYPE} 条)\n")

    # 额外：打印一个“家族分布矩阵”（帮助了解主题差异）
    print("[MATRIX] 主导家族组合分布（tf_family x pt_family）：")
    matrix = Counter((e["tf_family"], e["pt_family"]) for e in sum(buckets.values(), []))
    # 只展示出现次数 >= 5 的组合
    for (tf_fam, pt_fam), cnt in sorted(matrix.items(), key=lambda x: -x[1]):
        if cnt >= 5:
            print(f"  ({tf_fam:5s} -> {pt_fam:5s}) : {cnt}")

if __name__ == "__main__":
    main()
