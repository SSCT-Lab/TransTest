# core/pick_high_agreement_cases.py
import json, argparse, random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DEF_FUSED = DATA / "recall_pairs_fused.jsonl"
DEF_TF = DATA / "tests_tf.mapped.jsonl"
DEF_PT = DATA / "tests_pt.mapped.jsonl"
DEF_LLM = DATA / "diff_types_llm.jsonl"
DEF_OUT = DATA / "high_agreement_cases.jsonl"

LABEL_ORDER = [
    "CROSS_FAMILY_CONFLICT", "API_OVERLAP_LOW", "ASSERT_OVERLAP_LOW",
    "MANY_TO_ONE_TF", "MANY_TO_ONE_PT", "TRIVIAL_TEST", "IDENTICAL_SEMANTICS", "OTHER"
]


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, "r")]


def pair_key(r):
    return (r.get("tf_file"), r.get("tf_name"), r.get("pt_file"), r.get("pt_name"))


def jaccard(a, b):
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    u = len(sa | sb);
    i = len(sa & sb)
    return i / u if u else 0.0


def _family(api_id: str) -> str:
    s = (api_id or "").upper()
    if "CONV" in s: return "CONV"
    if any(k in s for k in ["DENSE", "LINEAR", "FC"]): return "DENSE"
    if "POOL" in s: return "POOL"
    if any(k in s for k in ["BATCHNORM", "NORM", "NORMALIZATION"]): return "NORM"
    if any(k in s for k in ["RELU", "SIGMOID", "TANH", "ACT"]): return "ACT"
    return "OTHER"


def _dom_family(apis):
    if not apis: return "NONE"
    c = Counter(_family(a) for a in apis)
    return c.most_common(1)[0][0]


def compute_rule_labels(fused, tf_tests, pt_tests, threshold=0.5):
    tf_idx = {(t.get("file"), t.get("name")): t for t in tf_tests}
    pt_idx = {(t.get("file"), t.get("name")): t for t in pt_tests}
    pairs = [p for p in fused if p.get("final_score", 0.0) >= threshold]

    tf_counts = Counter((p["tf_file"], p["tf_name"]) for p in pairs)
    pt_counts = Counter((p["pt_file"], p["pt_name"]) for p in pairs)

    rule = {}
    for p in pairs:
        tf_t = tf_idx.get((p["tf_file"], p["tf_name"]), {})
        pt_t = pt_idx.get((p["pt_file"], p["pt_name"]), {})

        apis_tf = tf_t.get("apis_mapped", []) or []
        apis_pt = pt_t.get("apis_mapped", []) or []
        asserts_tf = tf_t.get("asserts", []) or []
        asserts_pt = pt_t.get("asserts", []) or []

        api_j = jaccard(apis_tf, apis_pt)
        asr_j = jaccard(asserts_tf, asserts_pt)
        fam_tf = _dom_family(apis_tf);
        fam_pt = _dom_family(apis_pt)

        labels = set()
        if api_j < 0.5: labels.add("API_OVERLAP_LOW")
        if asr_j < 0.5: labels.add("ASSERT_OVERLAP_LOW")
        if fam_tf != "NONE" and fam_pt != "NONE" and fam_tf != fam_pt:
            labels.add("CROSS_FAMILY_CONFLICT")
        if tf_counts[(p["tf_file"], p["tf_name"])] > 1: labels.add("MANY_TO_ONE_TF")
        if pt_counts[(p["pt_file"], p["pt_name"])] > 1: labels.add("MANY_TO_ONE_PT")
        if (len(apis_tf) <= 1 and len(asserts_tf) == 0) or (len(apis_pt) <= 1 and len(asserts_pt) == 0):
            labels.add("TRIVIAL_TEST")
        if not labels: labels.add("IDENTICAL_SEMANTICS")

        # 主标签按优先序挑一个，保证稳定
        primary = next((lab for lab in LABEL_ORDER if lab in labels), "OTHER")
        k = (p["tf_file"], p["tf_name"], p["pt_file"], p["pt_name"])
        rule[k] = {
            "rule_label": primary,
            "api_jaccard": api_j, "assert_jaccard": asr_j,
            "tf_family": fam_tf, "pt_family": fam_pt,
            "final_score": p.get("final_score", 0.0)
        }
    return rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default=str(DEF_FUSED))
    ap.add_argument("--tf", default=str(DEF_TF))
    ap.add_argument("--pt", default=str(DEF_PT))
    ap.add_argument("--llm", default=str(DEF_LLM))
    ap.add_argument("--threshold", type=float, default=0.5, help="final_score 下限")
    ap.add_argument("--labels", type=str,
                    default="API_OVERLAP_LOW,TRIVIAL_TEST,IDENTICAL_SEMANTICS",
                    help="要抽样的一致标签(逗号分隔)")
    ap.add_argument("--topn", type=int, default=10, help="每个标签导出案例数")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    random.seed(args.seed)

    fused = load_jsonl(Path(args.fused))
    tf_tests = load_jsonl(Path(args.tf))
    pt_tests = load_jsonl(Path(args.pt))
    llm_rows = load_jsonl(Path(args.llm))

    # 现场算规则标签
    rule_map = compute_rule_labels(fused, tf_tests, pt_tests, threshold=args.threshold)

    # LLM 索引
    llm_idx = {pair_key(r): r for r in llm_rows}

    want_labels = {s.strip() for s in args.labels.split(",") if s.strip()}

    # 组装一致样本
    pool = []
    for k, rinfo in rule_map.items():
        if k not in llm_idx:  # LLM 未覆盖的跳过
            continue
        llm = llm_idx[k]
        # 取 LLM 主标签（按同一优先序）
        l_primary = next((lab for lab in LABEL_ORDER if lab in (llm.get("labels", []) or ["OTHER"])), "OTHER")
        if rinfo["rule_label"] == l_primary and rinfo["rule_label"] in want_labels:
            tf_file, tf_name, pt_file, pt_name = k
            pool.append({
                "tf_file": tf_file, "tf_name": tf_name,
                "pt_file": pt_file, "pt_name": pt_name,
                "label": rinfo["rule_label"],
                "final_score": rinfo["final_score"],
                "api_jaccard": rinfo["api_jaccard"],
                "assert_jaccard": rinfo["assert_jaccard"],
                "tf_family": rinfo["tf_family"],
                "pt_family": rinfo["pt_family"]
            })

    # 分标签取前 N（按 final_score ↓，再 api_jaccard/assert_jaccard 辅助）
    out = []
    for lab in want_labels:
        cand = [x for x in pool if x["label"] == lab]
        cand.sort(key=lambda x: (x["final_score"], x["api_jaccard"], x["assert_jaccard"]), reverse=True)
        out.extend(cand[:args.topn])

    # 写出
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 终端预览
    print(f"[OK] 一致样本导出 {len(out)} 条 -> {out_path}")
    by_lab = Counter([r["label"] for r in out])
    for lab, cnt in by_lab.items():
        print(f"  - {lab}: {cnt}")
    if out:
        s = out[0]
        print("\n[示例 CASE]\n"
              f"Label     : {s['label']}\n"
              f"TF        : {s['tf_file']} :: {s['tf_name']}\n"
              f"PT        : {s['pt_file']} :: {s['pt_name']}\n"
              f"final     : {s['final_score']:.3f}\n"
              f"apiJ/asrJ : {s['api_jaccard']:.2f} / {s['assert_jaccard']:.2f}\n"
              f"family    : {s['tf_family']} → {s['pt_family']}")


if __name__ == "__main__":
    main()
