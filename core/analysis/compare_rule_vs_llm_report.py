import json, csv, argparse, io, base64, textwrap
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# 默认输入（沿用你已有命名）
DEFAULT_RULE_SAMPLES = DATA / "diff_types_samples.jsonl"    # 仅在未 --compute-rule 时使用
DEFAULT_LLM_JSONL    = DATA / "diff_types_llm.jsonl"        # 大模型标注（labels 多选）
DEFAULT_FUSED        = DATA / "recall_pairs_fused.jsonl"    # 含 final_score
DEFAULT_TF           = DATA / "tests_tf.mapped.jsonl"
DEFAULT_PT           = DATA / "tests_pt.mapped.jsonl"

OUT_DIR              = DATA
OUT_CONFUSION_CSV    = OUT_DIR / "compare_rule_vs_llm_confusion.csv"
OUT_PER_LABEL_CSV    = OUT_DIR / "compare_rule_vs_llm_perlabel_acc.csv"
OUT_HTML             = OUT_DIR / "compare_rule_vs_llm_report.html"
OUT_MD               = OUT_DIR / "compare_rule_vs_llm_report.md"

LABEL_ORDER = [
    "CROSS_FAMILY_CONFLICT",
    "API_OVERLAP_LOW",
    "ASSERT_OVERLAP_LOW",
    "MANY_TO_ONE_TF",
    "MANY_TO_ONE_PT",
    "TRIVIAL_TEST",
    "IDENTICAL_SEMANTICS",
    "OTHER"
]

# ---------------- IO ----------------
def read_jsonl(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return [json.loads(l) for l in open(p, "r")]

def pair_key(r):
    return (r.get("tf_file"), r.get("tf_name"), r.get("pt_file"), r.get("pt_name"))

def primary_label_llm(r):
    labels = r.get("labels", []) or ["OTHER"]
    for k in LABEL_ORDER:
        if k in labels:
            return k
    return labels[0]

# ---------------- 规则标签即时计算 ----------------
def jaccard(a, b):
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    u = len(sa | sb); i = len(sa & sb)
    return i / u if u else 0.0

def _family(api_id: str) -> str:
    s = (api_id or "").upper()
    if "CONV" in s: return "CONV"
    if any(k in s for k in ["DENSE","LINEAR","FC"]): return "DENSE"
    if "POOL" in s: return "POOL"
    if any(k in s for k in ["BATCHNORM","NORM","NORMALIZATION"]): return "NORM"
    if any(k in s for k in ["OPT_","OPTIM"]): return "OPT"
    if any(k in s for k in ["LOSS","CROSSENTROPY","MSE"]): return "LOSS"
    if any(k in s for k in ["RELU","SIGMOID","TANH","ACT"]): return "ACT"
    return "OTHER"

def _dom_family(apis):
    if not apis: return "NONE"
    c = Counter(_family(a) for a in apis)
    return c.most_common(1)[0][0]

def compute_rule_labels_on_the_fly(
    fused_path: Path,
    tf_path: Path,
    pt_path: Path,
    final_score_threshold: float
):
    """返回 dict: key(pair_key) -> rule_label"""
    fused = read_jsonl(fused_path)
    tf_tests = read_jsonl(tf_path)
    pt_tests = read_jsonl(pt_path)

    tf_idx = {(t.get("file"), t.get("name")): t for t in tf_tests}
    pt_idx = {(t.get("file"), t.get("name")): t for t in pt_tests}

    # 仅保留高分对
    pairs = [p for p in fused if p.get("final_score", 0.0) >= final_score_threshold]

    # 统计一对多
    tf_counts = Counter((p["tf_file"], p["tf_name"]) for p in pairs)
    pt_counts = Counter((p["pt_file"], p["pt_name"]) for p in pairs)

    API_JACCARD_LOW = 0.5
    ASSERT_JACCARD_LOW = 0.5
    TRIVIAL_API_MAX = 1
    TRIVIAL_ASSERT_MAX = 0

    res = {}
    for p in pairs:
        tf_t = tf_idx.get((p["tf_file"], p["tf_name"]), {})
        pt_t = pt_idx.get((p["pt_file"], p["pt_name"]), {})

        apis_tf = tf_t.get("apis_mapped", []) or []
        apis_pt = pt_t.get("apis_mapped", []) or []
        asserts_tf = tf_t.get("asserts", []) or []
        asserts_pt = pt_t.get("asserts", []) or []

        api_j = jaccard(apis_tf, apis_pt)
        asr_j = jaccard(asserts_tf, asserts_pt)

        fam_tf = _dom_family(apis_tf)
        fam_pt = _dom_family(apis_pt)

        labels = set()
        if api_j < API_JACCARD_LOW: labels.add("API_OVERLAP_LOW")
        if asr_j < ASSERT_JACCARD_LOW: labels.add("ASSERT_OVERLAP_LOW")
        if fam_tf != "NONE" and fam_pt != "NONE" and fam_tf != fam_pt:
            labels.add("CROSS_FAMILY_CONFLICT")
        if tf_counts[(p["tf_file"], p["tf_name"])] > 1: labels.add("MANY_TO_ONE_TF")
        if pt_counts[(p["pt_file"], p["pt_name"])] > 1: labels.add("MANY_TO_ONE_PT")
        # 轻/泛测试
        if (len(apis_tf) <= TRIVIAL_API_MAX and len(asserts_tf) <= TRIVIAL_ASSERT_MAX) or \
           (len(apis_pt) <= TRIVIAL_API_MAX and len(asserts_pt) <= TRIVIAL_ASSERT_MAX) or \
           (str(tf_t.get("name","")).lower() in {"test","testfn","test_fn"} or
            str(pt_t.get("name","")).lower() in {"test","testfn","test_fn"}):
            labels.add("TRIVIAL_TEST")
        if not labels:
            labels.add("IDENTICAL_SEMANTICS")

        key = (p["tf_file"], p["tf_name"], p["pt_file"], p["pt_name"])
        # 主标签：取 LABEL_ORDER 中优先出现的一个，保证稳定
        primary = None
        for lab in LABEL_ORDER:
            if lab in labels:
                primary = lab; break
        primary = primary or "OTHER"
        res[key] = primary
    return res

# ---------------- 画图 ----------------
def fig_to_base64(fig, bbox_inches="tight", dpi=160):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches=bbox_inches, dpi=dpi)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def plot_confusion(cm_counts, labels):
    L = labels
    mat = np.zeros((len(L), len(L)), dtype=int)
    for (r, l), cnt in cm_counts.items():
        if r in L and l in L:
            mat[L.index(r), L.index(l)] = cnt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(L))); ax.set_xticklabels(L, rotation=45, ha="right")
    ax.set_yticks(range(len(L))); ax.set_yticklabels(L)
    ax.set_xlabel("LLM label"); ax.set_ylabel("Rule label")
    ax.set_title("Confusion Matrix (count)")
    for i in range(len(L)):
        for j in range(len(L)):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, str(v), va="center", ha="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def plot_per_label_acc(per_total, per_agree):
    labs = [l for l in LABEL_ORDER if per_total.get(l, 0) > 0]
    acc = [per_agree.get(l, 0) / per_total.get(l, 1) for l in labs]
    counts = [per_total.get(l, 0) for l in labs]
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(labs))
    ax.bar(x, acc)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Agreement w/ LLM")
    ax.set_title("Per-Label Agreement (Rule as reference)")
    for i, (a, n) in enumerate(zip(acc, counts)):
        ax.text(i, a + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)
    return fig

# ---------------- Top 分歧样例 ----------------
def pick_top_disagreements(joined_rows, topn=20):
    diffs = [r for r in joined_rows if r["rule_label"] != r["llm_label"]]
    diffs.sort(key=lambda r: (r.get("final_score", 0.0), ), reverse=True)
    return diffs[:topn]

# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser()
    # 两种输入模式：1) 读规则样例文件；2) 现场计算规则标签
    ap.add_argument("--compute-rule", action="store_true",
                    help="启用现场计算规则标签（免额外文件）")
    ap.add_argument("--fused", default=str(DEFAULT_FUSED),
                    help="当 --compute-rule 时：recall_pairs_fused.jsonl 路径")
    ap.add_argument("--tf", default=str(DEFAULT_TF),
                    help="当 --compute-rule 时：tests_tf.mapped.jsonl 路径")
    ap.add_argument("--pt", default=str(DEFAULT_PT),
                    help="当 --compute-rule 时：tests_pt.mapped.jsonl 路径")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="当 --compute-rule 时：final_score 阈值（默认 0.5）")

    # LLM 标注
    ap.add_argument("--llm-jsonl", default=str(DEFAULT_LLM_JSONL))
    # 若不 compute-rule，沿用原参数：规则样例（注意：这是“样例集”，不一定全量）
    ap.add_argument("--rule-samples", default=str(DEFAULT_RULE_SAMPLES))

    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--topn", type=int, default=20, help="报告中展示 Top-N 分歧样例")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    confusion_csv = out_dir / OUT_CONFUSION_CSV.name
    per_label_csv = out_dir / OUT_PER_LABEL_CSV.name
    out_html = out_dir / OUT_HTML.name
    out_md = out_dir / OUT_MD.name

    # 读 LLM 标注
    llm_rows  = read_jsonl(Path(args.llm_jsonl))
    llm_idx = {pair_key(r): r for r in llm_rows}

    # 规则侧：两种来源
    if args.compute_rule:
        print(f"[INFO] 启用现场计算规则标签：阈值 final_score >= {args.threshold}")
        rule_map = compute_rule_labels_on_the_fly(
            fused_path=Path(args.fused),
            tf_path=Path(args.tf),
            pt_path=Path(args.pt),
            final_score_threshold=args.threshold
        )
        # 将 rule_map 作为全集，与 LLM 取交集
        rule_keys = set(rule_map.keys())
        llm_keys = set(llm_idx.keys())
        keys = sorted(rule_keys & llm_keys)
        print(f"[COVERAGE] 高分对（规则侧）数量：{len(rule_keys)}；LLM 标注数量：{len(llm_keys)}；交集：{len(keys)}")
        missing_llm = len(rule_keys - llm_keys)
        if missing_llm > 0:
            print(f"[WARN] 有 {missing_llm} 个高分对没有 LLM 标注（不参与对比）。")
        # 构建 “规则行字典” 供后续 join 使用
        rule_idx = {}
        for k in keys:
            (tf_file, tf_name, pt_file, pt_name) = k
            rule_idx[k] = {
                "tf_file": tf_file, "tf_name": tf_name,
                "pt_file": pt_file, "pt_name": pt_name,
                "diff_type": rule_map[k],
            }
    else:
        print("[INFO] 使用现有规则样例文件（可能不是全量）：", args.rule_samples)
        rule_rows = read_jsonl(Path(args.rule_samples))
        rule_idx = {pair_key(r): r for r in rule_rows}
        keys = sorted(set(rule_idx.keys()) & set(llm_idx.keys()))
        print(f"[COVERAGE] 规则样例数：{len(rule_idx)}；LLM 标注数：{len(llm_idx)}；交集：{len(keys)}")

    if not keys:
        raise RuntimeError("没有重叠样本：请检查输入，或确保 LLM 标注覆盖了高分对。")

    # 构建比较集
    cm = defaultdict(int)
    per_total = Counter()
    per_agree = Counter()
    joined_rows = []

    for k in keys:
        rr = rule_idx[k]; llm = llm_idx[k]
        rlab = rr.get("diff_type", "OTHER")
        llab = primary_label_llm(llm)

        cm[(rlab, llab)] += 1
        per_total[rlab] += 1
        if rlab == llab:
            per_agree[rlab] += 1

        joined_rows.append({
            "tf_file": rr.get("tf_file"),
            "tf_name": rr.get("tf_name"),
            "pt_file": rr.get("pt_file"),
            "pt_name": rr.get("pt_name"),
            "rule_label": rlab,
            "llm_label": llab,
            "final_score": llm.get("final_score"),
            "api_jaccard": llm.get("api_jaccard"),
            "assert_jaccard": llm.get("assert_jaccard"),
            "tf_family": llm.get("tf_family"),
            "pt_family": llm.get("pt_family"),
            "many_to_one_tf": llm.get("many_to_one_tf"),
            "many_to_one_pt": llm.get("many_to_one_pt"),
            "rationale": (llm.get("rationale") or "")[:120],
        })

    # 汇总 CSV：混淆长表
    with open(confusion_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rule_label","llm_label","count"])
        for (a,b), cnt in sorted(cm.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            w.writerow([a,b,cnt])

    # 汇总 CSV：每标签一致率
    with open(per_label_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rule_label","agree","total","accuracy"])
        for lab in LABEL_ORDER:
            tot = per_total.get(lab, 0)
            agr = per_agree.get(lab, 0)
            acc = f"{(agr/max(tot,1)):.2%}" if tot else "NA"
            w.writerow([lab, agr, tot, acc])

    # 作图
    def fig_to_base64(fig, bbox_inches="tight", dpi=160):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches=bbox_inches, dpi=dpi)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def plot_confusion(cm_counts, labels):
        L = labels
        mat = np.zeros((len(L), len(L)), dtype=int)
        for (r, l), cnt in cm_counts.items():
            if r in L and l in L:
                mat[L.index(r), L.index(l)] = cnt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(L))); ax.set_xticklabels(L, rotation=45, ha="right")
        ax.set_yticks(range(len(L))); ax.set_yticklabels(L)
        ax.set_xlabel("LLM label"); ax.set_ylabel("Rule label")
        ax.set_title("Confusion Matrix (count)")
        for i in range(len(L)):
            for j in range(len(L)):
                v = mat[i, j]
                if v > 0:
                    ax.text(j, i, str(v), va="center", ha="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    def plot_per_label_acc(per_total, per_agree):
        labs = [l for l in LABEL_ORDER if per_total.get(l, 0) > 0]
        acc = [per_agree.get(l, 0) / per_total.get(l, 1) for l in labs]
        counts = [per_total.get(l, 0) for l in labs]
        fig = plt.figure(figsize=(8, 4.8))
        ax = fig.add_subplot(111)
        x = np.arange(len(labs))
        ax.bar(x, acc)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Agreement w/ LLM")
        ax.set_title("Per-Label Agreement (Rule as reference)")
        for i, (a, n) in enumerate(zip(acc, counts)):
            ax.text(i, a + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)
        return fig

    fig1 = plot_confusion(cm, LABEL_ORDER); img_conf = fig_to_base64(fig1)
    fig2 = plot_per_label_acc(per_total, per_agree); img_acc = fig_to_base64(fig2)

    # Top N 分歧样例
    topn = int(args.topn)
    diffs = [r for r in joined_rows if r["rule_label"] != r["llm_label"]]
    diffs.sort(key=lambda r: (r.get("final_score", 0.0), ), reverse=True)
    top_diff = diffs[:topn]

    # HTML 报告
    total = len(joined_rows)
    overall_acc = sum(1 for r in joined_rows if r["rule_label"] == r["llm_label"]) / total if total else 0.0

    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<title>规则 vs 大模型 差异对比报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif; margin: 24px; }}
h1,h2 {{ margin: 0.4em 0; }}
.small {{ color: #666; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
th {{ background: #f7f7f7; text-align: left; }}
.code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
.badge {{ display:inline-block; padding:2px 6px; border-radius:4px; background:#eef; margin-right:6px; }}
.badge.red {{ background:#ffecec; }}
.badge.green {{ background:#eaffea; }}
</style>
</head>
<body>
<h1>规则 vs 大模型 差异对比报告</h1>
<p class="small">
样本数：{total}；主标签一致率：{overall_acc:.2%}<br/>
来源：{"现场计算规则标签" if args.compute_rule else "规则样例文件"}；LLM：{Path(args.llm_jsonl).name}
</p>

<h2>混淆矩阵（规则标签 × 大模型标签）</h2>
<img src="{img_conf}" alt="confusion" />

<h2>按规则标签的一致率</h2>
<img src="{img_acc}" alt="per label acc" />

<h2>Top {topn} 分歧样例（按 final_score 降序）</h2>
<table>
<tr>
  <th>#</th><th>TF</th><th>PT</th>
  <th>Rule → LLM</th><th>final</th><th>apiJ</th><th>asrJ</th>
  <th>fam(TF→PT)</th><th>M2O</th><th>LLM理由</th>
</tr>
"""
    for i, r in enumerate(top_diff, 1):
        tf = f"{r['tf_file']} :: {r['tf_name']}"
        pt = f"{r['pt_file']} :: {r['pt_name']}"
        m2o = ""
        if r.get("many_to_one_tf"): m2o += "TF "
        if r.get("many_to_one_pt"): m2o += "PT"
        fam = f"{r.get('tf_family','?')}→{r.get('pt_family','?')}"
        html += f"""
<tr>
  <td>{i}</td>
  <td class="code">{tf}</td>
  <td class="code">{pt}</td>
  <td><span class="badge">{r['rule_label']}</span> → <span class="badge red">{r['llm_label']}</span></td>
  <td>{(r.get('final_score') or 0):.2f}</td>
  <td>{(r.get('api_jaccard') or 0):.2f}</td>
  <td>{(r.get('assert_jaccard') or 0):.2f}</td>
  <td>{fam}</td>
  <td>{m2o}</td>
  <td>{r.get('rationale','')}</td>
</tr>
"""
    html += """
</table>
<p class="small">注：一致率以“规则标签”为参照；热力图数值为计数。</p>
</body></html>
"""
    out_html.write_text(html, encoding="utf-8")

    # Markdown（简要）
    md = f"""# 规则 vs 大模型 差异对比报告

- 样本数：**{total}**
- 主标签一致率：**{overall_acc:.2%}**
- 来源：{"现场计算规则标签" if args.compute_rule else "规则样例文件"}
- LLM：`{Path(args.llm_jsonl).name}`

## 文件
- 混淆长表：`{confusion_csv.name}`
- 每标签一致率：`{per_label_csv.name}`
- HTML 报告：`{out_html.name}`

> 详细热力图与柱状图请打开 HTML 报告查看。
"""
    out_md.write_text(md, encoding="utf-8")

    print(f"[OK] 混淆长表  → {confusion_csv}")
    print(f"[OK] 每标签一致率 → {per_label_csv}")
    print(f"[OK] HTML 报告   → {out_html}")
    print(f"[OK] Markdown    → {out_md}")

if __name__ == "__main__":
    main()
