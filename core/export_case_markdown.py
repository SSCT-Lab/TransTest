# core/export_case_markdown.py
import argparse, json, re, ast
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def read_jsonl(p: Path):
    with open(p, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def find_function_by_ast(src: str, func_name: str) -> Optional[Tuple[int,int]]:
    """
    返回 (start_lineno, end_lineno) 1-based（包含 end）
    若找不到返回 None
    """
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    target_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target_nodes.append(node)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == func_name:
            target_nodes.append(node)
    if not target_nodes:
        return None
    # 选第一个匹配
    node = target_nodes[0]
    # Python 3.8+ ast nodes have end_lineno
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is not None and end is not None:
        return (start, end)
    # 退化：没有 end_lineno 时，粗略扩展到下一个顶层定义
    lines = src.splitlines()
    i = start - 1
    j = i + 1
    while j < len(lines) and not re.match(r"^\s*def\s+\w+\s*\(|^\s*class\s+\w+\s*:", lines[j]):
        j += 1
    return (start, j)

def find_function_by_regex(src: str, func_name: str) -> Optional[Tuple[int,int]]:
    """
    退路：用正则找 def func_name(...): 的起始行，再向后扫到下一个顶层 def/class
    """
    pat = re.compile(rf"^(\s*)def\s+{re.escape(func_name)}\s*\(", re.M)
    m = pat.search(src)
    if not m:
        return None
    start_idx = len(src[:m.start()].splitlines()) + 1
    lines = src.splitlines()
    # 找下一个顶级 def/class
    i = start_idx - 1
    j = i + 1
    while j < len(lines):
        if re.match(r"^\s*def\s+\w+\s*\(|^\s*class\s+\w+\s*:", lines[j]) and len(lines[j]) - len(lines[j].lstrip()) <= len(m.group(1)):
            break
        j += 1
    return (start_idx, j)

def slice_with_context(src: str, start: int, end: int, ctx: int) -> str:
    lines = src.splitlines()
    s = max(1, start - ctx)
    e = min(len(lines), end + ctx)
    # 附带行号注释方便口头讲解
    out = []
    for idx in range(s, e + 1):
        prefix = f"{idx:>5} │ "
        out.append(prefix + lines[idx-1])
    return "\n".join(out)

def pick_func_source(file_path: Path, func_name: str, context: int = 2) -> Tuple[str, str]:
    """
    返回 (snip, how) where how in {"ast","regex","not_found"}
    """
    if not file_path.exists():
        return (f"[文件不存在] {file_path}", "not_found")
    src = load_text(file_path)
    loc = find_function_by_ast(src, func_name)
    how = "ast"
    if not loc:
        loc = find_function_by_regex(src, func_name)
        how = "regex" if loc else "not_found"
    if not loc:
        return (f"[未能定位函数 `{func_name}`] 文件：{file_path}", "not_found")
    start, end = loc
    return (slice_with_context(src, start, end, context), how)

def build_markdown(case: dict, tf_repo: Path, pt_repo: Path, ctx: int) -> str:
    tf_file = tf_repo / case["tf_file"]
    pt_file = pt_repo / case["pt_file"]
    tf_name = case["tf_name"]
    pt_name = case["pt_name"]
    label  = case.get("label", case.get("rule_label", ""))
    meta = {
        "label": label,
        "final_score": case.get("final_score"),
        "api_jaccard": case.get("api_jaccard"),
        "assert_jaccard": case.get("assert_jaccard"),
        "tf_family": case.get("tf_family"),
        "pt_family": case.get("pt_family"),
    }

    tf_snip, tf_how = pick_func_source(tf_file, tf_name, ctx)
    pt_snip, pt_how = pick_func_source(pt_file, pt_name, ctx)

    md = []
    md.append(f"# 测试对案例（{label}）\n")
    md.append("## 元信息\n")
    md.append("| 项目 | 值 |")
    md.append("|---|---|")
    md.append(f"| TF | `{case['tf_file']} :: {tf_name}` |")
    md.append(f"| PT | `{case['pt_file']} :: {pt_name}` |")
    for k, v in meta.items():
        if v is not None:
            md.append(f"| {k} | `{v}` |")
    md.append("")
    md.append("## TensorFlow 片段\n")
    md.append("```python")
    md.append(tf_snip)
    md.append("```")
    md.append(f"> 定位方式：{tf_how}")
    md.append("")
    md.append("## PyTorch 片段\n")
    md.append("```python")
    md.append(pt_snip)
    md.append("```")
    md.append(f"> 定位方式：{pt_how}")
    md.append("")
    return "\n".join(md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DATA / "high_agreement_cases.jsonl"))
    # 你的仓库根（对应你 YAML 中的路径）
    ap.add_argument("--tf-repo", default="framework/tensorflow-master")
    ap.add_argument("--pt-repo", default="framework/pytorch-main")
    ap.add_argument("--label", default="", help="只导出某个标签的案例，如 IDENTICAL_SEMANTICS")
    ap.add_argument("--limit", type=int, default=3, help="导出案例数")
    ap.add_argument("--context", type=int, default=2, help="源码上下文行数")
    ap.add_argument("--out", default=str(DATA / "cases_md"))
    args = ap.parse_args()

    cases_path = Path(args.cases)
    tf_repo = Path(args.tf_repo)
    pt_repo = Path(args.pt_repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = []
    for r in read_jsonl(cases_path):
        if args.label and r.get("label") != args.label:
            continue
        picked.append(r)
    # 默认按 final_score 降序
    picked.sort(key=lambda x: (x.get("final_score") or 0.0,
                               x.get("api_jaccard") or 0.0,
                               x.get("assert_jaccard") or 0.0), reverse=True)
    picked = picked[: args.limit]

    if not picked:
        print("[WARN] 没有符合条件的案例。检查 --cases / --label / --limit")
        return

    index_md = ["# 案例索引\n"]
    for i, c in enumerate(picked, 1):
        md = build_markdown(c, tf_repo, pt_repo, args.context)
        out_file = out_dir / f"case_{i:02d}_{c.get('label','')}.md"
        out_file.write_text(md, encoding="utf-8")
        index_md.append(f"- [{out_file.name}]({out_file.name}) | {c['tf_name']} ↔ {c['pt_name']} | {c.get('label','')}")
        print(f"[OK] 写入 {out_file}")

    (out_dir / "README.md").write_text("\n".join(index_md), encoding="utf-8")
    print(f"[OK] 索引写入 {out_dir/'README.md'}")

if __name__ == "__main__":
    main()
