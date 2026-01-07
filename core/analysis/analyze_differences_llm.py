import json
import csv
import time
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
from openai import OpenAI

# ----------------- 常量与工具 -----------------
LABELS = [
    "API_OVERLAP_LOW",
    "ASSERT_OVERLAP_LOW",
    "CROSS_FAMILY_CONFLICT",
    "MANY_TO_ONE_TF",
    "MANY_TO_ONE_PT",
    "TRIVIAL_TEST",
    "IDENTICAL_SEMANTICS",
    "OTHER"
]


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in open(path, "r")]


def save_jsonl(path: Path, rows: List[dict], append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_api_key(path: Path) -> str:
    return path.read_text().strip()


def jaccard(a, b) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def family_from_api_id(api_id: str) -> str:
    s = (api_id or "").upper()
    if "CONV" in s: return "CONV"
    if any(k in s for k in ["DENSE", "LINEAR", "FC"]): return "DENSE"
    if any(k in s for k in ["POOL", "POOLING"]): return "POOL"
    if any(k in s for k in ["BATCHNORM", "NORM", "NORMALIZATION"]): return "NORM"
    if any(k in s for k in ["OPT_", "OPTIM"]): return "OPT"
    if any(k in s for k in ["LOSS", "CROSSENTROPY", "MSE"]): return "LOSS"
    if any(k in s for k in ["RELU", "SIGMOID", "TANH", "ACT"]): return "ACT"
    return "OTHER"


def dominant_family(apis_mapped: List[str]) -> str:
    if not apis_mapped:
        return "NONE"
    counts = Counter(family_from_api_id(a) for a in apis_mapped)
    return counts.most_common(1)[0][0]


def pair_id(p: dict) -> str:
    raw = f"{p.get('tf_file')}|{p.get('tf_name')}||{p.get('pt_file')}|{p.get('pt_name')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ----------------- Prompt 构造（批量） -----------------
def build_batch_prompt(batch: List[Dict[str, Any]]) -> str:
    """
    将若干 pair 信息（含已计算特征）打包给模型，要求只输出 JSON 数组。
    """
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["idx", "labels", "rationale"],
            "properties": {
                "idx": {"type": "integer"},
                "labels": {"type": "array", "items": {"type": "string", "enum": LABELS}},
                "rationale": {"type": "string"}
            }
        }
    }

    header = (
        "你是代码测试差异分析助手。给你若干对 TensorFlow 与 PyTorch 的测试匹配对，"
        "请从下列标签集合中多选最合适的差异类型，并给出极简理由（<=40字）。\n"
        f"可选标签：{', '.join(LABELS)}\n"
        "输出必须是一个 JSON 数组，数组元素与输入顺序一一对应，字段为："
        "{\"idx\": 序号, \"labels\": [标签...], \"rationale\": \"理由\"}。\n"
        "不要输出除 JSON 外的任何内容。"
    )

    # 将每个条目的关键信息压缩到一小段，控制 token
    lines = []
    for i, x in enumerate(batch):
        line = {
            "idx": i,
            "tf": {
                "file": x["tf_file"],
                "name": x["tf_name"],
                "apis": x.get("tf_apis", [])[:12],
                "asserts": x.get("tf_asserts", [])[:12],
                "dom_family": x.get("tf_family", "NONE"),
            },
            "pt": {
                "file": x["pt_file"],
                "name": x["pt_name"],
                "apis": x.get("pt_apis", [])[:12],
                "asserts": x.get("pt_asserts", [])[:12],
                "dom_family": x.get("pt_family", "NONE"),
            },
            "features": {
                "final_score": round(x.get("final_score", 0.0), 4),
                "api_jaccard": round(x.get("api_jaccard", 0.0), 4),
                "assert_jaccard": round(x.get("assert_jaccard", 0.0), 4),
                "many_to_one_tf": x.get("many_to_one_tf", False),
                "many_to_one_pt": x.get("many_to_one_pt", False),
            }
        }
        lines.append(line)

    prompt = header + "\n\n输入数据（JSON）：\n" + json.dumps(lines, ensure_ascii=False)
    return prompt


# ----------------- 调用 LLM -----------------
def get_client(key_path: Path) -> OpenAI:
    key = load_api_key(key_path)
    return OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def call_llm(client: OpenAI, model: str, prompt: str, retries=3, sleep_s=2) -> List[dict]:
    for r in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            # 只接受严格 JSON
            return json.loads(content)
        except Exception as e:
            if r == retries - 1:
                raise
            time.sleep(sleep_s * (r + 1))
    return []


# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="../data/recall_pairs_fused.jsonl")
    ap.add_argument("--tf", default="../data/tests_tf.mapped.jsonl")
    ap.add_argument("--pt", default="../data/tests_pt.mapped.jsonl")
    ap.add_argument("--key", default="../aliyun.key")
    ap.add_argument("--out-jsonl", default="../data/diff_types_llm.jsonl")
    ap.add_argument("--out-summary", default="../data/diff_types_llm_summary.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--model", default="qwen-flash")  # 可换 qwen2.5-7b-instruct 等
    ap.add_argument("--resume", action="store_true", help="断点续跑（跳过已存在 pair_id）")
    args = ap.parse_args()

    fused = load_jsonl(Path(args.fused))
    tf_tests = load_jsonl(Path(args.tf))
    pt_tests = load_jsonl(Path(args.pt))

    # 建索引
    tf_idx = {(t.get("file"), t.get("name")): t for t in tf_tests}
    pt_idx = {(t.get("file"), t.get("name")): t for t in pt_tests}

    # 过滤高分
    pairs = [p for p in fused if p.get("final_score", 0.0) >= args.threshold]
    print(f"[INFO] 高分对数量: {len(pairs)} (final_score >= {args.threshold})")

    # 统计一对多，用于提示模型
    tf_counts = Counter((p["tf_file"], p["tf_name"]) for p in pairs)
    pt_counts = Counter((p["pt_file"], p["pt_name"]) for p in pairs)

    # 预计算结构特征（落到 pair 上，便于 prompt 精简）
    enriched = []
    for p in pairs:
        tf_t = tf_idx.get((p["tf_file"], p["tf_name"]), {})
        pt_t = pt_idx.get((p["pt_file"], p["pt_name"]), {})
        apis_tf = tf_t.get("apis_mapped", []) or []
        apis_pt = pt_t.get("apis_mapped", []) or []
        asserts_tf = tf_t.get("asserts", []) or []
        asserts_pt = pt_t.get("asserts", []) or []
        enriched.append({
            **p,
            "tf_apis": apis_tf,
            "pt_apis": apis_pt,
            "tf_asserts": asserts_tf,
            "pt_asserts": asserts_pt,
            "api_jaccard": jaccard(apis_tf, apis_pt),
            "assert_jaccard": jaccard(asserts_tf, asserts_pt),
            "tf_family": dominant_family(apis_tf),
            "pt_family": dominant_family(apis_pt),
            "many_to_one_tf": tf_counts[(p["tf_file"], p["tf_name"])] > 1,
            "many_to_one_pt": pt_counts[(p["pt_file"], p["pt_name"])] > 1,
            "_pair_id": pair_id(p),
        })

    out_jsonl = Path(args.out_jsonl)
    # 断点续跑：读已有结果，跳过已处理
    done_ids = set()
    if args.resume and out_jsonl.exists():
        for line in open(out_jsonl, "r"):
            try:
                obj = json.loads(line)
                if "_pair_id" in obj:
                    done_ids.add(obj["_pair_id"])
            except:
                continue
        print(f"[INFO] 断点续跑：已存在 {len(done_ids)} 条，跳过这些 pair")

    client = get_client(Path(args.key))

    to_process = [x for x in enriched if x["_pair_id"] not in done_ids]
    print(f"[INFO] 本次需要处理: {len(to_process)} 条")

    results = []
    for i in range(0, len(to_process), args.batch_size):
        batch = to_process[i:i + args.batch_size]
        prompt = build_batch_prompt(batch)
        try:
            parsed = call_llm(client, args.model, prompt)
        except Exception as e:
            print(f"[ERROR] LLM 调用失败，跳过该批次 i={i}: {e}")
            continue

        if not isinstance(parsed, list) or len(parsed) != len(batch):
            print(f"[WARN] 返回数量与输入不一致 i={i}，尝试对齐/跳过")
            continue

        out_rows = []
        for local_idx, ann in enumerate(parsed):
            labels = ann.get("labels", [])
            rationale = ann.get("rationale", "")
            # 规整：过滤非法标签，至少给 OTHER
            labels = [L for L in labels if L in LABELS]
            if not labels:
                labels = ["OTHER"]

            row = {
                "_pair_id": batch[local_idx]["_pair_id"],
                "tf_file": batch[local_idx]["tf_file"],
                "tf_name": batch[local_idx]["tf_name"],
                "pt_file": batch[local_idx]["pt_file"],
                "pt_name": batch[local_idx]["pt_name"],
                "final_score": batch[local_idx].get("final_score"),
                "api_jaccard": batch[local_idx].get("api_jaccard"),
                "assert_jaccard": batch[local_idx].get("assert_jaccard"),
                "tf_family": batch[local_idx].get("tf_family"),
                "pt_family": batch[local_idx].get("pt_family"),
                "many_to_one_tf": batch[local_idx].get("many_to_one_tf"),
                "many_to_one_pt": batch[local_idx].get("many_to_one_pt"),
                "labels": labels,
                "rationale": rationale.strip(),
            }
            out_rows.append(row)

        # 逐批落盘（防中断丢失）
        save_jsonl(out_jsonl, out_rows, append=True)
        results.extend(out_rows)
        print(f"[INFO] 写入 {len(out_rows)} 条，进度 {i + len(batch)}/{len(to_process)}")

        # 轻微节流
        time.sleep(0.3)

    # 汇总统计（新产出 + 旧结果一体统计）
    all_rows = load_jsonl(out_jsonl)
    label_counter = Counter()
    for r in all_rows:
        for L in r.get("labels", []):
            label_counter[L] += 1

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    total = len(all_rows)
    with open(out_summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "count", "ratio"])
        for L, cnt in label_counter.most_common():
            ratio = f"{(cnt / total):.2%}" if total else "0.00%"
            w.writerow([L, cnt, ratio])

    print(f"[DONE] 共 {total} 条标注写入：{out_jsonl}")
    print(f"[DONE] 标签分布写入：{out_summary}")
    for L, cnt in label_counter.most_common():
        print(f"  - {L:22s}: {cnt:5d} ({(cnt / total):.2%})")

if __name__ == "__main__":
    main()