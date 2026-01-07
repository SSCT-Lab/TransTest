import json
import argparse
from pathlib import Path
from openai import OpenAI

def load_api_key(path="aliyun.key"):
    with open(path, "r") as f:
        return f.read().strip()

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

def build_prompt(tf_t, pt_t):
    return f"""
你是一个代码测试匹配助手。任务是判断下面两个 Python 测试函数是否在测试相同或等价的功能。

TensorFlow 测试函数:
文件: {tf_t.get("file")}
函数: {tf_t.get("name")}
API调用: {", ".join(tf_t.get("apis_mapped", []))}
断言: {", ".join(tf_t.get("asserts", []))}

PyTorch 测试函数:
文件: {pt_t.get("file")}
函数: {pt_t.get("name")}
API调用: {", ".join(pt_t.get("apis_mapped", []))}
断言: {", ".join(pt_t.get("asserts", []))}

请只输出一个 0 到 1 之间的小数（例如 0.85），表示它们的功能相似度。
不要输出其他解释。
"""

def get_qwen_client(key_path="aliyun.key"):
    api_key = load_api_key(key_path)
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def rerank_with_llm(client, pairs, tf_tests, pt_tests, limit=50, min_score=0.2):
    tf_index = {(t["file"], t["name"]): t for t in tf_tests}
    pt_index = {(t["file"], t["name"]): t for t in pt_tests}

    total = len(pairs)
    # 先筛选
    filtered = [
        p for p in pairs
        if p.get("score", 0) >= min_score or p.get("struct_score", 0) >= min_score
    ]
    print(f"总共有 {total} 条，筛选后 {len(filtered)} 条 (min_score={min_score})")

    # 如果 limit=-1，表示全量跑
    if limit > 0:
        filtered = filtered[:limit]
        print(f"最终将处理 {len(filtered)} 条 (limit={limit})")
    else:
        print(f"最终将处理 {len(filtered)} 条 (全量)")

    enriched = []
    for i, p in enumerate(filtered, 1):
        tf_t = tf_index.get((p["tf_file"], p["tf_name"]))
        pt_t = pt_index.get((p["pt_file"], p["pt_name"]))
        if not tf_t or not pt_t:
            continue

        prompt = build_prompt(tf_t, pt_t)
        resp = client.chat.completions.create(
            model="qwen-flash",  # 可替换成 qwen2.5-7b-instruct / qwen2.5-14b-instruct 等
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()

        try:
            score = float(raw)
        except:
            score = None

        p["llm_score"] = score
        enriched.append(p)

        print(f"[{i}/{len(filtered)}] TF:{p['tf_name']} ↔ PT:{p['pt_name']} -> {score}")

    return enriched

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20,
                        help="最多处理多少条，-1 表示全量")
    parser.add_argument("--min-score", type=float, default=0.2,
                        help="筛选阈值 (TF-IDF/struct_score)")
    parser.add_argument("--model", type=str, default="qwen-flash",
                        help="大模型名称，例如 qwen-flash / qwen2.5-7b-instruct")
    args = parser.parse_args()

    tf_tests = load_jsonl("../data/tests_tf.mapped.jsonl")
    pt_tests = load_jsonl("../data/tests_pt.mapped.jsonl")
    pairs = load_jsonl("../data/recall_pairs_struct.jsonl")

    client = get_qwen_client("../aliyun.key")
    # enriched = rerank_with_llm(
    #     client, pairs, tf_tests, pt_tests,
    #     limit=args.limit, min_score=args.min_score
    # )
    enriched = rerank_with_llm(
        client, pairs, tf_tests, pt_tests,
        limit=-1, min_score=0.3
    )
    Path("../data").mkdir(exist_ok=True)
    with open("../data/recall_pairs_sem_llm.jsonl", "w") as f:
        for p in enriched:
            f.write(json.dumps(p) + "\n")

# if __name__ == "__main__":
#     tf_tests = load_jsonl("../data/tests_tf.mapped.jsonl")
#     pt_tests = load_jsonl("../data/tests_pt.mapped.jsonl")
#     pairs = load_jsonl("../data/recall_pairs_struct.jsonl")
#
#     client = get_qwen_client("../aliyun.key")
#     enriched = rerank_with_llm(client, pairs, tf_tests, pt_tests, limit=20)
#
#     with open("../data/recall_pairs_sem_llm.jsonl", "w") as f:
#         for p in enriched:
#             f.write(json.dumps(p) + "\n")
