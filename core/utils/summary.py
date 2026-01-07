import json
import argparse
from pathlib import Path
from openai import OpenAI

def load_api_key(path="aliyun.key"):
    with open(path, "r") as f:
        return f.read().strip()

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

def save_jsonl(path, data):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def fusion_score(p, w_tf=0.2, w_struct=0.3, w_sem=0.0, w_llm=0.5):
    """融合分数计算，默认更信任 LLM"""
    return (
        w_tf * p.get("score", 0)
        + w_struct * p.get("struct_score", 0)
        + w_sem * p.get("semantic_score", 0)
        + w_llm * (p.get("llm_score", 0) or 0)
    )

def get_qwen_client(key_path="aliyun.key"):
    api_key = load_api_key(key_path)
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def build_summary_prompt(p):
    return f"""
你是一个深度学习框架测试迁移分析助手。现在给你一对 TensorFlow 和 PyTorch 的测试函数，请总结它们的共同点（在测试什么功能），并输出一句话描述。

TensorFlow 测试:
文件: {p['tf_file']}
函数: {p['tf_name']}
API: {", ".join(p.get("apis_mapped", []))}

PyTorch 测试:
文件: {p['pt_file']}
函数: {p['pt_name']}
API: {", ".join(p.get('apis_mapped', []))}

请用简短中文总结，例如：
“这两个测试都在验证 Conv2D 层的前向计算是否正确。”

"""

def summarize_matches(client, pairs, topk=20):
    summaries = []
    for i, p in enumerate(pairs[:topk], 1):
        prompt = build_summary_prompt(p)
        resp = client.chat.completions.create(
            model="qwen-flash",  # 可替换成 qwen2.5-7b-instruct
            messages=[{"role": "user", "content": prompt}],
        )
        summary = resp.choices[0].message.content.strip()
        p["summary"] = summary
        summaries.append(p)

        print(f"[{i}/{topk}] TF:{p['tf_name']} ↔ PT:{p['pt_name']} -> {summary}")

    return summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5, help="final_score 阈值")
    parser.add_argument("--topk", type=int, default=2000, help="最多总结多少条")
    parser.add_argument("--model", type=str, default="qwen-flash", help="大模型名称")
    args = parser.parse_args()

    # 读取 Step 7 结果（包含 llm_score）
    pairs = load_jsonl("../data/recall_pairs_sem_llm.jsonl")

    # Step 8: 融合排序
    for p in pairs:
        p["final_score"] = fusion_score(p)
    pairs_sorted = sorted(pairs, key=lambda x: x["final_score"], reverse=True)

    Path("../data").mkdir(exist_ok=True)
    save_jsonl("../data/recall_pairs_fused.jsonl", pairs_sorted)
    print(f"融合完成，共 {len(pairs_sorted)} 条，已保存到 ../data/recall_pairs_fused.jsonl")

    # Step 9: 筛选高分对并总结
    high_pairs = [p for p in pairs_sorted if p["final_score"] >= args.threshold]
    print(f"阈值 {args.threshold} 筛选后剩余 {len(high_pairs)} 条")

    if high_pairs:
        client = get_qwen_client("../aliyun.key")
        summarized = summarize_matches(client, high_pairs, topk=args.topk)
        save_jsonl("../data/recall_pairs_summary.jsonl", summarized)
        print(f"总结完成，已保存到 ../data/recall_pairs_summary.jsonl")
    else:
        print("没有超过阈值的匹配对，不执行总结。")
