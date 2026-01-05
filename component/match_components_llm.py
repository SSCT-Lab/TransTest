# ./component/match_components_llm.py
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from llm_utils import get_qwen_client

def load_jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(x) for x in open(path)]

PROMPT = """
你是深度学习框架 API 匹配专家。
判断下面两个 API 是否为等价或相似组件：

TF: {tf_api}
Signature: {tf_sig}

PT: {pt_api}
Signature: {pt_sig}

只输出一个 0~1 的数字。
"""

def key_from(obj):
    return obj["tf_api"] + "@@" + obj["pt_api"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand", default="data/component_candidates.jsonl")
    parser.add_argument("--out", default="data/component_pairs.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="qwen-flash")
    args = parser.parse_args()

    candidates = load_jsonl(args.cand)

    # load resume
    done = {}
    if args.resume and Path(args.out).exists():
        for x in load_jsonl(args.out):
            done[key_from(x)] = True
        print(f"[resume] 已完成 {len(done)} 条")

    client = get_qwen_client("aliyun.key")
    fout = open(args.out, "a", encoding="utf-8")

    for c in tqdm(candidates):
        k = key_from(c)
        if k in done:
            continue

        prompt = PROMPT.format(**c)

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role":"user","content":prompt}]
            )
            raw = resp.choices[0].message.content.strip()
            score = float(raw)
        except:
            score = 0

        c["llm_score"] = score
        c["final_score"] = 0.5*c["emb_sim"] + 0.5*score

        fout.write(json.dumps(c, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"[DONE] wrote to {args.out}")

if __name__ == "__main__":
    main()
