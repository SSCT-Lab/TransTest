import json
from pathlib import Path

def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return inter / union if union > 0 else 0.0

def enrich_with_structural(pairs, tf_tests, pt_tests, w_api=0.7, w_assert=0.3):
    # 建立索引，方便通过 (file, name) 快速取测试函数信息
    tf_index = {(t["file"], t["name"]): t for t in tf_tests}
    pt_index = {(t["file"], t["name"]): t for t in pt_tests}

    enriched = []
    for p in pairs:
        tf_t = tf_index.get((p["tf_file"], p["tf_name"]), {})
        pt_t = pt_index.get((p["pt_file"], p["pt_name"]), {})

        api_score = jaccard(tf_t.get("apis", []), pt_t.get("apis", []))
        assert_score = jaccard(tf_t.get("asserts", []), pt_t.get("asserts", []))

        struct_score = w_api * api_score + w_assert * assert_score

        p.update({
            "api_jaccard": api_score,
            "assert_jaccard": assert_score,
            "struct_score": struct_score
        })
        enriched.append(p)
    return enriched

if __name__ == "__main__":
    tf_tests = [json.loads(line) for line in open("../data/tests_tf.parsed.jsonl")]
    pt_tests = [json.loads(line) for line in open("../data/tests_pt.parsed.jsonl")]
    pairs = [json.loads(line) for line in open("../data/recall_pairs_func.jsonl")]

    enriched = enrich_with_structural(pairs, tf_tests, pt_tests)

    Path("data").mkdir(exist_ok=True)
    with open("../data/recall_pairs_struct.jsonl", "w") as f:
        for p in enriched:
            f.write(json.dumps(p) + "\n")
