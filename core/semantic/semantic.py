import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

def stringify_test(t):
    """把测试函数转成语义文本"""
    parts = []
    parts.append(t.get("name", ""))
    parts.extend(t.get("apis_mapped", []))
    parts.extend(t.get("asserts", []))
    if "doc" in t:  # 如果你在 AST 里加过 docstring，可以在这里加
        parts.append(t["doc"])
    return " ".join(parts)

def build_semantic_vectors(tf_tests, pt_tests):
    texts = [stringify_test(t) for t in tf_tests + pt_tests]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    n_tf = len(tf_tests)
    X_tf = X[:n_tf]
    X_pt = X[n_tf:]
    return X_tf, X_pt, vectorizer

def enrich_with_semantics(pairs, tf_tests, pt_tests):
    # 建立索引
    tf_index = {(t["file"], t["name"]): i for i, t in enumerate(tf_tests)}
    pt_index = {(t["file"], t["name"]): i for i, t in enumerate(pt_tests)}

    # 向量化
    X_tf, X_pt, _ = build_semantic_vectors(tf_tests, pt_tests)

    enriched = []
    for p in pairs:
        i = tf_index.get((p["tf_file"], p["tf_name"]))
        j = pt_index.get((p["pt_file"], p["pt_name"]))
        if i is None or j is None:
            continue
        score = cosine_similarity(X_tf[i], X_pt[j])[0, 0]
        p["semantic_score"] = float(score)
        enriched.append(p)
    return enriched

if __name__ == "__main__":
    tf_tests = load_jsonl("../data/tests_tf.mapped.jsonl")
    pt_tests = load_jsonl("../data/tests_pt.mapped.jsonl")
    pairs = load_jsonl("../data/recall_pairs_struct.jsonl")

    enriched = enrich_with_semantics(pairs, tf_tests, pt_tests)

    with open("../data/recall_pairs_sem.jsonl", "w") as f:
        for p in enriched:
            f.write(json.dumps(p) + "\n")
