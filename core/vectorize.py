import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_tests(path):
    tests = []
    with open(path) as f:
        for line in f:
            tests.append(json.loads(line))
    return tests

def stringify_test(test):
    """把测试函数转成文本串（用于向量化）"""
    parts = []
    parts.append(test.get("name", ""))
    parts.extend(test.get("apis", []))
    parts.extend(test.get("asserts", []))
    return " ".join(parts)

def build_vectors(tests):
    texts = [stringify_test(t) for t in tests]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def recall_topk(tf_tests, pt_tests, k=10):
    X_tf, vec = build_vectors(tf_tests)
    X_pt = vec.transform([stringify_test(t) for t in pt_tests])

    sim_matrix = cosine_similarity(X_tf, X_pt)

    results = []
    for i, tf_t in enumerate(tf_tests):
        sims = sim_matrix[i]
        topk_idx = np.argsort(sims)[::-1][:k]
        for j in topk_idx:
            results.append({
                "tf_file": tf_t["file"],
                "tf_name": tf_t["name"],
                "pt_file": pt_tests[j]["file"],
                "pt_name": pt_tests[j]["name"],
                "score": float(sims[j])
            })
    return results

if __name__ == "__main__":
    tf_tests = load_tests("../data/tests_tf.parsed.jsonl")
    pt_tests = load_tests("../data/tests_pt.parsed.jsonl")

    pairs = recall_topk(tf_tests, pt_tests, k=5)

    with open("../data/recall_pairs_func.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
