# component/migrate_identify.py
import json
from pathlib import Path
from tqdm import tqdm

def load_jsonl(p):
    return [json.loads(line) for line in open(p)]

def main():
    pairs = load_jsonl("data/component_pairs.jsonl")
    tf_usage = load_jsonl("data/tf_test_api_usage.jsonl")

    # ------------------------------
    # 建立 TF_API → PT_API 映射表
    # ------------------------------
    api_map = {}
    for p in pairs:
        tf_api = p["tf_api"]
        pt_api = p["pt_api"]
        score = p["final_score"]
        if score < 0.3:
            continue
        api_map.setdefault(tf_api, []).append((pt_api, score))

    print(f"[INFO] 可用 TF→PT API 映射数量: {len(api_map)}")

    # ------------------------------
    # 扫描测试中是否出现可迁移 API
    # ------------------------------
    migrated = []
    for item in tqdm(tf_usage, desc="扫描 TF 测试用例"):
        tf_file = item["file"]
        tf_apis = item.get("apis", [])

        matched = {}
        for api in tf_apis:
            if api in api_map:
                matched[api] = api_map[api]

        if matched:
            migrated.append({
                "file": tf_file,
                "apis": tf_apis,
                "matches": matched
            })

    out = "data/migration_candidates.jsonl"
    with open(out, "w") as f:
        for m in migrated:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[DONE] 可迁移测试数量: {len(migrated)}")
    print(f"结果已输出: {out}")

if __name__ == "__main__":
    main()
