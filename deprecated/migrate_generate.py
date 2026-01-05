# component/migrate_generate.py
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re

def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []

def safe_slug(s):
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/component_pairs.jsonl")
    parser.add_argument("--tf-tests", default="data/tests_tf.mapped.jsonl")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--out-dir", default="migrated_tests")
    args = parser.parse_args()

    print("[LOAD] reading mapping and TF tests...")
    pairs = load_jsonl(args.pairs)
    tests = load_jsonl(args.tf_tests)

    # API → API 替换字典
    mapping = {}
    for p in pairs:
        if p.get("final_score", 0) >= args.min_score:
            mapping[p["tf_api"]] = p["pt_api"]

    # 输出目录
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    print("[MIGRATE] generating migrated tests...")
    count = 0

    for t in tqdm(tests):
        file = t["file"]
        apis = t.get("apis", [])

        # 检查该测试是否含有可迁移 API
        migrate_pairs = [(a, mapping[a]) for a in apis if a in mapping]
        if not migrate_pairs:
            continue

        # 读取源代码
        src_path = Path("framework/tensorflow-master") / file
        if not src_path.exists():
            continue

        code = src_path.read_text()

        # 替换 API
        for tf_api, pt_api in migrate_pairs:
            code = code.replace(tf_api, pt_api)

        # 写入迁移后的测试文件
        new_file = out / f"{safe_slug(file)}__{t['name']}.py"
        with open(new_file, "w") as f:
            f.write(code)

        count += 1

    print(f"[DONE] generated {count} migrated tests in {args.out_dir}")


if __name__ == "__main__":
    main()
