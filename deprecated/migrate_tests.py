# ./component/migrate_tests.py
import argparse
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def index_tests_by_api(tests):
    """
    将测试按 API 反向索引：
    api -> [ {file, name, ...}, ... ]
    """
    index = defaultdict(list)
    for t in tests:
        apis = t.get("apis_mapped") or t.get("apis") or []
        for api in apis:
            index[api].append({
                "file": t.get("file"),
                "name": t.get("name"),
                "class": t.get("class"),
                # 如有额外字段也可以加进来
            })
    return index


def load_existing_pairs(out_path):
    """
    用于断点续跑，读取已有迁移结果中的 (tf_api, pt_api) 集合。
    """
    done = set()
    if not Path(out_path).exists():
        return done
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tf_api = rec.get("tf_api")
            pt_api = rec.get("pt_api")
            if tf_api and pt_api:
                done.add((tf_api, pt_api))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default="data/component_pairs.jsonl",
                        help="组件级 API 映射结果")
    parser.add_argument("--tf-tests", type=str, default="data/tests_tf.mapped.jsonl",
                        help="TensorFlow 测试元数据")
    parser.add_argument("--pt-tests", type=str, default="data/tests_pt.mapped.jsonl",
                        help="PyTorch 测试元数据")
    parser.add_argument("--out", type=str, default="data/migration_plan.jsonl",
                        help="测试迁移计划输出文件")
    parser.add_argument("--min-score", type=float, default=0.1,
                        help="只保留相似度 >= min-score 的映射")
    parser.add_argument("--resume", action="store_true",
                        help="启用断点续跑：保留已有输出，只补充未完成部分")
    parser.add_argument("--limit", type=int, default=-1,
                        help="最多处理多少条映射，-1 表示全量")
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    tf_tests_path = Path(args.tf_tests)
    pt_tests_path = Path(args.pt_tests)
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"[LOAD] component pairs from {pairs_path}")
    pairs = load_jsonl(pairs_path)

    print(f"[LOAD] TF tests from {tf_tests_path}")
    tf_tests = load_jsonl(tf_tests_path)
    print(f"[LOAD] PT tests from {pt_tests_path}")
    pt_tests = load_jsonl(pt_tests_path)

    print("[INDEX] building API -> tests index ...")
    tf_index = index_tests_by_api(tf_tests)
    pt_index = index_tests_by_api(pt_tests)
    print(f"[INDEX] TF apis={len(tf_index)}, PT apis={len(pt_index)}")

    # 断点续跑：读取已完成 (tf_api, pt_api)
    done_pairs = set()
    write_mode = "w"
    if args.resume and out_path.exists():
        done_pairs = load_existing_pairs(out_path)
        write_mode = "a"
        print(f"[RESUME] 已存在 {len(done_pairs)} 条迁移记录，将跳过这些 pair")
    else:
        print("[RESUME] 不使用断点续跑，将从头重写输出文件")

    # 打开输出文件
    fout = open(out_path, write_mode, encoding="utf-8")

    processed = 0
    kept = 0

    # 可选裁剪数量
    to_iter = pairs
    if args.limit > 0:
        to_iter = pairs[:args.limit]
        print(f"[INFO] 仅处理前 {args.limit} 条 pairs")

    for p in tqdm(to_iter, desc="migrate-plans"):
        tf_api = p.get("tf_api") or p.get("tf") or p.get("src_api")
        pt_api = p.get("pt_api") or p.get("pt") or p.get("tgt_api")

        if not tf_api or not pt_api:
            continue

        key = (tf_api, pt_api)
        if key in done_pairs:
            # 已处理过，跳过
            continue

        score = p.get("score") or p.get("sim") or p.get("similarity") or 0.0
        try:
            score = float(score)
        except Exception:
            score = 0.0

        # if score < args.min_score:
        #     continue

        # 找到所有使用该 API 的测试
        tf_list = tf_index.get(tf_api, [])
        pt_list = pt_index.get(pt_api, [])

        rec = {
            "tf_api": tf_api,
            "pt_api": pt_api,
            "score": score,
            "num_tf_tests": len(tf_list),
            "num_pt_tests": len(pt_list),
            "tf_tests": tf_list,
            "pt_tests": pt_list,
        }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
        processed += 1

    fout.close()

    print("\n==== MIGRATION PLAN SUMMARY ====")
    print(f"总 pairs: {len(pairs)}")
    if args.limit > 0:
        print(f"实际遍历: {len(to_iter)}")
    print(f"min_score >= {args.min_score} 的迁移映射数: {kept}")
    print(f"输出文件: {out_path}")


if __name__ == "__main__":
    main()
