# component/migrate_force.py
import ast
import astor
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def load_jsonl(p):
    return [json.loads(x) for x in open(p)] if Path(p).exists() else []


def is_test_file(p):
    n = p.name.lower()
    return (
        n.startswith("test_")
        or n.endswith("_test.py")
        or "test" in n
    )


def extract_test_functions(path):
    """提取 TF 测试文件中的所有 test_xxx 函数"""
    try:
        src = Path(path).read_text(encoding="utf-8")
    except:
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    tests = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            tests.append(node)
    return tests


def convert_tf_to_pt(src: str):
    """非常简单的占位转换：tf.constant → torch.tensor"""
    return (
        src.replace("tf.constant", "torch.tensor")
           .replace("tf.math", "torch")
           .replace("self.assertTrue", "assert")
           .replace("self.assertFalse", "assert not")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf-root", default="framework/tensorflow-master")
    parser.add_argument("--out", default="output/pt_migrated_tests")
    parser.add_argument("--report", default="data/migrate_force_plan.jsonl")
    args = parser.parse_args()

    tf_root = Path(args.tf_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report)
    fout = open(report_path, "w")

    tf_files = [p for p in tf_root.rglob("*.py") if is_test_file(p)]

    print(f"[SCAN] Found TF test files: {len(tf_files)}")

    migrated_count = 0

    for file in tqdm(tf_files):
        tests = extract_test_functions(file)
        if not tests:
            continue

        migrated_count += len(tests)

        new_path = out_dir / (file.stem + "_migrated.py")

        lines = ["import torch\nimport pytest\n\n"]

        for t in tests:
            src = astor.to_source(t)
            src_pt = convert_tf_to_pt(src)
            lines.append(src_pt + "\n")

            # 写入迁移报告项
            fout.write(json.dumps({
                "tf_file": str(file),
                "pt_file": str(new_path),
                "test_name": t.name,
            }) + "\n")

        new_path.write_text("".join(lines), encoding="utf-8")

    fout.close()

    print("==== FORCE MIGRATION DONE ====")
    print(f"共迁移 TF 测试函数: {migrated_count}")
    print(f"输出目录: {out_dir}")
    print(f"迁移记录报告: {report_path}")
