# component/full_migrate_and_run.py
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import subprocess
import re

def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def safe_slug(s):
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)

def migrate_tests(pairs, tests, min_score, out_dir):
    mapping = {p["tf_api"]: p["pt_api"] for p in pairs if p["final_score"] >= min_score}
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    count = 0

    for t in tqdm(tests, desc="Migrating tests"):
        file = t["file"]
        apis = t.get("apis", [])
        migrate_pairs = [(a, mapping[a]) for a in apis if a in mapping]
        if not migrate_pairs:
            continue

        src_path = Path("framework/tensorflow-master") / file
        if not src_path.exists():
            continue

        code = src_path.read_text()

        for tf_api, pt_api in migrate_pairs:
            code = code.replace(tf_api, pt_api)

        new_file = out_dir / f"{safe_slug(file)}__{t['name']}.py"
        with open(new_file, "w") as f:
            f.write(code)
        count += 1

    print(f"[DONE] Migrated {count} tests to {out_dir}")

def run_tests(test_dir, result_file):
    result_file = Path(result_file)
    result_file.parent.mkdir(exist_ok=True)

    tests = list(Path(test_dir).glob("*.py"))
    print(f"[RUN] Found {len(tests)} migrated tests")

    results = []
    for t in tqdm(tests, desc="Executing tests"):
        cmd = ["pytest", str(t), "-q", "--disable-warnings"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = proc.communicate()
        rc = proc.returncode

        result = {
            "file": str(t),
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "status": "pass" if rc == 0 else "fail"
        }
        results.append(result)

    with open(result_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[DONE] Execution results saved to {result_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/component_pairs.jsonl")
    parser.add_argument("--tf-tests", default="data/tests_tf.mapped.jsonl")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--out-dir", default="migrated_tests")
    parser.add_argument("--result-file", default="data/migrate_exec.jsonl")
    args = parser.parse_args()

    # 1. Load pairs and tests
    pairs = load
