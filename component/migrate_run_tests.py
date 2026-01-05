# ./component/migrate_run_tests.py
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import tempfile
import os

MIGRATED_DIR = Path("migrated_tests")
OUT_FILE = Path("data/migrate_exec.jsonl")
LOG_DIR = Path("data/migrate_logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)


def run_pytest(file_path):
    """单独执行一个 PyTorch 测试文件，确保不会互相污染环境。"""
    try:
        # 建立独立进程执行 pytest
        result = subprocess.run(
            ["pytest", str(file_path), "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,  # 超时保护
            text=True
        )
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired as e:
        return {
            "status": "timeout",
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }
    except Exception as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "returncode": -2
        }


def main():
    files = sorted(MIGRATED_DIR.glob("*.py"))
    print(f"[RUN] found {len(files)} migrated tests")

    fout = open(OUT_FILE, "w")

    for f in tqdm(files, desc="Running migrated tests"):
        result = run_pytest(f)

        # 保存单个文件的详细日志
        log_path = LOG_DIR / f"{f.stem}.log"
        with open(log_path, "w") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result["stdout"])
            lf.write("\n\n=== STDERR ===\n")
            lf.write(result["stderr"])

        # 简要记录 JSON
        rec = {
            "file": f.name,
            "path": str(f),
            "status": result["status"],
            "returncode": result["returncode"],
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    fout.close()

    print(f"[DONE] execution results saved to {OUT_FILE}")
    print(f"[LOG] detailed logs stored in {LOG_DIR}")


if __name__ == "__main__":
    main()
