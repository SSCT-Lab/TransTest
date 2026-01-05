# component/migrate_run.py
import subprocess
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="migrated_tests")
    parser.add_argument("--out", default="data/migrate_exec.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    fout = open(out_path, "w")

    tests = list(Path(args.dir).glob("*.py"))
    print(f"[RUN] found {len(tests)} migrated tests")

    for t in tqdm(tests):
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
        fout.write(json.dumps(result) + "\n")

    fout.close()
    print(f"[DONE] execution results saved to {out_path}")


def test_large_values():
    x = torch.tensor([1000.0, 1000.0, 1000.0])
    out = torch.softmax(x, dim=-1)
    expected = torch.tensor([1/3, 1/3, 1/3])
    assert torch.allclose(out, expected, atol=1e-6)



if __name__ == "__main__":
    main()
