import yaml
import json
from pathlib import Path
from glob import glob
from core import discover_test_files, normalize_file, parse_test_file

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def expand_roots(repo_root, sub_roots):
    """把相对路径和 glob 展开为绝对路径"""
    expanded = []
    for sub in sub_roots:
        full_pattern = str(Path(repo_root) / sub)
        for match in glob(full_pattern, recursive=True):
            if Path(match).is_dir():
                expanded.append(match)
    return expanded

def run_pipeline(config):
    Path("data").mkdir(exist_ok=True)

    for repo_key in ["tf", "pt"]:
        repo_root = config["repos"][repo_key]
        sub_roots = config["test_roots"][repo_key]
        root_dirs = expand_roots(repo_root, sub_roots)

        include = config["include_globs"]
        exclude = config["exclude_globs"]

        # Step 1: Discover
        files = discover_test_files(root_dirs, include, exclude)
        with open(f"data/files_{repo_key}.jsonl", "w") as f:
            for item in files:
                f.write(json.dumps(item) + "\n")

        # Step 2: Normalize
        normalized = [normalize_file(item) for item in files]
        with open(f"data/norm_{repo_key}.jsonl", "w") as f:
            for item in normalized:
                f.write(json.dumps(item) + "\n")

        # Step 3: Parse tests
        results = []
        for item in normalized:
            tests = parse_test_file(item["abs_path"])
            for t in tests:
                t["file"] = item["rel_path"]
                results.append(t)

        with open(f"data/tests_{repo_key}.parsed.jsonl", "w") as f:
            for t in results:
                f.write(json.dumps(t) + "\n")

if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)
