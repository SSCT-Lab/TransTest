import ast
import json
from pathlib import Path

def extract_imports(source_code: str):
    """用 AST 抽取 import 语句"""
    imports = []
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
    except SyntaxError:
        pass
    return imports

def normalize_file(meta_item):
    """给 discover 的结果补充 imports、行数、文件名 token"""
    path = Path(meta_item["abs_path"])
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""
    imports = extract_imports(text)
    n_lines = text.count("\n") + 1
    filename_tokens = path.stem.replace("-", "_").split("_")
    meta_item.update({
        "n_lines": n_lines,
        "imports": imports,
        "filename_tokens": filename_tokens
    })
    return meta_item

if __name__ == "__main__":
    files = []
    with open("data/files_tf.jsonl") as f:
        for line in f:
            files.append(json.loads(line))

    normalized = [normalize_file(item) for item in files]
    with open("../data/norm_tf.jsonl", "w") as f:
        for item in normalized:
            f.write(json.dumps(item) + "\n")
