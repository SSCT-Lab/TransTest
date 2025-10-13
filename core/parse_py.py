import ast
import json
from pathlib import Path

class TestVisitor(ast.NodeVisitor):
    def __init__(self):
        self.tests = []

    def visit_FunctionDef(self, node):
        if node.name.startswith("test"):
            apis = []
            asserts = []
            # 遍历子节点收集 API 调用 & 断言
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    # 提取函数调用完整名
                    if isinstance(child.func, ast.Attribute):
                        chain = []
                        current = child.func
                        while isinstance(current, ast.Attribute):
                            chain.append(current.attr)
                            current = current.value
                        if isinstance(current, ast.Name):
                            chain.append(current.id)
                        apis.append(".".join(reversed(chain)))
                    elif isinstance(child.func, ast.Name):
                        apis.append(child.func.id)
                if isinstance(child, ast.Assert):
                    asserts.append("assert")
            self.tests.append({
                "name": node.name,
                "lineno": node.lineno,
                "apis": apis,
                "asserts": asserts,
            })
        self.generic_visit(node)

def parse_test_file(abs_path):
    try:
        source = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        visitor = TestVisitor()
        visitor.visit(tree)
        return visitor.tests
    except SyntaxError:
        return []

if __name__ == "__main__":
    # 读取规范化后的文件清单
    results = []
    with open("data/norm_tf.jsonl") as f:
        for line in f:
            item = json.loads(line)
            tests = parse_test_file(item["abs_path"])
            for t in tests:
                t["file"] = item["rel_path"]
                results.append(t)

    with open("data/tests_tf.parsed.jsonl", "w") as f:
        for t in results:
            f.write(json.dumps(t) + "\n")
