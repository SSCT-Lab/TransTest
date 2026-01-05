import ast
import json
from pathlib import Path

class ImportCollector(ast.NodeVisitor):
    """收集文件中的所有 import 语句，识别 TensorFlow 相关模块"""
    def __init__(self):
        self.tf_modules = set()  # TensorFlow 相关的模块名
        self.tf_aliases = {}  # 别名映射，如 {constant_op: tensorflow.python.ops.constant_op}
        
    def visit_Import(self, node):
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            # 检查是否是 TensorFlow 相关
            if self._is_tf_module(name):
                self.tf_modules.add(asname)
                self.tf_aliases[asname] = name
                
    def visit_ImportFrom(self, node):
        if node.module:
            # 检查是否是 TensorFlow 相关模块
            if self._is_tf_module(node.module):
                # from tensorflow.xxx import yyy 或 from tensorflow.python.ops import constant_op
                for alias in node.names:
                    imported_name = alias.name
                    asname = alias.asname or imported_name
                    # 构建完整路径
                    full_path = f"{node.module}.{imported_name}"
                    self.tf_modules.add(asname)
                    self.tf_aliases[asname] = full_path
            # 即使 module 不是 tensorflow，也要检查导入的名称是否可能是 TF 相关的
            # 例如：from some_module import tensorflow_function
            # 这种情况较少，但为了全面性也处理
    
    def _is_tf_module(self, name):
        """判断是否是 TensorFlow 相关模块"""
        if not name:
            return False
        name_lower = name.lower()
        # 直接是 tensorflow
        if name == "tensorflow" or name.startswith("tensorflow."):
            return True
        # 常见的 TensorFlow 相关模块
        tf_keywords = ["tf", "keras", "tensorflow"]
        return any(kw in name_lower for kw in tf_keywords)

class TestVisitor(ast.NodeVisitor):
    def __init__(self, tf_modules, tf_aliases):
        self.tests = []
        self.tf_modules = tf_modules  # TensorFlow 相关的模块名集合
        self.tf_aliases = tf_aliases  # 别名到完整路径的映射
        
    def _extract_api_name(self, node):
        """提取 API 调用的完整名称"""
        if isinstance(node, ast.Attribute):
            chain = []
            current = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                chain.append(current.id)
                full_name = ".".join(reversed(chain))
                # 检查是否是 TensorFlow 相关
                if self._is_tf_api(full_name, current.id):
                    return full_name
        elif isinstance(node, ast.Name):
            # 直接函数调用，检查是否是 TensorFlow 模块
            if node.id in self.tf_modules:
                # 查找完整路径
                full_path = self.tf_aliases.get(node.id, node.id)
                return full_path
        return None
    
    def _is_tf_api(self, full_name, first_part):
        """判断是否是 TensorFlow API"""
        # 1. 直接以 tf. 开头
        if full_name.startswith("tf."):
            return True
        # 2. 第一个部分是 TensorFlow 相关模块（这是最重要的，因为很多 API 通过 import 导入）
        if first_part in self.tf_modules:
            return True
        # 3. 包含 tensorflow 关键字
        if "tensorflow" in full_name.lower() or "keras" in full_name.lower():
            return True
        # 4. 检查别名映射中是否有完整路径
        if first_part in self.tf_aliases:
            alias_path = self.tf_aliases[first_part]
            if "tensorflow" in alias_path.lower() or "keras" in alias_path.lower():
                return True
        return False

    def visit_FunctionDef(self, node):
        if node.name.startswith("test"):
            apis = []
            asserts = []
            # 遍历子节点收集 API 调用 & 断言
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    # 提取函数调用完整名
                    api_name = self._extract_api_name(child.func)
                    if api_name:
                        apis.append(api_name)
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
        
        # 第一步：收集所有 import 语句，识别 TensorFlow 相关模块
        import_collector = ImportCollector()
        import_collector.visit(tree)
        
        # 第二步：提取测试函数中的 API 调用
        visitor = TestVisitor(import_collector.tf_modules, import_collector.tf_aliases)
        visitor.visit(tree)
        return visitor.tests
    except SyntaxError:
        return []
    except Exception as e:
        # 其他错误也返回空列表，避免中断整个流程
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
