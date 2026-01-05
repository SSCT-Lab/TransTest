# ./component/migrate_extract_tf_tests.py
# 提取 TensorFlow 测试逻辑，创建独立的测试文件
import json
import ast
import argparse
from pathlib import Path
from tqdm import tqdm
import re

def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []

def extract_test_function_code(file_path, test_name):
    """从 TensorFlow 测试文件中提取测试函数代码"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        
        # 如果测试函数名是 unknown_test_*，尝试从文件中查找实际的测试函数
        actual_test_name = test_name
        if test_name.startswith("unknown_test_"):
            test_funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
            if test_funcs:
                actual_test_name = test_funcs[0]
        
        # 查找测试函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == actual_test_name:
                lines = source.split('\n')
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)
                func_lines = lines[start_line:end_line]
                
                # 移除基础缩进
                if func_lines:
                    base_indent = len(func_lines[0]) - len(func_lines[0].lstrip())
                    if base_indent > 0:
                        func_lines = [line[base_indent:] if len(line) > base_indent else line for line in func_lines]
                
                return '\n'.join(func_lines), actual_test_name
    except Exception as e:
        pass
    return None, test_name

def find_test_class(file_path, test_name):
    """查找包含测试函数的类名和类定义"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == test_name:
                        # 返回类名和类的完整代码
                        lines = source.split('\n')
                        class_start = node.lineno - 1
                        class_end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)
                        class_code = '\n'.join(lines[class_start:class_end])
                        return node.name, class_code
    except:
        pass
    return None, None

def extract_imports(file_path):
    """提取文件中的 import 语句"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.asname or alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join([alias.asname or alias.name for alias in node.names])
                imports.append(f"from {module} import {names}")
        
        return list(set(imports))  # 去重
    except:
        return []

def convert_tf_test_to_standalone(test_code, class_code=None, imports=None, test_name="test_extracted"):
    """将 TensorFlow 测试代码转换为独立可执行的测试"""
    # 检查是否是类方法
    is_class_method = 'self' in test_code and 'def ' in test_code
    
    # 准备 imports
    import_lines = []
    if imports:
        import_lines.extend(imports)
    else:
        import_lines.append("import tensorflow as tf")
        import_lines.append("import numpy as np")
    
    if is_class_method:
        # 如果是类方法，提取测试方法体
        # 提取函数定义行
        func_def_match = re.search(r'def\s+(\w+)\s*\([^)]*\)\s*:', test_code)
        if not func_def_match:
            return None
        
        # 提取函数体（去掉 def 行和缩进）
        test_lines = test_code.split('\n')
        test_body = []
        in_def = False
        base_indent = None
        
        for line in test_lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                in_def = True
                # 获取基础缩进
                base_indent = len(line) - len(line.lstrip())
                continue
            if in_def:
                if not stripped:  # 空行
                    continue
                # 移除基础缩进
                if base_indent and len(line) > base_indent:
                    line = line[base_indent:]
                # 简化 self. 调用（移除 self 参数）
                line = re.sub(r'\bself\.', '', line)
                # 处理测试框架方法
                line = re.sub(r'\bassertAllClose\(', 'np.testing.assert_allclose(', line)
                line = re.sub(r'\bassertAllEqual\(', 'np.testing.assert_array_equal(', line)
                line = re.sub(r'\bassertEqual\(', 'assert ', line)
                line = re.sub(r'\bassertTrue\(', 'assert ', line)
                line = re.sub(r'\bassertFalse\(', 'assert not ', line)
                # 处理 cached_session() 和 session()
                line = re.sub(r'\bcached_session\(\)', 'tf.compat.v1.Session()', line)
                line = re.sub(r'\bsession\(\)', 'tf.compat.v1.Session()', line)
                # 处理 .eval() 调用（在 session 中）
                if '.eval()' in line and 'Session()' not in line:
                    # 如果不在 session 上下文中，尝试使用 eager execution
                    line = line.replace('.eval()', '.numpy()')
                test_body.append(line)
        
        # 确保 test_body 中的代码有正确的缩进
        indented_body = []
        for line in test_body:
            if line.strip():  # 非空行
                # 如果行没有缩进，添加缩进
                if not line.startswith(' ') and not line.startswith('\t'):
                    indented_body.append('        ' + line)
                else:
                    # 保持原有缩进，但确保至少是 8 个空格
                    indented_body.append('        ' + line.lstrip())
            else:
                indented_body.append('')
        
        # 添加必要的辅助函数
        helper_functions = """
# Helper functions for test framework methods
def assertAllClose(a, b, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def assertAllEqual(a, b):
    np.testing.assert_array_equal(a, b)

# Enable eager execution for simpler evaluation
tf.config.run_functions_eagerly(True)
"""
        
        standalone_code = f"""# Extracted TensorFlow test logic
{chr(10).join(import_lines)}
{helper_functions}

# Test logic extracted from TensorFlow test class method
def {test_name}():
    try:
{chr(10).join(indented_body)}
        print("PASS")
    except Exception as e:
        print(f"FAIL: {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    {test_name}()
"""
    else:
        # 独立函数，直接使用
        func_name_match = re.search(r'def\s+(\w+)\s*\(', test_code)
        func_name = func_name_match.group(1) if func_name_match else "test_extracted"
        
        standalone_code = f"""# Extracted TensorFlow test logic
{chr(10).join(import_lines)}

# Test function extracted from TensorFlow test
{test_code}

if __name__ == "__main__":
    try:
        {func_name}()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {{e}}")
        import traceback
        traceback.print_exc()
"""
    
    return standalone_code

def safe_filename(s):
    """生成安全的文件名"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

def main():
    parser = argparse.ArgumentParser(description="提取 TensorFlow 测试逻辑，创建独立的测试文件")
    parser.add_argument("--input", default="data/migration_candidates_fuzzy.jsonl", 
                       help="输入文件（候选测试列表）")
    parser.add_argument("--tf-root", default="framework/tensorflow-master", 
                       help="TensorFlow 源码根目录")
    parser.add_argument("--output-dir", default="extracted_tf_tests", 
                       help="输出目录（存放提取的测试文件）")
    parser.add_argument("--limit", type=int, default=-1, 
                       help="限制测试数量，-1 表示全部")
    parser.add_argument("--tests-tf-mapped", default="data/tests_tf.mapped.jsonl",
                       help="测试元数据文件（用于查找真实测试函数名）")
    args = parser.parse_args()
    
    # 加载候选测试
    candidates = load_jsonl(args.input)
    print(f"[LOAD] 加载了 {len(candidates)} 个候选测试")
    
    # 加载测试元数据
    tests_tf_mapped = None
    if Path(args.tests_tf_mapped).exists():
        tests_tf_mapped = load_jsonl(args.tests_tf_mapped)
        print(f"[LOAD] 加载了 {len(tests_tf_mapped)} 条测试元数据")
    
    # 限制数量
    if args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"[INFO] 限制测试数量: {args.limit}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    tf_root = Path(args.tf_root)
    
    # 提取源文件信息
    def extract_source_info(file_path, test_name, tests_tf_mapped):
        """从文件路径和测试名中提取信息"""
        if tests_tf_mapped:
            file_key = file_path.replace("framework/tensorflow-master/", "")
            for test_data in tests_tf_mapped:
                if file_key in test_data.get("file", "") or test_data.get("file", "").endswith(Path(file_path).name):
                    real_name = test_data.get("name", test_name)
                    if real_name.startswith("test"):
                        return file_path, real_name
        return file_path, test_name
    
    extracted_count = 0
    failed_count = 0
    
    for item in tqdm(candidates, desc="Extracting TF tests"):
        file_path = item.get("file", "")
        test_name = item.get("name", "")
        
        # 提取真实测试函数名
        tf_file_path, actual_test_name = extract_source_info(file_path, test_name, tests_tf_mapped)
        
        # 处理文件路径
        if tf_file_path.startswith("framework/tensorflow-master/"):
            full_path = Path(tf_file_path)
        elif Path(tf_file_path).exists():
            full_path = Path(tf_file_path)
        else:
            full_path = tf_root / tf_file_path
        
        if not full_path.exists():
            failed_count += 1
            continue
        
        # 提取测试函数代码
        test_code, actual_test_name = extract_test_function_code(full_path, actual_test_name)
        
        if not test_code:
            failed_count += 1
            continue
        
        # 提取 imports
        imports = extract_imports(full_path)
        
        # 检查是否是类方法
        is_class_method = 'self' in test_code and 'def ' in test_code
        class_code = None
        if is_class_method:
            class_name, class_code = find_test_class(full_path, actual_test_name)
        
        # 转换为独立测试
        standalone_code = convert_tf_test_to_standalone(test_code, class_code, imports, actual_test_name)
        
        if not standalone_code:
            failed_count += 1
            continue
        
        # 保存到文件
        safe_name = safe_filename(f"{Path(file_path).stem}_{actual_test_name}")
        output_file = output_dir / f"{safe_name}.py"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Extracted from: {file_path}:{test_name}\n")
            f.write(f"# Original test: {actual_test_name}\n")
            f.write(standalone_code)
        
        extracted_count += 1
    
    print("\n==== EXTRACTION SUMMARY ====")
    print(f"总测试数: {len(candidates)}")
    print(f"成功提取: {extracted_count}")
    print(f"失败: {failed_count}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()

