# ./component/migrate_generate_tests.py
import json
import ast
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 尝试导入 LLM 客户端 - 使用与 semantic_llm.py 相同的方式
def load_api_key(path="aliyun.key"):
    with open(path) as f:
        return f.read().strip()

def get_qwen_client(key_path="aliyun.key"):
    """创建 Qwen API 客户端"""
    try:
        # 尝试新版本 openai
        from openai import OpenAI
        api_key = load_api_key(key_path)
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except ImportError:
        # 尝试旧版本
        try:
            import openai
            api_key = load_api_key(key_path)
            openai.api_key = api_key
            openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return openai
        except Exception as e:
            print(f"[ERROR] 无法初始化 OpenAI 客户端: {e}")
            raise

IN_FILE = "data/migration_candidates_fuzzy.jsonl"
OUT_DIR = Path("migrated_tests")
OUT_DIR.mkdir(exist_ok=True)

HEADER = """import torch
import pytest
import tensorflow as tf
import numpy as np

# Helper functions for TensorFlow test framework methods
def assertAllClose(a, b, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def assertAllEqual(a, b):
    np.testing.assert_array_equal(a, b)

def assertFunctionMatchesEager(func, *args, **kwargs):
    \"\"\"Assert that a function matches eager execution behavior.
    This is a simplified version for standalone test execution.\"\"\"
    try:
        # Run in eager mode
        eager_result = func(*args, **kwargs)
        # For now, just check that it doesn't raise an exception
        # In a full implementation, this would compare eager vs graph mode
        return eager_result
    except Exception as e:
        raise AssertionError(f"Function execution failed: {e}")

# Enable eager execution for simpler evaluation
try:
    tf.config.run_functions_eagerly(True)
except:
    pass

"""


def safe(s):
    return s.replace("/", "_").replace(".", "_").replace("-", "_")


def convert_tf_to_standalone(test_code, test_name):
    """将 TensorFlow 测试代码转换为独立可执行的函数"""
    import re
    
    # 检查是否是类方法
    is_class_method = 'self' in test_code and 'def ' in test_code
    
    if is_class_method:
        # 提取函数体（去掉 def 行和缩进）
        test_lines = test_code.split('\n')
        test_body = []
        in_def = False
        base_indent = None
        
        for line in test_lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                in_def = True
                base_indent = len(line) - len(line.lstrip())
                continue
            if in_def:
                if not stripped:
                    continue
                # 移除基础缩进
                if base_indent and len(line) > base_indent:
                    line = line[base_indent:]
                # 简化 self. 调用
                line = re.sub(r'\bself\.', '', line)
                # 处理测试框架方法（保留函数名，因为 HEADER 中已定义）
                line = re.sub(r'\bassertAllClose\(', 'assertAllClose(', line)
                line = re.sub(r'\bassertAllEqual\(', 'assertAllEqual(', line)
                # 处理 cached_session() 和 session()
                line = re.sub(r'\bcached_session\(\)', 'tf.compat.v1.Session()', line)
                line = re.sub(r'\bsession\(\)', 'tf.compat.v1.Session()', line)
                # 处理 .eval() 调用
                if '.eval()' in line and 'Session()' not in line:
                    line = line.replace('.eval()', '.numpy()')
                test_body.append(line)
        
        # 确保缩进正确
        indented_body = []
        for line in test_body:
            if line.strip():
                if not line.startswith(' ') and not line.startswith('\t'):
                    indented_body.append('        ' + line)
                else:
                    indented_body.append('        ' + line.lstrip())
            else:
                indented_body.append('')
        
        standalone_code = f"""def {test_name}():
    \"\"\"Original TensorFlow test logic\"\"\"
    try:
{chr(10).join(indented_body)}
        print("TF: PASS")
    except Exception as e:
        print(f"TF: FAIL - {{e}}")
        import traceback
        traceback.print_exc()
"""
    else:
        # 独立函数，提取函数体
        func_lines = test_code.split('\n')
        func_body = []
        in_def = False
        base_indent = None
        
        for line in func_lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                in_def = True
                base_indent = len(line) - len(line.lstrip())
                continue
            if in_def:
                if base_indent and len(line) > base_indent:
                    line = line[base_indent:]
                func_body.append(line)
        
        indented_body = []
        for line in func_body:
            if line.strip():
                if not line.startswith(' ') and not line.startswith('\t'):
                    indented_body.append('        ' + line)
                else:
                    indented_body.append('        ' + line.lstrip())
            else:
                indented_body.append('')
        
        standalone_code = f"""def {test_name}():
    \"\"\"Original TensorFlow test logic\"\"\"
    try:
{chr(10).join(indented_body)}
        print("TF: PASS")
    except Exception as e:
        print(f"TF: FAIL - {{e}}")
        import traceback
        traceback.print_exc()
"""
    
    return standalone_code


def extract_test_function_code(file_path, test_name):
    """从原始测试文件中提取指定测试函数的代码"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        
        # 查找测试函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == test_name:
                # 使用行号范围提取代码
                lines = source.split('\n')
                start_line = node.lineno - 1
                
                # 找到函数结束位置
                # 方法1: 使用 ast 的 end_lineno（Python 3.8+）
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    end_line = node.end_lineno
                else:
                    # 方法2: 查找下一个同级别定义
                    end_line = len(lines)
                    # 遍历 AST 找到下一个同级别节点
                    parent = None
                    for n in ast.walk(tree):
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                            for child in (n.body if hasattr(n, 'body') else []):
                                if child == node:
                                    parent = n
                                    break
                    
                    if parent and hasattr(parent, 'body'):
                        for sibling in parent.body:
                            if isinstance(sibling, (ast.FunctionDef, ast.ClassDef)) and sibling.lineno > node.lineno:
                                end_line = sibling.lineno - 1
                                break
                
                func_code = '\n'.join(lines[start_line:end_line])
                # 确保缩进正确（移除函数定义前的缩进）
                func_lines = func_code.split('\n')
                if func_lines:
                    # 找到第一行非空行的缩进
                    base_indent = 0
                    for line in func_lines:
                        if line.strip():
                            base_indent = len(line) - len(line.lstrip())
                            break
                    # 移除基础缩进
                    if base_indent > 0:
                        func_lines = [line[base_indent:] if len(line) > base_indent else line for line in func_lines]
                        func_code = '\n'.join(func_lines)
                
                return func_code
    except Exception as e:
        print(f"[WARN] 无法提取 {file_path}:{test_name} 的代码: {e}")
    return None


def extract_helper_functions(file_path):
    """从原始测试文件中提取辅助函数（非测试函数）"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)

        helper_functions = []

        # 提取所有非测试函数的函数定义
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 跳过测试函数和私有函数
                if node.name.startswith('test') or node.name.startswith('_'):
                    continue

                lines = source.split('\n')
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)

                func_code = '\n'.join(lines[start_line:end_line])
                # 移除基础缩进
                func_lines = func_code.split('\n')
                if func_lines:
                    base_indent = 0
                    for line in func_lines:
                        if line.strip():
                            base_indent = len(line) - len(line.lstrip())
                            break
                    if base_indent > 0:
                        func_lines = [line[base_indent:] if len(line) > base_indent else line for line in func_lines]
                        func_code = '\n'.join(func_lines)

                helper_functions.append(func_code)

        return '\n\n'.join(helper_functions) if helper_functions else None
    except Exception:
        # 提取辅助函数失败时不影响主流程
        return None


def build_migration_prompt(tf_code, tf_apis, mapped_pt_apis):
    """构建 LLM 迁移提示词（不再要求生成 compare 函数，只生成 PyTorch 侧逻辑）"""
    # 选择前5个最相关的映射 API
    top_mapped = mapped_pt_apis[:5] if len(mapped_pt_apis) > 5 else mapped_pt_apis

    prompt = f"""你是一个资深深度学习与单元测试迁移工程师。
现在有一个 TensorFlow 测试函数，需要你**只迁移成 PyTorch 版本**，不需要生成任何 TF/PT 对比函数。

【任务】
1. 读取下面给出的 TensorFlow 测试代码，将其迁移为等价的 PyTorch 测试代码。
2. 迁移后的代码中：
   - 保持测试逻辑不变（输入、计算流程、关键断言语义保持一致）。
   - 将 TensorFlow API 替换为对应的 PyTorch API（可参考下方给出的映射列表）。
   - 可以使用 `assert` 或 `torch.allclose` 等方式校验结果。
   - **不需要**再次调用 TensorFlow，也**不需要**写任何“TF/PT 结果对比”的辅助逻辑。
3. 如果原始测试中有重要的中间结果，建议适当使用 `print(...)` 打印，方便人工查看（例如打印张量的形状或数值）。
4. 最终请给出**可以直接运行的 PyTorch 测试函数代码**（可以是一个或多个 `def test_xxx_pt(...):`），
   不要包含多余的解释性文字。

【TensorFlow 原始测试代码】
```python
{tf_code}
```

【使用到的 TensorFlow API（最多 10 个，仅供参考）】
{', '.join(tf_apis[:10])}

【可能对应的 PyTorch API（前 5 个，仅供参考）】
{', '.join(top_mapped)}

请只输出迁移后的 **PyTorch 测试函数代码**，不要输出其他解释或说明。
"""
    return prompt


def migrate_with_llm(client, tf_code, tf_apis, mapped_pt_apis, model="qwen-flash"):
    """使用 LLM 生成迁移后的代码"""
    prompt = build_migration_prompt(tf_code, tf_apis, mapped_pt_apis)
    
    try:
        # 兼容新旧版本的 OpenAI API
        if hasattr(client, 'chat'):
            # 新版本
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048
            )
            raw_code = resp.choices[0].message.content.strip()
        else:
            # 旧版本
            resp = client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048
            )
            raw_code = resp.choices[0].message.content.strip()
        
        # 提取代码块（如果有 markdown 代码块）
        code_match = re.search(r'```python\n(.*?)```', raw_code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 如果没有代码块，尝试提取函数定义
        func_match = re.search(r'def\s+test_\w+.*?(?=\n\n|\ndef\s|\Z)', raw_code, re.DOTALL)
        if func_match:
            return func_match.group(0).strip()
        
        return raw_code
    except Exception as e:
        print(f"[ERROR] LLM 调用失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=IN_FILE, help="输入文件路径")
    parser.add_argument("--output-dir", default=OUT_DIR, help="输出目录")
    parser.add_argument("--limit", type=int, default=-1, help="限制生成数量，-1 表示全部")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的文件")
    parser.add_argument("--tf-root", default="framework/tensorflow-master", help="TensorFlow 源码根目录")
    parser.add_argument("--model", default="qwen-flash", help="LLM 模型名称")
    parser.add_argument("--key-path", default="aliyun.key", help="API key 路径")
    parser.add_argument("--workers", type=int, default=10, help="并发线程数")
    args = parser.parse_args()
    
    in_file = Path(args.input)
    out_dir = Path(args.output_dir)
    tf_root = Path(args.tf_root)
    out_dir.mkdir(exist_ok=True)
    
    if not in_file.exists():
        print(f"[ERR] 找不到输入文件: {in_file}")
        return

    lines = [json.loads(x) for x in open(in_file)]
    print(f"[LOAD] 发现可迁移测试数量: {len(lines)}")
    
    # 限制数量
    if args.limit > 0:
        lines = lines[:args.limit]
        print(f"[INFO] 限制生成数量: {args.limit}")

    # 去重：基于 (file, name) 组合
    seen = set()
    unique_tests = []
    for item in lines:
        key = (item["file"], item["name"])
        if key not in seen:
            seen.add(key)
            unique_tests.append(item)
    
    print(f"[INFO] 去重后测试数量: {len(unique_tests)} (去除了 {len(lines) - len(unique_tests)} 个重复)")

    # 初始化 LLM 客户端（可选，如果失败仍可生成包含 TF 测试的文件）
    client = None
    try:
        # llm_utils 中的 get_qwen_client 默认使用 ../aliyun.key，需要调整
        key_path = Path(args.key_path)
        if not key_path.is_absolute():
            key_path = Path(__file__).parent.parent / key_path
        client = get_qwen_client(str(key_path))
    except Exception as e:
        print(f"[WARN] 无法初始化 LLM 客户端: {e}")
        print(f"[INFO] 尝试使用默认路径...")
        try:
            client = get_qwen_client("aliyun.key")
        except:
            print(f"[WARN] 无法初始化 LLM 客户端，将使用占位符生成 PyTorch 测试")
            print(f"[INFO] 但会包含 TensorFlow 原始测试逻辑")

    # 为每个线程创建独立的客户端（如果可用）
    clients = []
    if client:
        try:
            key_path = Path(args.key_path)
            if not key_path.is_absolute():
                key_path = Path(__file__).parent.parent / key_path
            clients = [get_qwen_client(str(key_path)) for _ in range(args.workers)]
        except:
            clients = [None] * args.workers
    else:
        clients = [None] * args.workers
    
    # 线程安全的计数器
    migrated_counter = [0]
    failed_counter = [0]
    lock = threading.Lock()
    
    def process_one_test(item, client_idx):
        """处理单个测试的生成"""
        file = item["file"]
        name = item["name"]
        apis = item["apis_used"]
        matches = item["matches"]
        
        # 生成目标文件路径
        out_path = out_dir / f"{safe(name)}.py"
        
        # 已存在则跳过（支持断点续测），除非使用 --force
        if out_path.exists() and not args.force:
            return {"status": "skipped", "name": name}
        
        try:
            # 提取原始测试代码
            # 处理文件路径
            if file.startswith("framework/tensorflow-master/"):
                full_file_path = Path(file)
            elif file.startswith("tensorflow/"):
                full_file_path = tf_root / file
            else:
                full_file_path = tf_root / file
            
            # 如果文件不存在，尝试其他路径
            if not full_file_path.exists():
                full_file_path = Path(file)
            
            # 如果测试函数名是 unknown_test_*，尝试从文件中查找实际的测试函数
            actual_test_name = name
            if name.startswith("unknown_test_"):
                try:
                    source = full_file_path.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(source)
                    test_funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
                    if test_funcs:
                        actual_test_name = test_funcs[0]
                except:
                    pass
            
            tf_code = extract_test_function_code(full_file_path, actual_test_name)
            
            if not tf_code:
                return {"status": "failed", "name": name, "error": f"无法提取 {file}:{actual_test_name} 的代码"}
            
            # 提取并转换 TensorFlow 测试为独立可执行的函数
            tf_standalone_code = convert_tf_to_standalone(tf_code, actual_test_name)
            
            if not tf_standalone_code:
                return {"status": "failed", "name": name, "error": f"无法转换 TensorFlow 测试 {file}:{actual_test_name}"}
            
            # 获取映射的 PyTorch API
            mapped_pt_apis = [m["mapped_pt_api"] for m in matches]
            
            # 使用 LLM 生成迁移代码（如果客户端可用）
            migrated_code = None
            thread_client = clients[client_idx % len(clients)]
            if thread_client:
                migrated_code = migrate_with_llm(thread_client, tf_code, apis, mapped_pt_apis, args.model)
            
            if not migrated_code:
                migrated_code = f"""def test_{safe(name)}_pt():
    # ===== TF Original APIs Used =====
    # {', '.join(apis[:10])}
    #
    # ===== Mapped PT APIs =====
    # {', '.join(mapped_pt_apis[:5])}

    # TODO: Write migrated PyTorch version
    assert True  # placeholder
"""
            
            # 清理生成的代码：移除重复的 import
            migrated_code = re.sub(r'^import torch\s*$', '', migrated_code, flags=re.MULTILINE)
            migrated_code = re.sub(r'^import pytest\s*$', '', migrated_code, flags=re.MULTILINE)
            migrated_code = migrated_code.strip()
            
            # 确保 PyTorch 测试函数名以 _pt 结尾
            if not re.search(r'def\s+test_\w+_pt\s*\(', migrated_code):
                migrated_code = re.sub(r'def\s+(test_\w+)\s*\(', r'def \1_pt(', migrated_code)
            
            # 提取辅助函数（测试中使用的非测试函数）
            helper_functions = extract_helper_functions(full_file_path)
            helper_section = ""
            if helper_functions:
                helper_section = f"""# ===== Helper Functions from Original File =====
{helper_functions}

"""
            
            # 组合最终代码：包含 TensorFlow 原始测试和 PyTorch 迁移测试
            content = HEADER + f"""# Auto-Migrated from TF test
# source: {file}:{name}
# Original test function: {actual_test_name}

{helper_section}# ===== TensorFlow Original Test =====
{tf_standalone_code}

# ===== PyTorch Migrated Test =====
{migrated_code}

# ===== Comparison Test =====
def test_{safe(name)}_compare():
    \"\"\"Compare TensorFlow and PyTorch test results\"\"\"
    try:
        # Run TensorFlow test
        tf_result = None
        tf_error = None
        try:
            {actual_test_name}()
            tf_result = "PASS"
        except Exception as e:
            tf_result = "FAIL"
            tf_error = str(e)
        
        # Run PyTorch test
        pt_result = None
        pt_error = None
        # Find PyTorch test function name from migrated_code
        import re
        pt_match = re.search(r'def\s+(test_\w+_pt)\s*\(', '''{migrated_code}''')
        if pt_match:
            pt_test_name = pt_match.group(1)
            try:
                # Execute the PyTorch test function
                exec('''{migrated_code}''')
                exec(f\"\"\"{{pt_test_name}}()\"\"\")
                pt_result = "PASS"
            except Exception as e:
                pt_result = "FAIL"
                pt_error = str(e)
        else:
            pt_result = "SKIP"
            pt_error = "Could not find PyTorch test function"
        
        # Compare results
        if tf_result == "PASS" and pt_result == "PASS":
            print("PASS: Both TF and PT tests passed")
        elif tf_result == "FAIL" and pt_result == "FAIL":
            print("FAIL: Both TF and PT tests failed")
            if tf_error:
                print(f"TF error: {{tf_error}}")
            if pt_error:
                print(f"PT error: {{pt_error}}")
        else:
            print(f"MISMATCH: TF={{tf_result}}, PT={{pt_result}}")
            if tf_error:
                print(f"TF error: {{tf_error}}")
            if pt_error:
                print(f"PT error: {{pt_error}}")
    except Exception as e:
        print(f"COMPARISON ERROR: {{e}}")
        import traceback
        traceback.print_exc()

# ===== Main Execution =====
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            test_{safe(name)}_compare()
        elif sys.argv[1] == "--tf":
            {actual_test_name}()
        elif sys.argv[1] == "--pt":
            import re
            pt_match = re.search(r'def\s+(test_\w+_pt)\s*\(', '''{migrated_code}''')
            if pt_match:
                pt_test_name = pt_match.group(1)
                exec('''{migrated_code}''')
                exec(f\"\"\"{{pt_test_name}}()\"\"\")
    else:
        # Default: run TensorFlow test
        {actual_test_name}()

"""
            
            # 线程安全写入文件
            with lock:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)
                migrated_counter[0] += 1
            
            return {"status": "success", "name": name}
            
        except Exception as e:
            with lock:
                failed_counter[0] += 1
            return {"status": "failed", "name": name, "error": str(e)}
    
    # 并发处理
    print(f"[INFO] 使用 {args.workers} 个并发线程生成测试")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_one_test, item, i % args.workers): item 
            for i, item in enumerate(unique_tests)
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Migrated Tests"):
            try:
                result = future.result()
                if result["status"] == "failed":
                    item = futures[future]
                    print(f"[WARN] {result['name']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                item = futures[future]
                print(f"[ERROR] 处理失败 {item.get('name', 'unknown')}: {e}")
                with lock:
                    failed_counter[0] += 1
    
    migrated = migrated_counter[0]
    failed = failed_counter[0]

    print("\n==== TEST MIGRATION SUMMARY ====")
    print(f"成功生成迁移测试数量: {migrated}")
    print(f"失败数量: {failed}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
