# ./component/doc_analyzer.py
"""使用大模型分析测试问题，结合官方文档"""
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from component.doc.doc_crawler_factory import get_doc_content, detect_framework
from component.migration.migrate_generate_tests import get_qwen_client, load_api_key

DEFAULT_MODEL = "qwen-flash"
DEFAULT_KEY_PATH = "aliyun.key"


def build_analysis_prompt(
    error_message: str,
    tf_code: str,
    pt_code: str,
    tf_docs: List[str],
    pt_docs: List[str],
    context: Optional[str] = None
) -> str:
    """构建分析提示词"""
    
    tf_docs_text = "\n\n".join(tf_docs) if tf_docs else "未找到相关 TensorFlow 文档"
    pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "未找到相关 PyTorch 文档"
    
    prompt = f"""你是一个资深的深度学习框架专家，擅长分析 TensorFlow 和 PyTorch 之间的差异。

【任务】
分析以下测试迁移中的问题，判断该差异或异常是否正常。

【错误信息】
{error_message}

【TensorFlow 原始代码】
```python
{tf_code}
```

【PyTorch 迁移代码】
```python
{pt_code}
```

【TensorFlow 官方文档】
{tf_docs_text}

【PyTorch 官方文档】
{pt_docs_text}
"""
    
    if context:
        prompt += f"""
【额外上下文】
{context}
"""
    
    prompt += """
【分析要求】
1. 仔细阅读错误信息和代码
2. 参考官方文档，理解两个框架的 API 差异
3. 判断该错误是否是由于框架差异导致的正常现象
4. 如果是正常差异，说明原因和解决方案
5. 如果是迁移错误，指出问题所在

请给出详细的分析结果，包括：
- 错误原因分析
- 是否为正常差异
- 如果是正常差异，说明两个框架的差异点
- 建议的修复方案（如果需要）
"""
    
    return prompt


def analyze_with_llm(
    client,
    error_message: str,
    tf_code: str,
    pt_code: str,
    tf_apis: List[str],
    pt_apis: List[str],
    context: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Optional[str]:
    """使用 LLM 分析问题"""
    
    # 爬取相关文档
    print(f"[INFO] 正在爬取文档...")
    tf_docs = []
    pt_docs = []
    
    for api in tf_apis[:5]:  # 最多爬取5个相关 API 的文档
        try:
            doc_content = get_doc_content(api, "tensorflow")
            if doc_content and "无法获取" not in doc_content:
                tf_docs.append(doc_content)
        except Exception as e:
            print(f"[WARN] 爬取 TF 文档失败 {api}: {e}")
    
    for api in pt_apis[:5]:
        try:
            doc_content = get_doc_content(api, "pytorch")
            if doc_content and "无法获取" not in doc_content:
                pt_docs.append(doc_content)
        except Exception as e:
            print(f"[WARN] 爬取 PT 文档失败 {api}: {e}")
    
    print(f"[INFO] 已获取 {len(tf_docs)} 个 TF 文档，{len(pt_docs)} 个 PT 文档")
    
    # 构建提示词
    prompt = build_analysis_prompt(
        error_message, tf_code, pt_code, tf_docs, pt_docs, context
    )
    
    # 调用 LLM
    try:
        if hasattr(client, 'chat'):
            # 新版本
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            analysis = resp.choices[0].message.content.strip()
        else:
            # 旧版本
            resp = client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            analysis = resp.choices[0].message.content.strip()
        
        return analysis
    except Exception as e:
        print(f"[ERROR] LLM 调用失败: {e}")
        return None


def analyze_test_error(
    error_message: str,
    test_file: str,
    tf_apis: Optional[List[str]] = None,
    pt_apis: Optional[List[str]] = None,
    context: Optional[str] = None
) -> Optional[str]:
    """分析测试错误"""
    
    # 读取测试文件
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"[ERROR] 测试文件不存在: {test_file}")
        return None
    
    test_content = test_path.read_text(encoding='utf-8')
    
    # 提取 TF 和 PT 代码（简单提取，可以根据需要改进）
    import re
    
    # 提取 TF 测试函数
    tf_match = re.search(r'def\s+(test\w+)\(\):.*?(?=def\s+test.*?_pt|# ===== PyTorch|if __name__)', 
                         test_content, re.DOTALL)
    tf_code = tf_match.group(0) if tf_match else ""
    
    # 提取 PT 测试函数
    pt_match = re.search(r'def\s+(test.*?_pt)\(\):.*?(?=def\s+test.*?|# ===== Main|if __name__)', 
                         test_content, re.DOTALL)
    pt_code = pt_match.group(0) if pt_match else ""
    
    # 如果没有提供 API 列表，尝试从代码中提取
    if not tf_apis:
        tf_apis = re.findall(r'tf\.\w+(?:\.\w+)*', tf_code)
    if not pt_apis:
        pt_apis = re.findall(r'torch\.\w+(?:\.\w+)*', pt_code)
    
    # 初始化 LLM 客户端
    try:
        client = get_qwen_client(DEFAULT_KEY_PATH)
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return None
    
    # 调用分析
    return analyze_with_llm(
        client, error_message, tf_code, pt_code, tf_apis, pt_apis, context
    )


def main():
    """命令行工具"""
    parser = argparse.ArgumentParser(description="分析测试迁移问题")
    parser.add_argument("error", help="错误信息")
    parser.add_argument("--test-file", "-t", required=True, help="测试文件路径")
    parser.add_argument("--tf-apis", nargs="+", help="TensorFlow API 列表")
    parser.add_argument("--pt-apis", nargs="+", help="PyTorch API 列表")
    parser.add_argument("--context", "-c", help="额外上下文信息")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="LLM 模型名称")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH, help="API key 路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 分析问题
    analysis = analyze_test_error(
        args.error,
        args.test_file,
        args.tf_apis,
        args.pt_apis,
        args.context
    )
    
    if analysis:
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(analysis)
            print(f"[SUCCESS] 分析结果已保存到: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("分析结果")
            print("=" * 80)
            print(analysis)
    else:
        print("[ERROR] 分析失败")


if __name__ == "__main__":
    main()

