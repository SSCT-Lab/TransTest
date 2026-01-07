# ./component/doc_crawler_tensorflow.py
"""TensorFlow 文档爬取器"""
import re
from typing import Dict
from bs4 import BeautifulSoup
from component.doc.doc_crawler_base import DocCrawler

TF_DOC_BASE = "https://www.tensorflow.org/api_docs/python/tf/"


class TensorFlowDocCrawler(DocCrawler):
    """TensorFlow 文档爬取器"""
    
    def __init__(self):
        super().__init__("tensorflow")
    
    def build_doc_url(self, api_name: str) -> str:
        """构建 TensorFlow 文档 URL"""
        api_parts = api_name.split('.')
        
        if len(api_parts) == 1:
            return f"{TF_DOC_BASE}{api_name}"
        else:
            # 模块.函数名格式
            module_path = '/'.join(api_parts[:-1])
            func_name = api_parts[-1]
            return f"{TF_DOC_BASE}{module_path}/{func_name}"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """解析 TensorFlow 文档内容"""
        doc_content = {
            "api_name": api_name,
            "framework": "tensorflow",
            "url": url,
            "title": soup.find('title').text if soup.find('title') else "",
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": [],
            "raw_html": str(soup.find('main') or soup.find('body') or '')
        }
        
        # 提取主要描述
        main_content = soup.find('main') or soup.find('div', class_='devsite-article-body')
        if main_content:
            # 提取第一个段落作为描述
            first_p = main_content.find('p')
            if first_p:
                doc_content["description"] = first_p.get_text(strip=True)
            
            # 提取参数说明（TensorFlow 使用特定的 HTML 结构）
            params_section = main_content.find('section', {'id': 'args'}) or main_content.find('h2', string=re.compile(r'Args?', re.I))
            if params_section:
                params = []
                # TensorFlow 文档中参数通常在 <code> 标签中
                for code in params_section.find_all('code'):
                    param_name = code.get_text(strip=True)
                    # 查找参数描述
                    parent = code.find_parent()
                    if parent:
                        param_desc = parent.get_text(strip=True).replace(param_name, '').strip()
                        if param_desc:
                            params.append({"name": param_name, "description": param_desc})
                doc_content["parameters"] = params
            
            # 提取返回值说明
            returns_section = main_content.find('section', {'id': 'returns'}) or main_content.find('h2', string=re.compile(r'Returns?', re.I))
            if returns_section:
                doc_content["returns"] = returns_section.get_text(strip=True)
        
        return doc_content

