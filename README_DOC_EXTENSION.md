# 文档爬取器扩展指南

## 架构设计

文档爬取器采用**策略模式 + 工厂模式**设计，具有良好的可扩展性：

```
doc_crawler_base.py          # 抽象基类，定义接口
├── doc_crawler_pytorch.py   # PyTorch 实现
├── doc_crawler_tensorflow.py # TensorFlow 实现
├── doc_crawler_factory.py   # 工厂类，管理所有爬取器
└── doc_crawler.py           # 统一接口（向后兼容）
```

## 如何添加新框架（以 PaddlePaddle 为例）

### 步骤 1: 创建爬取器类

创建 `component/doc_crawler_paddle.py`：

```python
# ./component/doc_crawler_paddle.py
from component.doc_crawler_base import DocCrawler
from bs4 import BeautifulSoup

PADDLE_DOC_BASE = "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/"

class PaddleDocCrawler(DocCrawler):
    """PaddlePaddle 文档爬取器"""
    
    def __init__(self):
        super().__init__("paddle")
    
    def build_doc_url(self, api_name: str) -> str:
        """构建 PaddlePaddle 文档 URL"""
        # 实现 URL 构建逻辑
        api_parts = api_name.split('.')
        # ... 根据 PaddlePaddle 文档结构构建 URL
        return f"{PADDLE_DOC_BASE}paddle/{api_name}.html"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """解析 PaddlePaddle 文档内容"""
        doc_content = {
            "api_name": api_name,
            "framework": "paddle",
            "url": url,
            "title": "",
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": [],
            "raw_html": ""
        }
        
        # 实现 PaddlePaddle 特定的解析逻辑
        # ... 根据 PaddlePaddle 文档 HTML 结构提取信息
        
        return doc_content
```

### 步骤 2: 注册到工厂

在 `component/doc_crawler_factory.py` 中添加：

```python
from component.doc_crawler_paddle import PaddleDocCrawler

# 在 _CRAWLERS 字典中添加
_CRAWLERS['paddle'] = PaddleDocCrawler
_CRAWLERS['paddlepaddle'] = PaddleDocCrawler

# 在 detect_framework 函数中添加检测逻辑
def detect_framework(api_name: str) -> Optional[str]:
    api_lower = api_name.lower()
    
    # ... 现有代码 ...
    
    elif api_lower.startswith('paddle.'):
        return 'paddle'
    
    return None
```

### 步骤 3: 测试

```bash
# 测试爬取 PaddlePaddle 文档
python3 component/doc_crawler.py paddle.nn.Conv2D --framework paddle

# 自动检测
python3 component/doc_crawler.py paddle.nn.Conv2D
```

## 接口说明

### DocCrawler 基类

所有爬取器必须继承 `DocCrawler` 并实现以下方法：

#### 必须实现的方法

1. **`build_doc_url(api_name: str) -> str`**
   - 根据 API 名称构建文档 URL
   - 不同框架的 URL 格式可能不同

2. **`parse_doc_content(soup: BeautifulSoup, api_name: str, url: str) -> Dict`**
   - 解析 HTML 内容，提取文档信息
   - 返回标准格式的字典

#### 可选重写的方法

1. **`normalize_api_name(api_name: str) -> str`**
   - 规范化 API 名称
   - 默认实现会移除常见前缀，子类可以扩展

#### 继承的方法（无需重写）

- `crawl(api_name: str)` - 主爬取流程
- `get_doc_text(api_name: str)` - 获取文本格式文档
- `load_cached_doc()` / `save_cached_doc()` - 缓存管理

## 标准文档格式

`parse_doc_content` 方法应返回以下格式的字典：

```python
{
    "api_name": "nn.Conv2d",           # API 名称
    "framework": "paddle",              # 框架名称
    "url": "https://...",               # 文档 URL
    "title": "Conv2d",                  # 标题
    "description": "二维卷积层...",     # 描述
    "parameters": [                     # 参数列表
        {
            "name": "in_channels",
            "description": "输入通道数"
        },
        ...
    ],
    "returns": "Tensor",                # 返回值说明
    "examples": [],                     # 示例代码（可选）
    "raw_html": "<html>...</html>"      # 原始 HTML（可选）
}
```

## 最佳实践

1. **保持接口一致**：确保返回的字典格式符合标准
2. **错误处理**：在 `parse_doc_content` 中处理解析失败的情况
3. **缓存利用**：利用基类的缓存机制，避免重复请求
4. **文档结构分析**：先分析目标框架的文档 HTML 结构，再实现解析逻辑
5. **测试覆盖**：为新的爬取器编写测试用例

## 示例：MindSpore 扩展

参考 `component/doc_crawler_example_mindspore.py` 查看完整的 MindSpore 爬取器示例。

## 向后兼容

现有的 `doc_crawler.py` 接口保持不变，新框架会自动通过工厂模式集成，无需修改现有代码。

