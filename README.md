# TensorFlow → PyTorch 测试迁移系统

## 概述

这是一个自动化系统，用于将 TensorFlow 测试迁移到 PyTorch。系统通过语义匹配、LLM 分析和代码生成，实现从 TensorFlow 测试到 PyTorch 测试的自动化迁移。

## 系统架构

### 目录结构

```
TransTest/
├── core/                    # 核心功能模块
│   ├── discover.py         # 发现测试文件
│   ├── normalize.py        # 规范化文件路径
│   ├── parse_py.py         # 解析 Python 测试文件，提取 API 调用
│   ├── api_mapping.py      # 构建 API 映射关系
│   ├── semantic.py         # 语义相似度计算
│   └── ...
├── component/               # 组件和工具脚本
│   ├── migrate_identify_fuzzy.py    # 识别可迁移测试
│   ├── migrate_generate_tests.py   # 生成迁移后的测试
│   ├── migrate_compare.py           # 对比测试结果
│   ├── migrate_collect_tf_results.py # 收集 TensorFlow 测试结果
│   ├── migrate_report.py            # 生成迁移报告
│   └── ...
├── data/                   # 数据文件（中间结果）
│   ├── files_tf.jsonl     # TensorFlow 测试文件列表
│   ├── tests_tf.parsed.jsonl  # 解析后的 TensorFlow 测试
│   ├── tests_tf.mapped.jsonl  # API 映射后的 TensorFlow 测试
│   ├── migration_candidates_fuzzy.jsonl  # 可迁移测试候选
│   └── ...
├── migrated_tests/         # 生成的迁移测试文件
└── deprecated/             # 过时文件（已废弃）

```

## 完整工作流程

### 阶段 1: 数据准备和解析

#### 1.1 发现和解析测试文件

```bash
# 执行主流程：发现、规范化、解析测试文件
python3 main.py
```

**输出文件：**
- `data/files_tf.jsonl` - TensorFlow 测试文件列表
- `data/files_pt.jsonl` - PyTorch 测试文件列表
- `data/norm_tf.jsonl` - 规范化后的 TensorFlow 文件路径
- `data/norm_pt.jsonl` - 规范化后的 PyTorch 文件路径
- `data/tests_tf.parsed.jsonl` - 解析后的 TensorFlow 测试（包含 API 调用）
- `data/tests_pt.parsed.jsonl` - 解析后的 PyTorch 测试（包含 API 调用）

**核心模块：**
- `core/discover.py` - 发现测试文件
- `core/normalize.py` - 规范化文件路径
- `core/parse_py.py` - 解析 Python 文件，提取：
  - 测试函数名
  - TensorFlow/PyTorch API 调用
  - 断言语句

#### 1.2 构建 API 映射

```bash
# 构建 TensorFlow 到 PyTorch 的 API 映射
python3 core/api_mapping.py
```

**输出文件：**
- `data/tests_tf.mapped.jsonl` - 包含 API 映射信息的 TensorFlow 测试
- `data/tests_pt.mapped.jsonl` - 包含 API 映射信息的 PyTorch 测试

**功能：**
- 分析 TensorFlow 和 PyTorch 测试中使用的 API
- 建立 API 使用频率统计
- 为后续匹配提供基础数据

### 阶段 2: 组件匹配（可选，用于 API 映射）

如果需要更精确的 API 映射，可以执行组件匹配流程：

#### 2.1 扫描组件

```bash
# 扫描 TensorFlow 和 PyTorch 源码中的函数/类定义
python3 component/scan_components.py
```

**输出文件：**
- `data/tf_components.jsonl` - TensorFlow 组件列表
- `data/pt_components.jsonl` - PyTorch 组件列表

#### 2.2 生成 Embedding

```bash
# 使用本地模型生成 embedding
python3 component/embed_components.py

# 或使用在线 API（需要 API key）
python3 component/embed_components_ol.py
```

**输出文件：**
- `data/tf_vectors.npy` - TensorFlow 组件的 embedding 向量
- `data/pt_vectors.npy` - PyTorch 组件的 embedding 向量

#### 2.3 检索候选对

```bash
# 使用 embedding 相似度检索候选 API 对
python3 component/gen_candidates.py
```

**输出文件：**
- `data/component_candidates.jsonl` - 候选 API 对

#### 2.4 LLM 匹配

```bash
# 使用 LLM 分析并生成最终 API 映射
python3 component/match_components_llm.py
```

**输出文件：**
- `data/component_pairs.jsonl` - 最终 API 映射对

### 阶段 3: 识别可迁移测试

```bash
# 识别可以使用已知 API 映射进行迁移的测试
python3 component/migrate_identify_fuzzy.py
```

**输入文件：**
- `data/tests_tf.mapped.jsonl` - TensorFlow 测试（已映射 API）
- `data/component_pairs.jsonl` - API 映射对（如果执行了组件匹配）

**输出文件：**
- `data/migration_candidates_fuzzy.jsonl` - 可迁移测试候选列表

**功能：**
- 分析每个 TensorFlow 测试使用的 API
- 检查是否有对应的 PyTorch API 映射
- 计算匹配度，筛选可迁移的测试

### 阶段 4: 修复测试名称（如果需要）

如果候选列表中包含 `unknown_test_*` 名称，需要先修复：

```bash
python3 component/migrate_fix_names.py
```

**功能：**
- 从原始 TensorFlow 文件中提取真实的测试函数名
- 更新 `data/migration_candidates_fuzzy.jsonl` 文件

### 阶段 5: 生成迁移测试

```bash
# 生成包含 TensorFlow 原始测试和 PyTorch 迁移测试的文件
python3 component/migrate_generate_tests.py \
    --limit 50 \
    --force \
    --output-dir migrated_tests
```

**参数说明：**
- `--limit 50` - 限制生成 50 个测试（-1 表示全部）
- `--force` - 强制覆盖已存在的文件
- `--output-dir` - 输出目录（默认：`migrated_tests`）
- `--model` - LLM 模型名称（默认：`qwen-flash`）
- `--key-path` - API key 文件路径（默认：`aliyun.key`）
- `--tf-root` - TensorFlow 源码根目录（默认：`framework/tensorflow-master`）

**输出：**
- `migrated_tests/*.py` - 生成的迁移测试文件

**生成的文件结构：**
每个迁移测试文件包含：
1. **TensorFlow 原始测试逻辑** - 提取并转换为独立可执行的函数
2. **PyTorch 迁移后的测试** - 使用 LLM 生成的迁移代码
3. **对比测试函数** - 自动执行两个测试并对比结果

**示例：**
```python
import torch
import pytest
import tensorflow as tf
import numpy as np

# Helper functions
def assertAllClose(a, b, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

# ===== TensorFlow Original Test =====
def testRenames():
    """Original TensorFlow test logic"""
    try:
        assertAllClose(1.04719755, tf.acos(0.5))
        assertAllClose(0.5, tf.rsqrt(4.0))
        print("TF: PASS")
    except Exception as e:
        print(f"TF: FAIL - {e}")

# ===== PyTorch Migrated Test =====
def test_testRenames_pt():
    # PyTorch migrated code
    assert torch.allclose(torch.acos(torch.tensor(0.5)), torch.tensor(1.04719755))
    # ...

# ===== Comparison Test =====
def test_testRenames_compare():
    """Compare TensorFlow and PyTorch test results"""
    # 自动执行两个测试并对比
    ...
```

### 阶段 6: 执行和对比测试

#### 6.1 收集 TensorFlow 测试结果（可选）

如果需要静态对比，可以先收集 TensorFlow 测试结果：

```bash
# 执行所有 TensorFlow 测试并保存结果
python3 component/migrate_collect_tf_results.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl \
    --limit 50
```

**输出文件：**
- `data/tf_test_results.jsonl` - TensorFlow 测试执行结果

#### 6.2 执行和对比测试

**方式 1: 只执行 PyTorch 测试（推荐，速度快）**

```bash
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --skip-tf \
    --out data/migrate_exec.jsonl
```

**方式 2: 执行并对比 TensorFlow 和 PyTorch**

```bash
# 使用静态结果文件（推荐）
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --tf-results data/tf_test_results.jsonl \
    --out data/migrate_comparison.jsonl

# 或动态执行 TensorFlow 测试（较慢）
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --out data/migrate_comparison.jsonl
```

**方式 3: 使用内置对比函数**

```bash
# 执行迁移文件中的对比函数
pytest migrated_tests/ -k "_compare" -v
```

**输出文件：**
- `data/migrate_exec.jsonl` - PyTorch 测试执行结果
- `data/migrate_comparison.jsonl` - TensorFlow 和 PyTorch 对比结果

### 阶段 7: 生成报告

```bash
# 生成迁移报告
python3 component/migrate_report.py
```

**输出文件：**
- `reports/migration_report.csv` - 迁移报告（CSV 格式）
- `reports/migration_comparison.html` - 对比报告（HTML 格式）

## 快速开始

### 一键执行完整流程

```bash
# 执行完整流程：生成50个测试并检查
./scripts/generate_and_check.sh
```

### 手动执行步骤

1. **准备数据**（如果还没有）：
   ```bash
   python3 main.py
   python3 core/api_mapping.py
   ```

2. **识别可迁移测试**：
   ```bash
   python3 component/migrate_identify_fuzzy.py
   ```

3. **修复测试名称**（如果需要）：
   ```bash
   python3 component/migrate_fix_names.py
   ```

4. **生成迁移测试**：
   ```bash
   python3 component/migrate_generate_tests.py --limit 50 --force
   ```

5. **执行和对比**：
   ```bash
   python3 component/migrate_compare.py \
       --migrated-dir migrated_tests \
       --limit 50 \
       --skip-tf \
       --out data/migrate_exec.jsonl
   ```

6. **生成报告**：
   ```bash
   python3 component/migrate_report.py
   ```

## 核心模块说明

### core/ 目录

- **discover.py** - 发现测试文件
  - 扫描指定目录，查找测试文件
  - 支持 glob 模式匹配

- **normalize.py** - 规范化文件路径
  - 统一文件路径格式
  - 处理相对路径和绝对路径

- **parse_py.py** - 解析 Python 测试文件
  - 提取测试函数定义
  - 识别 TensorFlow/PyTorch API 调用
  - 提取断言语句
  - 支持类方法和独立函数

- **api_mapping.py** - 构建 API 映射
  - 分析 API 使用频率
  - 建立 API 映射关系

- **semantic.py** - 语义相似度计算
  - 使用 embedding 模型计算相似度

### component/ 目录

- **migrate_identify_fuzzy.py** - 识别可迁移测试
  - 基于 API 映射匹配测试
  - 计算匹配度

- **migrate_generate_tests.py** - 生成迁移测试
  - 提取 TensorFlow 测试代码
  - 使用 LLM 生成 PyTorch 迁移代码
  - 生成包含 TF 和 PT 测试的完整文件

- **migrate_compare.py** - 对比测试结果
  - 执行 PyTorch 测试
  - 可选执行 TensorFlow 测试
  - 对比结果并生成报告

- **migrate_collect_tf_results.py** - 收集 TensorFlow 测试结果
  - 批量执行 TensorFlow 测试
  - 保存静态结果供后续对比

- **migrate_report.py** - 生成迁移报告
  - 统计迁移成功率
  - 生成 HTML 和 CSV 报告

## 数据文件说明

### 输入文件

- `config.yaml` - 配置文件，定义 TensorFlow 和 PyTorch 源码路径
- `data/migration_candidates_fuzzy.jsonl` - 可迁移测试候选列表

### 中间文件

- `data/files_*.jsonl` - 测试文件列表
- `data/norm_*.jsonl` - 规范化后的文件路径
- `data/tests_*.parsed.jsonl` - 解析后的测试
- `data/tests_*.mapped.jsonl` - API 映射后的测试
- `data/component_*.jsonl` - 组件和映射数据

### 输出文件

- `migrated_tests/*.py` - 生成的迁移测试文件
- `data/migrate_exec.jsonl` - 测试执行结果
- `data/migrate_comparison.jsonl` - 对比结果
- `reports/*` - 迁移报告

## 配置说明

### config.yaml

```yaml
repos:
  tf: "framework/tensorflow-master"
  pt: "framework/pytorch-main"

test_roots:
  tf: ["tensorflow/python/**"]
  pt: ["test/**"]

include_globs:
  - "**/*test*.py"

exclude_globs:
  - "**/__pycache__/**"
  - "**/*_test_*.py"
```

### API Key 配置

LLM 功能需要 API key，默认从 `aliyun.key` 文件读取：

```bash
echo "your-api-key" > aliyun.key
```

## 常见问题

### 1. 如何只生成特定数量的测试？

使用 `--limit` 参数：
```bash
python3 component/migrate_generate_tests.py --limit 10 --force
```

### 2. 如何跳过 TensorFlow 测试执行？

使用 `--skip-tf` 参数：
```bash
python3 component/migrate_compare.py --skip-tf
```

### 3. 如何使用静态 TensorFlow 结果？

先收集结果，然后使用 `--tf-results` 参数：
```bash
python3 component/migrate_collect_tf_results.py --output data/tf_test_results.jsonl
python3 component/migrate_compare.py --tf-results data/tf_test_results.jsonl
```

### 4. 如何查看迁移统计？

```bash
python3 -c "
import json
results = [json.loads(l) for l in open('data/migrate_exec.jsonl')]
total = len(results)
passed = sum(1 for r in results if r.get('pt_result', {}).get('status') == 'pass')
print(f'Total: {total}, Passed: {passed}, Pass Rate: {passed/total*100:.1f}%')
"
```

## 相关文档

- `README_MIGRATE_WITH_TF.md` - 迁移测试包含 TensorFlow 逻辑的详细说明
- `README_MIGRATION_FLOW.md` - 迁移流程详细说明
- `README_TF_STATIC_RESULTS.md` - TensorFlow 静态结果收集指南
- `deprecated/README.md` - 过时文件说明

## 注意事项

1. **API Key**: LLM 功能需要配置 API key
2. **依赖环境**: 需要安装 TensorFlow 和 PyTorch
3. **测试框架**: 生成的测试使用 pytest 框架
4. **文件大小**: 大量测试可能生成较大的数据文件

## 贡献

欢迎提交 Issue 和 Pull Request！

