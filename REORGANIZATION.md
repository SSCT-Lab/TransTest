# 文件重组说明

## 概述

本次重组将 `component/` 和 `core/` 目录下的文件按功能分类组织到子目录中，提高了代码的可读性和可维护性。

## 新的目录结构

### component/ 目录

```
component/
├── migration/          # 测试迁移相关（7个文件）
│   ├── migrate_generate_tests.py
│   ├── migrate_compare.py
│   ├── migrate_collect_tf_results.py
│   ├── migrate_report.py
│   ├── migrate_fix_names.py
│   ├── migrate_identify_fuzzy.py
│   └── migrate_run_tests.py
├── matching/          # API/组件匹配相关（7个文件）
│   ├── scan_components.py
│   ├── scan_api_usage.py
│   ├── embed_components.py
│   ├── embed_components_ol.py
│   ├── gen_candidates.py
│   ├── match_components_llm.py
│   └── identify_components_llm.py
├── doc/               # 文档爬取和分析（8个文件）
│   ├── doc_crawler_base.py
│   ├── doc_crawler_factory.py
│   ├── doc_crawler_pytorch.py
│   ├── doc_crawler_tensorflow.py
│   ├── doc_crawler.py
│   ├── doc_analyzer.py
│   └── examples/      # 扩展示例
│       ├── doc_crawler_example_paddle.py
│       └── doc_crawler_example_mindspore.py
├── fuzz/              # 模糊测试（3个文件）
│   ├── fuzz_generate_tests_numeric.py
│   ├── fuzz_run_numeric.py
│   └── fuzz_seed_extract.py
└── utils/             # 工具函数（1个文件）
    └── llm_utils.py
```

### core/ 目录

```
core/
├── parsing/           # 文件解析相关（3个文件）
│   ├── discover.py
│   ├── normalize.py
│   └── parse_py.py
├── mapping/           # API映射（1个文件）
│   └── api_mapping.py
├── semantic/          # 语义分析（3个文件）
│   ├── semantic.py
│   ├── semantic_llm.py
│   └── vectorize.py
├── analysis/          # 差异分析和对比（4个文件）
│   ├── analyze_differences.py
│   ├── analyze_differences_llm.py
│   ├── compare_rule_vs_llm.py
│   └── compare_rule_vs_llm_report.py
├── fuzz/              # 模糊测试（3个文件）
│   ├── fuzz_pairs.py
│   ├── fuzz_report.py
│   └── conv2d_fuzz.py
└── utils/             # 其他工具（4个文件）
    ├── struct_features.py
    ├── summary.py
    ├── export_case_markdown.py
    └── pick_high_agreement_cases.py
```

## 已删除/移动的文件

### 删除的文件
- `component/test.py` - 临时测试文件
- `core/test.py` - 临时测试文件

### 移动到 deprecated/ 的文件
- `component/migrate_identify.py` - 已被 `migrate_identify_fuzzy.py` 替代
- `component/report_migration.py` - 功能与 `migrate_report.py` 不同，可能不再使用

## 向后兼容性

### core/ 模块
`core/__init__.py` 提供了向后兼容的导入：
```python
from core import discover_test_files, normalize_file, parse_test_file
```

这些导入会自动重定向到新的子模块：
- `core.parsing.discover`
- `core.parsing.normalize`
- `core.parsing.parse_py`

### component/ 模块
`component/doc/doc_crawler.py` 提供了向后兼容的统一接口。

## 更新的脚本

以下脚本中的路径已更新：
- `scripts/generate_and_check.sh`
- `scripts/run_migration.sh`
- `scripts/analyze_test_error.sh`
- `scripts/collect_all_tf_tests.sh`

## 导入路径变更

### 旧路径 → 新路径

**component/**
- `component.migrate_*` → `component.migration.migrate_*`
- `component.scan_*` → `component.matching.scan_*`
- `component.embed_*` → `component.matching.embed_*`
- `component.gen_candidates` → `component.matching.gen_candidates`
- `component.match_components_llm` → `component.matching.match_components_llm`
- `component.identify_components_llm` → `component.matching.identify_components_llm`
- `component.doc_*` → `component.doc.doc_*`
- `component.fuzz_*` → `component.fuzz.fuzz_*`
- `component.llm_utils` → `component.utils.llm_utils`

**core/**
- `core.discover` → `core.parsing.discover`
- `core.normalize` → `core.parsing.normalize`
- `core.parse_py` → `core.parsing.parse_py`
- `core.api_mapping` → `core.mapping.api_mapping`
- `core.semantic*` → `core.semantic.semantic*`
- `core.vectorize` → `core.semantic.vectorize`
- `core.analyze_*` → `core.analysis.analyze_*`
- `core.compare_*` → `core.analysis.compare_*`
- `core.fuzz_*` → `core.fuzz.fuzz_*`
- `core.conv2d_fuzz` → `core.fuzz.conv2d_fuzz`

## 验证

运行以下命令验证导入是否正常：
```bash
python3 -c "from core import discover_test_files; print('✓ core 导入正常')"
python3 -c "from component.doc.doc_crawler_factory import get_doc_content; print('✓ component.doc 导入正常')"
```

## 统计

- **component/** 目录：从 33 个文件重组为 5 个子目录，共 26 个文件
- **core/** 目录：从 20 个文件重组为 6 个子目录，共 18 个文件
- **删除**：2 个临时测试文件
- **移动到 deprecated/**：2 个过时文件

## 注意事项

1. 所有脚本路径已更新，可以直接使用
2. `core/__init__.py` 提供向后兼容，现有代码无需修改
3. 如果遇到导入错误，请检查是否使用了旧的导入路径
4. 建议使用新的导入路径以保持代码清晰

