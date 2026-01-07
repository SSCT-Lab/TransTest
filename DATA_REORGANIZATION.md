# Data 文件夹重组说明

## 概述

`data/` 文件夹已按功能分类重组，提高了文件的可读性和可维护性。

## 新的目录结构

```
data/
├── parsing/          # 文件解析相关
│   ├── files_tf.jsonl
│   ├── files_pt.jsonl
│   ├── norm_tf.jsonl
│   ├── norm_pt.jsonl
│   ├── tests_tf.parsed.jsonl
│   └── tests_pt.parsed.jsonl
├── mapping/          # API映射相关
│   ├── api_map.json
│   ├── tests_tf.mapped.jsonl
│   ├── tests_pt.mapped.jsonl
│   └── tf_test_api_usage.jsonl
├── components/       # 组件匹配相关
│   ├── tf_components.jsonl
│   ├── pt_components.jsonl
│   ├── tf_vectors.npy
│   ├── pt_vectors.npy
│   ├── component_candidates.jsonl
│   ├── component_candidates_filtered.jsonl
│   └── component_pairs.jsonl
├── migration/        # 迁移候选和计划
│   ├── migration_candidates.jsonl
│   ├── migration_candidates_fuzzy.jsonl
│   ├── migration_candidates_fuzzy_fixed.jsonl
│   └── migration_plan.jsonl
├── results/          # 测试执行结果
│   ├── migrate_exec.jsonl
│   ├── migrate_comparison.jsonl
│   └── tf_test_results.jsonl
├── logs/             # 执行日志
│   └── (迁移测试执行日志)
├── analysis/         # 分析和对比报告
│   ├── compare_rule_vs_llm_*.csv
│   ├── compare_rule_vs_llm_*.html
│   ├── diff_types_*.jsonl
│   ├── diff_types_*.csv
│   ├── recall_pairs_*.jsonl
│   └── high_agreement_cases.jsonl
├── fuzz/             # 模糊测试相关
│   ├── fuzz_report.jsonl
│   ├── fuzz_summary.csv
│   └── seed_patterns.jsonl
└── reports/          # 报告和案例
    ├── cases_md/
    ├── epoch1/
    └── migrated_tests_v2/
```

## 路径变更

### 解析相关
- `data/files_*.jsonl` → `data/parsing/files_*.jsonl`
- `data/norm_*.jsonl` → `data/parsing/norm_*.jsonl`
- `data/tests_*.parsed.jsonl` → `data/parsing/tests_*.parsed.jsonl`

### 映射相关
- `data/api_map.json` → `data/mapping/api_map.json`
- `data/tests_*.mapped.jsonl` → `data/mapping/tests_*.mapped.jsonl`
- `data/tf_test_api_usage.jsonl` → `data/mapping/tf_test_api_usage.jsonl`

### 组件相关
- `data/*_components.jsonl` → `data/components/*_components.jsonl`
- `data/*_vectors.npy` → `data/components/*_vectors.npy`
- `data/component_*.jsonl` → `data/components/component_*.jsonl`

### 迁移相关
- `data/migration_*.jsonl` → `data/migration/migration_*.jsonl`
- `data/migration_plan.jsonl` → `data/migration/migration_plan.jsonl`

### 结果相关
- `data/migrate_exec.jsonl` → `data/results/migrate_exec.jsonl`
- `data/migrate_comparison.jsonl` → `data/results/migrate_comparison.jsonl`
- `data/tf_test_results.jsonl` → `data/results/tf_test_results.jsonl`

### 日志相关
- `data/migrate_logs/` → `data/logs/`

## 更新的文件

以下文件中的路径已更新：
- `main.py`
- `core/mapping/api_mapping.py`
- `component/migration/*.py`
- `scripts/*.sh`

## 向后兼容性

所有脚本和代码已更新为新路径。如果遇到路径错误，请检查是否使用了旧的路径引用。

## 注意事项

1. 所有脚本路径已更新，可以直接使用
2. 如果遇到文件不存在错误，请检查路径是否正确
3. 建议使用新的路径结构以保持代码清晰

