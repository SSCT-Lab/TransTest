# TensorFlow 测试结果收集和使用指南

## 概述

为了避免每次对比时都执行 TensorFlow 测试（可能因为环境问题失败），可以先批量运行 TensorFlow 测试并保存结果到静态文件，后续直接读取这些结果。

## 使用流程

### 步骤 1: 收集 TensorFlow 测试结果

```bash
# 收集所有候选测试的 TensorFlow 执行结果
python3 component/migrate_collect_tf_results.py \
    --candidates data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl \
    --limit 50

# 或者收集全部
python3 component/migrate_collect_tf_results.py \
    --candidates data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl
```

**参数说明：**
- `--candidates`: 候选测试文件（默认：`data/migration_candidates_fuzzy.jsonl`）
- `--output`: 输出结果文件（默认：`data/tf_test_results.jsonl`）
- `--limit`: 限制测试数量（默认：-1，全部）
- `--tf-root`: TensorFlow 源码根目录（默认：`framework/tensorflow-master`）

### 步骤 2: 使用静态结果进行对比

```bash
# 使用静态结果文件进行对比（不会执行 TensorFlow 测试）
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --tf-results data/tf_test_results.jsonl
```

**注意：**
- 如果 `--tf-results` 指定的文件存在，会自动使用缓存的结果
- 如果缓存中找不到某个测试的结果，会尝试执行测试
- 使用 `--skip-tf` 可以完全跳过 TensorFlow 测试

### 步骤 3: 生成报告

```bash
python3 component/migrate_report.py \
    --comparison data/migrate_comparison.jsonl \
    --output reports/migration_comparison.html
```

## 结果文件格式

`data/tf_test_results.jsonl` 格式：

```json
{
  "file": "framework/tensorflow-master/tensorflow/tools/compatibility/testdata/test_file_v0_11.py",
  "test_name": "testArgRenames",
  "actual_test_name": "testArgRenames",
  "tf_file": "framework/tensorflow-master/tensorflow/tools/compatibility/testdata/test_file_v0_11.py",
  "tf_result": {
    "status": "pass",
    "stdout": "...",
    "stderr": "",
    "returncode": 0,
    "actual_test_name": "testArgRenames"
  },
  "timestamp": "..."
}
```

## 优势

1. **避免重复执行**：一次收集，多次使用
2. **避免环境问题**：可以在环境正常时收集，后续即使环境有问题也能使用
3. **提高速度**：读取静态文件比执行测试快得多
4. **可追溯性**：保存了完整的执行结果和日志

## 更新结果

如果需要更新 TensorFlow 测试结果：

```bash
# 重新收集（会覆盖旧文件）
python3 component/migrate_collect_tf_results.py --output data/tf_test_results.jsonl
```

## 完整工作流示例

```bash
# 1. 收集 TensorFlow 测试结果
python3 component/migrate_collect_tf_results.py --limit 50

# 2. 生成 PyTorch 测试
python3 component/migrate_generate_tests.py --limit 50 --force

# 3. 对比测试（使用静态 TF 结果）
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --tf-results data/tf_test_results.jsonl

# 4. 生成报告
python3 component/migrate_report.py
```

