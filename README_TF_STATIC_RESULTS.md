# TensorFlow 测试静态结果收集和使用指南

## 工作流程

### 1. 收集阶段：执行所有 TensorFlow 测试并保存结果

```bash
# 收集所有候选测试的 TensorFlow 执行结果
python3 component/migrate_collect_tf_results.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl

# 或者使用脚本
./scripts/collect_all_tf_tests.sh
```

**说明：**
- 对每个 TensorFlow 测试执行一次
- 记录执行结果：`status` (pass/fail/error/timeout), `stdout`, `stderr`, `returncode`
- 保存到静态文件 `data/tf_test_results.jsonl`
- 每个测试一行 JSON，格式如下：

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
  "apis_used": ["tf.acos", "tf.rsqrt"],
  "timestamp": "..."
}
```

### 2. 迁移阶段：生成 PyTorch 测试

```bash
# 生成迁移后的 PyTorch 测试
python3 component/migrate_generate_tests.py --limit 50 --force
```

### 3. 对比阶段：查表对照（不执行 TensorFlow 测试）

```bash
# 使用静态结果文件进行对比（直接查表，不执行 TensorFlow 测试）
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --tf-results data/tf_test_results.jsonl
```

**说明：**
- 自动从静态文件中查找对应的 TensorFlow 测试结果
- 支持多种路径格式匹配（完整路径、去掉前缀、只文件名）
- 如果静态文件中没有找到，会尝试执行测试（但建议先收集完整结果）

### 4. 生成报告

```bash
python3 component/migrate_report.py \
    --comparison data/migrate_comparison.jsonl \
    --output reports/migration_comparison.html
```

## 完整工作流示例

```bash
# 步骤 1: 收集所有 TensorFlow 测试结果（一次性，可能需要较长时间）
python3 component/migrate_collect_tf_results.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl

# 步骤 2: 生成 PyTorch 测试
python3 component/migrate_generate_tests.py --limit 50 --force

# 步骤 3: 对比测试（使用静态结果，快速查表）
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --tf-results data/tf_test_results.jsonl

# 步骤 4: 生成报告
python3 component/migrate_report.py
```

## 优势

1. **避免重复执行**：TensorFlow 测试只执行一次，结果保存后多次使用
2. **避免环境问题**：可以在环境正常时收集，后续即使环境有问题也能查表
3. **提高速度**：查表比执行测试快得多
4. **可追溯性**：保存了完整的执行结果、日志和输出
5. **离线使用**：静态文件可以分享、备份、版本控制

## 匹配逻辑

对比时会尝试多种键格式匹配：
1. `(完整文件路径, 测试名)` - 例如：`("framework/tensorflow-master/.../test.py", "testName")`
2. `(去掉前缀的路径, 测试名)` - 例如：`("tensorflow/.../test.py", "testName")`
3. `(只文件名, 测试名)` - 例如：`("test.py", "testName")`

这样可以处理不同的路径格式，提高匹配成功率。

## 更新结果

如果需要更新 TensorFlow 测试结果：

```bash
# 重新收集（会覆盖旧文件）
python3 component/migrate_collect_tf_results.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl
```

## 注意事项

1. **收集时间**：收集所有测试结果可能需要较长时间，建议在后台运行
2. **环境要求**：收集时需要 TensorFlow 环境正常
3. **文件大小**：结果文件可能较大（包含完整的 stdout/stderr）
4. **匹配准确性**：确保测试名称和文件路径一致，否则可能匹配失败

