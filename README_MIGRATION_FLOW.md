# TensorFlow -> PyTorch 测试迁移完整流程

## 快速开始（一键执行）

```bash
# 执行完整流程：生成50个测试并检查
./scripts/generate_and_check.sh
```

## 手动执行步骤

### 步骤 1: 修复测试函数名（如果需要）

如果 `data/migration_candidates_fuzzy.jsonl` 中包含 `unknown_test_*` 名称，需要先修复：

```bash
python3 component/migrate_fix_names.py
```

这会从原始 TensorFlow 文件中提取真实的测试函数名并更新数据文件。

### 步骤 2: 生成迁移测试

使用 LLM 生成 50 个 PyTorch 测试文件：

```bash
python3 component/migrate_generate_tests.py \
    --limit 50 \
    --force \
    --output-dir migrated_tests
```

**参数说明：**
- `--limit 50`: 限制生成 50 个测试
- `--force`: 强制覆盖已存在的文件
- `--output-dir`: 输出目录（默认：`migrated_tests`）
- `--model`: LLM 模型（默认：`qwen-flash`）
- `--key-path`: API key 路径（默认：`aliyun.key`）

### 步骤 3: 执行测试

**方式 1: 只执行 PyTorch 测试（推荐，速度快）**

```bash
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --skip-tf \
    --out data/migrate_exec.jsonl
```

**方式 2: 执行并对比 TensorFlow 和 PyTorch（较慢）**

```bash
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --out data/migrate_comparison.jsonl
```

**方式 3: 使用现有脚本（只执行 PyTorch）**

```bash
python3 component/migrate_run_tests.py
```

### 步骤 4: 查看结果

**查看 JSON 结果：**

```bash
# 查看执行结果
cat data/migrate_exec.jsonl | python3 -m json.tool | head -50

# 查看对比结果
cat data/migrate_comparison.jsonl | python3 -m json.tool | head -50
```

**快速统计：**

```bash
python3 -c "
import json
results = [json.loads(l) for l in open('data/migrate_exec.jsonl')]
total = len(results)
passed = sum(1 for r in results if r.get('status') == 'pass')
print(f'通过: {passed}/{total} ({passed/total*100:.1f}%)')
"
```

**查看详细日志：**

```bash
# 查看特定测试的详细日志
cat data/migrate_logs/unknown_test_199.log
```

### 步骤 5: 生成 HTML 报告

```bash
python3 component/migrate_report.py \
    --comparison data/migrate_comparison.jsonl \
    --output reports/migration_comparison.html
```

然后在浏览器中打开：

```bash
open reports/migration_comparison.html
```

## 完整命令示例

```bash
# 1. 修复测试名称（如果需要）
python3 component/migrate_fix_names.py

# 2. 生成50个测试
python3 component/migrate_generate_tests.py --limit 50 --force

# 3. 执行测试
python3 component/migrate_compare.py --migrated-dir migrated_tests --limit 50 --skip-tf

# 4. 生成报告
python3 component/migrate_report.py

# 5. 查看报告
open reports/migration_comparison.html
```

## 文件说明

### 输入文件
- `data/migration_candidates_fuzzy.jsonl`: 可迁移的测试候选列表
- `data/tests_tf.mapped.jsonl`: TensorFlow 测试元数据
- `data/component_pairs.jsonl`: TensorFlow -> PyTorch API 映射对

### 输出文件
- `migrated_tests/*.py`: 生成的 PyTorch 测试文件
- `data/migrate_exec.jsonl`: PyTorch 测试执行结果
- `data/migrate_comparison.jsonl`: TensorFlow vs PyTorch 对比结果
- `data/migrate_logs/*.log`: 每个测试的详细执行日志
- `reports/migration_comparison.html`: HTML 格式的对比报告

## 常见问题

### Q: 为什么有些测试显示 `unknown_test_*`？

A: 这是因为原始数据中没有提取到真实的测试函数名。运行 `python3 component/migrate_fix_names.py` 可以修复。

### Q: TensorFlow 测试执行失败怎么办？

A: TensorFlow 测试执行可能因为环境、依赖等问题失败。建议使用 `--skip-tf` 参数只执行 PyTorch 测试。

### Q: 如何生成更多测试？

A: 修改 `--limit` 参数，或设置为 `-1` 生成全部测试。

### Q: 如何查看失败的测试详情？

A: 查看 `data/migrate_logs/` 目录下的对应日志文件，或打开 HTML 报告查看详细信息。

## 性能优化

- 使用 `--skip-tf` 可以大幅加快执行速度
- 使用 `--limit` 限制测试数量进行快速验证
- 批量处理时可以使用 `--force` 覆盖已存在的文件

