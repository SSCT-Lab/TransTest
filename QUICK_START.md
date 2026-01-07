# 快速执行命令

## 当前状态
- ✓ 数据文件已准备
- ✓ 迁移候选：319 个
- ✓ 已生成：37 个迁移测试

## 执行命令

### 方式 1: 一键执行（推荐）

```bash
# 生成50个测试并执行完整流程
./scripts/run_migration.sh 50
```

### 方式 2: 分步执行

```bash
# 步骤 1: 生成50个迁移测试
python3 component/migrate_generate_tests.py --limit 50 --force

# 步骤 2: 执行 PyTorch 测试
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --skip-tf \
    --out data/results/migrate_exec.jsonl

# 步骤 3: 查看统计
python3 -c "
import json
results = [json.loads(l) for l in open('data/results/migrate_exec.jsonl')]
total = len(results)
passed = sum(1 for r in results if (r.get('pt_result') or {}).get('status') == 'pass')
failed = sum(1 for r in results if (r.get('pt_result') or {}).get('status') == 'fail')
error = sum(1 for r in results if (r.get('pt_result') or {}).get('status') in ['error', 'timeout', 'not_found'])
print(f'总测试数: {total}')
print(f'通过: {passed} ({passed/total*100:.1f}%)')
print(f'失败: {failed} ({failed/total*100:.1f}%)')
print(f'错误: {error} ({error/total*100:.1f}%)')
"

# 步骤 4: 生成报告
python3 component/migrate_report.py
```

### 方式 3: 只生成测试（不执行）

```bash
# 生成100个测试
python3 component/migrate_generate_tests.py --limit 100 --force
```

### 方式 4: 只执行已有测试

```bash
# 执行所有已生成的测试
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --skip-tf \
    --out data/results/migrate_exec.jsonl
```

## 查看结果

```bash
# 查看执行结果
cat data/results/migrate_exec.jsonl | python3 -m json.tool | head -50

# 查看报告
open reports/migration_comparison.html  # macOS
# 或
xdg-open reports/migration_comparison.html  # Linux
```

