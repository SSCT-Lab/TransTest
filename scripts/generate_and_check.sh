#!/bin/bash
# 完整的生成50个测试并检查的流程脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "TensorFlow -> PyTorch 测试迁移完整流程"
echo "=========================================="
echo ""

# 配置
LIMIT=50
MIGRATED_DIR="migrated_tests"
DATA_DIR="data"
REPORTS_DIR="reports"

# 创建必要的目录
mkdir -p $MIGRATED_DIR
mkdir -p $DATA_DIR
mkdir -p $REPORTS_DIR

echo "[STEP 1/5] 修复测试函数名（如果需要）"
echo "----------------------------------------"
if [ ! -f "$DATA_DIR/migration/migration_candidates_fuzzy.jsonl" ]; then
    echo "  [SKIP] migration_candidates_fuzzy.jsonl 不存在，跳过修复"
else
    echo "  检查是否需要修复..."
    python3 -c "
import json
from pathlib import Path
count = sum(1 for line in open('$DATA_DIR/migration/migration_candidates_fuzzy.jsonl') 
            if json.loads(line).get('name', '').startswith('unknown_test_'))
if count > 0:
    print(f'  发现 {count} 个 unknown_test_* 名称，需要修复')
    exit(1)
else:
    print('  所有测试名称都是真实的，无需修复')
    exit(0)
" && echo "  [OK] 测试名称正常" || {
    echo "  [FIX] 开始修复测试名称..."
    python3 component/migration/migrate_fix_names.py
    echo "  [OK] 修复完成"
}
fi
echo ""

echo "[STEP 2/5] 生成50个迁移测试"
echo "----------------------------------------"
echo "  生成 $LIMIT 个 PyTorch 测试文件..."
python3 component/migration/migrate_generate_tests.py \
    --limit $LIMIT \
    --force \
    --output-dir $MIGRATED_DIR

GENERATED_COUNT=$(find $MIGRATED_DIR -name "*.py" -type f | wc -l | tr -d ' ')
echo "  [OK] 已生成 $GENERATED_COUNT 个测试文件"
echo ""

echo "[STEP 3/5] 执行 PyTorch 测试"
echo "----------------------------------------"
echo "  运行迁移后的 PyTorch 测试..."
python3 component/migration/migrate_compare.py \
    --migrated-dir $MIGRATED_DIR \
    --limit $LIMIT \
    --skip-tf \
    --out $DATA_DIR/results/migrate_exec.jsonl

echo "  [OK] 测试执行完成"
echo ""

echo "[STEP 4/5] 统计执行结果"
echo "----------------------------------------"
python3 -c "
import json
from pathlib import Path

results = []
if Path('$DATA_DIR/results/migrate_exec.jsonl').exists():
    results = [json.loads(l) for l in open('$DATA_DIR/results/migrate_exec.jsonl')]
elif Path('$DATA_DIR/results/migrate_comparison.jsonl').exists():
    results = [json.loads(l) for l in open('$DATA_DIR/results/migrate_comparison.jsonl')]
    results = [{'status': r.get('pt_result', {}).get('status', 'unknown')} for r in results]

if results:
    total = len(results)
    passed = sum(1 for r in results if r.get('status') == 'pass')
    failed = sum(1 for r in results if r.get('status') == 'fail')
    error = sum(1 for r in results if r.get('status') in ['error', 'timeout', 'not_found'])
    
    print(f'  总测试数: {total}')
    print(f'  通过: {passed} ({passed/total*100:.1f}%)')
    print(f'  失败: {failed} ({failed/total*100:.1f}%)')
    print(f'  错误/超时: {error} ({error/total*100:.1f}%)')
else:
    print('  [WARN] 未找到执行结果')
"
echo ""

echo "[STEP 5/5] 生成 HTML 报告"
echo "----------------------------------------"
if [ -f "$DATA_DIR/results/migrate_comparison.jsonl" ]; then
    python3 component/migration/migrate_report.py \
        --comparison $DATA_DIR/results/migrate_comparison.jsonl \
        --output $REPORTS_DIR/migration_comparison.html
    echo "  [OK] 报告已生成: $REPORTS_DIR/migration_comparison.html"
elif [ -f "$DATA_DIR/results/migrate_exec.jsonl" ]; then
    echo "  [INFO] 只有执行结果，无法生成对比报告"
    echo "  如需对比报告，请运行:"
    echo "    python3 component/migration/migrate_compare.py --migrated-dir $MIGRATED_DIR --limit $LIMIT"
else
    echo "  [WARN] 未找到结果文件"
fi
echo ""

echo "=========================================="
echo "流程完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  1. 测试文件: $MIGRATED_DIR/"
echo "  2. 执行结果: $DATA_DIR/results/migrate_exec.jsonl"
echo "  3. HTML报告: $REPORTS_DIR/migration_comparison.html"
echo ""
echo "查看详细日志："
echo "  cat $DATA_DIR/migrate_logs/*.log"
echo ""

