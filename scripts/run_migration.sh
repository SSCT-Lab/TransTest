#!/bin/bash
# TensorFlow → PyTorch 测试迁移完整执行脚本

set -e  # 遇到错误立即退出

echo "============================================"
echo "TensorFlow → PyTorch 测试迁移执行流程"
echo "============================================"
echo ""

# 配置
LIMIT=${1:-50}  # 默认生成50个测试
MIGRATED_DIR="migrated_tests"
DATA_DIR="data"

# 创建必要的目录
mkdir -p $MIGRATED_DIR
mkdir -p $DATA_DIR

# 步骤 1: 检查并准备数据（如果还没有）
echo "[步骤 1/7] 检查数据文件..."
if [ ! -f "$DATA_DIR/tests_tf.parsed.jsonl" ]; then
    echo "  执行数据准备..."
    python3 main.py
    python3 core/api_mapping.py
    echo "  ✓ 数据准备完成"
else
    echo "  ✓ 数据文件已存在，跳过准备步骤"
fi
echo ""

# 步骤 2: 识别可迁移测试（如果还没有）
echo "[步骤 2/7] 检查迁移候选..."
if [ ! -f "$DATA_DIR/migration_candidates_fuzzy.jsonl" ]; then
    echo "  执行识别..."
    python3 component/migrate_identify_fuzzy.py
    echo "  ✓ 识别完成"
else
    CANDIDATE_COUNT=$(wc -l < "$DATA_DIR/migration_candidates_fuzzy.jsonl" | tr -d ' ')
    echo "  ✓ 迁移候选文件已存在（$CANDIDATE_COUNT 个候选）"
fi
echo ""

# 步骤 3: 修复测试名称（如果需要）
echo "[步骤 3/7] 检查测试名称..."
python3 -c "
import json
from pathlib import Path
if Path('$DATA_DIR/migration_candidates_fuzzy.jsonl').exists():
    count = sum(1 for line in open('$DATA_DIR/migration_candidates_fuzzy.jsonl') 
                if json.loads(line).get('name', '').startswith('unknown_test_'))
    if count > 0:
        print(f'  发现 {count} 个 unknown_test_* 名称，需要修复')
        exit(1)
    else:
        print('  ✓ 所有测试名称正常')
        exit(0)
" && echo "  跳过修复" || {
    echo "  执行修复..."
    python3 component/migrate_fix_names.py
    echo "  ✓ 修复完成"
}
echo ""

# 步骤 4: 生成迁移测试
echo "[步骤 4/7] 生成迁移测试（限制 $LIMIT 个）..."
python3 component/migrate_generate_tests.py \
    --limit $LIMIT \
    --force \
    --output-dir $MIGRATED_DIR

GENERATED_COUNT=$(find $MIGRATED_DIR -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "  ✓ 已生成 $GENERATED_COUNT 个测试文件"
echo ""

# 步骤 5: 执行和对比测试
echo "[步骤 5/7] 执行 PyTorch 测试..."
python3 component/migrate_compare.py \
    --migrated-dir $MIGRATED_DIR \
    --limit $LIMIT \
    --skip-tf \
    --out $DATA_DIR/migrate_exec.jsonl

echo "  ✓ 测试执行完成"
echo ""

# 步骤 6: 生成报告
echo "[步骤 6/7] 生成迁移报告..."
if [ -f "$DATA_DIR/migrate_exec.jsonl" ]; then
    python3 component/migrate_report.py
    echo "  ✓ 报告生成完成"
else
    echo "  ⚠ 未找到执行结果文件，跳过报告生成"
fi
echo ""

# 步骤 7: 显示统计
echo "[步骤 7/7] 统计结果..."
if [ -f "$DATA_DIR/migrate_exec.jsonl" ]; then
    python3 -c "
import json
from pathlib import Path

results = [json.loads(l) for l in open('$DATA_DIR/migrate_exec.jsonl')]
total = len(results)
passed = sum(1 for r in results if (r.get('pt_result') or {}).get('status') == 'pass')
failed = sum(1 for r in results if (r.get('pt_result') or {}).get('status') == 'fail')
error = sum(1 for r in results if (r.get('pt_result') or {}).get('status') in ['error', 'timeout', 'not_found'])

print(f'  总测试数: {total}')
if total > 0:
    print(f'  通过: {passed} ({passed/total*100:.1f}%)')
    print(f'  失败: {failed} ({failed/total*100:.1f}%)')
    print(f'  错误: {error} ({error/total*100:.1f}%)')
else:
    print('  没有测试结果')
"
else
    echo "  ⚠ 未找到执行结果文件"
fi

echo ""
echo "============================================"
echo "执行完成！"
echo "============================================"
echo ""
echo "查看结果："
echo "  - 迁移测试文件: $MIGRATED_DIR/"
echo "  - 执行结果: $DATA_DIR/migrate_exec.jsonl"
echo "  - 报告文件: reports/"
echo ""

