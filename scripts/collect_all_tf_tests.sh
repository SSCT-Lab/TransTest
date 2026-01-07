#!/bin/bash
# 收集所有 TensorFlow 测试的执行结果

echo "=========================================="
echo "收集所有 TensorFlow 测试执行结果"
echo "=========================================="
echo ""

# 配置
INPUT_FILE="data/migration/migration_candidates_fuzzy.jsonl"
OUTPUT_FILE="data/results/tf_test_results.jsonl"
TF_ROOT="framework/tensorflow-master"

echo "[STEP 1] 检查输入文件..."
if [ ! -f "$INPUT_FILE" ]; then
    echo "  [ERROR] 输入文件不存在: $INPUT_FILE"
    echo "  请先运行: python3 component/migration/migrate_identify_fuzzy.py"
    exit 1
fi

echo "  [OK] 输入文件存在: $INPUT_FILE"

echo ""
echo "[STEP 2] 开始收集 TensorFlow 测试结果..."
echo "  这将执行所有候选测试并保存结果到: $OUTPUT_FILE"
echo "  注意：这可能需要较长时间，请耐心等待..."
echo ""

# 执行收集脚本
python3 component/migration/migrate_collect_tf_results.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --tf-root "$TF_ROOT"

if [ $? -eq 0 ]; then
    echo ""
    echo "[DONE] 收集完成！"
    echo "  结果文件: $OUTPUT_FILE"
    echo ""
    echo "后续使用："
    echo "  python3 component/migration/migrate_compare.py --tf-results $OUTPUT_FILE"
else
    echo ""
    echo "[ERROR] 收集失败，请检查错误信息"
    exit 1
fi

