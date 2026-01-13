#!/bin/bash
# 清理所有过去的测试文件

set -e

echo "=========================================="
echo "清理所有过去的测试文件"
echo "=========================================="
echo ""

# 清理 dev 目录下的测试文件
echo "[1/5] 清理 dev/tf_core/ ..."
rm -rf dev/tf_core/*.py
rm -f dev/tf_core/tf_core_index.jsonl
echo "  ✓ 已清理 dev/tf_core/"

echo "[2/5] 清理 dev/tf_fuzz/ ..."
rm -rf dev/tf_fuzz/*.py
rm -f dev/tf_fuzz/tf_fuzz_index.jsonl
echo "  ✓ 已清理 dev/tf_fuzz/"

echo "[3/5] 清理 dev/pt_migrated/ ..."
rm -rf dev/pt_migrated/*.py
echo "  ✓ 已清理 dev/pt_migrated/"

echo "[4/5] 清理 dev/tf_seeds.jsonl ..."
rm -f dev/tf_seeds.jsonl
echo "  ✓ 已清理 dev/tf_seeds.jsonl"

echo "[5/5] 清理 dev/results/ 和 dev/logs/ ..."
rm -f dev/results/*.jsonl
rm -f dev/logs/*.log
echo "  ✓ 已清理 dev/results/ 和 dev/logs/"

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="


