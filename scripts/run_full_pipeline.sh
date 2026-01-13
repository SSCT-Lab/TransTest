#!/bin/bash
# 完整流程：清理 -> 提取 TF -> Fuzzing -> 迁移 -> 执行 -> 下载文档

set -e

echo "=========================================="
echo "TF -> PT 完整迁移流程"
echo "=========================================="
echo ""

# 配置
NUM_FUZZ_VARIANTS=50
CONDA_ENV_TF="tf2pt-dev-tf"
CONDA_ENV_PT="tf2pt-dev-pt"
LIMIT_TESTS=${1:-100}  # 默认提取 100 个测试

echo "配置:"
echo "  - 每个 seed 生成 ${NUM_FUZZ_VARIANTS} 个 fuzzing 变体"
echo "  - TF 环境: ${CONDA_ENV_TF}"
echo "  - PT 环境: ${CONDA_ENV_PT}"
echo "  - 提取测试数量: ${LIMIT_TESTS}"
echo ""

# 步骤 1: 清理
echo "[步骤 1/8] 清理所有过去的测试..."
bash scripts/clean_all_tests.sh
echo ""

# 步骤 2: 提取 TF core 测试
echo "[步骤 2/8] 提取 TF core 测试（限制 ${LIMIT_TESTS} 个）..."
python3 component/migration/dev_extract_tf_core.py \
    --limit ${LIMIT_TESTS} \
    --output-dir dev/tf_core
TF_CORE_COUNT=$(find dev/tf_core -name "*.py" -type f | wc -l | tr -d ' ')
echo "  ✓ 已提取 ${TF_CORE_COUNT} 个 TF core 测试"
echo ""

# 步骤 3: 筛选合规 seed
echo "[步骤 3/8] 筛选合规的 TF seed..."
python3 component/migration/dev_filter_tf_seeds.py \
    --limit ${LIMIT_TESTS} \
    --conda-env ${CONDA_ENV_TF} \
    --seeds-out dev/tf_seeds.jsonl
SEED_COUNT=$(wc -l < dev/tf_seeds.jsonl 2>/dev/null | tr -d ' ' || echo "0")
echo "  ✓ 找到 ${SEED_COUNT} 个合规 seed"
echo ""

# 步骤 4: Fuzzing 生成变体
echo "[步骤 4/8] 对每个 seed 进行 fuzzing（生成 ${NUM_FUZZ_VARIANTS} 个变体）..."
python3 component/migration/dev_fuzz_tf_seeds.py \
    --seeds-file dev/tf_seeds.jsonl \
    --output-dir dev/tf_fuzz \
    --num-variants ${NUM_FUZZ_VARIANTS} \
    --force
FUZZ_COUNT=$(find dev/tf_fuzz -name "*.py" -type f | wc -l | tr -d ' ')
echo "  ✓ 已生成 ${FUZZ_COUNT} 个 fuzzing 测试"
echo ""

# 步骤 5: 迁移到 PyTorch
echo "[步骤 5/8] 迁移所有测试到 PyTorch（使用 10 个线程）..."
python3 component/migration/dev_migrate_all_to_pt.py \
    --seeds-file dev/tf_seeds.jsonl \
    --fuzz-index dev/tf_fuzz/tf_fuzz_index.jsonl \
    --output-dir dev/pt_migrated \
    --key-path aliyun.key \
    --workers 10 \
    --force
PT_COUNT=$(find dev/pt_migrated -name "*.py" -type f | wc -l | tr -d ' ')
echo "  ✓ 已迁移 ${PT_COUNT} 个 PyTorch 测试"
echo ""

# 步骤 6: 执行 TF 测试（包括 core 和 fuzz）
echo "[步骤 6/8] 执行 TF 测试..."
echo "  6.1 执行 TF core 测试..."
python3 component/migration/migrate_run_tf_tests.py \
    --migrated-dir dev/tf_core \
    --out dev/results/tf_core_exec.jsonl \
    --log-dir dev/logs \
    --conda-env ${CONDA_ENV_TF} \
    --limit -1

echo "  6.2 执行 TF fuzzing 测试..."
if [ -d "dev/tf_fuzz" ] && [ "$(ls -A dev/tf_fuzz/*.py 2>/dev/null)" ]; then
    python3 component/migration/migrate_run_tf_tests.py \
        --migrated-dir dev/tf_fuzz \
        --out dev/results/tf_fuzz_exec.jsonl \
        --log-dir dev/logs \
        --conda-env ${CONDA_ENV_TF} \
        --limit -1
else
    echo "    (跳过：没有 fuzzing 测试)"
fi

# 合并结果
cat dev/results/tf_core_exec.jsonl dev/results/tf_fuzz_exec.jsonl 2>/dev/null > dev/results/tf_exec.jsonl || \
    cp dev/results/tf_core_exec.jsonl dev/results/tf_exec.jsonl 2>/dev/null || true

echo "  ✓ TF 测试执行完成"
echo ""

# 步骤 7: 执行 PT 测试
echo "[步骤 7/8] 执行 PT 测试..."
python3 component/migration/migrate_run_tests.py \
    --migrated-dir dev/pt_migrated \
    --out dev/results/pt_exec.jsonl \
    --log-dir dev/logs \
    --limit -1
echo "  ✓ PT 测试执行完成"
echo ""

# 步骤 8: 提取 API 并下载文档
echo "[步骤 8/8] 提取 API 并下载文档..."
echo "  8.1 从测试文件中提取所有 API..."
python3 component/doc/extract_apis_from_tests.py \
    --output data/analysis/test_apis.jsonl \
    --tf-core-dir dev/tf_core \
    --tf-fuzz-dir dev/tf_fuzz \
    --pt-migrated-dir dev/pt_migrated

echo "  8.2 批量下载文档..."
python3 component/doc/crawl_apis_from_tests.py \
    --input data/analysis/test_apis.jsonl \
    --output data/analysis/test_api_docs.jsonl
echo "  ✓ 文档下载完成"
echo ""

echo "=========================================="
echo "完整流程执行完成！"
echo "=========================================="
echo ""
echo "统计:"
echo "  - TF core 测试: ${TF_CORE_COUNT}"
echo "  - 合规 seed: ${SEED_COUNT}"
echo "  - Fuzzing 测试: ${FUZZ_COUNT}"
echo "  - PyTorch 测试: ${PT_COUNT}"
echo ""
echo "结果文件:"
echo "  - TF 执行结果: dev/results/tf_exec.jsonl"
echo "  - PT 执行结果: dev/results/pt_exec.jsonl"
echo "  - API 文档: data/analysis/test_api_docs.jsonl"
echo ""

