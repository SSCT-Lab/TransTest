#!/bin/bash
# 快速分析测试错误的脚本

if [ $# -lt 2 ]; then
    echo "用法: $0 <错误信息> <测试文件路径> [额外参数]"
    echo ""
    echo "示例:"
    echo "  $0 \"IndentationError: ...\" migrated_tests/testArgRenames.py"
    echo "  $0 \"TypeError: ...\" migrated_tests/test.py --tf-apis tf.reduce_sum --pt-apis torch.sum"
    exit 1
fi

ERROR_MSG="$1"
TEST_FILE="$2"
shift 2

# 运行分析工具
python3 component/doc/doc_analyzer.py "$ERROR_MSG" --test-file "$TEST_FILE" "$@"

