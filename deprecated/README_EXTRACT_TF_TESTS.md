# TensorFlow 测试提取和使用指南

## 工作流程

### 1. 提取 TensorFlow 测试逻辑

```bash
# 提取所有候选测试的逻辑，创建独立的测试文件
python3 component/migrate_extract_tf_tests.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output-dir extracted_tf_tests \
    --limit 50

# 或提取全部
python3 component/migrate_extract_tf_tests.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output-dir extracted_tf_tests
```

**说明：**
- 从 TensorFlow 原始测试文件中提取测试逻辑
- 创建独立的 Python 测试文件（不依赖 TensorFlow 测试框架）
- 处理类方法和独立函数
- 简化 `self.assert*` 调用为标准 `assert`
- 保存到 `extracted_tf_tests/` 目录

### 2. 执行提取的测试并保存结果

```bash
# 执行提取的测试并保存结果（优先使用提取的文件）
python3 component/migrate_collect_tf_results.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl \
    --extracted-dir extracted_tf_tests \
    --limit 50
```

**说明：**
- 优先使用提取的独立测试文件（避免 TensorFlow 测试框架问题）
- 如果提取的文件不存在，回退到原始测试文件
- 记录执行结果到静态文件

### 3. 生成 PyTorch 测试

```bash
python3 component/migrate_generate_tests.py --limit 50 --force
```

### 4. 对比测试（使用静态结果）

```bash
# 使用静态结果文件进行对比
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --tf-results data/tf_test_results.jsonl
```

### 5. 生成报告

```bash
python3 component/migrate_report.py
```

## 完整工作流示例

```bash
# 步骤 1: 提取 TensorFlow 测试逻辑
python3 component/migrate_extract_tf_tests.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output-dir extracted_tf_tests

# 步骤 2: 执行提取的测试并保存结果
python3 component/migrate_collect_tf_results.py \
    --input data/migration_candidates_fuzzy.jsonl \
    --output data/tf_test_results.jsonl \
    --extracted-dir extracted_tf_tests

# 步骤 3: 生成 PyTorch 测试
python3 component/migrate_generate_tests.py --limit 50 --force

# 步骤 4: 对比测试
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50 \
    --tf-results data/tf_test_results.jsonl

# 步骤 5: 生成报告
python3 component/migrate_report.py
```

## 提取的文件格式

提取的测试文件示例：

```python
# Extracted from: framework/tensorflow-master/.../test.py:testName
# Original test: testName

# Extracted TensorFlow test logic
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import test_util

# Test logic extracted from TensorFlow test class method
def testName():
    try:
        # 提取的测试逻辑（已移除 self. 调用）
        assert np.allclose(1.04719755, tf.acos(0.5))
        assert np.allclose(0.5, tf.rsqrt(4.0))
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    testName()
```

## 优势

1. **避免测试框架依赖**：不依赖 TensorFlow 的测试框架（`tf.test.TestCase`）
2. **独立执行**：提取的测试文件可以直接运行
3. **简化逻辑**：移除 `self.` 调用，转换为标准 Python 代码
4. **易于调试**：独立的测试文件更容易调试和修改
5. **可重复使用**：提取的文件可以多次执行，不需要重新提取

## 处理逻辑

### 类方法处理
- 提取测试方法体
- 移除 `self.` 前缀
- 转换 `self.assertAllClose` 为 `np.testing.assert_allclose`
- 转换 `self.assertEqual` 为 `assert`
- 保持测试逻辑不变

### 独立函数处理
- 直接使用原始函数代码
- 添加执行逻辑和错误处理

## 注意事项

1. **提取可能不完整**：复杂的类方法可能需要手动调整
2. **依赖处理**：某些测试可能需要额外的依赖，需要手动添加
3. **测试框架方法**：`self.assert*` 方法已简化，但可能不完全等价
4. **执行环境**：提取的测试仍需要 TensorFlow 环境

