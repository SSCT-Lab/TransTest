# 迁移测试包含 TensorFlow 原始测试逻辑

## 概述

现在生成的迁移测试文件包含：
1. **TensorFlow 原始测试逻辑** - 提取并转换为独立可执行的函数
2. **PyTorch 迁移后的测试** - 使用 LLM 生成的迁移代码
3. **对比测试函数** - 自动执行两个测试并对比结果

## 生成的文件结构

```python
import torch
import pytest
import tensorflow as tf
import numpy as np

# Helper functions for TensorFlow test framework methods
def assertAllClose(a, b, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def assertAllEqual(a, b):
    np.testing.assert_array_equal(a, b)

# ===== TensorFlow Original Test =====
def testRenames():
    """Original TensorFlow test logic"""
    try:
        assertAllClose(1.04719755, tf.acos(0.5))
        assertAllClose(0.5, tf.rsqrt(4.0))
        print("TF: PASS")
    except Exception as e:
        print(f"TF: FAIL - {e}")

# ===== PyTorch Migrated Test =====
def test_testRenames_pt():
    # PyTorch migrated code
    assert torch.allclose(torch.acos(torch.tensor(0.5)), torch.tensor(1.04719755))
    # ...

# ===== Comparison Test =====
def test_testRenames_compare():
    """Compare TensorFlow and PyTorch test results"""
    # 自动执行两个测试并对比
    # ...
```

## 使用方法

### 1. 生成迁移测试（包含 TensorFlow 逻辑）

```bash
python3 component/migrate_generate_tests.py --limit 50 --force
```

这会生成包含 TensorFlow 原始测试逻辑的迁移文件。

### 2. 执行单个测试文件

```bash
# 直接运行文件（会执行 TensorFlow 测试）
python3 migrated_tests/testRenames.py

# 或使用 pytest 运行对比测试
pytest migrated_tests/testRenames.py::test_testRenames_compare -v
```

### 3. 批量执行并对比

```bash
# 执行所有迁移测试的对比函数
pytest migrated_tests/ -k "_compare" -v

# 或使用自定义脚本
python3 component/migrate_compare.py \
    --migrated-dir migrated_tests \
    --limit 50
```

## 优势

1. **自包含**：每个测试文件包含完整的 TensorFlow 和 PyTorch 测试逻辑
2. **直接对比**：可以在同一个文件中执行两个测试并对比
3. **不依赖原始文件**：不需要访问原始的 TensorFlow 测试文件
4. **易于调试**：所有代码都在一个文件中，方便查看和调试
5. **可独立运行**：每个测试文件都可以独立执行

## 测试函数说明

- `testRenames()` - TensorFlow 原始测试逻辑（提取并转换）
- `test_testRenames_pt()` - PyTorch 迁移后的测试
- `test_testRenames_compare()` - 自动对比两个测试的结果

## 执行结果

对比测试会输出：
- `PASS: Both TF and PT tests passed` - 两者都通过
- `FAIL: Both TF and PT tests failed` - 两者都失败
- `MISMATCH: TF=..., PT=...` - 结果不一致
- 详细的错误信息（如果有）

