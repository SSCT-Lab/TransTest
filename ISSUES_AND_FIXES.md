# 问题诊断与修复建议

## 🔴 主要问题

### 1. 迁移测试输出为空/只有占位符

**问题**: `migrated_tests/*.py` 文件只有模板代码，没有真正的迁移逻辑

**原因**:

- `migrate_generate_tests.py` 只生成模板，没有读取原始测试代码并替换 API
- 应该读取原始 TensorFlow 测试文件，替换 API，然后生成 PyTorch 版本

**修复方案**:

- 修改 `migrate_generate_tests.py`，添加读取原始测试代码的逻辑
- 实现 API 替换功能（不仅仅是字符串替换，需要 AST 级别的替换）

### 2. 数据流问题

**问题**: `migration_candidates.jsonl` 为空（0B）

**原因**:

- `migrate_identify.py` 使用 `tf_test_api_usage.jsonl`（文件级别，没有测试函数信息）
- 应该使用 `tests_tf.mapped.jsonl`（测试函数级别）

**状态**: ✅ 已修复

- `migrate_identify_fuzzy.py` 已更新为使用 `tests_tf.mapped.jsonl`
- `core/parse_py.py` 已修复，只提取 `tf.*` 开头的 API

### 3. 重复/过时文件

#### 过时文件（建议删除或重命名）

1. **`component/migrate_identify.py`**
   - 使用错误的数据源 `tf_test_api_usage.jsonl`
   - 输出空文件 `migration_candidates.jsonl`
   - ✅ 已被 `migrate_identify_fuzzy.py` 替代

2. **`component/scan_api_usage.py`**
   - 生成 `tf_test_api_usage.jsonl`，但不再被使用
   - 如果不再需要，可以删除

3. **`component/migrate_generate.py`**
   - 简单字符串替换，不够准确
   - 与 `migrate_generate_tests.py` 功能不同但可能混淆
   - 如果不用，可以标记为过时

#### 重复文件

1. **`component/embed_components.py`** vs **`component/embed_components_ol.py`**
   - 功能相同，一个用本地模型，一个用在线 API
   - 建议保留一个，根据使用场景选择

2. **`component/migrate_run.py`**, **`migrate_run_tests.py`**, **`migrate_run_dynamic.py`**
   - 三个运行脚本，功能重复
   - 建议统一为一个（推荐 `migrate_run_dynamic.py`）

## ✅ 已修复的问题

1. **`core/parse_py.py`** - ✅ **已改进**：现在全面提取所有 TensorFlow 相关 API
   - 不仅限于 `tf.*` 开头
   - 通过分析 import 语句识别 TensorFlow 相关模块
   - 提取所有来自这些模块的 API 调用（如 `constant_op.constant`, `keras.layers.Dense` 等）
   - 支持各种 import 模式：`import tensorflow as tf`, `from tensorflow.xxx import yyy` 等
   - 测试验证：能正确提取 `tf.constant`, `constant_op.constant`, `keras.layers.Dense` 等

2. **`component/migrate_identify_fuzzy.py`** - ✅ 使用正确的数据源 `tests_tf.mapped.jsonl`

## 📋 完整流程（修复后）

```
1. scan_components.py → tf_components.jsonl, pt_components.jsonl
2. embed_components.py → tf_vectors.npy, pt_vectors.npy
3. gen_candidates.py → component_candidates.jsonl
4. match_components_llm.py → component_pairs.jsonl
5. main.py → tests_tf.parsed.jsonl, tests_pt.parsed.jsonl
6. core/api_mapping.py → tests_tf.mapped.jsonl, tests_pt.mapped.jsonl
7. migrate_identify_fuzzy.py → migration_candidates_fuzzy.jsonl
8. migrate_generate_tests.py → migrated_tests/*.py (需要改进)
```

## 🔧 下一步修复建议

1. **改进 `migrate_generate_tests.py`**:
   - 读取原始 TensorFlow 测试文件
   - 使用 AST 解析和替换，而不是简单字符串替换
   - 实现真正的 API 迁移逻辑

2. **清理过时文件**:
   - 重命名或删除 `migrate_identify.py`
   - 确认 `scan_api_usage.py` 是否还需要
   - 统一运行脚本

3. **验证数据流**:
   - 重新运行完整流程
   - 检查每个步骤的输出是否正确
