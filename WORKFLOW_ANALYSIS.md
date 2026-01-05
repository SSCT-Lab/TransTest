# 迁移测试流程分析与问题诊断

## 完整流程

### 阶段1: 组件扫描与匹配

1. **扫描组件**: `component/scan_components.py`
   - 输出: `data/tf_components.jsonl`, `data/pt_components.jsonl`
   - 扫描 TensorFlow 和 PyTorch 源码中的函数/类定义

2. **生成 Embedding**: `component/embed_components.py` 或 `component/embed_components_ol.py`
   - 输出: `data/tf_vectors.npy`, `data/pt_vectors.npy`
   - ⚠️ **重复文件**: 两个文件功能相同，一个用本地模型，一个用在线API

3. **检索候选对**: `component/gen_candidates.py`
   - 输入: `tf_components.jsonl`, `pt_components.jsonl`, `tf_vectors.npy`, `pt_vectors.npy`
   - 输出: `data/component_candidates.jsonl`
   - 使用 embedding 相似度检索

4. **LLM 匹配**: `component/match_components_llm.py`
   - 输入: `component_candidates.jsonl`
   - 输出: `data/component_pairs.jsonl`
   - 用 LLM 分析函数对，生成最终映射

### 阶段2: 测试扫描与解析

5. **扫描测试文件**: `main.py`
   - 输出: `data/files_tf.jsonl`, `data/files_pt.jsonl`
   - 输出: `data/norm_tf.jsonl`, `data/norm_pt.jsonl`
   - 输出: `data/tests_tf.parsed.jsonl`, `data/tests_pt.parsed.jsonl`

2. **API 映射**: `core/api_mapping.py`
   - 输入: `tests_tf.parsed.jsonl`, `tests_pt.parsed.jsonl`
   - 输出: `data/tests_tf.mapped.jsonl`, `data/tests_pt.mapped.jsonl`
   - ✅ **已改进**: `parse_py.py` 现在全面提取所有 TensorFlow 相关 API（不仅限于 `tf.*` 开头）

### 阶段3: 测试迁移

7. **识别可迁移测试**: `component/migrate_identify_fuzzy.py`
   - 输入: `component_pairs.jsonl`, `tests_tf.mapped.jsonl`
   - 输出: `data/migration_candidates_fuzzy.jsonl`
   - ⚠️ **已修复**: 使用 `tests_tf.mapped.jsonl` 而不是 `tf_test_api_usage.jsonl`

2. **生成迁移测试**: `component/migrate_generate_tests.py`
   - 输入: `migration_candidates_fuzzy.jsonl`
   - 输出: `migrated_tests/*.py`
   - 生成模板代码（目前只有占位符）

## 问题分析

### 🔴 过时/重复文件

1. **`component/embed_components_ol.py`**
   - 状态: 与 `embed_components.py` 功能重复
   - 建议: 保留一个（根据使用场景选择本地或在线）

2. **`component/migrate_identify.py`**
   - 状态: **过时** - 使用 `tf_test_api_usage.jsonl`（文件级别，无测试函数信息）
   - 输出: `migration_candidates.jsonl` 为空（0B）
   - 建议: 删除或标记为过时，使用 `migrate_identify_fuzzy.py` 替代

3. **`component/scan_api_usage.py`**
   - 状态: 生成 `tf_test_api_usage.jsonl`，但不再被使用
   - 建议: 如果不再需要，可以删除

4. **`component/migrate_generate.py`**
   - 状态: 简单字符串替换 API，不够准确
   - 与 `migrate_generate_tests.py` 功能不同但可能混淆
   - 建议: 如果不用，标记为过时

5. **`component/migrate_tests.py`**
   - 状态: 生成迁移计划 `migration_plan.jsonl`，但可能不被使用
   - 建议: 确认是否被使用，如果不用可以删除

6. **`component/migrate_run.py`, `migrate_run_tests.py`, `migrate_run_dynamic.py`**
   - 状态: 三个运行脚本，功能重复
   - 建议: 统一为一个，或明确各自用途

### ⚠️ 数据流问题

1. **`migration_candidates.jsonl`** - 空文件（0B）
   - 原因: `migrate_identify.py` 使用错误的数据源
   - 解决: 已修复，使用 `migrate_identify_fuzzy.py`

2. **`migrated_tests/*.py`** - 只有占位符
   - 原因: `migrate_generate_tests.py` 只生成模板
   - 需要: 实现真正的代码迁移逻辑

### ✅ 已修复问题

1. **`core/parse_py.py`** - 现在只提取 `tf.*` 开头的 API
2. **`component/migrate_identify_fuzzy.py`** - 使用正确的数据源 `tests_tf.mapped.jsonl`

## 建议的清理操作

1. 删除或重命名过时文件:
   - `component/migrate_identify.py` → 标记为 `_deprecated`
   - `component/scan_api_usage.py` → 如果不用，删除
   - `component/migrate_generate.py` → 如果不用，标记为过时

2. 统一运行脚本:
   - 保留 `migrate_run_dynamic.py`（功能最全）
   - 删除或标记其他两个

3. 完善迁移生成:
   - 改进 `migrate_generate_tests.py`，实现真正的代码迁移
