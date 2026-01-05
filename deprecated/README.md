# 过时文件说明

本目录包含已过时或不再使用的文件和目录。

## 文件列表

### 迁移相关脚本（已过时）

- `migrate_extract_tf_tests.py` - 用于提取 TensorFlow 测试到独立文件
  - **过时原因**: 现在 `migrate_generate_tests.py` 直接在生成的迁移文件中包含 TensorFlow 测试逻辑，不再需要单独提取
  
- `migrate_force.py` - 强制迁移脚本（旧版本）
- `migrate_force_run.py` - 强制运行脚本（旧版本）
- `migrate_run_dynamic.py` - 动态运行脚本（旧版本）
- `migrate_run.py` - 运行脚本（旧版本）
- `migrate_tests.py` - 测试脚本（旧版本）
- `migrate_generate.py` - 生成脚本（旧版本）
  - **过时原因**: 已被 `migrate_generate_tests.py` 替代，新版本功能更完善

### 目录

- `extracted_tf_tests/` - 提取的 TensorFlow 测试文件目录
  - **过时原因**: 现在 TensorFlow 测试逻辑直接包含在生成的迁移文件中，不再需要单独提取
  
- `migrated_test_old/` - 旧版本的迁移测试文件
  - **过时原因**: 已被 `migrated_tests/` 目录中的新版本替代

- `outputs/` - 旧的输出文件目录
  - **过时原因**: 包含旧的测试输出文件，现在使用 `migrated_tests/` 目录

### 文档

- `README_EXTRACT_TF_TESTS.md` - TensorFlow 测试提取指南
  - **过时原因**: 描述了旧的提取流程，现在已不再使用

## 当前推荐的工作流程

请参考 `README_MIGRATE_WITH_TF.md` 了解当前推荐的工作流程。

## 恢复文件

如果需要恢复这些文件，可以从 Git 历史中恢复，或从备份中复制。
