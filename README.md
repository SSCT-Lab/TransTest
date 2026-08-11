# CoSMiT

CoSMiT（Component Similarity-guided Test Knowledge Migration）是面向深度学习框架的组件级测试知识迁移方法。它从成熟框架的已有测试中抽取输入构造、执行逻辑、断言、参数约束和测试意图，将这些知识迁移到目标框架中测试薄弱但影响关键的组件，并通过执行、修复、差分验证和扩展形成可用测试资产。

当前版本是项目重启后的 `0.1` 双向 MVP，主范围固定为 TensorFlow ↔ PyTorch。六阶段已经可以端到端运行：读取组件画像、计算关键性和相似性、从 Python AST 抽取测试知识、按显式规则迁移、执行静态/动态验证、检查意图保持并输出排序与扩展计划。当前样例用于验证数据流和工程边界，尚不代表真实框架规模的实验结果。

## 方法概览

| 阶段 | 核心问题 | 主要产物 |
| --- | --- | --- |
| 1. 关键组件识别 | 哪些目标组件最需要补充测试？ | 关键组件及五维评分证据 |
| 2. 相似组件检索 | 哪些源组件可提供可迁移知识？ | 六维相似组件 Top-K |
| 3. 测试知识抽取 | 源测试真正验证了什么？ | 结构化测试知识与测试意图 |
| 4. 测试代码迁移 | 如何在目标框架中重建原意图？ | 可追溯的候选迁移测试 |
| 5. 验证与反馈修复 | 候选测试能否正确执行？ | 有效、可修复、丢弃三类结果 |
| 6. 排序与扩展 | 哪些测试最有价值，如何扩大覆盖？ | 排序测试、增强测试与报告 |

## 项目边界

CoSMiT 的输入是框架源码、API 文档、测试仓库、使用/变更/缺陷历史和运行环境；输出是目标框架原生、可执行、可追溯的测试代码及其验证证据。

CoSMiT 不做模型格式转换，不把 API 名称替换视为测试迁移，也不接受“代码可以运行”作为唯一成功标准。测试意图、核心输入和核心断言必须得到保留。

## 快速开始

```bash
python -m pip install -e .
cosmit show-pipeline
cosmit validate-config configs/cosmit.example.yaml
cosmit run configs/tf_pt_bidirectional.yaml
python -m unittest discover -s tests
```

也可以不安装包直接运行：

```bash
PYTHONPATH=src python -m cosmit show-pipeline
```

## 目录

```text
CoSMiT/
├── src/cosmit/             # 全新 CoSMiT 源码
│   ├── domain/             # 组件、测试知识、迁移候选等领域模型
│   ├── pipeline/           # 六阶段顺序与 artifact 契约
│   └── adapters/           # 框架差异边界
├── configs/                # 可版本化配置模板
├── examples/               # TF ↔ PT 双向最小可重复样例
├── docs/                   # 架构、相关工作、实验设计与路线图
├── references/cosmit/      # CoSMiT 正式项目参考材料
├── tests/                  # 新架构测试
└── others/                 # 本地冻结归档与排除材料，不进入新源码
```

## 设计文档

- [项目重设计说明](CoSMiT_REDESIGN.md)
- [系统架构](docs/architecture.md)
- [相关工作](docs/related_work.md)
- [实验设计](docs/evaluation.md)
- [实施路线](docs/roadmap.md)

旧项目和临时材料的分类见 [others/README.md](others/README.md)。它们不参与 CoSMiT 的设计与实现。

## 当前实现边界

- 已实现：双向配置、五维关键性评分、六维相似度、AST 测试知识抽取、API/导入/参数规则迁移、静态检查、隔离动态执行、有限规则修复、意图保持评分和扩展计划。
- 下一步：从真实 TensorFlow/PyTorch 仓库自动构建组件画像和测试索引，扩充 API/参数/断言规则，建立人工金标准并运行规模化实验。
- 动态执行默认关闭；正式配置 `configs/tf_pt_bidirectional.yaml` 显式开启。未经动态验证的候选只能进入预排序，不能进入最终测试集。
