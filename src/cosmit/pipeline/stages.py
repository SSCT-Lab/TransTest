from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class StageName(str, Enum):
    IDENTIFY = "identify-critical-components"
    RETRIEVE = "retrieve-similar-components"
    EXTRACT = "extract-test-knowledge"
    MIGRATE = "migrate-test-code"
    VALIDATE = "validate-and-repair"
    RANK = "rank-and-expand"


@dataclass(frozen=True, slots=True)
class StageSpec:
    order: int
    name: StageName
    goal: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    acceptance: str

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["name"] = self.name.value
        return result


PIPELINE_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        order=1,
        name=StageName.IDENTIFY,
        goal="识别目标框架中最需要测试知识迁移的关键组件",
        inputs=("target_component_catalog", "usage_history", "test_inventory", "change_history", "defect_history"),
        outputs=("critical_components",),
        acceptance="每个入选组件具备五维关键性证据、可解释分数和排序位置",
    ),
    StageSpec(
        order=2,
        name=StageName.RETRIEVE,
        goal="从一个或多个源框架检索功能相似组件",
        inputs=("critical_components", "source_component_catalogs", "framework_documents", "source_repositories"),
        outputs=("component_matches",),
        acceptance="每个候选对具备六维相似性证据、综合分数和 Top-K 排名",
    ),
    StageSpec(
        order=3,
        name=StageName.EXTRACT,
        goal="从源测试中抽取结构化测试知识和测试意图",
        inputs=("component_matches", "source_test_repositories"),
        outputs=("test_knowledge_bundles",),
        acceptance="知识单元覆盖六类代码结构、参数约束和至少一种测试意图",
    ),
    StageSpec(
        order=4,
        name=StageName.MIGRATE,
        goal="执行 API 映射、参数对齐、输入适配、断言重构和测试设施适配",
        inputs=("test_knowledge_bundles", "component_matches", "target_test_infrastructure"),
        outputs=("migration_candidates",),
        acceptance="候选测试可追溯到源知识单元，并显式记录所有语义转换",
    ),
    StageSpec(
        order=5,
        name=StageName.VALIDATE,
        goal="通过静态检查、隔离执行和反馈修复获得可执行测试",
        inputs=("migration_candidates", "runtime_profiles"),
        outputs=("validated_candidates", "repairable_candidates", "discarded_candidates"),
        acceptance="每个候选具有确定状态、执行证据、失败分类和修复历史",
    ),
    StageSpec(
        order=6,
        name=StageName.RANK,
        goal="检查意图保持，执行差分与数值分析，并排序和扩展有效测试",
        inputs=("validated_candidates", "source_execution_evidence", "runtime_profiles"),
        outputs=("ranked_tests", "expanded_tests", "migration_report"),
        acceptance="最终测试保持原始意图，且具备价值评分、覆盖增益和差分证据",
    ),
)


EXTERNAL_ARTIFACTS = {
    "target_component_catalog",
    "usage_history",
    "test_inventory",
    "change_history",
    "defect_history",
    "source_component_catalogs",
    "framework_documents",
    "source_repositories",
    "source_test_repositories",
    "target_test_infrastructure",
    "runtime_profiles",
    "source_execution_evidence",
}


def validate_pipeline_contract(stages: tuple[StageSpec, ...] = PIPELINE_STAGES) -> None:
    orders = [stage.order for stage in stages]
    if orders != list(range(1, len(stages) + 1)):
        raise ValueError(f"pipeline orders must be contiguous from 1, got {orders}")

    available = set(EXTERNAL_ARTIFACTS)
    for stage in stages:
        missing = set(stage.inputs) - available
        if missing:
            raise ValueError(f"stage {stage.name.value} has unavailable inputs: {sorted(missing)}")
        duplicated = set(stage.outputs) & available
        if duplicated:
            raise ValueError(f"stage {stage.name.value} overwrites artifacts: {sorted(duplicated)}")
        available.update(stage.outputs)
