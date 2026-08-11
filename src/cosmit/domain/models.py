from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Mapping


def _validate_unit_interval(instance: object) -> None:
    for field in fields(instance):
        value = getattr(instance, field.name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{field.name} must be a number")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field.name} must be within [0, 1]")


def _weighted_score(instance: object, weights: Mapping[str, float] | None) -> float:
    names = tuple(field.name for field in fields(instance))
    active = dict(weights) if weights is not None else {name: 1.0 for name in names}
    if set(active) != set(names):
        missing = sorted(set(names) - set(active))
        extra = sorted(set(active) - set(names))
        raise ValueError(f"weight keys mismatch; missing={missing}, extra={extra}")
    if any(weight < 0 for weight in active.values()):
        raise ValueError("weights must be non-negative")
    total = sum(active.values())
    if total <= 0:
        raise ValueError("weight sum must be positive")
    return sum(float(getattr(instance, name)) * active[name] for name in names) / total


@dataclass(frozen=True, slots=True)
class ComponentId:
    framework: str
    qualified_name: str

    def __post_init__(self) -> None:
        if not self.framework.strip():
            raise ValueError("framework must not be empty")
        if not self.qualified_name.strip():
            raise ValueError("qualified_name must not be empty")

    @property
    def canonical(self) -> str:
        return f"{self.framework}:{self.qualified_name}"


@dataclass(frozen=True, slots=True)
class CriticalitySignals:
    usage_frequency: float
    test_weakness: float
    change_activity: float
    defect_sensitivity: float
    impact_scope: float

    def __post_init__(self) -> None:
        _validate_unit_interval(self)

    def score(self, weights: Mapping[str, float] | None = None) -> float:
        return _weighted_score(self, weights)


@dataclass(frozen=True, slots=True)
class SimilaritySignals:
    name: float
    documentation: float
    parameters: float
    source_code: float
    call_context: float
    test_behavior: float

    def __post_init__(self) -> None:
        _validate_unit_interval(self)

    def score(self, weights: Mapping[str, float] | None = None) -> float:
        return _weighted_score(self, weights)


@dataclass(frozen=True, slots=True)
class ComponentProfile:
    component: ComponentId
    kind: str
    signature: str
    documentation: str
    source_code: str
    call_context: tuple[str, ...]
    test_behavior: tuple[str, ...]
    test_files: tuple[Path, ...]
    criticality: CriticalitySignals

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("component kind must not be empty")
        if not self.signature.strip():
            raise ValueError("component signature must not be empty")


@dataclass(frozen=True, slots=True)
class ComponentMatch:
    match_id: str
    source: ComponentId
    target: ComponentId
    similarity: SimilaritySignals
    similarity_score: float
    target_criticality_score: float
    rank: int

    def __post_init__(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id must not be empty")
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("similarity_score must be within [0, 1]")
        if not 0.0 <= self.target_criticality_score <= 1.0:
            raise ValueError("target_criticality_score must be within [0, 1]")
        if self.rank <= 0:
            raise ValueError("rank must be positive")


class TestIntent(str, Enum):
    FUNCTIONAL = "functional"
    NUMERICAL = "numerical"
    SHAPE = "shape"
    DTYPE = "dtype"
    DEVICE = "device"
    GRADIENT = "gradient"
    EXCEPTION = "exception"
    BOUNDARY = "boundary"
    REGRESSION = "regression"


@dataclass(frozen=True, slots=True)
class TestKnowledge:
    knowledge_id: str
    source_component: ComponentId
    source_test: str
    intents: tuple[TestIntent, ...]
    environment: tuple[str, ...] = ()
    input_construction: tuple[str, ...] = ()
    component_construction: tuple[str, ...] = ()
    execution: tuple[str, ...] = ()
    assertions: tuple[str, ...] = ()
    cleanup: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    match_id: str = ""
    target_component: ComponentId | None = None
    source_file: Path | None = None
    source_code: str = ""

    def __post_init__(self) -> None:
        if not self.knowledge_id.strip():
            raise ValueError("knowledge_id must not be empty")
        if not self.source_test.strip():
            raise ValueError("source_test must not be empty")
        if not self.intents:
            raise ValueError("at least one test intent is required")
        if not self.execution:
            raise ValueError("execution knowledge must not be empty")
        if not self.assertions:
            raise ValueError("assertion knowledge must not be empty")


class MigrationStatus(str, Enum):
    GENERATED = "generated"
    STATIC_REJECTED = "static_rejected"
    STATIC_VALIDATED = "static_validated"
    REPAIRABLE = "repairable"
    VALIDATED = "validated"
    DISCARDED = "discarded"


@dataclass(frozen=True, slots=True)
class MigrationCandidate:
    candidate_id: str
    knowledge_id: str
    target_component: ComponentId
    code: str
    status: MigrationStatus = MigrationStatus.GENERATED
    repair_round: int = 0
    source_framework: str = ""
    target_api: str = ""
    transformations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if not self.knowledge_id.strip():
            raise ValueError("knowledge_id must not be empty")
        if not self.code.strip():
            raise ValueError("candidate code must not be empty")
        if self.repair_round < 0:
            raise ValueError("repair_round must be non-negative")
