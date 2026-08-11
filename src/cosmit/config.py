from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from cosmit.domain.models import CriticalitySignals, SimilaritySignals


@dataclass(frozen=True, slots=True)
class FrameworkConfig:
    framework_id: str
    repository: Path
    component_catalog: Path


@dataclass(frozen=True, slots=True)
class MigrationDirection:
    source: str
    target: str

    @property
    def slug(self) -> str:
        return f"{self.source}-to-{self.target}"


@dataclass(frozen=True, slots=True)
class CoSMiTConfig:
    frameworks: Mapping[str, FrameworkConfig]
    directions: tuple[MigrationDirection, ...]
    mappings_file: Path
    artifacts_dir: Path
    critical_components_per_direction: int
    top_k_similar_components: int
    max_repair_rounds: int
    dynamic_validation: bool
    execution_timeout_seconds: int
    criticality_weights: Mapping[str, float]
    similarity_weights: Mapping[str, float]


def _resolve(base: Path, value: object, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} must not be empty")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _framework(
    framework_id: str,
    raw: Mapping[str, Any],
    base: Path,
) -> FrameworkConfig:
    if framework_id not in {"tensorflow", "pytorch"}:
        raise ValueError(f"unsupported framework in the initial scope: {framework_id}")
    return FrameworkConfig(
        framework_id=framework_id,
        repository=_resolve(base, raw.get("repository"), f"frameworks.{framework_id}.repository"),
        component_catalog=_resolve(
            base,
            raw.get("component_catalog"),
            f"frameworks.{framework_id}.component_catalog",
        ),
    )


def load_config(path: str | Path) -> CoSMiTConfig:
    config_path = Path(path).resolve()
    base = config_path.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    project = raw.get("project", {})
    if project.get("name") != "CoSMiT":
        raise ValueError("project.name must be exactly 'CoSMiT'")

    framework_raw = raw.get("frameworks", {})
    if set(framework_raw) != {"tensorflow", "pytorch"}:
        raise ValueError("initial CoSMiT scope requires exactly tensorflow and pytorch")
    frameworks = {
        framework_id: _framework(framework_id, value, base)
        for framework_id, value in framework_raw.items()
    }

    direction_raw = raw.get("directions", [])
    if not isinstance(direction_raw, list) or not direction_raw:
        raise ValueError("at least one migration direction is required")
    directions = tuple(
        MigrationDirection(source=str(item.get("source", "")), target=str(item.get("target", "")))
        for item in direction_raw
    )
    expected = {("tensorflow", "pytorch"), ("pytorch", "tensorflow")}
    actual = {(item.source, item.target) for item in directions}
    if actual != expected:
        raise ValueError("directions must contain tensorflow->pytorch and pytorch->tensorflow")

    pipeline = raw.get("pipeline", {})
    critical_limit = int(pipeline.get("critical_components_per_direction", 20))
    top_k = int(pipeline.get("top_k_similar_components", 5))
    max_repairs = int(pipeline.get("max_repair_rounds", 2))
    if critical_limit <= 0 or top_k <= 0:
        raise ValueError("component limits must be positive")
    if max_repairs < 0:
        raise ValueError("max_repair_rounds must be non-negative")

    validation = raw.get("validation", {})
    timeout = int(validation.get("timeout_seconds", 30))
    if timeout <= 0:
        raise ValueError("validation.timeout_seconds must be positive")

    criticality_weights = dict(raw.get("criticality_weights", {}))
    similarity_weights = dict(raw.get("similarity_weights", {}))
    CriticalitySignals(0, 0, 0, 0, 0).score(criticality_weights)
    SimilaritySignals(0, 0, 0, 0, 0, 0).score(similarity_weights)

    inputs = raw.get("inputs", {})
    return CoSMiTConfig(
        frameworks=frameworks,
        directions=directions,
        mappings_file=_resolve(base, inputs.get("api_mappings"), "inputs.api_mappings"),
        artifacts_dir=_resolve(base, pipeline.get("artifacts_dir", "../artifacts"), "pipeline.artifacts_dir"),
        critical_components_per_direction=critical_limit,
        top_k_similar_components=top_k,
        max_repair_rounds=max_repairs,
        dynamic_validation=bool(validation.get("dynamic", False)),
        execution_timeout_seconds=timeout,
        criticality_weights=criticality_weights,
        similarity_weights=similarity_weights,
    )
