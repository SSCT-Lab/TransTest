from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

from cosmit.config import CoSMiTConfig
from cosmit.domain.models import ComponentMatch, MigrationCandidate, TestKnowledge
from cosmit.engine.components import (
    critical_record,
    identify_critical_components,
    load_component_profiles,
    match_record,
    retrieve_similar_components,
)
from cosmit.engine.extraction import extract_for_matches, knowledge_record
from cosmit.engine.migration import candidate_record, load_api_rules, migrate_all
from cosmit.engine.ranking import rank_candidates
from cosmit.engine.validation import validate_candidate
from cosmit.io import write_json, write_jsonl
from cosmit.pipeline.stages import validate_pipeline_contract


class ArtifactStore:
    def __init__(self, root: Path, run_id: str) -> None:
        if not run_id.strip() or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a non-empty path-safe name")
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def direction_dir(self, slug: str) -> Path:
        path = self.run_dir / slug
        path.mkdir(parents=True, exist_ok=False)
        return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _git_commit(repository: Path) -> str | None:
    if not (repository / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _direction_summary(
    critical_count: int,
    match_count: int,
    knowledge_count: int,
    candidate_count: int,
    validation: list[dict[str, object]],
    ranked: list[dict[str, object]],
) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    for item in validation:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "critical_components": critical_count,
        "component_matches": match_count,
        "test_knowledge_units": knowledge_count,
        "migration_candidates": candidate_count,
        "validation_status": status_counts,
        "ranked_tests": len(ranked),
        "final_suite_eligible": sum(bool(item["eligible_for_final_suite"]) for item in ranked),
    }


def run_pipeline(
    config: CoSMiTConfig,
    run_id: str,
    artifacts_dir: Path | None = None,
) -> dict[str, object]:
    validate_pipeline_contract()
    store = ArtifactStore(artifacts_dir or config.artifacts_dir, run_id)
    profiles = {
        framework_id: load_component_profiles(framework.component_catalog)
        for framework_id, framework in config.frameworks.items()
    }
    for framework_id, items in profiles.items():
        if any(item.component.framework != framework_id for item in items):
            raise ValueError(f"component catalog framework mismatch: {framework_id}")
    rules = load_api_rules(config.mappings_file)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "directions": [direction.slug for direction in config.directions],
        "dynamic_validation": config.dynamic_validation,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": _package_version("tensorflow"),
            "pytorch": _package_version("torch"),
        },
        "frameworks": {
            framework_id: {
                "repository": str(framework.repository),
                "repository_commit": _git_commit(framework.repository),
                "component_catalog": str(framework.component_catalog),
                "component_catalog_sha256": _sha256(framework.component_catalog),
            }
            for framework_id, framework in config.frameworks.items()
        },
        "api_mappings": {
            "path": str(config.mappings_file),
            "sha256": _sha256(config.mappings_file),
        },
        "config": {
            "critical_components_per_direction": config.critical_components_per_direction,
            "top_k_similar_components": config.top_k_similar_components,
            "max_repair_rounds": config.max_repair_rounds,
        },
    }
    write_json(store.run_dir / "manifest.json", manifest)

    summaries: dict[str, object] = {}
    for direction in config.directions:
        direction_dir = store.direction_dir(direction.slug)
        critical = identify_critical_components(
            profiles[direction.target],
            config.criticality_weights,
            config.critical_components_per_direction,
        )
        write_jsonl(
            direction_dir / "01-critical-components" / "critical_components.jsonl",
            (critical_record(profile, score, rank) for profile, score, rank in critical),
        )

        matches = retrieve_similar_components(
            critical,
            profiles[direction.source],
            config.similarity_weights,
            config.top_k_similar_components,
        )
        write_jsonl(
            direction_dir / "02-component-matches" / "component_matches.jsonl",
            (match_record(match) for match in matches),
        )

        knowledge = extract_for_matches(matches, profiles[direction.source])
        write_jsonl(
            direction_dir / "03-test-knowledge" / "test_knowledge_bundles.jsonl",
            (knowledge_record(item) for item in knowledge),
        )

        candidates = migrate_all(knowledge, rules)
        write_jsonl(
            direction_dir / "04-migration-candidates" / "migration_candidates.jsonl",
            (candidate_record(candidate) for candidate in candidates),
        )

        validated_candidates: list[MigrationCandidate] = []
        validation_records: list[dict[str, object]] = []
        candidate_dir = direction_dir / "05-validation" / "candidates"
        for candidate in candidates:
            validated, record = validate_candidate(
                candidate=candidate,
                candidate_path=candidate_dir / f"{candidate.candidate_id}.py",
                dynamic=config.dynamic_validation,
                timeout_seconds=config.execution_timeout_seconds,
                max_repair_rounds=config.max_repair_rounds,
            )
            validated_candidates.append(validated)
            validation_records.append(record)
        write_jsonl(
            direction_dir / "05-validation" / "validation_results.jsonl",
            validation_records,
        )

        knowledge_by_id: Mapping[str, TestKnowledge] = {
            item.knowledge_id: item for item in knowledge
        }
        matches_by_id: Mapping[str, ComponentMatch] = {
            item.match_id: item for item in matches
        }
        validation_by_id: Mapping[str, Mapping[str, object]] = {
            str(item["candidate_id"]): item for item in validation_records
        }
        ranked = rank_candidates(
            validated_candidates,
            validation_by_id,
            knowledge_by_id,
            matches_by_id,
        )
        write_jsonl(
            direction_dir / "06-ranked-tests" / "ranked_tests.jsonl",
            ranked,
        )
        write_jsonl(
            direction_dir / "06-ranked-tests" / "expansion_plans.jsonl",
            (
                {
                    "candidate_id": item["candidate_id"],
                    "eligible_for_final_suite": item["eligible_for_final_suite"],
                    "expansion_plan": item["expansion_plan"],
                }
                for item in ranked
            ),
        )

        summary = _direction_summary(
            len(critical),
            len(matches),
            len(knowledge),
            len(candidates),
            validation_records,
            ranked,
        )
        write_json(direction_dir / "06-ranked-tests" / "migration_report.json", summary)
        summaries[direction.slug] = summary

    result = {"run_id": run_id, "run_dir": str(store.run_dir), "directions": summaries}
    write_json(store.run_dir / "summary.json", result)
    return result
