from __future__ import annotations

import ast
from typing import Iterable, Mapping

from cosmit.domain.models import ComponentMatch, MigrationCandidate, MigrationStatus, TestIntent, TestKnowledge
from cosmit.engine.extraction import ImportResolver, dotted_name


def _intent_checks(candidate: MigrationCandidate) -> dict[str, bool]:
    try:
        tree = ast.parse(candidate.code)
    except SyntaxError:
        return {"target_call_preserved": False, "assertion_preserved": False}
    resolver = ImportResolver(tree)
    calls = {
        resolver.canonical(dotted_name(node.func)) or dotted_name(node.func) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    has_assertion = any(isinstance(node, ast.Assert) for node in ast.walk(tree)) or any(
        "assert" in call.lower() for call in calls
    )
    return {
        "target_call_preserved": candidate.target_api in calls,
        "assertion_preserved": has_assertion,
    }


def _expansion_plan(knowledge: TestKnowledge, target_framework: str) -> list[dict[str, object]]:
    intents = set(knowledge.intents)
    plan: list[dict[str, object]] = []
    prefix = "tensorflow" if target_framework == "tensorflow" else "torch"
    if TestIntent.DTYPE in intents or TestIntent.NUMERICAL in intents:
        plan.append(
            {
                "dimension": "dtype",
                "values": [f"{prefix}.float16", f"{prefix}.float32", f"{prefix}.float64"],
                "requires_revalidation": True,
            }
        )
    if TestIntent.SHAPE in intents or TestIntent.BOUNDARY in intents:
        plan.append(
            {
                "dimension": "shape",
                "values": ["scalar", "empty-dimension", "singleton", "broadcastable"],
                "requires_revalidation": True,
            }
        )
    if TestIntent.DEVICE in intents:
        values = ["CPU:0", "GPU:0"] if target_framework == "tensorflow" else ["cpu", "cuda"]
        plan.append(
            {"dimension": "device", "values": values, "requires_revalidation": True}
        )
    if TestIntent.GRADIENT in intents:
        plan.append(
            {
                "dimension": "gradient",
                "values": ["first-order", "second-order"],
                "requires_revalidation": True,
            }
        )
    return plan


def rank_candidates(
    candidates: Iterable[MigrationCandidate],
    validation_records: Mapping[str, Mapping[str, object]],
    knowledge_by_id: Mapping[str, TestKnowledge],
    matches_by_id: Mapping[str, ComponentMatch],
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for candidate in candidates:
        knowledge = knowledge_by_id[candidate.knowledge_id]
        match = matches_by_id[knowledge.match_id]
        checks = _intent_checks(candidate)
        preservation = sum(checks.values()) / len(checks)
        validation_record = validation_records[candidate.candidate_id]
        validation_score = {
            MigrationStatus.VALIDATED.value: 1.0,
            MigrationStatus.STATIC_VALIDATED.value: 0.6,
            MigrationStatus.REPAIRABLE.value: 0.2,
        }.get(str(validation_record["status"]), 0.0)
        score = (
            0.35 * validation_score
            + 0.30 * preservation
            + 0.20 * match.similarity_score
            + 0.15 * match.target_criticality_score
        )
        ranked.append(
            {
                "candidate_id": candidate.candidate_id,
                "knowledge_id": candidate.knowledge_id,
                "source_framework": candidate.source_framework,
                "target_framework": candidate.target_component.framework,
                "target_api": candidate.target_api,
                "validation_status": validation_record["status"],
                "execution_verified": validation_record["execution_verified"],
                "eligible_for_final_suite": validation_record["execution_verified"] and all(checks.values()),
                "intent_checks": checks,
                "intent_preservation_score": preservation,
                "similarity_score": match.similarity_score,
                "criticality_score": match.target_criticality_score,
                "ranking_score": score,
                "expansion_plan": _expansion_plan(
                    knowledge,
                    candidate.target_component.framework,
                ),
            }
        )
    ranked.sort(key=lambda item: (-float(item["ranking_score"]), str(item["candidate_id"])))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked
