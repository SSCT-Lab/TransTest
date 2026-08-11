from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Mapping

from cosmit.domain.models import (
    ComponentId,
    ComponentMatch,
    ComponentProfile,
    CriticalitySignals,
    SimilaritySignals,
)
from cosmit.io import read_jsonl


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"expected a list, got {type(value).__name__}")
    return tuple(str(item) for item in value)


def load_component_profiles(path: Path) -> list[ComponentProfile]:
    profiles: list[ComponentProfile] = []
    base = path.parent
    for record in read_jsonl(path):
        framework = str(record.get("framework", "")).strip()
        qualified_name = str(record.get("qualified_name", "")).strip()
        criticality_raw = record.get("criticality", {})
        if not isinstance(criticality_raw, dict):
            raise ValueError(f"criticality must be a mapping for {qualified_name}")
        test_files = tuple(
            candidate if candidate.is_absolute() else (base / candidate).resolve()
            for candidate in (Path(item) for item in _sequence(record.get("test_files")))
        )
        profiles.append(
            ComponentProfile(
                component=ComponentId(framework, qualified_name),
                kind=str(record.get("kind", "function")),
                signature=str(record.get("signature", qualified_name)),
                documentation=str(record.get("documentation", "")),
                source_code=str(record.get("source_code", "")),
                call_context=_sequence(record.get("call_context")),
                test_behavior=_sequence(record.get("test_behavior")),
                test_files=test_files,
                criticality=CriticalitySignals(
                    usage_frequency=float(criticality_raw.get("usage_frequency", 0)),
                    test_weakness=float(criticality_raw.get("test_weakness", 0)),
                    change_activity=float(criticality_raw.get("change_activity", 0)),
                    defect_sensitivity=float(criticality_raw.get("defect_sensitivity", 0)),
                    impact_scope=float(criticality_raw.get("impact_scope", 0)),
                ),
            )
        )
    return profiles


def identify_critical_components(
    profiles: Iterable[ComponentProfile],
    weights: Mapping[str, float],
    limit: int,
) -> list[tuple[ComponentProfile, float, int]]:
    scored = sorted(
        ((profile, profile.criticality.score(weights)) for profile in profiles),
        key=lambda item: (-item[1], item[0].component.qualified_name),
    )
    return [(profile, score, rank) for rank, (profile, score) in enumerate(scored[:limit], start=1)]


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item.lower() for item in left if item}
    right_set = {item.lower() for item in right if item}
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _leaf_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1].lower()


def compute_similarity(source: ComponentProfile, target: ComponentProfile) -> SimilaritySignals:
    name = SequenceMatcher(
        None,
        _leaf_name(source.component.qualified_name),
        _leaf_name(target.component.qualified_name),
    ).ratio()
    return SimilaritySignals(
        name=name,
        documentation=_jaccard(_tokens(source.documentation), _tokens(target.documentation)),
        parameters=_jaccard(_tokens(source.signature), _tokens(target.signature)),
        source_code=_jaccard(_tokens(source.source_code), _tokens(target.source_code)),
        call_context=_jaccard(source.call_context, target.call_context),
        test_behavior=_jaccard(source.test_behavior, target.test_behavior),
    )


def retrieve_similar_components(
    critical_targets: Iterable[tuple[ComponentProfile, float, int]],
    source_profiles: Iterable[ComponentProfile],
    weights: Mapping[str, float],
    top_k: int,
) -> list[ComponentMatch]:
    sources = tuple(source_profiles)
    matches: list[ComponentMatch] = []
    for target, criticality_score, _ in critical_targets:
        candidates = []
        for source in sources:
            signals = compute_similarity(source, target)
            candidates.append((source, signals, signals.score(weights)))
        candidates.sort(key=lambda item: (-item[2], item[0].component.qualified_name))
        for rank, (source, signals, score) in enumerate(candidates[:top_k], start=1):
            raw_id = f"{source.component.canonical}->{target.component.canonical}"
            match_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
            matches.append(
                ComponentMatch(
                    match_id=match_id,
                    source=source.component,
                    target=target.component,
                    similarity=signals,
                    similarity_score=score,
                    target_criticality_score=criticality_score,
                    rank=rank,
                )
            )
    return matches


def critical_record(profile: ComponentProfile, score: float, rank: int) -> dict[str, object]:
    return {
        "framework": profile.component.framework,
        "qualified_name": profile.component.qualified_name,
        "kind": profile.kind,
        "signature": profile.signature,
        "criticality_signals": {
            field: getattr(profile.criticality, field)
            for field in (
                "usage_frequency",
                "test_weakness",
                "change_activity",
                "defect_sensitivity",
                "impact_scope",
            )
        },
        "criticality_score": score,
        "rank": rank,
    }


def match_record(match: ComponentMatch) -> dict[str, object]:
    return {
        "match_id": match.match_id,
        "source": {
            "framework": match.source.framework,
            "qualified_name": match.source.qualified_name,
        },
        "target": {
            "framework": match.target.framework,
            "qualified_name": match.target.qualified_name,
        },
        "similarity_signals": {
            field: getattr(match.similarity, field)
            for field in (
                "name",
                "documentation",
                "parameters",
                "source_code",
                "call_context",
                "test_behavior",
            )
        },
        "similarity_score": match.similarity_score,
        "target_criticality_score": match.target_criticality_score,
        "rank": match.rank,
    }
