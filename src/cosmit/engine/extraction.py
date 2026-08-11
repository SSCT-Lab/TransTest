from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Iterable

from cosmit.domain.models import ComponentId, ComponentMatch, ComponentProfile, TestIntent, TestKnowledge


FRAMEWORK_ALIASES = {
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "torch": "torch",
}


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


class ImportResolver:
    def __init__(self, tree: ast.Module) -> None:
        self.aliases = dict(FRAMEWORK_ALIASES)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def canonical(self, value: str | None) -> str | None:
        if not value:
            return None
        head, *tail = value.split(".")
        resolved = self.aliases.get(head, FRAMEWORK_ALIASES.get(head, head))
        return ".".join((resolved, *tail)) if tail else resolved


def _source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ast.unparse(node)


def _contains_call(node: ast.AST, resolver: ImportResolver, qualified_name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and resolver.canonical(dotted_name(child.func)) == qualified_name
        for child in ast.walk(node)
    )


def _is_assertion(node: ast.AST, resolver: ImportResolver) -> bool:
    if isinstance(node, ast.Assert):
        return True
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = (resolver.canonical(dotted_name(child.func)) or "").lower()
            if "assert" in name or name.endswith("raises"):
                return True
    return False


def _infer_intents(function_source: str) -> tuple[TestIntent, ...]:
    lowered = function_source.lower()
    intents = {TestIntent.FUNCTIONAL}
    if any(token in lowered for token in ("assert_near", "assert_close", "allclose", "tolerance")):
        intents.add(TestIntent.NUMERICAL)
    if "shape" in lowered or "reshape" in lowered:
        intents.add(TestIntent.SHAPE)
    if "dtype" in lowered or "float16" in lowered or "float64" in lowered:
        intents.add(TestIntent.DTYPE)
    if "device" in lowered or "cuda" in lowered or "gpu" in lowered:
        intents.add(TestIntent.DEVICE)
    if "gradient" in lowered or "backward" in lowered or "grad" in lowered:
        intents.add(TestIntent.GRADIENT)
    if "raises" in lowered or "exception" in lowered or "assert_error" in lowered:
        intents.add(TestIntent.EXCEPTION)
    if any(token in lowered for token in ("nan", "inf", "empty", "zero", "boundary")):
        intents.add(TestIntent.BOUNDARY)
    if "regression" in lowered or "issue" in lowered:
        intents.add(TestIntent.REGRESSION)
    return tuple(sorted(intents, key=lambda intent: intent.value))


def _constraints(function: ast.FunctionDef, source: str, resolver: ImportResolver) -> tuple[str, ...]:
    values: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.keyword) and node.arg:
            values.add(f"{node.arg}={_source_segment(source, node.value)}")
        elif isinstance(node, ast.Attribute):
            name = resolver.canonical(dotted_name(node)) or ""
            if name.startswith(("tensorflow.", "torch.")) and any(
                token in name.lower() for token in ("float", "int", "bool", "complex")
            ):
                values.add(name)
    return tuple(sorted(values))


def extract_test_knowledge(
    test_file: Path,
    match: ComponentMatch,
) -> list[TestKnowledge]:
    source = test_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(test_file))
    resolver = ImportResolver(tree)
    environment = tuple(
        _source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    extracted: list[TestKnowledge] = []

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test")
    ]
    for function in functions:
        if not _contains_call(function, resolver, match.source.qualified_name):
            continue
        function_source = _source_segment(source, function)
        inputs: list[str] = []
        component_construction: list[str] = []
        execution: list[str] = []
        assertions: list[str] = []
        seen_execution = False
        for statement in function.body:
            text = _source_segment(source, statement)
            is_execution = _contains_call(statement, resolver, match.source.qualified_name)
            if is_execution:
                execution.append(text)
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    component_construction.append(text)
                seen_execution = True
            elif _is_assertion(statement, resolver):
                assertions.append(text)
            elif not seen_execution and isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                inputs.append(text)

        if not execution or not assertions:
            continue
        raw_id = f"{match.match_id}:{test_file.resolve()}:{function.name}"
        knowledge_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
        extracted.append(
            TestKnowledge(
                knowledge_id=knowledge_id,
                source_component=match.source,
                source_test=function.name,
                intents=_infer_intents(function_source),
                environment=environment,
                input_construction=tuple(inputs),
                component_construction=tuple(component_construction),
                execution=tuple(execution),
                assertions=tuple(assertions),
                constraints=_constraints(function, source, resolver),
                match_id=match.match_id,
                target_component=match.target,
                source_file=test_file.resolve(),
                source_code=function_source,
            )
        )
    return extracted


def extract_for_matches(
    matches: Iterable[ComponentMatch],
    source_profiles: Iterable[ComponentProfile],
) -> list[TestKnowledge]:
    profiles = {profile.component.canonical: profile for profile in source_profiles}
    knowledge: list[TestKnowledge] = []
    for match in matches:
        profile = profiles.get(match.source.canonical)
        if profile is None:
            continue
        for test_file in profile.test_files:
            knowledge.extend(extract_test_knowledge(test_file, match))
    return knowledge


def knowledge_record(item: TestKnowledge) -> dict[str, object]:
    target = item.target_component
    return {
        "knowledge_id": item.knowledge_id,
        "match_id": item.match_id,
        "source_component": {
            "framework": item.source_component.framework,
            "qualified_name": item.source_component.qualified_name,
        },
        "target_component": {
            "framework": target.framework if target else "",
            "qualified_name": target.qualified_name if target else "",
        },
        "source_test": item.source_test,
        "source_file": str(item.source_file or ""),
        "intents": [intent.value for intent in item.intents],
        "environment": list(item.environment),
        "input_construction": list(item.input_construction),
        "component_construction": list(item.component_construction),
        "execution": list(item.execution),
        "assertions": list(item.assertions),
        "cleanup": list(item.cleanup),
        "constraints": list(item.constraints),
        "source_code": item.source_code,
    }
