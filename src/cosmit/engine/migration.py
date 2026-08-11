from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from cosmit.domain.models import MigrationCandidate, TestKnowledge
from cosmit.engine.extraction import ImportResolver, dotted_name


@dataclass(frozen=True, slots=True)
class ApiRule:
    source: str
    target: str
    source_api: str
    target_api: str
    parameter_map: Mapping[str, str]


def load_api_rules(path: Path) -> tuple[ApiRule, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("api_mappings"), list):
        raise ValueError("mapping file must contain an api_mappings list")
    rules = []
    for index, item in enumerate(raw["api_mappings"]):
        if not isinstance(item, dict):
            raise ValueError(f"api_mappings[{index}] must be a mapping")
        rule = ApiRule(
            source=str(item.get("source", "")),
            target=str(item.get("target", "")),
            source_api=str(item.get("source_api", "")),
            target_api=str(item.get("target_api", "")),
            parameter_map={
                str(key): str(value)
                for key, value in dict(item.get("parameter_map", {})).items()
            },
        )
        if not all((rule.source, rule.target, rule.source_api, rule.target_api)):
            raise ValueError(f"api_mappings[{index}] has empty fields")
        rules.append(rule)
    return tuple(rules)


def _expression(dotted: str) -> ast.expr:
    parsed = ast.parse(dotted, mode="eval")
    return parsed.body


class RuleBasedTransformer(ast.NodeTransformer):
    def __init__(
        self,
        source_framework: str,
        target_framework: str,
        resolver: ImportResolver,
        mappings: dict[str, ApiRule],
    ) -> None:
        self.source_framework = source_framework
        self.target_framework = target_framework
        self.resolver = resolver
        self.mappings = mappings
        self.transformations: list[str] = []

    def _rule(self, node: ast.AST) -> tuple[str | None, ApiRule | None]:
        raw = dotted_name(node)
        canonical = self.resolver.canonical(raw)
        rule = self.mappings.get(canonical or "") or self.mappings.get(raw or "")
        return canonical or raw, rule

    def _replacement(self, node: ast.AST) -> ast.expr | None:
        source, rule = self._rule(node)
        if not rule:
            return None
        self.transformations.append(f"api:{source}->{rule.target_api}")
        return ast.copy_location(_expression(rule.target_api), node)

    def visit_Import(self, node: ast.Import) -> ast.AST:
        rewritten: list[ast.alias] = []
        changed = False
        for alias in node.names:
            if self.source_framework == "tensorflow" and alias.name == "tensorflow":
                rewritten.append(ast.alias(name="torch", asname=None))
                changed = True
            elif self.source_framework == "pytorch" and alias.name == "torch":
                rewritten.append(ast.alias(name="tensorflow", asname=None))
                changed = True
            else:
                rewritten.append(alias)
        if changed:
            self.transformations.append(
                f"import:{self.source_framework}->{self.target_framework}"
            )
        node.names = rewritten
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        module = node.module or ""
        source_module = "tensorflow" if self.source_framework == "tensorflow" else "torch"
        if module == source_module or module.startswith(f"{source_module}."):
            self.transformations.append(
                f"import-from:{module}->{self.target_framework}"
            )
            if self.target_framework == "pytorch":
                return ast.copy_location(ast.Import(names=[ast.alias(name="torch")]), node)
            return ast.copy_location(ast.Import(names=[ast.alias(name="tensorflow")]), node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.args.args and node.args.args[0].arg == "self":
            node.args.args = node.args.args[1:]
            self.transformations.append("test-method:self-parameter-removed")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        source, rule = self._rule(node.func)
        if rule is not None:
            node.func = ast.copy_location(_expression(rule.target_api), node.func)
            self.transformations.append(f"api:{source}->{rule.target_api}")
            for keyword in node.keywords:
                if keyword.arg in rule.parameter_map:
                    previous = keyword.arg
                    keyword.arg = rule.parameter_map[previous]
                    self.transformations.append(
                        f"parameter:{previous}->{keyword.arg}"
                    )
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        replacement = self._replacement(node)
        return replacement if replacement is not None else node


def migrate_knowledge(
    knowledge: TestKnowledge,
    rules: Iterable[ApiRule],
) -> MigrationCandidate | None:
    target_component = knowledge.target_component
    if target_component is None:
        return None
    source_framework = knowledge.source_component.framework
    target_framework = target_component.framework
    active_rules = {
        rule.source_api: rule
        for rule in rules
        if rule.source == source_framework and rule.target == target_framework
    }
    component_rule = active_rules.get(knowledge.source_component.qualified_name)
    if component_rule is None or component_rule.target_api != target_component.qualified_name:
        return None

    combined = "\n".join((*knowledge.environment, "", knowledge.source_code))
    tree = ast.parse(combined, filename=str(knowledge.source_file or "<knowledge>"))
    resolver = ImportResolver(tree)
    transformer = RuleBasedTransformer(
        source_framework=source_framework,
        target_framework=target_framework,
        resolver=resolver,
        mappings=active_rules,
    )
    migrated = transformer.visit(tree)
    ast.fix_missing_locations(migrated)
    code = ast.unparse(migrated).strip() + "\n"
    raw_id = f"{knowledge.knowledge_id}:{target_component.canonical}"
    candidate_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
    return MigrationCandidate(
        candidate_id=candidate_id,
        knowledge_id=knowledge.knowledge_id,
        target_component=target_component,
        code=code,
        source_framework=source_framework,
        target_api=target_component.qualified_name,
        transformations=tuple(transformer.transformations),
    )


def migrate_all(
    knowledge: Iterable[TestKnowledge],
    rules: Iterable[ApiRule],
) -> list[MigrationCandidate]:
    active_rules = tuple(rules)
    candidates = []
    for item in knowledge:
        candidate = migrate_knowledge(item, active_rules)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def candidate_record(candidate: MigrationCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "knowledge_id": candidate.knowledge_id,
        "source_framework": candidate.source_framework,
        "target_component": {
            "framework": candidate.target_component.framework,
            "qualified_name": candidate.target_component.qualified_name,
        },
        "target_api": candidate.target_api,
        "status": candidate.status.value,
        "repair_round": candidate.repair_round,
        "transformations": list(candidate.transformations),
        "code": candidate.code,
    }
