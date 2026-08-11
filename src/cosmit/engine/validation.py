from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from cosmit.domain.models import MigrationCandidate, MigrationStatus
from cosmit.engine.extraction import ImportResolver, dotted_name


FORBIDDEN_CALLS = {"eval", "exec", "compile", "os.system", "subprocess.run", "subprocess.Popen"}


def _imports_framework(tree: ast.Module, framework: str) -> bool:
    module = "tensorflow" if framework == "tensorflow" else "torch"
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == module for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (
            node.module == module or (node.module or "").startswith(f"{module}.")
        ):
            return True
    return False


def static_issues(candidate: MigrationCandidate) -> list[str]:
    try:
        tree = ast.parse(candidate.code, filename=f"{candidate.candidate_id}.py")
        compile(tree, f"{candidate.candidate_id}.py", "exec")
    except SyntaxError as exc:
        return [f"syntax_error:{exc.msg}:{exc.lineno}"]

    issues: list[str] = []
    target_framework = candidate.target_component.framework
    if _imports_framework(tree, candidate.source_framework):
        issues.append("source_import_remaining")
    if not _imports_framework(tree, target_framework):
        issues.append("target_import_missing")

    resolver = ImportResolver(tree)
    calls = {
        resolver.canonical(dotted_name(node.func)) or dotted_name(node.func) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    if candidate.target_api not in calls:
        issues.append("target_call_missing")
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)) and not any(
        "assert" in call.lower() for call in calls
    ):
        issues.append("assertion_missing")
    forbidden = sorted(call for call in calls if call in FORBIDDEN_CALLS)
    issues.extend(f"forbidden_call:{call}" for call in forbidden)
    return issues


def _repair(candidate: MigrationCandidate, issues: list[str]) -> tuple[MigrationCandidate, list[str]]:
    code = candidate.code
    actions: list[str] = []
    target = candidate.target_component.framework
    source = candidate.source_framework
    if "source_import_remaining" in issues:
        if source == "tensorflow" and target == "pytorch":
            code = code.replace("import tensorflow as tf", "import torch")
            code = code.replace("import tensorflow", "import torch")
        elif source == "pytorch" and target == "tensorflow":
            code = code.replace("import torch", "import tensorflow")
        actions.append("repair:replace-source-import")
    if "target_import_missing" in issues:
        required = "import torch\n" if target == "pytorch" else "import tensorflow\n"
        code = required + code
        actions.append("repair:add-target-import")
    return replace(candidate, code=code, repair_round=candidate.repair_round + 1), actions


def _dynamic_execute(path: Path, timeout_seconds: int) -> dict[str, object]:
    harness = (
        "import runpy,sys; "
        "ns=runpy.run_path(sys.argv[1]); "
        "tests=[v for k,v in ns.items() if k.startswith('test') and callable(v)]; "
        "assert tests, 'no test functions found'; "
        "[test() for test in tests]"
    )
    allowed_env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "PYTHONHOME", "TMPDIR", "TEMP", "SYSTEMROOT", "CUDA_VISIBLE_DEVICES"}
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", harness, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=allowed_env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "stdout": "", "stderr": "timeout"}
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def validate_candidate(
    candidate: MigrationCandidate,
    candidate_path: Path,
    dynamic: bool,
    timeout_seconds: int,
    max_repair_rounds: int,
) -> tuple[MigrationCandidate, dict[str, object]]:
    working = candidate
    repair_history: list[dict[str, object]] = []
    issues = static_issues(working)
    while issues and working.repair_round < max_repair_rounds:
        repaired, actions = _repair(working, issues)
        if not actions:
            break
        repair_history.append(
            {"round": repaired.repair_round, "issues": issues, "actions": actions}
        )
        working = repaired
        issues = static_issues(working)

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(working.code, encoding="utf-8")
    if issues:
        status = MigrationStatus.STATIC_REJECTED
        dynamic_result: dict[str, object] = {"status": "not_run"}
    elif dynamic:
        dynamic_result = _dynamic_execute(candidate_path, timeout_seconds)
        status = (
            MigrationStatus.VALIDATED
            if dynamic_result["status"] == "passed"
            else MigrationStatus.REPAIRABLE
        )
    else:
        dynamic_result = {"status": "disabled"}
        status = MigrationStatus.STATIC_VALIDATED

    working = replace(working, status=status)
    record = {
        "candidate_id": working.candidate_id,
        "knowledge_id": working.knowledge_id,
        "status": status.value,
        "static_issues": issues,
        "dynamic": dynamic_result,
        "repair_history": repair_history,
        "candidate_path": str(candidate_path),
        "execution_verified": status is MigrationStatus.VALIDATED,
        "code": working.code,
    }
    return working, record
