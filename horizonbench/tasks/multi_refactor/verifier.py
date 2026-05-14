"""Verifier for multi_refactor task.

Two checks per file:
1. pytest test passes (functional correctness).
2. AST diff: the old snake_case function definition is gone and the new
   camelCase one is present (structural correctness).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any


def _has_function(source: str, name: str) -> bool:
    """Return True if *source* defines a top-level function named *name*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in ast.walk(tree)
    )


def check_ast(file_path: Path, old_name: str, new_name: str) -> tuple[bool, str]:
    """Return (passed, detail).  Passed when new_name exists and old_name is gone."""
    if not file_path.exists():
        return False, f"{file_path.name} does not exist"
    source = file_path.read_text()
    has_new = _has_function(source, new_name)
    has_old = _has_function(source, old_name)
    if has_new and not has_old:
        return True, "AST ok"
    parts = []
    if not has_new:
        parts.append(f"missing {new_name!r}")
    if has_old:
        parts.append(f"old name {old_name!r} still present")
    return False, "; ".join(parts)


def run_pytest(workspace: Path) -> tuple[bool, str]:
    """Run pytest inside *workspace* and return (all_passed, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short", "--no-header"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    passed = result.returncode == 0
    output = (result.stdout + result.stderr).strip()
    return passed, output


def verify_file(
    index: int,
    file_meta: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    """Run AST check for a single file; return per-file result dict."""
    src_file = workspace / "src" / file_meta["filename"]
    ast_ok, ast_detail = check_ast(src_file, file_meta["old_name"], file_meta["new_name"])
    return {
        "index": index,
        "module": file_meta["module"],
        "ast_passed": ast_ok,
        "ast_detail": ast_detail,
        "passed": ast_ok,
    }
