"""Generator for multi_refactor task instances.

Produces a small Python repo with k source files, each exposing one
snake_case function (``compute_value_<i>``) that the agent must rename to
camelCase (``computeValue<i>``).  The generated workspace also contains a
pytest suite that verifies the renaming, so the verifier can run it.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path


# ── templates ─────────────────────────────────────────────────────────────────

_SOURCE_TEMPLATE = textwrap.dedent("""\
    # File: {filename}
    # TODO: rename `{old_name}` → `{new_name}` (camelCase convention)

    def {old_name}(x: int) -> int:
        \"\"\"Return x squared plus a constant.\"\"\"
        return x * x + {constant}


    RESULT = {old_name}(10)
""")

_TEST_TEMPLATE = textwrap.dedent("""\
    # Auto-generated test suite for multi_refactor task.
    # Tests pass only after the refactor is applied correctly.
    import importlib, sys, types

    {imports}


    {tests}
""")

_CONFTEST = textwrap.dedent("""\
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
""")


def _snake(i: int) -> str:
    return f"compute_value_{i}"


def _camel(i: int) -> str:
    return f"computeValue{i}"


def _module_name(i: int) -> str:
    return f"module_{i:03d}"


def generate_workspace(k: int, seed: int, workspace: Path) -> dict:
    """Write a fresh workspace with k files and their test suite.

    Returns metadata about the generated files.
    """
    rng = random.Random(seed)
    src = workspace / "src"
    tests_dir = workspace / "tests"
    src.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (workspace / "conftest.py").write_text(_CONFTEST)

    files_meta = []
    for i in range(k):
        constant = rng.randint(1, 99)
        mod = _module_name(i)
        old = _snake(i)
        new = _camel(i)
        filename = f"{mod}.py"

        source = _SOURCE_TEMPLATE.format(
            filename=filename,
            old_name=old,
            new_name=new,
            constant=constant,
        )
        (src / filename).write_text(source)

        files_meta.append(
            {
                "index": i,
                "module": mod,
                "filename": filename,
                "old_name": old,
                "new_name": new,
                "constant": constant,
                "expected_result": 10 * 10 + constant,
            }
        )

    # Write pytest suite
    imports = "\n".join(
        f"import src.{m['module']} as mod_{m['index']}" for m in files_meta
    )
    tests = "\n\n".join(
        textwrap.dedent(f"""\
            def test_{m['module']}_renamed():
                assert hasattr(mod_{m['index']}, '{m['new_name']}'), (
                    "Function '{m['new_name']}' not found in {m['module']}"
                )
                assert not hasattr(mod_{m['index']}, '{m['old_name']}'), (
                    "Old name '{m['old_name']}' still present in {m['module']}"
                )

            def test_{m['module']}_result():
                fn = getattr(mod_{m['index']}, '{m['new_name']}')
                assert fn(10) == {m['expected_result']}
        """)
        for m in files_meta
    )

    test_src = _TEST_TEMPLATE.format(imports=imports, tests=tests)
    (tests_dir / "test_refactor.py").write_text(test_src)

    # Write a minimal pyproject so pytest can find the package
    (workspace / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n"
    )

    return {"files": files_meta, "k": k, "seed": seed}
