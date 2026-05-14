"""Docker sandbox runner using the Python docker SDK.

Lifecycle per run:
1. Start an ephemeral container with the workspace bind-mounted at /work/.
2. Tool calls (read_file, write_file, run_bash) are dispatched via docker exec.
3. On completion or timeout, the container is stopped and removed.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import docker
    from docker.errors import DockerException, NotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.12-slim"


@dataclass
class SandboxConfig:
    image: str = DEFAULT_IMAGE
    timeout_seconds: int = 300
    memory_limit: str = "512m"
    network_disabled: bool = True


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


class SandboxRunner:
    """Manages a single ephemeral Docker container for one task run."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._client: Any = None

    def _get_client(self) -> Any:
        if not DOCKER_AVAILABLE:
            raise RuntimeError("docker SDK not installed")
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    # ── container lifecycle ────────────────────────────────────────────────────

    def _start_container(self, workspace: Path, timeout: int) -> Any:
        """Start a detached container and return the Container object."""
        client = self._get_client()
        volume_spec = {str(workspace.resolve()): {"bind": "/work", "mode": "rw"}}
        container = client.containers.run(
            self.config.image,
            command=["sleep", str(timeout + 60)],
            detach=True,
            remove=True,
            volumes=volume_spec,
            working_dir="/work",
            network_disabled=self.config.network_disabled,
            mem_limit=self.config.memory_limit,
        )
        logger.debug("Started container %s", container.short_id)
        return container

    def _stop_container(self, container: Any) -> None:
        try:
            container.stop(timeout=5)
        except Exception:
            pass
        logger.debug("Stopped container %s", container.short_id)

    # ── tool dispatch ──────────────────────────────────────────────────────────

    def exec_in_container(self, container: Any, command: list[str]) -> ExecResult:
        exit_code, output = container.exec_run(command, demux=False)
        text = output.decode("utf-8", errors="replace") if output else ""
        return ExecResult(exit_code=exit_code or 0, stdout=text, stderr="")

    def read_file(self, container: Any, path: str) -> str:
        result = self.exec_in_container(container, ["cat", f"/work/{path}"])
        if result.exit_code != 0:
            raise FileNotFoundError(f"/work/{path}: not found")
        return result.stdout

    def write_file(self, container: Any, path: str, content: str) -> None:
        full_path = f"/work/{path}"
        parent = str(Path(full_path).parent)
        self.exec_in_container(container, ["mkdir", "-p", parent])
        # Write via sh heredoc piped through base64 to avoid quoting issues
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        self.exec_in_container(
            container,
            ["sh", "-c", f"echo '{encoded}' | base64 -d > {full_path}"],
        )

    def run_bash(self, container: Any, command: str) -> str:
        result = self.exec_in_container(container, ["sh", "-c", command])
        return result.output


class Sandbox:
    """Context manager for a single container lifetime."""

    def __init__(self, workspace: Path, config: SandboxConfig | None = None):
        self.workspace = workspace
        self.runner = SandboxRunner(config=config or SandboxConfig())
        self._container: Any = None

    def __enter__(self) -> "Sandbox":
        timeout = self.runner.config.timeout_seconds
        self._container = self.runner._start_container(self.workspace, timeout)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._container is not None:
            self.runner._stop_container(self._container)
            self._container = None

    def read_file(self, path: str) -> str:
        return self.runner.read_file(self._container, path)

    def write_file(self, path: str, content: str) -> None:
        self.runner.write_file(self._container, path, content)

    def run_bash(self, command: str) -> str:
        return self.runner.run_bash(self._container, command)


def build_local_tool_handler(workspace: Path):
    """Return a tool handler that operates directly on the local filesystem.

    Used when running without Docker (local dev, CI without Docker daemon).
    """
    import subprocess
    import sys

    def handler(tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "read_file":
            path = workspace / args.get("path", "")
            try:
                return path.read_text()
            except FileNotFoundError:
                return f"ERROR: file not found: {args.get('path')}"
        elif tool_name == "write_file":
            path = workspace / args.get("path", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.get("content", ""))
            return "ok"
        elif tool_name == "run_bash":
            cmd = args.get("command", "")
            # Redirect dangerous filesystem-wide finds to workspace only
            import re
            cmd = re.sub(r'\bfind\s+/\s+', f'find {workspace} ', cmd)
            cmd = re.sub(r'\bfind\s+/\b', f'find {workspace}', cmd)
            result = subprocess.run(
                ["sh", "-c", cmd],
                capture_output=True,
                text=True,
                cwd=workspace,
                timeout=60,
            )
            return (result.stdout + result.stderr).strip()
        elif tool_name == "read_document":
            # For research_synth: read a document from docs/ by id
            doc_id = args.get("doc_id", "")
            path = workspace / "docs" / f"{doc_id}.txt"
            try:
                return path.read_text()
            except FileNotFoundError:
                return f"ERROR: document not found: {doc_id}"
        elif tool_name == "list_documents":
            docs_dir = workspace / "docs"
            if not docs_dir.exists():
                return "[]"
            import json
            return json.dumps(sorted(p.stem for p in docs_dir.glob("*.txt")))
        else:
            return f"ERROR: unknown tool {tool_name!r}"

    return handler
