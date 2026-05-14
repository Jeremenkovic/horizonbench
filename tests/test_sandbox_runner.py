"""Tests for sandbox runner (docker SDK version) and mock doc server."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from horizonbench.sandbox.runner import (
    ExecResult,
    Sandbox,
    SandboxConfig,
    SandboxRunner,
    build_local_tool_handler,
)


# ── unit tests (no Docker daemon required) ────────────────────────────────────


def test_exec_result_output():
    r = ExecResult(exit_code=0, stdout="hello\n", stderr="warn\n")
    assert "hello" in r.output
    assert "warn" in r.output


def test_sandbox_config_defaults():
    cfg = SandboxConfig()
    assert cfg.network_disabled is True
    assert cfg.memory_limit == "512m"


def _make_container_mock(exec_output=b"file contents\n", exit_code=0):
    container = MagicMock()
    container.short_id = "abc123"
    container.exec_run.return_value = (exit_code, exec_output)
    return container


def _make_docker_client_mock(container_mock):
    client = MagicMock()
    client.containers.run.return_value = container_mock
    return client


def test_runner_start_uses_docker_sdk(tmp_path):
    runner = SandboxRunner()
    container_mock = _make_container_mock()
    client_mock = _make_docker_client_mock(container_mock)
    with patch("docker.from_env", return_value=client_mock):
        container = runner._start_container(tmp_path, 60)
    assert client_mock.containers.run.called
    call_kwargs = client_mock.containers.run.call_args
    assert call_kwargs[1]["detach"] is True
    assert call_kwargs[1]["network_disabled"] is True


def test_runner_read_file_ok(tmp_path):
    runner = SandboxRunner()
    container = _make_container_mock(exec_output=b"hello world\n", exit_code=0)
    content = runner.read_file(container, "src/mod.py")
    assert "hello world" in content


def test_runner_read_file_missing(tmp_path):
    runner = SandboxRunner()
    container = _make_container_mock(exec_output=b"", exit_code=1)
    with pytest.raises(FileNotFoundError):
        runner.read_file(container, "nonexistent.py")


def test_runner_run_bash():
    runner = SandboxRunner()
    container = _make_container_mock(exec_output=b"ok\n", exit_code=0)
    out = runner.run_bash(container, "echo ok")
    assert "ok" in out


# ── local tool handler ────────────────────────────────────────────────────────


def test_local_tool_read_write(tmp_path):
    handler = build_local_tool_handler(tmp_path)
    result = handler("write_file", {"path": "test.txt", "content": "hello"})
    assert result == "ok"
    content = handler("read_file", {"path": "test.txt"})
    assert content == "hello"


def test_local_tool_read_missing(tmp_path):
    handler = build_local_tool_handler(tmp_path)
    result = handler("read_file", {"path": "nope.txt"})
    assert "ERROR" in result


def test_local_tool_run_bash(tmp_path):
    handler = build_local_tool_handler(tmp_path)
    result = handler("run_bash", {"command": "echo hello-world"})
    assert "hello-world" in result


def test_local_tool_unknown(tmp_path):
    handler = build_local_tool_handler(tmp_path)
    result = handler("unknown_tool", {})
    assert "ERROR" in result


def test_local_tool_list_documents(tmp_path):
    import json
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc_000.txt").write_text("first")
    (docs / "doc_001.txt").write_text("second")
    handler = build_local_tool_handler(tmp_path)
    result = handler("list_documents", {})
    ids = json.loads(result)
    assert ids == ["doc_000", "doc_001"]


def test_local_tool_read_document(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc_000.txt").write_text("the answer is 42")
    handler = build_local_tool_handler(tmp_path)
    result = handler("read_document", {"doc_id": "doc_000"})
    assert "42" in result


# ── mock doc server ────────────────────────────────────────────────────────────


def test_doc_server_list_and_get(tmp_path):
    """Start the mock server and hit its endpoints."""
    import httpx
    from horizonbench.sandbox.mock_servers.doc_server import DocServerThread

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc_000.txt").write_text("content of doc zero")

    with DocServerThread(docs, port=18765) as srv:
        ids = httpx.get(f"{srv.base_url}/documents").json()
        assert "doc_000" in ids
        text = httpx.get(f"{srv.base_url}/documents/doc_000").text
        assert "content of doc zero" in text
        resp = httpx.get(f"{srv.base_url}/documents/nonexistent")
        assert resp.status_code == 404


# ── docker integration (requires daemon) ──────────────────────────────────────


def _docker_available() -> bool:
    try:
        import docker
        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
def test_sandbox_integration(tmp_path):
    cfg = SandboxConfig(timeout_seconds=30, network_disabled=False)
    with Sandbox(tmp_path, config=cfg) as sb:
        sb.write_file("hello.txt", "world\n")
        content = sb.read_file("hello.txt")
        assert "world" in content
        out = sb.run_bash("echo hello-from-container")
        assert "hello-from-container" in out
