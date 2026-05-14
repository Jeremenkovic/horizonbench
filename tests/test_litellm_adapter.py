"""Tests for LiteLLM adapter — mocked to avoid live API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from horizonbench.adapters.base import AdapterConfig, Message, Usage
from horizonbench.adapters.litellm import LiteLLMAdapter, _to_litellm_message


# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_response(content="hello", finish_reason="stop", tool_calls=None):
    """Build a minimal object that mimics a LiteLLM ModelResponse."""
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model_dump.return_value = {}
    return resp


# ── _to_litellm_message ────────────────────────────────────────────────────────


def test_to_litellm_message_basic():
    m = Message(role="user", content="hi")
    d = _to_litellm_message(m)
    assert d["role"] == "user"
    assert d["content"] == "hi"
    assert "tool_calls" not in d


def test_to_litellm_message_with_tool_calls():
    tc = [{"id": "1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
    m = Message(role="assistant", content="", tool_calls=tc)
    d = _to_litellm_message(m)
    assert d["tool_calls"] == tc


def test_to_litellm_message_tool_result():
    m = Message(role="tool", content="result", tool_call_id="call-1")
    d = _to_litellm_message(m)
    assert d["tool_call_id"] == "call-1"


# ── LiteLLMAdapter.complete ────────────────────────────────────────────────────


def test_adapter_complete_basic():
    cfg = AdapterConfig(model="claude-sonnet-4-6")
    adapter = LiteLLMAdapter(cfg)
    mock_resp = _mock_response(content="world")
    with patch("litellm.completion", return_value=mock_resp) as mock_completion:
        with patch("litellm.completion_cost", return_value=0.001):
            response = adapter.complete([Message(role="user", content="hello")])
    mock_completion.assert_called_once()
    assert response.message.content == "world"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 15
    assert response.usage.cost_usd == pytest.approx(0.001)


def test_adapter_complete_with_tools():
    cfg = AdapterConfig(model="gpt-4o")
    adapter = LiteLLMAdapter(cfg)
    tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
    mock_resp = _mock_response()
    with patch("litellm.completion", return_value=mock_resp) as mock_completion:
        with patch("litellm.completion_cost", return_value=0.0):
            adapter.complete([Message(role="user", content="do it")], tools=tools)
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"


def test_adapter_finish_reason_tool_calls():
    cfg = AdapterConfig(model="gpt-4o")
    adapter = LiteLLMAdapter(cfg)
    mock_resp = _mock_response(finish_reason="tool_calls")
    with patch("litellm.completion", return_value=mock_resp):
        with patch("litellm.completion_cost", return_value=0.0):
            resp = adapter.complete([Message(role="user", content="x")])
    assert resp.finish_reason == "tool_calls"


def test_adapter_tool_loop_no_tools():
    """When model returns stop on first call, loop exits immediately."""
    cfg = AdapterConfig(model="claude-sonnet-4-6")
    adapter = LiteLLMAdapter(cfg)
    mock_resp = _mock_response(content="done")
    calls = []

    def tool_handler(name, args):
        calls.append((name, args))
        return "ok"

    with patch("litellm.completion", return_value=mock_resp):
        with patch("litellm.completion_cost", return_value=0.0):
            messages = [Message(role="user", content="start")]
            final_messages, usage = adapter.run_tool_loop(messages, [], tool_handler)

    assert len(calls) == 0
    assert final_messages[-1].content == "done"


def test_adapter_tool_loop_one_tool_call():
    cfg = AdapterConfig(model="claude-sonnet-4-6")
    adapter = LiteLLMAdapter(cfg)

    tc = [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]
    resp_with_tool = _mock_response(finish_reason="tool_calls", tool_calls=tc)
    resp_final = _mock_response(content="all done", finish_reason="stop")

    call_seq = iter([resp_with_tool, resp_final])
    calls = []

    def tool_handler(name, args):
        calls.append((name, args))
        return "file contents"

    with patch("litellm.completion", side_effect=lambda **kw: next(call_seq)):
        with patch("litellm.completion_cost", return_value=0.0):
            messages = [Message(role="user", content="go")]
            final_messages, usage = adapter.run_tool_loop(messages, [], tool_handler)

    assert calls == [("read_file", {"path": "a.py"})]
    assert final_messages[-1].content == "all done"
