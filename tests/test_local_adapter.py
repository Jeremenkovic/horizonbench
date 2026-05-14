"""Tests for local adapter (vLLM / Ollama) — mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from horizonbench.adapters.local import OllamaAdapter, VLLMAdapter, make_local_adapter


def _mock_litellm_response():
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = "result"
    msg.tool_calls = []
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 5
    usage.total_tokens = 10
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model_dump.return_value = {}
    return resp


def test_vllm_adapter_model_prefix():
    adapter = VLLMAdapter("llama-3-70b")
    assert adapter.config.model.startswith("openai/")


def test_vllm_adapter_api_base():
    adapter = VLLMAdapter("llama-3-70b", api_base="http://myserver:8000/v1")
    assert adapter.config.extra["api_base"] == "http://myserver:8000/v1"


def test_ollama_adapter_model_prefix():
    adapter = OllamaAdapter("mistral")
    assert adapter.config.model.startswith("ollama/")


def test_make_local_adapter_vllm():
    adapter = make_local_adapter("vllm", "llama-3-70b")
    assert isinstance(adapter, VLLMAdapter)


def test_make_local_adapter_ollama():
    adapter = make_local_adapter("ollama", "mistral")
    assert isinstance(adapter, OllamaAdapter)


def test_make_local_adapter_unknown():
    with pytest.raises(ValueError, match="Unknown local backend"):
        make_local_adapter("unknown", "model")


def test_vllm_adapter_complete():
    from horizonbench.adapters.base import Message
    adapter = VLLMAdapter("llama-3-70b")
    mock_resp = _mock_litellm_response()
    with patch("litellm.completion", return_value=mock_resp):
        with patch("litellm.completion_cost", return_value=0.0):
            response = adapter.complete([Message(role="user", content="hello")])
    assert response.message.content == "result"
