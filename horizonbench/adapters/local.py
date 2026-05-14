"""Local model adapter — vLLM (OpenAI-compatible) or Ollama.

Both expose an OpenAI-compatible HTTP endpoint.  We route them through
LiteLLM using the appropriate model prefix:
  - vLLM:  "openai/<model_name>" with api_base pointing to the vLLM server.
  - Ollama: "ollama/<model_name>" (LiteLLM has native Ollama support).
"""

from __future__ import annotations

import logging
from typing import Any

from horizonbench.adapters.base import AdapterConfig, AdapterResponse, Message, ModelAdapter
from horizonbench.adapters.litellm import LiteLLMAdapter

logger = logging.getLogger(__name__)

# Default endpoints
VLLM_DEFAULT_BASE = "http://localhost:8000/v1"
OLLAMA_DEFAULT_BASE = "http://localhost:11434"


class VLLMAdapter(ModelAdapter):
    """Adapter for a locally running vLLM server (OpenAI-compatible API)."""

    def __init__(
        self,
        model: str,
        api_base: str = VLLM_DEFAULT_BASE,
        **kwargs: Any,
    ):
        config = AdapterConfig(
            model=f"openai/{model}",
            extra={"api_base": api_base, "api_key": "EMPTY", **kwargs},
        )
        super().__init__(config)
        self._inner = LiteLLMAdapter(config)

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AdapterResponse:
        return self._inner.complete(messages, tools=tools)


class OllamaAdapter(ModelAdapter):
    """Adapter for a locally running Ollama server."""

    def __init__(
        self,
        model: str,
        api_base: str = OLLAMA_DEFAULT_BASE,
        **kwargs: Any,
    ):
        config = AdapterConfig(
            model=f"ollama/{model}",
            extra={"api_base": api_base, **kwargs},
        )
        super().__init__(config)
        self._inner = LiteLLMAdapter(config)

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AdapterResponse:
        return self._inner.complete(messages, tools=tools)


def make_local_adapter(
    backend: str,
    model: str,
    api_base: str | None = None,
    **kwargs: Any,
) -> ModelAdapter:
    """Factory: backend ∈ {'vllm', 'ollama'}."""
    if backend == "vllm":
        return VLLMAdapter(model, api_base=api_base or VLLM_DEFAULT_BASE, **kwargs)
    elif backend == "ollama":
        return OllamaAdapter(model, api_base=api_base or OLLAMA_DEFAULT_BASE, **kwargs)
    else:
        raise ValueError(f"Unknown local backend: {backend!r}. Choose 'vllm' or 'ollama'.")
