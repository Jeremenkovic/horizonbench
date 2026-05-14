"""LiteLLM-based model adapter — single interface across all vendors."""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from horizonbench.adapters.base import (
    AdapterConfig,
    AdapterResponse,
    Message,
    ModelAdapter,
    Usage,
)

logger = logging.getLogger(__name__)

# Silence LiteLLM's chatty success/warning logs by default.
litellm.suppress_debug_info = True


def _to_litellm_message(msg: Message) -> dict[str, Any]:
    """Convert our Message dataclass to the dict format LiteLLM expects."""
    d: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.name:
        d["name"] = msg.name
    return d


def _from_litellm_response(response: Any) -> AdapterResponse:
    """Convert a LiteLLM ModelResponse to our AdapterResponse."""
    choice = response.choices[0]
    raw_msg = choice.message

    tool_calls = []
    if hasattr(raw_msg, "tool_calls") and raw_msg.tool_calls:
        for tc in raw_msg.tool_calls:
            if isinstance(tc, dict):
                tool_calls.append(tc)
            else:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

    message = Message(
        role=raw_msg.role,
        content=raw_msg.content or "",
        tool_calls=tool_calls,
    )

    usage_data = response.usage if hasattr(response, "usage") else None
    usage = Usage(
        prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage_data, "total_tokens", 0) or 0,
        cost_usd=_estimate_cost(response),
    )

    finish_reason = choice.finish_reason or "stop"
    # Normalise OpenAI/Anthropic finish-reason variants
    if finish_reason in ("tool_calls", "tool_use"):
        finish_reason = "tool_calls"

    return AdapterResponse(
        message=message,
        usage=usage,
        finish_reason=finish_reason,
        raw=response.model_dump() if hasattr(response, "model_dump") else {},
    )


def _estimate_cost(response: Any) -> float:
    """Use LiteLLM's built-in cost estimator if available."""
    try:
        return litellm.completion_cost(completion_response=response)
    except Exception:
        return 0.0


class LiteLLMAdapter(ModelAdapter):
    """Wraps litellm.completion for unified multi-provider access."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AdapterResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_to_litellm_message(m) for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.extra,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("LiteLLM call: model=%s, messages=%d", self.config.model, len(messages))
        response = litellm.completion(**kwargs)
        return _from_litellm_response(response)
