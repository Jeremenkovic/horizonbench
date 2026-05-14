"""Adapter base class — defines the interface every model adapter must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AdapterResponse:
    message: Message
    usage: Usage
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length" | "error"
    raw: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Unified interface for model inference across all providers."""

    def __init__(self, config: AdapterConfig):
        self.config = config

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AdapterResponse:
        """Send messages to the model and return its response."""

    def run_tool_loop(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        tool_handler: "ToolHandler",
        max_turns: int = 50,
    ) -> tuple[list[Message], Usage]:
        """Drive the full tool-call loop until the model stops or max_turns is reached.

        Returns the full updated message list and aggregated usage.
        """
        total_usage = Usage()
        for _ in range(max_turns):
            response = self.complete(messages, tools=tools)
            total_usage = Usage(
                prompt_tokens=total_usage.prompt_tokens + response.usage.prompt_tokens,
                completion_tokens=total_usage.completion_tokens + response.usage.completion_tokens,
                total_tokens=total_usage.total_tokens + response.usage.total_tokens,
                cost_usd=total_usage.cost_usd + response.usage.cost_usd,
            )
            messages.append(response.message)

            if response.finish_reason != "tool_calls" or not response.message.tool_calls:
                break

            # Dispatch each tool call
            for tc in response.message.tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                tool_args_raw = tc.get("function", {}).get("arguments", "{}")
                import json
                try:
                    tool_args = json.loads(tool_args_raw)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}
                result = tool_handler(tool_name, tool_args)
                messages.append(
                    Message(
                        role="tool",
                        content=str(result),
                        tool_call_id=tc.get("id", ""),
                    )
                )

        return messages, total_usage


# Type alias for tool handlers
ToolHandler = Any  # Callable[[str, dict], str]
