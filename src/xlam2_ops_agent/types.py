from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

JsonObject = dict[str, Any]
Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonObject
    raw_arguments: str

    def as_openai_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments,
            },
        }


@dataclass(slots=True)
class AssistantTurn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: JsonObject | None = None


class ChatClient(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[JsonObject],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AssistantTurn: ...
