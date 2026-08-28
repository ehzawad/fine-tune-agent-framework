from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import httpx

from .types import AssistantTurn, JsonObject, Message, ToolCall


def _parse_openai_tool_call(raw_call: Mapping[str, Any]) -> ToolCall:
    function = raw_call.get("function")
    if not isinstance(function, Mapping):
        raise RuntimeError(f"Malformed tool call from vLLM: missing function object: {raw_call!r}")

    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Malformed tool call from vLLM: missing function name: {raw_call!r}")

    raw_arguments = function.get("arguments", "{}")
    if isinstance(raw_arguments, Mapping):
        arguments = dict(raw_arguments)
        serialized_arguments = json.dumps(
            arguments,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    elif isinstance(raw_arguments, str):
        serialized_arguments = raw_arguments
        try:
            decoded = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSON arguments for tool {name!r}: {raw_arguments!r}"
            ) from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"Tool arguments for {name!r} must decode to an object, got "
                f"{type(decoded).__name__}"
            )
        arguments = decoded
    else:
        raise RuntimeError(
            f"Tool arguments for {name!r} must be a JSON string or object, got "
            f"{type(raw_arguments).__name__}"
        )

    call_id = raw_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex[:24]}"
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=serialized_arguments,
    )


class VLLMChatClient:
    """Minimal client for vLLM's OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local-token",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> VLLMChatClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[JsonObject],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AssistantTurn:
        payload: JsonObject = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self._http.post(f"{self.base_url}/chat/completions", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:2000]
            raise RuntimeError(f"vLLM request failed ({response.status_code}): {body}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"vLLM returned non-JSON content ({response.status_code}): "
                f"{response.text[:2000]}"
            ) from exc
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat-completions response: {data!r}") from exc
        if not isinstance(message, Mapping):
            raise RuntimeError(f"Unexpected assistant message: {message!r}")

        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise RuntimeError(f"Unexpected tool_calls payload: {raw_tool_calls!r}")
        tool_calls = [
            _parse_openai_tool_call(raw_call)
            for raw_call in raw_tool_calls
            if isinstance(raw_call, Mapping)
        ]
        if len(tool_calls) != len(raw_tool_calls):
            raise RuntimeError(f"Malformed tool_calls payload: {raw_tool_calls!r}")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise RuntimeError(f"Unexpected assistant content type: {type(content).__name__}")
        return AssistantTurn(
            content=content,
            tool_calls=tool_calls,
            raw=data,
        )


class ScriptedChatClient:
    """Deterministic model adapter used by tests and the offline proof."""

    def __init__(self, turns: Iterable[AssistantTurn]) -> None:
        self._turns = deque(turns)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[JsonObject],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AssistantTurn:
        self.requests.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._turns:
            raise RuntimeError("ScriptedChatClient has no remaining turns")
        return self._turns.popleft()
