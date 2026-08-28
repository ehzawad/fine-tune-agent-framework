from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from typing import Any

from .types import ToolCall

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_CALLS_TAG_RE = re.compile(r"\[TOOL_CALLS\]\s*(.*)", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _is_json_document_candidate(value: str) -> bool:
    stripped = value.lstrip()
    return bool(stripped) and stripped[0] in "[{"


def _candidate_texts(content: str) -> Iterable[str]:
    stripped = content.strip()
    if not stripped:
        return

    # Raw fallback execution is intentionally limited to output that is itself a JSON
    # document. A valid-looking call embedded in explanatory prose must remain prose.
    if _is_json_document_candidate(stripped):
        yield stripped

    without_think = _THINK_RE.sub("", stripped).strip()
    if (
        without_think
        and without_think != stripped
        and _is_json_document_candidate(without_think)
    ):
        yield without_think

    for regex in (_CODE_FENCE_RE, _TOOL_TAG_RE, _TOOL_CALLS_TAG_RE):
        for match in regex.finditer(stripped):
            candidate = match.group(1).strip()
            if candidate and _is_json_document_candidate(candidate):
                yield candidate


def _decode_json_document(candidate: str) -> Any | None:
    decoder = json.JSONDecoder()
    candidate = candidate.lstrip()
    try:
        value, end = decoder.raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    if candidate[end:].strip():
        return None
    return value


def _normalize_arguments(value: Any) -> tuple[dict[str, Any], str] | None:
    if isinstance(value, dict):
        return value, json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            return decoded, json.dumps(decoded, separators=(",", ":"), ensure_ascii=False)
    return None


def parse_xlam_tool_calls(
    content: str | None,
    *,
    allowed_names: set[str] | None = None,
) -> list[ToolCall]:
    """Parse xLAM's raw JSON tool-call output conservatively.

    The parser accepts a direct JSON document and wrappers supported by current
    vLLM's xLAM parser. It returns no calls unless every item has a known name and
    object-shaped arguments. A valid tool-call object embedded in ordinary prose is
    deliberately not executable.
    """

    if not content:
        return []

    for candidate in _candidate_texts(content):
        value = _decode_json_document(candidate)
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not value:
            continue

        calls: list[ToolCall] = []
        valid = True
        for item in value:
            if not isinstance(item, dict):
                valid = False
                break
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(name, str) or not name:
                valid = False
                break
            if allowed_names is not None and name not in allowed_names:
                valid = False
                break
            normalized = _normalize_arguments(arguments)
            if normalized is None:
                valid = False
                break
            args_dict, raw_args = normalized
            call_id = str(item.get("id") or f"call_{uuid.uuid4().hex[:24]}")
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=args_dict,
                    raw_arguments=raw_args,
                )
            )
        if valid and calls:
            return calls
    return []
