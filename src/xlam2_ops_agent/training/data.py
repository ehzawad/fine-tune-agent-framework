from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


class TrajectoryValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DataStats:
    trajectories: int
    assistant_targets: int
    tool_call_targets: int
    text_targets: int
    max_messages: int
    system_message_trajectories: int

    def as_dict(self) -> dict[str, int]:
        return {
            "trajectories": self.trajectories,
            "assistant_targets": self.assistant_targets,
            "tool_call_targets": self.tool_call_targets,
            "text_targets": self.text_targets,
            "max_messages": self.max_messages,
            "system_message_trajectories": self.system_message_trajectories,
        }


def load_jsonl(path: str | Path) -> list[JsonObject]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file not found: {source}")
    rows: list[JsonObject] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryValidationError(
                f"{source}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise TrajectoryValidationError(
                f"{source}:{line_number}: each JSONL row must be an object"
            )
        value.setdefault("_source", f"{source}:{line_number}")
        rows.append(value)
    if not rows:
        raise TrajectoryValidationError(f"No trajectories found in {source}")
    return rows


def _normalize_arguments(value: Any, *, where: str) -> JsonObject:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TrajectoryValidationError(
                f"{where}: tool arguments string is not valid JSON"
            ) from exc
        if isinstance(decoded, dict):
            return decoded
    raise TrajectoryValidationError(f"{where}: tool arguments must be a JSON object")


def canonicalize_tools(raw_tools: Any, *, where: str) -> list[JsonObject]:
    if not isinstance(raw_tools, list) or not raw_tools:
        raise TrajectoryValidationError(f"{where}: tools must be a non-empty list")
    tools: list[JsonObject] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_tools):
        tool_where = f"{where}.tools[{index}]"
        if not isinstance(raw, dict):
            raise TrajectoryValidationError(f"{tool_where}: tool must be an object")
        function = raw.get("function") if raw.get("type") == "function" else raw
        if not isinstance(function, dict):
            raise TrajectoryValidationError(f"{tool_where}: function definition is missing")
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name.strip():
            raise TrajectoryValidationError(f"{tool_where}: name must be a non-empty string")
        if name in names:
            raise TrajectoryValidationError(f"{tool_where}: duplicate tool name {name!r}")
        if not isinstance(description, str) or not description.strip():
            raise TrajectoryValidationError(
                f"{tool_where}: description must be a non-empty string"
            )
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise TrajectoryValidationError(
                f"{tool_where}: parameters must be an object-shaped JSON Schema"
            )
        names.add(name)
        tools.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return tools


def normalize_message(raw: Any, *, where: str, tool_names: set[str]) -> JsonObject:
    if not isinstance(raw, dict):
        raise TrajectoryValidationError(f"{where}: message must be an object")
    role = raw.get("role")
    if role not in _ALLOWED_ROLES:
        raise TrajectoryValidationError(
            f"{where}: role must be one of {sorted(_ALLOWED_ROLES)}, got {role!r}"
        )

    message: JsonObject = {"role": role}
    if role == "assistant" and raw.get("tool_calls"):
        calls = raw["tool_calls"]
        if not isinstance(calls, list):
            raise TrajectoryValidationError(f"{where}.tool_calls must be a list")
        normalized_calls: list[JsonObject] = []
        for index, raw_call in enumerate(calls):
            call_where = f"{where}.tool_calls[{index}]"
            if not isinstance(raw_call, dict):
                raise TrajectoryValidationError(f"{call_where}: call must be an object")
            function = raw_call.get("function", raw_call)
            if not isinstance(function, dict):
                raise TrajectoryValidationError(f"{call_where}: function is missing")
            name = function.get("name")
            if not isinstance(name, str) or name not in tool_names:
                raise TrajectoryValidationError(
                    f"{call_where}: unknown tool name {name!r}; declared={sorted(tool_names)}"
                )
            arguments = _normalize_arguments(
                function.get("arguments", {}), where=f"{call_where}.function.arguments"
            )
            normalized_calls.append(
                {
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        message["tool_calls"] = normalized_calls
        if raw.get("content") is not None:
            message["content"] = str(raw["content"])
        return message

    content = raw.get("content")
    if content is None:
        raise TrajectoryValidationError(f"{where}: content is required for role {role!r}")
    if role == "tool" and isinstance(content, (dict, list)):
        message["content"] = content
    elif isinstance(content, str):
        message["content"] = content
    else:
        message["content"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    for key in ("name", "tool_call_id"):
        if key in raw:
            message[key] = str(raw[key])
    if role == "tool":
        name = message.get("name")
        if not isinstance(name, str) or name not in tool_names:
            raise TrajectoryValidationError(
                f"{where}: tool result requires a declared tool name; got {name!r}"
            )
    return message


def _validate_message_sequence(messages: Sequence[Mapping[str, Any]], *, source: str) -> None:
    index = 1 if messages[0]["role"] == "system" else 0
    expected = "user"
    pending_tools: list[str] = []

    for message_index in range(index, len(messages)):
        message = messages[message_index]
        role = str(message["role"])
        where = f"{source}.messages[{message_index}]"

        if expected == "user":
            if role != "user":
                raise TrajectoryValidationError(f"{where}: expected user, got {role!r}")
            expected = "assistant"
            continue

        if expected == "assistant":
            if role != "assistant":
                raise TrajectoryValidationError(f"{where}: expected assistant, got {role!r}")
            calls = message.get("tool_calls") or []
            if calls:
                pending_tools = [str(call["function"]["name"]) for call in calls]
                expected = "tool"
            else:
                expected = "user"
            continue

        if expected == "tool":
            if role != "tool":
                raise TrajectoryValidationError(
                    f"{where}: expected one of the pending tool results "
                    f"{pending_tools}, got {role!r}"
                )
            name = str(message["name"])
            if name not in pending_tools:
                raise TrajectoryValidationError(
                    f"{where}: result for {name!r} does not match pending calls {pending_tools}"
                )
            pending_tools.remove(name)
            expected = "tool" if pending_tools else "assistant"
            continue

        raise AssertionError(f"unexpected sequence state: {expected}")

    if expected == "assistant":
        raise TrajectoryValidationError(f"{source}: trajectory ends before an assistant response")
    if expected == "tool":
        raise TrajectoryValidationError(
            f"{source}: trajectory ends before tool results for {pending_tools}"
        )


def validate_trajectory(raw: Mapping[str, Any], *, index: int = 0) -> JsonObject:
    source = str(raw.get("_source") or raw.get("id") or f"trajectory[{index}]")
    tools = canonicalize_tools(raw.get("tools"), where=source)
    tool_names = {tool["name"] for tool in tools}
    raw_messages = raw.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise TrajectoryValidationError(f"{source}: messages must be a non-empty list")
    messages = [
        normalize_message(message, where=f"{source}.messages[{i}]", tool_names=tool_names)
        for i, message in enumerate(raw_messages)
    ]
    if messages[0]["role"] not in {"system", "user"}:
        raise TrajectoryValidationError(
            f"{source}: first message must be system or user, got {messages[0]['role']!r}"
        )
    if not any(message["role"] == "assistant" for message in messages):
        raise TrajectoryValidationError(f"{source}: at least one assistant message is required")
    for i, message in enumerate(messages):
        if message["role"] == "system" and i != 0:
            raise TrajectoryValidationError(f"{source}: system message is only valid at index 0")
    _validate_message_sequence(messages, source=source)
    return {
        "id": str(raw.get("id") or f"trajectory-{index:06d}"),
        "tools": tools,
        "messages": messages,
        "source": source,
    }


def validate_trajectories(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonObject], DataStats]:
    normalized = [validate_trajectory(row, index=i) for i, row in enumerate(rows)]
    assistant_targets = 0
    tool_call_targets = 0
    text_targets = 0
    max_messages = 0
    system_count = 0
    for trajectory in normalized:
        messages = trajectory["messages"]
        max_messages = max(max_messages, len(messages))
        system_count += int(messages[0]["role"] == "system")
        for message in messages:
            if message["role"] != "assistant":
                continue
            assistant_targets += 1
            if message.get("tool_calls"):
                tool_call_targets += 1
            else:
                text_targets += 1
    return normalized, DataStats(
        trajectories=len(normalized),
        assistant_targets=assistant_targets,
        tool_call_targets=tool_call_targets,
        text_targets=text_targets,
        max_messages=max_messages,
        system_message_trajectories=system_count,
    )


def read_template(path: str | Path) -> str:
    template_path = Path(path)
    if not template_path.is_file():
        raise FileNotFoundError(f"Chat template not found: {template_path}")
    value = template_path.read_text(encoding="utf-8")
    if "<|im_start|>" not in value or "tool_calls" not in value:
        raise ValueError(f"Unexpected xLAM chat template contents: {template_path}")
    return value


def render_assistant_examples(
    tokenizer: Any,
    trajectories: Sequence[Mapping[str, Any]],
    *,
    chat_template: str,
) -> list[JsonObject]:
    rendered: list[JsonObject] = []
    for trajectory in trajectories:
        messages = list(trajectory["messages"])
        tools = list(trajectory["tools"])
        for target_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prefix_messages = messages[:target_index]
            full_messages = messages[: target_index + 1]
            prompt = tokenizer.apply_chat_template(
                prefix_messages,
                tools=tools,
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=True,
            )
            full = tokenizer.apply_chat_template(
                full_messages,
                tools=tools,
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not isinstance(prompt, str) or not isinstance(full, str):
                raise RuntimeError("Tokenizer chat template did not return text")
            if not full.startswith(prompt):
                raise TrajectoryValidationError(
                    f"{trajectory['source']}: rendered assistant target is not a prompt prefix; "
                    "the tokenizer/template contract changed"
                )
            completion = full[len(prompt) :]
            if not completion:
                raise TrajectoryValidationError(
                    f"{trajectory['source']}: assistant target rendered to an empty completion"
                )
            rendered.append(
                {
                    "trajectory_id": trajectory["id"],
                    "source": trajectory["source"],
                    "assistant_index": target_index,
                    "target_kind": "tool_calls" if message.get("tool_calls") else "text",
                    "prompt": prompt,
                    "completion": completion,
                }
            )
    if not rendered:
        raise TrajectoryValidationError("No assistant targets were rendered")
    return rendered


def _encode_with_offsets(
    tokenizer: Any, text: str
) -> tuple[list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError, ValueError):
        return None
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(input_ids, list) or not isinstance(offsets, list):
        return None
    return list(input_ids), [tuple(pair) for pair in offsets]


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for first, second in zip(left, right, strict=False):
        if first != second:
            break
        count += 1
    return count


def tokenize_completion_only(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    max_seq_length: int,
) -> JsonObject:
    full_text = prompt + completion
    with_offsets = _encode_with_offsets(tokenizer, full_text)
    if with_offsets is not None:
        input_ids, offsets = with_offsets
        boundary = next(
            (index for index, (_start, end) in enumerate(offsets) if end > len(prompt)),
            len(input_ids),
        )
    else:
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        input_ids = list(full_ids)
        boundary = _longest_common_prefix(prompt_ids, full_ids)
        if boundary < len(prompt_ids):
            boundary = max(0, boundary - 1)

    if boundary >= len(input_ids):
        raise TrajectoryValidationError("Completion produced no trainable tokens")

    target_length = len(input_ids) - boundary
    if target_length >= max_seq_length:
        raise TrajectoryValidationError(
            f"Assistant completion is {target_length} tokens, which does not fit in "
            f"max_seq_length={max_seq_length}; targets are never truncated"
        )

    if len(input_ids) > max_seq_length:
        prompt_tokens_to_keep = max_seq_length - target_length
        start = boundary - prompt_tokens_to_keep
        input_ids = input_ids[start:]
        boundary = prompt_tokens_to_keep

    labels = [-100] * boundary + input_ids[boundary:]
    if all(label == -100 for label in labels):
        raise TrajectoryValidationError("All labels are masked")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
        "prompt_tokens": boundary,
        "target_tokens": len(input_ids) - boundary,
    }


def build_tokenized_dataset(
    tokenizer: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    chat_template: str,
    max_seq_length: int,
    num_proc: int = 1,
    overwrite_cache: bool = False,
) -> tuple[Any, DataStats]:
    normalized, stats = validate_trajectories(rows)
    rendered = render_assistant_examples(tokenizer, normalized, chat_template=chat_template)

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies with .[train]") from exc

    dataset = Dataset.from_list(rendered)

    def tokenize_row(row: Mapping[str, Any]) -> JsonObject:
        return tokenize_completion_only(
            tokenizer,
            prompt=str(row["prompt"]),
            completion=str(row["completion"]),
            max_seq_length=max_seq_length,
        )

    dataset = dataset.map(
        tokenize_row,
        num_proc=num_proc if num_proc > 1 else None,
        remove_columns=dataset.column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing assistant-only xLAM targets",
    )
    return dataset, stats


def iter_validation_errors(rows: Iterable[Mapping[str, Any]]) -> Iterable[str]:
    for index, row in enumerate(rows):
        try:
            validate_trajectory(row, index=index)
        except (TrajectoryValidationError, TypeError, ValueError) as exc:
            yield str(exc)
