from pathlib import Path
from typing import Any

import pytest

from xlam2_ops_agent.training.data import (
    TrajectoryValidationError,
    load_jsonl,
    render_assistant_examples,
    tokenize_completion_only,
    validate_trajectories,
)


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        chat_template: str,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert tools
        assert chat_template
        parts = ["SYSTEM:tools\n"]
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                value = str(message["tool_calls"])
            else:
                value = str(message.get("content", ""))
            parts.append(f"{role.upper()}:{value}<END>\n")
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "".join(parts)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        assert not add_special_tokens
        assert not truncation
        input_ids = [ord(character) for character in text]
        output: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


def test_demo_trajectories_are_well_formed() -> None:
    rows = load_jsonl(ROOT / "data" / "demo_train.jsonl")
    normalized, stats = validate_trajectories(rows)
    assert len(normalized) == 6
    assert stats.trajectories == 6
    assert stats.assistant_targets == 12
    assert stats.tool_call_targets == 6
    assert stats.text_targets == 6


def test_orphan_tool_result_is_rejected() -> None:
    row = {
        "tools": [
            {
                "name": "lookup",
                "description": "Lookup an item.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "tool_calls": [{"name": "lookup", "arguments": {}}],
            },
            {"role": "tool", "name": "lookup", "content": {"ok": True}},
            {"role": "tool", "name": "lookup", "content": {"ok": True}},
        ],
    }
    with pytest.raises(TrajectoryValidationError, match="expected assistant"):
        validate_trajectories([row])


def test_rendering_creates_one_target_per_assistant_turn() -> None:
    rows = load_jsonl(ROOT / "data" / "demo_eval.jsonl")
    normalized, stats = validate_trajectories(rows)
    rendered = render_assistant_examples(
        CharacterTokenizer(), normalized, chat_template="tool_calls template"
    )
    assert len(rendered) == stats.assistant_targets == 3
    assert {row["target_kind"] for row in rendered} == {"tool_calls", "text"}
    assert all(row["completion"] for row in rendered)


def test_completion_only_labels_mask_prompt_and_keep_target() -> None:
    tokenized = tokenize_completion_only(
        CharacterTokenizer(),
        prompt="abcdefghij",
        completion="TARGET",
        max_seq_length=10,
    )
    assert len(tokenized["input_ids"]) == 10
    assert tokenized["prompt_tokens"] == 4
    assert tokenized["target_tokens"] == 6
    assert tokenized["labels"][:4] == [-100] * 4
    assert tokenized["labels"][4:] == tokenized["input_ids"][4:]


def test_completion_is_never_silently_truncated() -> None:
    with pytest.raises(TrajectoryValidationError, match="targets are never truncated"):
        tokenize_completion_only(
            CharacterTokenizer(),
            prompt="x",
            completion="too-long",
            max_seq_length=8,
        )
