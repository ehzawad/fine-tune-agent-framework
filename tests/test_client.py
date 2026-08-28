import pytest

from xlam2_ops_agent.client import _parse_openai_tool_call


def test_parses_openai_tool_call_and_generates_missing_id() -> None:
    call = _parse_openai_tool_call(
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "arguments": '{"order_id":"ORD-1001"}',
            },
        }
    )
    assert call.id.startswith("call_")
    assert call.name == "get_order"
    assert call.arguments == {"order_id": "ORD-1001"}


def test_malformed_tool_arguments_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="Malformed JSON arguments"):
        _parse_openai_tool_call(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_order", "arguments": "{not-json"},
            }
        )


def test_non_object_tool_arguments_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="must decode to an object"):
        _parse_openai_tool_call(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_order", "arguments": "[]"},
            }
        )
