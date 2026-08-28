from xlam2_ops_agent.protocol import parse_xlam_tool_calls


def test_direct_json_array() -> None:
    calls = parse_xlam_tool_calls(
        '[{"name":"get_order","arguments":{"order_id":"ORD-1001"}}]',
        allowed_names={"get_order"},
    )
    assert len(calls) == 1
    assert calls[0].name == "get_order"
    assert calls[0].arguments == {"order_id": "ORD-1001"}


def test_wrapped_json() -> None:
    calls = parse_xlam_tool_calls(
        '<think>Need a lookup.</think><tool_call>[{"name":"get_order",'
        '"arguments":{"order_id":"ORD-1001"}}]</tool_call>',
        allowed_names={"get_order"},
    )
    assert calls[0].name == "get_order"


def test_unknown_name_is_not_executable() -> None:
    calls = parse_xlam_tool_calls(
        '[{"name":"delete_everything","arguments":{}}]',
        allowed_names={"get_order"},
    )
    assert calls == []


def test_normal_json_prose_is_left_alone() -> None:
    calls = parse_xlam_tool_calls(
        'Here is a list: [1, 2, 3].',
        allowed_names={"get_order"},
    )
    assert calls == []


def test_valid_tool_json_embedded_in_prose_is_not_executable() -> None:
    calls = parse_xlam_tool_calls(
        'I would call [{"name":"get_order","arguments":{"order_id":"ORD-1001"}}].',
        allowed_names={"get_order"},
    )
    assert calls == []


def test_trailing_prose_after_raw_json_is_not_executable() -> None:
    calls = parse_xlam_tool_calls(
        '[{"name":"get_order","arguments":{"order_id":"ORD-1001"}}] maybe',
        allowed_names={"get_order"},
    )
    assert calls == []
