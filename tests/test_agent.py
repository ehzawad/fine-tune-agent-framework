import json
from pathlib import Path

from xlam2_ops_agent.agent import XlamAgent
from xlam2_ops_agent.audit import AuditLogger
from xlam2_ops_agent.client import ScriptedChatClient
from xlam2_ops_agent.policy import PolicyEngine
from xlam2_ops_agent.store import OrderStore
from xlam2_ops_agent.tools import build_order_registry
from xlam2_ops_agent.types import AssistantTurn, ToolCall


def call(call_id: str, name: str, args: dict[str, object]) -> ToolCall:
    return ToolCall(call_id, name, args, json.dumps(args))


def test_agent_executes_read_then_returns_answer(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "orders.db")
    store.initialize_demo(reset=True)
    client = ScriptedChatClient(
        [
            AssistantTurn(tool_calls=[call("c1", "get_order", {"order_id": "ORD-1001"})]),
            AssistantTurn(content="ORD-1001 is processing."),
        ]
    )
    agent = XlamAgent(client=client, registry=build_order_registry(store))
    result = agent.run_turn("What is the status of ORD-1001?")
    assert result.content == "ORD-1001 is processing."
    assert any(message.get("role") == "tool" for message in result.messages)


def test_write_without_approval_does_not_execute(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "orders.db")
    store.initialize_demo(reset=True)
    client = ScriptedChatClient(
        [
            AssistantTurn(
                tool_calls=[
                    call(
                        "c1",
                        "cancel_order",
                        {"order_id": "ORD-1001", "reason": "Ordered by mistake"},
                    )
                ]
            ),
            AssistantTurn(content="The cancellation requires confirmation and was not executed."),
        ]
    )
    agent = XlamAgent(client=client, registry=build_order_registry(store))
    result = agent.run_turn("Cancel ORD-1001")
    assert "not executed" in result.content
    assert store.get_order("ORD-1001")["status"] == "processing"


def test_write_with_external_approval_executes_and_audits(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "orders.db")
    store.initialize_demo(reset=True)
    audit_path = tmp_path / "audit.jsonl"
    client = ScriptedChatClient(
        [
            AssistantTurn(
                tool_calls=[
                    call(
                        "c1",
                        "cancel_order",
                        {"order_id": "ORD-1001", "reason": "Ordered by mistake"},
                    )
                ]
            ),
            AssistantTurn(content="The order was cancelled."),
        ]
    )
    agent = XlamAgent(
        client=client,
        registry=build_order_registry(store),
        policy=PolicyEngine(),
        audit=AuditLogger(audit_path),
    )
    result = agent.run_turn("Cancel ORD-1001", approval_callback=lambda _s, _a: True)
    assert result.content == "The order was cancelled."
    assert store.get_order("ORD-1001")["status"] == "cancelled"
    events = [json.loads(line)["event"] for line in audit_path.read_text().splitlines()]
    assert "policy_decision" in events
    assert "tool_executed" in events


def test_raw_xlam_content_fallback_executes(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "orders.db")
    store.initialize_demo(reset=True)
    client = ScriptedChatClient(
        [
            AssistantTurn(
                content='[{"name":"check_inventory","arguments":{"sku":"KB-75"}}]'
            ),
            AssistantTurn(content="There are 23 units available."),
        ]
    )
    agent = XlamAgent(client=client, registry=build_order_registry(store))
    result = agent.run_turn("Check KB-75")
    assert result.content == "There are 23 units available."
