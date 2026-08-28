from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .agent import XlamAgent
from .audit import AuditLogger
from .client import ScriptedChatClient
from .policy import PolicyEngine
from .store import OrderStore
from .tools import build_order_registry
from .types import AssistantTurn, ToolCall


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments, separators=(",", ":")),
    )


def run_offline_demo() -> str:
    with tempfile.TemporaryDirectory(prefix="xlam2-demo-") as temp_dir:
        base = Path(temp_dir)
        store = OrderStore(base / "orders.db")
        store.initialize_demo(reset=True)
        client = ScriptedChatClient(
            [
                AssistantTurn(tool_calls=[_call("call_1", "get_order", {"order_id": "ORD-1001"})]),
                AssistantTurn(
                    tool_calls=[
                        _call(
                            "call_2",
                            "cancel_order",
                            {"order_id": "ORD-1001", "reason": "Ordered by mistake"},
                        )
                    ]
                ),
                AssistantTurn(
                    content=(
                        "ORD-1001 was verified as processing and then cancelled after explicit "
                        "approval."
                    )
                ),
            ]
        )
        agent = XlamAgent(
            client=client,
            registry=build_order_registry(store),
            policy=PolicyEngine(require_write_confirmation=True),
            audit=AuditLogger(base / "audit.jsonl"),
        )
        result = agent.run_turn(
            "Please verify and cancel ORD-1001; I ordered it by mistake.",
            approval_callback=lambda _spec, _args: True,
        )
        final_order = store.get_order("ORD-1001")
        assert final_order and final_order["status"] == "cancelled"
        audit_lines = (base / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        return (
            f"{result.content}\n"
            f"steps={result.steps}; final_status={final_order['status']}; "
            f"audit_events={len(audit_lines)}"
        )
