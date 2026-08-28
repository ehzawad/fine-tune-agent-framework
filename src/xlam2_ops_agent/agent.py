from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .audit import AuditLogger, NullAuditLogger
from .policy import PolicyAction, PolicyEngine
from .protocol import parse_xlam_tool_calls
from .tools import ToolRegistry, ToolSpec
from .types import ChatClient, Message, ToolCall

ApprovalCallback = Callable[[ToolSpec[Any], BaseModel], bool]

DEFAULT_SYSTEM_PROMPT = """You are a policy-bound operations assistant.
You have access to tools. When using tools, emit the calls as one JSON array in this
exact shape: [{"name":"tool_name","arguments":{"argument":"value"}}]. Use parallel
entries when independent calls can run together. If required arguments are missing,
ask for them rather than inventing them. Do not interpret a proposed call as an
executed action: wait for the tool result. Never claim that a side effect succeeded
until a tool result confirms it. A tool result with status confirmation_required,
denied, dry_run, invalid_arguments, or error means the requested action did not
execute. For requests that need no tool, answer directly. Keep final answers concise
and state what was actually verified."""


@dataclass(slots=True)
class AgentResult:
    content: str
    messages: list[Message]
    steps: int
    stop_reason: str


class XlamAgent:
    def __init__(
        self,
        *,
        client: ChatClient,
        registry: ToolRegistry,
        policy: PolicyEngine | None = None,
        audit: AuditLogger | NullAuditLogger | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.registry = registry
        self.policy = policy or PolicyEngine()
        self.audit = audit or NullAuditLogger()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run_turn(
        self,
        user_input: str,
        *,
        history: Sequence[Message] | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> AgentResult:
        messages = [dict(message) for message in (history or [])]
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})

        call_counts: Counter[str] = Counter()
        tools = self.registry.as_openai_tools()

        for step in range(1, self.max_steps + 1):
            turn = self.client.complete(
                messages=messages,
                tools=tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            calls = turn.tool_calls
            if not calls:
                calls = parse_xlam_tool_calls(
                    turn.content,
                    allowed_names=self.registry.names,
                )

            if not calls:
                content = (turn.content or "").strip()
                messages.append({"role": "assistant", "content": content})
                return AgentResult(
                    content=content,
                    messages=messages,
                    steps=step,
                    stop_reason="final_response",
                )

            assistant_message: Message = {
                "role": "assistant",
                "content": turn.content if turn.tool_calls else None,
                "tool_calls": [call.as_openai_dict() for call in calls],
            }
            messages.append(assistant_message)

            for call in calls:
                signature = self._signature(call)
                call_counts[signature] += 1
                if call_counts[signature] > 2:
                    result = {
                        "status": "error",
                        "message": "duplicate tool-call limit exceeded",
                    }
                    self.audit.record(
                        "tool_call_blocked",
                        tool=call.name,
                        arguments=call.arguments,
                        reason="duplicate limit",
                    )
                    messages.append(self._tool_message(call, result))
                    continue

                result = self._process_call(call, approval_callback)
                messages.append(self._tool_message(call, result))

        content = "Stopped because the maximum number of model/tool steps was reached."
        messages.append({"role": "assistant", "content": content})
        return AgentResult(
            content=content,
            messages=messages,
            steps=self.max_steps,
            stop_reason="max_steps",
        )

    def _process_call(
        self,
        call: ToolCall,
        approval_callback: ApprovalCallback | None,
    ) -> dict[str, Any]:
        try:
            spec, validated = self.registry.validate(call.name, call.arguments)
        except KeyError as exc:
            result = {"status": "unknown_tool", "message": str(exc)}
            self.audit.record("tool_call_rejected", tool=call.name, result=result)
            return result
        except ValueError as exc:
            result = {"status": "invalid_arguments", "errors": exc.args[0]}
            self.audit.record(
                "tool_call_rejected",
                tool=call.name,
                arguments=call.arguments,
                result=result,
            )
            return result

        decision = self.policy.decide(spec, validated, approved=False)
        approved = False
        if decision.action == PolicyAction.REQUIRE_CONFIRMATION and approval_callback:
            approved = bool(approval_callback(spec, validated))
            decision = self.policy.decide(spec, validated, approved=approved)

        self.audit.record(
            "policy_decision",
            tool=spec.name,
            risk=spec.risk.value,
            arguments=validated.model_dump(mode="json"),
            action=decision.action.value,
            reason=decision.reason,
            approved=approved,
        )

        if decision.action == PolicyAction.REQUIRE_CONFIRMATION:
            return {
                "status": "confirmation_required",
                "tool": spec.name,
                "arguments": validated.model_dump(mode="json"),
                "message": decision.reason,
            }
        if decision.action == PolicyAction.DENY:
            return {"status": "denied", "message": decision.reason}
        if decision.action == PolicyAction.DRY_RUN:
            return {
                "status": "dry_run",
                "tool": spec.name,
                "arguments": validated.model_dump(mode="json"),
                "message": decision.reason,
            }

        execution = self.registry.execute_validated(spec, validated)
        self.audit.record(
            "tool_executed",
            tool=spec.name,
            arguments=validated.model_dump(mode="json"),
            ok=execution.ok,
            output=execution.output,
        )
        return execution.output

    @staticmethod
    def _tool_message(call: ToolCall, result: dict[str, Any]) -> Message:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        }

    @staticmethod
    def _signature(call: ToolCall) -> str:
        return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
