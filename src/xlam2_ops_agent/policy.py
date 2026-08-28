from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .tools import IssueRefundArgs, ToolRisk, ToolSpec


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DRY_RUN = "dry_run"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str


class PolicyEngine:
    def __init__(
        self,
        *,
        require_write_confirmation: bool = True,
        max_refund_cents: int = 50_000,
        dry_run: bool = False,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.require_write_confirmation = require_write_confirmation
        self.max_refund_cents = max_refund_cents
        self.dry_run = dry_run
        self.allowed_tools = allowed_tools

    def decide(
        self,
        spec: ToolSpec[Any],
        args: BaseModel,
        *,
        approved: bool = False,
    ) -> PolicyDecision:
        if self.allowed_tools is not None and spec.name not in self.allowed_tools:
            return PolicyDecision(PolicyAction.DENY, "tool is not in the deployment allowlist")

        if isinstance(args, IssueRefundArgs):
            cents = int(args.amount * 100)
            if cents > self.max_refund_cents:
                return PolicyDecision(
                    PolicyAction.DENY,
                    f"refund exceeds the policy limit of {self.max_refund_cents / 100:.2f}",
                )

        if self.dry_run and spec.risk == ToolRisk.WRITE:
            return PolicyDecision(PolicyAction.DRY_RUN, "deployment is in dry-run mode")

        if spec.risk == ToolRisk.WRITE and self.require_write_confirmation and not approved:
            return PolicyDecision(
                PolicyAction.REQUIRE_CONFIRMATION,
                "write tools require an explicit approval outside the model",
            )

        return PolicyDecision(PolicyAction.ALLOW, "policy checks passed")
