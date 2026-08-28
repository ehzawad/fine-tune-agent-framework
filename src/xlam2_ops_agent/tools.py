from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .store import OrderStore

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class ToolRisk(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class ToolSpec(Generic[ArgsT]):
    name: str
    description: str
    args_model: type[ArgsT]
    risk: ToolRisk
    handler: Callable[[ArgsT], Any]

    def as_openai_tool(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


@dataclass(slots=True)
class ToolExecution:
    ok: bool
    output: dict[str, Any]


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec[Any]]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Tool names must be unique")

    @property
    def names(self) -> set[str]:
        return set(self._specs)

    def get(self, name: str) -> ToolSpec[Any] | None:
        return self._specs.get(name)

    def as_openai_tools(self) -> list[dict[str, Any]]:
        return [spec.as_openai_tool() for spec in self._specs.values()]

    def validate(self, name: str, arguments: dict[str, Any]) -> tuple[ToolSpec[Any], BaseModel]:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool: {name}")
        try:
            validated = spec.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(exc.errors(include_url=False)) from exc
        return spec, validated

    def execute_validated(self, spec: ToolSpec[Any], arguments: BaseModel) -> ToolExecution:
        try:
            result = spec.handler(arguments)
        except Exception as exc:  # The boundary converts tool exceptions to structured results.
            return ToolExecution(
                ok=False,
                output={
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        return ToolExecution(ok=True, output={"status": "ok", "data": result})


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GetOrderArgs(StrictArgs):
    order_id: str = Field(description="Order identifier such as ORD-1001")

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, value: str) -> str:
        return value.upper()


class FindOrdersArgs(StrictArgs):
    customer_email: str = Field(description="Exact customer email address")


class CheckInventoryArgs(StrictArgs):
    sku: str = Field(description="Inventory SKU such as KB-75")

    @field_validator("sku")
    @classmethod
    def normalize_sku(cls, value: str) -> str:
        return value.upper()


class CancelOrderArgs(StrictArgs):
    order_id: str = Field(description="Order identifier such as ORD-1001")
    reason: str = Field(min_length=3, max_length=300, description="Customer-provided reason")

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, value: str) -> str:
        return value.upper()


class IssueRefundArgs(StrictArgs):
    order_id: str = Field(description="Order identifier such as ORD-1001")
    amount: Decimal = Field(gt=0, description="Refund amount in US dollars, at most two decimals")
    reason: str = Field(min_length=3, max_length=300)

    @field_validator("amount")
    @classmethod
    def validate_currency_scale(cls, value: Decimal) -> Decimal:
        if value != value.quantize(Decimal("0.01")):
            raise ValueError("amount must have at most two decimal places")
        return value

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, value: str) -> str:
        return value.upper()


def build_order_registry(store: OrderStore) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="get_order",
                description="Get one order by its order ID. This does not modify state.",
                args_model=GetOrderArgs,
                risk=ToolRisk.READ,
                handler=lambda args: store.get_order(args.order_id)
                or {"status": "not_found", "order_id": args.order_id},
            ),
            ToolSpec(
                name="find_orders",
                description="List orders for an exact customer email. This does not modify state.",
                args_model=FindOrdersArgs,
                risk=ToolRisk.READ,
                handler=lambda args: store.find_orders(args.customer_email),
            ),
            ToolSpec(
                name="check_inventory",
                description="Check current inventory for one SKU. This does not reserve stock.",
                args_model=CheckInventoryArgs,
                risk=ToolRisk.READ,
                handler=lambda args: store.check_inventory(args.sku)
                or {"status": "not_found", "sku": args.sku},
            ),
            ToolSpec(
                name="cancel_order",
                description=(
                    "Cancel an order that is still processing. This modifies state and requires "
                    "explicit authorization."
                ),
                args_model=CancelOrderArgs,
                risk=ToolRisk.WRITE,
                handler=lambda args: store.cancel_order(args.order_id, args.reason),
            ),
            ToolSpec(
                name="issue_refund",
                description=(
                    "Issue a refund in US dollars. This modifies state and requires explicit "
                    "authorization."
                ),
                args_model=IssueRefundArgs,
                risk=ToolRisk.WRITE,
                handler=lambda args: store.issue_refund(
                    args.order_id,
                    int(args.amount * 100),
                    args.reason,
                ),
            ),
        ]
    )
