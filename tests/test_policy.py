from xlam2_ops_agent.policy import PolicyAction, PolicyEngine
from xlam2_ops_agent.store import OrderStore
from xlam2_ops_agent.tools import CancelOrderArgs, IssueRefundArgs, build_order_registry


def _spec(name: str):
    store = OrderStore(":memory:")
    registry = build_order_registry(store)
    found = registry.get(name)
    assert found is not None
    return found


def test_read_is_allowed() -> None:
    spec = _spec("get_order")
    args = spec.args_model.model_validate({"order_id": "ORD-1001"})
    assert PolicyEngine().decide(spec, args).action == PolicyAction.ALLOW


def test_write_requires_confirmation() -> None:
    spec = _spec("cancel_order")
    args = CancelOrderArgs(order_id="ORD-1001", reason="Ordered by mistake")
    policy = PolicyEngine()
    assert policy.decide(spec, args).action == PolicyAction.REQUIRE_CONFIRMATION
    assert policy.decide(spec, args, approved=True).action == PolicyAction.ALLOW


def test_refund_limit_is_deterministic() -> None:
    spec = _spec("issue_refund")
    args = IssueRefundArgs(order_id="ORD-1001", amount=600, reason="Test refund")
    decision = PolicyEngine(max_refund_cents=50_000).decide(spec, args, approved=True)
    assert decision.action == PolicyAction.DENY
