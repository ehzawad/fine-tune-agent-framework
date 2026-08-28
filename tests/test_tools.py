from pathlib import Path

from xlam2_ops_agent.store import OrderStore
from xlam2_ops_agent.tools import build_order_registry


def test_transactional_cancel_and_idempotent_repeat(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "orders.db")
    store.initialize_demo(reset=True)
    registry = build_order_registry(store)
    spec, args = registry.validate(
        "cancel_order",
        {"order_id": "ord-1001", "reason": "Ordered by mistake"},
    )
    first = registry.execute_validated(spec, args)
    second = registry.execute_validated(spec, args)
    assert first.ok
    assert first.output["data"]["status"] == "cancelled"
    assert second.ok
    assert second.output["data"]["status"] == "already_cancelled"


def test_invalid_extra_argument_is_rejected(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "orders.db")
    store.initialize_demo(reset=True)
    registry = build_order_registry(store)
    try:
        registry.validate("get_order", {"order_id": "ORD-1001", "admin": True})
    except ValueError as exc:
        assert "extra_forbidden" in str(exc)
    else:
        raise AssertionError("extra argument should fail validation")


def test_refund_exact_retry_is_idempotent(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "orders.db")
    store.initialize_demo(reset=True)
    first = store.issue_refund("ORD-1001", 1000, "Courtesy refund")
    second = store.issue_refund("ORD-1001", 1000, "Courtesy refund")
    assert first["status"] == "refunded"
    assert second["status"] == "already_refunded"
    assert store.get_order("ORD-1001")["refunded_amount"] == "10.00"
