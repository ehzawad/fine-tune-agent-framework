from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OrderStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize_demo(self, *, reset: bool = False) -> None:
        if reset and self.path.exists():
            self.path.unlink()
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    customer_email TEXT NOT NULL,
                    sku TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    status TEXT NOT NULL,
                    paid_amount_cents INTEGER NOT NULL CHECK (paid_amount_cents >= 0),
                    refunded_amount_cents INTEGER NOT NULL DEFAULT 0
                        CHECK (refunded_amount_cents >= 0),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inventory (
                    sku TEXT PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    available_units INTEGER NOT NULL CHECK (available_units >= 0)
                );

                CREATE TABLE IF NOT EXISTS refunds (
                    operation_key TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES orders(order_id),
                    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            if count == 0:
                conn.executemany(
                    """
                    INSERT INTO orders (
                        order_id, customer_name, customer_email, sku, quantity,
                        status, paid_amount_cents, refunded_amount_cents, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "ORD-1001",
                            "Alice Rahman",
                            "alice@example.com",
                            "KB-75",
                            1,
                            "processing",
                            12999,
                            0,
                            "2026-08-20T10:00:00Z",
                        ),
                        (
                            "ORD-1002",
                            "Bob Chen",
                            "bob@example.com",
                            "MON-27",
                            2,
                            "shipped",
                            59998,
                            0,
                            "2026-08-18T12:30:00Z",
                        ),
                        (
                            "ORD-1003",
                            "Priya Sen",
                            "priya@example.com",
                            "MOU-01",
                            1,
                            "processing",
                            4999,
                            0,
                            "2026-08-25T09:15:00Z",
                        ),
                    ],
                )
            inventory_count = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
            if inventory_count == 0:
                conn.executemany(
                    "INSERT INTO inventory (sku, product_name, available_units) VALUES (?, ?, ?)",
                    [
                        ("KB-75", "Mechanical Keyboard 75%", 23),
                        ("MON-27", "27-inch Monitor", 7),
                        ("MOU-01", "Ergonomic Mouse", 41),
                    ],
                )

    @staticmethod
    def _money(cents: int) -> str:
        return f"{cents / 100:.2f}"

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id.upper(),)
            ).fetchone()
        return self._row_to_order(row) if row else None

    def find_orders(self, customer_email: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE lower(customer_email) = lower(?) "
                "ORDER BY created_at DESC",
                (customer_email,),
            ).fetchall()
        return [self._row_to_order(row) for row in rows]

    def check_inventory(self, sku: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM inventory WHERE sku = ?", (sku.upper(),)
            ).fetchone()
        return dict(row) if row else None

    def cancel_order(self, order_id: str, reason: str) -> dict[str, Any]:
        order_id = order_id.upper()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown order: {order_id}")
            if row["status"] == "cancelled":
                return {
                    "status": "already_cancelled",
                    "order_id": order_id,
                    "reason": reason,
                }
            if row["status"] != "processing":
                raise ValueError(
                    f"Order {order_id} cannot be cancelled from status {row['status']!r}"
                )
            conn.execute(
                "UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,)
            )
        return {"status": "cancelled", "order_id": order_id, "reason": reason}

    def issue_refund(self, order_id: str, amount_cents: int, reason: str) -> dict[str, Any]:
        order_id = order_id.upper()
        reason = reason.strip()
        if amount_cents <= 0:
            raise ValueError("Refund amount must be positive")
        operation_key = hashlib.sha256(
            f"{order_id}|{amount_cents}|{reason.casefold()}".encode("utf-8")
        ).hexdigest()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT * FROM refunds WHERE operation_key = ?", (operation_key,)
            ).fetchone()
            if prior is not None:
                order = conn.execute(
                    "SELECT refunded_amount_cents FROM orders WHERE order_id = ?", (order_id,)
                ).fetchone()
                return {
                    "status": "already_refunded",
                    "order_id": order_id,
                    "amount": self._money(prior["amount_cents"]),
                    "total_refunded": self._money(order["refunded_amount_cents"]),
                    "reason": prior["reason"],
                    "operation_key": operation_key,
                }

            row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown order: {order_id}")
            remaining = row["paid_amount_cents"] - row["refunded_amount_cents"]
            if amount_cents > remaining:
                raise ValueError(
                    f"Refund exceeds remaining refundable amount {self._money(remaining)}"
                )
            new_total = row["refunded_amount_cents"] + amount_cents
            conn.execute(
                "UPDATE orders SET refunded_amount_cents = ? WHERE order_id = ?",
                (new_total, order_id),
            )
            conn.execute(
                """
                INSERT INTO refunds (operation_key, order_id, amount_cents, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (operation_key, order_id, amount_cents, reason, datetime.now(UTC).isoformat()),
            )
        return {
            "status": "refunded",
            "order_id": order_id,
            "amount": self._money(amount_cents),
            "total_refunded": self._money(new_total),
            "reason": reason,
            "operation_key": operation_key,
        }

    def _row_to_order(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "order_id": row["order_id"],
            "customer_name": row["customer_name"],
            "customer_email": row["customer_email"],
            "sku": row["sku"],
            "quantity": row["quantity"],
            "status": row["status"],
            "paid_amount": self._money(row["paid_amount_cents"]),
            "refunded_amount": self._money(row["refunded_amount_cents"]),
            "created_at": row["created_at"],
        }
