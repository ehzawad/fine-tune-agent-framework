from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .agent import XlamAgent
from .audit import AuditLogger
from .client import VLLMChatClient
from .demo import run_offline_demo
from .policy import PolicyEngine
from .store import OrderStore
from .tools import ToolSpec, build_order_registry


def _defaults() -> dict[str, str]:
    return {
        "base_url": os.getenv("XLAM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "model": os.getenv("XLAM_MODEL", "xlam-2-32b-fc-r"),
        "api_key": os.getenv(
            "XLAM_API_KEY", os.getenv("VLLM_API_KEY", "local-token")
        ),
        "db": os.getenv("XLAM_DB_PATH", ".xlam2_ops/orders.db"),
        "audit": os.getenv("XLAM_AUDIT_PATH", ".xlam2_ops/audit.jsonl"),
    }


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    defaults = _defaults()
    parser.add_argument("--base-url", default=defaults["base_url"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--api-key", default=defaults["api_key"])
    parser.add_argument("--db", default=defaults["db"])
    parser.add_argument("--audit", default=defaults["audit"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--approve-writes",
        action="store_true",
        help="Approve model-proposed write tools without an interactive prompt.",
    )


def _build_agent(args: argparse.Namespace) -> tuple[XlamAgent, VLLMChatClient]:
    store = OrderStore(args.db)
    store.initialize_demo(reset=False)
    client = VLLMChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )
    policy = PolicyEngine(dry_run=args.dry_run)
    agent = XlamAgent(
        client=client,
        registry=build_order_registry(store),
        policy=policy,
        audit=AuditLogger(args.audit),
    )
    return agent, client


def _interactive_approval(spec: ToolSpec[Any], arguments: BaseModel) -> bool:
    print(f"\nProposed write: {spec.name}({arguments.model_dump(mode='json')})")
    answer = input("Execute it? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _approval_callback(args: argparse.Namespace, *, interactive: bool):
    if args.approve_writes:
        return lambda _spec, _arguments: True
    if interactive:
        return _interactive_approval
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xlam2-agent",
        description="Policy-bounded Salesforce xLAM-2 tool-calling reference runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Initialize or reset the demo SQLite data")
    init_db.add_argument("--db", default=_defaults()["db"])
    init_db.add_argument("--reset", action="store_true")

    run = subparsers.add_parser("run", help="Run one non-interactive user turn")
    _add_connection_args(run)
    run.add_argument("query")

    chat = subparsers.add_parser("chat", help="Open an interactive multi-turn session")
    _add_connection_args(chat)

    subparsers.add_parser("demo", help="Run deterministic offline proof without a model")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "init-db":
        store = OrderStore(args.db)
        store.initialize_demo(reset=args.reset)
        print(f"Initialized demo database at {Path(args.db).resolve()}")
        return

    if args.command == "demo":
        print(run_offline_demo())
        return

    agent, client = _build_agent(args)
    try:
        if args.command == "run":
            result = agent.run_turn(
                args.query,
                approval_callback=_approval_callback(args, interactive=False),
            )
            print(result.content)
            return

        history = None
        print("Interactive xLAM-2 operations session. Commands: :quit, :reset")
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input == ":quit":
                break
            if user_input == ":reset":
                history = None
                print("Conversation reset.")
                continue
            result = agent.run_turn(
                user_input,
                history=history,
                approval_callback=_approval_callback(args, interactive=True),
            )
            history = result.messages
            print(f"agent> {result.content}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
