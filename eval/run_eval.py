#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from xlam2_ops_agent.client import VLLMChatClient
from xlam2_ops_agent.store import OrderStore
from xlam2_ops_agent.tools import build_order_registry


def arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Starter xLAM tool-selection evaluation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=os.getenv("XLAM_MODEL", "xlam-2-32b-fc-r"))
    parser.add_argument(
        "--api-key", default=os.getenv("VLLM_API_KEY", "local-token")
    )
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    registry = build_order_registry(OrderStore(":memory:"))
    tools = registry.as_openai_tools()
    passed = 0

    with VLLMChatClient(
        base_url=args.base_url, model=args.model, api_key=args.api_key
    ) as client:
        for case in cases:
            turn = client.complete(
                messages=[{"role": "user", "content": case["prompt"]}],
                tools=tools,
                temperature=0.0,
            )
            expected_tool = case.get("expected_tool")
            ok = False
            if expected_tool is None:
                text = (turn.content or "").lower()
                needles = [item.lower() for item in case.get("content_should_contain_any", [])]
                ok = not turn.tool_calls and (not needles or any(item in text for item in needles))
            elif turn.tool_calls:
                first = turn.tool_calls[0]
                ok = first.name == expected_tool and arguments_match(
                    first.arguments, case.get("expected_arguments", {})
                )
            passed += int(ok)
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "passed": ok,
                        "content": turn.content,
                        "tool_calls": [
                            {"name": call.name, "arguments": call.arguments}
                            for call in turn.tool_calls
                        ],
                    },
                    ensure_ascii=False,
                )
            )

    print(f"score={passed}/{len(cases)} ({passed / len(cases):.1%})")
    raise SystemExit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
