#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from xlam2_ops_agent.client import VLLMChatClient

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check current inventory for one SKU.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    }
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one xLAM tool-call smoke request")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="xlam-2-32b-fc-r")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "local-token"))
    args = parser.parse_args()

    with VLLMChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    ) as client:
        turn = client.complete(
            messages=[{"role": "user", "content": "Check inventory for SKU KB-75."}],
            tools=TOOLS,
            temperature=0.0,
        )
    print(f"content={turn.content!r}")
    print(
        json.dumps(
            [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in turn.tool_calls
            ],
            indent=2,
        )
    )
    if not turn.tool_calls:
        raise SystemExit("No parsed tool call. Confirm --tool-call-parser xlam is enabled.")
    first = turn.tool_calls[0]
    if first.name != "check_inventory" or first.arguments.get("sku") != "KB-75":
        raise SystemExit(f"Unexpected tool call: {first}")


if __name__ == "__main__":
    main()
