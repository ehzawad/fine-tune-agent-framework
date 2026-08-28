"""OpenAI Python client against the local vLLM endpoint.

Install with: pip install -e '.[openai]'
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("VLLM_API_KEY", "local-token"),
    base_url=os.getenv("XLAM_BASE_URL", "http://127.0.0.1:8000/v1"),
)
model = os.getenv("XLAM_MODEL", "xlam-2-32b-fc-r")
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check inventory for one SKU.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    }
]

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Check inventory for KB-75."}],
    tools=tools,
    tool_choice="auto",
    temperature=0,
)
message = response.choices[0].message
for call in message.tool_calls or []:
    print(call.function.name, json.loads(call.function.arguments))
