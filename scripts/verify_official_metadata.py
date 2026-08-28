#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import httpx

MODEL = "Salesforce/xLAM-2-32b-fc-r"
DEFAULT_REVISION = "5ddef330ce01999a05ff56726c543bd6a5fe7142"

EXPECTED_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 5120,
    "intermediate_size": 27648,
    "num_hidden_layers": 64,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
    "rope_theta": 1_000_000.0,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}

EXPECTED_GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
}


def fetch_json(
    client: httpx.Client, *, revision: str, filename: str
) -> dict[str, Any]:
    base = f"https://huggingface.co/{MODEL}/resolve/{revision}"
    response = client.get(f"{base}/{filename}", follow_redirects=True)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected an object in {filename}")
    return data


def check_subset(actual: dict[str, Any], expected: dict[str, Any], label: str) -> list[str]:
    errors = []
    for key, value in expected.items():
        if actual.get(key) != value:
            errors.append(f"{label}.{key}: expected {value!r}, got {actual.get(key)!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify current official xLAM-2-32B metadata without downloading weights."
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face branch, tag, or commit to verify",
    )
    args = parser.parse_args()

    with httpx.Client(timeout=args.timeout) as client:
        config = fetch_json(client, revision=args.revision, filename="config.json")
        generation = fetch_json(
            client, revision=args.revision, filename="generation_config.json"
        )
        tokenizer = fetch_json(
            client, revision=args.revision, filename="tokenizer_config.json"
        )

    errors = check_subset(config, EXPECTED_CONFIG, "config")
    errors.extend(check_subset(generation, EXPECTED_GENERATION, "generation"))

    template = str(tokenizer.get("chat_template") or "")
    for fragment in (
        "You are a helpful assistant that can use tools",
        '"name":"tool_call_name"',
        "If there are no tools",
    ):
        if fragment not in template:
            errors.append(f"tokenizer.chat_template is missing expected fragment: {fragment!r}")

    print(f"model: {MODEL}")
    print(f"revision: {args.revision}")
    print(f"architecture: {config.get('architectures')}")
    print(
        "shape: "
        f"layers={config.get('num_hidden_layers')}, "
        f"hidden={config.get('hidden_size')}, "
        f"heads={config.get('num_attention_heads')}, "
        f"kv_heads={config.get('num_key_value_heads')}"
    )
    print(f"config context: {config.get('max_position_embeddings')}")
    print(f"tokenizer model_max_length: {tokenizer.get('model_max_length')}")
    if tokenizer.get("model_max_length") != config.get("max_position_embeddings"):
        print("warning: tokenizer and model config advertise different context lengths")

    if errors:
        print("verification: FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("verification: PASSED")


if __name__ == "__main__":
    main()
