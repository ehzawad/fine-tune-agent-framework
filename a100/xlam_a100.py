from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Salesforce/xLAM-2-32b-fc-r"
MODEL_REVISION = os.environ.get("XLAM_MODEL_REVISION", "main")
PARAMS = 32_763_900_000


class ContractError(ValueError):
    pass


def obj(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def check(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    predicates = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
    }
    if expected and (expected not in predicates or not predicates[expected]()):
        raise ContractError(f"{path}: expected {expected}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ContractError(f"{path}: missing {missing}")
        for key, item in value.items():
            if key in properties:
                check(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties", True) is False:
                raise ContractError(f"{path}.{key}: extra property")


def validate(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record.get("id"), str):
        raise ContractError("id required")
    tools: dict[str, dict[str, Any]] = {}
    for raw_tool in record.get("tools", []):
        tool = raw_tool["function"] if raw_tool.get("type") == "function" else raw_tool
        if not isinstance(tool.get("name"), str):
            raise ContractError("tool name required")
        tools[tool["name"]] = tool
    pending: dict[str, str] = {}
    messages: list[dict[str, Any]] = []
    targets = 0
    for index, message in enumerate(record.get("messages", [])):
        role = message.get("role")
        if role in {"system", "user"}:
            if pending:
                raise ContractError("message before tool results")
            if not isinstance(message.get("content"), str):
                raise ContractError("text content required")
            messages.append(dict(message))
            continue
        if role == "assistant":
            if pending:
                raise ContractError("assistant before tool results")
            normalized_calls = []
            for call_index, raw_call in enumerate(message.get("tool_calls", []) or []):
                function = raw_call.get("function", raw_call)
                name = function.get("name")
                arguments = obj(function.get("arguments", {}), "arguments")
                if name not in tools:
                    raise ContractError(f"unknown tool {name}")
                check(arguments, tools[name]["parameters"])
                call_id = raw_call.get("id", f"c{index}_{call_index}")
                pending[call_id] = name
                normalized_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            content = message.get("content", "")
            if not normalized_calls and not isinstance(content, str):
                raise ContractError("assistant target required")
            normalized = {"role": "assistant", "content": content or ""}
            if normalized_calls:
                normalized["tool_calls"] = normalized_calls
            messages.append(normalized)
            targets += 1
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id is None and len(pending) == 1:
                call_id = next(iter(pending))
            if call_id not in pending:
                raise ContractError("orphan tool result")
            name = message.get("name", pending[call_id])
            if name != pending[call_id]:
                raise ContractError("tool name mismatch")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": message.get("content"),
                }
            )
            del pending[call_id]
            continue
        raise ContractError(f"invalid role {role}")
    if pending:
        raise ContractError("missing tool results")
    if not targets:
        raise ContractError("assistant target required")
    return {"id": record["id"], "tools": list(tools.values()), "messages": messages}


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(validate(json.loads(line)))
        except Exception as exc:
            raise ContractError(f"line {line_number}: {exc}") from exc
    if not rows:
        raise ContractError("empty dataset")
    return rows


def token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return value


def encode(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    previous = None
    labels: list[int] = []
    current: list[int] = []
    for index, message in enumerate(row["messages"]):
        current = token_ids(
            tokenizer.apply_chat_template(
                row["messages"][: index + 1],
                tools=row["tools"],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if previous is None:
            labels = [-100] * len(current)
        else:
            if current[: len(previous)] != previous:
                raise RuntimeError("chat template is not append-only")
            labels += [-100] * (len(current) - len(labels))
            if message["role"] == "assistant":
                labels[len(previous) : len(current)] = current[len(previous) : len(current)]
        previous = current
    current = current[:max_length]
    labels = labels[:max_length]
    if all(label == -100 for label in labels):
        raise ContractError("no assistant tokens in window")
    return {"input_ids": current, "attention_mask": [1] * len(current), "labels": labels}


class Dataset:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        length = (max(len(item["input_ids"]) for item in items) + 7) // 8 * 8

        def padded(key: str, value: int) -> list[list[int]]:
            return [item[key] + [value] * (length - len(item[key])) for item in items]

        return {
            "input_ids": torch.tensor(padded("input_ids", self.pad_token_id)),
            "attention_mask": torch.tensor(padded("attention_mask", 0)),
            "labels": torch.tensor(padded("labels", -100)),
        }


def preflight() -> None:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(rows) != 1:
        raise SystemExit(f"expected one visible GPU, got {len(rows)}")
    name, memory_mib = [part.strip() for part in rows[0].split(",")]
    if "A100" not in name or float(memory_mib) / 1024 < 39:
        raise SystemExit(f"need A100 40GB; got {rows[0]}")
    if shutil.disk_usage(Path.cwd()).free / 1024**3 < 90:
        raise SystemExit("need at least 90 GiB free disk")
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 unavailable")
    weight_gib = PARAMS * 0.55 / 1024**3
    kv_gib = 2 * 64 * 8 * 128 * 2 * 8192 / 1024**3
    print(
        json.dumps(
            {
                "gpu": rows[0],
                "nf4_weight_estimate_gib": round(weight_gib, 2),
                "8k_single_sequence_kv_gib": round(kv_gib, 2),
            },
            indent=2,
        )
    )


def train(smoke: bool = False) -> None:
    import torch
    import yaml
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    config = yaml.safe_load(Path("train.yaml").read_text(encoding="utf-8"))
    rows = load(Path(config["train_file"]))
    if smoke:
        rows = rows[:4]
        config["max_steps"] = 2
        config["max_length"] = 512
        config["output_dir"] += "-smoke"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset([encode(tokenizer, row, config["max_length"]) for row in rows])
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    arguments = TrainingArguments(
        output_dir=config["output_dir"] + "/checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_steps=max(2, min(25, config["max_steps"])),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        use_cache=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()
    output = Path(config["output_dir"]) / "final_adapter"
    trainer.model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(output)


@dataclass(frozen=True)
class Call:
    id: str
    name: str
    arguments: dict[str, Any]


def calls(message: dict[str, Any]) -> list[Call]:
    raw = message.get("tool_calls")
    content = message.get("content")
    if not raw and isinstance(content, str) and content.strip().startswith("["):
        raw = json.loads(content)
    parsed = []
    for index, raw_call in enumerate(raw or []):
        function = raw_call.get("function", raw_call)
        parsed.append(
            Call(
                raw_call.get("id", f"c{index}"),
                function["name"],
                obj(function.get("arguments", {}), "arguments"),
            )
        )
    if len(parsed) > 8:
        raise ContractError("too many calls")
    return parsed


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as connection:
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,status TEXT,total INTEGER);"
                "CREATE TABLE IF NOT EXISTS idem(k TEXT PRIMARY KEY,r TEXT);"
                "INSERT OR IGNORE INTO orders VALUES('ORD-1001','processing',12900);"
            )

    def db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, isolation_level=None)

    def get(self, order_id: str) -> dict[str, Any]:
        with self.db() as connection:
            row = connection.execute(
                "SELECT id,status,total FROM orders WHERE id=?", (order_id,)
            ).fetchone()
        return {
            "ok": bool(row),
            "order": dict(zip(["order_id", "status", "total_cents"], row)) if row else None,
        }

    def cancel(self, order_id: str, reason: str, key: str) -> dict[str, Any]:
        with self.db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT r FROM idem WHERE k=?", (key,)).fetchone()
            if old:
                connection.execute("COMMIT")
                return json.loads(old[0]) | {"replay": True}
            row = connection.execute(
                "SELECT status,total FROM orders WHERE id=?", (order_id,)
            ).fetchone()
            if not row or row[0] != "processing":
                result = {"ok": False, "error": "not_cancellable"}
            else:
                connection.execute(
                    "UPDATE orders SET status='cancelled' WHERE id=?", (order_id,)
                )
                result = {
                    "ok": True,
                    "status": "cancelled",
                    "refund_cents": row[1],
                    "reason": reason,
                }
            connection.execute("INSERT INTO idem VALUES(?,?)", (key, json.dumps(result)))
            connection.execute("COMMIT")
            return result


def post(base_url: str, model: str, api_key: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": [{"type": "function", "function": tool} for tool in tools],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["choices"][0]["message"]


def demo(online: bool, model: str) -> None:
    tools = [
        {
            "name": "get_order",
            "description": "Get order",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cancel_order",
            "description": "Cancel after confirmation",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False,
            },
        },
    ]
    store = Store(Path("runs/orders.sqlite3"))
    session_id = str(uuid.uuid4())
    messages = [
        {
            "role": "system",
            "content": "Use tools. Tool calls must be one JSON array. Writes require external approval.",
        },
        {"role": "user", "content": "Cancel ORD-1001. I confirm."},
    ]
    for turn in range(6):
        if online:
            message = post(
                os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                model,
                os.getenv("VLLM_API_KEY", "EMPTY"),
                messages,
                tools,
            )
        elif turn == 0:
            message = {
                "tool_calls": [
                    {
                        "id": "g",
                        "function": {
                            "name": "get_order",
                            "arguments": {"order_id": "ORD-1001"},
                        },
                    }
                ]
            }
        elif turn == 1:
            message = {
                "tool_calls": [
                    {
                        "id": "x",
                        "function": {
                            "name": "cancel_order",
                            "arguments": {
                                "order_id": "ORD-1001",
                                "reason": "confirmed",
                            },
                        },
                    }
                ]
            }
        else:
            message = {
                "content": "ORD-1001 is cancelled and the confirmed refund is $129.00."
            }
        parsed_calls = calls(message)
        if not parsed_calls:
            print(message["content"])
            return
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in parsed_calls
                ],
            }
        )
        for call in parsed_calls:
            if call.name == "get_order":
                check(call.arguments, tools[0]["parameters"])
                result = store.get(call.arguments["order_id"])
            elif call.name == "cancel_order":
                check(call.arguments, tools[1]["parameters"])
                order = store.get(call.arguments["order_id"])
                if not order["ok"] or order["order"]["total_cents"] > 50_000:
                    result = {"ok": False, "error": "policy_denied"}
                else:
                    approval = os.getenv("AUTO_APPROVE", "0") == "1" or input(
                        f"Approve cancellation/refund ${order['order']['total_cents'] / 100:.2f}? [y/N] "
                    ).lower() in {"y", "yes"}
                    result = (
                        store.cancel(
                            call.arguments["order_id"],
                            call.arguments["reason"],
                            session_id + call.id,
                        )
                        if approval
                        else {"ok": False, "error": "approval_denied"}
                    )
            else:
                result = {"ok": False, "error": "unknown_tool"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result),
                }
            )
    raise RuntimeError("agent round limit")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--smoke", action="store_true")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("path", nargs="?", default="sample_train.jsonl")
    demo_parser = commands.add_parser("demo")
    demo_parser.add_argument("--online", action="store_true")
    demo_parser.add_argument("--model", default="xlam-2-32b")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "train":
        train(args.smoke)
    elif args.command == "validate":
        print(f"validated {len(load(Path(args.path)))} trajectories")
    elif args.command == "demo":
        demo(args.online, args.model)


if __name__ == "__main__":
    main()
