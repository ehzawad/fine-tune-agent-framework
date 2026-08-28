# Architecture

## Two independent planes

The repository separates model adaptation from action execution.

```text
Training plane
trajectory JSONL
  -> strict validation
  -> xLAM/Qwen template rendering
  -> assistant-only labels
  -> 4-bit base load
  -> LoRA optimization
  -> PEFT adapter + manifest

Runtime plane
user
  -> xLAM base or adapter
  -> proposed tool calls
  -> schema validation
  -> policy and approval
  -> transactional connector
  -> audit record
  -> structured result
  -> grounded final response
```

A better tool-calling model improves proposals. It does not enlarge authority. The policy and connector layers remain authoritative after fine-tuning.

## Training components

`training/config.py` loads strict YAML and rejects unknown knobs. `training/data.py` normalizes tools and messages, verifies conversation state, renders every assistant turn, and builds completion-only labels. `training/model.py` performs NF4 loading and PEFT injection. `training/train.py` constructs current Transformers arguments, refuses silently dropped API fields, runs Trainer, saves the adapter, and writes metrics and a manifest.

The base model remains frozen. Only LoRA parameters are trainable. The code refuses to proceed when no parameters are trainable or when the entire model is unexpectedly trainable.

## Runtime components

`client.py` calls vLLM's OpenAI-compatible endpoint. `protocol.py` supplies a conservative fallback for direct xLAM JSON arrays. `tools.py` owns JSON Schema and Pydantic validation. `policy.py` owns deterministic read/write decisions. `store.py` demonstrates transactions and idempotency. `agent.py` owns the bounded model-tool loop. `audit.py` records decisions and outcomes.

## Trust boundaries

Untrusted values:

- user messages;
- retrieved or tool-provided text;
- model prose;
- model-selected tool names;
- model-generated arguments;
- parser output;
- connector responses until interpreted according to their contract.

Trusted only within their defined scope:

- reviewed tool registry;
- strict argument models;
- authenticated identity and authorization inputs supplied by an external system;
- deterministic policy code;
- transactional connector invariants;
- append-only or tamper-evident production audit storage.

The model is not one of the trusted authorities.

## Replacing the demo domain

To integrate a real system:

1. implement a connector class with explicit read and write methods;
2. define one strict Pydantic argument model per operation;
3. register a `ToolSpec` with a risk class;
4. encode business limits in policy and connector invariants;
5. add idempotency keys for every retriable write;
6. add deterministic tests without involving the model;
7. create realistic trajectories using the same schemas;
8. evaluate base and adapter against hidden stateful tasks;
9. expose only least-privilege credentials to the connector process.

Do not put credentials in prompts, model weights, training examples, or tool descriptions.
