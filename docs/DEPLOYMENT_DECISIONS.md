# Deployment decisions

## 4-bit rather than BF16

One A100 40 GB cannot hold approximately 61.0 GiB of BF16 xLAM-2-32B weights. The base serving path uses vLLM's current BitsAndBytes plugin for in-flight 4-bit loading. Training uses the same conceptual memory reduction through QLoRA, while preserving BF16 compute on the A100.

This is a capacity decision, not a claim that 4-bit outputs are identical to BF16. Base and adapter quality must be evaluated on the target tasks.

## Separate training and serving environments

vLLM pins a tightly coupled PyTorch/CUDA stack. Transformers, PEFT, BitsAndBytes, Datasets, and Accelerate evolve independently. Two virtual environments make dependency ownership explicit and prevent a trainer upgrade from silently replacing the serving engine's PyTorch build.

## Explicit model revision

The default is Salesforce's latest published repository commit as of 2026-08-28. Both model and tokenizer revisions are passed to vLLM and Transformers. The manifest records the resolved commit exposed by Transformers.

## Vendored chat template

The tool template is vendored from vLLM 0.28.0 rather than fetched at runtime. Training and serving therefore use the same reviewed serialization contract. Any future template update should be treated as a data-format migration and regression-tested before retraining or serving.

## Assistant-only supervision

Every assistant turn becomes a target; all preceding context is masked. This avoids optimizing the model to predict user messages or tool observations. Prompt truncation removes old context from the left, while a target too long to fit causes an explicit error.

## Model server versus authority runtime

vLLM owns model loading, batching, tokenization, KV cache, LoRA routing, and xLAM syntax extraction. The Python application owns tool registration, schema validation, policy, approval, execution, state, and audit. A parser success is not an authorization decision.

## Eager serving baseline

The default enables `--enforce-eager` to avoid CUDA-graph capture consuming uncertain memory during the first proof on 40 GB. Once the baseline is stable, the operator can set `ENFORCE_EAGER=0` and measure the throughput/memory tradeoff.

## SQLite demo connector

SQLite provides actual transactions, idempotency, and state transitions while remaining locally runnable. Replace it with a real connector without moving authorization into the model.
