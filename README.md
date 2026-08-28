# Fine-Tune Agent Framework: xLAM-2-32B on One A100 40 GB

This repository is a complete, auditable path for doing useful work with Salesforce's `Salesforce/xLAM-2-32b-fc-r` on a **single NVIDIA A100 40 GB**:

1. serve the base model with on-the-fly 4-bit BitsAndBytes quantization through vLLM;
2. fine-tune tool-calling behavior with single-GPU QLoRA;
3. serve the resulting PEFT adapter through the same OpenAI-compatible endpoint;
4. run a policy-bounded agent that validates, authorizes, executes, and audits tool calls;
5. evaluate tool selection and argument construction with reproducible local cases.

The central engineering boundary is deliberate: **xLAM proposes actions; deterministic software decides whether they may execute.** Model output is never treated as authorization or proof that a side effect occurred.

This repository does not redistribute Salesforce model weights. Its application code is Apache-2.0. The Salesforce checkpoint is separately labeled `CC-BY-NC-4.0` and described as a research release. Review the upstream terms before use, especially before any commercial deployment.

## Why the A100 path uses 4-bit loading

The official configuration contains approximately 32.764 billion parameters. BF16 weights alone require roughly:

```text
32,763,900,000 parameters x 2 bytes = 61.0 GiB
```

That cannot fit on a 40 GB GPU before KV cache, activations, temporary buffers, allocator reserve, and framework overhead. The default project therefore uses:

- **Inference:** vLLM 0.28.0 plus the official out-of-tree BitsAndBytes plugin for in-flight 4-bit loading.
- **Training:** QLoRA with NF4, double quantization, BF16 compute, paged 8-bit AdamW, gradient checkpointing, and LoRA adapters.
- **Initial context:** 8,192 tokens for serving and 2,048 tokens for training. Raise these only after measuring memory on your host.

The raw 4-bit weight arithmetic is about 15.26 GiB, but real runtime usage is higher. The provided settings are conservative starting points, not a universal VRAM guarantee.

## Pinned upstream contract

The project pins the Salesforce model revision:

```text
5ddef330ce01999a05ff56726c543bd6a5fe7142
```

It also pins the training and serving stacks in `pyproject.toml` and `requirements/`. The vendored xLAM/Qwen tool template comes from vLLM 0.28.0. Training manifests record the requested revision, resolved model commit, package versions, GPU information, configuration, and SHA-256 hashes of the data and template.

## Repository map

```text
configs/                         A100 QLoRA and three-step smoke configurations
data/                            small, independently written trajectory examples
deployment/                      base-model and adapter vLLM launchers
docs/                            A100 runbook, data contract, architecture, evidence
examples/                        direct Transformers and OpenAI-client examples
eval/                            starter online tool-selection evaluation
requirements/                    isolated training and serving dependency pins
scripts/                         setup, preflight, smoke, metadata, and memory tools
src/xlam2_ops_agent/             policy-bounded agent runtime
src/xlam2_ops_agent/training/    validation, rendering, masking, QLoRA, manifests
templates/                       pinned xLAM/Qwen vLLM chat template
tests/                           deterministic CPU-side verification
```

## Fastest useful path: serve the base model

These commands assume Linux, one visible A100 40 GB, a working NVIDIA driver, Git, and Python 3.10-3.13. They are written for zsh.

```zsh
git clone https://github.com/ehzawad/fine-tune-agent-framework.git
cd fine-tune-agent-framework
./scripts/init_env.zsh
```

The initializer creates `.env` with a random local API key and mode `0600`. Review `HF_HOME`, `HOST`, and the other paths. `HF_TOKEN` is optional for this public checkpoint but useful for authenticated downloads and rate limits.

Create the isolated serving environment:

```zsh
./scripts/setup_serve_env.zsh
```

Start the 4-bit server:

```zsh
./deployment/serve_vllm.zsh
```

The first launch downloads the original BF16 checkpoint to the Hugging Face cache and quantizes it while loading. Keep at least 110 GiB of free disk. In a second shell:

```zsh
cd fine-tune-agent-framework
set -a; source .env; set +a
source .venv-serve/bin/activate
python scripts/smoke_http.py
```

A successful smoke test returns a parsed `check_inventory` call for `KB-75`. The server binds to `127.0.0.1` by default. From your workstation, prefer an SSH tunnel instead of exposing the raw vLLM port:

```zsh
ssh -L 8000:127.0.0.1:8000 user@your-a100-host
```

Then run the complete policy-bounded operations agent:

```zsh
xlam2-agent init-db --reset
xlam2-agent chat
```

Try:

```text
Which orders does alice@example.com have?
Check inventory for SKU KB-75.
Cancel ORD-1001 because it was ordered by mistake.
```

Read tools may execute automatically. Write tools require an approval outside the model. Every decision and execution result is appended to `.xlam2_ops/audit.jsonl`.

## Fine-tune with QLoRA on the same A100

Serving and training use separate virtual environments because vLLM tightly pins PyTorch and inference dependencies, while the trainer pins Transformers, PEFT, Datasets, Accelerate, and BitsAndBytes. Do not merge these environments casually.

Create the training environment:

```zsh
./scripts/setup_train_env.zsh
```

Run the strict host check:

```zsh
./scripts/preflight_a100.zsh
```

Validate the trajectory schema without loading the model:

```zsh
source .venv-train/bin/activate
xlam2-train --config configs/a100_40gb_qlora.yaml --validate-only
```

Run the three-step GPU smoke training first:

```zsh
./scripts/smoke_train.zsh
```

Only after that succeeds, run the full configured job:

```zsh
./scripts/train_qlora.zsh
```

The included data is a **pipeline proof**, not a useful domain corpus. Replace `data/demo_train.jsonl` and `data/demo_eval.jsonl` with reviewed trajectories following `docs/TRAINING_DATA_FORMAT.md`, then update `configs/a100_40gb_qlora.yaml`.

The default full run writes:

```text
runs/xlam2-a100-qlora/
  checkpoints...
  final_adapter/
  preflight.json
  resolved_config.json
  training_manifest.json
  train_results.json
  eval_results.json
  tensorboard/
```

Monitor it with:

```zsh
tensorboard --logdir runs/xlam2-a100-qlora/tensorboard --host 127.0.0.1
```

## Serve the trained adapter

After `final_adapter/adapter_config.json` exists:

```zsh
./deployment/serve_adapter_vllm.zsh
```

The server exposes two model IDs:

```text
xlam-2-32b-base
xlam-2-32b-a100-adapter
```

Use the adapter explicitly:

```zsh
export XLAM_MODEL=xlam-2-32b-a100-adapter
source .venv-serve/bin/activate
python scripts/smoke_http.py --model "$XLAM_MODEL"
xlam2-agent chat --model "$XLAM_MODEL"
```

vLLM routes requests to a LoRA adapter through the request's `model` field. Calling the base model ID does not activate the adapter.

## Training data contract

Each JSONL row is one complete trajectory with its own tool registry and message sequence. The trainer:

- accepts either direct xLAM-style tool definitions or OpenAI-style function wrappers;
- rejects undeclared tools, malformed JSON Schema, extra sequence transitions, orphan tool results, and incomplete calls;
- creates one example for every assistant turn;
- renders with the pinned xLAM/Qwen template;
- masks system, user, prior assistant, and tool-result tokens with label `-100`;
- trains only the current assistant completion;
- left-truncates old prompt context when necessary;
- never silently truncates the assistant target.

That last point matters: a training pipeline that learns user text or tool observations as labels is quietly training the wrong conditional distribution.

## Agent execution architecture

```text
user request
    |
    v
xLAM-2 proposes tool calls
    |
    v
known-tool allowlist
    |
    v
strict Pydantic argument validation
    |
    v
deterministic policy decision
    |             |              |
    |             |              +--> deny / dry-run
    |             +-----------------> external approval
    v
transactional tool execution
    |
    v
structured result + audit evidence
    |
    v
xLAM-2 explains only the confirmed outcome
```

Implemented controls include strict schemas, read/write risk classes, refund limits, explicit write approval, transactional SQLite state, idempotent cancellation/refunds, bounded model/tool loops, duplicate-call protection, structured errors, conservative raw-JSON recovery, and audit redaction.

This is not an identity provider, secrets manager, production CRM connector, tamper-evident ledger, or enterprise authorization service. Those systems belong outside the model and should connect at the existing policy and tool boundaries.

## Verification commands

CPU-side proof, requiring no model download:

```zsh
python -m compileall -q src scripts examples eval tests
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m xlam2_ops_agent.training.train \
  --config configs/a100_40gb_smoke.yaml \
  --validate-only
PYTHONPATH=src python -m xlam2_ops_agent demo
python scripts/estimate_memory.py \
  --context 8192 \
  --tensor-parallel 1 \
  --weight-bits 4 \
  --lora-rank 16
```

Online checks after the server starts:

```zsh
set -a; source .env; set +a
python scripts/smoke_http.py
python eval/run_eval.py
python scripts/verify_official_metadata.py
curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" \
  http://127.0.0.1:8000/v1/models
```

The evaluation cases are wiring tests, not BFCL or tau-bench reproduction.

## Tuning the A100 recipe

When an out-of-memory error occurs, change one variable at a time in this order:

1. Training: lower `data.max_seq_length` from 2048 to 1536 or 1024.
2. Training: switch `lora.target_modules` from `all-linear` to the attention projections used by the smoke config.
3. Training: lower LoRA rank from 16 to 8.
4. Serving: lower `MAX_MODEL_LEN` from 8192 to 4096.
5. Serving: lower `GPU_MEMORY_UTILIZATION` slightly if startup allocation fails.
6. Serving: keep `MAX_NUM_SEQS=1` until a measured concurrency test passes.

Do not “solve” memory pressure by turning off schema checks, assistant-only masking, or approval controls; those are correctness boundaries, not performance knobs.

## Known limitations

The full 32B model and A100 kernels could not be executed in the environment where this repository was assembled. CPU-side code, validation, policy, storage, audit, rendering logic, configuration, and tests were exercised; real checkpoint loading, CUDA memory, throughput, quantization quality, and adapter quality must be established on your A100 host. See `docs/TEST_REPORT.md` for the exact evidence boundary.

The complete xLAM-2 training mixture and exact Salesforce training manifest are not public, so this repository implements domain QLoRA adaptation rather than claiming an exact reproduction of Salesforce's training run.

## Primary documentation

See `docs/SOURCES.md` for the dated, primary-source reading list. The most operationally important documents are:

- Salesforce xLAM-2-32B model card and repository files;
- Salesforce xLAM repository, APIGen-MT paper, and ActionStudio paper;
- Hugging Face Transformers BitsAndBytes and PEFT quantization guides;
- vLLM 0.28.0 BitsAndBytes, LoRA, tool-calling, and server documentation;
- PyTorch's official installation guidance.
