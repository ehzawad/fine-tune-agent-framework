# Verification report

Verification date: 2026-08-28.

## Executed in the assembly environment

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
python -m pip install -e . --no-deps --no-build-isolation
xlam2-agent demo
xlam2-train --config configs/a100_40gb_smoke.yaml --validate-only
```

Observed:

```text
33 passed
train trajectories: 6
train assistant targets: 12
train tool-call targets: 6
train text targets: 6
eval trajectories: 2
eval assistant targets: 3
ORD-1001 was verified as processing and then cancelled after explicit approval.
steps=3; final_status=cancelled; audit_events=4
ideal 4-bit weights: approximately 15.257 GiB
BF16 KV cache at 8192 tokens, one sequence: 2.000 GiB
rank-16 all-linear LoRA parameters: approximately 134.218M
```

## What these checks establish

They exercise syntax compilation, strict configuration parsing, trajectory normalization,
conversation-state validation, xLAM template rendering, completion-only labels,
nontruncation of assistant targets, padding, conservative raw-JSON recovery, fail-closed
OpenAI/vLLM tool-call parsing, known-tool enforcement, policy decisions, write approval,
refund limits, SQLite transactions, idempotency, bounded loops, audit records, package
entry points, and architecture arithmetic.

The shell files were also parsed with Bash as a portability sanity check. The assembly
environment did not contain Zsh, so a native `zsh -n` check was not available here.

## What remains unverified here

The assembly environment had no NVIDIA GPU and no external package/model download access.
It therefore did not execute:

- the 32B checkpoint load;
- A100 CUDA kernels or measured VRAM;
- vLLM 0.28.0 with the BitsAndBytes plugin;
- an actual three-step QLoRA optimization;
- adapter loading through vLLM;
- online function-calling quality;
- throughput, latency, long-context behavior, or quantization loss;
- Salesforce benchmark reproduction.

The repository supplies explicit smoke commands for establishing those facts on the A100
server. No claim should be upgraded from "configured" to "verified" until those commands
pass and their logs are preserved.
