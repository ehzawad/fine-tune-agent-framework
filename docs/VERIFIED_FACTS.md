# Verified facts, engineering inferences, and unknowns

Verification date: 2026-08-28.

## Verified from primary sources

- `Salesforce/xLAM-2-32b-fc-r` is a roughly 32.764B-parameter, Qwen2-class causal language model specialized for multi-turn conversation and function calling.
- Its repository config contains 64 layers, hidden size 5,120, 40 attention heads, 8 KV heads, and 32,768 maximum positions.
- The native tool contract asks for one JSON array of objects with `name` and `arguments`, asks for clarification when required parameters are missing, and waits for tool results before interpreting an action as complete.
- The checkpoint is BF16, labeled CC-BY-NC-4.0, and described as a research release.
- Salesforce's APIGen-MT pipeline creates verified task blueprints and simulated multi-turn user-agent-environment trajectories.
- Salesforce released a 5,000-trajectory APIGen-MT subset and ActionStudio code, while stating that xLAM-related data is only partially released.
- Current vLLM 0.28.0 exposes a built-in `xlam` tool parser, an OpenAI-compatible server, LoRA routing, and BitsAndBytes through its official out-of-tree plugin.
- Current Transformers/PEFT guidance supports 4-bit base loading with trainable adapters; NF4 is recommended for training normally distributed pretrained weights, and `all-linear` is the QLoRA-style target choice.

## Engineering conclusions

- BF16 xLAM-2-32B cannot fit in one A100 40 GB; the raw weights alone are about 61.0 GiB.
- Four-bit inference and QLoRA are appropriate first experiments for this hardware, but fitting arithmetic is not evidence of acceptable latency or output quality.
- A language model should be an action planner behind schema validation, deterministic authorization, approval, transactions, and audit—not the authority layer itself.
- Parser output remains untrusted input. The parser extracts structure; the runtime decides semantics and permissions.
- Temperature zero is a reasonable default for action selection, while model comparison should still include repeated trials because GPU kernels and generation systems can exhibit residual nondeterminism.

## Important unknowns

- The complete xLAM-2-32B training mixture, filters, sample counts, and exact Salesforce training manifest are not public.
- This repository has not independently reproduced Salesforce's BFCL or tau-bench results.
- The full checkpoint, vLLM BitsAndBytes path, QLoRA training loop, and LoRA serving combination have not been executed in the assembly environment; they require the user's A100 host.
- Quantization loss, domain adaptation quality, throughput, latency, and maximum stable context are workload-specific measurements.
