# Verified xLAM-2-32B specification

Verification date: 2026-08-28. Default pinned revision: `5ddef330ce01999a05ff56726c543bd6a5fe7142`.

| Field | Verified value |
|---|---:|
| Repository | `Salesforce/xLAM-2-32b-fc-r` |
| Model class | `Qwen2ForCausalLM` |
| Parameters | approximately 32.764B |
| Checkpoint tensor type | BF16 |
| Hidden size | 5,120 |
| MLP intermediate size | 27,648 |
| Transformer layers | 64 |
| Attention heads | 40 |
| KV heads | 8 |
| Head dimension | 128 |
| Attention form | grouped-query attention, 5 query heads per KV head |
| Vocabulary | 152,064 |
| Activation | SiLU |
| Normalization epsilon | 1e-6 |
| Position setting | 32,768 in config; model card describes optional YaRN extension to 128K |
| RoPE theta | 1,000,000 |
| License label | CC-BY-NC-4.0; research release |

## Weight memory

```text
BF16: 32,763,900,000 x 2 bytes = 61.028 GiB
4-bit ideal payload: 32,763,900,000 x 0.5 bytes = 15.257 GiB
```

Four-bit storage has scales, metadata, non-quantized modules, allocator overhead, and runtime buffers, so 15.257 GiB is a lower-bound arithmetic value rather than observed VRAM.

## KV-cache arithmetic

For 64 layers, 8 KV heads, head dimension 128, BF16 keys and values:

```text
bytes per token = 2 x 64 x 8 x 128 x 2 = 262,144 bytes = 256 KiB
8192 tokens     = 2 GiB for one sequence
32768 tokens    = 8 GiB for one sequence
131072 tokens   = 32 GiB for one sequence
```

The default server therefore begins at 8K and one sequence. Real vLLM allocation includes more than KV cache.

## LoRA parameter arithmetic

For rank 16 across the Qwen attention and MLP linear layers, the project's architecture estimator reports approximately 134.2 million adapter parameters. Raw BF16 adapter weights are roughly 0.25 GiB, but training also needs gradients, optimizer state, activations, and temporary buffers.

## Context discrepancy

The model config advertises 32,768 positions, while the tokenizer metadata historically advertised 16,384. The model card describes 32K default and optional 128K YaRN extension. The repository therefore:

1. pins a revision;
2. passes context length explicitly;
3. starts below both advertised values on the 40 GB GPU;
4. treats any longer context as an empirical quality and capacity experiment.
