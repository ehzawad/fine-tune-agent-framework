# vLLM deployment on one A100 40 GB

The launchers target vLLM 0.28.0 and its official out-of-tree BitsAndBytes plugin. They use in-flight 4-bit quantization because the 61.0 GiB BF16 weight payload cannot fit on a 40 GB GPU.

## Install

```zsh
./scripts/setup_serve_env.zsh
```

The serving environment is intentionally separate from `.venv-train`.

## Base model

```zsh
./deployment/serve_vllm.zsh
```

Defaults:

```text
model              Salesforce/xLAM-2-32b-fc-r
revision           5ddef330ce01999a05ff56726c543bd6a5fe7142
served model       xlam-2-32b-fc-r
bind address       127.0.0.1
quantization       BitsAndBytes in-flight 4-bit
context            8192
concurrency        1
GPU utilization    0.90
tool parser        xlam
chat template      templates/tool_chat_template_xlam_qwen.jinja
generation config  vLLM defaults
execution mode     eager
```

Override through environment variables, preferably in `.env`:

```zsh
MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.88 \
  ./deployment/serve_vllm.zsh
```

## Adapter

```zsh
./deployment/serve_adapter_vllm.zsh
```

The adapter defaults to `runs/xlam2-a100-qlora/final_adapter` and is registered as `xlam-2-32b-a100-adapter`. Select that exact model ID in requests. The base ID remains available separately.

## Health and contract tests

```zsh
set -a; source .env; set +a
source .venv-serve/bin/activate
python scripts/smoke_http.py
curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" \
  http://127.0.0.1:8000/v1/models
```

The xLAM parser extracts syntax. Pydantic schemas and policy code still determine whether a call is valid and executable.

## Operational cautions

- The first start must retain the original checkpoint on disk even though GPU-resident weights are quantized.
- Do not start the trainer and server on the same A100 simultaneously.
- Keep `MAX_NUM_SEQS=1` until measured concurrency and long-prompt tests establish headroom.
- A larger context consumes KV cache linearly and can make a previously successful startup fail.
- `--enforce-eager` reduces CUDA-graph memory uncertainty but may sacrifice throughput. Set `ENFORCE_EAGER=0` only after a successful baseline.
- Pin both model and tokenizer revisions. Do not rely on mutable `main` in a reproducible deployment.
- The server binds to loopback by default. Prefer SSH port forwarding for remote use. If `HOST=0.0.0.0` is intentional, place the endpoint behind real network authentication and firewall policy.
- The vLLM API key is a development control, not an enterprise identity or authorization layer.
- The Salesforce checkpoint remains research/noncommercially licensed regardless of the serving framework.
