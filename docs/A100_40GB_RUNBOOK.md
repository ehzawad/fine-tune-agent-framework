# A100 40 GB runbook

Verification date: 2026-08-28.

This runbook is the literal execution order for one NVIDIA A100 40 GB. Start with base inference, then run a three-step QLoRA smoke job, then train on real trajectories, then serve the adapter.

## 1. Host prerequisites

Required:

- Linux x86-64;
- one NVIDIA A100 40 GB visible through `nvidia-smi`;
- a driver compatible with the PyTorch/vLLM CUDA wheels installed on the host;
- Python 3.10, 3.11, 3.12, or 3.13;
- Git and zsh;
- at least 110 GiB free disk for the original checkpoint cache and outputs;
- preferably 64 GiB or more host RAM.

Inspect first:

```zsh
nvidia-smi
python3 --version
df -h .
free -h
```

## 2. Clone and configure

```zsh
git clone https://github.com/ehzawad/fine-tune-agent-framework.git
cd fine-tune-agent-framework
./scripts/init_env.zsh
```

The initializer writes a random API key and sets `.env` to mode `0600`. Review the generated file. Leave the pinned model revision unchanged for the first proof.

## 3. Base-model serving proof

```zsh
./scripts/setup_serve_env.zsh
./deployment/serve_vllm.zsh
```

Expected startup characteristics:

- vLLM downloads the original BF16 shards into `HF_HOME`;
- the BitsAndBytes plugin performs in-flight 4-bit loading;
- only GPU 0 is visible;
- context is explicitly capped at 8,192 tokens;
- concurrency begins at one sequence;
- CUDA graphs are disabled by default through `--enforce-eager` to reduce startup memory uncertainty;
- the built-in `xlam` parser returns OpenAI-style `tool_calls`;
- the HTTP server binds to `127.0.0.1` unless `HOST` is deliberately overridden.

In a second shell:

```zsh
cd fine-tune-agent-framework
set -a; source .env; set +a
source .venv-serve/bin/activate
python scripts/smoke_http.py
curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" \
  http://127.0.0.1:8000/v1/models
```


From a workstation, tunnel the loopback-only endpoint rather than opening port 8000 publicly:

```zsh
ssh -L 8000:127.0.0.1:8000 user@your-a100-host
```

If startup runs out of memory:

```zsh
MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.88 ./deployment/serve_vllm.zsh
```

If the plugin is not discovered, verify:

```zsh
source .venv-serve/bin/activate
python - <<'PY'
import importlib.metadata
import vllm
print(vllm.__version__)
print(importlib.metadata.version("vllm-bnb-plugin"))
PY
```

## 4. Agent proof

```zsh
set -a; source .env; set +a
source .venv-serve/bin/activate
xlam2-agent init-db --reset
xlam2-agent chat
```

Use a read request first. For a write request, inspect the proposed arguments before approving.

Audit evidence:

```zsh
tail -n 20 .xlam2_ops/audit.jsonl
```

## 5. Training environment and preflight

Stop the vLLM process before training; do not make the server and trainer compete for the same GPU.

```zsh
./scripts/setup_train_env.zsh
./scripts/preflight_a100.zsh
```

The preflight fails on missing pinned packages, unavailable CUDA, insufficient reported VRAM, or absent BF16 support. Disk and host-memory findings are warnings because local caching layouts differ.

When the default PyTorch wheel does not match the host driver, recreate the environment using the exact index selected from PyTorch's official installation page:

```zsh
rm -rf .venv-train
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/<official-cuda-index> \
  ./scripts/setup_train_env.zsh
```

Do not guess the CUDA index; match it to the server's driver and PyTorch's current official matrix.

## 6. Data validation and three-step smoke training

```zsh
source .venv-train/bin/activate
xlam2-train --config configs/a100_40gb_smoke.yaml --validate-only
./scripts/smoke_train.zsh
```

The smoke configuration uses:

- 1,024-token examples;
- rank-8 LoRA;
- attention projection targets only;
- three optimizer steps;
- the same NF4/double-quant/BF16 training path as the full configuration.

A successful smoke run should create:

```text
runs/xlam2-a100-smoke/final_adapter/adapter_config.json
runs/xlam2-a100-smoke/training_manifest.json
```

## 7. Domain training

Replace the demo trajectories and validate them before loading 32B weights:

```zsh
xlam2-train --config configs/a100_40gb_qlora.yaml --validate-only
./scripts/train_qlora.zsh
```

Resume from a Trainer checkpoint:

```zsh
./scripts/train_qlora.zsh \
  --resume-from-checkpoint runs/xlam2-a100-qlora/checkpoint-<step>
```

A configuration override can isolate an experiment:

```zsh
./scripts/train_qlora.zsh \
  --max-steps 100 \
  --output-dir runs/experiment-001
```

## 8. Serve and verify the adapter

Stop the trainer, then use the serving environment:

```zsh
./deployment/serve_adapter_vllm.zsh
```

In another shell:

```zsh
set -a; source .env; set +a
source .venv-serve/bin/activate
export XLAM_MODEL=xlam-2-32b-a100-adapter
python scripts/smoke_http.py --model "$XLAM_MODEL"
python eval/run_eval.py --model "$XLAM_MODEL"
xlam2-agent chat --model "$XLAM_MODEL"
```

Compare base and adapter on a held-out evaluation set. Do not infer improvement from training loss alone.

## 9. Evidence to save from each run

Preserve:

- Git commit SHA;
- `.env` with secrets removed;
- YAML configuration;
- `preflight.json`;
- `training_manifest.json`;
- train/eval metrics;
- TensorBoard logs;
- exact evaluation cases and outputs;
- `nvidia-smi` output;
- vLLM startup command and logs;
- model and adapter IDs returned by `/v1/models`.

The training manifest already records most reproducibility-critical fields, but operational logs remain necessary for CUDA and serving behavior.
