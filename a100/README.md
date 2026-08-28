# Single-A100 xLAM-2 runner

This directory is a self-contained fallback entry point for one NVIDIA A100
40GB. It provides strict trajectory validation, assistant-only loss masking,
NF4 QLoRA training, 4-bit vLLM serving, and a policy-gated SQLite agent demo.
The richer modular package at the repository root uses the same contracts.

```zsh
cd a100
bash run.sh train-env
export HF_TOKEN='...'
bash run.sh preflight
bash run.sh smoke
bash run.sh train

# Separate shell/environment for serving
bash run.sh serve-env
export HF_TOKEN='...'
export VLLM_API_KEY='replace-me'
bash run.sh serve outputs/xlam-a100/final_adapter

# Another shell
source .venv-serve/bin/activate
python xlam_a100.py demo --online --model xlam-2-32b-custom
```

A 32B BF16 checkpoint does not fit in 40GB. The code therefore trains LoRA
adapters over NF4 weights and serves through bitsandbytes. Start at 2K training
sequences and 8K serving context. The model page labels the checkpoint
CC-BY-NC-4.0; review that separate license before use.
