#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-help}"; shift || true
case "$CMD" in
  train-env)
    "${PYTHON_BIN:-python3.12}" -m venv "$ROOT/.venv-train"
    source "$ROOT/.venv-train/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "$ROOT/requirements-train.txt"
    ;;
  serve-env)
    "${PYTHON_BIN:-python3.12}" -m venv "$ROOT/.venv-serve"
    source "$ROOT/.venv-serve/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "$ROOT/requirements-serve.txt"
    ;;
  preflight)
    source "$ROOT/.venv-train/bin/activate"
    python "$ROOT/xlam_a100.py" preflight
    ;;
  smoke)
    source "$ROOT/.venv-train/bin/activate"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$ROOT/xlam_a100.py" train --smoke
    ;;
  train)
    source "$ROOT/.venv-train/bin/activate"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$ROOT/xlam_a100.py" train "$@"
    ;;
  serve)
    source "$ROOT/.venv-serve/bin/activate"
    ADAPTER="${1:-}"
    ARGS=(serve Salesforce/xLAM-2-32b-fc-r --served-model-name xlam-2-32b --dtype bfloat16 --quantization bitsandbytes --load-format bitsandbytes --max-model-len "${MAX_MODEL_LEN:-8192}" --max-num-seqs 1 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" --enable-auto-tool-choice --tool-call-parser xlam --chat-template "$ROOT/tool_chat_template.jinja" --enforce-eager --host 0.0.0.0 --port "${PORT:-8000}")
    if [[ -n "$ADAPTER" ]]; then ARGS+=(--enable-lora --max-lora-rank 64 --lora-modules "xlam-2-32b-custom=$ADAPTER"); fi
    if [[ -n "${VLLM_API_KEY:-}" ]]; then ARGS+=(--api-key "$VLLM_API_KEY"); fi
    exec vllm "${ARGS[@]}"
    ;;
  test)
    PYTHONPATH="$ROOT" python -m unittest -v "$ROOT/test_xlam_a100.py"
    ;;
  *) echo "usage: bash run.sh {train-env|serve-env|preflight|smoke|train|serve [adapter]|test}"; exit 2 ;;
esac
