#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR=${0:A:h:h}
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_load_env.zsh"
ENV_DIR=${SERVE_ENV_DIR:-.venv-serve}
if [[ ! -x "$ENV_DIR/bin/vllm" ]]; then
  print -u2 "Missing $ENV_DIR. Run ./scripts/setup_serve_env.zsh first."
  exit 2
fi
source "$ENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
MODEL_ID=${MODEL_ID:-Salesforce/xLAM-2-32b-fc-r}
MODEL_REVISION=${MODEL_REVISION:-5ddef330ce01999a05ff56726c543bd6a5fe7142}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-xlam-2-32b-base}
ADAPTER_NAME=${ADAPTER_NAME:-xlam-2-32b-a100-adapter}
ADAPTER_PATH=${ADAPTER_PATH:-$ROOT_DIR/runs/xlam2-a100-qlora/final_adapter}
MAX_LORA_RANK=${MAX_LORA_RANK:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
API_KEY=${VLLM_API_KEY:-}
DOWNLOAD_DIR=${HF_HOME:-}

if [[ -z "$API_KEY" || "$API_KEY" == local-token || "$API_KEY" == replace-with-* ]]; then
  print -u2 "Set a non-placeholder VLLM_API_KEY in .env before starting the server."
  print -u2 "Run ./scripts/init_env.zsh to generate a safe local configuration."
  exit 2
fi

if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
  print -u2 "Adapter not found at $ADAPTER_PATH"
  exit 2
fi

args=(
  serve "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --tokenizer-revision "$MODEL_REVISION"
  --served-model-name "$BASE_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --dtype bfloat16
  --generation-config vllm
  --seed 42
  --quantization bitsandbytes
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser xlam
  --chat-template "$ROOT_DIR/templates/tool_chat_template_xlam_qwen.jinja"
  --enable-lora
  --lora-modules "$ADAPTER_NAME=$ADAPTER_PATH"
  --max-loras 1
  --max-lora-rank "$MAX_LORA_RANK"
)

if [[ -n "$DOWNLOAD_DIR" ]]; then
  args+=(--download-dir "$DOWNLOAD_DIR")
fi
if [[ ${ENFORCE_EAGER:-1} == 1 ]]; then
  args+=(--enforce-eager)
fi

exec vllm "${args[@]}"
