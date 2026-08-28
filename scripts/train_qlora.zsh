#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR=${0:A:h:h}
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_load_env.zsh"
ENV_DIR=${TRAIN_ENV_DIR:-.venv-train}
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  print -u2 "Missing $ENV_DIR. Run ./scripts/setup_train_env.zsh first."
  exit 2
fi
source "$ENV_DIR/bin/activate"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

CONFIG=${CONFIG:-configs/a100_40gb_qlora.yaml}
exec python -m xlam2_ops_agent.training.train --config "$CONFIG" "$@"
