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
exec python -m xlam2_ops_agent.training.preflight --strict --path "$ROOT_DIR"
