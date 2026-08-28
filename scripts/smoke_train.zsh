#!/usr/bin/env zsh
set -euo pipefail
ROOT_DIR=${0:A:h:h}
cd "$ROOT_DIR"
CONFIG=configs/a100_40gb_smoke.yaml exec ./scripts/train_qlora.zsh "$@"
