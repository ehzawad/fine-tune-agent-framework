#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR=${0:A:h:h}
cd "$ROOT_DIR"
PYTHON_BIN=${PYTHON_BIN:-python3}
ENV_DIR=${TRAIN_ENV_DIR:-.venv-train}
TORCH_VERSION=${TORCH_VERSION:-2.13.0}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-}

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(f"Python 3.10-3.13 is required; found {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if ! python -c 'import torch' >/dev/null 2>&1; then
  if [[ -n "$PYTORCH_INDEX_URL" ]]; then
    python -m pip install "torch==$TORCH_VERSION" --index-url "$PYTORCH_INDEX_URL"
  else
    python -m pip install "torch==$TORCH_VERSION"
  fi
fi

# torch is installed first so the CUDA wheel choice is not silently replaced.
python -m pip install -e '.[train,dev]'
python - <<'PY'
import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers

print("training environment ready")
print("torch", torch.__version__, "cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu", props.name, f"{props.total_memory / 1024**3:.2f} GiB")
    print("bf16 supported", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY
