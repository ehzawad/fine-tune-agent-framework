#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR=${0:A:h:h}
cd "$ROOT_DIR"
PYTHON_BIN=${PYTHON_BIN:-python3}
ENV_DIR=${SERVE_ENV_DIR:-.venv-serve}

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(f"Python 3.10-3.13 is required; found {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/serve.txt
python -m pip install -e '.[dev]'
python - <<'PY'
import importlib.metadata
import torch
import vllm

print("serving environment ready")
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda runtime", torch.version.cuda)
print("vllm-bnb-plugin", importlib.metadata.version("vllm-bnb-plugin"))
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu", props.name, f"{props.total_memory / 1024**3:.2f} GiB")
PY
