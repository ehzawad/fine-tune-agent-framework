#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m compileall -q src scripts examples eval tests
python -m pytest -q
PYTHONPATH=src python -m xlam2_ops_agent.training.train \
  --config configs/a100_40gb_smoke.yaml \
  --validate-only
PYTHONPATH=src python -m xlam2_ops_agent demo
python scripts/estimate_memory.py \
  --context 8192 \
  --tensor-parallel 1 \
  --weight-bits 4 \
  --lora-rank 16
