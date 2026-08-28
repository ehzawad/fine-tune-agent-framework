#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR=${0:A:h:h}
cd "$ROOT_DIR"

if [[ -e .env && ${1:-} != --force ]]; then
  print -u2 ".env already exists. Use --force only if replacing it is intentional."
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
API_KEY=$(
  "$PYTHON_BIN" - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
)

sed "s#replace-with-a-long-random-local-token#$API_KEY#" .env.example > .env
chmod 600 .env
print "Created $ROOT_DIR/.env with a random local API key and mode 0600."
print "Review HOST, HF_HOME, and other paths before starting vLLM."
