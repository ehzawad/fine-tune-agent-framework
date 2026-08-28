# Source this file only from repository scripts after ROOT_DIR has been set.
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi
