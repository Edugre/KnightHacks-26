#!/usr/bin/env bash
# One command to run the app.
#   ./run.sh          GPU server: starts the resident model server (if not up), then the API + test UI on :8000
#   ./run.sh stub     laptop / no GPU: stub detector, no models, API + test UI on :8000
#   ./run.sh test     run the test suite (no GPU needed)
# On AUP Learning Cloud, open the UI at  https://dub.aupcloud.io/user/<your-user>/proxy/8000/
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs data
PORT="${PORT:-8000}"
MODEL_PORT="${CENSOR_MODEL_PORT:-8100}"

case "${1:-gpu}" in
  test)
    exec python3 -m pytest -q ;;
  stub)
    export CENSOR_STUB=1
    unset CENSOR_MODEL_URL || true
    echo "stub mode: upload a clip from scripts/make_test_clip.py plus its .tracks.json sidecar"
    exec python3 -m uvicorn censor.server.app:app --host 0.0.0.0 --port "$PORT" ;;
  gpu)
    export CENSOR_MODEL_URL="http://127.0.0.1:$MODEL_PORT"
    if ! curl -sf "$CENSOR_MODEL_URL/health" >/dev/null; then
      echo "starting model server (first load takes ~3-4 min; log: logs/model_server.log)"
      nohup python3 -m censor.server.model_server > logs/model_server.log 2>&1 &
      for _ in $(seq 1 120); do
        sleep 5
        curl -sf "$CENSOR_MODEL_URL/health" >/dev/null && break
      done
    fi
    curl -sf "$CENSOR_MODEL_URL/health" && echo || { echo "model server not healthy; see logs/model_server.log" >&2; exit 1; }
    exec python3 -m uvicorn censor.server.app:app --host 0.0.0.0 --port "$PORT" ;;
  *)
    echo "usage: ./run.sh [gpu|stub|test]" >&2; exit 2 ;;
esac
