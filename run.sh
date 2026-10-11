#!/usr/bin/env bash
# One command to run the app.
#   ./run.sh          GPU server: starts the resident model server (if not up), then the API + web UI on :8000
#   ./run.sh stub     laptop / no GPU: stub detector, no models, API + web UI on :8000
#   ./run.sh test     run the test suite (no GPU needed)
#   ./run.sh bench    benchmark -> bench/report.md (extra args go to bench/run_bench.py; CENSOR_STUB=1 for no GPU)
# On AUP Learning Cloud, open the UI at  https://dub.aupcloud.io/user/<your-user>/proxy/8000/
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs data
PORT="${PORT:-8000}"
MODEL_PORT="${CENSOR_MODEL_PORT:-8100}"

# Inference never touches the network (IMPLEMENTATION.md success condition 4). Models load from the caches
# scripts/setup.sh fills, and censor/netguard.py refuses and logs anything else to logs/egress.log.
# CENSOR_ALLOW_NETWORK=1 skips this, e.g. to fetch a model setup.sh doesn't know about.
offline() {
  [ "${CENSOR_ALLOW_NETWORK:-0}" = 1 ] && return
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=true CENSOR_OFFLINE=1
}

case "${1:-gpu}" in
  test)
    exec python3 -m pytest -q ;;
  stub)
    export CENSOR_STUB=1
    unset CENSOR_MODEL_URL || true
    echo "stub mode: upload a clip from scripts/make_test_clip.py plus its .tracks.json sidecar"
    exec python3 -m uvicorn censor.server.app:app --host 0.0.0.0 --port "$PORT" ;;
  gpu)
    offline
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
    if [ "${CENSOR_OFFLINE:-0}" = 1 ] && ! curl -sf "$CENSOR_MODEL_URL/health" | grep -q '"offline":true'; then
      echo "warning: the running model server was started without the egress guard; restart it to cover it" >&2
    fi
    exec python3 -m uvicorn censor.server.app:app --host 0.0.0.0 --port "$PORT" ;;
  bench)
    shift
    offline
    # Use the resident model server for the agent and VLM checks when it's up; it is not started here.
    curl -sf "http://127.0.0.1:$MODEL_PORT/health" >/dev/null && export CENSOR_MODEL_URL="http://127.0.0.1:$MODEL_PORT"
    exec python3 -m bench.run_bench "$@" ;;
  *)
    echo "usage: ./run.sh [gpu|stub|test|bench]" >&2; exit 2 ;;
esac
