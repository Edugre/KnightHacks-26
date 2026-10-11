#!/usr/bin/env bash
# Start (or restart) the whole backend on the GPU server, in the background: model server on :8100, app on :8000
# with CORS for blurr.photo, and the Cloudflare tunnel to https://api.blurr.photo. Running it again stops all three
# and starts them fresh. Logs go to logs/model_server.log, logs/app.log and logs/tunnel.log.
#   bash scripts/serve.sh                 pull main, restart everything (the VLM loads in seconds on Strix Halo, minutes on slow storage)
#   bash scripts/serve.sh --keep-models   keep a healthy model server running, restart the app and the tunnel
#   bash scripts/serve.sh --no-pull       don't git pull first
#   bash scripts/serve.sh stop            stop all three
# CENSOR_CORS_ORIGINS overrides the allowed origins (comma-separated); PORT and CENSOR_MODEL_PORT the ports.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
mkdir -p logs data
PORT="${PORT:-8000}"
MODEL_PORT="${CENSOR_MODEL_PORT:-8100}"
export CENSOR_CORS_ORIGINS="${CENSOR_CORS_ORIGINS:-https://blurr.photo}"
TUNNEL=blurr

keep_models=0 pull=1 stop_only=0
for a in "$@"; do
  case "$a" in
    --keep-models) keep_models=1 ;;
    --no-pull) pull=0 ;;
    stop) stop_only=1 ;;
    *) echo "usage: bash scripts/serve.sh [--keep-models] [--no-pull] | stop" >&2; exit 2 ;;
  esac
done

# Pids listening on a TCP port
port_pids() { ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u; }
# Stop what matches a pattern and whatever still holds the port, then wait for the port to be free
stop() {
  local name=$1 pattern=$2 port=${3:-}
  if pgrep -f "$pattern" >/dev/null || { [ -n "$port" ] && [ -n "$(port_pids "$port")" ]; }; then
    echo "stopping $name"
    pkill -f "$pattern" || true
    for _ in $(seq 1 20); do
      pgrep -f "$pattern" >/dev/null || { [ -n "$port" ] && [ -n "$(port_pids "$port")" ]; } || return 0
      sleep 0.5
    done
    pkill -9 -f "$pattern" || true
    [ -n "$port" ] && for p in $(port_pids "$port"); do kill -9 "$p" 2>/dev/null || true; done
    sleep 1
  fi
}

stop tunnel "cloudflared tunnel run.* $TUNNEL"
stop app "run\.sh( gpu)?$|uvicorn censor.server.app:app .*--port $PORT" "$PORT"
model_up() { curl -sf "http://127.0.0.1:$MODEL_PORT/health" >/dev/null; }
if [ "$stop_only" = 1 ] || [ "$keep_models" = 0 ] || ! model_up; then
  stop "model server" "censor.server.model_server" "$MODEL_PORT"
fi
[ "$stop_only" = 1 ] && { echo "stopped"; exit 0; }

if [ "$pull" = 1 ]; then
  git pull --ff-only || echo "warning: git pull failed; running the code as it is" >&2
fi
for f in ego_blur_face_gen1.jit ego_blur_lp_gen1.jit; do
  [ -f "${CENSOR_EGOBLUR_DIR:-$HOME/datasets/egoblur}/$f" ] \
    || echo "warning: $f missing from ~/datasets/egoblur; faces and plates fall back to YOLOE (see README)" >&2
done

# run.sh starts the model server if it isn't up, waits for it, then runs the app
echo "starting the app on :$PORT (CORS: $CENSOR_CORS_ORIGINS); the model server loads first if it's down"
nohup bash run.sh gpu > logs/app.log 2>&1 &
app=$!
for _ in $(seq 1 150); do  # up to 12.5 min
  curl -sf "http://127.0.0.1:$PORT/api/config" >/dev/null && break
  kill -0 "$app" 2>/dev/null || { echo "the app exited; last lines of logs/app.log:" >&2; tail -20 logs/app.log >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/api/config" >/dev/null || { echo "the app isn't answering on :$PORT; see logs/app.log" >&2; exit 1; }
echo "app up on :$PORT"

if ! command -v cloudflared >/dev/null; then
  echo "warning: cloudflared not found in ~/bin; skipping the tunnel (setup in README, \"Frontend hosted on its own\")" >&2
  exit 0
fi
nohup cloudflared tunnel run --protocol http2 --url "http://localhost:$PORT" "$TUNNEL" > logs/tunnel.log 2>&1 &
for _ in $(seq 1 12); do
  curl -sf https://api.blurr.photo/api/config >/dev/null && { echo "tunnel up: https://api.blurr.photo"; exit 0; }
  sleep 5
done
echo "warning: https://api.blurr.photo isn't answering yet; see logs/tunnel.log" >&2
