#!/usr/bin/env bash
# Start the live Paper-2 stack: XAIR HTTP API (:$XAIR_PORT, default 8080) + adapter
# (:$ADAPTER_HTTP_PORT HTTP, default 9092). Experiments reach it via PAPER2_XAIR / PAPER2_ADAPTER.
# Used only by the live checks (B2 fidelity, gate-latency microbenchmark,
# verify_e2e.sh); all B1-B5 headline numbers are offline replays and need no stack.
# The adapter publishes to ROS 2 only if rclpy is importable; otherwise it runs HTTP-only.
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_resolve_layout.sh"
PID_DIR="$REPO_ROOT/.run"
mkdir -p "$PID_DIR"

# Ports are configurable because 8080 may be taken on shared hosts.
XAIR_PORT="${XAIR_PORT:-8080}"
ADAPTER_WS_PORT="${ADAPTER_WS_PORT:-9091}"
ADAPTER_HTTP_PORT="${ADAPTER_HTTP_PORT:-9092}"
export XAIR_URL="${XAIR_URL:-http://127.0.0.1:$XAIR_PORT}"
PY="${XAIR_ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY=python3

echo "=== XAIR + adapter stack ==="
echo "  XAIR_Runtime: $XAIR_ROOT"

alive() { [ -f "$PID_DIR/$1.pid" ] && kill -0 "$(cat "$PID_DIR/$1.pid")" 2>/dev/null; }

if alive xair; then
  echo "[OK] XAIR already running"
else
  echo "Starting XAIR on :$XAIR_PORT..."
  # No subshell: $! must be the uvicorn process itself so stop_full_stack.sh can kill it.
  pushd "$XAIR_ROOT" >/dev/null
  # Paper 2 validates AI intents against the AIS AI extension (cell action vocabulary).
  XAIR_SCHEMA="${XAIR_SCHEMA:-action-intent-v1-ai.json}" nohup "$PY" -m uvicorn xair.adapters.http_server:app \
    --host 127.0.0.1 --port "$XAIR_PORT" < /dev/null > "$PID_DIR/xair.log" 2>&1 &
  echo $! > "$PID_DIR/xair.pid"
  popd >/dev/null
fi

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf "$XAIR_URL/v1/metrics" >/dev/null; then
    echo "[OK] XAIR /v1/metrics"
    break
  fi
  sleep 1
done

if alive adapter; then
  echo "[OK] adapter already running"
else
  echo "Starting adapter ($ADAPTER_WS_PORT WS / $ADAPTER_HTTP_PORT HTTP)..."
  nohup env XAIR_URL="$XAIR_URL" XAIR_ALLOW_MODE_OVERRIDE="${XAIR_ALLOW_MODE_OVERRIDE:-1}" PYTHONPATH="$XAIR_ROOT:$SCRIPTS" \
    "$PY" "$SCRIPTS/adaptix_quest_adapter.py" "$ADAPTER_WS_PORT" "$ADAPTER_HTTP_PORT" < /dev/null > "$PID_DIR/adapter.log" 2>&1 &
  echo $! > "$PID_DIR/adapter.pid"
  sleep 2
fi

echo ""
echo "Stack ready:"
echo "  XAIR API: $XAIR_URL"
echo "  Adapter:  http://127.0.0.1:$ADAPTER_HTTP_PORT (/intent /context /health)"
echo "  Verify:   $SCRIPTS/verify_e2e.sh"
echo "  Logs:     $PID_DIR/"
