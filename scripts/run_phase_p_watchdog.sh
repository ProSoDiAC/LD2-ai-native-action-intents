#!/usr/bin/env bash
# Watchdog for Phase P: if the perception cache stops growing, bounce Ollama and resume.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XAIR="$ROOT/XAIR_Runtime"
CACHE="$XAIR/experiments/results/perception_cache/phase_p.jsonl"
LOG="${PHASE_P_LOG:-/tmp/phase_p_watchdog.log}"
STALL_SECS="${STALL_SECS:-600}"          # retries+runner respawn can exceed 3 minutes
POLL_SECS="${POLL_SECS:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"

MODELS=(qwen2.5vl:3b llama3.2-vision:11b gemma3:12b qwen2.5vl:32b)
# Keep 7b gaps for a later pass; they were hanging on blind_noctx timeouts.

source "$ROOT/config/compute-gpu.env" 2>/dev/null || true
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"
export OLLAMA_TIMEOUT_S="${OLLAMA_TIMEOUT_S:-60}"
export OLLAMA_MAX_RETRIES="${OLLAMA_MAX_RETRIES:-2}"
export OLLAMA_IMAGE_MAX_SIDE="${OLLAMA_IMAGE_MAX_SIDE:-1024}"
export PYTHONPATH="$XAIR:${PYTHONPATH:-}"

PY="${XAIR}/.venv/bin/python"
[ -x "$PY" ] || PY=python3

mkdir -p "$(dirname "$CACHE")"
touch "$CACHE"

cache_ok_lines() {
  "$PY" - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/perception_cache/phase_p.jsonl")
n = 0
if p.is_file():
    for line in p.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("error"):
            n += 1
print(n)
PY
}

strip_errors() {
  "$PY" - <<'PY'
import json
from pathlib import Path
src = Path("experiments/results/perception_cache/phase_p.jsonl")
if not src.is_file():
    raise SystemExit(0)
tmp = src.with_suffix(".tmp")
kept = dropped = 0
with src.open() as fin, tmp.open("w") as fout:
    for line in fin:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            dropped += 1
            continue
        fout.write(json.dumps(r) + "\n")
        kept += 1
tmp.replace(src)
print(f"[watchdog] cache kept={kept} dropped_errors={dropped}", flush=True)
PY
}

restart_ollama() {
  echo "[watchdog] restarting ollama $(date -Is)" | tee -a "$LOG"
  # Ollama runs on this host (the L40 node); -n fails fast instead of prompting.
  sudo -n systemctl restart ollama >>"$LOG" 2>&1 || true
  sleep 5
  for _ in $(seq 1 30); do
    if curl -sf --max-time 3 "${OLLAMA_HOST}/api/tags" >/dev/null; then
      echo "[watchdog] ollama healthy" | tee -a "$LOG"
      return 0
    fi
    sleep 2
  done
  echo "[watchdog] ollama still unhealthy" | tee -a "$LOG"
  return 1
}

start_campaign() {
  strip_errors
  cd "$XAIR"
  nohup "$PY" experiments/perception_cache.py \
    --models "${MODELS[@]}" \
    --variants blind \
    --primary-model qwen2.5vl:3b \
    --tag phase_p \
    >>"$LOG" 2>&1 &
  echo $! > /tmp/phase_p_campaign.pid
  echo "[watchdog] started campaign pid=$(cat /tmp/phase_p_campaign.pid)" | tee -a "$LOG"
}

campaign_alive() {
  local pid
  pid="$(cat /tmp/phase_p_campaign.pid 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

cache_lines() {
  cache_ok_lines
}

status() {
  echo "[watchdog] $1" | tee -a "$LOG"
}

cd "$XAIR"
echo "[watchdog] boot $(date -Is) stall=${STALL_SECS}s ctx=$OLLAMA_NUM_CTX" | tee -a "$LOG"
status "Phase P watchdog active (num_ctx=$OLLAMA_NUM_CTX, timeout=${OLLAMA_TIMEOUT_S}s); resuming missing models."

restarts=0
if ! campaign_alive; then
  restart_ollama || true
  start_campaign
fi

last_lines="$(cache_lines)"
last_change="$(date +%s)"

while true; do
  sleep "$POLL_SECS"
  now="$(date +%s)"
  lines="$(cache_lines)"

  if [ "$lines" -ge 18000 ]; then
    echo "[watchdog] cache complete ($lines) $(date -Is)" | tee -a "$LOG"
    status "Phase P OK: cache at $lines rows (target 18000)."
    break
  fi

  # Done when no pending work for our model list (approx: 7b kept + 4*2000).
  # Prefer checking process exit + log.
  if ! campaign_alive; then
    # Check whether perception_cache exited cleanly
    if grep -q '"jobs_run"' "$LOG" 2>/dev/null && ! campaign_alive; then
      # May have finished its planned jobs (only missing models)
      echo "[watchdog] campaign process exited; checking residual work $(date -Is)" | tee -a "$LOG"
      start_campaign
      # If immediately nothing pending it will exit fast again
      sleep 20
      if ! campaign_alive; then
        echo "[watchdog] no more pending jobs for selected models" | tee -a "$LOG"
        status "Phase P (missing models) finished; cache=$lines. Fill 7B gaps separately if needed."
        break
      fi
    else
      echo "[watchdog] campaign died unexpectedly; restarting" | tee -a "$LOG"
      restarts=$((restarts + 1))
      if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        status "FAILED: too many restarts ($restarts)."
        exit 1
      fi
      restart_ollama || true
      start_campaign
      last_lines="$(cache_lines)"
      last_change="$(date +%s)"
      continue
    fi
  fi

  if [ "$lines" -gt "$last_lines" ]; then
    echo "[watchdog] progress $last_lines -> $lines $(date -Is)" | tee -a "$LOG"
    last_lines="$lines"
    last_change="$now"
    continue
  fi

  stalled=$((now - last_change))
  if [ "$stalled" -ge "$STALL_SECS" ]; then
    echo "[watchdog] STALL ${stalled}s at $lines lines; bouncing $(date -Is)" | tee -a "$LOG"
    status "Stalled for ${stalled}s at $lines rows; restarting Ollama and resuming."
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
      status "FAILED: too many restarts ($restarts)."
      exit 1
    fi
    pid="$(cat /tmp/phase_p_campaign.pid 2>/dev/null || true)"
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null || true
    restart_ollama || true
    start_campaign
    last_lines="$(cache_lines)"
    last_change="$(date +%s)"
  fi
done
