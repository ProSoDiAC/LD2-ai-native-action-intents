#!/usr/bin/env bash
# Stop the stack started by start_full_stack.sh.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$REPO_ROOT/.run"

echo "=== Stop XAIR + adapter ==="
for name in adapter xair; do
  if [ -f "$PID_DIR/$name.pid" ]; then
    pid=$(cat "$PID_DIR/$name.pid")
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_DIR/$name.pid"
    echo "  Stopped $name (pid $pid)"
  fi
done
# Only PID-file processes are stopped: other checkouts (e.g. LD1) may run the same
# programs on this host.
echo "Done."
