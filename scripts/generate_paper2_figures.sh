#!/usr/bin/env bash
# Regenerate the Paper-2 manuscript figures from experiments/results/.
# Output: journal/figures/ when the (gitignored) manuscript folder exists,
# otherwise XAIR_Runtime/experiments/results/figures/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XAIR="$ROOT/XAIR_Runtime"
if [ -d "$ROOT/journal" ]; then
  FIG="$ROOT/journal/figures"
else
  FIG="$XAIR/experiments/results/figures"
fi
cd "$XAIR"
PY="${XAIR}/.venv/bin/python"
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$XAIR${PYTHONPATH:+:$PYTHONPATH}"

echo "== Paper 2 figures -> $FIG =="
"$PY" experiments/plot_paper2_figures.py --out "$FIG"
"$PY" experiments/plot_drift_rate_curve.py
"$PY" experiments/plot_break_even.py
echo "Done."
[ -d "$ROOT/journal" ] && echo "Rebuild the PDF with: make -C journal"
