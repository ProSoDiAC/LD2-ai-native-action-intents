#!/usr/bin/env bash
# Offline Paper-2 reproduce path (CPU): B1–B5 from the Phase-P perception cache.
# Does NOT re-run Phase P GPU inference and never contacts Ollama.
#
#   PAPER2_FULL=0  skip the slow exploratory steps (B2 frontier, repaired sensitivity,
#                  category hold-out, p_recover sweep, revision analyses); default 1.
#   N_BOOT         bootstrap resamples for run_paper2_stats.py (default 1000).
# After a deliberate change of results, refresh the manifest with
#   .venv/bin/python experiments/write_results_manifest.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TAG="${PAPER2_TAG:-phase_p}"
FULL="${PAPER2_FULL:-1}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

CACHE="experiments/results/perception_cache/${TAG}.jsonl"
MANIFEST="experiments/datasets/manufacturing-a1/manifest.jsonl"
EXPECTED_SHA="943fbe80afd9f421e9a1dbb34ed19f8035ea9b85661debff6d1026f74fc3e38c"

if [[ ! -f "$CACHE" ]]; then
  echo "[reproduce] ERROR: missing perception cache: $CACHE" >&2
  echo "[reproduce] The Phase-P cache is part of the repository; restore it from a fresh copy." >&2
  exit 1
fi
if [[ "$TAG" == "phase_p" ]] && ! echo "$EXPECTED_SHA  $CACHE" | sha256sum -c --quiet - 2>/dev/null; then
  echo "[reproduce] ERROR: $CACHE does not match its published SHA-256 (truncated or modified cache)" >&2
  exit 1
fi
if [[ ! -s "$MANIFEST" ]]; then
  echo "[reproduce] ERROR: missing dataset manifest: $MANIFEST" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
  echo "[reproduce] WARN: Python < 3.12: float sum() differs in the last ulp, so SHA-256 of the" >&2
  echo "           JSON/CSV outputs will not match the manifest (values agree to <=3e-14)." >&2
fi

echo "[reproduce] split"
$PY experiments/paper2_splits.py

echo "[reproduce] B1 baselines + grounding"
$PY experiments/run_b1_baselines.py --tag "$TAG" --no-plot
$PY experiments/run_b1_grounding.py --tag "$TAG"

echo "[reproduce] GT threshold sensitivity"
$PY experiments/run_gt_threshold_sensitivity.py

if [[ "$FULL" == "1" ]]; then
  echo "[reproduce] B2 validity frontier (exploratory grid, ~5 min)"
  $PY experiments/run_b2_validity_frontier.py --tag "$TAG"
  echo "[reproduce] B2 precondition-repair sensitivity"
  $PY experiments/run_b2_repaired_sensitivity.py
fi

echo "[reproduce] B3 validity budget (headline + privileged ablation)"
$PY experiments/run_b3_validity_budget.py --tag "$TAG"

echo "[reproduce] B4 post-hoc selection (test frames, 5 seeds)"
$PY experiments/run_b4_model_routing.py --tag "$TAG"

echo "[reproduce] B5 post-revocation policy simulation"
$PY experiments/run_b5_agent_policy.py --tag "$TAG"

echo "[reproduce] paper2 stats + utility sensitivity"
$PY experiments/run_paper2_stats.py --tag "$TAG" --n-boot "${N_BOOT:-1000}"
$PY experiments/run_utility_sensitivity.py

if [[ "$FULL" == "1" ]]; then
  echo "[reproduce] category hold-out split + B3 on it"
  $PY experiments/build_category_holdout_split.py
  $PY experiments/run_b3_validity_budget.py --tag "$TAG" --no-privileged-severity \
    --split-path experiments/results/paper2_frame_split_category_holdout.json --out-suffix _cat_holdout
  echo "[reproduce] B4 with a fixed w=4000 ms window (latency binds)"
  $PY experiments/run_b4_model_routing.py --tag "$TAG" --freshness-policy fixed --out-suffix _fixedw
  echo "[reproduce] B5 p_recover sweep"
  $PY experiments/run_b5_precover_sweep.py --tag "$TAG"
  echo "[reproduce] revision analyses (stratified, paired, sensitivity)"
  $PY experiments/run_paper2_revision_analyses.py --tag "$TAG"
  echo "[reproduce] corrected-prompt and repeatability analysis (cached revision runs)"
  $PY experiments/run_paper2_prompt_v2_analysis.py
  echo "[reproduce] corrected prompts for every model: B1 and B2 invariance, B3-B5 rerun (suffix _v2)"
  $PY experiments/run_paper2_v2_invariance.py --adaptive
fi

echo "[reproduce] controlled detector producer: calibration on training frames and B2 replay"
$PY experiments/run_detector_analysis.py

echo "[reproduce] latency factorial (evidence age crossed with source, model and correctness)"
$PY experiments/run_paper2_latency_factorial.py

echo "[reproduce] drift timing: Poisson drift times and a real job-shop event log (4TU, SHA-256 checked)"
$PY experiments/run_paper2_drift_timing.py

echo "[reproduce] review analyses: schema baseline, hidden-path drift, recheck delay, drift-rate curve, break-even"
$PY experiments/run_paper2_review_analyses.py
$PY experiments/run_paper2_constrained_newer.py
$PY experiments/run_paper2_audit_extras.py
$PY experiments/run_paper2_b1_cluster.py
$PY experiments/run_paper2_live_cluster.py
$PY experiments/run_paper2_expectation_prevalence.py
$PY experiments/run_paper2_benign_races.py

echo "[reproduce] summaries of the archived live checks (fidelity, gate latency)"
$PY experiments/summarize_live_runs.py


echo "[reproduce] compare with the published SHA-256 manifest"
if [[ -f experiments/results/paper2_results_manifest.json ]]; then
  if ! "$PY" experiments/write_results_manifest.py --check; then
    echo "[reproduce] manifest mismatch (expected only with Python < 3.12 or PAPER2_FULL=0)" >&2
    [[ "${ALLOW_MANIFEST_DIFF:-0}" == "1" ]] || exit 1
  fi
fi

echo "[reproduce] done"
