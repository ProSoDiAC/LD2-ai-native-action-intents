# Paper 2 artifact (B1–B5)

Manuscript: *Release-Time Validation of Actions Proposed by Vision-Language Models in Cyber-Physical
Systems: Architecture, Replay, and Software-in-the-Loop Evaluation*.
Public repository: https://github.com/ProSoDiAC/LD2-ai-native-action-intents

This artifact reproduces every offline (CPU) number of the manuscript from the cached
Phase-P perception outputs. It does not re-run GPU inference and never calls Ollama.

**Trusted network only, no authentication:** the HTTP API and the adapter (bind `127.0.0.1` by default, `XAIR_BIND_HOST` to change; `?mode=` other than the default needs `XAIR_ALLOW_MODE_OVERRIDE=1`) must not be exposed to untrusted hosts.

## Environment

- Python **3.12+** (`pip install -e ".[dev]"`): numpy, scipy, matplotlib, pillow, pytest.
  Bit-identical SHA-256 of the outputs needs Python ≥ 3.12 (float `sum()` changed in 3.12;
  older versions differ in the last ulp, ≤ 3e-14).
- Inputs (tracked in the repository): the perception caches in
  `experiments/results/perception_cache/` (`phase_p.jsonl`, SHA-256 `943fbe80…c38c`, checked by
  the script; `phase_p_v2.jsonl` and `phase_p_rep.jsonl` from the later reruns) and the dataset
  manifest `experiments/datasets/manufacturing-a1/manifest.jsonl`. Image frames are not needed
  offline.

## One command

```bash
cd XAIR_Runtime
./scripts/reproduce_paper2_offline.sh          # ~8 min on 48 cores; PAPER2_FULL=0 skips the slow extras
.venv/bin/python experiments/write_results_manifest.py --check   # also run at the end of the script
../scripts/generate_paper2_figures.sh          # manuscript figures
```

## Script → manuscript mapping

| Result | Script | Output |
|--------|--------|--------|
| 70/30 split (seed 42, 612 test frames) | `paper2_splits.py` | `paper2_frame_split.json` |
| B1 grounding, leakage, precondition quality | `run_b1_grounding.py`, `run_b1_baselines.py` | `b1_grounding.{json,csv}`, `b1_baselines*.{json,csv}` |
| B1 diagnostics (vs. chance/majority under every tie-break, κ, defect-judgement BA, self-catch by syntax) | `run_paper2_revision_analyses.py` | `paper2_revision_analyses.json` |
| GT severity-cut sensitivity | `run_gt_threshold_sensitivity.py` | `gt_threshold_sensitivity.json` |
| B2 exploratory grid and anchor blind spot | `run_b2_validity_frontier.py` | `b2_validity_frontier.{json,csv}` |
| B2 reference cell: bootstrap U and paired frame-cluster contrasts (descriptive) | `run_paper2_stats.py` | `paper2_stats.json` |
| B2 by image source, other cells, harness-echo baseline, gain sources, adverse-publication split (safety-relevant / conservative), revoke-all reference | `run_paper2_revision_analyses.py` | `paper2_revision_analyses.json` |
| B2 precondition-repair sensitivity | `run_b2_repaired_sensitivity.py` | `b2_repaired_comparison.json` |
| B3 learned budgets (+ privileged ablation) | `run_b3_validity_budget.py` | `b3_*` |
| B3 re-selection under other λ / p_drift | `run_paper2_revision_analyses.py` | `paper2_revision_analyses.json` |
| Live recheck injection | `run_live_recheck_injection.py --xair http://127.0.0.1:<port>` | `live_recheck_injection.json` |
| Selective prediction | `run_paper2_selective_prediction.py` | `paper2_selective_prediction.json` |
| Cost-weight sensitivity | `run_paper2_weight_sensitivity.py` | `paper2_weight_sensitivity.json` |
| B3 category hold-out | `build_category_holdout_split.py` + `run_b3_validity_budget.py --no-privileged-severity --split-path … --out-suffix _cat_holdout` | `b3_*_cat_holdout*` |
| B4 post-hoc selection (adaptive / fixed w) | `run_b4_model_routing.py` (`--freshness-policy fixed --out-suffix _fixedw`) | `b4_*`, `b4_*_fixedw*` |
| B5 post-revocation policies, p_recover sweep | `run_b5_agent_policy.py`, `run_b5_precover_sweep.py` | `b5_*` |
| λ, μ re-weighting | `run_utility_sensitivity.py` | `utility_sensitivity.{json,csv}` |
| Latency factorial (fixed and permuted evidence ages) | `run_paper2_latency_factorial.py` | `paper2_latency_factorial.json` |
| Drift timing: Poisson drift times and residual times of a real job-shop event log (one machine, and all machines superposed) (downloaded once from 4TU.ResearchData, DOI 10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399, SHA-256 checked, cached in `datasets/raw/`) | `run_paper2_drift_timing.py` | `paper2_drift_timing.json` |
| Corrected prompts for every model: B1 and B2 reference cell with Phase-P and with own latencies, B3-B5 rerun on the latency-matched decisions | `run_paper2_v2_invariance.py --adaptive` (writes the derived caches `phase_p_v2_matched`, `phase_p_v2_own`) | `paper2_v2_invariance.json`, `b3_*_v2*`, `b4_*_v2*`, `b5_*_v2*` |
| Controlled detector producer (mechanism probe): calibration on training frames, B2 replay at its own and at 7B evidence ages | `run_detector_analysis.py` (reads `results/detector/`) | `perception_cache/detector_patchcore.jsonl`, `detector_analysis.json` |
| Schema+freshness baseline, defect-flag ablation, hidden-path drift with integrator-declared predicate, queueing delay with gateway recheck, drift-rate curve and job-shop operating periods, break-even weight λ* (adverse-publication weight), grounding ceiling and table-mapped accuracy | `run_paper2_review_analyses.py` | `paper2_review_analyses.json` |
| Constrained decoding (7B) and Qwen3-VL 8B on the test frames, B2 reference cell at own and original 7B ages | `run_paper2_constrained_newer.py` | `paper2_constrained_newer.json` |
| Additional contrasts: XAIR minus schema+freshness in every sensitivity row, λ* with over-conservative costs, grounding on decidable frames, recheck split into age and predicate parts with the analytic in-queue drift share, confidence-threshold gate, Qwen3-VL with the integrator-declared predicate | `run_paper2_audit_extras.py` | `paper2_audit_extras.json` |
| Exact expectation over drift at the reference cell (no drift seeds), frame- and category-cluster intervals, Monte Carlo error of the five-seed replay, adverse publications split into safety-relevant and conservative, reweighting to defect prevalences of 1–50% and per use case, perception-free nominal producer and its paired contrast with the 7B model | `run_paper2_expectation_prevalence.py` | `paper2_expectation_prevalence.json` |
| Benign races in the queue (invalidating, predicate-preserving, irrelevant changes): global-version, read-set-version, semantic, and full-XAIR rechecks | `run_paper2_benign_races.py` | `paper2_benign_races.json` |
| Drift-rate figure | `plot_drift_rate_curve.py` | `journal/figures/drift_rate_curve.pdf` (generated locally; `journal/` is not published) |
| Break-even weight λ* per producer (figure) | `plot_break_even.py` | `journal/figures/break_even.pdf` (generated locally; `journal/` is not published) |
| Ordering of the paths versus λ, balanced and 10% prevalence (figure) | `plot_lambda_sensitivity.py` (reads `paper2_expectation_prevalence.json`) | `journal/figures/lambda_sensitivity.pdf` (generated locally; `journal/` is not published) |
| Latency gap per image source and origin of the 7B schema-validity drop (review checks) | `run_paper2_latency_schema_checks.py` | printed to stdout (cited in the Results text) |
| Live check with an OPC UA cell in the loop (7B live on 200 test frames at drift rates 0.05, 0.2 and 1 per s; freshness-only and full XAIR run concurrently on the same drifting world and each is judged by the node value read right after its own commit, with a synchronised-trials subset reported; XAIR gate behind HTTP, OPC UA server and bridge from `../live_cell_ld1clone`) | start a Redis container, `uvicorn xair.adapters.http_server:app --port 18090` with `REDIS_URL`, and `live_cell_ld1clone/simulation/opcua_hil_bridge.py --daemon --port 4841` with `XAIR_URL`; then `run_paper2_live_opcua.py --rate R --out paper2_live_opcua_rR.json` | `paper2_live_opcua_r{0.05,0.2,1.0}.json` |
| B1 grounding with category-cluster inference (27 categories, 10,000 resamples, original and corrected prompts, five models) | `run_paper2_b1_cluster.py` | `paper2_b1_cluster.json` |
| Category-cluster bootstrap of the live OPC UA contrasts | `run_paper2_live_cluster.py` | `paper2_live_cluster.json` |
| Graphical abstract (problem → gate → three outcomes, gain versus drift rate) | `plot_graphical_abstract.py --out ../journal/submission` | `graphical_abstract.{png,pdf}` |
| Figures | `plot_paper2_figures.py` | `journal/figures/` or `results/figures/` |

Notes on quantities quoted in the paper:

- **Self-catch and false-block denominators.** The B1 table of the paper reports both rates on syntactically valid emissions only (`paper2_revision_analyses.json` → `b1_judgement_and_catch.<model>.valid_syntax`). `b1_grounding.json` computes them over all emissions, where a malformed precondition blocks every intent, so its values differ (e.g. Llama 3.2-Vision self-catch 0.765 there, 0.676 in the paper).
- **`defect.absent` in the blind prompt.** `xair/ai/prompts/ais_blind.txt` lists `defect.absent` as a known path, and the replay evaluates it against the ground-truth flag. The bound quoted in the paper comes from `paper2_review_analyses.json` → `defect_signal` (detector flag instead of ground truth, and defect preconditions dropped) and from the emission counts in `paper2_revision_analyses.json` (`emissions_referencing_defect_absent`: 7B 189, Llama 118) and `paper2_v2_analysis.json` (`emits_bare_defect_absent`: 0.0835 of 2,000 = 167 bare paths for the 7B model).
- **Exploratory-analysis intervals.** The ΔU intervals of the exploratory analyses (learned budget, model selection, recovery; tables in `docs/SECONDARY_RESULTS.md`) are paired frame-cluster contrasts (`paper2_stats.json` → `paired_frame_contrasts`, and `paper2_revision_analyses.json` → `b4_best_minus_primary`), not the coarse seed-level bootstrap in `b3_seed_bootstrap`.

## Archived measurements (not regenerated by the script)

| Measurement | How it was produced | File |
|-------------|--------------------|------|
| Phase P (5 VLMs, ~18,000 calls) | `perception_cache.py` on the L40 with Ollama 0.17.7 | `perception_cache/phase_p.jsonl` |
| Corrected prompts (blind-v2, all five models; the 7B run on 2026-09-23, the others on 2026-09-30/10-01 on the shared L40) and repeatability (7B, 400 frames) | `perception_cache.py --models M --primary-model M --variants blind_v2 --tag phase_p_v2`; `--variants blind --limit 400 --tag phase_p_rep`; analysed offline by `run_paper2_prompt_v2_analysis.py` | `perception_cache/phase_p_v2.jsonl`, `phase_p_rep.jsonl`, `paper2_v2_analysis.json` |
| Offline-online consistency (XAIR 1.4.0): 5 seeds on 500 primary-model frames, and 1,990 frames of each of the five models | `run_b2_fidelity.py --seed {1..5} --out-suffix _seed{s}`; `--model M --n 1990 --seed 1 --out-suffix _allframes_M` against the live stack | `b2_fidelity_seed*.{json,csv}`, `b2_fidelity_allframes_*.{json,csv}` |
| Gate latency (6 runs, XAIR 1.4.0) | `run_gate_latency_bench.py --out-suffix _run{r}` against the live stack | `gate_latency_run*.json` |
| Gate load and concurrency test (single uvicorn worker, Redis; 1/4/16/64 clients; 20 ms queue with no writer, an irrelevant-path writer or a writer flipping `line.state`, 5 repeats each) | start Redis and `uvicorn xair.adapters.http_server:app --port 18391` (with `REDIS_URL`, `XAIR_SCHEMA=action-intent-v1-ai.json`), then `PAPER2_XAIR=http://127.0.0.1:18391 run_paper2_gate_load.py` | `paper2_gate_load.json`; mutation control with the commit-time version comparison disabled: serve `experiments/gate_load_nocheck_app.py` (`uvicorn gate_load_nocheck_app:app --app-dir experiments`), same script with `--out paper2_gate_load_nocheck.json` -> `paper2_gate_load_nocheck.json` |
| Grammar-constrained decoding (7B, blind prompt, JSON schema with action enum and precondition grammar) and Qwen3-VL 8B Instruct, 612 test frames, 2026-10-02 on the shared L40 | `perception_cache.py --models qwen2.5vl:7b --variants blind_constrained --split test --tag phase_p_constrained`; `--models qwen3-vl:8b-instruct --primary-model qwen3-vl:8b-instruct --variants blind --split test --tag phase_p_qwen3`; reference image plus constrained decoding: `--models qwen2.5vl:7b --variants blind_ref_constrained --split test --tag phase_p_ref_constrained` | `perception_cache/phase_p_constrained.jsonl`, `phase_p_qwen3.jsonl`, `phase_p_ref_constrained.jsonl` (all analysed by `run_paper2_constrained_newer.py`) |
| 200-frame resolution ablation | `run_b1_downscale_ablation.py` | `b1_downscale_ablation.json` |
| Legacy saturated pilot (A1a–A1d) | `run_a1_vlm_ais.py` | `a1_*` |
| Controlled detector producer (mechanism probe) scores, anomaly maps, latencies (GPU, torch 2.2 / torchvision 0.17) | `detector_producer.py` on the L40 | `detector/patchcore_{scores.jsonl,maps.npz,fit.json}` |

### Re-running Phase P (GPU)

Phase P used Ollama 0.17.7 on one NVIDIA L40 (46 GB), temperature 0, `num_ctx` 8192, images
downscaled to a longest side of 1024 px. The knobs are in `config/compute-gpu.env.example`.
The five models must first be pulled: `qwen2.5vl:7b` (primary), `qwen2.5vl:3b`,
`qwen2.5vl:32b`, `gemma3:12b`, `llama3.2-vision:11b`. All default tags are Q4_K_M quantizations.
The weight digests of the original campaign (2026-08) were not recorded; the Ollama model IDs
installed for the corrected-prompt, constrained-decoding, and Qwen3-VL runs (2026-09/10) are
`qwen2.5vl:3b` fb90415cde1e, `qwen2.5vl:7b` 5ced39dfa4ba, `qwen2.5vl:32b` 3edc3a52fe98,
`gemma3:12b` f4031aab637d, `llama3.2-vision:11b` 6f2f9757ae97, `qwen3-vl:8b-instruct` 0533d74300e4.

```bash
cp ../config/compute-gpu.env.example ../config/compute-gpu.env && source ../config/compute-gpu.env
PYTHONPATH=$PWD .venv/bin/python experiments/perception_cache.py --help
../scripts/run_phase_p_watchdog.sh     # long runs: restarts a stalled local Ollama
```

The 31 blind no-context calls that repeatedly stalled at 1024 px were re-queried at 768 or
512 px by `experiments/fill_blind_noctx_gaps.py` (`perception_cache/blind_noctx_gap_fill.json`);
`phase_p_closure.json` records the final per-model counts and `phase_p.errors.jsonl` the
failed attempts.

Each live file records the host, its load, and the runtime build identifiers of the development
checkout used for the run (`xair_version`, `code_commit`); these are internal identifiers and
do not resolve in this snapshot.
`experiments/summarize_live_runs.py` (run by the reproduce script) aggregates them into
`b2_fidelity_runs.json` and `gate_latency_runs.json`.

The live checks need the HTTP stack: `../scripts/start_full_stack.sh` (ports configurable
with `XAIR_PORT`, `ADAPTER_HTTP_PORT`; point the experiments to it with `PAPER2_XAIR`,
`PAPER2_ADAPTER`).

## Integrity

- `experiments/results/paper2_results_manifest.json`: SHA-256 of every result file; the
  reproduce script checks the regenerated files against it.

## Scope / non-goals

- No physical cell or hardware-in-the-loop campaign: the live check runs the 7B model against a simulated OPC UA cell (software in the loop).
- Phase P GPU inference is not re-run by the reproduce script.
