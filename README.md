# LD2: release-time validation of actions proposed by vision-language models (Paper 2)

Code, data pointers and results for
*Release-Time Validation of Actions Proposed by Vision-Language Models in Cyber-Physical
Systems: Architecture, Replay, and Software-in-the-Loop Evaluation*. Vision-language models (VLMs)
propose actions as AI-native AIS intents; the XAIR runtime validates each one against
current context before publication. On a stratified 2,000-frame VisA / MVTec AD corpus the
evaluation comprises blind grounding (B1), the validity frontier of the publication paths
(B2), a latency factorial, Poisson and job-shop drift timing, drift on context the prompt
does not show, a queueing-delay replay of the gateway recheck, a grammar-constrained decoding
arm and a newer VLM, a PatchCore-style detector as a controlled detector producer, and
three secondary suites (B3-B5). Everything after the one-off perception campaign (Phase P)
is replayed offline from cached model outputs.

The companion *execution-gap* study (AIS/XAIR, experiments E0-E25; arXiv:2610.27971,
[doi:10.48550/arXiv.2610.27971](https://doi.org/10.48550/arXiv.2610.27971)) is
[LD1-execution-gap-xair](https://github.com/ProSoDiAC/LD1-execution-gap-xair); the runtime
in `XAIR_Runtime/xair/` is that runtime plus the AI intent builder.

## Layout

```
LD2/
├── XAIR_Runtime/                 # Python package + experiments (the public artifact)
│   ├── xair/                     # runtime: core gate, AI intent builder, HTTP adapter
│   ├── schemas/                  # AIS JSON schemas (v1, v1-ai)
│   ├── examples/                 # sample intents used by the live smoke test
│   ├── tests/                    # pytest suite
│   ├── experiments/
│   │   ├── paper2_common.py      # shared harness: gate model, classify(), metrics
│   │   ├── perception_cache.py   # Phase P (GPU/Ollama) inference driver
│   │   ├── run_b1_* ... run_b5_* # suites B1-B5 (offline replay)
│   │   ├── run_paper2_stats.py   # bootstrap / McNemar / power
│   │   ├── run_a1_vlm_ais.py     # legacy saturated pilot (quoted once in the paper)
│   │   ├── detector_producer.py  # PatchCore-style detector (GPU, torch; run once)
│   │   ├── run_detector_analysis.py, run_paper2_latency_factorial.py
│   │   ├── rl/                   # LinUCB and Q-learning environments/policies
│   │   ├── datasets/manufacturing-a1/   # manifest (tracked) + builder scripts
│   │   └── results/              # result files behind every paper number (+ SHA-256 manifest)
│   ├── scripts/reproduce_paper2_offline.sh
│   └── PAPER2_ARTIFACT.md        # artifact description and exact commands
├── scripts/                      # live stack, Phase-P watchdog
├── config/                       # *.env.example templates (real *.env are gitignored)
├── docs/DESIGN.md                # design rationale of the study
├── docs/SECONDARY_RESULTS.md     # secondary suites B3-B5 (protocols, full tables) and extra details
├── docs/MODEL_DIGESTS.md         # Ollama tags, digests and quantization of the models
├── .github/workflows/ci.yml      # tests + live HTTP smoke test
└── journal/                      # manuscript (LaTeX), local only, gitignored
```

## Quick start (offline, CPU only)

```bash
cd XAIR_Runtime
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Python >= 3.12
.venv/bin/pytest -q
./scripts/reproduce_paper2_offline.sh    # ~8 min; checks all results against the SHA-256 manifest
../scripts/generate_paper2_figures.sh
```

Nine per-trial outputs listed in the manifest are not stored in the repository because
of their size (4–57 MB each): `b2_headline_trials.csv`, `b3_eval_trials*.jsonl`,
`b4_eval_trials*.jsonl` and `b5_eval_trials*.jsonl`. The reproduction script regenerates
them deterministically and checks them against the manifest. All other result files are
in `XAIR_Runtime/experiments/results/`.

See [`XAIR_Runtime/PAPER2_ARTIFACT.md`](XAIR_Runtime/PAPER2_ARTIFACT.md) for which
script produces which result, and for the archived GPU and live measurements that the
script does not regenerate.

## Live checks (optional)

The B2 fidelity check and the gate-latency microbenchmark use the live HTTP path:

```bash
./scripts/start_full_stack.sh      # XAIR :8080 + adapter :9092 (HTTP-only without ROS 2)
./scripts/verify_e2e.sh            # EXECUTE on RUN, REVOKE on PAUSED
./scripts/stop_full_stack.sh
```

## Phase P (GPU)

Phase P ran once on an NVIDIA L40 with a local Ollama 0.17.7; the offline replay does not
need a GPU. Re-running it (model list, settings, watchdog) is described in
`XAIR_Runtime/PAPER2_ARTIFACT.md`.

## Datasets

VisA (CC BY 4.0) and MVTec AD (CC BY-NC-SA 4.0, non-commercial). Download and sampling
instructions: [`XAIR_Runtime/experiments/datasets/manufacturing-a1/README.md`](XAIR_Runtime/experiments/datasets/manufacturing-a1/README.md).
Raw archives and sampled frames are gitignored.

## Manuscript

`journal/` holds the LaTeX sources; it is not part of the public repository.

## License

Apache-2.0 (code). Dataset licenses as above.
