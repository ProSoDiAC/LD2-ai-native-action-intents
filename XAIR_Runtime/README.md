# XAIR Runtime

Execution-time validation layer for AI-driven industrial cyber-physical systems. Producers
(here: vision-language models) submit Action Intent Schema (AIS) intents; XAIR checks
schema, resource conflicts, freshness/deadline, contextual preconditions and any
operator-mandated predicates (`PUT /v1/policy`); before release, the gateway re-checks the
read-set or predicate-truth version (optimistic mode) or asks XAIR for an atomic
authorization commit (`POST /v1/intents/{id}/commit`, log at `GET /v1/actuations`).

This directory is the Paper-2 artifact (publication-time validation of VLM proposals). Reproduction of the
offline B1-B5 analyses from the Phase-P perception cache is described in
[PAPER2_ARTIFACT.md](PAPER2_ARTIFACT.md).

**Trusted network only, no authentication:** the HTTP API and the adapter (bind `127.0.0.1` by default, `XAIR_BIND_HOST` to change; `?mode=` other than the default needs `XAIR_ALLOW_MODE_OVERRIDE=1`) must not be exposed to untrusted hosts.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
./scripts/reproduce_paper2_offline.sh      # needs experiments/results/perception_cache/phase_p.jsonl
```

| Path | Content |
|------|---------|
| `xair/core/` | intent receiver, temporal and context validators, coordinator, lifecycle FSM, runtime |
| `xair/ai/` | Ollama client and VLM-output to AIS intent builder |
| `xair/adapters/` | FastAPI HTTP server (`/v1/intents`, `/v1/intents/{id}/commit`, `/v1/actuations`, `/v1/policy`, `/v1/context/snapshot`, `/v1/metrics`) and the actuation-log consumer |
| `schemas/` | AIS v1 and AI-extension JSON schemas |
| `experiments/` | Phase P driver, B1-B5 suites, statistics, figures, results |
| `tests/` | unit and HTTP integration tests |

Repository: https://github.com/ProSoDiAC/LD2-ai-native-action-intents
