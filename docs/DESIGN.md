# Design rationale: AI-native action intents

This note summarizes the design choices behind the evaluation. Full protocols and results of
the secondary suites are in [SECONDARY_RESULTS.md](SECONDARY_RESULTS.md).

## Problem

Vision-language models (VLMs) and tool-using agents can propose actions for industrial
cyber-physical systems, but a plausible model output is not the same as an action that is
admissible to publish now. Between evidence capture and middleware publication the plant state
can change, and inference itself takes several seconds.

## Approach

AI producers emit Action Intent Schema (AIS) intents: a typed action, explicit
preconditions, evidence provenance, and a validity anchor. They never publish to plant
middleware directly; every intent passes through XAIR before publication, which validates the intent at
t_v and rechecks freshness, predicates, and the read-set context version at the gateway
(t_g, the release) before middleware publication (t_m); evidence capture is t_e. The runtime is shared with the companion study
(LD1). XAIR does not improve perceptual accuracy; it bounds authorization to publish.

Three design choices are operationalized:

1. **Evidence anchoring.** Freshness is measured from evidence capture, so slow inference
   consumes the freshness budget instead of being invisible to the gate.
2. **Stochastic drift.** Plant volatility (probability and timing of context invalidation)
   is a controlled factor, so admissibility is measured rather than guaranteed.
3. **Validity budgets.** A fixed, learned, or oracle policy selects the freshness window
   and precondition strictness to trade successful release against adverse publication
   and wrongful revocation.

## Suites

| Suite | Question |
|-------|----------|
| B1 | Blind grounding, label-leakage control, precondition quality (five VLMs; corrected prompts for all, repeatability, constrained decoding, and Qwen3-VL 8B) |
| B2 | Validity frontier: freshness × drift probability × timing × anchor; paired contrasts of direct, freshness-only, schema+freshness, and full XAIR publication |
| Replays | Latency factorial; Poisson and job-shop drift timing; hidden-path drift with an integrator-declared predicate; queueing delay between validation and release; controlled detector producer |
| B3 | Learned validity budgets (LinUCB, tabular Q-learning) vs. a train-selected fixed budget |
| B4 | Post-hoc selection among preserved model outputs (adaptive and fixed window) |
| B5 | Post-revocation policy simulation (abstain / retry / re-observe / escalate) |

The legacy A1 pilot (unconditional drift) is kept only to document why naive VLM–gate
experiments saturate.

## What the evaluation does and does not show

- Offline replay of cached perception; drift is simulated; one software-in-the-loop check against a simulated OPC UA cell, no physical cell.
- Several effects are analytic consequences of the design (anchor blind spot offline,
  B3 convergence to revoke-all at λ = 5 when drift precedes submission, the B5 adverse-rate bound) and are reported as such.
- Whether validated publication pays at all follows from two measured rates: a path beats
  revoking every intent iff λ < λ* = (1 + μ) SAR / Adverse (the break-even weight).
- XAIR is a publication-boundary authorization layer, not a replacement for safety PLCs,
  interlocks, control barrier functions, or certified fallback controllers.
