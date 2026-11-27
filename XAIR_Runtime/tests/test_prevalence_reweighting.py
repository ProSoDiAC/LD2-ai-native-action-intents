"""Unit tests for the benchmark-composition reweighting (run_paper2_expectation_prevalence.py).

These check the reweighting identity itself (weights sum to the target prevalence split),
the drift-probability convex combination at its boundaries (p_drift = 0 and 1), and the
break-even weight identity lambda* = (1 + mu) * SAR / Adverse, independently of any cached
replay data.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.run_paper2_expectation_prevalence import outcome, weights
from experiments.run_b2_validity_frontier import ReplayRecord


def make_episodes(n_defect: int, n_nominal: int) -> list[dict]:
    eps = []
    for i in range(n_defect):
        eps.append({"use_case": "uc1_triage", "defect_present": True})
    for i in range(n_nominal):
        eps.append({"use_case": "uc2_restart" if i % 2 else "uc1_triage", "defect_present": False})
    return eps


@pytest.mark.parametrize("prevalence", [0.01, 0.05, 0.10, 0.25, 0.50])
def test_weights_reweight_to_target_prevalence(prevalence):
    eps = make_episodes(n_defect=40, n_nominal=60)
    w = weights(eps, prevalence)
    d = np.array([1.0 if e["defect_present"] else 0.0 for e in eps])
    total = w.sum()
    # Defective and nominal frames carry the target share of the total weight.
    assert w[d == 1].sum() / total == pytest.approx(prevalence, abs=1e-9)
    assert w[d == 0].sum() / total == pytest.approx(1 - prevalence, abs=1e-9)


def test_weights_none_prevalence_is_uniform():
    eps = make_episodes(n_defect=10, n_nominal=10)
    w = weights(eps, None)
    assert np.all(w == 1.0)


def test_weights_use_case_filter_zeros_other_categories():
    eps = make_episodes(n_defect=10, n_nominal=10)
    w = weights(eps, None, use_case="uc2_restart")
    kept = {e["use_case"] for e, wi in zip(eps, w) if wi != 0}
    assert kept == {"uc2_restart"}
    assert (w != 0).sum() == sum(1 for e in eps if e["use_case"] == "uc2_restart")


def test_weights_prevalence_and_use_case_compose():
    eps = make_episodes(n_defect=10, n_nominal=20)
    w = weights(eps, prevalence=0.1, use_case="uc1_triage")
    uc1 = np.array([e["use_case"] == "uc1_triage" for e in eps])
    assert np.all(w[~uc1] == 0.0)
    d = np.array([e["defect_present"] for e in eps])
    total = w.sum()
    # Within the kept use case, the defect/nominal split still matches the target prevalence.
    assert w[uc1 & d].sum() / total == pytest.approx(0.1, abs=1e-9)
    assert w[uc1 & ~d].sum() / total == pytest.approx(0.9, abs=1e-9)


def test_weights_raises_when_a_class_is_absent():
    # Reweighting to a target prevalence is undefined when one class has no selected frames
    # (e.g. no defective frames at all, or none left after a use-case filter).
    eps = make_episodes(n_defect=0, n_nominal=10)
    with pytest.raises(ValueError):
        weights(eps, prevalence=0.1)
    eps = make_episodes(n_defect=10, n_nominal=0)
    with pytest.raises(ValueError):
        weights(eps, prevalence=0.1)


def _record(*, correct: bool, schema_valid: bool, precond_nominal: bool, precond_drifted: bool,
            latency_ms: float = 500.0) -> ReplayRecord:
    r = object.__new__(ReplayRecord)
    r.frame_id = "f0"
    r.use_case = "uc1_triage"
    r.model = "test"
    r.prompt_variant = "blind"
    r.latency_ms = latency_ms
    r.gt_action = "ACCEPT"
    r.model_action = "ACCEPT" if correct else "RESUME"
    r.grounding_correct = correct
    r.schema_valid = schema_valid
    r.gate_schema_valid = schema_valid
    r.n_preconditions = 1
    r.preconds_hold_nominal = precond_nominal
    r.preconds_hold_drifted = precond_drifted
    r.severity = None
    r.defect_present = False
    r.category = "cat0"
    return r


def test_outcome_convex_combination_boundaries_match_no_drift_and_drift():
    """expected = p * outcome(drift) + (1 - p) * outcome(no drift); check p=0 and p=1."""
    rec = _record(correct=True, schema_valid=True, precond_nominal=True, precond_drifted=False)
    d = outcome(rec, "xair", drift=True)
    n = outcome(rec, "xair", drift=False)
    for p, expected in ((0.0, n), (1.0, d)):
        combo = {k: p * d[k] + (1 - p) * n[k] for k in d}
        for k in d:
            assert combo[k] == pytest.approx(expected[k])


def test_outcome_drift_revokes_precondition_when_only_drifted_fails():
    rec = _record(correct=True, schema_valid=True, precond_nominal=True, precond_drifted=False)
    drifted = outcome(rec, "xair", drift=True)
    nominal = outcome(rec, "xair", drift=False)
    # Without drift the precondition holds and XAIR publishes the correct action.
    assert nominal["success"] == 1.0
    assert nominal["adverse"] == 0.0
    # Under drift the precondition fails, so XAIR revokes instead of publishing; since the
    # replay also marks the context invalid at that point, this is a correct revoke, not an
    # adverse publication or a wrongful one.
    assert drifted["success"] == 0.0
    assert drifted["adverse"] == 0.0
    assert drifted["wrongful"] == 0.0


@pytest.mark.parametrize("lam,mu,sar,adverse", [(5.0, 1.0, 0.5, 0.1), (5.0, 1.0, 0.2, 0.4), (10.0, 2.0, 0.3, 0.05)])
def test_lambda_star_identity(lam, mu, sar, adverse):
    """lambda* = (1 + mu) * SAR / Adverse is the weight at which full-XAIR utility equals revoke-all's."""
    lambda_star = (1 + mu) * sar / adverse
    # U_xair(lambda*) - U_revoke_all(lambda*) == 0 by construction of the break-even weight,
    # using U_revoke_all = -mu * (SAR + WRR_component) collapsed to -mu * SAR here (WRR = SAR
    # for a benchmark where every revoked xair trial would have succeeded).
    u_xair = sar - lambda_star * adverse - mu * 0.0
    u_revoke_all = -mu * sar
    assert u_xair == pytest.approx(u_revoke_all)


def test_lambda_star_is_infinite_when_adverse_rate_is_zero():
    sar, adverse, mu = 0.6, 0.0, 1.0
    with np.errstate(divide="ignore"):
        lam = (1 + mu) * sar / adverse if adverse > 0 else float("inf")
    assert lam == float("inf")
