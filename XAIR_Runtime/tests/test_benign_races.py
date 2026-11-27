"""Release policies of the benign-race replay (run_paper2_benign_races.py)."""

from __future__ import annotations

import pytest

from experiments.run_paper2_benign_races import POLICIES, release


def rel(policy, cls, *, race=True, pred_g=True, passed_v=True, age_ok_g=True):
    return release(policy, passed_v=passed_v, age_ok_g=age_ok_g, race_in_queue=race, cls=cls, pred_g=pred_g)


def test_invalidating_change_is_caught_by_every_recheck():
    for p in POLICIES:
        expected = p == "validate_only"
        assert rel(p, "invalidating", pred_g=False) is expected


def test_predicate_preserving_change_revoked_only_by_version_checks():
    assert rel("semantic", "predicate_preserving") is True
    for p in ("global_version", "readset_version", "full_xair"):
        assert rel(p, "predicate_preserving") is False


def test_irrelevant_change_revoked_only_by_global_version():
    assert rel("global_version", "irrelevant") is False
    for p in ("readset_version", "semantic", "full_xair", "validate_only"):
        assert rel(p, "irrelevant") is True


@pytest.mark.parametrize("p", [p for p in POLICIES if p != "validate_only"])
def test_age_and_validation_bound_every_recheck(p):
    assert rel(p, "irrelevant", race=False, age_ok_g=False) is False
    assert rel(p, "irrelevant", race=False, passed_v=False) is False
