"""Regression tests for the fixes made during the 2026-09 academic review."""

from __future__ import annotations

from experiments.paper2_common import evidence_age_ms
from experiments.run_b5_agent_policy import (
    aggregate_episodes,
    fixed_chooser,
    run_policy_on_frames,
    single_shot_chooser,
)
from experiments.run_paper2_stats import holm_adjust, paired_frame_contrast
from xair.ai.structured_intent import precondition_syntax_ok
from xair.core.context_validator import ContextValidator


class _Rec:
    def __init__(self, frame_id, model, *, latency=3000.0, correct=False):
        self.frame_id = frame_id
        self.use_case = "uc2_restart"
        self.model = model
        self.prompt_variant = "blind"
        self.latency_ms = latency
        self.gt_action = "STOP"
        self.model_action = "STOP" if correct else "RESUME"
        self.grounding_correct = correct
        self.n_preconditions = 1
        self.schema_valid = True
        self.gate_schema_valid = True
        self.severity = "minor"
        self.defect_present = True
        self.category = "cable"
        self.preconds_hold_nominal = True
        self.preconds_hold_drifted = False


def _frames(n=60):
    models = ("qwen2.5vl:7b", "qwen2.5vl:32b", "gemma3:12b", "llama3.2-vision:11b")
    return [{m: _Rec(f"f{i}", m, correct=(i % 3 == 0)) for m in models} for i in range(n)]


def test_evidence_age_uses_wall_clock_after_retry():
    rec = {"latency_ms": 2000.0, "capture_ts": "2026-08-12T08:00:00.000Z",
           "emitted_ts": "2026-08-12T08:02:00.000Z"}
    assert evidence_age_ms(rec) == 120000.0
    rec["emitted_ts"] = "2026-08-12T08:00:02.004Z"  # bookkeeping overhead only
    assert evidence_age_ms(rec) == 2000.0
    assert evidence_age_ms({"latency_ms": 1500.0}) == 1500.0


def test_type_mismatch_fails_closed():
    ok, reason = ContextValidator({"line": {"state": "RUN"}})._check("line.state < 5")
    assert not ok and reason.startswith("ill_typed")


def test_bare_equals_only_normalised_after_path():
    v = ContextValidator({"line": {"state": "RUN"}, "x": {"v": "a=b"}})
    assert v._check("line.state='RUN'")[0]
    assert v._check("x.v == 'a=b'")[0]
    assert precondition_syntax_ok("line.state='RUN'")
    assert precondition_syntax_ok("x.v == 'a=b'")


def test_b5_common_random_numbers_first_shot_identical():
    """Drift is drawn per (seed, frame): first attempts coincide across policies."""
    frames = _frames()
    setting = {"label": "t", "p_drift": 0.5, "drift_offset_ms": 250}
    a = run_policy_on_frames(frames, policy_name="ss", choose_action=single_shot_chooser(),
                             setting=setting, seed=3)
    b = run_policy_on_frames(frames, policy_name="ro", choose_action=fixed_chooser("reobserve"),
                             setting=setting, seed=3)
    drift_a = {r["frame_id"]: r["plant_drifted"] for r in a}
    drift_b = {r["frame_id"]: r["plant_drifted"] for r in b}
    assert drift_a == drift_b


def test_b5_wrr_denominator_is_first_shot_opportunity():
    frames = _frames()
    setting = {"label": "t", "p_drift": 0.0, "drift_offset_ms": 250}
    rows = run_policy_on_frames(frames, policy_name="ss", choose_action=single_shot_chooser(),
                                setting=setting, seed=1)
    agg = aggregate_episodes(rows)
    assert 0.0 <= agg["WRR"] <= 1.0


def test_paired_contrast_and_holm():
    a = [{"frame_id": f"f{i}", "reward": 1.0, "successful_actuation": 1,
          "hazardous_publish": 0, "wrongful_revoke": 0} for i in range(50)]
    b = [{"frame_id": f"f{i}", "reward": 0.0, "successful_actuation": 0,
          "hazardous_publish": 0, "wrongful_revoke": 0} for i in range(50)]
    out = paired_frame_contrast(a, b, n_boot=200, seed=0)
    assert out["fields"]["U"]["mean_delta_a_minus_b"] == 1.0
    assert out["fields"]["U"]["ci_excludes_zero"]
    adj = holm_adjust({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == 0.03 and adj["z"] == 0.06 and adj["y"] == 0.06
