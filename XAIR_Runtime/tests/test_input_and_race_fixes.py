"""Regression tests: lock ownership, commit/revoke atomicity, and HTTP input validation."""

from __future__ import annotations

import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("REDIS_URL", "")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from xair.adapters.http_server import app, store  # noqa: E402
from xair.core.models import ActionIntent, IntentState  # noqa: E402
from xair.core.runtime import XAIRRuntime  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from xair_http_client import XAIRHttpClient  # noqa: E402

client = TestClient(app)


def _body(target=None, **extra):
    return {"id": str(uuid.uuid4()), "source": "ai", "timestamp_decision": datetime.now(timezone.utc).isoformat(),
            "freshness_window_ms": 5000, "preconditions": [{"expr": "line.state == 'RUN'"}],
            "payload": {"action_type": "RESUME", "target_entity": target or f"t_{uuid.uuid4().hex[:6]}"}, **extra}


def _authorized(target=None):
    client.post("/v1/context/snapshot", json={"line": {"state": "RUN"}})
    body = _body(target)
    out = client.post("/v1/intents", json=body).json()
    assert out["state"] == "AUTHORIZED", out
    return body["id"]


def _rt_intent(target, **kw):
    return ActionIntent.from_dict({**_body(target), **kw})


def test_release_by_non_owner_keeps_the_target_lock():
    rt = XAIRRuntime(context={"line": {"state": "RUN"}})
    a, b, c = (_rt_intent("T1") for _ in range(3))
    assert rt.process_intent(a).state == IntentState.AUTHORIZED
    assert rt.process_intent(b).state == IntentState.DELAYED
    rt.supervisory_revoke(b.id)
    assert rt.process_intent(c).state == IntentState.DELAYED
    rt.confirm_publication(a.id, True, "ok")
    d = _rt_intent("T1")
    assert rt.process_intent(d).state == IntentState.AUTHORIZED


def test_revoke_racing_a_commit_waits_for_the_lock_and_the_commit_is_logged(monkeypatch):
    intent_id = _authorized()
    real = store.commit_authorization

    def commit(*a, **kw):
        t = threading.Thread(target=lambda: client.delete(f"/v1/intents/{intent_id}"))
        t.start()
        t.join(0.5)
        commit.thread = t
        return real(*a, **kw)

    monkeypatch.setattr(store, "commit_authorization", commit)
    before = len(store.actuation_log())
    r = client.post(f"/v1/intents/{intent_id}/commit")
    commit.thread.join(5)
    assert r.status_code == 200 and r.json()["released"] is True
    assert client.get(f"/v1/intents/{intent_id}").json()["state"] == "EXECUTED"
    assert len(store.actuation_log()) == before + 1


def test_duplicate_commit_closes_the_authorized_intent_and_frees_the_target():
    intent_id = _authorized("t_dup")
    real = store.commit_authorization

    def twice(*a, **kw):
        real(*a, **kw)
        return real(*a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(store, "commit_authorization", twice)
        before = len(store.actuation_log())
        r = client.post(f"/v1/intents/{intent_id}/commit")
    assert r.status_code == 200 and r.json()["status"] == "duplicate" and r.json()["released"] is False
    assert len(store.actuation_log()) == before + 1
    assert client.get(f"/v1/intents/{intent_id}").json()["state"] == "EXECUTED"
    assert client.post(f"/v1/intents/{intent_id}/commit").status_code == 409
    assert client.post("/v1/intents", json=_body("t_dup")).json()["state"] == "AUTHORIZED"


def test_concurrent_commit_and_revoke_are_consistent():
    for _ in range(10):
        intent_id = _authorized()
        before = len(store.actuation_log())
        out = {}
        ts = [threading.Thread(target=lambda: out.__setitem__("c", client.post(f"/v1/intents/{intent_id}/commit"))),
              threading.Thread(target=lambda: out.__setitem__("d", client.delete(f"/v1/intents/{intent_id}")))]
        [t.start() for t in ts]
        [t.join(10) for t in ts]
        assert out["c"].status_code in (200, 409) and out["d"].status_code in (200, 409)
        logged = len(store.actuation_log()) - before
        state = client.get(f"/v1/intents/{intent_id}").json()["state"]
        assert (logged, state) in ((1, "EXECUTED"), (0, "REVOKED"))


def test_concurrent_commits_log_once_and_keep_lifecycle():
    intent_id = _authorized()
    before = len(store.actuation_log())
    res = []
    ts = [threading.Thread(target=lambda: res.append(client.post(f"/v1/intents/{intent_id}/commit").status_code))
          for _ in range(4)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    assert sorted(res) == [200, 409, 409, 409]
    assert len(store.actuation_log()) == before + 1
    assert client.get(f"/v1/intents/{intent_id}").json()["state"] == "EXECUTED"


@pytest.mark.parametrize("ts", ["not-a-date", "2026-13-45T99:00:00Z", "2026-01-01T00:00:00", 12])
def test_malformed_timestamp_is_rejected_not_500(ts):
    body = {**_body(), "timestamp_decision": ts}
    r = client.post("/v1/intents", json=body)
    assert r.status_code == 200 and r.json()["outcome"] == "REVOKE" and "timestamp_decision" in r.json()["reason"]
    assert client.get(f"/v1/intents/{body['id']}").status_code == 404
    batch = client.post("/v1/intents/batch", json=[body])
    assert batch.status_code == 200 and batch.json()["results"][0]["outcome"] == "REVOKE"


@pytest.mark.parametrize("q", ["not json", "{\"a\": 1}", "[1, 2]"])
def test_bad_predicates_query_is_422(q):
    assert client.get("/v1/context/snapshot", params={"predicates": q}).status_code == 422
    assert client.get("/v1/context/snapshot", params={"predicates": '["line.state == \'RUN\'"]'}).status_code == 200


def test_commit_request_is_validated():
    intent_id = _authorized()
    for bad in ('{"scope": "zzz"}', '{"margin_ms": -1}', '{"margin_ms": NaN}', '{"margin_ms": Infinity}'):
        assert client.post(f"/v1/intents/{intent_id}/commit", content=bad,
                           headers={"Content-Type": "application/json"}).status_code == 422
    assert client.get(f"/v1/intents/{intent_id}").json()["state"] == "AUTHORIZED"


def test_withheld_for_unknown_intent_is_404_and_not_logged():
    before = len(store.actuation_log())
    assert client.post("/v1/intents/nope/withheld", json={"reason": "x"}).status_code == 404
    assert len(store.actuation_log()) == before
    intent_id = _authorized()
    assert client.post(f"/v1/intents/{intent_id}/withheld", json={"reason": "x"}).status_code == 200


def test_negative_actuation_start_is_422():
    assert client.get("/v1/actuations", params={"start": -1}).status_code == 422


def test_client_quotes_intent_id_in_the_path():
    seen = []
    c = XAIRHttpClient("http://x")
    c._request = lambda method, path, body=None, timeout=15.0: seen.append(path) or {}
    c.commit("a/b?c#d")
    c.withheld("a/b?c#d", "r")
    c.report_publication("a/b?c#d", True, "r")
    assert all("a%2Fb%3Fc%23d" in p for p in seen) and len(seen) == 3
