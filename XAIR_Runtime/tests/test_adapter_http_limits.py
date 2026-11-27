"""Adapter HTTP hardening: body limits, malformed Content-Length, mode override gate."""

from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import adaptix_quest_adapter as adapter  # noqa: E402


@pytest.fixture(scope="module")
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    threading.Thread(target=adapter.run_http_server, args=(p,), daemon=True).start()
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", p), 0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    return p


def _post(port, path, body=b"{}", headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("POST", path)
    for k, v in (headers or {"Content-Length": str(len(body))}).items():
        conn.putheader(k, v)
    conn.endheaders()
    if body and "Content-Length" in (headers or {"Content-Length": 1}) and (headers is None or headers["Content-Length"].isdigit()):
        conn.send(body[:1 << 16])
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def test_non_numeric_content_length_is_400(port):
    status, out = _post(port, "/intent", b"", {"Content-Length": "abc"})
    assert status == 400 and out["error"] == "invalid_content_length"


def test_oversized_body_is_413(port):
    status, out = _post(port, "/intent", b"", {"Content-Length": str(adapter.MAX_BODY_BYTES + 1)})
    assert status == 413


def test_mode_override_is_forbidden_by_default(port):
    assert adapter.ALLOW_MODE_OVERRIDE is False
    status, out = _post(port, "/intent?mode=direct")
    assert status == 403 and out["error"] == "mode_override_disabled"


def test_mode_override_allowed_when_enabled(port, monkeypatch):
    monkeypatch.setattr(adapter, "ALLOW_MODE_OVERRIDE", True)
    status, _ = _post(port, "/intent?mode=direct")
    assert status != 403
