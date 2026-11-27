"""The live AI schema rejects actions outside the cell vocabulary, as the offline replay does."""

import importlib
import json
from pathlib import Path

from xair.ai.structured_intent import ACTION_TYPES

ROOT = Path(__file__).resolve().parents[1]


def _server(monkeypatch):
    monkeypatch.setenv("XAIR_SCHEMA", "action-intent-v1-ai.json")
    import xair.adapters.http_server as h
    return importlib.reload(h)


def test_ai_schema_vocabulary_matches_offline_action_types():
    schema = json.loads((ROOT / "schemas" / "action-intent-v1-ai.json").read_text())
    enum = schema["allOf"][1]["properties"]["payload"]["properties"]["action_type"]["enum"]
    assert tuple(enum) == ACTION_TYPES


def test_unknown_action_is_a_schema_violation(monkeypatch):
    h = _server(monkeypatch)
    body = json.loads((ROOT / "examples" / "manufacturing-resume-stale.json").read_text())
    assert h._schema_error(body) is None
    body["payload"] = dict(body["payload"], action_type="RESUMED")
    assert "action_type" in (h._schema_error(body) or "")
    monkeypatch.delenv("XAIR_SCHEMA")
    importlib.reload(h)
