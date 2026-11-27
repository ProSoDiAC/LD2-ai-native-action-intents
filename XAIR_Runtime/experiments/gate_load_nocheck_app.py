"""Scratch launcher: the gate with the read-set version comparison at commit disabled (mutation control)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xair.adapters import http_server
from xair.adapters.runtime_state import store

_orig = store.commit_authorization

def patched(intent_id, paths, expected, record, **kw):
    return _orig(intent_id, [], 0, record, **kw)  # empty read set: observed==expected==0, version check never fails

store.commit_authorization = patched
app = http_server.app
