"""
bot_state.py — shared mutable runtime state (no circular imports).
Both main.py and master_agent_v5.py import from here.

Agent enables are persisted (AGENT_ENABLES_FILE, default next to the performance ledger) so a
strategy switched off from the dashboard stays off across restarts and redeploys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULTS = {"intraday": True, "fno": True, "swing": True, "scalping": True, "attn_on": True}
LEGACY_AGENTS = ("intraday", "scalping", "swing", "fno")
# lab-protocol agents (pre-registered DISCOVERY H1/H2 screen + one FINAL look), driven by platform_scheduler jobs
VALIDATED_AGENTS = ("auc_fade", "attn_on", "vrp_winged")
_FILE = Path(os.getenv("AGENT_ENABLES_FILE")
             or Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
             .with_name("agent_enables.json"))


def _load() -> dict[str, bool]:
    try:
        saved = json.loads(_FILE.read_text())
        return {k: bool(saved.get(k, v)) for k, v in _DEFAULTS.items()}
    except Exception:
        return dict(_DEFAULTS)


_agent_enabled: dict[str, bool] = _load()


def _save() -> None:
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_agent_enabled, indent=1))
        os.replace(tmp, _FILE)
    except Exception:
        pass                                   # in-memory state still applies for this process


def is_agent_enabled(name: str) -> bool:
    return _agent_enabled.get(name, True)


def set_agent_enabled(name: str, val: bool) -> None:
    if name in _agent_enabled:
        _agent_enabled[name] = bool(val)
        _save()
