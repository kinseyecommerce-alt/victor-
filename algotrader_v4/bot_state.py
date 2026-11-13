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

_DEFAULTS = {"intraday": True, "fno": True, "swing": True, "scalping": True,
             "auc_fade": True, "attn_on": True, "vrp_winged": True}
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


# An operator Stop (/bot/stop) of the whole bot, persisted until the next /bot/start. The validated agents are set
# running by platform_scheduler (08:55 preflight, restart catch-up), which honours it.
_STOP_FILE = Path(os.getenv("BOT_STOP_FILE") or _FILE.with_name("bot_stopped.json"))


def _load_stopped() -> bool:
    try:
        return bool(json.loads(_STOP_FILE.read_text()).get("stopped"))
    except Exception:
        return False


_operator_stopped: bool = _load_stopped()


def operator_stopped() -> bool:
    return _operator_stopped


def set_operator_stopped(val: bool) -> None:
    global _operator_stopped
    _operator_stopped = bool(val)
    try:
        _STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STOP_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stopped": _operator_stopped}))
        os.replace(tmp, _STOP_FILE)
    except Exception:
        pass
