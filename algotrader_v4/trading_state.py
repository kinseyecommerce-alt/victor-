"""
trading_state.py — Nautilus-style trading states, global and per strategy: ACTIVE | REDUCE_ONLY | HALTED.

    ACTIVE       normal: entries and exits by the rule.
    REDUCE_ONLY  no new entries (an entry stage skips with reason "trading_state:REDUCE_ONLY"; an order that would
                 open or enlarge a position is refused). Every exit the rule schedules — time exits, stops,
                 settlements, the MIS square-off, restore reconciliations — runs exactly as before.
    HALTED       no new entries, pending semi-auto approvals are cancelled, and the order path accepts RISK EXITS
                 ONLY: orders that reduce an open position. Every exit the four kept rules produce is a risk exit
                 (a time stop, a stop-loss, an expiry settlement or a square-off), so open positions still leave by
                 their rule — a halted engine never holds a position past its tested exit, which would add risk the
                 lab never measured. Anything that would increase exposure is refused.

The effective state of a strategy is the most restrictive of: the global state, the strategy's own state and the
portfolio drawdown brake's level (risk_budget.py). Operator states are set through the v2 API (auth-protected),
the brake by the risk budget; both are persisted next to the performance ledger (TRADING_STATE_FILE) and every
change is logged with who / why. Defaults: everything ACTIVE (today's behaviour).

Enforcement points: ValidatedAgent.entry_block_reason() (every entry stage of every kept agent) and the paper
order helpers in validated_base (order-level reduce-only check by the paper row's sign).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

from ist_clock import now_ist

STATES = ("ACTIVE", "REDUCE_ONLY", "HALTED")
_RANK = {s: i for i, s in enumerate(STATES)}
HISTORY_MAX = 200
_lock = threading.Lock()


class TradingStateRefused(RuntimeError):
    """An order the current trading state does not allow (raised by the paper order helpers)."""


def path() -> Path:
    return Path(os.getenv("TRADING_STATE_FILE")
                or Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
                .with_name("trading_state.json"))


def _empty() -> dict:
    return {"global": {"state": "ACTIVE"}, "strategies": {}, "brake": {"state": "ACTIVE"}, "history": []}


_cache: Optional[dict] = None
_cache_path: Optional[Path] = None


def _load() -> dict:
    global _cache, _cache_path
    p = path()
    if _cache is not None and _cache_path == p:
        return _cache
    d = _empty()
    try:
        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            for k in d:
                if k in raw:
                    d[k] = raw[k]
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.error("[trading_state] {} unreadable ({}) — everything ACTIVE", p, exc)
    _cache, _cache_path = d, p
    return d


def _save(d: dict) -> None:
    p = path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".trading_state.")
        with os.fdopen(fd, "w") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, p)
    except Exception as exc:
        logger.error("[trading_state] not saved: {}", exc)


def reload() -> None:
    """Forget the cached file (tests; a file replaced by hand)."""
    global _cache
    _cache = None


def _norm(state: str) -> str:
    s = str(state or "").strip().upper().replace("-", "_")
    if s not in STATES:
        raise ValueError(f"state must be one of {', '.join(STATES)}")
    return s


def _record(d: dict, scope: str, old: str, new: str, reason: str, by: str) -> dict:
    ev = {"at": now_ist().isoformat(timespec="seconds"), "scope": scope, "from": old, "to": new,
          "reason": (reason or "")[:200], "by": (by or "")[:60]}
    d["history"] = (d.get("history") or [])[-(HISTORY_MAX - 1):] + [ev]
    logger.warning("[trading_state] {}: {} → {} ({}; by {})", scope, old, new, reason or "no reason", by or "?")
    return ev


def _entry(state: str, reason: str, by: str) -> dict:
    return {"state": state, "reason": (reason or "")[:200], "by": (by or "")[:60],
            "at": now_ist().isoformat(timespec="seconds")}


def set_global(state: str, reason: str = "", by: str = "operator") -> dict:
    state = _norm(state)
    with _lock:
        d = _load()
        old = d["global"].get("state", "ACTIVE")
        d["global"] = _entry(state, reason, by)
        ev = _record(d, "global", old, state, reason, by)
        _save(d)
    if state == "HALTED":
        _cancel_approvals("global HALTED")
    return ev


def set_strategy(name: str, state: str, reason: str = "", by: str = "operator") -> dict:
    state = _norm(state)
    with _lock:
        d = _load()
        old = (d["strategies"].get(name) or {}).get("state", "ACTIVE")
        d["strategies"][name] = _entry(state, reason, by)
        ev = _record(d, f"strategy:{name}", old, state, reason, by)
        _save(d)
    if state == "HALTED":
        _cancel_approvals(f"{name} HALTED", name)
    return ev


def set_brake(state: str, reason: str = "", by: str = "risk_budget", extra: Optional[dict] = None) -> Optional[dict]:
    """The drawdown brake's level (only risk_budget and the brake-reset endpoint call this)."""
    state = _norm(state)
    with _lock:
        d = _load()
        old = d["brake"].get("state", "ACTIVE")
        if old == state:
            return None
        d["brake"] = {**_entry(state, reason, by), **(extra or {})}
        ev = _record(d, "brake", old, state, reason, by)
        _save(d)
    if state == "HALTED":
        _cancel_approvals("drawdown brake HALTED")
    return ev


def _cancel_approvals(why: str, strategy: Optional[str] = None) -> None:
    try:
        import approvals
        approvals.cancel_pending(why, strategy)
    except Exception as exc:
        logger.debug("[trading_state] approvals not cancelled: {}", exc)


def global_state() -> str:
    return _load()["global"].get("state", "ACTIVE")


def strategy_state(name: str) -> str:
    return (_load()["strategies"].get(name) or {}).get("state", "ACTIVE")


def brake_state() -> str:
    return _load()["brake"].get("state", "ACTIVE")


def effective(name: Optional[str] = None) -> str:
    states = [global_state(), brake_state()] + ([strategy_state(name)] if name else [])
    return max(states, key=lambda s: _RANK.get(s, 0))


def entry_block(name: str) -> Optional[str]:
    """None when the strategy may take new entries, else the skip reason."""
    st = effective(name)
    return None if st == "ACTIVE" else f"trading_state:{st}"


def check_order(name: Optional[str], increases: bool) -> None:
    """Order-level guard: raises TradingStateRefused for an order that would open / enlarge a position while the
    effective state is not ACTIVE. Reducing orders (exits, stops, settlements, square-offs) always pass."""
    if not increases:
        return
    st = effective(name)
    if st != "ACTIVE":
        raise TradingStateRefused(f"{name or 'order'}: {st} — only orders that reduce a position are allowed")


def snapshot() -> dict:
    d = _load()
    try:
        from agents.strategy_agents import ALL_AGENTS
        names = list(ALL_AGENTS)
    except Exception:
        names = list(d["strategies"])
    return {"states": list(STATES), "global": dict(d["global"]), "brake": dict(d["brake"]),
            "strategies": {n: {**(d["strategies"].get(n) or {"state": "ACTIVE"}), "effective": effective(n)}
                           for n in names},
            "history": list(d.get("history") or [])[-50:]}
