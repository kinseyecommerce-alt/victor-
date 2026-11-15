"""
approvals.py — semi-auto approval mode (default OFF): a human approves a strategy's entries before they go out.

Intended for the first LIVE weeks (the kept strategies refuse LIVE today, so in PAPER it stays OFF and Monday's
paper trading is unchanged). Granularity: ONE ENTRY STAGE of one strategy on one day — the batch the rule decides
at its fixed time (auction fade 09:15:05, PEAD 09:15:10, strangle 10:20, attention 15:21). A per-order approval is
not possible for these rules: the auction fade's fill is the 09:15 auction print one second after the decision.

    * With APPROVAL_MODE on (setting, or toggled through the v2 API and persisted in APPROVALS_FILE), an entry stage
      proceeds only if today's ticket for that strategy is APPROVED; otherwise the entry skips with
      "awaiting_approval" (or "approval_rejected" / "approval_expired").
    * Tickets are opened automatically (pending) by the 08:55 preflight and by the first gated entry, or through
      POST /api/v2/approvals. Each ticket has a deadline: the strategy's entry time that day (DEADLINES) unless
      given; a ticket still pending at its deadline expires. An approved ticket covers that strategy's entries for
      that day only.
    * HALTED (global, strategy or brake) cancels pending tickets.
    * Exits never need an approval.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

from loguru import logger

from ist_clock import now_ist

DEADLINES = {"auc_fade": dtime(9, 16), "pead_h60": dtime(9, 16), "vrp_winged": dtime(10, 21),
             "attn_on": dtime(15, 24)}
DEFAULT_DEADLINE = dtime(15, 30)
_lock = threading.Lock()
_cache: Optional[dict] = None
_cache_path: Optional[Path] = None


def path() -> Path:
    return Path(os.getenv("APPROVALS_FILE")
                or Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
                .with_name("approvals.json"))


def reload() -> None:
    global _cache
    _cache = None


def _load() -> dict:
    global _cache, _cache_path
    p = path()
    if _cache is not None and _cache_path == p:
        return _cache
    d = {"mode": None, "tickets": {}}
    try:
        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            d["mode"] = raw.get("mode")
            d["tickets"] = raw.get("tickets") or {}
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.error("[approvals] {} unreadable ({}) — no tickets", p, exc)
    _cache, _cache_path = d, p
    return d


def _save(d: dict) -> None:
    p = path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # keep the file small: the last 200 tickets
        if len(d["tickets"]) > 200:
            keep = sorted(d["tickets"].values(), key=lambda t: t.get("created", ""))[-200:]
            d["tickets"] = {t["id"]: t for t in keep}
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".approvals.")
        with os.fdopen(fd, "w") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, p)
    except Exception as exc:
        logger.error("[approvals] not saved: {}", exc)


def mode_on() -> bool:
    d = _load()
    if d.get("mode") is not None:
        return bool(d["mode"])
    from config import settings
    return bool(getattr(settings, "approval_mode", False))


def set_mode(on: bool, by: str = "operator") -> dict:
    with _lock:
        d = _load()
        d["mode"] = bool(on)
        _save(d)
    logger.warning("[approvals] semi-auto approval mode {} (by {})", "ON" if on else "OFF", by)
    return status()


def _deadline(strategy: str, day: str, deadline: Optional[str] = None) -> str:
    if deadline:
        dt = datetime.fromisoformat(deadline)
        if dt.tzinfo is None:                     # a deadline without a timezone is IST (the exchange clock)
            dt = dt.replace(tzinfo=now_ist().tzinfo)
        return dt.isoformat(timespec="seconds")
    t = DEADLINES.get(strategy, DEFAULT_DEADLINE)
    return datetime.combine(datetime.fromisoformat(day).date(), t, tzinfo=now_ist().tzinfo).isoformat(timespec="seconds")


def _aware(dt: datetime, ref: datetime) -> datetime:
    """A stored deadline without a timezone (written before this fix) is read as IST, like new ones."""
    return dt.replace(tzinfo=ref.tzinfo or now_ist().tzinfo) if dt.tzinfo is None else dt


def _expire(d: dict, now: datetime) -> None:
    for t in d["tickets"].values():
        if t["status"] == "pending" and _aware(datetime.fromisoformat(t["deadline"]), now) < now:
            t["status"], t["decided_at"], t["decided_by"] = "expired", now.isoformat(timespec="seconds"), "deadline"


def ticket_for(strategy: str, day: str) -> Optional[dict]:
    d = _load()
    hits = [t for t in d["tickets"].values() if t["strategy"] == strategy and t["day"] == day]
    return sorted(hits, key=lambda t: t["created"])[-1] if hits else None


def request(strategy: str, day: Optional[str] = None, deadline: Optional[str] = None, note: str = "",
            by: str = "system") -> dict:
    """Open (or return) the pending ticket for (strategy, day). Idempotent while one is pending / approved."""
    now = now_ist()
    day = day or now.date().isoformat()
    with _lock:
        d = _load()
        _expire(d, now)
        cur = ticket_for(strategy, day)
        if cur and cur["status"] in ("pending", "approved"):
            return dict(cur)
        t = {"id": uuid.uuid4().hex[:12], "strategy": strategy, "day": day, "stage": "entries", "status": "pending",
             "created": now.isoformat(timespec="seconds"), "deadline": _deadline(strategy, day, deadline),
             "requested_by": by, "note": note[:200], "decided_at": None, "decided_by": None}
        d["tickets"][t["id"]] = t
        _save(d)
    logger.info("[approvals] ticket {} opened: {} entries on {} (deadline {})", t["id"], strategy, day, t["deadline"])
    return dict(t)


def decide(ticket_id: str, approve: bool, by: str = "operator", note: str = "") -> dict:
    now = now_ist()
    with _lock:
        d = _load()
        _expire(d, now)
        t = d["tickets"].get(ticket_id)
        if t is None:
            raise KeyError(ticket_id)
        if t["status"] != "pending":
            raise ValueError(f"ticket is {t['status']}")
        t["status"] = "approved" if approve else "rejected"
        t["decided_at"], t["decided_by"] = now.isoformat(timespec="seconds"), by[:60]
        if note:
            t["note"] = note[:200]
        _save(d)
    logger.warning("[approvals] ticket {} {} by {} ({} {})", ticket_id, t["status"], by, t["strategy"], t["day"])
    return dict(t)


def cancel_pending(why: str, strategy: Optional[str] = None) -> int:
    now = now_ist()
    n = 0
    with _lock:
        d = _load()
        for t in d["tickets"].values():
            if t["status"] == "pending" and (strategy is None or t["strategy"] == strategy):
                t["status"], t["decided_at"], t["decided_by"], t["note"] = \
                    "cancelled", now.isoformat(timespec="seconds"), "system", why[:200]
                n += 1
        if n:
            _save(d)
    if n:
        logger.warning("[approvals] {} pending ticket(s) cancelled: {}", n, why)
    return n


def entry_gate(strategy: str, create: bool = True, now: Optional[datetime] = None) -> Optional[str]:
    """None when the strategy's entries may go out (mode off, or today's ticket approved); else the skip reason.
    create=False (status reads) never opens a ticket."""
    if not mode_on():
        return None
    now = now or now_ist()
    day = now.date().isoformat()
    with _lock:
        d = _load()
        _expire(d, now)
        t = ticket_for(strategy, day)
    if t is None:
        if create:
            request(strategy, day)
        return "awaiting_approval"
    return {"approved": None, "pending": "awaiting_approval", "rejected": "approval_rejected",
            "expired": "approval_expired", "cancelled": "approval_cancelled"}.get(t["status"], "awaiting_approval")


def status() -> dict:
    now = now_ist()
    with _lock:
        d = _load()
        _expire(d, now)
        tickets = sorted(d["tickets"].values(), key=lambda t: t["created"], reverse=True)[:50]
    return {"mode": mode_on(), "granularity": "one entry stage of one strategy per day",
            "deadlines": {k: v.strftime("%H:%M") for k, v in DEADLINES.items()},
            "pending": [t for t in tickets if t["status"] == "pending"], "recent": tickets}
