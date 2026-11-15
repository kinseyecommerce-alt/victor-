"""
paper_book.py — the PAPER book survives a server restart.

In PAPER mode every simulated order and position lives in memory: kite_client's paper orders / positions, the
validated agents' own positions and day state (export_book / import_book), and each agent's day counters. This
module writes one JSON snapshot of that state (atomic tmp + replace) whenever it changes — checked every
PAPER_BOOK_SAVE_INTERVAL_S and immediately after a position closes — and restores it at startup.

* A snapshot of today's session is restored fully.
* A snapshot of an earlier session (restart after midnight, weekend): its order book is history; open paper rows
  are restored and each validated agent decides what its restored positions need (attention / strangle / PEAD
  keep holding, the auction fade closes a missed exit at the snapshot's last price — import_book(stale=True)).
* A position the performance ledger already closed (closed after the last snapshot) is not restored twice.

Snapshots written before the core rebuild also carry blocks of removed engines ("tracked", "tsl", "ai", "risk",
"guard", legacy agents' day state). They are ignored — never an error — and the first restore that finds them
keeps a copy of the file as paper_book.pre-rebuild.json. Their paper rows stay in the book (an MIS / NRML row is
flattened by the 15:15 square-off like any other leftover; a CNC row stays) and are listed as orphans in status().

Never in LIVE mode (the broker holds the real book there).
Setting: PERSIST_PAPER_BOOK (default on). File: PAPER_BOOK_FILE (default next to performance.json).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import date, time as dtime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

SCHEMA = 2
LEGACY_BLOCKS = ("tracked", "tsl", "ai", "risk", "guard")

_restore_attempted = False      # nothing is written before the startup restore ran (it would overwrite the snapshot)
_last_body: Optional[str] = None
_save_task: Optional[asyncio.Task] = None
last_restore: dict = {}
# The trading session the in-memory day counters belong to: from DAY_START (09:15) to the next day's 09:15.
_book_day: Optional[date] = None
DAY_START = dtime(9, 15)
_WORKING = ("TRIGGER PENDING", "OPEN")


def path() -> Path:
    return Path(os.getenv("PAPER_BOOK_FILE")
                or Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
                .parent / "paper_book.json")


def enabled() -> bool:
    return bool(settings.persist_paper_book) and settings.trading_mode == "PAPER"


def _agents() -> dict:
    from agents.strategy_agents import ALL_AGENTS
    return ALL_AGENTS


# ── Snapshot ──────────────────────────────────────────────────────────────────

def snapshot() -> dict:
    from kite_client import kite_client
    agents, validated = {}, {}
    for name, a in _agents().items():
        try:
            day_state = a.export_day_state()
        except Exception as exc:
            logger.debug("[paper_book] {} day state not exported: {}", name, exc)
            day_state = {}
        agents[name] = {"trades_today": a.state.trades_today, "pnl_today": a.state.pnl_today,
                        "signals_fired": a.state.signals_fired, "skip_counts": dict(a.state.skip_counts),
                        "last_skip": a.state.last_skip, "day_state": day_state}
        if getattr(a, "scheduled", False) and hasattr(a, "export_book"):
            try:
                validated[name] = a.export_book()
            except Exception as exc:
                logger.error("[paper_book] {} book not exported: {}", name, exc)
    return {
        "validated": validated,
        "schema": SCHEMA,
        "session": now_ist().date().isoformat(),
        "book_day": (_book_day or session_day()).isoformat(),
        "mode": settings.trading_mode,
        "kite": {"orders": list(kite_client._paper_orders), "positions": list(kite_client._paper_positions),
                 "gtts": {str(k): v for k, v in kite_client._paper_gtts.items()}},
        "agents": agents,
    }


def save(force: bool = False) -> bool:
    """Write the snapshot if the book changed since the last write. PAPER only, and only once the startup
    restore has run. Returns True when a file was written."""
    global _last_body
    if not enabled() or not _restore_attempted:
        return False
    try:
        snap = snapshot()
        body = json.dumps(snap, default=str, sort_keys=True)
        if body == _last_body and not force:
            return False
        f = path()
        f.parent.mkdir(parents=True, exist_ok=True)
        out = json.dumps({**snap, "saved_at": now_ist().isoformat(timespec="seconds")}, default=str)
        fd, tmp = tempfile.mkstemp(dir=f.parent, prefix=".paper_book.")
        with os.fdopen(fd, "w") as fh:
            fh.write(out)
        os.replace(tmp, f)
        _last_body = body
        return True
    except Exception as exc:
        logger.error("[paper_book] snapshot not written: {}", exc)
        return False


# ── Restore ───────────────────────────────────────────────────────────────────

def _closed_ids(since: str) -> dict:
    """Entry order ids the performance ledger recorded on or after `since` → their ledger row."""
    from performance import performance
    return {t["oid"]: t for t in performance.trades if t.get("oid") and str(t.get("t", ""))[:10] >= since[:10]}


def restore() -> dict:
    """Startup: re-load the paper book. Never raises; returns a summary."""
    global _restore_attempted, last_restore, _book_day
    _book_day = session_day()
    try:
        out = _restore()
    except Exception as exc:                      # a broken snapshot must never stop the server
        logger.error("[paper_book] restore failed, starting with an empty paper book: {}", exc)
        _quarantine()
        out = {"restored": False, "reason": f"error: {str(exc)[:200]}"}
        _book_day = session_day()
    _restore_attempted = True
    last_restore = out
    if out.get("restored") and roll_day_if_due():
        out["day_reset"] = True
    return out


def session_day(now=None) -> date:
    """The trading session a moment belongs to: from 09:15 to the next 09:15."""
    now = now or now_ist()
    return now.date() if now.time() >= DAY_START else now.date() - timedelta(days=1)


def _parse_day(v) -> Optional[date]:
    try:
        return date.fromisoformat(str(v)) if v else None
    except ValueError:
        return None


def mark_day_reset(now=None) -> None:
    global _book_day
    _book_day = session_day(now)


def roll_day_if_due(now=None) -> bool:
    """Reset the agents' day counters once the session they belong to is over (PAPER, persistence on)."""
    if not enabled() or not _restore_attempted:
        return False
    cur = session_day(now)
    if _book_day == cur:
        return False
    prev = _book_day
    mark_day_reset(now)
    for a in _agents().values():
        a.reset_daily()
    logger.warning("[paper_book] new trading session {} — day counters of {} reset", cur, prev)
    save(force=True)
    return True


def _quarantine() -> None:
    try:
        f = path()
        if f.exists():
            os.replace(f, f.with_suffix(".corrupt.json"))
    except Exception:
        pass


def _keep_pre_rebuild_copy(raw: str) -> None:
    try:
        f = path().with_name("paper_book.pre-rebuild.json")
        if not f.exists():
            f.write_text(raw)
            logger.warning("[paper_book] snapshot from before the core rebuild: copy kept at {}", f)
    except Exception as exc:
        logger.warning("[paper_book] could not keep the pre-rebuild snapshot: {}", exc)


def _restore() -> dict:
    global _book_day
    if settings.trading_mode != "PAPER":
        return {"restored": False, "reason": "LIVE mode: the broker holds the book"}
    if not enabled():
        return {"restored": False, "reason": "disabled (PERSIST_PAPER_BOOK=false)"}
    f = path()
    try:
        raw = f.read_text()
    except FileNotFoundError:
        return {"restored": False, "reason": "no snapshot"}
    try:
        d = json.loads(raw)
        if not isinstance(d, dict):
            raise ValueError("snapshot is not an object")
    except Exception as exc:
        logger.error("[paper_book] snapshot unreadable ({}) — ignored", exc)
        _quarantine()
        return {"restored": False, "reason": f"unreadable: {str(exc)[:120]}"}
    legacy = [k for k in LEGACY_BLOCKS if d.get(k)]
    if legacy:
        _keep_pre_rebuild_copy(raw)
    today = now_ist().date().isoformat()
    stale = d.get("session") != today
    if stale:
        try:                                   # keep it: the next save overwrites it
            f.with_name(f"{f.stem}.{d.get('session') or 'old'}.json").write_text(raw)
        except Exception as exc:
            logger.warning("[paper_book] could not keep the old snapshot: {}", exc)
        s_day = _parse_day(d.get("session"))
        if s_day is None or s_day > now_ist().date():
            return {"restored": False, "reason": f"stale snapshot from {d.get('session')}"}
    if d.get("mode", "PAPER") != "PAPER":
        return {"restored": False, "reason": "snapshot not from PAPER mode"}

    from kite_client import kite_client
    k = d.get("kite") or {}
    orders = [dict(o) for o in k.get("orders") or [] if isinstance(o, dict)]
    positions = [dict(p) for p in k.get("positions") or [] if isinstance(p, dict)]
    gtts = {}
    for gid, g in (k.get("gtts") or {}).items():
        try:
            gtts[int(gid)] = dict(g)
        except (TypeError, ValueError):
            continue
    if stale:
        orders = [o for o in orders if o.get("status") in _WORKING]
        positions = [p for p in positions if p.get("quantity")]
    closed = _closed_ids(str(d.get("book_day") or d.get("session") or today))

    if kite_client._paper_orders or kite_client._paper_positions:
        logger.warning("[paper_book] paper book not empty at restore — merging")
    kite_client._paper_orders[:0] = orders
    kite_client._paper_positions[:0] = positions
    for gid, g in gtts.items():
        kite_client._paper_gtts.setdefault(gid, g)
    _book_day = _parse_day(d.get("book_day")) or _parse_day(d.get("session"))

    agents = _agents()
    ignored = sorted(n for n in (d.get("agents") or {}) if n not in agents)
    for name, st in (d.get("agents") or {}).items():
        a = agents.get(name)
        if not a or not isinstance(st, dict):
            continue
        a.state.trades_today = int(st.get("trades_today", 0))
        a.state.pnl_today = float(st.get("pnl_today", 0.0))
        a.state.signals_fired = int(st.get("signals_fired", 0))
        a.state.skip_counts.update(st.get("skip_counts") or {})
        a.state.last_skip = st.get("last_skip", "")
        try:
            a.import_day_state(st.get("day_state") or {})
        except Exception as exc:
            logger.warning("[paper_book] {} day state not restored: {}", name, exc)
    for name, a in agents.items():
        if getattr(a, "scheduled", False) and hasattr(a, "import_book"):
            try:
                a.import_book((d.get("validated") or {}).get(name, {}), closed=set(closed), stale=stale)
            except Exception as exc:
                logger.error("[paper_book] {} book not restored: {}", name, exc)

    orphans = orphan_rows()
    msg = (f"restored {len(positions)} paper row(s), validated positions: "
           + ", ".join(f"{n} {len(a.positions())}" for n, a in agents.items() if hasattr(a, "positions"))
           + (f"; ignored blocks of removed engines: {', '.join(legacy)}" if legacy else "")
           + (f"; ignored removed agents: {', '.join(ignored)}" if ignored else "")
           + (f"; orphan rows (no kept owner): {', '.join(r['tradingsymbol'] for r in orphans)}" if orphans else ""))
    logger.warning("[paper_book] {} (snapshot {})", msg, d.get("saved_at"))
    return {"restored": True, "stale": stale, "saved_at": d.get("saved_at"), "legacy_blocks_ignored": legacy,
            "agents_ignored": ignored, "orphans": [r["tradingsymbol"] for r in orphans], "message": msg}


def orphan_rows() -> list[dict]:
    """Open paper rows no kept agent owns (positions of engines removed in the rebuild)."""
    from kite_client import kite_client
    owned: dict[tuple[str, str], int] = {}
    for a in _agents().values():
        for p in (a.positions() if hasattr(a, "positions") else []):
            k = (p.symbol, p.product)
            owned[k] = owned.get(k, 0) + p.qty * p.sign
    out = []
    for r in kite_client._paper_positions:
        q = int(r.get("quantity") or 0)
        if q and q != owned.get((r.get("tradingsymbol"), r.get("product") or "MIS"), 0):
            out.append(r)
    return out


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def _save_loop() -> None:
    while True:
        await asyncio.sleep(max(0.5, float(settings.paper_book_save_interval_s)))
        try:
            roll_day_if_due()
        except Exception as exc:
            logger.error("[paper_book] day rollover failed: {}", exc)
        save()


def start() -> None:
    """After restore(): periodic snapshots."""
    global _save_task
    if not enabled():
        return
    if not _save_task or _save_task.done():
        _save_task = asyncio.create_task(_save_loop())


def status() -> dict:
    f = path()
    try:
        orphans = [{k: r.get(k) for k in ("tradingsymbol", "product", "quantity", "average_price", "last_price")}
                   for r in orphan_rows()]
    except Exception:
        orphans = []
    return {"enabled": enabled(), "file": str(f), "exists": f.exists(), "book_day": str(_book_day or ""),
            "last_restore": last_restore, "orphan_rows": orphans}
