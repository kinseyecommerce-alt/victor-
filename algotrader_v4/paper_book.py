"""
paper_book.py — the PAPER book survives a server restart on the same trading day.

In PAPER mode every simulated order, position and stop lives in memory: kite_client's paper orders /
positions / GTTs, the trailing-stop engine's positions (current stop, best price, trail flags), the
agents' open-position bookkeeping (base_agent._tsl_sl_orders), the order guard's active orders and
trade counts, the risk manager's realised P&L and counters, each agent's trades_today / pnl_today and
day state, and the AI trader's targets / trails / spreads. On 12 Nov 2026 five restarts during market
hours wiped all of it and the open trades vanished from the ledger.

This module writes one JSON snapshot of that state (atomic tmp + replace) whenever it changes — checked
every PAPER_BOOK_SAVE_INTERVAL_S and immediately after a position closes — and restores it at startup
fully when the snapshot was written on today's IST date. A snapshot from an earlier session restores the
positions held overnight (CNC delivery / swing); its intraday MIS / NRML positions missed their square-off
and are closed at their last price and recorded in the ledger instead of being dropped. Restored positions keep being managed: the
owning agent's tick loop handles them once it runs, and until then (or when the agent is not running
or not watching the symbol) a small watchdog here drives the trailing engine and the agent's own exit
policy (stops, targets, trailing, the MIS square-off at SQUAREOFF_TIME). A restored trade that closes
goes through base_agent._close_tracked like any other, so it lands in the performance ledger once,
with its real entry price and costs. A trade the ledger already holds (closed after the last snapshot)
is reconciled instead of being restored a second time.

Never in LIVE mode (the broker holds the real book there) and never in demo mode.
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

SCHEMA = 1

_restore_attempted = False      # nothing is written before the startup restore ran (it would overwrite the snapshot)
_last_body: Optional[str] = None
_restored: set[str] = set()     # entry order ids restored for the strategy agents (watchdog candidates)
_save_task: Optional[asyncio.Task] = None
_watch_task: Optional[asyncio.Task] = None
_WATCH_NAME = "paper_book"
last_restore: dict = {}
# The trading session the in-memory day counters (risk, order guard, agents, AI) belong to. A session runs
# from DAY_START (09:15, master's daily reset) to the next day's 09:15, so a snapshot written after midnight
# still says its counters are the previous session's. Set at restore, advanced by every daily reset.
_book_day: Optional[date] = None
_squared_off_day: Optional[date] = None   # restored positions squared off by this module on that date
DAY_START = dtime(9, 15)


def path() -> Path:
    return Path(os.getenv("PAPER_BOOK_FILE")
                or Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
                .parent / "paper_book.json")


def enabled() -> bool:
    return bool(settings.persist_paper_book) and settings.trading_mode == "PAPER" and not settings.demo_active


# ── Snapshot ──────────────────────────────────────────────────────────────────

_TSL_FIELDS = ("symbol", "strategy", "side", "entry_price", "quantity", "order_id", "current_sl", "best_price",
               "trail_active", "breakeven_hit", "target1_hit", "target2_hit", "sl_moves", "max_profit",
               "locked_profit", "external_trail", "opened_at", "atr", "target", "target_hit")


def _agents() -> dict:
    from agents.strategy_agents import ALL_AGENTS
    return ALL_AGENTS


def snapshot() -> dict:
    """Everything a same-day restart needs to carry on trading the paper book."""
    from kite_client import kite_client
    from trailing_sl_engine import trailing_sl_engine, SLStatus
    from agents.base_agent import _tsl_sl_orders
    from risk_manager import risk_manager as rm
    from order_guard import order_guard as og

    with trailing_sl_engine._lock:
        tsl = [{f: getattr(p, f) for f in _TSL_FIELDS} for p in trailing_sl_engine._positions.values()
               if p.status == SLStatus.ACTIVE]
    with og._lock:
        guard = {"active": [[k[0], k[1], k[2], v.order_id, v.placed_at] for k, v in og._active.items()],
                 "trade_count": dict(og._trade_count), "cooldown_until": dict(og._cooldown_until)}
    agents = {}
    for name, a in _agents().items():
        try:
            day_state = a.export_day_state()
        except Exception as exc:
            logger.debug("[paper_book] {} day state not exported: {}", name, exc)
            day_state = {}
        agents[name] = {"trades_today": a.state.trades_today, "pnl_today": a.state.pnl_today,
                        "signals_fired": a.state.signals_fired, "skip_counts": dict(a.state.skip_counts),
                        "last_skip": a.state.last_skip, "day_state": day_state}
    try:
        from ai_trader import ai_trader
        ai = ai_trader.export_book()
    except Exception as exc:
        logger.debug("[paper_book] AI book not exported: {}", exc)
        ai = {}
    validated = {}
    for name, a in _agents().items():
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
        "tsl": tsl,
        "tracked": dict(_tsl_sl_orders),
        "risk": {k: getattr(rm, k) for k in ("daily_realised_pnl", "open_position_count", "trades_today",
                                             "wins_today", "losses_today", "gross_profit", "gross_loss",
                                             "costs_today", "is_trading_halted")},
        "guard": guard,
        "agents": agents,
        "ai": ai,
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
    """Entry order ids (and spread ids) the performance ledger recorded on or after `since` (the session the
    snapshot's book belongs to) → their ledger row."""
    from performance import performance
    return {t["oid"]: t for t in performance.trades if t.get("oid") and str(t.get("t", ""))[:10] >= since[:10]}


def _flatten_paper(kite_client, symbol: str, price: float, note: str, product: Optional[str] = None) -> None:
    """Close a restored paper position the ledger shows as already closed (synthetic exit at `price`)."""
    pos = next((p for p in kite_client._paper_positions if p["tradingsymbol"] == symbol
                and (product is None or (p.get("product") or "MIS") == product) and p.get("quantity")), None)
    if not pos or not pos.get("quantity"):
        return
    px = price or pos.get("last_price") or pos.get("average_price") or 0.0
    rec = {"order_id": f"PAPER-RECON-{len(kite_client._paper_orders)}", "tradingsymbol": symbol,
           "exchange": pos.get("exchange", "NSE"), "transaction_type": "SELL" if pos["quantity"] > 0 else "BUY",
           "quantity": abs(int(pos["quantity"])), "order_type": "MARKET", "product": pos.get("product", "MIS"),
           "price": px, "average_price": px, "trigger_price": 0.0, "status": "COMPLETE", "tag": note[:20],
           "placed_at": now_ist().isoformat()}
    kite_client._paper_orders.append(rec)
    kite_client._update_paper_position(rec)


def restore() -> dict:
    """Startup: re-load today's paper book. Never raises; returns a summary."""
    global _restore_attempted, last_restore, _book_day
    _book_day = session_day()                     # fresh (empty) counters belong to the current session
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
        out["day_reset"] = True                   # the restored counters were an earlier session's
    return out


# ── Day rollover (independent of master.start()) ──────────────────────────────

def session_day(now=None) -> date:
    """The trading session a moment belongs to: from 09:15 (master's daily reset) to the next 09:15."""
    now = now or now_ist()
    return now.date() if now.time() >= DAY_START else now.date() - timedelta(days=1)


def _parse_day(v) -> Optional[date]:
    try:
        return date.fromisoformat(str(v)) if v else None
    except ValueError:
        return None


def mark_day_reset(now=None) -> None:
    """master._daily_reset just reset every day counter: they belong to the current session now."""
    global _book_day
    _book_day = session_day(now)


def roll_day_if_due(now=None) -> bool:
    """Reset the day counters exactly as master._daily_reset does when the session they belong to is over.
    master's 09:15 job only exists once master.start() ran, and after a restart the bot is started at 09:16
    (or, as a catch-up, only until 15:00), so a restored or evening process would otherwise carry one day's
    halted flag, day P&L and trade caps into the next. PAPER (persistence on) only; returns True on a reset."""
    if not enabled() or not _restore_attempted:
        return False
    cur = session_day(now)
    if _book_day == cur:
        return False
    prev = _book_day
    mark_day_reset(now)
    from risk_manager import risk_manager as rm
    from order_guard import order_guard as og
    rm.reset_daily()
    og.reset_daily()
    for a in _agents().values():
        a.reset_daily()
    try:
        from agents.base_agent import reapply_overnight
        reapply_overnight()                       # positions held from earlier days still count (every agent)
    except Exception as exc:
        logger.warning("[paper_book] overnight positions not re-applied: {}", exc)
    logger.warning("[paper_book] new trading session {} — day counters of {} reset", cur, prev)
    save(force=True)
    return True


def squareoff_restored_if_due(now=None) -> list[str]:
    """Restored intraday positions (MIS / NRML F&O legs, not CNC / swing) are closed at SQUAREOFF_TIME even
    when master is not running to do it (after a restart past the 15:00 auto-start catch-up, for example).
    Same path as master's square-off; a position master already closed is no longer tracked, so nothing is
    exited twice. Restored positions only, and once per day."""
    global _squared_off_day
    if not _restored or not enabled():
        return []
    now = now or now_ist()
    h, m = (int(x) for x in settings.squareoff_time.split(":"))
    if now.weekday() >= 5 or now.time() < dtime(h, m) or _squared_off_day == now.date():
        return []
    from agents.base_agent import close_tracked_for_squareoff, _tsl_sl_orders
    from kite_client import kite_client
    only = {oid for oid in _restored if oid in _tsl_sl_orders}
    _restored.intersection_update(_tsl_sl_orders)
    if not only:
        return []
    _squared_off_day = now.date()
    syms = {_tsl_sl_orders[oid].get("symbol") for oid in only
            if _tsl_sl_orders[oid].get("product", "MIS") != "CNC" and _tsl_sl_orders[oid].get("kind") != "swing"}
    closed = close_tracked_for_squareoff("restored square-off", only=only)
    # a failed exit is flattened at the (paper) broker, as master's square-off does: restored symbols only
    from agents.base_agent import overnight_hold_qty
    hold = overnight_hold_qty()
    for p in kite_client.positions().get("net", []):
        sym = p.get("tradingsymbol")
        net = int(p.get("quantity") or 0) - sum(q for (s, _p), q in hold.items() if s == sym)
        if sym in syms and sym not in closed and p.get("quantity") and p.get("product") != "CNC" \
                and net and (net > 0) == (p["quantity"] > 0):
            try:
                kite_client.place_order(tradingsymbol=sym, exchange=p.get("exchange", "NSE"),
                                        transaction_type="SELL" if net > 0 else "BUY",
                                        quantity=abs(net), order_type="MARKET",
                                        product=p.get("product", "MIS"), tag="SquareOff")
            except Exception as exc:
                logger.critical("[paper_book] square-off of {} failed: {}", sym, exc)
    _restored.intersection_update(_tsl_sl_orders)
    if closed:
        logger.warning("[paper_book] squared off restored position(s) at {}: {}", settings.squareoff_time,
                       ", ".join(closed))
    return closed


def _quarantine() -> None:
    try:
        f = path()
        if f.exists():
            os.replace(f, f.with_suffix(".corrupt.json"))
    except Exception:
        pass


def _restore() -> dict:
    global _book_day
    if settings.trading_mode != "PAPER":
        return {"restored": False, "reason": "LIVE mode: the broker holds the book"}
    if not enabled():
        return {"restored": False, "reason": "disabled (PERSIST_PAPER_BOOK=false or demo mode)"}
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
    today = now_ist().date().isoformat()
    stale = d.get("session") != today
    if stale:
        try:                                   # keep it: the next save overwrites it
            f = path()
            f.with_name(f"{f.stem}.{d.get('session') or 'old'}.json").write_text(raw)
        except Exception as exc:
            logger.warning("[paper_book] could not keep the old snapshot: {}", exc)
        s_day = _parse_day(d.get("session"))
        if s_day is None or s_day > now_ist().date():
            logger.info("[paper_book] snapshot session {} is not usable today ({}); starting fresh",
                        d.get("session"), today)
            return {"restored": False, "reason": f"stale snapshot from {d.get('session')}"}
        # An earlier session's book (restart after midnight, a weekend, a crash before 15:15 that stayed down):
        # delivery / swing holdings are still held and are restored; intraday MIS / NRML positions that missed
        # their square-off are closed at their last price and recorded below — never silently dropped.
        logger.warning("[paper_book] snapshot is from an earlier session ({}): restoring the overnight holdings, "
                       "settling the intraday positions", d.get("session"))
    if d.get("mode", "PAPER") != "PAPER":
        return {"restored": False, "reason": "snapshot not from PAPER mode"}

    from kite_client import kite_client
    from trailing_sl_engine import trailing_sl_engine, PositionSL, SLStatus
    import agents.base_agent as ba
    from risk_manager import risk_manager as rm
    from order_guard import order_guard as og, ActiveOrder
    from tick_engine import tick_engine

    # Parse everything before touching live state: a malformed snapshot changes nothing
    k = d.get("kite") or {}
    orders = [dict(o) for o in k.get("orders") or []]
    positions = [dict(p) for p in k.get("positions") or []]
    gtts = {int(gid): dict(g) for gid, g in (k.get("gtts") or {}).items()}
    tracked = {str(oid): dict(e) for oid, e in (d.get("tracked") or {}).items()}
    tsl = {}
    for row in d.get("tsl") or []:
        pos = PositionSL(**{f: row[f] for f in _TSL_FIELDS if f in row})
        pos.status = SLStatus.ACTIVE
        tsl[pos.order_id] = pos
    risk = dict(d.get("risk") or {})
    guard = d.get("guard") or {}
    g_active = [(str(a[0]), str(a[1]), str(a[2]), str(a[3]), float(a[4])) for a in guard.get("active") or []]
    agents_state = d.get("agents") or {}
    if stale:
        # an earlier session's order book is history: only the orders still working (stops) carry over,
        # and only the positions still open
        orders = [o for o in orders if o.get("status") in _WORKING]
        positions = [p for p in positions if p.get("quantity")]

    closed = _closed_ids(str(d.get("book_day") or d.get("session") or today))
    tracked = {oid: e for oid, e in tracked.items() if oid in tsl}       # a position needs both halves

    # ── apply ──
    if kite_client._paper_orders or kite_client._paper_positions:
        logger.warning("[paper_book] paper book not empty at restore — merging")
    kite_client._paper_orders[:0] = orders
    kite_client._paper_positions[:0] = positions
    for gid, g in gtts.items():
        kite_client._paper_gtts.setdefault(gid, g)

    for name in ("daily_realised_pnl", "gross_profit", "gross_loss", "costs_today"):
        if name in risk:
            setattr(rm, name, float(risk[name]))
    for name in ("open_position_count", "trades_today", "wins_today", "losses_today"):
        if name in risk:
            setattr(rm, name, int(risk[name]))
    rm.is_trading_halted = bool(risk.get("is_trading_halted", False))
    _book_day = _parse_day(d.get("book_day")) or _parse_day(d.get("session"))   # the session these counters are from

    with og._lock:
        for sym, strat, side, oid, placed in g_active:
            og._active[(sym, strat, side)] = ActiveOrder(symbol=sym, strategy=strat, side=side,
                                                         order_id=oid, placed_at=placed)
        for strat, n in (guard.get("trade_count") or {}).items():
            og._trade_count[strat] = max(og._trade_count.get(strat, 0), int(n))
        for strat, until in (guard.get("cooldown_until") or {}).items():
            og._cooldown_until[strat] = max(og._cooldown_until.get(strat, 0.0), float(until))

    agents = _agents()
    for name, st in agents_state.items():
        a = agents.get(name)
        if not a:
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

    # Positions: skip any the ledger already closed (closed after the last snapshot) and reconcile the book
    restored, reconciled = [], []
    ba._setup_tsl_callbacks()
    for oid, e in tracked.items():
        if oid in ba._tsl_sl_orders:
            continue
        pos = tsl[oid]
        if oid in closed:
            row = closed[oid]
            _reconcile_closed(kite_client, og, rm, agents, e, pos, row)
            reconciled.append(e["symbol"])
            continue
        ba._tsl_sl_orders[oid] = e
        with trailing_sl_engine._lock:
            trailing_sl_engine._positions[oid] = pos
        exch = e.get("exchange", "NSE")
        tick_engine.subscribe([{"symbol": e["symbol"], "exchange": exch, "seed_price": pos.best_price or pos.entry_price}])
        a = agents.get(e.get("agent"))
        if a is not None:
            if exch == "NFO":
                a._exit_only.add(e["symbol"])      # option legs: their ticks only drive exits (as on entry)
            _restored.add(oid)
        restored.append(e["symbol"])

    # validated (lab-protocol) agents: their own positions, after the kite rows are restored
    for name, a in agents.items():
        if getattr(a, "scheduled", False) and hasattr(a, "import_book"):
            try:
                a.import_book((d.get("validated") or {}).get(name, {}), closed=set(closed), stale=stale)
            except Exception as exc:
                logger.error("[paper_book] {} book not restored: {}", name, exc)

    settled = _settle_missed_squareoff(kite_client, d) if stale else []
    for sym in settled:
        if sym in restored:
            restored.remove(sym)

    ai_n = 0
    if d.get("ai"):
        try:
            ai_d = dict(d["ai"])
            spreads = dict(ai_d.get("spreads") or {})
            for sid in [s for s in spreads if s in closed]:
                _reconcile_spread(kite_client, og, rm, spreads[sid], closed[sid])
                reconciled.append(sid)
            if stale:                            # intraday credit spreads missed their exit too
                for sid in [s for s in spreads if s not in closed]:
                    _settle_spread(kite_client, og, rm, sid, spreads.pop(sid), d.get("session"))
                    settled.append(sid)
                ai_d["spreads"] = spreads
            from ai_trader import ai_trader
            ai_n = ai_trader.import_book(ai_d, closed=set(closed), day=_book_day)
        except Exception as exc:
            logger.error("[paper_book] AI book not restored: {}", exc)

    if _restored:
        _ensure_watchdog()
    msg = (f"restored {len(restored)} paper position(s) ({', '.join(restored) or 'none'}), "
           f"{ai_n} AI, day P&L ₹{rm.daily_realised_pnl:,.0f}, {rm.trades_today} trades"
           + (f"; reconciled {len(reconciled)} already closed: {', '.join(reconciled)}" if reconciled else "")
           + (f"; closed {len(settled)} intraday position(s) that missed the {d.get('session')} square-off: "
              f"{', '.join(settled)}" if settled else ""))
    logger.warning("[paper_book] {} (snapshot {})", msg, d.get("saved_at"))
    return {"restored": True, "positions": restored, "ai_positions": ai_n, "reconciled": reconciled,
            "settled": settled, "stale": stale, "saved_at": d.get("saved_at"), "message": msg}


_WORKING = ("TRIGGER PENDING", "OPEN")


def _overnight(e: dict) -> bool:
    """Held past the close: delivery (CNC) or a swing position. Everything else is squared off every day."""
    return e.get("product", "MIS") == "CNC" or e.get("kind") == "swing"


def _paper_row(kite_client, symbol: str, product: Optional[str] = None) -> Optional[dict]:
    """The paper-broker row of `symbol` (one row per (symbol, product)): of that product when given, else the
    first open one. A CNC swing holding and an intraday MIS position in the same symbol are separate rows."""
    rows = [p for p in kite_client._paper_positions if p["tradingsymbol"] == symbol
            and (product is None or (p.get("product") or "MIS") == product)]
    return next((p for p in rows if p.get("quantity")), rows[0] if rows else None)


def _last_price(kite_client, symbol: str, fallback: float, product: Optional[str] = None) -> float:
    pos = _paper_row(kite_client, symbol, product)
    px = float((pos or {}).get("last_price") or 0.0)
    return px if px > 0 else float(fallback or 0.0)


def _paper_exit(kite_client, symbol: str, side: str, qty: int, px: float, note: str,
                product: Optional[str] = None) -> int:
    """Close `qty` of a paper position at `px` (synthetic fill: the session it belongs to is over).
    Never more than is open; returns the quantity closed."""
    pos = _paper_row(kite_client, symbol, product)
    sign = 1 if side == "BUY" else -1
    q = min(int(qty), max(0, int((pos or {}).get("quantity") or 0) * sign))
    if q <= 0:
        return 0
    rec = {"order_id": f"PAPER-SETTLE-{len(kite_client._paper_orders)}", "tradingsymbol": symbol,
           "exchange": pos.get("exchange", "NSE"), "transaction_type": "SELL" if side == "BUY" else "BUY",
           "quantity": q, "order_type": "MARKET", "product": pos.get("product", "MIS"),
           "price": px, "average_price": px, "trigger_price": 0.0, "status": "COMPLETE", "tag": note[:20],
           "placed_at": now_ist().isoformat()}
    kite_client._paper_orders.append(rec)
    kite_client._update_paper_position(rec)
    return q


def _settle_missed_squareoff(kite_client, d: dict) -> list[str]:
    """Restored from an earlier session: its intraday (MIS / NRML) positions should have been squared off at
    SQUAREOFF_TIME but the process was down. Close each at its last snapshot price, exactly once, through
    _close_tracked (ledger with its entry order id, order guard, risk counters, agent P&L)."""
    import agents.base_agent as ba
    from trailing_sl_engine import trailing_sl_engine
    agents = _agents()
    out = []
    for oid, e in list(ba._tsl_sl_orders.items()):
        if _overnight(e):
            continue
        pos = trailing_sl_engine._positions.get(oid)
        if pos is None:
            continue
        product = e.get("product", "MIS")        # never the CNC row of the same symbol
        px = _last_price(kite_client, pos.symbol, pos.entry_price, product)
        q = _paper_exit(kite_client, pos.symbol, pos.side, pos.quantity, px, "SquareOff-missed", product)
        if q <= 0:                               # already flat at the broker (its stop filled): close at the stop
            px = pos.current_sl if pos.current_sl and pos.current_sl > 0 else px
        sign = 1 if pos.side == "BUY" else -1
        qty = q or pos.quantity
        pnl = round((px - pos.entry_price) * qty * sign, 2)
        ba._close_tracked(oid, pos.symbol, pos.side, qty, pnl,
                          f"missed square-off {d.get('session')}", pos.strategy, send_exit=False)
        a = agents.get(e.get("agent"))
        if a is not None:
            a.state.pnl_today += pnl
            a._exit_only.discard(pos.symbol)
        _restored.discard(oid)
        out.append(pos.symbol)
        logger.warning("[paper_book] {} {} missed the {} square-off — closed at ₹{:.2f}, P&L ₹{:.0f}",
                       pos.strategy, pos.symbol, d.get("session"), px, pnl)
    return out


def _settle_spread(kite_client, og, rm, sid: str, sp: dict, session) -> None:
    """An earlier session's AI credit spread (intraday) that missed its exit: close both legs at their last price."""
    from performance import performance
    short_px = _last_price(kite_client, sp["short"]["symbol"], 0.0)
    long_px = _last_price(kite_client, sp["long"]["symbol"], 0.0)
    _paper_exit(kite_client, sp["short"]["symbol"], "SELL", sp["qty"], short_px, "SquareOff-missed")
    _paper_exit(kite_client, sp["long"]["symbol"], "BUY", sp["qty"], long_px, "SquareOff-missed")
    value = max(short_px - long_px, 0.0)
    pnl = round((sp["credit"] - value) * sp["qty"], 2)
    og.release_order(sp["underlying"], "ai_spread", "SELL", pnl)
    rm.record_trade(pnl)
    rm.position_closed()
    performance.record_trade("ai_spread", sid, "SELL", sp["qty"] * 2, sp["credit"], pnl, "NFO", "MIS",
                             exit_price=value, note=f"missed square-off {session}", order_id=sid)


def _reconcile_closed(kite_client, og, rm, agents, e: dict, pos, row: dict) -> None:
    """The ledger already has this trade: settle the restored counters and paper position to match."""
    for o in kite_client._paper_orders:
        if o["order_id"] == e.get("sl_order_id") and o.get("status") == "TRIGGER PENDING":
            o["status"] = "CANCELLED"
    _flatten_paper(kite_client, e["symbol"], float(row.get("exit") or 0.0), "RESTORE-RECON",
                   e.get("product", "MIS"))
    gross = float(row.get("gross") or 0.0)
    og.release_order(e.get("guard_symbol", e["symbol"]), pos.strategy, e.get("side", pos.side), gross)
    rm.record_trade(gross)
    rm.record_cost(float(row.get("cost") or 0.0))
    rm.position_closed()
    a = agents.get(e.get("agent"))
    if a is not None:
        a.state.pnl_today += gross


def _reconcile_spread(kite_client, og, rm, sp: dict, row: dict) -> None:
    for leg in ("short", "long"):
        _flatten_paper(kite_client, sp[leg]["symbol"], 0.0, "RESTORE-RECON")
    gross = float(row.get("gross") or 0.0)
    og.release_order(sp["underlying"], "ai_spread", "SELL", gross)
    rm.record_trade(gross)
    rm.record_cost(float(row.get("cost") or 0.0))
    rm.position_closed()


# ── Watchdog: restored positions whose agent is not (yet) handling the symbol ──────

def _handled_by_agent(a, snap) -> bool:
    """True when the agent's own tick loop runs the exits for this tick (BaseAgent._run_loop: exit-only legs
    always; watchlist symbols once they have the agent's minimum candles)."""
    if not a.state.running:
        return False
    if snap.symbol in a._exit_only:
        return True
    return snap.symbol in a._approved and len(snap.candles_1min) >= a.min_candles_1min


async def on_snap(snap) -> None:
    """One tick for the restored positions nobody else is managing: the trailing engine (stop, breakeven,
    trail, target 2) and the owning agent's exit policy (rule_exit — the MIS square-off for the validated
    agents), exactly what the agent's own tick loop would run."""
    from agents.base_agent import _tsl_sl_orders
    from trailing_sl_engine import trailing_sl_engine
    agents = _agents()
    for oid in list(_restored):
        e = _tsl_sl_orders.get(oid)
        if e is None:
            _restored.discard(oid)               # closed
            continue
        if e.get("symbol") != snap.symbol:
            continue
        a = agents.get(e.get("agent"))
        if a is None or _handled_by_agent(a, snap):
            continue
        await trailing_sl_engine.on_tick(snap.symbol, snap.tick.ltp, snap.indicators.atr_14)
        if oid in _tsl_sl_orders:
            await a._check_exits_on_tick(snap)
        if oid not in _tsl_sl_orders:
            _restored.discard(oid)


async def _watch_loop() -> None:
    from tick_engine import tick_engine
    q = tick_engine.add_subscriber(_WATCH_NAME)
    try:
        while _restored:
            try:
                snap = await asyncio.wait_for(q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                from agents.base_agent import _tsl_sl_orders
                _restored.intersection_update(_tsl_sl_orders)
                continue
            try:
                await on_snap(snap)
            except Exception as exc:
                logger.error("[paper_book] watchdog error on {}: {}", getattr(snap, "symbol", "?"), exc)
    finally:
        tick_engine.remove_subscriber(_WATCH_NAME)


def _ensure_watchdog() -> None:
    global _watch_task
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return                                   # started later by start()
    if not _watch_task or _watch_task.done():
        _watch_task = asyncio.create_task(_watch_loop())


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def _save_loop() -> None:
    while True:
        await asyncio.sleep(max(0.5, float(settings.paper_book_save_interval_s)))
        for step in (roll_day_if_due, squareoff_restored_if_due):
            try:
                step()
            except Exception as exc:
                logger.error("[paper_book] {} failed: {}", step.__name__, exc)
        save()


def start() -> None:
    """After restore(): periodic snapshots and the watchdog for restored positions."""
    global _save_task
    if not enabled():
        return
    if not _save_task or _save_task.done():
        _save_task = asyncio.create_task(_save_loop())
    if _restored:
        _ensure_watchdog()


def status() -> dict:
    f = path()
    return {"enabled": enabled(), "file": str(f), "exists": f.exists(), "book_day": str(_book_day or ""),
            "watching_restored": len(_restored), "last_restore": last_restore}
