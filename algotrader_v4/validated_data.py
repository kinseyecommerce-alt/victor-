"""
validated_data.py — the market data the lab-validated agents read, built exactly as the lab built it.

The lab (stratlab/data.py) keeps the 1-minute bars from 09:15 to 15:29, drops sessions with fewer than
MIN_SESSION_BARS (200) bars, and defines "previous session" as the symbol's previous FULL (valid) session.
`summarize_sessions` is the single place that turns Kite minute candles into per-session summaries; the live
store and the lab-parity test both go through it.

Store: STORE_FILE (next to the performance ledger), refreshed from Kite historical minute candles (a full refetch,
so split adjustments in Kite's history are absorbed), holding the last 25 valid sessions of every symbol.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

import forward_test
from stratlab import data as _SD

UNIVERSE: tuple[str, ...] = tuple(forward_test.UNIVERSE)       # the lab's 97 symbols (single source)
KEEP_SESSIONS = 25
LOOKBACK = 20


def _store_file() -> Path:
    return Path(os.getenv("VALIDATED_SESSIONS_FILE") or
                Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
                .parent / "validated_sessions.json")


@dataclass(frozen=True)
class SessionSummary:
    day: int            # yyyymmdd
    bars: int
    volume: float
    last_close: float
    first_t: int        # minute of day of the first bar


# ── pure ──────────────────────────────────────────────────────────────────────
def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def summarize_sessions(rows: list[dict]) -> list[SessionSummary]:
    """Kite minute candles {"date": datetime, open, high, low, close, volume} → one summary per date, with the
    lab's cleaning (stratlab.data._clean_minute): bars 09:15..15:29 only, finite positive OHLC, duplicate minutes
    keep the last. Sessions of every length are returned; callers keep `valid_sessions` as the lab does."""
    by_day: dict[int, dict[int, tuple]] = {}
    for r in rows:
        ts = r.get("date")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts is None:
            continue
        t = ts.hour * 60 + ts.minute
        if t < _SD.SESSION_OPEN or t > _SD.SESSION_LAST:
            continue
        o, h, l, c = (_num(r.get(k)) for k in ("open", "high", "low", "close"))
        if not all(math.isfinite(x) and x > 0 for x in (o, h, l, c)):
            continue
        v = _num(r.get("volume"))
        d = ts.year * 10000 + ts.month * 100 + ts.day
        by_day.setdefault(d, {})[t] = (c, v if math.isfinite(v) else 0.0)        # later duplicate wins
    out = []
    for d in sorted(by_day):
        bars = by_day[d]
        ts = sorted(bars)
        out.append(SessionSummary(day=d, bars=len(ts), volume=float(sum(bars[t][1] for t in ts)),
                                  last_close=float(bars[ts[-1]][0]), first_t=int(ts[0])))
    return out


def valid_sessions(summ) -> list[SessionSummary]:
    return [s for s in summ if s.bars >= _SD.MIN_SESSION_BARS]


def _ymd(d) -> int:
    if isinstance(d, int):
        return d
    return d.year * 10000 + d.month * 100 + d.day


def prev_close_from(summ, day) -> Optional[float]:
    """last_close of the last valid session strictly before `day` (the lab's prev_close)."""
    k = _ymd(day)
    prior = [s for s in valid_sessions(summ) if s.day < k]
    return prior[-1].last_close if prior else None


def prev20_from(summ, day) -> Optional[tuple[float, ...]]:
    """Full-day volumes of the last 20 valid sessions strictly before `day`; None with fewer than 20."""
    k = _ymd(day)
    prior = [s for s in valid_sessions(summ) if s.day < k]
    if len(prior) < LOOKBACK:
        return None
    return tuple(s.volume for s in prior[-LOOKBACK:])


def _as_dt(ts) -> datetime:
    return datetime.fromisoformat(ts) if isinstance(ts, str) else ts


# today's raw minute candles per symbol from the last after-close refresh (the 15:40 parity audits read them)
last_day_rows: dict[str, list[dict]] = {}

# ── store ─────────────────────────────────────────────────────────────────────
_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_store_file().read_text())
        except FileNotFoundError:
            _cache = {"symbols": {}}
        except Exception as exc:
            logger.error("[validated_data] store unreadable ({}) — empty", exc)
            _cache = {"symbols": {}}
    return _cache


def _save(d: dict) -> None:
    global _cache
    f = _store_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=f.parent, prefix=".validated_sessions.")
    with os.fdopen(fd, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    os.replace(tmp, f)
    _cache = d


def set_store(symbols: dict[str, list[SessionSummary]], persist: bool = False, missing=()) -> None:
    """Replace the store (tests, refresh_store). Keeps the last KEEP_SESSIONS valid sessions per symbol."""
    global _cache
    d = {"updated": datetime.now().isoformat(timespec="seconds"), "missing": list(missing),
         "symbols": {s: [list(asdict(x).values()) for x in valid_sessions(v)[-KEEP_SESSIONS:]]
                     for s, v in symbols.items()}}
    if persist:
        _save(d)
    else:
        _cache = d


def stored_sessions(symbol: str) -> list[SessionSummary]:
    return [SessionSummary(int(r[0]), int(r[1]), float(r[2]), float(r[3]), int(r[4]))
            for r in _load().get("symbols", {}).get(symbol, [])]


def prev_close(symbol: str, day) -> Optional[float]:
    return prev_close_from(stored_sessions(symbol), day)


def prev20_volumes(symbol: str, day) -> Optional[tuple[float, ...]]:
    return prev20_from(stored_sessions(symbol), day)


def newest_session() -> Optional[int]:
    days = [r[-1][0] for r in _load().get("symbols", {}).values() if r]
    return max(days) if days else None


def previous_trading_day(d: date) -> date:
    from ist_clock import is_trading_day
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def store_current(today: date) -> bool:
    """The store reaches the previous trading day (symbols missing from it are skipped one by one)."""
    n = newest_session()
    return n is not None and n >= _ymd(previous_trading_day(today))


def store_fresh(today: date) -> bool:
    """The newest stored session is the previous trading day (today's decisions read sessions < today)."""
    n = newest_session()
    return n is not None and n >= _ymd(previous_trading_day(today)) and not _load().get("missing")


def refresh_store(days: int = 40) -> dict:
    """Kite historical minute candles for UNIVERSE (full refetch), last KEEP_SESSIONS valid sessions each.
    Today's session is kept only after the close (a partial session must never become tomorrow's 'previous')."""
    from kite_client import kite_client
    from ist_clock import now_ist
    now = now_ist()
    today_k = _ymd(now.date())
    complete_today = now.time() >= dtime(15, 35)
    try:
        tokens = kite_client.get_instrument_tokens(list(UNIVERSE), "NSE")
    except Exception as exc:
        logger.error("[validated_data] instrument tokens unavailable: {}", exc)
        return {"ok": False, "error": str(exc)[:200]}
    frm = datetime.combine(now.date() - timedelta(days=days), dtime(9, 0))
    to = datetime.combine(now.date(), dtime(15, 30))
    # a symbol whose fetch failed is left out (never kept on older sessions: its "previous session" would be wrong)
    out, missing = {}, []
    fetched = 0
    for sym in UNIVERSE:
        tok = tokens.get(sym)
        if not tok:
            missing.append(sym)
            continue
        try:
            rows = kite_client.historical_data(tok, frm, to, "minute")
        except Exception as exc:
            logger.warning("[validated_data] {} history failed: {}", sym, exc)
            missing.append(sym)
            continue
        if complete_today:
            last_day_rows[sym] = [r for r in rows if _ymd(_as_dt(r.get("date")).date()) == today_k]
        summ = [s for s in summarize_sessions(rows) if complete_today or s.day < today_k]
        if not summ:
            missing.append(sym)
            continue
        out[sym] = summ
        fetched += 1
    set_store(out, persist=True, missing=missing)
    res = {"ok": fetched > 0, "fetched": fetched, "missing": missing, "newest": newest_session()}
    logger.info("[validated_data] store refreshed: {}", {k: v for k, v in res.items() if k != "missing"}
                | {"missing": len(missing)})
    return res


# ── live quotes / calendars ───────────────────────────────────────────────────
def quotes(instruments: list[str]) -> dict:
    """Kite quotes compacted like forward_test._compact (ltp, open, prev_close, bid, ask, volume ...) keyed by the
    bare symbol. Never raises: {} on failure (logged)."""
    from kite_client import kite_client
    try:
        raw = kite_client.quote_kite(list(instruments)) or {}
        return {k.split(":", 1)[1] if ":" in k else k: forward_test._compact(q) for k, q in raw.items()}
    except Exception as exc:
        logger.error("[validated_data] quote of {} instruments failed: {}", len(instruments), exc)
        return {}


def price_adjusting_actions(symbol: str, d: date) -> list[str]:
    try:
        import event_calendar
        return event_calendar.price_adjusting_actions(symbol, d)
    except Exception as exc:
        logger.warning("[validated_data] corporate actions unavailable: {}", exc)
        return []


def ca_calendar_loaded() -> bool:
    try:
        import event_calendar
        return event_calendar._loaded_at is not None
    except Exception:
        return False


def special_sessions() -> set[date]:
    from config import settings
    out = set()
    for part in (getattr(settings, "validated_special_sessions", "") or "").split(","):
        part = part.strip()
        if part:
            try:
                out.add(date.fromisoformat(part))
            except ValueError:
                logger.warning("[validated_data] bad special session {!r} ignored", part)
    return out
