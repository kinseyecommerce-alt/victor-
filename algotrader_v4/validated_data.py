"""
validated_data.py — market data for the validated (lab-protocol) agents, built the way the lab built it.

The lab (stratlab/) reads 1-minute bars from 09:15 to 15:29, drops sessions with fewer than 200 bars
(stratlab.data.MIN_SESSION_BARS), and defines "previous close" as the close of the last 1-minute bar of the
symbol's previous valid session. The live agents need the same numbers, so this module keeps a small store of
per-session summaries built from Kite historical minute candles by `summarize_sessions` — the SAME function the
lab-parity tests feed with the lab's own bars.

  UNIVERSE         the lab's 97 symbols (forward_test.UNIVERSE, single source)
  refresh_store()  Kite minute candles for UNIVERSE (full refetch every time: absorbs split adjustments), keeps
                   the last 25 valid sessions per symbol in STORE_FILE (atomic write)
  prev_close()     last_close of the last valid session before a day
  quotes()         one Kite quote batch, compacted like forward_test (never raises)
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

import forward_test
from ist_clock import is_trading_day, now_ist
from stratlab.data import MIN_SESSION_BARS, SESSION_LAST, SESSION_OPEN

UNIVERSE: tuple[str, ...] = tuple(forward_test.UNIVERSE)
STORE_FILE = Path(os.getenv("VALIDATED_SESSIONS_FILE") or Path(os.getenv("PERFORMANCE_FILE") or
                  Path(__file__).resolve().parent / "data" / "performance.json").parent / "validated_sessions.json")
KEEP_VALID = 25


@dataclass(frozen=True)
class SessionSummary:
    day: int                  # yyyymmdd
    bars: int                 # 1-minute bars 09:15..15:29
    volume: float
    last_close: float         # close of the last bar of the session
    first_t: int              # minute of day of the first bar (555 = 09:15)
    first_open: float = 0.0   # open of the first bar (the auction print when first_t == 555)


def _ymd(d) -> int:
    if isinstance(d, int):
        return d
    if isinstance(d, str):
        return int(d[:10].replace("-", ""))
    return d.year * 10000 + d.month * 100 + d.day


def _bar_key(ts) -> tuple[int, int]:
    """(yyyymmdd, minute of day) of a Kite candle timestamp (IST wall clock; tz-aware or naive)."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.year * 10000 + ts.month * 100 + ts.day, ts.hour * 60 + ts.minute


def summarize_sessions(rows: Iterable[dict]) -> list[SessionSummary]:
    """Kite minute candles {"date", "open", "high", "low", "close", "volume"} → one summary per date, oldest
    first. Same cleaning as stratlab.data._clean_minute: 09:15..15:29 only, positive prices, a duplicate minute
    keeps its last bar. Short sessions are NOT dropped here (callers use valid_sessions, as the lab does)."""
    by_min: dict[tuple[int, int], dict] = {}
    for r in rows or []:
        try:
            d, t = _bar_key(r["date"])
            o, h, l, c = (float(r[k]) for k in ("open", "high", "low", "close"))
        except Exception:
            continue
        if not (SESSION_OPEN <= t <= SESSION_LAST) or min(o, h, l, c) <= 0 or any(x != x for x in (o, h, l, c)):
            continue
        by_min[(d, t)] = {"o": o, "c": c, "v": float(r.get("volume") or 0.0)}
    out: list[SessionSummary] = []
    cur, acc = None, None
    for (d, t) in sorted(by_min):
        b = by_min[(d, t)]
        if d != cur:
            if acc:
                out.append(SessionSummary(**acc))
            cur, acc = d, {"day": d, "bars": 0, "volume": 0.0, "last_close": 0.0, "first_t": t, "first_open": b["o"]}
        acc["bars"] += 1
        acc["volume"] += b["v"]
        acc["last_close"] = b["c"]
    if acc:
        out.append(SessionSummary(**acc))
    return out


def valid_sessions(summ: Iterable[SessionSummary]) -> list[SessionSummary]:
    return [s for s in summ if s.bars >= MIN_SESSION_BARS]


# ── store ────────────────────────────────────────────────────────────────────
_store: Optional[dict] = None


def _load() -> dict:
    global _store
    if _store is None:
        try:
            raw = json.loads(STORE_FILE.read_text())
            _store = {"symbols": {s: [SessionSummary(**x) for x in v] for s, v in raw.get("symbols", {}).items()},
                      "updated": raw.get("updated")}
        except FileNotFoundError:
            _store = {"symbols": {}, "updated": None}
        except Exception as exc:
            logger.error("[validated_data] store unreadable ({}) — starting empty", exc)
            _store = {"symbols": {}, "updated": None}
    return _store


def _save(st: dict) -> None:
    try:
        STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=STORE_FILE.parent, prefix=".validated_sessions.")
        with os.fdopen(fd, "w") as fh:
            json.dump({"symbols": {s: [asdict(x) for x in v] for s, v in st["symbols"].items()},
                       "updated": st.get("updated")}, fh)
        os.replace(tmp, STORE_FILE)
    except Exception as exc:
        logger.error("[validated_data] store not written: {}", exc)


def set_sessions(symbol: str, summ: list[SessionSummary], save: bool = True) -> None:
    """Replace one symbol's sessions (keeps the last KEEP_VALID valid sessions and the short ones among them)."""
    st = _load()
    summ = sorted(summ, key=lambda s: s.day)
    valid = valid_sessions(summ)
    lo = valid[-KEEP_VALID].day if len(valid) >= KEEP_VALID else (summ[0].day if summ else 0)
    st["symbols"][symbol] = [s for s in summ if s.day >= lo]
    st["updated"] = now_ist().isoformat(timespec="seconds")
    if save:
        _save(st)


def sessions(symbol: str) -> list[SessionSummary]:
    return list(_load()["symbols"].get(symbol, []))


def session_on(symbol: str, day) -> Optional[SessionSummary]:
    d = _ymd(day)
    return next((s for s in sessions(symbol) if s.day == d), None)


def prev_close(symbol: str, day) -> Optional[float]:
    """last_close of the last valid session strictly before `day` (the lab's prev_close), None if unknown."""
    d = _ymd(day)
    prev = [s for s in valid_sessions(sessions(symbol)) if s.day < d]
    return prev[-1].last_close if prev else None


def prev20_volumes(symbol: str, day) -> Optional[tuple[float, ...]]:
    d = _ymd(day)
    prev = [s for s in valid_sessions(sessions(symbol)) if s.day < d]
    return tuple(s.volume for s in prev[-20:]) if len(prev) >= 20 else None


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def newest_session() -> Optional[int]:
    days = [s.day for v in _load()["symbols"].values() for s in v]
    return max(days) if days else None


def store_fresh(today: date) -> bool:
    """The newest stored session is the previous trading day (or today, after the 15:40 refresh)."""
    n = newest_session()
    return n is not None and n >= _ymd(previous_trading_day(today))


def refresh_store(days: int = 40) -> dict:
    """Kite historical minute candles for UNIVERSE over the last `days` calendar days (~97 calls at 3/s)."""
    from kite_client import kite_client
    out = {"symbols": 0, "missing": [], "newest": None}
    try:
        tokens = kite_client.get_instrument_tokens(list(UNIVERSE), "NSE")
    except Exception as exc:
        logger.error("[validated_data] instrument tokens unavailable: {}", exc)
        return {**out, "error": str(exc)[:200]}
    now = now_ist().replace(tzinfo=None)
    frm = (now - timedelta(days=days)).replace(hour=9, minute=0, second=0, microsecond=0)
    for sym in UNIVERSE:
        tok = tokens.get(sym)
        if not tok:
            out["missing"].append(sym)
            continue
        try:
            rows = kite_client.historical_data(tok, frm, now, "minute")
        except Exception as exc:
            logger.warning("[validated_data] {} history failed: {}", sym, exc)
            rows = []
        summ = summarize_sessions(rows)
        if not summ:
            out["missing"].append(sym)
            continue
        set_sessions(sym, summ, save=False)
        out["symbols"] += 1
    _save(_load())
    out["newest"] = newest_session()
    logger.info("[validated_data] store refreshed: {} symbols, newest session {}, missing {}",
                out["symbols"], out["newest"], out["missing"][:10])
    return out


# ── quotes / calendars ───────────────────────────────────────────────────────
def quotes(instruments: Iterable[str]) -> dict:
    """One Kite quote batch → {tradingsymbol: forward_test._compact(quote)}. Bare names default to NSE.
    Never raises: {} (logged) on failure."""
    ins = [i if ":" in i else f"NSE:{i}" for i in instruments]
    if not ins:
        return {}
    try:
        from kite_client import kite_client
        raw = kite_client.quote_kite(ins) or {}
        return {k.split(":", 1)[1]: forward_test._compact(q) for k, q in raw.items()}
    except Exception as exc:
        logger.warning("[validated_data] quote batch of {} failed: {}", len(ins), exc)
        return {}


def price_adjusting_actions(symbol: str, d: date) -> list[str]:
    try:
        import event_calendar
        return event_calendar.price_adjusting_actions(symbol, d)
    except Exception:
        return []


def ca_calendar_loaded() -> bool:
    try:
        import event_calendar
        return event_calendar.ca_calendar_loaded()
    except Exception:
        return False


def special_sessions() -> set[date]:
    from config import settings
    out = set()
    for part in str(getattr(settings, "validated_special_sessions", "") or "").split(","):
        try:
            out.add(date.fromisoformat(part.strip()))
        except ValueError:
            continue
    return out
