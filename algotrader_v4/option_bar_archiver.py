"""
option_bar_archiver.py — keeps 1-minute history of index option contracts before they disappear.

WHY: Kite serves history only for contracts that are still listed.  Once a NIFTY / BANKNIFTY option expires its
minute bars are gone, so intraday option research has nothing older than the contracts listed today.  This job
saves the bars of the contracts about to expire, every trading day after the close, so history accumulates.

WHAT IT DOES (and nothing else): after the close on a trading day it selects
  * NIFTY and BANKNIFTY option contracts whose expiry is today or within the next 2 trading sessions and whose
    strike is within +-8 % of the index's close, and
  * the NIFTY and BANKNIFTY futures expiring in the same window,
fetches each contract's whole 1-minute history (with open interest) through kite_client.historical_data and writes
one gzip CSV per contract:  <root>/<expiry yyyymmdd>/<tradingsymbol>.csv.gz   (ts,open,high,low,close,volume,oi).
A contract is selected on up to 3 consecutive sessions; each run rewrites its file with the full history fetched
that day (idempotent; the last run, on the expiry day, is the complete life).  One line per run goes to
<root>/manifest.jsonl.

SAFETY
  * OFF by default: OPTION_BAR_ARCHIVER_ENABLED must be "true".  With the flag off the scheduler does not register
    the job, and daily_job() returns {"skipped": ...} without touching Kite or the disk.
  * Read-only towards trading: the only Kite calls are get_instruments, get_instrument_tokens and historical_data
    (history requests go through kite_client's own rate limiter).  It never places, modifies or cancels an order,
    never reads or changes an agent, a position, risk or settings, and sends no Telegram message.
  * Bounded disk: before writing, the archive size is measured; at or above OPTION_BAR_ARCHIVE_MAX_MB (default
    1024) the run writes nothing and reports "disk cap reached" (history is never deleted automatically).
    Estimate from the 2026-11 contracts: ~6 MB in a normal week, ~25 MB in a monthly-expiry week, ~37 MB a
    month (see ESTIMATE_MB_PER_WEEK).

ROOT: OPTION_BAR_ARCHIVE_DIR, else "<dir of PERFORMANCE_FILE>/option_bars" (the Railway volume), else ./data/option_bars.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

UNDERLYINGS = ("NIFTY", "BANKNIFTY")
INDEX_QUOTE = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}      # NSE index names in Kite's instrument master
STRIKE_PCT = 8.0                 # strikes within +-8 % of the index close
SESSIONS_AHEAD = 2               # expiry today or within the next 2 trading sessions
MAX_LIFE_DAYS = 120              # history window requested per contract (index monthlies list ~3 months ahead)
DEFAULT_MAX_MB = 1024
# Disk estimate, measured on the contracts listed on 2026-11-14 (whole life, gzip CSV without OI; +25 % allowed for
# the OI column): one NIFTY weekly expiry at +-8 % = 160 contracts = 4.2 MB (-> ~5.5 MB), the NIFTY monthly expiry
# 8.9 MB (-> ~11 MB), the BANKNIFTY monthly expiry (196 contracts) 6.5 MB (-> ~8 MB), two futures 0.4 MB.
# A normal week holds one NIFTY weekly expiry: ~6 MB; the monthly-expiry week ~20-25 MB; about 37 MB a month,
# 0.45 GB a year, so the default 1024 MB cap lasts a little over two years.
ESTIMATE_MB_PER_WEEK = 6.0       # normal week; ~25 MB in a monthly-expiry week; ~37 MB / month
_LOCK = threading.Lock()


def enabled() -> bool:
    """OPTION_BAR_ARCHIVER_ENABLED (default false)."""
    return os.getenv("OPTION_BAR_ARCHIVER_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def default_root() -> Path:
    p = os.getenv("OPTION_BAR_ARCHIVE_DIR")
    if p:
        return Path(p)
    perf = os.getenv("PERFORMANCE_FILE")
    base = Path(perf).parent if perf else Path(__file__).resolve().parent / "data"
    return base / "option_bars"


def max_bytes() -> int:
    try:
        return int(float(os.getenv("OPTION_BAR_ARCHIVE_MAX_MB", DEFAULT_MAX_MB)) * 1024 * 1024)
    except ValueError:
        return DEFAULT_MAX_MB * 1024 * 1024


def archive_bytes(root: Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def _as_date(x) -> Optional[date]:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str) and x:
        try:
            return date.fromisoformat(x[:10])
        except ValueError:
            return None
    return None


def expiry_window(day: date, is_trading_day: Callable[[date], bool], ahead: int = SESSIONS_AHEAD) -> date:
    """The last date of the window: the `ahead`-th trading session after `day`."""
    d, n = day, 0
    while n < ahead:
        d += timedelta(days=1)
        if is_trading_day(d):
            n += 1
        if (d - day).days > 30:          # broken calendar: never loop for ever
            break
    return d


def select_contracts(instruments: list[dict], day: date, last: date, spot: dict, pct: float = STRIKE_PCT) -> list[dict]:
    """Rows of the NFO instrument master to archive on `day`: NIFTY / BANKNIFTY futures and options with
    day <= expiry <= last; options only with |strike / spot - 1| <= pct % (an underlying without a spot is skipped)."""
    out = []
    for r in instruments:
        und = r.get("name")
        if und not in UNDERLYINGS:
            continue
        exp = _as_date(r.get("expiry"))
        if exp is None or not (day <= exp <= last):
            continue
        typ = r.get("instrument_type")
        if typ == "FUT":
            out.append({**r, "expiry": exp})
        elif typ in ("CE", "PE"):
            s = spot.get(und)
            k = float(r.get("strike") or 0)
            if s and k > 0 and abs(k / s - 1.0) <= pct / 100.0 + 1e-9:
                out.append({**r, "expiry": exp})
    out.sort(key=lambda r: (r["expiry"], r["name"], r["instrument_type"] != "FUT", float(r.get("strike") or 0),
                            r["instrument_type"]))
    return out


def _spot(kite, day: date) -> dict:
    """Index closes from Kite's daily candles (read-only history call)."""
    out = {}
    try:
        toks = kite.get_instrument_tokens(list(INDEX_QUOTE.values()), "NSE")
    except Exception as exc:
        logger.warning("[option-archive] index tokens failed: {}", exc)
        return out
    for und, name in INDEX_QUOTE.items():
        tok = toks.get(name)
        if not tok:
            continue
        try:
            rows = kite.historical_data(tok, datetime.combine(day - timedelta(days=7), datetime.min.time()),
                                        datetime.combine(day, datetime.max.time().replace(microsecond=0)), "day")
        except Exception as exc:
            logger.warning("[option-archive] {} close failed: {}", und, exc)
            continue
        if rows:
            out[und] = float(rows[-1]["close"])
    return out


def _ts(d) -> str:
    return (d.replace(tzinfo=None) if getattr(d, "tzinfo", None) else d).isoformat(timespec="minutes")


def write_contract(root: Path, row: dict, bars: list[dict]) -> int:
    """Atomically (re)write one contract's file; returns the bytes written."""
    d = Path(root) / row["expiry"].strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ts", "open", "high", "low", "close", "volume", "oi"])
    for b in bars:
        w.writerow([_ts(b["date"]), b["open"], b["high"], b["low"], b["close"], b.get("volume") or 0, b.get("oi") or 0])
    data = gzip.compress(buf.getvalue().encode(), mtime=0)
    path = d / (row["tradingsymbol"].replace("/", "_") + ".csv.gz")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return len(data)


def run_daily(day: Optional[date] = None, kite=None, root: Optional[Path] = None,
              is_trading_day: Optional[Callable[[date], bool]] = None) -> dict:
    """One archive run for `day` (default: today IST).  Does not check the feature flag (daily_job does)."""
    if day is None or is_trading_day is None:
        import ist_clock
        day = day or ist_clock.now_ist().date()
        is_trading_day = is_trading_day or ist_clock.is_trading_day
    if not is_trading_day(day):
        return {"day": day.isoformat(), "skipped": "not a trading day"}
    if kite is None:
        from kite_client import kite_client as kite
    root = Path(root) if root is not None else default_root()
    out = {"day": day.isoformat(), "contracts": 0, "bars": 0, "bytes": 0, "errors": 0, "empty": 0}
    with _LOCK:
        used = archive_bytes(root)
        if used >= max_bytes():
            out["skipped"] = f"disk cap reached ({used / 1048576:.0f} MB >= {max_bytes() / 1048576:.0f} MB)"
            logger.warning("[option-archive] {}", out["skipped"])
            return out
        instruments = kite.get_instruments("NFO") or []
        if not instruments:
            out["skipped"] = "no NFO instrument master (no Kite session?)"
            return out
        last = expiry_window(day, is_trading_day)
        spot = _spot(kite, day)
        rows = select_contracts(instruments, day, last, spot)
        out["window"] = [day.isoformat(), last.isoformat()]
        out["spot"] = spot
        start = datetime.combine(day - timedelta(days=MAX_LIFE_DAYS), datetime.min.time())
        end = datetime.combine(day, datetime.max.time().replace(microsecond=0))
        for r in rows:
            if used + out["bytes"] >= max_bytes():
                out["stopped"] = "disk cap reached during the run"
                break
            try:
                bars = kite.historical_data(int(r["instrument_token"]), start, end, "minute", oi=True)
            except Exception as exc:
                out["errors"] += 1
                logger.warning("[option-archive] {} failed: {}", r.get("tradingsymbol"), exc)
                continue
            if not bars:                 # never traded, or no session: keep an earlier file rather than blank it
                out["empty"] += 1
                continue
            out["bytes"] += write_contract(root, r, bars)
            out["contracts"] += 1
            out["bars"] += len(bars)
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "manifest.jsonl", "a") as fh:
            fh.write(json.dumps({**out, "selected": len(rows), "archive_mb": round((used + out["bytes"]) / 1048576, 2)})
                     + "\n")
    logger.info("[option-archive] {}: {} contracts, {} bars, {:.1f} MB written ({} empty, {} errors)", out["day"],
                out["contracts"], out["bars"], out["bytes"] / 1048576, out["empty"], out["errors"])
    return out


def daily_job(**kw) -> dict:
    """Scheduler entry point: re-checks the flag (off -> nothing is read or written)."""
    if not enabled():
        return {"skipped": "OPTION_BAR_ARCHIVER_ENABLED is off"}
    return run_daily(**kw)


def status(root: Optional[Path] = None) -> dict:
    root = Path(root) if root is not None else default_root()
    used = archive_bytes(root)
    return {"enabled": enabled(), "root": str(root), "archive_mb": round(used / 1048576, 2),
            "max_mb": round(max_bytes() / 1048576, 2), "estimate_mb_per_week": ESTIMATE_MB_PER_WEEK}
