"""
universe.py — the rolling trading universe for the opening-range agents.

Evidence (replay on real Kite minute bars, live capacity of a ₹2L MIS bucket with 2 positions per agent,
judged walk-forward): which stocks the two intraday slots go to matters more than any entry filter.
Trading only the stocks whose intraday + scalping signals made money over the previous sessions beat
"first come, first served" on data the rule had not seen.

Each trading day after the close (16:30 IST) this module replays the intraday and scalping agents over the
last `lookback` sessions for the NIFTY 100 in the background worker, sums every signal's net result per
stock (all signals, not only the ones capacity allowed), and keeps the stocks with a positive total as the
next session's watchlist. Auto-start uses it when it was computed for today; otherwise the fixed
AUTO_START_WATCHLIST applies.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist, is_trading_day, next_trading_day

_FILE = Path(os.getenv("UNIVERSE_FILE")
             or Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
             .parent / "universe.json")
AGENTS = ("intraday", "scalping")


def candidates() -> list[str]:
    from nifty100 import NIFTY_100
    from tick_engine import canonical_symbol
    return sorted({canonical_symbol(s) for s in NIFTY_100})


def load() -> dict:
    try:
        if _FILE.exists():
            return json.loads(_FILE.read_text())
    except Exception as exc:
        logger.debug("[universe] cannot read {}: {}", _FILE, exc)
    return {}


def _save(st: dict) -> None:
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, default=str))
    tmp.replace(_FILE)


def today_watchlist(today: Optional[date] = None) -> list[str]:
    """The rolling universe computed for `today` (default: today IST), or [] when there is none."""
    st = load()
    d = (today or now_ist().date()).isoformat()
    return list(st.get("symbols") or []) if st.get("for_date") == d else []


def recent_sessions(lookback: int, now=None) -> list[str]:
    """The last `lookback` NSE sessions that are complete at `now` (today counts after 15:30)."""
    from datetime import timedelta
    now = now or now_ist()
    d = now.date()
    if not (is_trading_day(d) and (now.hour, now.minute) >= (15, 30)):
        d -= timedelta(days=1)
    out: list[str] = []
    while len(out) < lookback:
        if is_trading_day(d):
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return sorted(out)


def select(trades_by_agent: dict[str, list[dict]], session_days: list[str], lookback: int,
           min_symbols: int) -> tuple[list[str], dict[str, float]]:
    """Per-stock net of every signal in the last `lookback` sessions; keep the positive ones."""
    keep_days = set(sorted(session_days)[-lookback:])
    per: dict[str, float] = defaultdict(float)
    for trades in trades_by_agent.values():
        for t in trades:
            if str(t["entry_ts"])[:10] in keep_days:
                per[t["symbol"]] += float(t["net"])
    chosen = sorted((s for s, v in per.items() if v > 0), key=lambda s: -per[s])
    if len(chosen) < min_symbols:
        return [], dict(per)                        # too thin to trust: fall back to the fixed list
    return chosen, dict(per)


async def refresh(lookback: Optional[int] = None, symbols: Optional[list[str]] = None) -> dict:
    """Replay the opening-range agents over the recent sessions and store the next session's universe."""
    from bg_compute import run_in_process
    lookback = lookback or settings.universe_lookback_sessions
    syms = symbols or candidates()
    days = int(lookback * 1.6) + 6                   # calendar days that cover `lookback` sessions
    trades_by_agent: dict[str, list[dict]] = {}
    for a in AGENTS:
        out = await run_in_process("replay_job", a, syms, days, 5.0, None, True)
        trades_by_agent[a] = out.get("trades") or []
    sessions = recent_sessions(lookback)
    chosen, per = select(trades_by_agent, sessions, lookback, settings.universe_min_symbols)
    today = now_ist().date()
    for_date = today if (is_trading_day(today) and now_ist().hour < 9) else next_trading_day(today)
    st = {"for_date": for_date.isoformat(), "computed_at": now_ist().isoformat(timespec="seconds"),
          "lookback_sessions": lookback, "candidates": len(syms), "symbols": chosen,
          "per_symbol_net": {k: round(v, 2) for k, v in sorted(per.items(), key=lambda kv: -kv[1])},
          "signals": {a: len(v) for a, v in trades_by_agent.items()},
          "fallback": not chosen}
    _save(st)
    logger.info("[universe] {} stocks for {} (of {} with signals in the last {} sessions){}",
                len(chosen), st["for_date"], len(per), lookback, " — too few, fixed list will be used" if not chosen else "")
    return st
