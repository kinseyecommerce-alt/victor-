"""
daily_data.py — daily candles and swing indicators for the AI swing trader.

Source, in order:
  * Kite historical data (LIVE, or PAPER with PAPER_DATA_SOURCE=KITE)
  * yfinance, if Kite is unavailable on a real-price setup
  * PAPER + SIM: a synthetic, repeatable history ending at the simulator's current price
    (the simulator's prices are made up, so real history would not line up with them)

Completed days are cached for the day; today's bar is rebuilt from the live tick on every call.
"""
from __future__ import annotations

import hashlib
import math
import random
from datetime import date, datetime, timedelta
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

_LOOKBACK_DAYS = 400                     # calendar days → ~260 sessions, enough for EMA200 / 52w
_cache: dict[tuple[str, date], list[dict]] = {}


def _from_kite(symbol: str) -> list[dict]:
    from kite_client import kite_client
    tok = kite_client.get_instrument_tokens([symbol], "NSE").get(symbol)
    if not tok:
        return []
    end = now_ist().replace(tzinfo=None)
    rows = kite_client.historical_data(tok, end - timedelta(days=_LOOKBACK_DAYS), end, "day")
    return [{"date": r["date"].date() if hasattr(r["date"], "date") else r["date"],
             "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
             "close": float(r["close"]), "volume": int(r.get("volume") or 0)} for r in rows]


def _from_yfinance(symbol: str) -> list[dict]:
    from market_data import yf_client
    df = yf_client.historical(symbol, "NSE", interval="1d", period="2y")
    return [{"date": r.date.date(), "open": float(r.open), "high": float(r.high), "low": float(r.low),
             "close": float(r.close), "volume": int(r.volume)} for r in df.itertuples()]


def _synthetic(symbol: str, last_price: float, today: date) -> list[dict]:
    """Repeatable random walk with a per-symbol drift, scaled to end at last_price."""
    seed = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    drift = rng.uniform(-0.0006, 0.0012)        # most symbols mildly up, some down
    vol = rng.uniform(0.010, 0.020)
    days, d = [], today - timedelta(days=1)
    while len(days) < 260:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    px, raw = 100.0, []
    for day in days:
        o = px
        c = o * math.exp(drift + rng.gauss(0, vol))
        h = max(o, c) * (1 + abs(rng.gauss(0, vol / 2)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, vol / 2)))
        raw.append((day, o, h, lo, c, int(rng.uniform(0.6, 1.6) * 1_000_000)))
        px = c
    k = last_price / raw[-1][4] if last_price > 0 else 1.0
    return [{"date": day, "open": o * k, "high": h * k, "low": lo * k, "close": c * k, "volume": v}
            for day, o, h, lo, c, v in raw]


def _completed_days(symbol: str, last_price: float) -> list[dict]:
    today = now_ist().date()
    key = (symbol, today)
    if key in _cache:
        return _cache[key]
    rows: list[dict] = []
    if settings.uses_kite_market_data:
        try:
            rows = _from_kite(symbol)
        except Exception as exc:
            logger.warning("[daily] Kite history for {} failed: {}", symbol, exc)
        if not rows:
            try:
                rows = _from_yfinance(symbol)
            except Exception as exc:
                logger.warning("[daily] yfinance history for {} failed: {}", symbol, exc)
    elif last_price > 0:
        rows = _synthetic(symbol, last_price, today)
    rows = [r for r in rows if r["date"] < today]
    if rows:
        _cache[key] = rows
    return rows


def _ema(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    k = 2 / (n + 1)
    e = sum(values[:n]) / n
    for v in values[n:]:
        e = v * k + e * (1 - k)
    return e


def _rsi(closes: list[float], n: int = 14) -> Optional[float]:
    if len(closes) <= n:
        return None
    gains = losses = 0.0
    for a, b in zip(closes[:n], closes[1:n + 1]):
        gains += max(b - a, 0); losses += max(a - b, 0)
    ag, al = gains / n, losses / n
    for a, b in zip(closes[n:], closes[n + 1:]):
        ag = (ag * (n - 1) + max(b - a, 0)) / n
        al = (al * (n - 1) + max(a - b, 0)) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def _atr(bars: list[dict], n: int = 14) -> Optional[float]:
    if len(bars) <= n:
        return None
    trs = [max(b["high"] - b["low"], abs(b["high"] - p["close"]), abs(b["low"] - p["close"]))
           for p, b in zip(bars, bars[1:])]
    a = sum(trs[:n]) / n
    for t in trs[n:]:
        a = (a * (n - 1) + t) / n
    return a


def daily_view(symbol: str, candles: int = 30) -> Optional[dict]:
    """Daily indicators + the last `candles` daily bars (today's bar is live, from the tick)."""
    from tick_engine import tick_engine
    tick, _ = tick_engine.latest(symbol)
    ltp = float(tick.ltp) if tick and tick.ltp > 0 else 0.0
    bars = list(_completed_days(symbol, ltp))
    if len(bars) < 30:
        return None
    if ltp > 0:
        bars.append({"date": now_ist().date(), "open": tick.open or ltp, "high": max(tick.high or ltp, ltp),
                     "low": min(tick.low or ltp, ltp) if tick.low else ltp, "close": ltp,
                     "volume": int(tick.volume or 0), "live": True})
    closes = [b["close"] for b in bars]
    prior = bars[:-1]
    last = bars[-1]
    e20, e50, e200 = _ema(closes, 20), _ema(closes, 50), _ema(closes, 200)
    hi20 = max(b["high"] for b in prior[-20:])
    lo20 = min(b["low"] for b in prior[-20:])
    year = prior[-250:]
    vols = [b["volume"] for b in prior[-20:] if b["volume"]]
    trend = "UNKNOWN"
    if e20 and e50:
        if last["close"] > e20 > e50 and (e200 is None or e50 > e200):
            trend = "UP"
        elif last["close"] < e20 < e50 and (e200 is None or e50 < e200):
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"
    r = lambda v: round(v, 2) if isinstance(v, float) else v
    return {
        "symbol": symbol, "ltp": r(last["close"]), "trend": trend,
        "ema20": r(e20), "ema50": r(e50), "ema200": r(e200),
        "rsi14": r(_rsi(closes)), "atr14": r(_atr(bars)),
        "atr_pct": r(_atr(bars) / last["close"] * 100) if _atr(bars) else None,
        "high_20d": r(hi20), "low_20d": r(lo20),
        "high_52w": r(max(b["high"] for b in year)), "low_52w": r(min(b["low"] for b in year)),
        "pct_from_20d_high": r((last["close"] - hi20) / hi20 * 100),
        "volume_vs_20d_avg": r(last["volume"] / (sum(vols) / len(vols))) if vols and last["volume"] else None,
        "source": ("kite/yfinance" if settings.uses_kite_market_data else "synthetic (paper simulator)"),
        "candles_1d": {"columns": ["date", "open", "high", "low", "close", "volume"],
                       "rows": [[str(b["date"]), r(b["open"]), r(b["high"]), r(b["low"]), r(b["close"]),
                                 b["volume"]] for b in bars[-candles:]]},
    }


def swing_candidates(symbols: list[str], limit: int = 5) -> list[dict]:
    """A shortlist for the AI to look at, not a buy signal:
      BULLISH — uptrend at a 20-day breakout or a pullback to the 20-day EMA
      BEARISH — downtrend at a 20-day breakdown or a bounce up into the 20-day EMA"""
    out = []
    for sym in symbols:
        try:
            v = daily_view(sym, candles=0)
        except Exception as exc:
            logger.debug("[daily] {} skipped: {}", sym, exc)
            continue
        if not v or v["rsi14"] is None or not v["ema20"]:
            continue
        near_ema = abs(v["ltp"] - v["ema20"]) / v["ema20"] * 100 <= 1.5
        vol_bonus = 1 if (v["volume_vs_20d_avg"] or 0) >= 1.3 else 0
        if v["trend"] == "UP" and 45 <= v["rsi14"] <= 72:
            breakout = v["pct_from_20d_high"] >= -1.0
            if not (breakout or near_ema):
                continue
            direction, setup = "BULLISH", "breakout" if breakout else "pullback to EMA20"
            score = (2 if breakout else 1) + vol_bonus + (1 if 55 <= v["rsi14"] <= 68 else 0)
        elif v["trend"] == "DOWN" and 28 <= v["rsi14"] <= 55:
            breakdown = (v["ltp"] - v["low_20d"]) / v["low_20d"] * 100 <= 1.0
            if not (breakdown or near_ema):
                continue
            direction, setup = "BEARISH", "breakdown" if breakdown else "bounce into EMA20"
            score = (2 if breakdown else 1) + vol_bonus + (1 if 32 <= v["rsi14"] <= 45 else 0)
        else:
            continue
        out.append({"symbol": sym, "direction": direction, "setup": setup, "score": score,
                    **{k: v[k] for k in ("ltp", "rsi14", "atr_pct", "ema20", "ema50", "high_20d", "low_20d",
                                         "volume_vs_20d_avg")}})
    return sorted(out, key=lambda c: -c["score"])[:limit]
