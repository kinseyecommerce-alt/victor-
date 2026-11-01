"""
market_filter.py — NIFTY 50 trend filter (price vs 200-day EMA).

The single most robust finding in trend-following research is that long positions held
while the index is below its long-term average suffer most of the big drawdowns. This
filter does not try to predict; it only says whether the index is above or below that
average and the AI / momentum rotation act on it:
  RISK_ON   NIFTY ≥ EMA200  → trade normally
  RISK_OFF  NIFTY <  EMA200 → intraday AI size × TREND_FILTER_RISK_OFF_SIZE, no new swing
                              or momentum buys (puts and shorts are unaffected)
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

INDEX = "NIFTY 50"
_cache: dict[date, dict] = {}


def _ema(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    k = 2 / (n + 1)
    e = sum(values[:n]) / n
    for v in values[n:]:
        e = v * k + e * (1 - k)
    return e


def state(refresh: bool = False) -> dict:
    """{'state': RISK_ON|RISK_OFF|UNKNOWN, 'nifty': ..., 'ema': ..., 'size_factor': ...}"""
    today = now_ist().date()
    if not refresh and today in _cache:
        return _cache[today]
    out = {"state": "UNKNOWN", "nifty": None, "ema": None, "ema_days": settings.trend_filter_ema_days,
           "pct_from_ema": None, "size_factor": 1.0, "enabled": settings.trend_filter_enabled, "as_of": str(today)}
    if not settings.trend_filter_enabled:
        out["state"] = "DISABLED"
        _cache[today] = out
        return out
    try:
        from daily_data import daily_closes
        from tick_engine import tick_engine
        closes = [c for _, c in daily_closes(INDEX, fallback_price=24000.0)]
        tick, _ = tick_engine.latest(INDEX)
        last = float(tick.ltp) if tick and tick.ltp > 0 else (closes[-1] if closes else None)
        ema = _ema(closes + ([last] if last else []), settings.trend_filter_ema_days)
        if ema and last:
            out.update(nifty=round(last, 2), ema=round(ema, 2),
                       pct_from_ema=round((last - ema) / ema * 100, 2))
            if last >= ema:
                out["state"] = "RISK_ON"
            else:
                out["state"], out["size_factor"] = "RISK_OFF", settings.trend_filter_risk_off_size
    except Exception as exc:
        logger.warning("[trend] filter unavailable: {}", exc)
    _cache[today] = out
    return out


def risk_off() -> bool:
    return state()["state"] == "RISK_OFF"


def size_factor() -> float:
    return state()["size_factor"]


def blocks_new_longs() -> bool:
    return settings.trend_filter_block_swing_longs and risk_off()
