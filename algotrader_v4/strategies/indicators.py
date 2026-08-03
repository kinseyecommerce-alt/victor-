"""
strategies/indicators.py
Pure-python daily-bar indicators for the positional strategies.
No numpy/pandas dependency — operates on lists of DailyBar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as _date


@dataclass(frozen=True)
class DailyBar:
    date:   _date
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float = 0.0


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values: list[float], period: int) -> list[float]:
    """EMA seeded with SMA of the first `period` values (standard convention)."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def ema(values: list[float], period: int) -> float | None:
    series = ema_series(values, period)
    return series[-1] if series else None


def true_ranges(bars: list[DailyBar]) -> list[float]:
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            trs.append(b.high - b.low)
        else:
            pc = bars[i - 1].close
            trs.append(max(b.high - b.low, abs(b.high - pc), abs(b.low - pc)))
    return trs


def atr(bars: list[DailyBar], period: int = 20) -> float | None:
    """Wilder-smoothed ATR. period=20 gives the Turtle 'N'."""
    if len(bars) < period + 1:
        return None
    trs = true_ranges(bars)
    val = sum(trs[1:period + 1]) / period
    for tr in trs[period + 1:]:
        val = (val * (period - 1) + tr) / period
    return val


def donchian(bars: list[DailyBar], period: int, exclude_last: bool = True
             ) -> tuple[float, float] | None:
    """(highest high, lowest low) over the prior `period` bars.
    exclude_last=True means the channel is built from bars BEFORE the current
    one, so today's close can be compared against the prior N-day extreme."""
    window = bars[:-1] if exclude_last else bars
    if len(window) < period:
        return None
    w = window[-period:]
    return max(b.high for b in w), min(b.low for b in w)


def daily_returns(closes: list[float]) -> list[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))
            if closes[i - 1] > 0]


def annualized_vol(closes: list[float], lookback: int = 60) -> float | None:
    """Ex-ante volatility estimate: stdev of last `lookback` daily returns, annualized."""
    rets = daily_returns(closes)
    if len(rets) < lookback:
        return None
    r = rets[-lookback:]
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
    return math.sqrt(var) * math.sqrt(252)
