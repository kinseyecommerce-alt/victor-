"""
multi_timeframe.py
Derives 5-min and 15-min candles from existing 1-min candle history.
No extra API calls — works entirely from data already in the tick engine.

Returns alignment score (0-3): how many of (1m, 5m, 15m) agree with the signal.
Minimum required alignment is configurable (default 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger


Direction = Literal["bullish", "bearish", "neutral"]


@dataclass
class MTFResult:
    aligned: bool          # meets minimum_alignment threshold
    score: int             # 0-3: how many TFs agree
    tf_1m:  Direction = "neutral"
    tf_5m:  Direction = "neutral"
    tf_15m: Direction = "neutral"


def _candles_to_df(candles: list) -> pd.DataFrame:
    """Convert candle objects to a DataFrame with OHLCV columns."""
    rows = []
    for c in candles:
        rows.append({
            "open":   getattr(c, "open",   c[0] if isinstance(c, (list, tuple)) else 0),
            "high":   getattr(c, "high",   c[1] if isinstance(c, (list, tuple)) else 0),
            "low":    getattr(c, "low",    c[2] if isinstance(c, (list, tuple)) else 0),
            "close":  getattr(c, "close",  c[3] if isinstance(c, (list, tuple)) else 0),
            "volume": getattr(c, "volume", c[4] if isinstance(c, (list, tuple)) else 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )


def _ema(series: pd.Series, period: int) -> float:
    """Fast EMA of the last N values."""
    if len(series) < period:
        return float(series.iloc[-1]) if len(series) else 0.0
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def _trend_direction(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> Direction:
    """EMA crossover direction on a candle DataFrame."""
    if len(df) < slow + 2:
        return "neutral"
    closes = df["close"]
    ema_f = closes.ewm(span=fast, adjust=False).mean()
    ema_s = closes.ewm(span=slow, adjust=False).mean()
    diff     = float(ema_f.iloc[-1]) - float(ema_s.iloc[-1])
    diff_lag = float(ema_f.iloc[-2]) - float(ema_s.iloc[-2])
    if diff > 0 and diff > abs(diff_lag) * 0.1:
        return "bullish"
    if diff < 0 and abs(diff) > abs(diff_lag) * 0.1:
        return "bearish"
    return "neutral"


def _aggregate(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1-min OHLCV into N-minute bars."""
    n = len(df_1m)
    if n < minutes:
        return pd.DataFrame(columns=df_1m.columns)
    bars = []
    for i in range(0, n - minutes + 1, minutes):
        chunk = df_1m.iloc[i:i + minutes]
        bars.append({
            "open":   float(chunk["open"].iloc[0]),
            "high":   float(chunk["high"].max()),
            "low":    float(chunk["low"].min()),
            "close":  float(chunk["close"].iloc[-1]),
            "volume": float(chunk["volume"].sum()),
        })
    return pd.DataFrame(bars)


def check(snap, action: str) -> MTFResult:
    """
    Check multi-timeframe alignment for a signal.
    snap.candles_1min — list of 1-min candle objects, oldest first.
    action — "BUY" or "SELL"
    """
    from config import settings

    candles = getattr(snap, "candles_1min", [])
    if not candles:
        return MTFResult(aligned=True, score=3)  # no data → don't block

    try:
        df_1m = _candles_to_df(candles[-90:])  # last 90 min
        df_5m  = _aggregate(df_1m, 5)
        df_15m = _aggregate(df_1m, 15)

        t_1m  = _trend_direction(df_1m, fast=9,  slow=21)
        t_5m  = _trend_direction(df_5m, fast=9,  slow=21)
        t_15m = _trend_direction(df_15m, fast=5, slow=14)

        wanted: Direction = "bullish" if action == "BUY" else "bearish"

        # neutral is NOT a veto — it just doesn't count as agreement
        score = sum([
            t_1m  == wanted,
            t_5m  == wanted,
            t_15m == wanted,
        ])

        # Veto only if a higher TF actively OPPOSES the signal
        actively_against = (
            (t_5m  not in (wanted, "neutral")) or
            (t_15m not in (wanted, "neutral"))
        )
        min_align = getattr(settings, "mtf_min_alignment", 2)
        aligned = (score >= min_align) and not actively_against

        # Attach to snapshot for claude_trade_gate context
        snap.mtf_alignment = {
            "1m": t_1m, "5m": t_5m, "15m": t_15m,
            "score": score, "aligned": aligned,
        }
        return MTFResult(aligned=aligned, score=score, tf_1m=t_1m, tf_5m=t_5m, tf_15m=t_15m)

    except Exception as exc:
        logger.debug("[mtf] {} check error: {} — not blocking", snap.symbol, exc)
        return MTFResult(aligned=True, score=0)
