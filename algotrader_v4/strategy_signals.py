"""
strategy_signals.py
Central registry of strategy signal functions, per-strategy configuration,
and backtest gate thresholds.  Imported by adaptive_engine.py and
auto_backtest_runner.py.
"""
from __future__ import annotations

import pandas as pd
import ta

# ── Gate thresholds (pass/fail for backtest qualification) ─────────────────
GATE_THRESHOLDS: dict[str, dict] = {
    "intraday":       {"win_rate": 55, "sharpe": 1.0,  "drawdown": 15, "min_trades": 20, "pf": 1.2},
    "swing":          {"win_rate": 52, "sharpe": 0.9,  "drawdown": 18, "min_trades": 15, "pf": 1.15},
    "futures":        {"win_rate": 53, "sharpe": 0.95, "drawdown": 16, "min_trades": 18, "pf": 1.2},
    "options":        {"win_rate": 40, "sharpe": 0.8,  "drawdown": 30, "min_trades": 12, "pf": 1.5},
    "iron_condor":    {"win_rate": 60, "sharpe": 1.0,  "drawdown": 20, "min_trades": 10, "pf": 1.3},
    "short_straddle": {"win_rate": 58, "sharpe": 0.9,  "drawdown": 25, "min_trades": 10, "pf": 1.25},
    "orb":            {"win_rate": 55, "sharpe": 1.0,  "drawdown": 15, "min_trades": 20, "pf": 1.2},
    "vwap_reversion": {"win_rate": 58, "sharpe": 1.0,  "drawdown": 12, "min_trades": 25, "pf": 1.3},
    # live-agent strategy aliases
    "scalping":       {"win_rate": 55, "sharpe": 0.9,  "drawdown": 10, "min_trades": 30, "pf": 1.15},
    "fno":            {"win_rate": 50, "sharpe": 0.9,  "drawdown": 20, "min_trades": 15, "pf": 1.3},
}

# ── Per-strategy configuration (SL %, target %, product, hold) ────────────
STRATEGY_CONFIG: dict[str, dict] = {
    "intraday":       {"sl": 1.5,  "t1": 3.0,  "t2": 5.0,  "product": "MIS",  "hold_bars": 30},
    "swing":          {"sl": 2.0,  "t1": 5.0,  "t2": 8.0,  "product": "CNC",  "hold_bars": 10},
    "futures":        {"sl": 1.5,  "t1": 3.0,  "t2": 5.0,  "product": "NRML", "hold_bars": 20},
    "options":        {"sl": 30.0, "t1": 60.0, "t2": 100.0,"product": "NRML", "hold_bars": 5},
    "iron_condor":    {"sl": 50.0, "t1": 30.0, "t2": 50.0, "product": "NRML", "hold_bars": 7},
    "short_straddle": {"sl": 40.0, "t1": 35.0, "t2": 60.0, "product": "NRML", "hold_bars": 7},
    "orb":            {"sl": 1.0,  "t1": 2.0,  "t2": 3.5,  "product": "MIS",  "hold_bars": 10},
    "vwap_reversion": {"sl": 0.8,  "t1": 1.5,  "t2": 2.5,  "product": "MIS",  "hold_bars": 8},
    "scalping":       {"sl": 0.5,  "t1": 1.0,  "t2": 1.8,  "product": "MIS",  "hold_bars": 5},
    "fno":            {"sl": 25.0, "t1": 50.0, "t2": 80.0, "product": "NRML", "hold_bars": 3},
}


# ── Signal functions (return pd.Series of {-1, 0, 1}) ─────────────────────

def _signals_intraday(df: pd.DataFrame) -> pd.Series:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    s = pd.Series(0, index=df.index)
    if len(df) < 30:
        return s
    ema9   = ta.trend.EMAIndicator(close, 9).ema_indicator()
    ema21  = ta.trend.EMAIndicator(close, 21).ema_indicator()
    rsi    = ta.momentum.RSIIndicator(close, 14).rsi()
    vwap   = ta.volume.VolumeWeightedAveragePrice(high, low, close, vol).volume_weighted_average_price()
    vol_ma = vol.rolling(20).mean()
    macd_h = ta.trend.MACD(close).macd_diff()
    for i in range(2, len(df)):
        if (close.iloc[i-1] < vwap.iloc[i-1] and close.iloc[i] > vwap.iloc[i]
                and ema9.iloc[i] > ema21.iloc[i] and 45 < rsi.iloc[i] < 67
                and macd_h.iloc[i] > 0 and vol.iloc[i] > vol_ma.iloc[i] * 1.25):
            s.iloc[i] = 1
    return s


def _signals_swing(df: pd.DataFrame) -> pd.Series:
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 55:
        return s
    ema50  = ta.trend.EMAIndicator(close, 50).ema_indicator()
    ema20  = ta.trend.EMAIndicator(close, 20).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close, min(200, len(df) - 1)).ema_indicator()
    rsi    = ta.momentum.RSIIndicator(close, 14).rsi()
    adx    = ta.trend.ADXIndicator(high, low, close, 14).adx()
    for i in range(55, len(df)):
        if (abs(close.iloc[i] - ema50.iloc[i]) / ema50.iloc[i] < 0.018
                and ema20.iloc[i] > ema50.iloc[i] and close.iloc[i] > ema200.iloc[i]
                and 38 < rsi.iloc[i] < 60 and adx.iloc[i] > 20):
            s.iloc[i] = 1
    return s


def _signals_futures(df: pd.DataFrame) -> pd.Series:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    s = pd.Series(0, index=df.index)
    if len(df) < 30:
        return s
    ema9   = ta.trend.EMAIndicator(close, 9).ema_indicator()
    ema21  = ta.trend.EMAIndicator(close, 21).ema_indicator()
    adx    = ta.trend.ADXIndicator(high, low, close, 14).adx()
    vol_ma = vol.rolling(20).mean()
    for i in range(2, len(df)):
        bull = ema9.iloc[i-1] < ema21.iloc[i-1] and ema9.iloc[i] > ema21.iloc[i]
        if bull and adx.iloc[i] > 23 and vol.iloc[i] > vol_ma.iloc[i] * 1.3:
            s.iloc[i] = 1
        bear = ema9.iloc[i-1] > ema21.iloc[i-1] and ema9.iloc[i] < ema21.iloc[i]
        if bear and adx.iloc[i] > 23:
            s.iloc[i] = -1
    return s


def _signals_orb(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(0, index=df.index)
    if len(df) < 10:
        return s
    vol    = df["volume"]
    vol_ma = vol.rolling(10).mean()
    i = 0
    while i < len(df) - 4:
        or_high = df["high"].iloc[i:i+2].max()
        or_low  = df["low"].iloc[i:i+2].min()
        width   = (or_high - or_low) / or_low * 100
        if 0.3 < width < 2.5:
            for j in range(i + 2, min(i + 6, len(df))):
                if df["close"].iloc[j] > or_high and vol.iloc[j] > vol_ma.iloc[j] * 1.4:
                    s.iloc[j] = 1
                    break
                if df["close"].iloc[j] < or_low and vol.iloc[j] > vol_ma.iloc[j] * 1.4:
                    s.iloc[j] = -1
                    break
        i += 26
    return s


def _signals_vwap_reversion(df: pd.DataFrame) -> pd.Series:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    s = pd.Series(0, index=df.index)
    if len(df) < 20:
        return s
    vwap = ta.volume.VolumeWeightedAveragePrice(high, low, close, vol).volume_weighted_average_price()
    rsi  = ta.momentum.RSIIndicator(close, 14).rsi()
    adx  = ta.trend.ADXIndicator(high, low, close, 14).adx()
    for i in range(15, len(df)):
        if not vwap.iloc[i]:
            continue
        dev = abs(close.iloc[i] - vwap.iloc[i]) / vwap.iloc[i] * 100
        if 0.5 < dev < 2.5 and adx.iloc[i] < 22:
            if close.iloc[i] < vwap.iloc[i] and rsi.iloc[i] < 40:
                s.iloc[i] = 1
            if close.iloc[i] > vwap.iloc[i] and rsi.iloc[i] > 60:
                s.iloc[i] = -1
    return s


def _signals_iron_condor(df: pd.DataFrame) -> pd.Series:
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 25:
        return s
    adx  = ta.trend.ADXIndicator(high, low, close, 14).adx()
    bb   = ta.volatility.BollingerBands(close, 20, 2)
    bb_w = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    rsi  = ta.momentum.RSIIndicator(close, 14).rsi()
    for i in range(20, len(df)):
        if adx.iloc[i] < 20 and bb_w.iloc[i] < 2.5 and 38 < rsi.iloc[i] < 62:
            s.iloc[i] = 1
    return s


def _signals_short_straddle(df: pd.DataFrame) -> pd.Series:
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 25:
        return s
    adx    = ta.trend.ADXIndicator(high, low, close, 14).adx()
    atr    = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    atr_ma = atr.rolling(20).mean()
    for i in range(20, len(df)):
        iv_low = atr.iloc[i] / atr_ma.iloc[i] < 0.85 if atr_ma.iloc[i] > 0 else False
        if adx.iloc[i] < 18 and iv_low:
            s.iloc[i] = 1
    return s


def _signals_options(df: pd.DataFrame) -> pd.Series:
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 28:
        return s
    adx  = ta.trend.ADXIndicator(high, low, close, 14).adx()
    bb   = ta.volatility.BollingerBands(close, 20, 2)
    rsi  = ta.momentum.RSIIndicator(close, 14).rsi()
    macd = ta.trend.MACD(close).macd_diff()
    for i in range(28, len(df)):
        if (close.iloc[i] > bb.bollinger_hband().iloc[i] and adx.iloc[i] > 26
                and rsi.iloc[i] > 58 and macd.iloc[i] > 0):
            s.iloc[i] = 1
        if (close.iloc[i] < bb.bollinger_lband().iloc[i] and adx.iloc[i] > 26
                and rsi.iloc[i] < 42 and macd.iloc[i] < 0):
            s.iloc[i] = -1
    return s


# ── Public registry ────────────────────────────────────────────────────────
SIGNAL_REGISTRY: dict[str, callable] = {
    "intraday":       _signals_intraday,
    "swing":          _signals_swing,
    "futures":        _signals_futures,
    "orb":            _signals_orb,
    "vwap_reversion": _signals_vwap_reversion,
    "iron_condor":    _signals_iron_condor,
    "short_straddle": _signals_short_straddle,
    "options":        _signals_options,
}
