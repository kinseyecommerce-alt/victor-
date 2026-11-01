"""
tick_engine.py  (v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-time market data engine.

Data sources:
  Live mode  → KiteConnect WebSocket (true real-time) + kite.quote() REST batch fallback
  Paper mode → GBM simulator seeded from yfinance last price

Architecture:
  LIVE: KiteConnect WebSocket (threaded) → _ingest_kite_tick()
        kite.quote() batch REST fallback for symbols not yet received via WebSocket
  PAPER: GBM simulator seeded from yfinance last price
  Both → TickBuffer (1s / 1min / 5min candles)
       → IndicatorCalc (EMA, RSI, MACD, BB, VWAP, ATR…)
       → MarketSnapshot → asyncio.Queue per agent
       → FastAPI WebSocket → dashboard
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Callable, Optional

import pandas as pd
import numpy as np
import ta
from loguru import logger

from config import settings
from kite_client import kite_client
from market_data import (
    Quote, NSEClient, YFinanceClient, PaperTickSimulator,
    nse_client, yf_client, paper_sim, is_market_open,
)


# ── Core data structures ──────────────────────────────────────────────────────

@dataclass
class Tick:
    symbol:    str
    ltp:       float
    bid:       float
    ask:       float
    volume:    int
    change:    float
    change_pct:float
    high:      float
    low:       float
    open:      float
    timestamp: datetime

    @classmethod
    def from_quote(cls, q: Quote) -> "Tick":
        return cls(
            symbol=q.symbol, ltp=q.ltp, bid=q.bid, ask=q.ask,
            volume=q.volume, change=q.change, change_pct=q.change_pct,
            high=q.high, low=q.low, open=q.open, timestamp=q.ts,
        )


@dataclass
class Candle:
    open: float; high: float; low: float; close: float
    volume: int; ts: datetime


@dataclass
class LiveIndicators:
    symbol:      str
    ltp:         float = 0.0
    bid:         float = 0.0
    ask:         float = 0.0
    spread:      float = 0.0
    # EMA
    ema9:        float = 0.0
    ema21:       float = 0.0
    ema50:       float = 0.0
    ema200:      float = 0.0
    # VWAP
    vwap:        float = 0.0
    # Momentum
    rsi_14:      float = 50.0
    rsi_7:       float = 50.0
    macd:        float = 0.0
    macd_signal: float = 0.0
    macd_hist:   float = 0.0
    # Volatility
    bb_upper:    float = 0.0
    bb_lower:    float = 0.0
    bb_mid:      float = 0.0
    atr_14:      float = 0.0
    # Volume
    volume_ratio:float = 1.0
    obv:         float = 0.0
    # Day context
    day_high:    float = 0.0
    day_low:     float = 0.0
    day_open:    float = 0.0
    change_pct:  float = 0.0
    # Derived labels
    trend:       str = "NEUTRAL"
    momentum:    str = "NEUTRAL"
    volatility:  str = "NORMAL"
    # Supertrend (period=10, mult=3.0)
    supertrend:       float = 0.0
    supertrend_dir:   str   = "NEUTRAL"
    # Hull Moving Average (period=20)
    hma:              float = 0.0
    hma_dir:          str   = "NEUTRAL"
    # TTM Squeeze
    squeeze_on:       bool  = False
    squeeze_momentum: float = 0.0
    # VWAP Bands (2σ / 3σ standard deviation)
    vwap_upper2: float = 0.0
    vwap_lower2: float = 0.0
    vwap_upper3: float = 0.0
    vwap_lower3: float = 0.0
    # Stochastic RSI (14, smooth_k=3, smooth_d=3)
    stoch_rsi_k: float = 50.0
    stoch_rsi_d: float = 50.0
    # Williams %R (14) — range -100 to 0; >-20 overbought, <-80 oversold
    williams_r:  float = -50.0
    # ADX (14) — trend strength, 0-100; >20 means a trend is present
    adx_14:      float = 0.0
    plus_di:     float = 0.0    # +DI (14) — bullish directional pressure
    minus_di:    float = 0.0    # -DI (14) — bearish directional pressure
    # Keltner Channel (EMA20 ± 2×ATR10)
    keltner_upper: float = 0.0
    keltner_mid:   float = 0.0
    keltner_lower: float = 0.0
    # CCI (20) — >+100 overbought / strong up, <-100 oversold / strong down
    cci_20:      float = 0.0
    # Money Flow Index (14) — volume-weighted RSI, 0-100
    mfi_14:      float = 50.0
    # Ichimoku (9, 26, 52): conversion/base lines and the cloud under the current bar
    ichimoku_tenkan:     float = 0.0
    ichimoku_kijun:      float = 0.0
    ichimoku_cloud_top:  float = 0.0
    ichimoku_cloud_bot:  float = 0.0
    computed_at: float = 0.0


@dataclass
class MarketSnapshot:
    symbol:       str
    tick:         Tick
    indicators:   LiveIndicators
    candles_1min: list[Candle] = field(default_factory=list)
    candles_5min: list[Candle] = field(default_factory=list)


# ── Tick buffer ───────────────────────────────────────────────────────────────

class TickBuffer:
    def __init__(self, resolution_sec: int, maxlen: int = 500):
        self.resolution = resolution_sec
        self._candles: deque[Candle] = deque(maxlen=maxlen)
        self._current: Optional[Candle] = None
        self._current_ts: Optional[datetime] = None
        self._lock = Lock()

    def push(self, ltp: float, volume: int, ts: datetime) -> Optional[Candle]:
        # Floor to the bar size within the day (the old seconds-only floor made every
        # bar one minute long, so the "5-minute" buffer held 1-minute candles)
        day_s = ts.hour * 3600 + ts.minute * 60 + ts.second
        b = day_s // self.resolution * self.resolution
        bar_ts = datetime(ts.year, ts.month, ts.day, b // 3600, (b % 3600) // 60, b % 60,
                          tzinfo=ts.tzinfo)
        completed = None
        with self._lock:
            if self._current is None or bar_ts != self._current_ts:
                if self._current:
                    self._candles.append(self._current)
                    completed = self._current
                self._current = Candle(ltp, ltp, ltp, ltp, volume, bar_ts)
                self._current_ts = bar_ts
            else:
                c = self._current
                c.high   = max(c.high, ltp)
                c.low    = min(c.low,  ltp)
                c.close  = ltp
                c.volume += volume
        return completed

    def candles(self) -> list[Candle]:
        with self._lock:
            result = list(self._candles)
            if self._current:
                result.append(self._current)
            return result

    def as_dataframe(self) -> pd.DataFrame:
        cs = self.candles()
        if not cs:
            return pd.DataFrame()
        return pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume, "date": c.ts,
        } for c in cs])


# ── Indicator helpers (Supertrend, HMA, TTM Squeeze) ─────────────────────────

def _wma(series, period: int):
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def _supertrend(high, low, close, period: int = 10, mult: float = 3.0):
    hl2   = (high + low) / 2
    atr   = ta.volatility.AverageTrueRange(high, low, close, period).average_true_range()
    upper = (hl2 + mult * atr).fillna(0)
    lower = (hl2 - mult * atr).fillna(0)
    n     = len(close)
    st_val, st_dir = [0.0] * n, ["NEUTRAL"] * n
    for i in range(1, n):
        if close.iloc[i] > upper.iloc[i - 1]:
            st_dir[i] = "UP"
        elif close.iloc[i] < lower.iloc[i - 1]:
            st_dir[i] = "DOWN"
        else:
            st_dir[i] = st_dir[i - 1]
        if st_dir[i] == "UP":
            st_val[i] = max(float(lower.iloc[i]), st_val[i - 1]) if st_val[i - 1] else float(lower.iloc[i])
        else:
            st_val[i] = min(float(upper.iloc[i]), st_val[i - 1]) if st_val[i - 1] else float(upper.iloc[i])
    return st_val[-1], st_dir[-1]


def _hma(close, period: int = 20):
    half  = max(int(period / 2), 1)
    sqrtp = max(int(period ** 0.5), 1)
    raw   = 2 * _wma(close, half) - _wma(close, period)
    hma_s = _wma(raw, sqrtp)
    v     = hma_s.dropna()
    if len(v) < 2:
        return 0.0, "NEUTRAL"
    direction = "UP" if float(v.iloc[-1]) > float(v.iloc[-2]) else "DOWN"
    return float(v.iloc[-1]), direction


def _ttm_squeeze(close, high, low, period: int = 20, kc_mult: float = 1.5):
    sma     = close.rolling(period).mean()
    atr     = ta.volatility.AverageTrueRange(high, low, close, period).average_true_range()
    bb_obj  = ta.volatility.BollingerBands(close, period, 2)
    bb_u    = bb_obj.bollinger_hband()
    bb_l    = bb_obj.bollinger_lband()
    kc_u    = sma + kc_mult * atr
    kc_l    = sma - kc_mult * atr
    squeeze = bool((bb_u.iloc[-1] < kc_u.iloc[-1]) and (bb_l.iloc[-1] > kc_l.iloc[-1]))
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    delta   = close - (((highest + lowest) / 2) + sma) / 2
    tail    = delta.dropna().iloc[-period:]
    if len(tail) >= 2:
        x = np.arange(len(tail), dtype=float)
        y = tail.values.astype(float)
        if not np.any(np.isnan(y)):
            c   = np.polyfit(x, y, 1)
            mom = float(c[0] * (len(tail) - 1) + c[1])
        else:
            mom = float(tail.iloc[-1])
    else:
        mom = 0.0
    return squeeze, round(mom, 4)


def _vwap_bands(close: pd.Series, high: pd.Series, low: pd.Series,
                volume: pd.Series) -> tuple[float, float, float, float]:
    tp = (high + low + close) / 3.0
    vol_arr = volume.values.astype(float)
    cum_vol = float(vol_arr.sum())
    if cum_vol <= 0:
        mid = float(close.iloc[-1])
        return mid, mid, mid, mid
    vwap = float((tp.values * vol_arr).sum()) / cum_vol
    dev  = float(np.sqrt((vol_arr * (tp.values - vwap) ** 2).sum() / cum_vol))
    return vwap + 2 * dev, vwap - 2 * dev, vwap + 3 * dev, vwap - 3 * dev


def _stoch_rsi(close: pd.Series, period: int = 14,
               smooth_k: int = 3, smooth_d: int = 3) -> tuple[float, float]:
    try:
        k = ta.momentum.stochrsi_k(close, window=period, smooth1=smooth_k, smooth2=smooth_d)
        d = ta.momentum.stochrsi_d(close, window=period, smooth1=smooth_k, smooth2=smooth_d)
        kv = float(k.iloc[-1]) * 100 if not k.empty and not pd.isna(k.iloc[-1]) else 50.0
        dv = float(d.iloc[-1]) * 100 if not d.empty and not pd.isna(d.iloc[-1]) else 50.0
        return round(kv, 2), round(dv, 2)
    except Exception:
        return 50.0, 50.0


def _williams_r(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> float:
    """Williams %R — ranges from -100 to 0. >-20 overbought, <-80 oversold."""
    try:
        wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=period)
        val = float(wr.williams_r().iloc[-1])
        return round(val, 2) if not pd.isna(val) else -50.0
    except Exception:
        return -50.0


def _num(series: pd.Series, idx: int = -1, default: float = 0.0) -> float:
    """Float value at idx, or default when missing/NaN."""
    try:
        val = float(series.iloc[idx])
        return default if pd.isna(val) else round(val, 2)
    except Exception:
        return default


def _adx(close: pd.Series, high: pd.Series, low: pd.Series,
         period: int = 14) -> tuple[float, float, float]:
    """(ADX, +DI, -DI). ADX = trend strength (0-100); DI lines give its direction."""
    try:
        adx = ta.trend.ADXIndicator(high, low, close, window=period)
        return _num(adx.adx()), _num(adx.adx_pos()), _num(adx.adx_neg())
    except Exception:
        return 0.0, 0.0, 0.0


def _keltner(close: pd.Series, high: pd.Series, low: pd.Series) -> tuple[float, float, float]:
    """Keltner Channel (EMA20 ± 2×ATR10) → (upper, mid, lower)."""
    try:
        kc = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=10,
                                          original_version=False, multiplier=2)
        return _num(kc.keltner_channel_hband()), _num(kc.keltner_channel_mband()), \
               _num(kc.keltner_channel_lband())
    except Exception:
        return 0.0, 0.0, 0.0


def _ichimoku(high: pd.Series, low: pd.Series) -> tuple[float, float, float, float]:
    """Ichimoku → (tenkan, kijun, cloud_top, cloud_bottom).
    The cloud under today's bar is the span pair computed 26 bars ago."""
    try:
        ich = ta.trend.IchimokuIndicator(high, low, 9, 26, 52, visual=False)
        tenkan, kijun = _num(ich.ichimoku_conversion_line()), _num(ich.ichimoku_base_line())
        top = bot = 0.0
        if len(high) >= 52 + 26:
            a, b = _num(ich.ichimoku_a(), -27), _num(ich.ichimoku_b(), -27)
            top, bot = max(a, b), min(a, b)
        return tenkan, kijun, top, bot
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


# ── Indicator calculator ──────────────────────────────────────────────────────

class IndicatorCalc:

    @staticmethod
    def compute(sym: str, tick: Tick, df: pd.DataFrame) -> LiveIndicators:
        ind = LiveIndicators(
            symbol=sym, ltp=tick.ltp, bid=tick.bid, ask=tick.ask,
            spread=round(tick.ask - tick.bid, 2),
            day_high=tick.high, day_low=tick.low,
            day_open=tick.open, change_pct=tick.change_pct,
            computed_at=time.time(),
        )
        if df.empty or len(df) < 5:
            return ind

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        n      = len(df)

        try:
            if n >= 9:
                ind.ema9  = float(ta.trend.EMAIndicator(close, 9 ).ema_indicator().iloc[-1])
            if n >= 21:
                ind.ema21 = float(ta.trend.EMAIndicator(close, 21).ema_indicator().iloc[-1])
            if n >= 50:
                ind.ema50 = float(ta.trend.EMAIndicator(close, 50).ema_indicator().iloc[-1])
            if n >= 200:
                ind.ema200= float(ta.trend.EMAIndicator(close, 200).ema_indicator().iloc[-1])

            if n >= 5:
                ind.vwap  = float(ta.volume.VolumeWeightedAveragePrice(
                    high, low, close, volume).volume_weighted_average_price().iloc[-1])

            if n >= 15:
                ind.rsi_14= float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])
            if n >= 8:
                ind.rsi_7 = float(ta.momentum.RSIIndicator(close,  7).rsi().iloc[-1])

            if n >= 26:
                m = ta.trend.MACD(close)
                ind.macd        = float(m.macd().iloc[-1])
                ind.macd_signal = float(m.macd_signal().iloc[-1])
                ind.macd_hist   = float(m.macd_diff().iloc[-1])

            if n >= 20:
                bb = ta.volatility.BollingerBands(close, 20, 2)
                ind.bb_upper = float(bb.bollinger_hband().iloc[-1])
                ind.bb_lower = float(bb.bollinger_lband().iloc[-1])
                ind.bb_mid   = float(bb.bollinger_mavg().iloc[-1])
                vol_avg = volume.rolling(20).mean().iloc[-1]
                ind.volume_ratio = float(volume.iloc[-1] / vol_avg) if vol_avg > 0 else 1.0

            if n >= 14:
                ind.atr_14 = float(ta.volatility.AverageTrueRange(
                    high, low, close, 14).average_true_range().iloc[-1])

            if n >= 5:
                ind.obv = float(ta.volume.OnBalanceVolumeIndicator(
                    close, volume).on_balance_volume().iloc[-1])

            if n >= 11:
                ind.supertrend, ind.supertrend_dir = _supertrend(high, low, close)

            if n >= 25:
                ind.hma, ind.hma_dir = _hma(close)

            if n >= 20:
                ind.squeeze_on, ind.squeeze_momentum = _ttm_squeeze(close, high, low)

            if n >= 10:
                ind.vwap_upper2, ind.vwap_lower2, ind.vwap_upper3, ind.vwap_lower3 = \
                    _vwap_bands(close, high, low, volume)

            if n >= 20:
                ind.stoch_rsi_k, ind.stoch_rsi_d = _stoch_rsi(close)

            if n >= 14:
                ind.williams_r = _williams_r(close, high, low)

            if n >= 28:   # ADX smoothing needs ~2× the window
                ind.adx_14, ind.plus_di, ind.minus_di = _adx(close, high, low)

            if n >= 20:
                ind.keltner_upper, ind.keltner_mid, ind.keltner_lower = _keltner(close, high, low)
                ind.cci_20 = _num(ta.trend.CCIIndicator(high, low, close, 20).cci())

            if n >= 15:
                ind.mfi_14 = _num(ta.volume.MFIIndicator(
                    high, low, close, volume, 14).money_flow_index(), default=50.0)

            if n >= 52:
                (ind.ichimoku_tenkan, ind.ichimoku_kijun,
                 ind.ichimoku_cloud_top, ind.ichimoku_cloud_bot) = _ichimoku(high, low)

        except Exception as exc:
            logger.debug("Indicator compute error {}: {}", sym, exc)

        # Derived labels
        ltp = tick.ltp
        if ind.ema9 and ind.ema21:
            if ltp > ind.ema9 > ind.ema21:   ind.trend = "UP"
            elif ltp < ind.ema9 < ind.ema21: ind.trend = "DOWN"

        if ind.rsi_14:
            if   ind.rsi_14 > 65 and ind.macd_hist > 0: ind.momentum = "STRONG_UP"
            elif ind.rsi_14 > 55:                        ind.momentum = "WEAK_UP"
            elif ind.rsi_14 < 35 and ind.macd_hist < 0: ind.momentum = "STRONG_DOWN"
            elif ind.rsi_14 < 45:                        ind.momentum = "WEAK_DOWN"

        if ind.atr_14 and ind.bb_mid:
            bw = (ind.bb_upper - ind.bb_lower) / ind.bb_mid if ind.bb_mid else 0
            ind.volatility = "HIGH" if bw > 0.04 else "LOW" if bw < 0.01 else "NORMAL"

        return ind


    # ── Vectorised twin of compute(): every bar at once (replay backtests) ───────
    # Same formulas; the value in row i depends only on rows ≤ i, so it equals what the
    # live path would have produced at that bar (up to ewm warm-up noise).

    @staticmethod
    def compute_frame(df: pd.DataFrame, window_cap: int = 500) -> pd.DataFrame:
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"].astype(float)
        n = len(df)
        out = pd.DataFrame(index=df.index)
        idx = np.arange(n)
        out["n"] = idx + 1
        out["ema9"]  = ta.trend.EMAIndicator(close, 9).ema_indicator()
        out["ema21"] = ta.trend.EMAIndicator(close, 21).ema_indicator()
        out["ema50"] = ta.trend.EMAIndicator(close, 50).ema_indicator()
        out["ema200"] = ta.trend.EMAIndicator(close, 200).ema_indicator()
        out["vwap"] = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
        out["rsi_14"] = ta.momentum.RSIIndicator(close, 14).rsi()
        out["rsi_7"] = ta.momentum.RSIIndicator(close, 7).rsi()
        m = ta.trend.MACD(close)
        out["macd"], out["macd_signal"], out["macd_hist"] = m.macd(), m.macd_signal(), m.macd_diff()
        bb = ta.volatility.BollingerBands(close, 20, 2)
        out["bb_upper"], out["bb_lower"], out["bb_mid"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
        vol_avg = volume.rolling(20).mean()
        out["volume_ratio"] = (volume / vol_avg).where(vol_avg > 0, 1.0)
        out["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
        out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        # supertrend (one pass)
        hl2 = (high + low) / 2
        atr10 = ta.volatility.AverageTrueRange(high, low, close, 10).average_true_range()
        upper = (hl2 + 3.0 * atr10).fillna(0).values; lower = (hl2 - 3.0 * atr10).fillna(0).values
        cv = close.values
        st_val, st_dir = np.zeros(n), np.array(["NEUTRAL"] * n, dtype=object)
        for i in range(1, n):
            if cv[i] > upper[i - 1]:   st_dir[i] = "UP"
            elif cv[i] < lower[i - 1]: st_dir[i] = "DOWN"
            else:                      st_dir[i] = st_dir[i - 1]
            if st_dir[i] == "UP":
                st_val[i] = max(lower[i], st_val[i - 1]) if st_val[i - 1] else lower[i]
            else:
                st_val[i] = min(upper[i], st_val[i - 1]) if st_val[i - 1] else upper[i]
        out["supertrend"], out["supertrend_dir"] = st_val, st_dir
        # HMA
        half, sqrtp = max(int(20 / 2), 1), max(int(20 ** 0.5), 1)
        hma = _wma(2 * _wma(close, half) - _wma(close, 20), sqrtp)
        out["hma"] = hma
        out["hma_dir"] = np.where(hma > hma.shift(1), "UP", np.where(hma < hma.shift(1), "DOWN", "NEUTRAL"))
        # TTM squeeze + rolling-regression momentum (fitted value at the window end)
        sma20 = close.rolling(20).mean()
        atr20 = ta.volatility.AverageTrueRange(high, low, close, 20).average_true_range()
        kc_u, kc_l = sma20 + 1.5 * atr20, sma20 - 1.5 * atr20
        out["squeeze_on"] = (out["bb_upper"] < kc_u) & (out["bb_lower"] > kc_l)
        delta = close - (((high.rolling(20).max() + low.rolling(20).min()) / 2) + sma20) / 2
        w = 20; x = np.arange(w, dtype=float); xm = x.mean(); sxx = ((x - xm) ** 2).sum()
        def _fit_end(y):
            if np.isnan(y).any(): return y[-1]
            slope = ((x - xm) * (y - y.mean())).sum() / sxx
            return slope * (w - 1) + (y.mean() - slope * xm)
        out["squeeze_momentum"] = delta.rolling(w).apply(_fit_end, raw=True).round(4)
        # VWAP ±2/3σ bands over the trailing buffer (the live buffer holds ≤ window_cap bars)
        tp = (high + low + close) / 3.0
        cv_ = volume.rolling(window_cap, min_periods=1).sum()
        pv = (tp * volume).rolling(window_cap, min_periods=1).sum()
        vw = (pv / cv_).where(cv_ > 0, close)
        dev = np.sqrt(((volume * (tp - vw) ** 2).rolling(window_cap, min_periods=1).sum() / cv_).where(cv_ > 0, 0.0))
        out["vwap_upper2"], out["vwap_lower2"] = vw + 2 * dev, vw - 2 * dev
        out["vwap_upper3"], out["vwap_lower3"] = vw + 3 * dev, vw - 3 * dev
        out["stoch_rsi_k"] = (ta.momentum.stochrsi_k(close, window=14, smooth1=3, smooth2=3) * 100).round(2)
        out["stoch_rsi_d"] = (ta.momentum.stochrsi_d(close, window=14, smooth1=3, smooth2=3) * 100).round(2)
        out["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().round(2)
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        out["adx_14"], out["plus_di"], out["minus_di"] = adx.adx().round(2), adx.adx_pos().round(2), adx.adx_neg().round(2)
        kc = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=10, original_version=False, multiplier=2)
        out["keltner_upper"], out["keltner_mid"], out["keltner_lower"] = \
            kc.keltner_channel_hband().round(2), kc.keltner_channel_mband().round(2), kc.keltner_channel_lband().round(2)
        out["cci_20"] = ta.trend.CCIIndicator(high, low, close, 20).cci().round(2)
        out["mfi_14"] = ta.volume.MFIIndicator(high, low, close, volume, 14).money_flow_index().round(2)
        ich = ta.trend.IchimokuIndicator(high, low, 9, 26, 52, visual=False)
        out["ichimoku_tenkan"], out["ichimoku_kijun"] = ich.ichimoku_conversion_line().round(2), ich.ichimoku_base_line().round(2)
        a, b = ich.ichimoku_a().shift(26), ich.ichimoku_b().shift(26)      # the cloud under today's bar
        out["ichimoku_cloud_top"], out["ichimoku_cloud_bot"] = np.maximum(a, b).round(2), np.minimum(a, b).round(2)
        return out

    _MIN_BARS = {"ema9": 9, "ema21": 21, "ema50": 50, "ema200": 200, "vwap": 5, "rsi_14": 15, "rsi_7": 8,
                 "macd": 26, "macd_signal": 26, "macd_hist": 26, "bb_upper": 20, "bb_lower": 20, "bb_mid": 20,
                 "volume_ratio": 20, "atr_14": 14, "obv": 5, "supertrend": 11, "supertrend_dir": 11, "hma": 25,
                 "hma_dir": 25, "squeeze_on": 20, "squeeze_momentum": 20, "vwap_upper2": 10, "vwap_lower2": 10,
                 "vwap_upper3": 10, "vwap_lower3": 10, "stoch_rsi_k": 20, "stoch_rsi_d": 20, "williams_r": 14,
                 "adx_14": 28, "plus_di": 28, "minus_di": 28, "keltner_upper": 20, "keltner_mid": 20,
                 "keltner_lower": 20, "cci_20": 20, "mfi_14": 15, "ichimoku_tenkan": 52, "ichimoku_kijun": 52,
                 "ichimoku_cloud_top": 78, "ichimoku_cloud_bot": 78}
    _DEFAULTS = {"stoch_rsi_k": 50.0, "stoch_rsi_d": 50.0, "williams_r": -50.0, "mfi_14": 50.0, "volume_ratio": 1.0}

    @classmethod
    def from_row(cls, sym: str, tick: Tick, row, n: int) -> LiveIndicators:
        """LiveIndicators for bar `row` (a compute_frame row) when the buffer held n bars."""
        ind = LiveIndicators(symbol=sym, ltp=tick.ltp, bid=tick.bid, ask=tick.ask, spread=round(tick.ask - tick.bid, 2),
                             day_high=tick.high, day_low=tick.low, day_open=tick.open, change_pct=tick.change_pct,
                             computed_at=time.time())
        if n < 5:
            return ind
        for k, need in cls._MIN_BARS.items():
            if n < need:
                continue
            v = row[k]
            if isinstance(v, str) or isinstance(v, (bool, np.bool_)):
                setattr(ind, k, bool(v) if k == "squeeze_on" else str(v))
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                if k in cls._DEFAULTS:
                    setattr(ind, k, cls._DEFAULTS[k])
            else:
                setattr(ind, k, float(v))
        ltp = tick.ltp
        if ind.ema9 and ind.ema21:
            if ltp > ind.ema9 > ind.ema21:   ind.trend = "UP"
            elif ltp < ind.ema9 < ind.ema21: ind.trend = "DOWN"
        if ind.rsi_14:
            if   ind.rsi_14 > 65 and ind.macd_hist > 0: ind.momentum = "STRONG_UP"
            elif ind.rsi_14 > 55:                        ind.momentum = "WEAK_UP"
            elif ind.rsi_14 < 35 and ind.macd_hist < 0: ind.momentum = "STRONG_DOWN"
            elif ind.rsi_14 < 45:                        ind.momentum = "WEAK_DOWN"
        if ind.atr_14 and ind.bb_mid:
            bw = (ind.bb_upper - ind.bb_lower) / ind.bb_mid if ind.bb_mid else 0
            ind.volatility = "HIGH" if bw > 0.04 else "LOW" if bw < 0.01 else "NORMAL"
        return ind


# ── Kite quote converter ──────────────────────────────────────────────────────

def _kite_quote_to_quote(symbol: str, data: dict) -> Quote:
    ohlc  = data.get("ohlc", {})
    depth = data.get("depth", {})
    buys  = depth.get("buy",  [{}])
    sells = depth.get("sell", [{}])
    ltp   = data.get("last_price", 0.0)
    return Quote(
        symbol    = symbol,
        ltp       = ltp,
        open_     = ohlc.get("open",  ltp),
        high      = ohlc.get("high",  ltp),
        low       = ohlc.get("low",   ltp),
        prev_close= ohlc.get("close", ltp),
        change    = data.get("change", 0.0),
        change_pct= data.get("change", 0.0),
        volume    = data.get("volume_traded", 0),
        bid       = buys[0].get("price",  ltp) if buys  else ltp,
        ask       = sells[0].get("price", ltp) if sells else ltp,
    )


# ── Tick Engine ───────────────────────────────────────────────────────────────

class TickEngine:
    """
    In LIVE mode: KiteConnect WebSocket (true real-time sub-second ticks) with
    NSE India API fallback for symbols not yet received via WebSocket.
    In PAPER mode: GBM simulator seeded from yfinance last price, or real Kite
    REST quotes when PAPER_DATA_SOURCE=KITE (orders stay simulated either way).
    """

    def __init__(self) -> None:
        self._running        = False
        self._symbols:  list[str]       = []
        self._exchange: dict[str, str]  = {}   # symbol → NSE/BSE

        self._bufs_1min: dict[str, TickBuffer] = {}
        self._bufs_5min: dict[str, TickBuffer] = {}

        self._latest_tick: dict[str, Tick]            = {}
        self._latest_ind:  dict[str, LiveIndicators]  = {}

        self._subscribers:  dict[str, asyncio.Queue]  = {}
        self.ws_broadcast:  Optional[Callable]        = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task]              = None

        # KiteConnect WebSocket state
        self._kite_ticker = None
        self._use_ws: bool = False
        self._ws_received: set[str] = set()
        self._last_quote_warn: float = 0.0

    # ── Setup ─────────────────────────────────────────────────────────

    def subscribe(self, watchlist: list[dict]) -> None:
        added: list[str] = []
        seeded: dict[str, float] = {}
        for item in watchlist:
            sym  = item["symbol"]
            exch = item.get("exchange", "NSE")
            if sym in self._bufs_1min:   # already subscribed (e.g. bot restarted)
                continue
            added.append(sym)
            if item.get("seed_price"):
                seeded[sym] = float(item["seed_price"])
            self._symbols.append(sym)
            self._exchange[sym]  = exch
            self._bufs_1min[sym] = TickBuffer(60,  maxlen=400)
            self._bufs_5min[sym] = TickBuffer(300, maxlen=200)

        if not settings.uses_kite_market_data:
            for sym, px in seeded.items():           # e.g. option legs — yfinance can't price them
                paper_sim.set_price(sym, px)
            to_seed = [s for s in added if s not in seeded]
            if to_seed:
                paper_sim.seed(to_seed, {s: self._exchange[s] for s in to_seed})
            if settings.demo_active:
                self._demo_backfill(added)

        logger.info("TickEngine: subscribed {} symbols via {}",
                    len(self._symbols), self.data_source())

        # The WebSocket needs the symbol list, so (re)start it once symbols exist
        # Only equities ride the WebSocket (it resolves NSE tokens); option legs use REST
        if self._running and any(self._exchange[s] in ("NSE", "BSE") for s in added):
            self._start_ws()

    def _demo_backfill(self, symbols: list[str], minutes: int = 90) -> None:
        """DEMO_MODE only: give new symbols 90 minutes of simulated history so strategies
        that need 10-200 candles can trade straight away instead of after 20+ minutes."""
        import random
        now = datetime.now()
        for sym in symbols:
            px = paper_sim._prices.get(sym)
            if not px:
                continue
            path = [px]
            for _ in range(minutes * 4):
                path.append(path[-1] * (1 + random.gauss(0, 0.0012)))
            scale = px / path[-1]                       # end the history at the current price
            for i, p in enumerate(path[:-1]):
                ts = now - timedelta(seconds=15 * (len(path) - 1 - i))
                vol = random.randint(800, 4000)
                self._bufs_1min[sym].push(p * scale, vol, ts)
                self._bufs_5min[sym].push(p * scale, vol, ts)

    def data_source(self) -> str:
        """Human-readable label for where ticks come from."""
        if settings.trading_mode == "LIVE":
            return "KiteConnect REST+WS"
        if settings.paper_data_source == "KITE":
            return "KiteConnect REST+WS (paper orders)"
        return "Paper simulator"

    def start_loop(self) -> None:
        """Starts the async polling loop and WebSocket. Called at FastAPI startup and on bot restart."""
        if self._running:
            return
        self._running = True
        self._use_ws = False
        self._ws_received.clear()
        self._loop = asyncio.get_event_loop()
        self._task = asyncio.create_task(self._poll_loop())

        self._start_ws()
        logger.info("TickEngine poll loop started (ws={})", self._use_ws)

    def _start_ws(self) -> None:
        """(Re)start the tick WebSocket for all subscribed symbols — TrueData preferred, Kite as fallback.
        Needs symbols, so it runs from subscribe() (or start_loop() on a restart), not at server startup."""
        if not settings.uses_kite_market_data or not self._symbols or not self._loop:
            return
        if self._kite_ticker:
            self._kite_ticker.stop()
            self._kite_ticker = None
        self._use_ws = False
        self._ws_received.clear()

        if settings.use_truedata_websocket and settings.truedata_username:
            try:
                from truedata_client import truedata_ticker
                self._kite_ticker = truedata_ticker  # reuse slot; same interface
                truedata_ticker.start(
                    self._symbols,
                    self._ingest_kite_tick,
                    self._loop,
                )
                self._use_ws = True
                logger.info("TickEngine: TrueData WebSocket started for {} symbols",
                            len(self._symbols))
            except Exception as exc:
                logger.error("TickEngine: TrueData WebSocket failed: {} — "
                             "falling back to Kite REST", exc)
        elif settings.use_kite_websocket and kite_client.has_session():
            try:
                from kite_ticker import KiteTicker
                self._kite_ticker = KiteTicker()
                self._use_ws = self._kite_ticker.start(
                    self._symbols,
                    self._ingest_kite_tick,
                    self._loop,
                )
                if self._use_ws:
                    logger.info("TickEngine: KiteConnect WebSocket started for {} symbols",
                                len(self._symbols))
            except Exception as exc:
                logger.error("TickEngine: KiteConnect WebSocket failed to start: {} — "
                             "falling back to Kite REST", exc)
                self._use_ws = False

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if self._kite_ticker:
            self._kite_ticker.stop()
            self._kite_ticker = None
        self._use_ws = False
        logger.info("TickEngine stopped")

    # ── Subscriber management ─────────────────────────────────────────

    def add_subscriber(self, name: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subscribers[name] = q
        return q

    def remove_subscriber(self, name: str) -> None:
        self._subscribers.pop(name, None)

    # ── Shared tick processing ────────────────────────────────────────

    async def _process_tick(self, symbol: str, tick: Tick, source: str = "KITE") -> None:
        """Candle buffer push → indicator calc → snapshot broadcast. Used by both WS and REST paths."""
        if symbol not in self._bufs_1min:
            return

        self._bufs_1min[symbol].push(tick.ltp, tick.volume, tick.timestamp)
        self._bufs_5min[symbol].push(tick.ltp, tick.volume, tick.timestamp)

        df  = self._bufs_1min[symbol].as_dataframe()
        ind = IndicatorCalc.compute(symbol, tick, df)

        self._latest_tick[symbol] = tick
        self._latest_ind[symbol]  = ind

        snap = MarketSnapshot(
            symbol=symbol, tick=tick, indicators=ind,
            candles_1min=self._bufs_1min[symbol].candles()[-60:],
            candles_5min=self._bufs_5min[symbol].candles()[-30:],
        )

        for q in self._subscribers.values():
            try:    q.put_nowait(snap)
            except asyncio.QueueFull: pass

        if self.ws_broadcast:
            try:
                await self.ws_broadcast({
                    "event":      "tick",
                    "symbol":     symbol,
                    "ltp":        tick.ltp,
                    "bid":        tick.bid,
                    "ask":        tick.ask,
                    "change_pct": round(tick.change_pct, 2),
                    "volume":     tick.volume,
                    "day_high":   tick.high,
                    "day_low":    tick.low,
                    "trend":      ind.trend,
                    "momentum":   ind.momentum,
                    "volatility": ind.volatility,
                    "rsi":        round(ind.rsi_14, 1),
                    "vwap":       round(ind.vwap, 2),
                    "ema9":       round(ind.ema9,  2),
                    "ema21":      round(ind.ema21, 2),
                    "macd_hist":  round(ind.macd_hist, 4),
                    "vol_ratio":   round(ind.volume_ratio, 2),
                    "supertrend":  ind.supertrend_dir,
                    "squeeze_on":  ind.squeeze_on,
                    "stoch_rsi_k": ind.stoch_rsi_k,
                    "williams_r":  ind.williams_r,
                    "source":     source,
                    "ts":         tick.timestamp.isoformat(),
                })
            except Exception:
                pass

    # ── KiteConnect WebSocket ingest ──────────────────────────────────

    async def _ingest_kite_tick(self, symbol: str, tick: Tick) -> None:
        """Process a tick received directly from KiteConnect WebSocket."""
        self._ws_received.add(symbol)
        await self._process_tick(symbol, tick, source="KITE_WS")

    # ── Main poll loop ────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Polls all subscribed symbols every 1 second.
        During off-market hours, slows to every 30 seconds (to check status).
        Symbols already receiving WebSocket ticks are skipped to avoid duplicate processing.
        """
        while self._running:
            t_start = time.monotonic()

            if not is_market_open() and settings.uses_kite_market_data:
                await asyncio.sleep(30)
                continue

            if settings.uses_kite_market_data:
                await self._fetch_kite_batch()
            else:
                tasks = [self._fetch_and_process(sym) for sym in self._symbols]
                await asyncio.gather(*tasks, return_exceptions=True)

            # Sleep the remainder of 1 second
            elapsed = time.monotonic() - t_start
            await asyncio.sleep(max(0, 1.0 - elapsed))

    async def _fetch_and_process(self, symbol: str) -> None:
        """PAPER mode only — generate next GBM tick and process it."""
        quote = paper_sim.next_tick(symbol)
        tick  = Tick.from_quote(quote)
        await self._process_tick(symbol, tick, source="PAPER")

    async def _fetch_kite_batch(self) -> None:
        """Single Kite quote() call for all non-WebSocket symbols (LIVE, or PAPER with Kite data)."""
        # Poll over REST for symbols the WebSocket isn't covering — including all of
        # them while the socket is disconnected (otherwise they'd go stale)
        ws_live = self._use_ws and getattr(self._kite_ticker, "is_connected", True)
        pending = [s for s in self._symbols
                   if not (ws_live and s in self._ws_received)]
        if not pending:
            return
        instruments = [f"{self._exchange.get(s, 'NSE')}:{s}" for s in pending]
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: kite_client.quote_kite(instruments)
            )
        except Exception as exc:
            # Throttle: a missing/expired token would otherwise log every second
            if time.monotonic() - self._last_quote_warn > 60:
                self._last_quote_warn = time.monotonic()
                logger.warning("[tick] Kite batch quote failed: {}", exc)
            return
        for sym in pending:
            exch = self._exchange.get(sym, "NSE")
            key  = f"{exch}:{sym}"
            data = raw.get(key)
            if not data:
                continue
            quote = _kite_quote_to_quote(sym, data)
            tick  = Tick.from_quote(quote)
            await self._process_tick(sym, tick, source="KITE_REST")

    # ── Query helpers ─────────────────────────────────────────────────

    def latest(self, symbol: str) -> tuple[Optional[Tick], Optional[LiveIndicators]]:
        return self._latest_tick.get(symbol), self._latest_ind.get(symbol)

    def all_latest(self) -> dict[str, dict]:
        result = {}
        for sym in self._latest_tick:
            tick = self._latest_tick[sym]
            ind  = self._latest_ind.get(sym)
            if tick and ind:
                result[sym] = {
                    "ltp":        tick.ltp,
                    "bid":        tick.bid,
                    "ask":        tick.ask,
                    "change_pct": round(tick.change_pct, 2),
                    "day_high":   tick.high,
                    "day_low":    tick.low,
                    "trend":      ind.trend,
                    "momentum":   ind.momentum,
                    "volatility": ind.volatility,
                    "rsi_14":     round(ind.rsi_14, 1),
                    "vwap":       round(ind.vwap, 2),
                    "ema9":       round(ind.ema9, 2),
                    "ema21":      round(ind.ema21, 2),
                    "macd_hist":      round(ind.macd_hist, 4),
                    "vol_ratio":      round(ind.volume_ratio, 2),
                    "source":         "NSE" if settings.uses_kite_market_data else "PAPER",
                    "supertrend":     round(ind.supertrend, 2),
                    "supertrend_dir": ind.supertrend_dir,
                    "hma":            round(ind.hma, 2),
                    "squeeze_on":     ind.squeeze_on,
                    "squeeze_mom":    ind.squeeze_momentum,
                    "vwap_u2":        round(ind.vwap_upper2, 2),
                    "vwap_l2":        round(ind.vwap_lower2, 2),
                    "vwap_u3":        round(ind.vwap_upper3, 2),
                    "vwap_l3":        round(ind.vwap_lower3, 2),
                    "stoch_rsi_k":    ind.stoch_rsi_k,
                    "stoch_rsi_d":    ind.stoch_rsi_d,
                    "williams_r":     ind.williams_r,
                    "adx_14":         ind.adx_14,
                    "plus_di":        ind.plus_di,
                    "minus_di":       ind.minus_di,
                    "keltner_u":      ind.keltner_upper,
                    "keltner_l":      ind.keltner_lower,
                    "cci_20":         ind.cci_20,
                    "mfi_14":         ind.mfi_14,
                    "ichimoku_tenkan":    ind.ichimoku_tenkan,
                    "ichimoku_kijun":     ind.ichimoku_kijun,
                    "ichimoku_cloud_top": ind.ichimoku_cloud_top,
                    "ichimoku_cloud_bot": ind.ichimoku_cloud_bot,
                    "ts":             tick.timestamp.isoformat(),
                }
        return result

    def symbols(self) -> list[str]:
        return list(self._symbols)

    # ── Historical data (for backtesting + warm-up) ───────────────────

    def get_historical(
        self, symbol: str, exchange: str = "NSE",
        interval: str = "1m", period: str = "5d",
    ) -> pd.DataFrame:
        """Fetch OHLCV from yfinance — used by backtest engine and signal engine."""
        return yf_client.historical(symbol, exchange, interval, period)

    # ── Market status ─────────────────────────────────────────────────

    async def get_market_status(self) -> dict:
        return await nse_client.market_status()

    async def get_option_chain(self, symbol: str) -> Optional[dict]:
        return await nse_client.option_chain(symbol)


tick_engine = TickEngine()
