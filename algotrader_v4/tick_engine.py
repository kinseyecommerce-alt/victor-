"""
tick_engine.py  (v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-time market data engine.

Data sources (NO Kite for polling):
  Live mode  → KiteConnect WebSocket (true real-time) with NSE India API fallback
  Paper mode → GBM simulator seeded from yfinance last price

Kite is completely absent from order-placement concerns in this file.
Kite WebSocket is used ONLY for tick streaming; orders go via kite_client.py.

Architecture:
  LIVE: KiteConnect WebSocket (threaded) → _ingest_kite_tick()
        NSE India API fallback for symbols not yet received via WebSocket
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
from datetime import datetime
from threading import Lock
from typing import Callable, Optional

import pandas as pd
import numpy as np
import ta
from loguru import logger

from config import settings
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
        bar_ts = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute,
                          (ts.second // self.resolution) * self.resolution)
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


# ── Tick Engine ───────────────────────────────────────────────────────────────

class TickEngine:
    """
    In LIVE mode: KiteConnect WebSocket (true real-time sub-second ticks) with
    NSE India API fallback for symbols not yet received via WebSocket.
    In PAPER mode: GBM simulator seeded from yfinance last price.
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

    # ── Setup ─────────────────────────────────────────────────────────

    def subscribe(self, watchlist: list[dict]) -> None:
        for item in watchlist:
            sym  = item["symbol"]
            exch = item.get("exchange", "NSE")
            self._symbols.append(sym)
            self._exchange[sym]  = exch
            self._bufs_1min[sym] = TickBuffer(60,  maxlen=400)
            self._bufs_5min[sym] = TickBuffer(300, maxlen=200)

        if settings.trading_mode == "PAPER":
            exchanges = {s: self._exchange[s] for s in self._symbols}
            paper_sim.seed(self._symbols, exchanges)

        logger.info("TickEngine: subscribed {} symbols via {}",
                    len(self._symbols),
                    "NSE India API" if settings.trading_mode == "LIVE" else "Paper simulator")

    def start_loop(self) -> None:
        """Called once FastAPI is running — starts the async polling loop and WebSocket."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._task = asyncio.create_task(self._poll_loop())

        # Start KiteConnect WebSocket in LIVE mode if configured
        if (settings.trading_mode == "LIVE"
                and settings.use_kite_websocket
                and settings.kite_access_token):
            try:
                from kite_ticker import KiteTicker
                self._kite_ticker = KiteTicker()
                self._kite_ticker.start(
                    self._symbols,
                    self._ingest_kite_tick,
                    self._loop,
                )
                self._use_ws = True
                logger.info("TickEngine: KiteConnect WebSocket started for {} symbols",
                            len(self._symbols))
            except Exception as exc:
                logger.error("TickEngine: KiteConnect WebSocket failed to start: {} — "
                             "falling back to NSE polling", exc)
                self._use_ws = False

        logger.info("TickEngine poll loop started (ws={})", self._use_ws)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if self._kite_ticker:
            self._kite_ticker.stop()
        logger.info("TickEngine stopped")

    # ── Subscriber management ─────────────────────────────────────────

    def add_subscriber(self, name: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subscribers[name] = q
        return q

    def remove_subscriber(self, name: str) -> None:
        self._subscribers.pop(name, None)

    # ── KiteConnect WebSocket ingest ──────────────────────────────────

    async def _ingest_kite_tick(self, symbol: str, tick: Tick) -> None:
        """Process a tick received directly from KiteConnect WebSocket."""
        if symbol not in self._bufs_1min:
            return
        self._ws_received.add(symbol)

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
                    "vol_ratio":  round(ind.volume_ratio, 2),
                    "source":     "KITE_WS",
                    "ts":         tick.timestamp.isoformat(),
                })
            except Exception:
                pass

    # ── Main poll loop ────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Polls all subscribed symbols every 1 second.
        During off-market hours, slows to every 30 seconds (to check status).
        Symbols already receiving WebSocket ticks are skipped to avoid duplicate processing.
        """
        while self._running:
            t_start = time.monotonic()

            if not is_market_open() and settings.trading_mode == "LIVE":
                await asyncio.sleep(30)
                continue

            # Fetch all symbols concurrently (skip those already fed by WebSocket)
            tasks = [self._fetch_and_process(sym) for sym in self._symbols]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Sleep the remainder of 1 second
            elapsed = time.monotonic() - t_start
            await asyncio.sleep(max(0, 1.0 - elapsed))

    async def _fetch_and_process(self, symbol: str) -> None:
        # Skip NSE polling if KiteConnect WebSocket is already feeding this symbol
        if self._use_ws and symbol in self._ws_received:
            return

        exch = self._exchange.get(symbol, "NSE")

        if settings.trading_mode == "PAPER":
            quote = paper_sim.next_tick(symbol)
        else:
            # Live: fetch from NSE India API (free, no Kite)
            quote = await nse_client.quote_equity(symbol)
            if not quote:
                # Fallback: yfinance current price
                price = yf_client.current_price(symbol, exch)
                if price > 0:
                    quote = Quote(
                        symbol=symbol, ltp=price, open_=price,
                        high=price, low=price, prev_close=price,
                        change=0, change_pct=0, volume=0,
                        bid=price, ask=price,
                    )
                else:
                    return

        tick = Tick.from_quote(quote)

        # Push to candle buffers
        self._bufs_1min[symbol].push(tick.ltp, tick.volume, tick.timestamp)
        self._bufs_5min[symbol].push(tick.ltp, tick.volume, tick.timestamp)

        # Compute live indicators from 1-min candles
        df  = self._bufs_1min[symbol].as_dataframe()
        ind = IndicatorCalc.compute(symbol, tick, df)

        self._latest_tick[symbol] = tick
        self._latest_ind[symbol]  = ind

        snap = MarketSnapshot(
            symbol=symbol, tick=tick, indicators=ind,
            candles_1min=self._bufs_1min[symbol].candles()[-60:],
            candles_5min=self._bufs_5min[symbol].candles()[-30:],
        )

        # Push to all subscriber queues
        for q in self._subscribers.values():
            try:    q.put_nowait(snap)
            except asyncio.QueueFull: pass

        # Broadcast to WebSocket dashboard
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
                    "vol_ratio":  round(ind.volume_ratio, 2),
                    "source":     "NSE" if settings.trading_mode == "LIVE" else "PAPER",
                    "ts":         tick.timestamp.isoformat(),
                })
            except Exception:
                pass

    # ── Query helpers ─────────────────────────────────────────────────

    def latest(self, symbol: str) -> tuple[Optional[Tick], Optional[LiveIndicators]]:
        return self._latest_tick.get(symbol), self._latest_ind.get(symbol)

    def all_latest(self) -> dict[str, dict]:
        result = {}
        for sym in self._latest_tick:
            tick = self._latest_tick[sym]
            ind  = self._latest_ind[sym]
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
                    "source":         "NSE" if settings.trading_mode == "LIVE" else "PAPER",
                    "supertrend":     round(ind.supertrend, 2),
                    "supertrend_dir": ind.supertrend_dir,
                    "hma":            round(ind.hma, 2),
                    "squeeze_on":     ind.squeeze_on,
                    "squeeze_mom":    ind.squeeze_momentum,
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
