"""
signal_engine.py
Generates trade signals two ways:
  1. Pure AI  → asks Claude for a signal given symbol + strategy
  2. Technical → computes RSI, MACD, EMA, VWAP from real OHLCV data,
                 then sends the computed indicators to Claude for a final verdict.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Literal

import anthropic
import pandas as pd
import ta                        # technical analysis library
from loguru import logger

from config import settings
from market_data import yf_client

STRATEGY = Literal["intraday", "fno", "swing", "scalping"]

SYSTEM_PROMPT = """
You are a senior NSE/BSE quant analyst. You receive either:
  a) A raw stock symbol + strategy name, OR
  b) Pre-computed technical indicator values for a stock.

Return ONLY a valid JSON object with NO markdown, no extra text, no code fences:
{
  "signal":       "BUY" | "SELL" | "HOLD",
  "confidence":   0-100,
  "entry_price":  number,
  "stop_loss":    number,
  "target_1":     number,
  "target_2":     number,
  "risk_reward":  number,
  "timeframe":    string,
  "strategy":     string,
  "reasoning":    string (max 80 words),
  "indicators": [
    {"name": string, "value": string, "signal": "BUY"|"SELL"|"Neutral"}
  ]
}
"""


class SignalEngine:

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        symbol: str,
        exchange: str = "NSE",
        strategy: STRATEGY = "intraday",
    ) -> dict:
        """
        Full pipeline:
          1. Fetch historical OHLCV from Kite
          2. Compute technical indicators with `ta`
          3. Send computed values to Claude
          4. Return parsed signal dict
        """
        try:
            df = self._fetch_candles(symbol, exchange, strategy)
            if df is not None and len(df) >= 50:
                indicators = self._compute_indicators(df, strategy)
                signal = self._claude_with_indicators(symbol, strategy, indicators, df)
            else:
                # Fallback: ask Claude without real data (still useful for paper mode)
                signal = self._claude_raw(symbol, strategy)
        except Exception as exc:
            logger.warning("Signal engine error for {}: {}. Falling back to raw Claude.", symbol, exc)
            signal = self._claude_raw(symbol, strategy)

        logger.info("Signal {} {} → {} (conf {}%)", symbol, strategy, signal.get("signal"), signal.get("confidence"))
        return signal

    # ── Data fetching ─────────────────────────────────────────────────────────

    # yfinance interval / period map
    _YF_MAP = {
        "intraday": ("15m",  "5d"),
        "scalping": ("5m",   "2d"),
        "swing":    ("1d",   "6mo"),
        "fno":      ("60m",  "1mo"),
    }

    def _fetch_candles(
        self, symbol: str, exchange: str, strategy: STRATEGY
    ) -> pd.DataFrame | None:
        """Fetch candles from yfinance — no Kite dependency."""
        yf_interval, period = self._YF_MAP.get(strategy, ("15m", "5d"))
        df = yf_client.historical(symbol, exchange, yf_interval, period)
        if df.empty:
            return None
        # signal_engine expects "datetime" column
        df = df.rename(columns={"date": "datetime"})
        return df.sort_values("datetime").reset_index(drop=True)

    # ── Technical indicators ──────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame, strategy: STRATEGY) -> list[dict]:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        results: list[dict] = []

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        rsi_sig = "BUY" if rsi < 40 else "SELL" if rsi > 65 else "Neutral"
        results.append({"name": "RSI (14)", "value": f"{rsi:.1f}", "signal": rsi_sig})

        # MACD
        macd_obj = ta.trend.MACD(close)
        macd_line = macd_obj.macd().iloc[-1]
        macd_sig_line = macd_obj.macd_signal().iloc[-1]
        macd_cross = "BUY" if macd_line > macd_sig_line else "SELL"
        results.append({"name": "MACD", "value": f"{macd_line:.2f} / {macd_sig_line:.2f}", "signal": macd_cross})

        # EMA 20 & 50
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
        ltp   = close.iloc[-1]
        ema_sig = "BUY" if ltp > ema20 > ema50 else "SELL" if ltp < ema20 < ema50 else "Neutral"
        results.append({"name": "EMA 20/50", "value": f"{ema20:.0f} / {ema50:.0f}", "signal": ema_sig})

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_low  = bb.bollinger_lband().iloc[-1]
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_sig  = "BUY" if ltp < bb_low else "SELL" if ltp > bb_high else "Neutral"
        results.append({"name": "Bollinger Bands", "value": f"{bb_low:.0f}–{bb_high:.0f}", "signal": bb_sig})

        # Volume ratio
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg else 1.0
        vol_sig = "BUY" if vol_ratio > 1.5 else "Neutral"
        results.append({"name": "Volume", "value": f"{vol_ratio:.1f}x avg", "signal": vol_sig})

        # VWAP (intraday only — meaningful on intraday / 5-min candles)
        if strategy in ("intraday", "scalping"):
            vwap = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price().iloc[-1]
            vwap_sig = "BUY" if ltp > vwap else "SELL"
            results.append({"name": "VWAP", "value": f"{vwap:.0f}", "signal": vwap_sig})

        # ADX (trend strength)
        adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        adx_sig = "BUY" if adx > 25 else "Neutral"
        results.append({"name": "ADX (14)", "value": f"{adx:.1f}", "signal": adx_sig})

        return results

    # ── Claude calls ──────────────────────────────────────────────────────────

    def _claude_with_indicators(
        self, symbol: str, strategy: str,
        indicators: list[dict], df: pd.DataFrame,
    ) -> dict:
        ltp   = df["close"].iloc[-1]
        high  = df["high"].iloc[-5:].max()
        low   = df["low"].iloc[-5:].min()

        user_msg = (
            f"Stock: {symbol} | Strategy: {strategy} | LTP: ₹{ltp:.2f} "
            f"| Recent High: ₹{high:.2f} | Recent Low: ₹{low:.2f}\n\n"
            f"Pre-computed indicators:\n{json.dumps(indicators, indent=2)}\n\n"
            "Generate a precise trade signal with entry, SL, and targets based on these levels."
        )
        return self._call_claude(user_msg)

    def _claude_raw(self, symbol: str, strategy: str) -> dict:
        user_msg = (
            f"Analyze {symbol} listed on NSE for a {strategy} trade today. "
            "Use realistic approximate current price levels for this Indian stock. "
            "Return a complete trade signal with entry price, stop-loss, and two targets."
        )
        return self._call_claude(user_msg)

    def _call_claude(self, user_msg: str) -> dict:
        message = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)


signal_engine = SignalEngine()