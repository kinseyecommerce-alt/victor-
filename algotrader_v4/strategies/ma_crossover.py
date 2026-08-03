"""
strategies/ma_crossover.py
Strategy 2 — Moving Average Crossover + Trend Filter, daily bars.

Spec:
  Entry:  20-EMA crosses above 50-EMA → long; crosses below → exit / short.
          Trend filter: longs only when close > 200-day SMA
          (shorts, if enabled, only when close < 200-day SMA).
  Exit:   opposite crossover, or ATR trailing stop.
  Stop:   2.5–3 × ATR(14) trailing (default 2.5; ratchets, never loosens).
  Params: 20 / 50 / 200 — fixed, no optimization.
"""
from __future__ import annotations

from typing import Optional

from strategies.base import Action, OpenPosition, PositionalStrategy, Signal
from strategies.indicators import DailyBar, atr, ema_series, sma


class MACrossoverStrategy(PositionalStrategy):
    name = "ma_crossover"

    FAST, SLOW, TREND = 20, 50, 200
    ATR_PERIOD        = 14
    ATR_MULT          = 2.5      # spec range 2.5–3

    def __init__(self, allow_short: bool = False, atr_mult: float = ATR_MULT) -> None:
        self.allow_short = allow_short
        self.atr_mult    = atr_mult
        self._positions: dict[str, OpenPosition] = {}
        self._extreme:   dict[str, float] = {}   # high-water (low-water) mark since entry

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        return self._positions.get(symbol)

    def evaluate(self, symbol: str, bars: list[DailyBar]) -> Signal:
        if len(bars) < self.TREND + 1:
            return Signal(Action.HOLD, symbol, reason=f"need ≥{self.TREND + 1} bars")

        closes = [b.close for b in bars]
        fast = ema_series(closes, self.FAST)
        slow = ema_series(closes, self.SLOW)
        if len(fast) < 2 or len(slow) < 2:
            return Signal(Action.HOLD, symbol, reason="EMA warm-up")

        # Align tails: last two values of each series compare the same bars
        f_now, f_prev = fast[-1], fast[-2]
        s_now, s_prev = slow[-1], slow[-2]
        cross_up   = f_prev <= s_prev and f_now > s_now
        cross_down = f_prev >= s_prev and f_now < s_now

        close     = closes[-1]
        trend_sma = sma(closes, self.TREND)
        a         = atr(bars, self.ATR_PERIOD) or 0.0

        pos = self._positions.get(symbol)
        if pos:
            return self._manage_open(symbol, bars, pos, cross_up, cross_down, a)

        if cross_up and trend_sma is not None and close > trend_sma and a > 0:
            stop = close - self.atr_mult * a
            self._positions[symbol] = OpenPosition(
                symbol=symbol, side="LONG", system="MA_XOVER",
                entries=[close], stop=stop, entry_n=a, last_add_ref=close)
            self._extreme[symbol] = close
            return Signal(Action.ENTER_LONG, symbol, price=close, stop=stop,
                          n_atr=a, system="MA_XOVER",
                          reason=f"{self.FAST}EMA>{self.SLOW}EMA cross, "
                                 f"close>{self.TREND}SMA")

        if (cross_down and self.allow_short and trend_sma is not None
                and close < trend_sma and a > 0):
            stop = close + self.atr_mult * a
            self._positions[symbol] = OpenPosition(
                symbol=symbol, side="SHORT", system="MA_XOVER",
                entries=[close], stop=stop, entry_n=a, last_add_ref=close)
            self._extreme[symbol] = close
            return Signal(Action.ENTER_SHORT, symbol, price=close, stop=stop,
                          n_atr=a, system="MA_XOVER",
                          reason=f"{self.FAST}EMA<{self.SLOW}EMA cross, "
                                 f"close<{self.TREND}SMA")

        return Signal(Action.HOLD, symbol)

    def _manage_open(self, symbol: str, bars: list[DailyBar], pos: OpenPosition,
                     cross_up: bool, cross_down: bool, a: float) -> Signal:
        bar   = bars[-1]
        close = bar.close
        long  = pos.side == "LONG"

        # Ratchet the ATR trailing stop off the best price since entry
        if a > 0:
            ext = self._extreme.get(symbol, pos.entries[0])
            ext = max(ext, bar.high) if long else min(ext, bar.low)
            self._extreme[symbol] = ext
            new_stop = ext - self.atr_mult * a if long else ext + self.atr_mult * a
            pos.stop = max(pos.stop, new_stop) if long else min(pos.stop, new_stop)

        if (long and bar.low <= pos.stop) or (not long and bar.high >= pos.stop):
            return self._close(symbol, pos, close,
                               f"{self.atr_mult}×ATR trailing stop @ {pos.stop:.2f}")
        if (long and cross_down) or (not long and cross_up):
            return self._close(symbol, pos, close, "opposite EMA crossover")
        return Signal(Action.HOLD, symbol, system="MA_XOVER", stop=pos.stop)

    def _close(self, symbol: str, pos: OpenPosition,
               close: float, reason: str) -> Signal:
        del self._positions[symbol]
        self._extreme.pop(symbol, None)
        return Signal(Action.EXIT, symbol, price=close,
                      system="MA_XOVER", reason=reason)
