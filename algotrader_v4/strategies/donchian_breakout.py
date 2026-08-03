"""
strategies/donchian_breakout.py
Strategy 1 — Donchian Channel Breakout (Turtle-style S1/S2), daily bars.

Spec:
  Entry  S1: close breaks prior 20-day high → long / prior 20-day low → short.
             Skip the signal if the PREVIOUS S1 breakout was a winner
             (whipsaw filter); the 55-day S2 acts as the safety net.
  Entry  S2: close breaks prior 55-day high/low — every S2 signal is taken.
  Exit   S1: opposite 10-day channel breakout.  S2: opposite 20-day.
  Stop:      2N from entry (N = 20-day ATR). After a pyramid add, the stop
             for the whole position moves to (latest entry ± 2N).
  Pyramid:   add one unit every 0.5N favorable move, max 4 units.

Signals are generated at EOD on completed bars; execution is assumed
next-day at open. Entry prices recorded here use the signal-day close as
reference — the caller should reconcile with actual fills.
"""
from __future__ import annotations

from typing import Optional

from strategies.base import (
    Action,
    OpenPosition,
    PositionalStrategy,
    Signal,
    _positions_from_dict,
    _positions_to_dict,
)
from strategies.indicators import DailyBar, atr, donchian


class DonchianBreakoutStrategy(PositionalStrategy):
    name = "donchian"

    S1_ENTRY, S1_EXIT = 20, 10
    S2_ENTRY, S2_EXIT = 55, 20
    ATR_PERIOD        = 20
    STOP_N            = 2.0
    PYRAMID_STEP_N    = 0.5
    MAX_UNITS         = 4

    def __init__(self) -> None:
        self._positions: dict[str, OpenPosition] = {}
        # Whipsaw filter state: result of the last completed S1 breakout
        # (tracked theoretically, whether or not the trade was taken)
        self._last_s1_won: dict[str, bool] = {}
        self._s1_tracker:  dict[str, dict] = {}   # symbol → open theoretical S1 trade

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        return self._positions.get(symbol)

    def to_state(self) -> dict:
        return {
            "positions":   _positions_to_dict(self._positions),
            "last_s1_won": dict(self._last_s1_won),
            "s1_tracker":  dict(self._s1_tracker),
        }

    def load_state(self, state: dict) -> None:
        self._positions   = _positions_from_dict(state.get("positions"))
        self._last_s1_won = dict(state.get("last_s1_won") or {})
        self._s1_tracker  = dict(state.get("s1_tracker") or {})

    # ── Main EOD evaluation ────────────────────────────────────────────────

    def evaluate(self, symbol: str, bars: list[DailyBar]) -> Signal:
        min_bars = self.S2_ENTRY + 1
        if len(bars) < min_bars:
            return Signal(Action.HOLD, symbol, reason=f"need ≥{min_bars} bars")

        n = atr(bars, self.ATR_PERIOD)
        if not n or n <= 0:
            return Signal(Action.HOLD, symbol, reason="no ATR")

        self._track_theoretical_s1(symbol, bars)

        pos = self._positions.get(symbol)
        if pos:
            return self._manage_open(symbol, bars, pos, n)
        return self._check_entry(symbol, bars, n)

    # ── Entry ──────────────────────────────────────────────────────────────

    def _check_entry(self, symbol: str, bars: list[DailyBar], n: float) -> Signal:
        close = bars[-1].close
        ch20 = donchian(bars, self.S1_ENTRY)
        ch55 = donchian(bars, self.S2_ENTRY)
        if not ch20 or not ch55:
            return Signal(Action.HOLD, symbol, reason="channel warm-up")
        hi20, lo20 = ch20
        hi55, lo55 = ch55

        side = ""
        if close > hi20:
            side = "LONG"
        elif close < lo20:
            side = "SHORT"

        if side:
            skip_s1 = self._last_s1_won.get(symbol, False)
            s2_hit  = (side == "LONG" and close > hi55) or \
                      (side == "SHORT" and close < lo55)
            if not skip_s1:
                return self._open(symbol, side, "S1", close, n)
            if s2_hit:
                return self._open(symbol, side, "S2", close, n)
            return Signal(Action.HOLD, symbol, system="S1",
                          reason="S1 skipped (last S1 breakout won)")

        if close > hi55:
            return self._open(symbol, "LONG", "S2", close, n)
        if close < lo55:
            return self._open(symbol, "SHORT", "S2", close, n)
        return Signal(Action.HOLD, symbol)

    def _open(self, symbol: str, side: str, system: str,
              close: float, n: float) -> Signal:
        stop = close - self.STOP_N * n if side == "LONG" else close + self.STOP_N * n
        self._positions[symbol] = OpenPosition(
            symbol=symbol, side=side, system=system,
            entries=[close], stop=stop, entry_n=n, last_add_ref=close,
        )
        action = Action.ENTER_LONG if side == "LONG" else Action.ENTER_SHORT
        return Signal(action, symbol, price=close, stop=stop, n_atr=n,
                      system=system,
                      reason=f"{system} {self.S1_ENTRY if system == 'S1' else self.S2_ENTRY}"
                             f"-day breakout {side.lower()}")

    # ── Open-position management ───────────────────────────────────────────

    def _manage_open(self, symbol: str, bars: list[DailyBar],
                     pos: OpenPosition, n: float) -> Signal:
        bar   = bars[-1]
        close = bar.close
        exit_period = self.S1_EXIT if pos.system == "S1" else self.S2_EXIT
        ch = donchian(bars, exit_period)
        long = pos.side == "LONG"

        # 1. Protective stop hit intraday (2N from latest entry)
        if (long and bar.low <= pos.stop) or (not long and bar.high >= pos.stop):
            return self._close(symbol, pos, close, f"2N stop hit @ {pos.stop:.2f}")

        # 2. Opposite channel breakout exit
        if ch:
            hi, lo = ch
            if long and close < lo:
                return self._close(symbol, pos, close,
                                   f"{exit_period}-day low exit ({pos.system})")
            if not long and close > hi:
                return self._close(symbol, pos, close,
                                   f"{exit_period}-day high exit ({pos.system})")

        # 3. Pyramid: every 0.5N favorable move, up to 4 units
        if pos.units < self.MAX_UNITS:
            step = self.PYRAMID_STEP_N * pos.entry_n
            if (long and close >= pos.last_add_ref + step) or \
               (not long and close <= pos.last_add_ref - step):
                pos.entries.append(close)
                pos.last_add_ref = close
                pos.stop = close - self.STOP_N * n if long else close + self.STOP_N * n
                return Signal(Action.PYRAMID, symbol, price=close, stop=pos.stop,
                              n_atr=n, system=pos.system,
                              reason=f"pyramid unit {pos.units}/{self.MAX_UNITS} "
                                     f"(+0.5N move), stop → {pos.stop:.2f}")

        return Signal(Action.HOLD, symbol, system=pos.system)

    def _close(self, symbol: str, pos: OpenPosition,
               close: float, reason: str) -> Signal:
        del self._positions[symbol]
        return Signal(Action.EXIT, symbol, price=close,
                      system=pos.system, reason=reason)

    # ── Theoretical S1 tracker (whipsaw filter) ────────────────────────────
    # Every S1 breakout is tracked to completion (2N stop = loss, 10-day
    # opposite exit = win/loss by P&L) regardless of whether it was taken,
    # exactly like the original Turtle skip rule.

    def _track_theoretical_s1(self, symbol: str, bars: list[DailyBar]) -> None:
        bar = bars[-1]
        trade = self._s1_tracker.get(symbol)

        if trade:
            long = trade["side"] == "LONG"
            ch = donchian(bars, self.S1_EXIT)
            if (long and bar.low <= trade["stop"]) or \
               (not long and bar.high >= trade["stop"]):
                self._last_s1_won[symbol] = False
                del self._s1_tracker[symbol]
            elif ch:
                hi, lo = ch
                if long and bar.close < lo:
                    self._last_s1_won[symbol] = bar.close > trade["entry"]
                    del self._s1_tracker[symbol]
                elif not long and bar.close > hi:
                    self._last_s1_won[symbol] = bar.close < trade["entry"]
                    del self._s1_tracker[symbol]
            if symbol in self._s1_tracker:
                return   # still open — don't start a new one

        # Start tracking a fresh S1 breakout (taken or skipped)
        n = atr(bars, self.ATR_PERIOD)
        ch20 = donchian(bars, self.S1_ENTRY)
        if not n or not ch20:
            return
        hi20, lo20 = ch20
        if bar.close > hi20:
            self._s1_tracker[symbol] = {
                "side": "LONG", "entry": bar.close,
                "stop": bar.close - self.STOP_N * n,
            }
        elif bar.close < lo20:
            self._s1_tracker[symbol] = {
                "side": "SHORT", "entry": bar.close,
                "stop": bar.close + self.STOP_N * n,
            }
