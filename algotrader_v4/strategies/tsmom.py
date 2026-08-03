"""
strategies/tsmom.py
Strategy 3 — Time-Series Momentum (TSMOM), daily bars.
Moskowitz, Ooi & Pedersen (2012, JFE); Hurst, Ooi & Pedersen (2017, JPM).

Spec:
  Entry:     12-month (252-day) return positive → long; negative → short
             (or flat when long_only=True, recommended for a low-leverage
             profile).
  Rebalance: monthly — signals fire only on the first evaluation of a new
             calendar month (align the caller's run with the pre-set
             review day).
  Sizing:    volatility-targeted, position ∝ 1 / ex-ante volatility, so
             every position contributes equal risk. The Signal carries
             `weight`; convert to lots with position_sizing.vol_target_lots.
  Exit:      signal flip (12-month return changes sign), checked at
             rebalance.

Note: the paper uses excess returns (over the risk-free rate); price
return is used here as the standard practitioner approximation for
futures, where carry is embedded in the futures curve.
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
from strategies.indicators import DailyBar, annualized_vol, atr


class TSMOMStrategy(PositionalStrategy):
    name = "tsmom"

    LOOKBACK     = 252    # ~12 months of trading days
    VOL_LOOKBACK = 60     # ex-ante volatility estimation window

    def __init__(self, long_only: bool = True) -> None:
        self.long_only = long_only
        self._positions:  dict[str, OpenPosition] = {}
        self._last_rebal: dict[str, tuple[int, int]] = {}   # symbol → (year, month)

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        return self._positions.get(symbol)

    def to_state(self) -> dict:
        return {
            "positions":  _positions_to_dict(self._positions),
            "last_rebal": {sym: list(v) for sym, v in self._last_rebal.items()},
        }

    def load_state(self, state: dict) -> None:
        self._positions  = _positions_from_dict(state.get("positions"))
        self._last_rebal = {sym: tuple(v) for sym, v in
                            (state.get("last_rebal") or {}).items()}

    def evaluate(self, symbol: str, bars: list[DailyBar]) -> Signal:
        if len(bars) < self.LOOKBACK + 1:
            return Signal(Action.HOLD, symbol,
                          reason=f"need ≥{self.LOOKBACK + 1} bars")

        bar = bars[-1]
        month_key = (bar.date.year, bar.date.month)
        if self._last_rebal.get(symbol) == month_key:
            return Signal(Action.HOLD, symbol, system="TSMOM",
                          reason="already rebalanced this month")
        self._last_rebal[symbol] = month_key

        closes = [b.close for b in bars]
        past   = closes[-(self.LOOKBACK + 1)]
        if past <= 0:
            return Signal(Action.HOLD, symbol, reason="bad history")
        mom = closes[-1] / past - 1.0

        vol = annualized_vol(closes, self.VOL_LOOKBACK)
        weight = (1.0 / vol) if vol and vol > 0 else None
        a = atr(bars, 20)

        want = "LONG" if mom > 0 else ("FLAT" if self.long_only else "SHORT")
        pos  = self._positions.get(symbol)

        if pos and pos.side != want:
            del self._positions[symbol]
            return Signal(Action.EXIT, symbol, price=bar.close, system="TSMOM",
                          reason=f"12-mo momentum flipped ({mom:+.1%})")

        if not pos and want in ("LONG", "SHORT"):
            self._positions[symbol] = OpenPosition(
                symbol=symbol, side=want, system="TSMOM",
                entries=[bar.close], entry_n=a or 0.0, last_add_ref=bar.close)
            action = Action.ENTER_LONG if want == "LONG" else Action.ENTER_SHORT
            return Signal(action, symbol, price=bar.close, n_atr=a,
                          system="TSMOM", weight=weight,
                          reason=f"12-mo return {mom:+.1%}, "
                                 f"vol {vol:.1%}" if vol else
                                 f"12-mo return {mom:+.1%}")

        return Signal(Action.HOLD, symbol, system="TSMOM", weight=weight,
                      reason=f"hold {pos.side if pos else 'flat'} "
                             f"(12-mo {mom:+.1%})")
