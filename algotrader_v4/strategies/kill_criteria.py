"""
strategies/kill_criteria.py
Pre-committed numeric kill criteria for the positional strategies.
The thresholds are written down BEFORE going live and only ever compared
mechanically — no discretionary overrides.

Spec:
  Min evaluation window   6 months or 30 trades (whichever is LATER) —
                          no strategy may be killed before it completes.
  Max drawdown (live)     live DD > backtest max DD × 1.5 → AUTO-HALT
                          (this one triggers even inside the min window —
                          it is a safety stop, not an evaluation).
  Expectancy              rolling 30-trade expectancy < 0 → REVIEW.
  Live vs backtest Sharpe live Sharpe < 50% of backtest Sharpe
                          (after min window) → KILL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Verdict(str, Enum):
    OK        = "OK"
    AUTO_HALT = "AUTO_HALT"   # stop trading immediately (drawdown breach)
    REVIEW    = "REVIEW"      # flag for the monthly review day
    KILL      = "KILL"        # retire the strategy


MIN_WINDOW_MONTHS   = 6
MIN_WINDOW_TRADES   = 30
DD_MULTIPLE         = 1.5
SHARPE_FLOOR_RATIO  = 0.5
EXPECTANCY_WINDOW   = 30


@dataclass
class KillCriteria:
    """One instance per strategy; feed it live results as they happen."""
    strategy:           str
    backtest_max_dd:    float           # fraction, e.g. 0.12 = 12%
    backtest_sharpe:    float
    live_start:         date
    trade_pnls:         list[float] = field(default_factory=list)

    dd_multiple:        float = DD_MULTIPLE
    sharpe_floor_ratio: float = SHARPE_FLOOR_RATIO

    def record_trade(self, pnl: float) -> None:
        self.trade_pnls.append(pnl)

    # ── Window guard ───────────────────────────────────────────────────────

    def min_window_complete(self, today: date) -> bool:
        """6 months AND 30 trades must both have elapsed (whichever is later)."""
        months = (today.year - self.live_start.year) * 12 \
                 + (today.month - self.live_start.month)
        if months == MIN_WINDOW_MONTHS and today.day < self.live_start.day:
            months -= 1
        return months >= MIN_WINDOW_MONTHS and \
            len(self.trade_pnls) >= MIN_WINDOW_TRADES

    # ── Metrics ────────────────────────────────────────────────────────────

    def rolling_expectancy(self, window: int = EXPECTANCY_WINDOW) -> float | None:
        if len(self.trade_pnls) < window:
            return None
        recent = self.trade_pnls[-window:]
        return sum(recent) / window

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, today: date, live_drawdown: float,
                 live_sharpe: float | None = None) -> tuple[Verdict, str]:
        """Run all checks; the most severe verdict wins.
        live_drawdown is the current live peak-to-trough fraction."""
        # Safety stop: applies even before the min window completes
        dd_limit = self.backtest_max_dd * self.dd_multiple
        if live_drawdown > dd_limit:
            return Verdict.AUTO_HALT, (
                f"live DD {live_drawdown:.1%} > backtest max DD "
                f"{self.backtest_max_dd:.1%} × {self.dd_multiple} = {dd_limit:.1%}")

        if not self.min_window_complete(today):
            return Verdict.OK, (
                f"min evaluation window not complete "
                f"({len(self.trade_pnls)}/{MIN_WINDOW_TRADES} trades, "
                f"{MIN_WINDOW_MONTHS} months from {self.live_start}) — no kill allowed")

        if live_sharpe is not None and self.backtest_sharpe > 0 and \
                live_sharpe < self.backtest_sharpe * self.sharpe_floor_ratio:
            return Verdict.KILL, (
                f"live Sharpe {live_sharpe:.2f} < "
                f"{self.sharpe_floor_ratio:.0%} of backtest "
                f"({self.backtest_sharpe:.2f})")

        exp = self.rolling_expectancy()
        if exp is not None and exp < 0:
            return Verdict.REVIEW, (
                f"rolling {EXPECTANCY_WINDOW}-trade expectancy "
                f"₹{exp:,.0f} < 0")

        return Verdict.OK, "all kill criteria clear"
