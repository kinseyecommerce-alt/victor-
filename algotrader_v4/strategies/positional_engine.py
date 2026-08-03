"""
strategies/positional_engine.py
EOD orchestrator for the three positional trend-following strategies.

Pure decision logic: feed it daily bar histories after the session close
and it returns a list of OrderPlan for next-day-open execution. The
caller (scheduler / main) owns all I/O — placing NRML orders + GTT stops
via kite_client, persisting state, reconciliation and the kill-switch.

Capital is split equally across the three strategies (spec: 3 strategies
→ ~33% each) and a single shared PortfolioRiskBook enforces the Turtle
caps across ALL strategies, so overlapping signals on correlated
instruments scale down automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from strategies.base import Action, PositionalStrategy, Signal
from strategies.contracts import ContractSpec, get_contract
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.indicators import DailyBar, annualized_vol
from strategies.ma_crossover import MACrossoverStrategy
from strategies.position_sizing import (
    DEFAULT_RISK_FRACTION,
    PortfolioRiskBook,
    turtle_unit_lots,
    vol_target_lots,
)
from strategies.tsmom import TSMOMStrategy


@dataclass
class OrderPlan:
    """One intended order for next-day-open execution."""
    symbol:    str
    exchange:  str
    action:    str            # "BUY" | "SELL"
    lots:      int
    quantity:  int            # lots × lot_size (Kite wants units)
    product:   str = "NRML"
    stop:      float | None = None   # place as GTT after fill
    strategy:  str = ""
    system:    str = ""
    reason:    str = ""
    tag:       str = ""       # order tag for internal tracking

    @property
    def is_entry(self) -> bool:
        return self.tag.endswith("-entry") or self.tag.endswith("-add")


@dataclass
class PositionalEngine:
    equity:        float
    risk_fraction: float = DEFAULT_RISK_FRACTION
    strategies:    list[PositionalStrategy] = field(default_factory=lambda: [
        DonchianBreakoutStrategy(),
        MACrossoverStrategy(),
        TSMOMStrategy(long_only=True),
    ])

    def __post_init__(self) -> None:
        self.book = PortfolioRiskBook(equity=self.equity)
        # Equal capital allocation per strategy (spec: tala ~33%)
        self._alloc = self.equity / max(len(self.strategies), 1)
        # Lots held per "strategy:symbol" so exits know what to unwind
        self._exit_lots:  dict[str, int] = {}
        self._exit_sides: dict[str, str] = {}

    # ── EOD run ────────────────────────────────────────────────────────────

    def run_eod(self, bars_by_symbol: dict[str, list[DailyBar]]
                ) -> list[OrderPlan]:
        """Evaluate every strategy on every instrument; return orders to
        place at next open. Also updates the shared risk book."""
        plans: list[OrderPlan] = []
        for strat in self.strategies:
            for symbol, bars in bars_by_symbol.items():
                sig = strat.evaluate(symbol, bars)
                plan = self._to_plan(strat, symbol, bars, sig)
                if plan:
                    plans.append(plan)
        return plans

    # ── Signal → sized order ───────────────────────────────────────────────

    def _to_plan(self, strat: PositionalStrategy, symbol: str,
                 bars: list[DailyBar], sig: Signal) -> OrderPlan | None:
        if sig.action == Action.HOLD:
            return None
        contract = get_contract(symbol)
        key = f"{strat.name}:{symbol}"

        if sig.action == Action.EXIT:
            self.book.remove(symbol, owner=strat.name)
            lots = self._exit_lots.pop(key, 0)
            if lots <= 0:
                return None
            side = "SELL" if self._exit_sides.pop(key, "LONG") == "LONG" else "BUY"
            return OrderPlan(
                symbol=symbol, exchange=contract.exchange, action=side,
                lots=lots, quantity=lots * contract.lot_size,
                strategy=strat.name, system=sig.system, reason=sig.reason,
                tag=f"maran_{strat.name}-exit")

        # Entries / pyramids
        if sig.action == Action.PYRAMID:
            pos = strat.get_position(symbol)
            direction = pos.side if pos else "LONG"
        else:
            direction = "LONG" if sig.action == Action.ENTER_LONG else "SHORT"

        lots, unit_risk = self._size(strat, contract, bars, sig)
        if lots <= 0:
            return None

        ok, why = self.book.can_add(contract, direction, unit_risk)
        if not ok:
            # Undo the strategy's optimistic position state on a blocked entry
            if sig.action != Action.PYRAMID:
                self._rollback_position(strat, symbol)
            return None

        self.book.add(contract, direction, unit_risk, owner=strat.name)
        prev = self._exit_lots.get(key, 0)
        self._exit_lots[key]  = prev + lots
        self._exit_sides[key] = direction
        side = "BUY" if direction == "LONG" else "SELL"
        suffix = "add" if sig.action == Action.PYRAMID else "entry"
        return OrderPlan(
            symbol=symbol, exchange=contract.exchange, action=side,
            lots=lots, quantity=lots * contract.lot_size,
            stop=sig.stop, strategy=strat.name, system=sig.system,
            reason=sig.reason, tag=f"maran_{strat.name}-{suffix}")

    def _size(self, strat: PositionalStrategy, contract: ContractSpec,
              bars: list[DailyBar], sig: Signal) -> tuple[int, float]:
        """Return (lots, ₹ open-risk per unit-block) for this signal."""
        alloc = self._alloc
        if isinstance(strat, TSMOMStrategy):
            closes = [b.close for b in bars]
            vol = annualized_vol(closes, TSMOMStrategy.VOL_LOOKBACK)
            if not vol:
                return 0, 0.0
            lots = vol_target_lots(alloc, bars[-1].close, vol, contract)
            # Open risk proxy: one day of 2-sigma move against the position
            unit_risk = 2 * (vol / (252 ** 0.5)) * bars[-1].close \
                        * contract.rupees_per_point * lots
            return lots, unit_risk

        if not sig.n_atr or sig.n_atr <= 0:
            return 0, 0.0
        lots = turtle_unit_lots(alloc, sig.n_atr, contract, self.risk_fraction)
        # 2N stop → open risk = 2 × N × ₹/point × lots
        unit_risk = 2.0 * sig.n_atr * contract.rupees_per_point * lots
        return lots, unit_risk

    @staticmethod
    def _rollback_position(strat: PositionalStrategy, symbol: str) -> None:
        pos_store = getattr(strat, "_positions", None)
        if pos_store is not None:
            pos_store.pop(symbol, None)
