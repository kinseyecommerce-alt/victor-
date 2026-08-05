"""
strategies/position_sizing.py
ATR-based position sizing and Turtle-derived portfolio risk caps.

Fixed-fractional (percent-risk):
    Risk_per_trade = Equity × risk_fraction          # 0.005–0.01
    Stop_distance  = ATR_multiple × ATR(n)           # points
    Risk_per_lot   = Stop_distance × point_value × lot_size
    Lots           = floor(Risk_per_trade / Risk_per_lot)

Turtle unit:
    N         = ATR(20)
    Unit_lots = floor((Equity × 0.01) / (N × point_value × lot_size))

Worked example from the spec (must hold — see test_strategies.py):
    Equity ₹10,00,000, 1% risk, crude ATR(20)=₹150, stop 2N=₹300
    CRUDEOIL  (lot 100): risk/lot ₹30,000 → 0 lots (doesn't fit)
    CRUDEOILM (lot 10):  risk/lot  ₹3,000 → 3 lots

Portfolio caps (Turtle-derived):
    single market ≤ 4 units | closely-correlated cluster ≤ 6 units per
    direction | loosely-correlated direction ≤ 10 units | total open
    risk (portfolio heat) ≤ 6% of equity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from strategies.contracts import ContractSpec


DEFAULT_RISK_FRACTION = 0.01     # 1% (spec range 0.5–1%)
MAX_UNITS_PER_MARKET  = 4
MAX_UNITS_CLUSTER     = 6        # closely-correlated, same direction
MAX_UNITS_DIRECTION   = 10       # loosely-correlated, same direction
MAX_PORTFOLIO_HEAT    = 0.06     # sum of open risks ≤ 6% of equity


def fixed_fractional_lots(equity: float, atr_value: float, contract: ContractSpec,
                          risk_fraction: float = DEFAULT_RISK_FRACTION,
                          atr_multiple: float = 2.0) -> int:
    """Lots such that a stop-out loses ≈ equity × risk_fraction."""
    if equity <= 0 or atr_value <= 0:
        return 0
    risk_per_trade = equity * risk_fraction
    stop_distance  = atr_multiple * atr_value
    risk_per_lot   = stop_distance * contract.rupees_per_point
    if risk_per_lot <= 0:
        return 0
    return math.floor(risk_per_trade / risk_per_lot)


def turtle_unit_lots(equity: float, n_atr: float, contract: ContractSpec,
                     risk_fraction: float = DEFAULT_RISK_FRACTION) -> int:
    """1 Unit = (Equity × 1%) / (N × point value × lot size)."""
    if equity <= 0 or n_atr <= 0:
        return 0
    denom = n_atr * contract.rupees_per_point
    return math.floor((equity * risk_fraction) / denom) if denom > 0 else 0


def vol_target_lots(equity: float, price: float, daily_vol_annualized: float,
                    contract: ContractSpec,
                    target_vol_per_position: float = 0.02) -> int:
    """TSMOM vol-targeted sizing: position ∝ 1 / ex-ante volatility.
    Lots such that the position's annualized ₹ volatility ≈
    equity × target_vol_per_position (equal risk contribution)."""
    if min(equity, price, daily_vol_annualized) <= 0:
        return 0
    rupee_vol_per_lot = daily_vol_annualized * price * contract.rupees_per_point
    if rupee_vol_per_lot <= 0:
        return 0
    return math.floor(equity * target_vol_per_position / rupee_vol_per_lot)


# ── Portfolio-level caps ───────────────────────────────────────────────────

@dataclass
class OpenRisk:
    """One open unit-block: what the portfolio currently has at risk."""
    symbol:    str
    cluster:   str
    direction: str      # "LONG" | "SHORT"
    units:     int      # Turtle units held
    open_risk: float    # ₹ lost if the current stop is hit
    owner:     str = ""  # which strategy owns this block ("" = unattributed)


@dataclass
class PortfolioRiskBook:
    """Tracks open units/risk and answers 'may I add one more unit?'.

    All three strategies share one book so overlapping signals on the
    same/correlated instruments scale down automatically, per the spec:
    a single instrument traded by all 3 strategies in the same direction
    is 3× effective risk unless capped here.
    """
    equity: float
    positions: list[OpenRisk] = field(default_factory=list)

    max_units_market:    int   = MAX_UNITS_PER_MARKET
    max_units_cluster:   int   = MAX_UNITS_CLUSTER
    max_units_direction: int   = MAX_UNITS_DIRECTION
    max_heat:            float = MAX_PORTFOLIO_HEAT

    def units_in(self, symbol: str) -> int:
        return sum(p.units for p in self.positions if p.symbol == symbol)

    def units_in_cluster(self, cluster: str, direction: str) -> int:
        return sum(p.units for p in self.positions
                   if p.cluster == cluster and p.direction == direction)

    def units_in_direction(self, direction: str) -> int:
        return sum(p.units for p in self.positions if p.direction == direction)

    def heat(self) -> float:
        """Portfolio heat: sum of open risks as a fraction of equity."""
        if self.equity <= 0:
            return 0.0
        return sum(p.open_risk for p in self.positions) / self.equity

    def can_add(self, contract: ContractSpec, direction: str,
                unit_risk: float, units: int = 1) -> tuple[bool, str]:
        """Check every cap before adding `units` with ₹`unit_risk` each."""
        if self.units_in(contract.symbol) + units > self.max_units_market:
            return False, (f"market cap: {contract.symbol} would exceed "
                           f"{self.max_units_market} units")
        if self.units_in_cluster(contract.cluster, direction) + units \
                > self.max_units_cluster:
            return False, (f"cluster cap: {contract.cluster}/{direction} would "
                           f"exceed {self.max_units_cluster} units")
        if self.units_in_direction(direction) + units > self.max_units_direction:
            return False, (f"direction cap: {direction} would exceed "
                           f"{self.max_units_direction} units")
        if self.equity > 0 and \
                self.heat() + (unit_risk * units) / self.equity > self.max_heat:
            return False, (f"portfolio heat would exceed "
                           f"{self.max_heat:.0%} of equity")
        return True, "OK"

    def add(self, contract: ContractSpec, direction: str,
            unit_risk: float, units: int = 1, owner: str = "") -> None:
        self.positions.append(OpenRisk(
            symbol=contract.symbol, cluster=contract.cluster,
            direction=direction, units=units, open_risk=unit_risk * units,
            owner=owner))

    def remove(self, symbol: str, owner: str | None = None) -> None:
        """Remove open-risk blocks for a symbol; restrict to one strategy's
        blocks by passing owner (None removes across all strategies)."""
        self.positions = [
            p for p in self.positions
            if not (p.symbol == symbol and (owner is None or p.owner == owner))
        ]
