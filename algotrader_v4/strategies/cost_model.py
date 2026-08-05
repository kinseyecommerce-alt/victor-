"""
strategies/cost_model.py
Realistic MCX / NSE futures transaction-cost model for backtesting and
live expectancy checks.

Rates per the spec (Zerodha, post-April-2026 STT — verify against the
Zerodha brokerage calculator monthly; budget changes are frequent):

MCX commodity futures:
  brokerage      min(₹20, 0.03%) per executed order
  CTT            0.01% sell side (non-agri; agri exempt)
  exchange txn   0.0021%
  SEBI fee       ₹10 per crore
  GST            18% × (brokerage + exchange txn + SEBI)
  stamp duty     0.002% buy side

NSE equity futures:
  brokerage      min(₹20, 0.03%) per executed order
  STT            0.05% sell side (raised from 0.02% effective Apr 1, 2026)
  exchange txn   0.00173%
  SEBI fee       ₹10 per crore
  GST            18% × (brokerage + exchange txn + SEBI)
  stamp duty     0.002% buy side

Slippage: assume ≥1 tick on market orders; 2–3 ticks (or 0.05–0.1%) for
illiquid / near-expiry contracts. Breakout entries can gap — model
conservatively. Rollover = spread cost + a full extra round trip.
"""
from __future__ import annotations

from dataclasses import dataclass

from strategies.contracts import ContractSpec

GST_RATE       = 0.18
SEBI_PER_CRORE = 10.0
BROKERAGE_FLAT = 20.0
BROKERAGE_PCT  = 0.0003     # 0.03%

MCX_CTT_SELL_PCT   = 0.0001    # 0.01% (non-agri)
MCX_TXN_PCT        = 0.000021  # 0.0021%
NSE_STT_SELL_PCT   = 0.0005    # 0.05% futures, post Apr-2026
NSE_TXN_PCT        = 0.0000173 # 0.00173%
STAMP_BUY_PCT      = 0.00002   # 0.002%


@dataclass
class CostBreakdown:
    brokerage: float = 0.0
    stt_ctt:   float = 0.0
    exchange:  float = 0.0
    sebi:      float = 0.0
    gst:       float = 0.0
    stamp:     float = 0.0
    slippage:  float = 0.0

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt_ctt + self.exchange
                + self.sebi + self.gst + self.stamp + self.slippage)


def order_cost(price: float, lots: int, contract: ContractSpec,
               side: str, slippage_ticks: int = 1) -> CostBreakdown:
    """Cost of one executed futures order (one side)."""
    if price <= 0 or lots <= 0:
        return CostBreakdown()
    turnover = price * contract.lot_size * lots * contract.point_value
    c = CostBreakdown()
    c.brokerage = min(BROKERAGE_FLAT, turnover * BROKERAGE_PCT)
    if contract.exchange == "MCX":
        c.exchange = turnover * MCX_TXN_PCT
        if side == "SELL":
            c.stt_ctt = turnover * MCX_CTT_SELL_PCT
    else:   # NFO
        c.exchange = turnover * NSE_TXN_PCT
        if side == "SELL":
            c.stt_ctt = turnover * NSE_STT_SELL_PCT
    c.sebi  = turnover / 1e7 * SEBI_PER_CRORE
    c.gst   = GST_RATE * (c.brokerage + c.exchange + c.sebi)
    if side == "BUY":
        c.stamp = turnover * STAMP_BUY_PCT
    c.slippage = slippage_ticks * contract.tick_size * contract.lot_size \
                 * lots * contract.point_value
    return c


def round_trip_cost(entry_price: float, exit_price: float, lots: int,
                    contract: ContractSpec, long: bool = True,
                    slippage_ticks: int = 1) -> float:
    """Total ₹ cost of entering and exiting one position."""
    buy_px, sell_px = (entry_price, exit_price) if long else (exit_price, entry_price)
    return (order_cost(buy_px, lots, contract, "BUY", slippage_ticks).total
            + order_cost(sell_px, lots, contract, "SELL", slippage_ticks).total)


def rollover_cost(price: float, lots: int, contract: ContractSpec,
                  calendar_spread: float = 0.0, long: bool = True,
                  slippage_ticks: int = 1) -> float:
    """Monthly rollover: square-off near month + re-enter next month
    (a full round trip) plus the near/next calendar spread paid."""
    trip = round_trip_cost(price, price, lots, contract, long, slippage_ticks)
    spread = abs(calendar_spread) * contract.lot_size * lots * contract.point_value
    return trip + spread
