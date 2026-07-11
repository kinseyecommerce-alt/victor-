"""
cost_model.py — MCX transaction-cost & slippage model.

Realistic round-trip cost for a commodity-futures trade, so that both the
backtester and live sizing account for what the exchange and broker actually
take. A TA edge that looks positive gross is often negative net once brokerage,
exchange charges, GST, stamp duty and slippage are subtracted — this module
makes that explicit.

Figures follow the standard Indian discount-broker (Zerodha) MCX futures
schedule; they are configurable via `settings` and should be verified against
your contract note before relying on them.

Components (per side unless noted):
  • Brokerage     — flat ₹ per executed order (min of flat vs %-of-turnover)
  • Exchange txn  — % of turnover (MCX transaction charge)
  • GST           — 18% on (brokerage + exchange txn)
  • SEBI charges  — ₹ per crore of turnover
  • Stamp duty    — % of turnover, buy side only
  • Slippage      — N ticks × tick value × lots (both sides)
"""
from __future__ import annotations

from dataclasses import dataclass

import mcx_universe


# Defaults (Zerodha MCX futures, 2025 schedule) — override via settings.*
DEF_BROKERAGE_FLAT   = 20.0      # ₹ per order
DEF_BROKERAGE_PCT    = 0.0003    # 0.03% of turnover (min vs flat)
DEF_EXCHANGE_TXN_PCT = 0.000026  # MCX transaction charge ~0.0026% of turnover
DEF_GST_PCT          = 0.18      # on brokerage + exchange txn
DEF_SEBI_PER_CRORE   = 10.0      # ₹10 per ₹1 crore turnover
DEF_STAMP_PCT_BUY    = 0.00002   # 0.002% on buy turnover
DEF_SLIPPAGE_TICKS   = 1.0       # ticks of slippage assumed per side


def _cfg(name: str, default: float) -> float:
    try:
        from config import settings
        return float(getattr(settings, name, default))
    except Exception:
        return default


@dataclass
class CostBreakdown:
    brokerage:    float
    exchange_txn: float
    gst:          float
    sebi:         float
    stamp:        float
    slippage:     float

    @property
    def total(self) -> float:
        return round(self.brokerage + self.exchange_txn + self.gst
                     + self.sebi + self.stamp + self.slippage, 2)

    def as_dict(self) -> dict:
        d = {k: round(v, 2) for k, v in self.__dict__.items()}
        d["total"] = self.total
        return d


def _turnover(symbol: str, lots: int, price: float) -> float:
    return abs(price) * mcx_universe.lot_size(symbol) * max(lots, 1)


def slippage_cost(symbol: str, lots: int) -> float:
    """Assumed slippage per side = N ticks × tick_size × lot_size × lots (in ₹)."""
    c = mcx_universe.contract(symbol)
    if not c:
        return 0.0
    ticks = _cfg("slippage_ticks", DEF_SLIPPAGE_TICKS)
    return ticks * c.tick_size * c.lot_size * max(lots, 1)


def _one_side(symbol: str, lots: int, price: float, is_buy: bool) -> CostBreakdown:
    turnover = _turnover(symbol, lots, price)
    flat = _cfg("brokerage_flat", DEF_BROKERAGE_FLAT)
    pct  = _cfg("brokerage_pct",  DEF_BROKERAGE_PCT)
    brokerage    = min(flat, turnover * pct) if pct > 0 else flat
    exchange_txn = turnover * _cfg("exchange_txn_pct", DEF_EXCHANGE_TXN_PCT)
    gst          = (brokerage + exchange_txn) * _cfg("gst_pct", DEF_GST_PCT)
    sebi         = turnover / 1e7 * _cfg("sebi_per_crore", DEF_SEBI_PER_CRORE)
    stamp        = turnover * _cfg("stamp_pct_buy", DEF_STAMP_PCT_BUY) if is_buy else 0.0
    slip         = slippage_cost(symbol, lots)
    return CostBreakdown(brokerage, exchange_txn, gst, sebi, stamp, slip)


def round_trip_cost(symbol: str, lots: int, price: float) -> float:
    """Total ₹ cost to open AND close `lots` of `symbol` near `price` (both sides)."""
    return round(entry_cost(symbol, lots, price) + exit_cost(symbol, lots, price), 2)


def entry_cost(symbol: str, lots: int, price: float) -> float:
    return _one_side(symbol, lots, price, is_buy=True).total


def exit_cost(symbol: str, lots: int, price: float) -> float:
    return _one_side(symbol, lots, price, is_buy=False).total


def round_trip_breakdown(symbol: str, lots: int, price: float) -> dict:
    e = _one_side(symbol, lots, price, is_buy=True)
    x = _one_side(symbol, lots, price, is_buy=False)
    return {"entry": e.as_dict(), "exit": x.as_dict(),
            "round_trip_total": round(e.total + x.total, 2)}


def cost_per_unit(symbol: str, lots: int, price: float) -> float:
    """Round-trip cost expressed as ₹ per unit of the underlying (per lot_size×lots)."""
    qty = mcx_universe.lot_size(symbol) * max(lots, 1)
    return round(round_trip_cost(symbol, lots, price) / qty, 4) if qty else 0.0


def min_profitable_move(symbol: str, lots: int, price: float) -> float:
    """Price move (in ₹/unit) the trade must capture just to break even on costs."""
    return cost_per_unit(symbol, lots, price)


def covers_cost(symbol: str, lots: int, entry: float, target: float) -> bool:
    """True if the target move is larger than the round-trip cost per unit."""
    return abs(target - entry) > min_profitable_move(symbol, lots, entry)
