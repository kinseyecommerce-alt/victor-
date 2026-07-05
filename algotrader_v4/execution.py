"""
execution.py — Entry order-type & price resolution (slippage-aware).

Pure market orders pay the full spread and give no price control. This resolves
a better entry per `settings.entry_order_type`:

  • MARKET            — cross at market (max fill certainty, max slippage)
  • LIMIT             — passive: rest at bid (BUY) / ask (SELL); best price, may not fill
  • MARKETABLE_LIMIT  — cross the spread by N ticks (default): near-certain fill
                        with a hard price cap, so slippage can never exceed N ticks.

All prices are snapped to the contract's tick grid.
"""
from __future__ import annotations

import mcx_universe


def _cfg(name: str, default):
    try:
        from config import settings
        return getattr(settings, name, default)
    except Exception:
        return default


def resolve_entry(symbol: str, action: str, ltp: float,
                  bid: float = 0.0, ask: float = 0.0) -> tuple[str, float]:
    """Return (order_type, price) for an entry. price=0.0 for MARKET orders."""
    mode = str(_cfg("entry_order_type", "MARKETABLE_LIMIT")).upper()
    if mode == "MARKET":
        return "MARKET", 0.0

    c = mcx_universe.contract(symbol)
    tick = c.tick_size if c else 0.05
    bid = bid or (ltp - tick)
    ask = ask or (ltp + tick)
    cross = int(_cfg("entry_limit_cross_ticks", 2))

    if mode == "LIMIT":
        # passive — rest on the near side
        price = bid if action == "BUY" else ask
    else:  # MARKETABLE_LIMIT — cross the book by `cross` ticks, capping slippage
        price = (ask + cross * tick) if action == "BUY" else (bid - cross * tick)

    return "LIMIT", mcx_universe.round_to_tick(symbol, price)
