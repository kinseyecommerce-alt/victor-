"""
execution.py — how entries are sent and how well they filled.

* entry_order(side, tick) → a marketable LIMIT: at the ask (BUY) or bid (SELL) plus
  ENTRY_LIMIT_BUFFER_PCT. It fills like a market order in normal conditions but can never
  fill at a wild price (Zerodha market orders on a thin book can). Falls back to MARKET if
  USE_LIMIT_ENTRIES is off or there is no two-sided quote.
* fill_after(order_id, fallback) → the actual average fill, read back from the broker
  (paper: immediate; live: usually immediate for a marketable limit) so the trailing stop,
  P&L and the slippage ledger start from the real price, not the signal price.
* slippage_bps(signal, fill, side) → how much worse (positive) or better the fill was.
"""
from __future__ import annotations

from typing import Optional

from config import settings
from options_resolver import round_tick


def entry_order(side: str, tick, ltp: float) -> dict:
    """kwargs for kite_client.place_order: {"order_type": ..., "price": ...}."""
    if not settings.use_limit_entries:
        return {"order_type": "MARKET"}
    bid = float(getattr(tick, "bid", 0) or 0)
    ask = float(getattr(tick, "ask", 0) or 0)
    ref = (ask if ask > 0 else ltp) if side == "BUY" else (bid if bid > 0 else ltp)
    if ref <= 0:
        return {"order_type": "MARKET"}
    buf = settings.entry_limit_buffer_pct / 100
    px = ref * (1 + buf) if side == "BUY" else ref * (1 - buf)
    return {"order_type": "LIMIT", "price": round_tick(px)}


def fill_after(order_id: str, fallback: float) -> float:
    from kite_client import kite_client
    px = kite_client.fill_price(order_id)
    return float(px) if px and px > 0 else float(fallback)


def slippage_bps(signal_price: float, fill_price: float, side: str) -> Optional[float]:
    if not signal_price or not fill_price:
        return None
    adverse = (fill_price - signal_price) if side == "BUY" else (signal_price - fill_price)
    return round(adverse / signal_price * 10000, 1)
