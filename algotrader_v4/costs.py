"""
costs.py — estimated Indian transaction costs per round trip (Zerodha, NSE, 2026 rates).

Every closed trade is charged these so the scoreboard shows what you keep, not gross P&L.
They are estimates: Zerodha's contract note is the truth. Rates (per side unless noted):

                      brokerage           STT                 exchange     stamp (buy)
  equity intraday     ₹20 or 0.03%        0.025% sell only    0.00297%     0.003%
  equity delivery     ₹0                  0.1% both sides     0.00297%     0.015%
  index option        ₹20 flat            0.1% on sell premium 0.03503%    0.003%
  index future        ₹20 or 0.03%        0.02% sell only     0.00173%     0.002%
GST 18% on brokerage + exchange charges; SEBI ₹10/crore; ₹15.93 IPFT etc. ignored.
"""
from __future__ import annotations

_RATES = {
    #                 brokerage      stt_buy  stt_sell  exch      stamp_buy
    "equity_mis": {"bro": ("pct", 0.0003, 20.0), "stt_buy": 0.0,     "stt_sell": 0.00025, "exch": 0.0000297, "stamp": 0.00003},
    "equity_cnc": {"bro": ("flat", 0.0, 0.0),    "stt_buy": 0.001,   "stt_sell": 0.001,   "exch": 0.0000297, "stamp": 0.00015},
    "option":     {"bro": ("flat", 0.0, 20.0),   "stt_buy": 0.0,     "stt_sell": 0.001,   "exch": 0.0003503, "stamp": 0.00003},
    "future":     {"bro": ("pct", 0.0003, 20.0), "stt_buy": 0.0,     "stt_sell": 0.0002,  "exch": 0.0000173, "stamp": 0.00002},
}
GST = 0.18
SEBI = 10 / 1e7


def instrument_class(exchange: str, product: str, symbol: str = "") -> str:
    if exchange in ("NFO", "BFO"):
        return "future" if symbol.upper().endswith("FUT") else "option"
    return "equity_cnc" if product == "CNC" else "equity_mis"


def round_trip_cost(buy_value: float, sell_value: float, kind: str) -> float:
    """₹ cost of buying `buy_value` and selling `sell_value` worth of the instrument."""
    r = _RATES.get(kind, _RATES["equity_mis"])
    mode, pct, flat = r["bro"]
    bro = 0.0
    for v in (buy_value, sell_value):
        if v <= 0:
            continue
        bro += flat if mode == "flat" else min(v * pct, flat)
    stt  = buy_value * r["stt_buy"] + sell_value * r["stt_sell"]
    exch = (buy_value + sell_value) * r["exch"]
    sebi = (buy_value + sell_value) * SEBI
    stamp = buy_value * r["stamp"]
    return round(bro + stt + exch + sebi + stamp + GST * (bro + exch + sebi), 2)


def trade_cost(entry_price: float, exit_price: float, quantity: int, side: str,
               exchange: str = "NSE", product: str = "MIS", symbol: str = "") -> float:
    """Cost of one closed trade. side = the entry side (BUY = long, SELL = short)."""
    kind = instrument_class(exchange, product, symbol)
    ev, xv = entry_price * quantity, (exit_price or entry_price) * quantity
    buy_v, sell_v = (ev, xv) if side == "BUY" else (xv, ev)
    return round_trip_cost(buy_v, sell_v, kind)
