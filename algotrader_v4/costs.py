"""
costs.py — estimated Indian transaction costs per round trip (Zerodha, NSE, 2026 rates).

Every closed trade is charged these so the scoreboard shows what you keep, not gross P&L.
They are estimates: Zerodha's contract note is the truth. Rates (per side unless noted):

                      brokerage           STT                       exchange (NSE)  stamp (buy)
  equity intraday     ₹20 or 0.03%        0.025% sell only          0.00307%        0.003%
  equity delivery     ₹0                  0.1% both sides           0.00307%        0.015%   + DP ₹15.34 per sell
  index/stock option  ₹20 flat            0.15% of premium, sell    0.03553%        0.003%
  index/stock future  ₹20 or 0.03%        0.05% sell only           0.00183%        0.002%
GST 18% on brokerage + exchange + SEBI charges; SEBI ₹10/crore.

Sources (checked 2026-11-11):
  * zerodha.com/charges — every rate in the table above, including the post-Budget-2026 F&O STT (futures 0.05% on
    the sell side; options 0.15% of premium on the sell side and 0.15% of intrinsic value on bought options that
    are exercised) and the NSE transaction charges (equity 0.00307%, futures 0.00183%, options 0.03553% of premium).
  * DP charge, same page: ₹15.34 per scrip per day on a delivery sale = ₹3.50 CDSL + ₹9.50 Zerodha + ₹2.34 GST.
    The ₹15.34 already INCLUDES the 18% GST (13.00 × 1.18 = 15.34), so no further GST is added on top. It is charged
    once per scrip per day whatever the quantity; `round_trip_cost` charges it once per delivery round trip that
    sells (two delivery sells of one scrip on one day would really pay it once — a small over-charge).
  * NSE IPFT (₹0.01 per crore of turnover per Zerodha's page) is too small to model and is ignored.
The previous version (before COST_MODEL_VERSION "2026-11") used the pre-Budget-2026 F&O STT (futures 0.02%,
options 0.1%), the 2024 exchange charges (0.00297% / 0.00173% / 0.03503%) and no DP charge.
"""
from __future__ import annotations

# Bump whenever a rate or rule below changes: research ledgers key results on it so that a re-costed run is
# recognised as a different cost model (see stratlab/protocol.py ledger_key).
COST_MODEL_VERSION = "2026-11"

_RATES = {
    #                 brokerage      stt_buy  stt_sell  exch      stamp_buy  dp (₹ per sell, GST incl.)
    "equity_mis": {"bro": ("pct", 0.0003, 20.0), "stt_buy": 0.0,   "stt_sell": 0.00025, "exch": 0.0000307, "stamp": 0.00003, "dp": 0.0},
    "equity_cnc": {"bro": ("flat", 0.0, 0.0),    "stt_buy": 0.001, "stt_sell": 0.001,   "exch": 0.0000307, "stamp": 0.00015, "dp": 15.34},
    "option":     {"bro": ("flat", 0.0, 20.0),   "stt_buy": 0.0,   "stt_sell": 0.0015,  "exch": 0.0003553, "stamp": 0.00003, "dp": 0.0},
    "future":     {"bro": ("pct", 0.0003, 20.0), "stt_buy": 0.0,   "stt_sell": 0.0005,  "exch": 0.0000183, "stamp": 0.00002, "dp": 0.0},
}
GST = 0.18
SEBI = 10 / 1e7
DP_CHARGE = 15.34               # ₹ per scrip per sell day on delivery, GST included (zerodha.com/charges)
STT_OPTION_EXERCISE = 0.0015    # 0.15 % of intrinsic value, on bought options that are exercised


def instrument_class(exchange: str, product: str, symbol: str = "") -> str:
    if exchange in ("NFO", "BFO"):
        return "future" if symbol.upper().endswith("FUT") else "option"
    return "equity_cnc" if product == "CNC" else "equity_mis"


def round_trip_cost(buy_value: float, sell_value: float, kind: str) -> float:
    """₹ cost of buying `buy_value` and selling `sell_value` worth of the instrument (for options: premium value)."""
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
    dp = r["dp"] if sell_value > 0 else 0.0
    return round(bro + stt + exch + sebi + stamp + GST * (bro + exch + sebi) + dp, 2)


def exercise_stt(intrinsic_value: float) -> float:
    """STT on a bought option that expires in the money and is exercised (0.15 % of intrinsic value, 2026)."""
    return round(max(0.0, intrinsic_value) * STT_OPTION_EXERCISE, 2)


def trade_cost(entry_price: float, exit_price: float, quantity: int, side: str,
               exchange: str = "NSE", product: str = "MIS", symbol: str = "") -> float:
    """Cost of one closed trade. side = the entry side (BUY = long, SELL = short)."""
    kind = instrument_class(exchange, product, symbol)
    ev, xv = entry_price * quantity, (exit_price or entry_price) * quantity
    buy_v, sell_v = (ev, xv) if side == "BUY" else (xv, ev)
    return round_trip_cost(buy_v, sell_v, kind)
