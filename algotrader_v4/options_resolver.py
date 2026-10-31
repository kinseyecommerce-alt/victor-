"""
options_resolver.py — find the real NFO option contract to trade.

Expiry days, weekly vs monthly series, strike steps and lot sizes are set by
NSE and change over time, so they are read from Kite's instrument master
(kite.instruments("NFO")) instead of being computed. Only the paper simulator
(no Kite session) falls back to synthetic contracts, named "<UNDERLYING>SIM…"
so they can never be mistaken for a real tradingsymbol.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

from config import settings

# NFO underlying name → NSE index tradingsymbol streamed for its spot price
INDEX_QUOTE: dict[str, str] = {
    "NIFTY":      "NIFTY 50",
    "BANKNIFTY":  "NIFTY BANK",
    "FINNIFTY":   "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MID SELECT",
}
_ALIASES = {
    "NIFTY": "NIFTY", "NIFTY50": "NIFTY", "NIFTY 50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY", "NIFTY BANK": "BANKNIFTY", "NIFTYBANK": "BANKNIFTY",
    "FINNIFTY": "FINNIFTY", "NIFTY FIN SERVICE": "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY", "NIFTY MID SELECT": "MIDCPNIFTY",
}
# Paper-simulator fallbacks only — live trading always uses the instrument master
_SIM_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25}
_SIM_LOT  = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 65, "MIDCPNIFTY": 120}

TICK = 0.05


def normalize_underlying(symbol: str) -> Optional[str]:
    return _ALIASES.get(symbol.strip().upper())


def round_tick(price: float, tick: float = TICK) -> float:
    return round(round(price / tick) * tick, 2)


@dataclass
class OptionContract:
    tradingsymbol: str
    underlying:    str
    expiry:        Optional[date]
    strike:        float
    option_type:   str        # CE / PE
    lot_size:      int
    live:          bool       # True = from the Kite instrument master


def _instruments() -> list[dict]:
    from kite_client import kite_client
    try:
        return kite_client.get_instruments("NFO")
    except Exception:
        return []


def resolve(underlying: str, spot: float, option_type: str, strike_offset: int,
            today: date, skip_expiry_today: bool = True) -> Optional[OptionContract]:
    """Nearest-expiry contract `strike_offset` strikes from ATM.
    Positive offset = out of the money (higher strike for CE, lower for PE)."""
    und = normalize_underlying(underlying)
    if not und or option_type not in ("CE", "PE") or spot <= 0:
        return None

    rows = [i for i in _instruments()
            if i.get("name") == und and i.get("instrument_type") == option_type
            and str(i.get("segment", "")).endswith("OPT")]
    if rows:
        expiries = sorted({r["expiry"] for r in rows if r.get("expiry") and r["expiry"] >= today
                           and not (skip_expiry_today and r["expiry"] == today)})
        if not expiries:
            return None
        exp = expiries[0]
        by_strike = {float(r["strike"]): r for r in rows if r["expiry"] == exp}
        strikes = sorted(by_strike)
        atm = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        idx = atm + (strike_offset if option_type == "CE" else -strike_offset)
        if not 0 <= idx < len(strikes):
            return None
        r = by_strike[strikes[idx]]
        return OptionContract(r["tradingsymbol"], und, exp, float(r["strike"]), option_type,
                              int(r.get("lot_size") or 0), live=True)

    if settings.uses_kite_market_data:
        return None              # real feed but no instrument master: never guess a live symbol
    step = _SIM_STEP[und]
    atm = round(spot / step) * step
    strike = atm + strike_offset * step * (1 if option_type == "CE" else -1)
    return OptionContract(f"{und}SIM{int(strike)}{option_type}", und, None, float(strike),
                          option_type, _SIM_LOT[und], live=False)


def estimate_premium(spot: float, strike: float, option_type: str, iv: float = 0.15,
                     days: float = 5.0) -> float:
    """Rough premium for the paper simulator only (ATM ≈ 0.4·S·σ·√T, decaying with distance)."""
    t = max(days, 0.5) / 365.0
    atm = 0.4 * spot * iv * math.sqrt(t)
    moneyness = (spot - strike) if option_type == "CE" else (strike - spot)
    intrinsic = max(moneyness, 0.0)
    time_value = atm * math.exp(-abs(moneyness) / max(spot * iv * math.sqrt(t), 1e-6))
    return round_tick(max(intrinsic + time_value, TICK))


@dataclass
class FutureContract:
    tradingsymbol: str
    underlying:    str
    expiry:        Optional[date]
    lot_size:      int
    live:          bool


def resolve_future(underlying: str, today: date, skip_expiry_today: bool = True) -> Optional[FutureContract]:
    """Nearest-month index future (on expiry day the next month, so an intraday trade never
    holds a contract that is about to settle)."""
    und = normalize_underlying(underlying)
    if not und:
        return None
    rows = [i for i in _instruments()
            if i.get("name") == und and i.get("instrument_type") == "FUT"
            and str(i.get("segment", "")).endswith("FUT")]
    if rows:
        rows = sorted((r for r in rows if r.get("expiry") and r["expiry"] >= today
                       and not (skip_expiry_today and r["expiry"] == today)), key=lambda r: r["expiry"])
        if not rows:
            return None
        r = rows[0]
        return FutureContract(r["tradingsymbol"], und, r["expiry"], int(r.get("lot_size") or 0), live=True)
    if settings.uses_kite_market_data:
        return None              # real feed but no instrument master: never guess a live symbol
    return FutureContract(f"{und}SIMFUT", und, None, _SIM_LOT[und], live=False)
