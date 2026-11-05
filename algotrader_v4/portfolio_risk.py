"""
portfolio_risk.py — portfolio-level risk applied to every entry, AI or rule agent.

Three rules the per-trade limits don't cover:

1. Volatility-targeted size: a trade may lose at most RISK_PER_TRADE_PCT of capital if its
   stop is hit (quantity ≤ capital × pct / |entry − stop|). A wide stop means a small
   position, a tight stop a larger one — the rupee risk per trade is constant.
2. Correlation: no new position that would make the book one bet (correlation_guard),
   plus a cap on average pairwise correlation ("heat"); event risk scales size the same way.
3. Daily VaR cap: the sum of what the open book could lose on a bad day (1.65σ, 95%,
   one day) plus the new trade must stay under PORTFOLIO_VAR_CAP_PCT of capital. Options
   and spreads count their maximum loss; equity and futures count notional × σ.

Everything returns a reason string so the AI (or the log) says exactly why a size was cut.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

from loguru import logger

from config import settings

Z_95 = 1.65
_vol_cache: dict[tuple[str, date], float] = {}


def daily_vol(symbol: str, fallback: float = 0.015) -> float:
    """Daily close-to-close volatility (fraction), 60-session window, cached per day."""
    from ist_clock import now_ist
    key = (symbol, now_ist().date())
    if key in _vol_cache:
        return _vol_cache[key]
    vol = fallback
    try:
        from daily_data import daily_closes
        closes = [c for _, c in daily_closes(symbol)][-61:]
        if len(closes) >= 21:
            rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
            m = sum(rets) / len(rets)
            vol = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
    except Exception as exc:
        logger.debug("[risk] vol {}: {}", symbol, exc)
    vol = max(0.004, min(vol, 0.08))
    _vol_cache[key] = vol
    return vol


def _open_book() -> list[dict]:
    """Open positions as {symbol, kind, value, max_loss}: equity/future → notional, options → premium."""
    from trailing_sl_engine import trailing_sl_engine, SLStatus
    from agents.base_agent import _tsl_sl_orders
    book = []
    for oid, pos in list(trailing_sl_engine._positions.items()):
        if pos.status != SLStatus.ACTIVE:
            continue
        e = _tsl_sl_orders.get(oid, {})
        exch = e.get("exchange", "NSE")
        is_opt = exch == "NFO" and not pos.symbol.endswith("FUT")
        notional = pos.entry_price * pos.quantity
        book.append({"symbol": pos.symbol, "kind": "option" if is_opt else ("future" if exch == "NFO" else "equity"),
                     "notional": notional, "max_loss": notional if is_opt else None,
                     "underlying": e.get("guard_symbol", pos.symbol)})
    try:
        from ai_trader import ai_trader
        for sid, sp in ai_trader._spreads.items():
            book.append({"symbol": sid, "kind": "spread", "notional": sp["max_loss"], "max_loss": sp["max_loss"],
                         "underlying": sp["underlying"]})
    except Exception:
        pass
    return book


def var_of(entry: dict) -> float:
    if entry.get("max_loss") is not None:
        return float(entry["max_loss"])
    ref = entry.get("underlying") or entry["symbol"]
    return float(entry["notional"]) * daily_vol(ref) * Z_95


def portfolio_var() -> dict:
    book = _open_book()
    total = sum(var_of(b) for b in book)
    cap = settings.total_capital * settings.portfolio_var_cap_pct / 100
    return {"var_1d_95": round(total, 2), "cap": round(cap, 2), "used_pct": round(total / cap * 100, 1) if cap else None,
            "positions": len(book)}


def size_for_stop(price: float, stop: float) -> int:
    """Largest quantity whose loss at the stop stays within RISK_PER_TRADE_PCT of capital."""
    dist = abs(price - stop)
    if dist <= 0:
        return 0
    return int(settings.total_capital * settings.risk_per_trade_pct / 100 // dist)


def pre_entry(symbol: str, side: str, quantity: int, price: float, stop: Optional[float] = None,
              kind: str = "equity", max_loss: Optional[float] = None, underlying: Optional[str] = None,
              unit_size: int = 1) -> dict:
    """Apply the portfolio rules to a proposed entry.

    unit_size: lot size for F&O — max_quantity is rounded down to whole lots, and less than one
    lot means "no".

    Returns {"ok": bool, "reason": str, "max_quantity": int, "size_factor": float, "var": {...}}.
    ok=False means do not trade; otherwise quantity must be ≤ max_quantity (the caller either
    scales down or rejects with the reason, the AI trader does the latter so the model sees it).
    """
    from event_calendar import get_event_risk
    from correlation_guard import check as corr_check, portfolio_heat
    from market_context import open_symbols

    reasons: list[str] = []
    factor = 1.0
    max_q = quantity

    # 1. events
    ev = get_event_risk(underlying or symbol)
    if ev["size_factor"] == 0.0:
        return {"ok": False, "reason": f"event block: {ev['description'] or ev['event_type']} in "
                                       f"{ev['hours_until']}h", "max_quantity": 0, "size_factor": 0.0}
    if ev["size_factor"] < 1.0:
        factor *= ev["size_factor"]
        reasons.append(f"{ev['risk_level']} event risk ({ev['event_type']}) → size ×{ev['size_factor']}")

    # 2. correlation
    held = [s for s in open_symbols() if s not in (symbol, underlying)]
    if kind in ("equity",):
        c = corr_check(symbol, held)
        if not c["allowed"]:
            return {"ok": False, "reason": f"correlation block: {c['reason']}", "max_quantity": 0, "size_factor": 0.0}
        if c["size_factor"] < 1.0:
            factor *= c["size_factor"]
            reasons.append(f"correlated with open positions → size ×{c['size_factor']}")
        heat = portfolio_heat(held + [symbol]) if held else 0.0
        if heat > settings.max_portfolio_heat:
            return {"ok": False, "reason": f"portfolio heat {heat:.2f} > {settings.max_portfolio_heat} — the book "
                                           "would be one correlated bet", "max_quantity": 0, "size_factor": 0.0}

    # 3. volatility-targeted size (rupee risk per trade)
    if stop is not None and kind in ("equity", "future"):
        cap_q = size_for_stop(price, stop)
        if cap_q < 1:
            return {"ok": False, "reason": "stop is so far that even one unit risks more than "
                                           f"{settings.risk_per_trade_pct}% of capital", "max_quantity": 0, "size_factor": 0.0}
        if cap_q < max_q:
            max_q = cap_q
            reasons.append(f"risk per trade {settings.risk_per_trade_pct}% of capital "
                           f"(₹{settings.total_capital * settings.risk_per_trade_pct / 100:,.0f}) at this stop → max qty {cap_q}")
    max_q = max(0, int(max_q * factor))

    # 4. daily VaR cap
    pv = portfolio_var()
    if kind in ("option", "spread"):
        new_var = float(max_loss if max_loss is not None else price * max_q)
    else:
        new_var = price * max_q * daily_vol(underlying or symbol) * Z_95
    if pv["cap"] and pv["var_1d_95"] + new_var > pv["cap"]:
        room = pv["cap"] - pv["var_1d_95"]
        if room <= 0:
            return {"ok": False, "reason": f"daily VaR cap reached (₹{pv['var_1d_95']:,.0f} of ₹{pv['cap']:,.0f}); "
                                           "close something first", "max_quantity": 0, "size_factor": 0.0, "var": pv}
        per_unit = new_var / max(max_q, 1)
        max_q = int(room // per_unit) if per_unit > 0 else 0
        reasons.append(f"daily VaR cap ₹{pv['cap']:,.0f} (used ₹{pv['var_1d_95']:,.0f}, room ₹{room:,.0f}) → max qty {max_q}")
    if unit_size > 1:
        max_q = (max_q // unit_size) * unit_size
        if max_q < unit_size:
            return {"ok": False, "reason": ("; ".join(reasons) or "no room") + f" — less than one lot of {unit_size}",
                    "max_quantity": 0, "size_factor": 0.0, "var": pv}
    if max_q < 1:
        return {"ok": False, "reason": "; ".join(reasons) or "no room for this trade", "max_quantity": 0,
                "size_factor": 0.0, "var": pv}
    return {"ok": True, "reason": "; ".join(reasons), "max_quantity": max_q, "size_factor": round(factor, 3), "var": pv}


def status() -> dict:
    pv = portfolio_var()
    from correlation_guard import portfolio_heat
    from market_context import open_symbols
    held = open_symbols()
    return {**pv, "portfolio_heat": round(portfolio_heat(held), 2) if len(held) > 1 else 0.0,
            "open_symbols": held, "risk_per_trade_pct": settings.risk_per_trade_pct,
            "max_loss_per_trade": round(settings.total_capital * settings.risk_per_trade_pct / 100, 2),
            "max_portfolio_heat": settings.max_portfolio_heat}
