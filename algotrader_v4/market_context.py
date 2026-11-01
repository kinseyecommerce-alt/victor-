"""
market_context.py — one compact, cache-only view of everything the app knows about a
symbol beyond its price: regime and India VIX, event risk (results, RBI, ex-dates), key
levels, institutional flow, option-chain intelligence, multi-timeframe alignment,
correlation with what is already held, the trend filter, and the trade memory's
recent lessons.

Used by the AI trader (get_market_context tool and every real-time alert) and by the
research loop. Every read is a cache lookup: this never does network I/O, so it is safe
on the tick path. Missing data is reported as missing, never guessed.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger


def _safe(fn, default):
    try:
        return fn()
    except Exception as exc:                         # one broken feed must not hide the rest
        logger.debug("[context] {}: {}", getattr(fn, "__name__", "field"), exc)
        return default


def open_symbols() -> list[str]:
    """Symbols with an open position anywhere (agents + AI), via the trailing-stop registry."""
    from trailing_sl_engine import trailing_sl_engine, SLStatus
    return sorted({p.symbol for p in trailing_sl_engine._positions.values() if p.status == SLStatus.ACTIVE})


def regime_block() -> dict:
    from market_regime import regime_detector
    sig = regime_detector.current_signals
    return {
        "regime": regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
        "india_vix": round(sig.india_vix, 2) if sig and sig.india_vix else None,
        "vix_change_pct": round(sig.vix_chg_pct, 2) if sig and sig.vix_chg_pct else None,
        "size_factor": regime_detector.current_plan.size_factor if regime_detector.current_plan else 1.0,
        "paused_strategies": list(regime_detector.current_plan.paused) if regime_detector.current_plan else [],
    }


def event_block(symbol: str) -> dict:
    from event_calendar import get_event_risk
    e = get_event_risk(symbol)
    return {"risk_level": e.get("risk_level", "NONE"), "event": e.get("event_type") or None,
            "hours_until": e.get("hours_until"), "size_factor": e.get("size_factor", 1.0),
            "description": e.get("description") or None}


def levels_block(symbol: str, ltp: float) -> dict:
    from levels_engine import get_levels
    lv = get_levels(symbol)
    if not lv:
        return {"available": False}
    keys = ("prev_high", "prev_low", "prev_close", "pivot", "r1", "s1", "r2", "s2",
            "week_high", "week_low", "vwap", "vwap_upper_1", "vwap_lower_1")
    out = {k: lv[k] for k in keys if k in lv}
    if ltp and out:
        above = {k: v for k, v in out.items() if isinstance(v, (int, float)) and v > ltp}
        below = {k: v for k, v in out.items() if isinstance(v, (int, float)) and v < ltp}
        if above:
            k = min(above, key=above.get); out["nearest_resistance"] = {k: above[k], "pct_away": round((above[k] / ltp - 1) * 100, 2)}
        if below:
            k = max(below, key=below.get); out["nearest_support"] = {k: below[k], "pct_away": round((1 - below[k] / ltp) * 100, 2)}
    out["available"] = True
    return out


def institutional_block(symbol: str) -> dict:
    from institutional_flow import get_cached_score
    d = get_cached_score(symbol) or {}
    return {"score_0_100": round(d.get("institutional_score", 50.0), 1),
            "delivery_pct": d.get("delivery_pct"), "block_deal": d.get("block_deal") or d.get("has_block_deal"),
            "bulk_deal": d.get("bulk_deal") or d.get("has_bulk_deal")}


def options_block(symbol: str) -> dict:
    from options_intelligence import get_cached
    from options_resolver import normalize_underlying, INDEX_QUOTE
    und = normalize_underlying(symbol)
    keys = [symbol] + ([und, INDEX_QUOTE[und]] if und else [])
    for k in keys:
        d = get_cached(k)
        if d:
            return {kk: d.get(kk) for kk in ("iv_rank", "iv_percentile", "pcr", "max_pain", "oi_buildup",
                                             "support_oi", "resistance_oi", "atm_iv") if kk in d}
    return {"available": False}


def mtf_block(snap) -> dict:
    from multi_timeframe import check
    out = {}
    for side in ("BUY", "SELL"):
        r = check(snap, side)
        out[side] = {"aligned": r.aligned, "timeframes_agreeing": r.score}
    return out


def correlation_block(symbol: str) -> dict:
    from correlation_guard import check, portfolio_heat
    held = [s for s in open_symbols() if s != symbol]
    c = check(symbol, held)
    return {"open_positions": held, "allowed": c["allowed"], "size_factor": c["size_factor"],
            "reason": c["reason"], "portfolio_heat_0_1": round(portfolio_heat(held + [symbol]), 2) if held else 0.0}


def memory_block(n: int = 5) -> list[dict]:
    from trade_memory import get_recent_insights
    return [{"symbol": e.get("symbol"), "strategy": e.get("strategy"), "pnl": e.get("pnl"),
             "lesson": e.get("lesson")} for e in get_recent_insights(n) if e.get("lesson")]


def context_for(symbol: str, snap=None, include_memory: bool = True) -> dict:
    """Everything known about `symbol` right now (cache reads only)."""
    from tick_engine import tick_engine
    import market_filter
    if snap is None:
        tick, ind = tick_engine.latest(symbol)
        ltp = float(tick.ltp) if tick else 0.0
        buf = tick_engine._bufs_1min.get(symbol)
        candles = buf.candles() if buf else []
        snap_like = type("S", (), {"candles_1min": candles})() if candles else None
    else:
        ltp = float(snap.tick.ltp)
        snap_like = snap
    out = {
        "symbol": symbol,
        "market": _safe(regime_block, {"regime": "UNKNOWN"}),
        "trend_filter": _safe(lambda: {k: market_filter.state()[k] for k in ("state", "pct_from_ema", "size_factor")}, {}),
        "event_risk": _safe(lambda: event_block(symbol), {"risk_level": "UNKNOWN"}),
        "key_levels": _safe(lambda: levels_block(symbol, ltp), {"available": False}),
        "institutional": _safe(lambda: institutional_block(symbol), {}),
        "options": _safe(lambda: options_block(symbol), {"available": False}),
        "multi_timeframe": _safe(lambda: mtf_block(snap_like), {}) if snap_like else {"available": False},
        "correlation": _safe(lambda: correlation_block(symbol), {}),
    }
    if include_memory:
        out["recent_lessons"] = _safe(memory_block, [])
    return out


def compact_for_alert(symbol: str, snap=None) -> dict:
    """The short form that rides on every real-time alert (keeps the prompt small)."""
    c = context_for(symbol, snap, include_memory=False)
    lv = c.get("key_levels", {})
    return {
        "regime": c["market"].get("regime"), "india_vix": c["market"].get("india_vix"),
        "trend_filter": c["trend_filter"].get("state"),
        "event_risk": {k: c["event_risk"].get(k) for k in ("risk_level", "event", "hours_until", "size_factor")},
        "nearest_support": lv.get("nearest_support"), "nearest_resistance": lv.get("nearest_resistance"),
        "institutional_score": c["institutional"].get("score_0_100"),
        "iv_rank": c["options"].get("iv_rank"), "pcr": c["options"].get("pcr"),
        "mtf": c.get("multi_timeframe"),
        "correlation": {k: c["correlation"].get(k) for k in ("allowed", "size_factor", "portfolio_heat_0_1")},
    }
