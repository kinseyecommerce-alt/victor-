"""
levels_engine.py
Computes key support/resistance levels per symbol: prior day H/L/C, pivot points,
weekly high/low, and VWAP ± ATR bands. Fed to Claude trade gate for better SL/target placement.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import yfinance as yf
from loguru import logger


# ── Module-level cache ────────────────────────────────────────────────────────
# Keyed by symbol. Populated by refresh_daily(), read by get_levels() / level_context().
_levels_cache: dict[str, dict] = {}


# ── Refresh ───────────────────────────────────────────────────────────────────

async def refresh_daily(symbols: list[str]) -> None:
    """
    Download the last 10 trading days of daily OHLCV from yfinance for each symbol
    and populate _levels_cache with prior-day H/L/C, weekly H/L, and classic pivot points.

    Designed to be called once per day at ~9:05 AM IST, before the market opens.
    All downloads are done concurrently via asyncio.to_thread to avoid blocking.
    """
    tasks = [_refresh_symbol(sym) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            logger.warning("[levels] refresh failed for {}: {}", sym, res)
    logger.info("[levels] refreshed {} symbols", sum(1 for r in results if not isinstance(r, Exception)))


async def _refresh_symbol(symbol: str) -> None:
    """Download and compute all daily-bar levels for a single symbol."""
    ticker = symbol + ".NS"
    try:
        df = await asyncio.to_thread(
            yf.download, ticker, period="10d", interval="1d", progress=False, auto_adjust=True
        )
    except Exception as exc:
        logger.debug("[levels] yfinance download error {}: {}", symbol, exc)
        raise

    if df is None or df.empty:
        logger.debug("[levels] empty data for {}", symbol)
        return

    # Flatten MultiIndex columns that yfinance sometimes produces
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    df = df.dropna(subset=["high", "low", "close"]).sort_index()

    if len(df) < 2:
        logger.debug("[levels] insufficient rows for {}", symbol)
        return

    # Prior day (second-to-last complete session)
    prior = df.iloc[-2]
    pdh = float(prior["high"])
    pdl = float(prior["low"])
    pdc = float(prior["close"])

    # Weekly high/low: last 5 complete sessions
    week_slice = df.iloc[-6:-1] if len(df) >= 6 else df.iloc[:-1]
    weekly_high = float(week_slice["high"].max())
    weekly_low  = float(week_slice["low"].min())

    # Classic pivot points (Floor Trader Pivots)
    pivot  = (pdh + pdl + pdc) / 3.0
    r1     = 2.0 * pivot - pdl
    r2     = pivot + (pdh - pdl)
    s1     = 2.0 * pivot - pdh
    s2     = pivot - (pdh - pdl)

    _levels_cache[symbol] = {
        "pdh":         round(pdh, 2),
        "pdl":         round(pdl, 2),
        "pdc":         round(pdc, 2),
        "weekly_high": round(weekly_high, 2),
        "weekly_low":  round(weekly_low, 2),
        "pivot":       round(pivot, 2),
        "r1":          round(r1, 2),
        "r2":          round(r2, 2),
        "s1":          round(s1, 2),
        "s2":          round(s2, 2),
    }
    logger.debug(
        "[levels] {} → PDH={} PDL={} PDC={} P={} R1={} S1={}",
        symbol, pdh, pdl, pdc, round(pivot, 2), round(r1, 2), round(s1, 2),
    )


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_levels(symbol: str) -> dict:
    """
    Return all cached key levels for *symbol*, enriched with live VWAP ± ATR bands
    sourced from tick_engine.

    Returns an empty dict if no daily data has been loaded yet.
    """
    base = _levels_cache.get(symbol)
    if not base:
        return {}

    result = dict(base)

    # Enrich with live VWAP / ATR from tick_engine (best-effort)
    try:
        from tick_engine import tick_engine  # local import avoids circular dependency
        snap = tick_engine.all_latest().get(symbol, {})
        vwap = snap.get("vwap")
        atr  = snap.get("atr_14")
        if vwap and vwap > 0:
            result["vwap"] = round(vwap, 2)
        if vwap and vwap > 0 and atr and atr > 0:
            result["atr"]           = round(atr, 2)
            result["vwap_upper_1"]  = round(vwap + atr, 2)
            result["vwap_lower_1"]  = round(vwap - atr, 2)
    except Exception as exc:
        logger.debug("[levels] tick_engine enrich error {}: {}", symbol, exc)

    return result


def level_context(symbol: str, ltp: float) -> str:
    """
    Return a compact human-readable string describing where *ltp* sits relative to
    key support and resistance levels.

    Example output:
        "0.4% below R1 resistance (2580.50); 1.2% above S1 support (2540.00); VWAP+ATR: 2565.00"

    Returns "" when no levels are cached for the symbol.
    """
    levels = get_levels(symbol)
    if not levels:
        return ""

    ltp = float(ltp)

    # ── Resistance candidates (above ltp) ──────────────────────────────
    resistance_candidates: dict[str, Optional[float]] = {
        "R2":         levels.get("r2"),
        "R1":         levels.get("r1"),
        "Weekly High": levels.get("weekly_high"),
        "PDH":        levels.get("pdh"),
        "VWAP+ATR":   levels.get("vwap_upper_1"),
        "Pivot":      levels.get("pivot"),
    }
    nearest_resistance: Optional[tuple[str, float]] = None
    for label, price in resistance_candidates.items():
        if price is not None and price > ltp:
            if nearest_resistance is None or price < nearest_resistance[1]:
                nearest_resistance = (label, price)

    # ── Support candidates (below ltp) ─────────────────────────────────
    support_candidates: dict[str, Optional[float]] = {
        "S2":         levels.get("s2"),
        "S1":         levels.get("s1"),
        "Weekly Low": levels.get("weekly_low"),
        "PDL":        levels.get("pdl"),
        "VWAP-ATR":   levels.get("vwap_lower_1"),
        "Pivot":      levels.get("pivot"),
    }
    nearest_support: Optional[tuple[str, float]] = None
    for label, price in support_candidates.items():
        if price is not None and price < ltp:
            if nearest_support is None or price > nearest_support[1]:
                nearest_support = (label, price)

    parts: list[str] = []

    if nearest_resistance:
        r_label, r_price = nearest_resistance
        r_dist_pct = abs((r_price - ltp) / ltp * 100)
        parts.append(f"{r_dist_pct:.1f}% below {r_label} resistance ({r_price:.2f})")

    if nearest_support:
        s_label, s_price = nearest_support
        s_dist_pct = abs((ltp - s_price) / ltp * 100)
        parts.append(f"{s_dist_pct:.1f}% above {s_label} support ({s_price:.2f})")

    vwap_upper = levels.get("vwap_upper_1")
    if vwap_upper is not None:
        parts.append(f"VWAP+ATR: {vwap_upper:.2f}")

    return "; ".join(parts)


__all__ = ["get_levels", "level_context", "refresh_daily"]
