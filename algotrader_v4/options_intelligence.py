"""
options_intelligence.py
Options chain analysis: IV rank, IV percentile, max pain, PCR, OI buildup.
Uses NSE's free option chain API (no auth required).

Public exports:
  async get_iv_context(symbol)   → full analysis dict or {} on failure
  async update_cache(symbols)    → parallel refresh of multiple symbols
  get_cached(symbol)             → sync, returns cached dict or {}
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from market_data import nse_client


# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE_DIR     = Path(__file__).parent
_LOGS_DIR     = _BASE_DIR / "logs"
_IV_HIST_FILE = _LOGS_DIR / "iv_history.json"

_LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── In-memory cache ────────────────────────────────────────────────────────────

_cache: dict[str, dict] = {}   # symbol → {data..., updated_at: float}
_CACHE_TTL_SEC = 300           # 5 minutes


# ── IV history I/O ─────────────────────────────────────────────────────────────

def _load_iv_history() -> dict[str, list[dict]]:
    """Load IV history from disk. Returns {} on any failure."""
    try:
        if _IV_HIST_FILE.exists():
            with _IV_HIST_FILE.open("r") as fh:
                return json.load(fh)
    except Exception as exc:
        logger.debug("[options_intel] iv_history load failed: {}", exc)
    return {}


def _save_iv_history(history: dict[str, list[dict]]) -> None:
    """Persist IV history to disk. Silent on failure."""
    try:
        with _IV_HIST_FILE.open("w") as fh:
            json.dump(history, fh)
    except Exception as exc:
        logger.debug("[options_intel] iv_history save failed: {}", exc)


def _append_iv_history(symbol: str, atm_iv: float) -> None:
    """Append today's ATM IV to the 60-day rolling history for symbol."""
    history = _load_iv_history()
    entries = history.get(symbol, [])
    today_str = date.today().isoformat()

    # Replace today's entry if it already exists
    entries = [e for e in entries if e.get("date") != today_str]
    entries.append({"date": today_str, "atm_iv": round(atm_iv, 4)})

    # Keep only the last 60 calendar days
    entries.sort(key=lambda e: e["date"])
    entries = entries[-60:]

    history[symbol] = entries
    _save_iv_history(history)


def _compute_iv_rank_percentile(symbol: str, current_iv: float) -> tuple[float, float]:
    """
    Returns (iv_rank, iv_percentile) using the last 60 days of history.
    Falls back to (50.0, 50.0) when history is insufficient.
    """
    history = _load_iv_history()
    entries = history.get(symbol, [])

    # Need at least 2 historical points to compute meaningful statistics
    if len(entries) < 2:
        return 50.0, 50.0

    ivs = [e["atm_iv"] for e in entries]
    iv_min = min(ivs)
    iv_max = max(ivs)

    iv_rank = (
        round((current_iv - iv_min) / (iv_max - iv_min) * 100, 1)
        if iv_max > iv_min
        else 50.0
    )
    iv_rank = max(0.0, min(100.0, iv_rank))

    below_count = sum(1 for iv in ivs if iv < current_iv)
    iv_percentile = round(below_count / len(ivs) * 100, 1)

    return iv_rank, iv_percentile


# ── Option chain parsing ───────────────────────────────────────────────────────

def _parse_chain(data: dict) -> Optional[dict]:
    """
    Parse NSE option chain response into a structured analysis.
    Returns None if the response is malformed or empty.
    """
    try:
        records    = data.get("records", {})
        chain_data = records.get("data", [])
        spot_price = float(records.get("underlyingValue", 0))

        if not chain_data or spot_price <= 0:
            return None

        # Collect all expiry dates to isolate the nearest (current) expiry
        expiry_dates: set[str] = set()
        for row in chain_data:
            for side in ("CE", "PE"):
                if side in row:
                    exp = row[side].get("expiryDate", "")
                    if exp:
                        expiry_dates.add(exp)

        if not expiry_dates:
            return None

        # "Current" expiry = smallest date string (ISO or DD-Mon-YYYY — sort lexicographically
        # after normalising; NSE uses "DD-Mon-YYYY" which sorts naturally within same month)
        current_expiry = sorted(expiry_dates)[0]

        # Build per-strike aggregates for the current expiry only
        strikes: dict[float, dict] = {}
        for row in chain_data:
            for side in ("CE", "PE"):
                item = row.get(side)
                if not item:
                    continue
                if item.get("expiryDate") != current_expiry:
                    continue
                k = float(item.get("strikePrice", 0))
                if k <= 0:
                    continue
                if k not in strikes:
                    strikes[k] = {"CE": None, "PE": None}
                strikes[k][side] = {
                    "oi":         int(item.get("openInterest", 0)),
                    "oi_change":  int(item.get("changeinOpenInterest", 0)),
                    "iv":         float(item.get("impliedVolatility", 0) or 0),
                    "ltp":        float(item.get("lastPrice", 0) or 0),
                }

        if not strikes:
            return None

        sorted_strikes = sorted(strikes.keys())

        # ── ATM IV ────────────────────────────────────────────────────────────
        # Find strike closest to spot
        atm_strike = min(sorted_strikes, key=lambda k: abs(k - spot_price))
        atm_entry  = strikes[atm_strike]
        ce_iv = atm_entry["CE"]["iv"] if atm_entry["CE"] else 0.0
        pe_iv = atm_entry["PE"]["iv"] if atm_entry["PE"] else 0.0

        valid_ivs = [v for v in (ce_iv, pe_iv) if v > 0]
        atm_iv = round(sum(valid_ivs) / len(valid_ivs), 2) if valid_ivs else 0.0

        # ── PCR ───────────────────────────────────────────────────────────────
        total_pe_oi = sum(
            strikes[k]["PE"]["oi"] for k in sorted_strikes if strikes[k]["PE"]
        )
        total_ce_oi = sum(
            strikes[k]["CE"]["oi"] for k in sorted_strikes if strikes[k]["CE"]
        )
        pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 0.0

        # ── Max Pain ──────────────────────────────────────────────────────────
        # For each candidate strike K, compute total pain to all option holders:
        # CE holders lose when spot < K  → pain per CE = max(0, K - spot) * OI
        # PE holders lose when spot > K  → pain per PE = max(0, spot - K) * OI
        # We iterate K over all strikes and compute pain assuming expiry at K.
        max_pain_strike = _compute_max_pain(strikes, sorted_strikes)

        # ── OI Buildup ────────────────────────────────────────────────────────
        oi_changes: list[dict] = []
        for k in sorted_strikes:
            for side in ("CE", "PE"):
                entry = strikes[k].get(side)
                if entry and entry["oi_change"] != 0:
                    oi_changes.append({
                        "strike":    k,
                        "type":      side,
                        "oi_change": entry["oi_change"],
                    })

        # Sort by absolute OI change, take top 3
        oi_buildup = sorted(
            oi_changes, key=lambda x: abs(x["oi_change"]), reverse=True
        )[:3]

        return {
            "spot_price":   round(spot_price, 2),
            "atm_strike":   atm_strike,
            "atm_iv":       atm_iv,
            "pcr":          pcr,
            "max_pain":     max_pain_strike,
            "oi_buildup":   oi_buildup,
            "expiry":       current_expiry,
            "total_ce_oi":  total_ce_oi,
            "total_pe_oi":  total_pe_oi,
        }

    except Exception as exc:
        logger.debug("[options_intel] chain parse error: {}", exc)
        return None


def _compute_max_pain(
    strikes: dict[float, dict],
    sorted_strikes: list[float],
) -> float:
    """
    For each strike K (potential expiry price), compute the total monetary
    pain inflicted on all option holders, then return K that minimises it.
    """
    best_strike = sorted_strikes[0] if sorted_strikes else 0.0
    best_pain   = float("inf")

    for candidate in sorted_strikes:
        total_pain = 0.0
        for k in sorted_strikes:
            entry = strikes[k]
            # CE holders lose max(0, candidate - k) * OI  (in-the-money CEs expire worthless)
            if entry["CE"]:
                ce_oi = entry["CE"]["oi"]
                total_pain += max(0.0, candidate - k) * ce_oi
            # PE holders lose max(0, k - candidate) * OI
            if entry["PE"]:
                pe_oi = entry["PE"]["oi"]
                total_pain += max(0.0, k - candidate) * pe_oi

        if total_pain < best_pain:
            best_pain   = total_pain
            best_strike = candidate

    return best_strike


# ── Public API ────────────────────────────────────────────────────────────────

async def get_iv_context(symbol: str) -> dict:
    """
    Fetch and analyse the full options chain for `symbol`.

    Returns a dict with keys:
      symbol, spot_price, atm_strike, atm_iv, pcr, max_pain,
      oi_buildup, iv_rank, iv_percentile, expiry, updated_at

    Returns {} on any failure.
    """
    symbol = symbol.upper()
    try:
        data = await nse_client.option_chain(symbol)
        if not data:
            logger.debug("[options_intel] no chain data for {}", symbol)
            return {}

        parsed = _parse_chain(data)
        if not parsed:
            logger.debug("[options_intel] chain parse returned nothing for {}", symbol)
            return {}

        atm_iv = parsed["atm_iv"]

        # Persist today's IV and compute rank/percentile
        if atm_iv > 0:
            _append_iv_history(symbol, atm_iv)

        iv_rank, iv_percentile = _compute_iv_rank_percentile(symbol, atm_iv)

        result = {
            "symbol":        symbol,
            "spot_price":    parsed["spot_price"],
            "atm_strike":    parsed["atm_strike"],
            "atm_iv":        atm_iv,
            "pcr":           parsed["pcr"],
            "max_pain":      parsed["max_pain"],
            "oi_buildup":    parsed["oi_buildup"],
            "iv_rank":       iv_rank,
            "iv_percentile": iv_percentile,
            "expiry":        parsed["expiry"],
            "total_ce_oi":   parsed["total_ce_oi"],
            "total_pe_oi":   parsed["total_pe_oi"],
            "updated_at":    datetime.now().isoformat(),
        }

        # Store in in-memory cache
        _cache[symbol] = {**result, "_updated_ts": time.time()}

        logger.info(
            "[options_intel] {} — spot={} ATM_IV={} PCR={} MaxPain={} IVRank={} IVPct={}",
            symbol,
            parsed["spot_price"],
            atm_iv,
            parsed["pcr"],
            parsed["max_pain"],
            iv_rank,
            iv_percentile,
        )
        return result

    except Exception as exc:
        logger.warning("[options_intel] get_iv_context({}) failed: {}", symbol, exc)
        return {}


async def update_cache(symbols: list[str]) -> None:
    """
    Refresh the in-memory cache for all symbols in parallel.
    Errors for individual symbols are logged but do not propagate.
    """
    if not symbols:
        return
    tasks = [get_iv_context(sym) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for sym, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.debug("[options_intel] update_cache({}) exception: {}", sym, result)


def get_cached(symbol: str) -> dict:
    """
    Synchronous accessor for cached option context.
    Returns cached dict if fresh (< 5 min), else {}.
    """
    symbol = symbol.upper()
    entry  = _cache.get(symbol)
    if not entry:
        return {}
    age = time.time() - entry.get("_updated_ts", 0)
    if age > _CACHE_TTL_SEC:
        return {}
    # Return a copy without the internal timestamp key
    return {k: v for k, v in entry.items() if k != "_updated_ts"}


