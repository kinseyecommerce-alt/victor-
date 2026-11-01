"""
institutional_flow.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tracks institutional activity: delivery percentage, block deals, bulk deals.
High delivery% + block deal BUY = strong institutional accumulation signal.

Data sources (via NSEClient):
  Delivery data : /api/deliveryArchiveData?date=DD-MM-YYYY
  Block deals   : /api/block-deal
  Bulk deals    : /api/bulk-deal

Score breakdown (0-100):
  delivery_score  = 0-100 from delivery %
  institutional_score = delivery_score*0.6
                      + 20 if block deal direction is BUY
                      + 10 if delivery_pct > 65
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Optional

from loguru import logger

from market_data import nse_client

# ── NSE API endpoints ──────────────────────────────────────────────────────────
NSE_BASE             = "https://www.nseindia.com"
DELIVERY_URL         = NSE_BASE + "/api/deliveryArchiveData?date={date}"
BLOCK_DEAL_URL       = NSE_BASE + "/api/block-deal"
BULK_DEAL_URL        = NSE_BASE + "/api/bulk-deal"

# ── Module-level in-memory cache ───────────────────────────────────────────────
# Structure: { symbol: { delivery_pct, delivery_score, has_block_deal,
#                        block_deal_direction, block_deal_qty,
#                        institutional_score, refreshed_at } }
_cache: dict[str, dict] = {}

# Tracks which API URLs have already had their failure logged today
# to avoid spamming the log on every query call.
_warned_urls: set[str] = set()

# Date of last successful full refresh (used to gate once-daily logic)
_last_refresh_date: Optional[date] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _today_str() -> str:
    """Return today's date as DD-MM-YYYY for NSE API."""
    return datetime.now().strftime("%d-%m-%Y")


def _compute_delivery_score(delivery_pct: float) -> float:
    """
    Map delivery percentage to a 0-100 score.
    Higher delivery (less intraday speculation) is rewarded; >40% gets a boost.
    """
    if delivery_pct > 40:
        return min(100.0, delivery_pct * 1.2)
    return delivery_pct * 0.8


def _compute_institutional_score(
    delivery_score: float,
    delivery_pct: float,
    has_block_deal: bool,
    block_deal_direction: Optional[str],
) -> float:
    """Combine delivery and block-deal signals into a single 0-100 score."""
    score = delivery_score * 0.6
    if has_block_deal and block_deal_direction == "BUY":
        score += 20.0
    if delivery_pct > 65.0:
        score += 10.0
    return min(100.0, score)


def _default_entry(symbol: str) -> dict:
    """Return a safe default score dict when data is unavailable."""
    return {
        "symbol":               symbol,
        "delivery_pct":         50.0,
        "delivery_score":       _compute_delivery_score(50.0),
        "has_block_deal":       False,
        "block_deal_direction": None,
        "block_deal_qty":       0,
        "institutional_score":  _compute_institutional_score(
                                    _compute_delivery_score(50.0),
                                    50.0, False, None
                                ),
        "refreshed_at":         None,
        "is_default":           True,
    }


def _warn_once(url: str, exc: object) -> None:
    """Log an API failure at WARNING level, but only once per session per URL."""
    if url not in _warned_urls:
        logger.warning("institutional_flow: NSE API failed for {} — {}", url, exc)
        _warned_urls.add(url)


# ── Data parsers ───────────────────────────────────────────────────────────────

def _parse_delivery(raw: Optional[dict]) -> dict[str, float]:
    """
    Parse deliveryArchiveData response.
    Returns { symbol: delivery_pct }.

    NSE uses at least two naming conventions across API versions:
      v1: { "deliveryQuantity": ..., "tradedQuantity": ... }
      v2: { "delQty":           ..., "ttlTradedQty":   ... }
    """
    result: dict[str, float] = {}
    if not raw:
        return result

    records = raw.get("data") or []
    if not isinstance(records, list):
        return result

    for rec in records:
        symbol = (rec.get("symbol") or rec.get("SYMBOL") or "").strip().upper()
        if not symbol:
            continue

        # Delivery quantity — try multiple key names
        del_qty = (
            rec.get("deliveryQuantity")
            or rec.get("delQty")
            or rec.get("DELIVERY_QTY")
            or 0
        )
        # Traded quantity — try multiple key names
        traded_qty = (
            rec.get("tradedQuantity")
            or rec.get("ttlTradedQty")
            or rec.get("TRADED_QTY")
            or rec.get("totalTradedQty")
            or 0
        )

        try:
            del_qty    = float(del_qty)
            traded_qty = float(traded_qty)
            if traded_qty > 0:
                result[symbol] = round((del_qty / traded_qty) * 100.0, 2)
        except (TypeError, ValueError):
            continue

    return result


def _parse_block_deals(raw: Optional[dict]) -> dict[str, dict]:
    """
    Parse block-deal response.
    Returns { symbol: { direction, qty, price } } — last deal per symbol wins.
    BUY side net-dominance takes priority when multiple deals exist.
    """
    result: dict[str, dict] = {}
    if not raw:
        return result

    records = raw.get("data") or []
    if not isinstance(records, list):
        return result

    # Accumulate BUY / SELL volumes per symbol first
    buy_qty:  dict[str, int] = {}
    sell_qty: dict[str, int] = {}
    price_map: dict[str, float] = {}

    for rec in records:
        symbol    = (rec.get("symbol") or rec.get("SYMBOL") or "").strip().upper()
        direction = (rec.get("buySell") or rec.get("buy_sell") or "").strip().upper()
        if not symbol or direction not in ("BUY", "SELL"):
            continue

        try:
            qty   = int(float(rec.get("quantity") or rec.get("qty") or 0))
            price = float(rec.get("tradePrice") or rec.get("trade_price") or 0)
        except (TypeError, ValueError):
            qty, price = 0, 0.0

        if direction == "BUY":
            buy_qty[symbol]  = buy_qty.get(symbol, 0) + qty
        else:
            sell_qty[symbol] = sell_qty.get(symbol, 0) + qty

        price_map[symbol] = price   # last price wins (good enough for signal)

    # Determine net direction per symbol
    all_syms = set(buy_qty) | set(sell_qty)
    for sym in all_syms:
        b = buy_qty.get(sym, 0)
        s = sell_qty.get(sym, 0)
        net_direction = "BUY" if b >= s else "SELL"
        result[sym] = {
            "direction": net_direction,
            "qty":       b + s,
            "price":     price_map.get(sym, 0.0),
        }

    return result


# ── Core refresh logic ─────────────────────────────────────────────────────────

async def _fetch_delivery() -> dict[str, float]:
    """Fetch and parse today's delivery data from NSE."""
    url = DELIVERY_URL.format(date=_today_str())
    try:
        raw = await nse_client.get(url)
        if raw is None:
            _warn_once(url, "returned None (HTTP error or no session)")
            return {}
        parsed = _parse_delivery(raw)
        logger.debug("institutional_flow: delivery data loaded for {} symbols", len(parsed))
        return parsed
    except Exception as exc:
        _warn_once(url, exc)
        return {}


async def _fetch_block_deals() -> dict[str, dict]:
    """Fetch and parse today's block deals from NSE."""
    try:
        raw = await nse_client.get(BLOCK_DEAL_URL)
        if raw is None:
            _warn_once(BLOCK_DEAL_URL, "returned None (HTTP error or no session)")
            return {}
        parsed = _parse_block_deals(raw)
        logger.debug("institutional_flow: block deals loaded for {} symbols", len(parsed))
        return parsed
    except Exception as exc:
        _warn_once(BLOCK_DEAL_URL, exc)
        return {}


async def _fetch_bulk_deals() -> dict[str, dict]:
    """
    Fetch bulk deals.  Bulk deals use the same schema as block deals,
    so we reuse _parse_block_deals.
    """
    try:
        raw = await nse_client.get(BULK_DEAL_URL)
        if raw is None:
            _warn_once(BULK_DEAL_URL, "returned None (HTTP error or no session)")
            return {}
        parsed = _parse_block_deals(raw)
        logger.debug("institutional_flow: bulk deals loaded for {} symbols", len(parsed))
        return parsed
    except Exception as exc:
        _warn_once(BULK_DEAL_URL, exc)
        return {}


def _build_cache_entry(
    symbol: str,
    delivery_map: dict[str, float],
    deal_map: dict[str, dict],
) -> dict:
    """
    Construct and return a fully-scored cache entry for one symbol.

    `is_default` is True when neither delivery data nor deal data was available
    for the symbol, meaning all values are filled with safe defaults.
    """
    sym_upper = symbol.upper()

    has_delivery_data = sym_upper in delivery_map
    delivery_pct      = delivery_map.get(sym_upper, 50.0)
    delivery_score    = _compute_delivery_score(delivery_pct)

    deal = deal_map.get(sym_upper)
    has_block_deal       = deal is not None
    block_deal_direction = deal["direction"] if deal else None
    block_deal_qty       = deal["qty"]       if deal else 0

    institutional_score = _compute_institutional_score(
        delivery_score, delivery_pct, has_block_deal, block_deal_direction
    )

    # Mark as default only when we genuinely had no data for this symbol
    is_default = not has_delivery_data and not has_block_deal

    return {
        "symbol":               sym_upper,
        "delivery_pct":         delivery_pct,
        "delivery_score":       round(delivery_score, 2),
        "has_block_deal":       has_block_deal,
        "block_deal_direction": block_deal_direction,
        "block_deal_qty":       block_deal_qty,
        "institutional_score":  round(institutional_score, 2),
        "refreshed_at":         datetime.now().isoformat(),
        "is_default":           is_default,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

async def refresh_daily(symbols: Optional[list[str]] = None) -> None:
    """
    Fetch all institutional data sources and populate the in-memory cache.

    Call once per trading day (e.g. from platform_scheduler after market open).
    If `symbols` is provided only those symbols are cached; otherwise every
    symbol present in the NSE delivery response is cached.

    Clears stale warnings from the previous session so fresh errors are logged.
    """
    global _last_refresh_date
    _warned_urls.clear()

    logger.info("institutional_flow: starting daily refresh ({})", _today_str())

    # Run all three fetches concurrently to minimise wall time
    delivery_map, block_map, bulk_map = await asyncio.gather(
        _fetch_delivery(),
        _fetch_block_deals(),
        _fetch_bulk_deals(),
    )

    # Merge block + bulk deal maps; block deals take precedence for BUY signals
    combined_deal_map: dict[str, dict] = {**bulk_map, **block_map}

    # Determine which symbols to cache
    if symbols:
        target_symbols = [s.upper() for s in symbols]
    else:
        # Union of all symbols we have any data for
        target_symbols = list(
            set(delivery_map.keys()) | set(combined_deal_map.keys())
        )

    refreshed = 0
    for sym in target_symbols:
        try:
            _cache[sym] = _build_cache_entry(sym, delivery_map, combined_deal_map)
            refreshed += 1
        except Exception as exc:
            logger.warning("institutional_flow: cache build failed for {}: {}", sym, exc)

    _last_refresh_date = date.today()
    logger.info(
        "institutional_flow: refresh complete — {} symbols cached, "
        "{} with delivery data, {} with deal data",
        refreshed,
        len(delivery_map),
        len(combined_deal_map),
    )


def get_cached_score(symbol: str) -> dict:
    """Synchronous accessor — returns cached entry or neutral default. Never raises."""
    return _cache.get(symbol.upper(), _default_entry(symbol))


async def get_institutional_score(symbol: str) -> dict:
    """
    Return the institutional score dict for `symbol`.

    If the cache has a fresh entry it is returned immediately.
    If the cache is empty (no refresh has run today) a background refresh is
    triggered for this symbol only, then the result is returned.
    Falls back to safe defaults on any error — never raises.

    Return shape:
        {
          "symbol":               str,
          "delivery_pct":         float,   # 0-100
          "delivery_score":       float,   # 0-100
          "has_block_deal":       bool,
          "block_deal_direction": "BUY" | "SELL" | None,
          "block_deal_qty":       int,
          "institutional_score":  float,   # 0-100
          "refreshed_at":         str | None,
          "is_default":           bool,    # True when data is unavailable
        }
    """
    sym_upper = symbol.upper()

    if sym_upper in _cache:
        return _cache[sym_upper]

    # Cache miss — try a targeted refresh before falling back to defaults
    try:
        await refresh_daily(symbols=[sym_upper])
        if sym_upper in _cache:
            return _cache[sym_upper]
    except Exception as exc:
        logger.warning(
            "institutional_flow: on-demand refresh failed for {}: {}", sym_upper, exc
        )

    # Last resort: return safe defaults so callers are never blocked
    default = _default_entry(sym_upper)
    _cache[sym_upper] = default   # store so repeated calls don't re-fetch
    return default
