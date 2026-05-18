"""
event_calendar.py
Guards against trading into earnings, ex-dividend, bonus, splits, and key economic events.
Reduces position size or blocks entry when a significant event is imminent.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, date, timedelta
from typing import Optional

from loguru import logger

from market_data import nse_client


# ── RBI MPC announcement dates ────────────────────────────────────────────────
# Full FY2025-26 calendar — add future dates as published by RBI.
RBI_DATES: list[str] = [
    "2025-06-04",
    "2025-08-06",
    "2025-10-08",
    "2025-12-05",
    "2026-02-06",
    "2026-04-08",
]

# ── Risk classification ───────────────────────────────────────────────────────
# Ordered from most- to least-critical; first match wins.
_RISK_THRESHOLDS: list[tuple[str, float, float]] = [
    # (risk_level, hours_threshold, size_factor)
    ("CRITICAL", 1.0,   0.0),
    ("HIGH",     4.0,   0.3),
    ("MEDIUM",  24.0,   0.6),
    ("LOW",     48.0,  0.85),
]
_SAFE_RESULT: dict = {
    "risk_level":   "NONE",
    "event_type":   "",
    "hours_until":  float("inf"),
    "size_factor":  1.0,
    "description":  "",
}

# ── Module-level cache ────────────────────────────────────────────────────────
# symbol → list of event dicts: {"event_type": str, "dt": datetime, "description": str}
_event_cache: dict[str, list] = {}
# "MARKET" key holds broad macro events (RBI, budget, etc.) not tied to a single symbol.
_MARKET_KEY = "__MARKET__"


# ── Public API ────────────────────────────────────────────────────────────────

def get_event_risk(symbol: str, current_time: Optional[datetime] = None) -> dict:
    """
    Return an event-risk assessment for *symbol* at *current_time*.

    Never blocks on I/O — reads only the in-process cache.
    Falls back to NONE risk on any error so trading is never blocked
    by a calendar-data issue.
    """
    try:
        now = current_time or datetime.now()
        events = _event_cache.get(symbol, []) + _event_cache.get(_MARKET_KEY, [])
        if not events:
            return dict(_SAFE_RESULT)

        # Find the soonest upcoming event
        min_hours: float = float("inf")
        worst_event: Optional[dict] = None
        for ev in events:
            dt: datetime = ev["dt"]
            delta_hours = (dt - now).total_seconds() / 3600.0
            if -1.0 <= delta_hours < min_hours:   # allow 1h past (event day, pre-open)
                min_hours = delta_hours
                worst_event = ev

        if worst_event is None:
            return dict(_SAFE_RESULT)

        hours = min_hours
        risk_level, size_factor = "NONE", 1.0

        # Results *today* are treated as HIGH regardless of exact time
        if worst_event.get("results_today"):
            risk_level, size_factor = "HIGH", 0.3
        else:
            for level, threshold, factor in _RISK_THRESHOLDS:
                if hours < threshold:
                    risk_level, size_factor = level, factor
                    break

        return {
            "risk_level":  risk_level,
            "event_type":  worst_event.get("event_type", ""),
            "hours_until": round(hours, 2),
            "size_factor": size_factor,
            "description": worst_event.get("description", ""),
        }
    except Exception as exc:
        logger.debug("[events] get_event_risk error {}: {}", symbol, exc)
        return dict(_SAFE_RESULT)


def has_results_today(symbol: str) -> bool:
    """Return True if *symbol* has earnings / annual results scheduled for today."""
    today = date.today()
    for ev in _event_cache.get(symbol, []):
        if ev.get("results_today") and ev["dt"].date() == today:
            return True
    return False


async def refresh_calendar() -> None:
    """
    Fetch NSE corporate actions (next 7 days) and the NSE event calendar,
    then rebuild *_event_cache*.

    Designed to be called once daily at 09:05 AM IST via the platform scheduler.
    Never raises — all errors are caught and logged so the system keeps running.
    """
    logger.info("[events] refreshing event calendar...")
    try:
        await asyncio.gather(
            _fetch_corporate_actions(),
            _fetch_event_calendar(),
            return_exceptions=True,
        )
        _inject_rbi_dates()
        logger.info(
            "[events] calendar loaded — {} symbols cached",
            sum(1 for k in _event_cache if k != _MARKET_KEY),
        )
    except Exception as exc:
        logger.warning("[events] refresh_calendar top-level error: {}", exc)


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _fetch_corporate_actions() -> None:
    today    = date.today()
    to_date  = today + timedelta(days=7)
    from_str = today.strftime("%d-%m-%Y")
    to_str   = to_date.strftime("%d-%m-%Y")

    url = (
        "https://www.nseindia.com/api/corporates-corporateActions"
        f"?index=equities&from_date={from_str}&to_date={to_str}"
    )

    try:
        data = await nse_client.get(url)
        if not data or not isinstance(data, list):
            logger.debug("[events] corporate actions: empty or non-list response")
            return
        _parse_corporate_actions(data)
    except Exception as exc:
        logger.warning("[events] corporate actions fetch error: {}", exc)


def _parse_corporate_actions(data: list) -> None:
    """Parse the NSE corporate-actions response and insert events into the cache."""
    today_date = date.today()
    for item in data:
        symbol  = (item.get("symbol") or "").strip().upper()
        purpose = (item.get("purpose") or "").strip()
        ex_date_str = (item.get("exDate") or "").strip()
        if not symbol or not purpose:
            continue

        event_dt = _parse_dmy(ex_date_str)
        if event_dt is None:
            continue

        # Classify event importance
        purpose_lower = purpose.lower()
        is_results = any(kw in purpose_lower for kw in (
            "annual results", "quarterly results", "financial results",
            "board meeting", "earnings",
        ))
        is_corporate = any(kw in purpose_lower for kw in (
            "dividend", "bonus", "split", "rights", "buyback", "demerger",
        ))
        if not (is_results or is_corporate):
            continue

        results_today = is_results and event_dt.date() == today_date

        ev = {
            "event_type":    "RESULTS" if is_results else "CORPORATE_ACTION",
            "dt":            event_dt,
            "description":   f"{purpose} (ex-date {ex_date_str})",
            "results_today": results_today,
        }
        _event_cache.setdefault(symbol, []).append(ev)


async def _fetch_event_calendar() -> None:
    url = "https://www.nseindia.com/api/event-calendar"
    try:
        data = await nse_client.get(url)
        if not data or not isinstance(data, list):
            logger.debug("[events] event-calendar: empty or non-list response")
            return
        _parse_event_calendar(data)
    except Exception as exc:
        logger.warning("[events] event-calendar fetch error: {}", exc)


def _parse_event_calendar(data: list) -> None:
    """Parse the NSE event-calendar response."""
    today_date = date.today()
    for item in data:
        symbol  = (item.get("symbol") or "").strip().upper()
        purpose = (item.get("purpose") or "").strip()
        date_str = (item.get("date") or "").strip()
        if not symbol or not purpose:
            continue

        event_dt = _parse_iso_or_dmy(date_str)
        if event_dt is None:
            continue

        purpose_lower = purpose.lower()
        is_results = any(kw in purpose_lower for kw in (
            "results", "earnings", "financial", "board meeting",
        ))
        if not is_results:
            continue

        results_today = event_dt.date() == today_date

        ev = {
            "event_type":    "RESULTS",
            "dt":            event_dt,
            "description":   f"{purpose} ({date_str})",
            "results_today": results_today,
        }
        _event_cache.setdefault(symbol, []).append(ev)


def _inject_rbi_dates() -> None:
    """Add hard-coded RBI MPC dates as MARKET-wide HIGH-risk events."""
    macro_events: list[dict] = _event_cache.get(_MARKET_KEY, [])
    # Clear stale RBI entries before re-injecting
    macro_events = [e for e in macro_events if e.get("event_type") != "RBI_MPC"]

    today = date.today()
    for date_str in RBI_DATES:
        try:
            ev_date = datetime.strptime(date_str, "%Y-%m-%d")
            if ev_date.date() >= today:
                macro_events.append({
                    "event_type":    "RBI_MPC",
                    "dt":            ev_date,
                    "description":   f"RBI MPC policy announcement ({date_str})",
                    "results_today": ev_date.date() == today,
                })
        except ValueError:
            logger.debug("[events] bad RBI date: {}", date_str)

    _event_cache[_MARKET_KEY] = macro_events


# ── Date parsing utilities ────────────────────────────────────────────────────

def _parse_dmy(s: str) -> Optional[datetime]:
    """Parse DD-MM-YYYY (NSE ex-date format). Returns None on failure."""
    try:
        return datetime.strptime(s.strip(), "%d-%m-%Y")
    except (ValueError, AttributeError):
        return None


def _parse_iso_or_dmy(s: str) -> Optional[datetime]:
    """Try ISO YYYY-MM-DD first, then DD-MM-YYYY."""
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


__all__ = ["get_event_risk", "has_results_today", "refresh_calendar", "RBI_DATES"]
