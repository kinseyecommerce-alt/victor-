"""
mcx_instruments.py — Resolve MCX base symbols to live broker contracts.

Our universe (mcx_universe.py) uses base names like CRUDEOIL / GOLDM. The broker
(Zerodha Kite) trades dated futures — CRUDEOIL25JULFUT etc. — each with its own
instrument_token. This module bridges the two: for each base name it picks the
nearest non-expired monthly futures contract from the broker's instrument dump
and returns its tradingsymbol + instrument_token (needed for WebSocket subscribe,
REST quotes and historical data).

All data comes from the broker — `kite_client.get_instruments("MCX")`. If the
broker session is not established the resolver returns an empty mapping (the
caller then has no market data, which is the intended "broker-only" behaviour).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from loguru import logger

from kite_client import kite_client
from ist_clock import now_ist


@dataclass(frozen=True)
class ResolvedContract:
    base:             str
    tradingsymbol:    str
    instrument_token: int
    exchange:         str
    expiry:           Optional[date]
    lot_size:         int


# base → resolved contract (cached until refresh)
_cache: dict[str, ResolvedContract] = {}


def _to_date(v) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and v:
        try:
            return datetime.strptime(v[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def resolve(bases: list[str], refresh: bool = False) -> dict[str, ResolvedContract]:
    """Map each base name to its nearest non-expired MCX futures contract.

    Reads instruments from the broker. Returns only the bases that resolved;
    missing/unconnected bases are simply absent from the result.
    """
    if not refresh:
        cached = {b: _cache[b] for b in bases if b in _cache}
        if len(cached) == len(bases):
            return cached

    if not kite_client.is_connected():
        logger.warning("[mcx_instruments] broker not connected — no contracts resolved")
        return {b: _cache[b] for b in bases if b in _cache}

    try:
        instruments = kite_client.get_instruments("MCX")
    except Exception as exc:
        logger.warning("[mcx_instruments] instrument fetch failed: {}", exc)
        return {b: _cache[b] for b in bases if b in _cache}

    today = now_ist().date()
    want  = set(bases)
    # base → list of (expiry, instrument) for FUT contracts not yet expired
    candidates: dict[str, list[tuple[date, dict]]] = {b: [] for b in bases}
    for inst in instruments:
        if inst.get("instrument_type") != "FUT":
            continue
        name = inst.get("name") or ""
        if name not in want:
            continue
        exp = _to_date(inst.get("expiry"))
        if exp is None or exp < today:
            continue
        candidates[name].append((exp, inst))

    resolved: dict[str, ResolvedContract] = {}
    for base, rows in candidates.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])           # nearest expiry first
        exp, inst = rows[0]
        rc = ResolvedContract(
            base=base,
            tradingsymbol=inst.get("tradingsymbol", base),
            instrument_token=int(inst.get("instrument_token", 0)),
            exchange=inst.get("exchange", "MCX"),
            expiry=exp,
            lot_size=int(inst.get("lot_size", 0)) or 1,
        )
        resolved[base] = rc
        _cache[base] = rc

    missing = want - set(resolved) - set(_cache)
    if missing:
        logger.warning("[mcx_instruments] no live contract for: {}", sorted(missing))
    logger.info("[mcx_instruments] resolved {}/{} MCX contracts from broker",
                len(resolved), len(bases))
    return {b: _cache[b] for b in bases if b in _cache}


def token_map(bases: list[str]) -> dict[str, int]:
    """base → instrument_token for resolved contracts."""
    return {b: rc.instrument_token for b, rc in resolve(bases).items()}


def tradingsymbol_map(bases: list[str]) -> dict[str, str]:
    """base → live futures tradingsymbol for resolved contracts."""
    return {b: rc.tradingsymbol for b, rc in resolve(bases).items()}


def clear_cache() -> None:
    _cache.clear()
