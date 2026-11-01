"""
gamma_scalp.py
Gamma Exposure (GEX) analysis — identifies gamma walls, pin risk, and
whether dealers are long or short gamma (market-stabilising vs accelerating).

GEX per strike = Σ(gamma × OI × spot² × 0.01)
Positive GEX → long gamma (market stabilises near high-GEX strikes)
Negative GEX → short gamma (market accelerates through high-GEX strikes)

All public functions are synchronous, cache-backed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

RISK_FREE_RATE = 0.065
_SQRT2PI = math.sqrt(2 * math.pi)


@dataclass
class GammaWall:
    strike:       int
    gex:          float    # positive=call wall, negative=put wall
    side:         str      # "CALL_WALL" | "PUT_WALL"
    distance_pct: float    # from spot (negative = below spot)
    oi:           int


@dataclass
class GEXProfile:
    symbol:            str
    spot:              float
    net_gex:           float          # positive = long-gamma regime
    regime:            str            # "LONG_GAMMA" | "SHORT_GAMMA" | "NEUTRAL"
    top_call_wall:     Optional[GammaWall]
    top_put_wall:      Optional[GammaWall]
    zero_gamma_level:  Optional[float]
    flip_pct:          Optional[float]   # % from spot to zero-gamma
    pin_risk:          bool              # within 0.5% of a dominant wall
    pin_strike:        Optional[int]
    updated_at:        float = field(default_factory=time.time)


_cache: dict[str, GEXProfile] = {}
_TTL = 180


def get_cached_gex(symbol: str) -> Optional[GEXProfile]:
    g = _cache.get(symbol.upper())
    return g if (g and time.time() - g.updated_at < _TTL) else None


def _n(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def build_gex_profile(symbol: str, chain: list[dict], spot: float,
                      days_to_expiry: int = 7, r: float = RISK_FREE_RATE) -> GEXProfile:
    """
    Build GEX profile from NSE chain list.
    chain items: {strike, CE:{iv:%, oi:int}, PE:{iv:%, oi:int}}
    """
    T = max(days_to_expiry / 365.0, 0.5 / 365.0)
    strike_gex: dict[int, tuple[float, int]] = {}   # strike → (gex, total_oi)

    for row in chain:
        strike = int(float(row.get("strike", 0)))
        if strike <= 0:
            continue
        net = 0.0
        oi_tot = 0
        for side, sign in [("CE", +1.0), ("PE", -1.0)]:
            leg = row.get(side) or {}
            oi  = int(leg.get("oi", 0) or 0)
            iv_raw = float(leg.get("iv", 25) or 25)
            sigma  = (iv_raw / 100.0) if iv_raw > 1.0 else iv_raw
            sigma  = max(sigma, 0.05)

            d1_val = _d1(spot, strike, T, r, sigma)
            gamma  = _n(d1_val) / (spot * sigma * math.sqrt(T))
            # Dollar GEX: gamma × OI × spot² × 0.01  (per 1% spot move)
            gex    = sign * gamma * oi * (spot ** 2) * 0.01
            net   += gex
            oi_tot += oi
        strike_gex[strike] = (net, oi_tot)

    if not strike_gex:
        empty = GEXProfile(symbol=symbol, spot=spot, net_gex=0.0, regime="NEUTRAL",
                           top_call_wall=None, top_put_wall=None, zero_gamma_level=None,
                           flip_pct=None, pin_risk=False, pin_strike=None)
        _cache[symbol.upper()] = empty
        return empty

    net_gex = sum(v[0] for v in strike_gex.values())

    # Top call/put walls
    call_strikes = [(k, v[0], v[1]) for k, v in strike_gex.items() if v[0] > 0]
    put_strikes  = [(k, abs(v[0]), v[1]) for k, v in strike_gex.items() if v[0] < 0]

    def _make_wall(k, gex_abs, oi, side):
        return GammaWall(strike=k, gex=round(gex_abs, 2), side=side,
                         distance_pct=round((k - spot) / spot * 100, 2), oi=oi)

    call_wall = None
    if call_strikes:
        k, g, o = max(call_strikes, key=lambda x: x[1])
        call_wall = _make_wall(k, g, o, "CALL_WALL")

    put_wall = None
    if put_strikes:
        k, g, o = max(put_strikes, key=lambda x: x[1])
        put_wall = _make_wall(k, g, o, "PUT_WALL")

    # Zero-gamma level
    zero_level = _zero_gamma(strike_gex, spot)
    flip_pct   = round((zero_level - spot) / spot * 100, 2) if zero_level else None

    # Pin risk: within 0.5% of top wall
    pin_risk   = False
    pin_strike = None
    for w in [call_wall, put_wall]:
        if w and abs(w.distance_pct) < 0.5:
            pin_risk   = True
            pin_strike = w.strike
            break

    regime = "LONG_GAMMA" if net_gex > 0 else ("SHORT_GAMMA" if net_gex < 0 else "NEUTRAL")

    profile = GEXProfile(
        symbol=symbol, spot=spot, net_gex=round(net_gex, 2), regime=regime,
        top_call_wall=call_wall, top_put_wall=put_wall,
        zero_gamma_level=zero_level, flip_pct=flip_pct,
        pin_risk=pin_risk, pin_strike=pin_strike,
    )
    _cache[symbol.upper()] = profile
    return profile


def _zero_gamma(strike_gex: dict[int, tuple], spot: float) -> Optional[float]:
    """Approximate the price level where net GEX flips sign (walk outward from spot)."""
    if not strike_gex:
        return None
    sorted_strikes = sorted(strike_gex.keys(), key=lambda k: abs(k - spot))
    cum = 0.0
    prev_sign = None
    for k in sorted_strikes:
        cum += strike_gex[k][0]
        sign = 1 if cum >= 0 else -1
        if prev_sign is not None and sign != prev_sign:
            return float(k)
        prev_sign = sign
    return None


def gex_context(symbol: str) -> str:
    """One-line GEX summary for Claude context."""
    g = get_cached_gex(symbol)
    if not g:
        return ""
    walls = ""
    if g.top_call_wall:
        walls += f" CallWall@{g.top_call_wall.strike}({g.top_call_wall.distance_pct:+.1f}%)"
    if g.top_put_wall:
        walls += f" PutWall@{g.top_put_wall.strike}({g.top_put_wall.distance_pct:+.1f}%)"
    return (
        f"GEX_regime={g.regime} net={g.net_gex:+.0f}"
        f"{walls}"
        f"{' PIN_RISK@' + str(g.pin_strike) if g.pin_risk else ''}"
        f"{f' ZeroGamma@{g.zero_gamma_level:.0f}({g.flip_pct:+.1f}%)' if g.zero_gamma_level else ''}"
    )
