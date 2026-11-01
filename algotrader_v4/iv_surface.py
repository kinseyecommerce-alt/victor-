"""
iv_surface.py
IV smile construction, skew analysis, and term-structure for NSE options.
Built from the parsed option chain (list of {strike, CE:{iv,oi,ltp}, PE:{iv,oi,ltp}}).
All public functions are synchronous.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger


@dataclass
class SkewData:
    symbol:         str
    spot:           float
    atm_iv:         float          # annualised (0–1)
    put_skew:       float          # 25Δ put IV – ATM IV  (positive → puts expensive)
    call_skew:      float          # 25Δ call IV – ATM IV
    skew_ratio:     float          # put_iv / call_iv at 25Δ
    risk_reversal:  float          # call_iv – put_iv  at 25Δ (positive → bullish)
    butterfly:      float          # (call+put)/2 – ATM at 25Δ
    skew_direction: str            # "BEARISH" | "NEUTRAL" | "BULLISH"
    pcr_oi:         float
    gex_net:        float          # rough net gamma exposure (+ → long-gamma regime)
    smile:          dict           # strike → mid_iv (annualised)
    updated_at:     float = field(default_factory=time.time)


_cache: dict[str, SkewData] = {}
_TTL = 300


def get_surface(symbol: str) -> Optional[SkewData]:
    s = _cache.get(symbol.upper())
    return s if (s and time.time() - s.updated_at < _TTL) else None


def build_surface(symbol: str, chain: list[dict], spot: float) -> SkewData:
    """
    Build IV surface from NSE chain list.
    Each item: {"strike": int, "CE": {"iv":%, "oi":int, "ltp":float},
                               "PE": {"iv":%, "oi":int, "ltp":float}}
    IV in chain is in % (e.g. 22.5 means 22.5%) — we convert to 0–1.
    """
    smile:      dict[int, float] = {}
    call_ivs:   dict[int, float] = {}
    put_ivs:    dict[int, float] = {}
    call_oi_total = put_oi_total = 0

    for row in chain:
        strike = int(float(row.get("strike", 0)))
        if strike <= 0:
            continue
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}

        c_iv_raw = float(ce.get("iv", 0) or 0)
        p_iv_raw = float(pe.get("iv", 0) or 0)
        c_iv = c_iv_raw / 100.0 if c_iv_raw > 1.0 else c_iv_raw
        p_iv = p_iv_raw / 100.0 if p_iv_raw > 1.0 else p_iv_raw

        c_oi = int(ce.get("oi", 0) or 0)
        p_oi = int(pe.get("oi", 0) or 0)
        call_oi_total += c_oi
        put_oi_total  += p_oi

        if c_iv > 0.01:
            call_ivs[strike] = c_iv
        if p_iv > 0.01:
            put_ivs[strike] = p_iv

        mid = ((c_iv + p_iv) / 2) if (c_iv > 0.01 and p_iv > 0.01) else (c_iv or p_iv)
        if mid > 0.01:
            smile[strike] = mid

    pcr_oi = put_oi_total / max(call_oi_total, 1)

    # ATM IV
    atm = int(round(spot / 50) * 50)
    atm_iv = smile.get(atm) or _nearest(smile, atm) or 0.25

    # 25-delta proxy strikes (~5% OTM)
    otm5_up  = int(round(spot * 1.05 / 50) * 50)
    otm5_dn  = int(round(spot * 0.95 / 50) * 50)
    iv_25c   = call_ivs.get(otm5_up) or _nearest(call_ivs, otm5_up) or atm_iv
    iv_25p   = put_ivs.get(otm5_dn)  or _nearest(put_ivs,  otm5_dn)  or atm_iv

    put_skew      = round(iv_25p - atm_iv, 4)
    call_skew     = round(iv_25c - atm_iv, 4)
    skew_ratio    = round(iv_25p / max(iv_25c, 0.001), 3)
    risk_reversal = round(iv_25c - iv_25p, 4)   # positive → call premium (bullish)
    butterfly     = round((iv_25c + iv_25p) / 2 - atm_iv, 4)

    if put_skew > 0.03:
        direction = "BEARISH"     # heavy put protection buying
    elif call_skew > 0.025:
        direction = "BULLISH"
    else:
        direction = "NEUTRAL"

    # Net GEX proxy: Σ call_oi×iv² – Σ put_oi×iv²  (dimensionless relative measure)
    gex_net = sum(
        int(float((row.get("CE") or {}).get("oi", 0) or 0))
        * ((float((row.get("CE") or {}).get("iv", 25) or 25) / 100) ** 2)
        - int(float((row.get("PE") or {}).get("oi", 0) or 0))
        * ((float((row.get("PE") or {}).get("iv", 25) or 25) / 100) ** 2)
        for row in chain if int(float(row.get("strike", 0))) > 0
    )

    sd = SkewData(
        symbol=symbol, spot=spot, atm_iv=round(atm_iv, 4),
        put_skew=put_skew, call_skew=call_skew, skew_ratio=skew_ratio,
        risk_reversal=risk_reversal, butterfly=butterfly,
        skew_direction=direction, pcr_oi=round(pcr_oi, 3),
        gex_net=round(gex_net, 2), smile=smile,
    )
    _cache[symbol.upper()] = sd
    return sd


def _nearest(d: dict, target: int) -> Optional[float]:
    return min(d.items(), key=lambda x: abs(x[0] - target))[1] if d else None


def skew_context(symbol: str) -> str:
    """One-line summary for Claude context. Empty string if no data."""
    s = get_surface(symbol)
    if not s:
        return ""
    return (
        f"skew={s.skew_direction} ATM_IV={s.atm_iv:.1%} "
        f"RR={s.risk_reversal:+.3f} PutSkew={s.put_skew:+.3f} "
        f"PCR={s.pcr_oi:.2f} GEX={'LONG' if s.gex_net > 0 else 'SHORT'}"
    )
