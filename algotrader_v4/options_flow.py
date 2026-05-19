"""
options_flow.py
Unusual options activity and smart-money flow detection.

Signals detected:
  - Unusual volume spikes (volume/OI > 5×)
  - Block trades (volume ≥ 500 lots at a strike)
  - Call/put sweeps (3+ strikes with unusual activity same direction)
  - IV spikes (IV jumps >15% vs recent average)
  - Net OI momentum (call vs put OI accumulation)

All public functions are synchronous, cache-backed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from loguru import logger

UNUSUAL_VOL_RATIO   = 5.0     # volume/OI > 5× is unusual
BLOCK_LOT_MIN       = 500     # lots ≥ 500 = block trade
IV_SPIKE_THRESHOLD  = 0.15    # 15% relative IV jump
_TTL = 120                    # 2-minute cache


@dataclass
class OptionsFlow:
    symbol:          str
    direction:       Literal["BULLISH", "BEARISH", "NEUTRAL"]
    score:           int              # 0–100
    total_call_vol:  int
    total_put_vol:   int
    call_put_ratio:  float
    unusual_calls:   list[dict]       # top unusual CE strikes
    unusual_puts:    list[dict]       # top unusual PE strikes
    block_trades:    list[dict]       # large volume strikes
    sweep_detected:  bool
    sweep_direction: Optional[str]
    iv_spikes:       list[dict]       # strikes with IV spike
    smart_bias:      str              # "BULLISH" | "BEARISH" | "NEUTRAL"
    oi_momentum:     float            # +ve = call OI growing faster
    updated_at:      float = field(default_factory=time.time)


_cache:      dict[str, OptionsFlow] = {}
_iv_history: dict[str, dict[str, list[float]]] = {}   # sym → "strike_CE" → [iv, ...]


def get_cached_flow(symbol: str) -> Optional[OptionsFlow]:
    f = _cache.get(symbol.upper())
    return f if (f and time.time() - f.updated_at < _TTL) else None


def analyze_flow(symbol: str, chain: list[dict], spot: float) -> OptionsFlow:
    """
    Analyze option chain for unusual flow.
    chain items: {strike, CE:{oi, volume, iv, ltp}, PE:{oi, volume, iv, ltp}}
    """
    symbol = symbol.upper()
    unusual_calls: list[dict] = []
    unusual_puts:  list[dict] = []
    blocks:        list[dict] = []
    iv_spikes:     list[dict] = []
    total_cv = total_pv = 0
    oi_call_delta = oi_put_delta = 0.0

    sym_hist = _iv_history.setdefault(symbol, {})

    for row in chain:
        strike = int(float(row.get("strike", 0)))
        if not strike:
            continue
        for side, is_call in [("CE", True), ("PE", False)]:
            leg = row.get(side) or {}
            oi  = int(leg.get("oi", 0) or 0)
            vol = int(leg.get("volume", 0) or 0)
            iv_raw = float(leg.get("iv", 25) or 25)
            iv  = (iv_raw / 100.0) if iv_raw > 1.0 else iv_raw
            ltp = float(leg.get("ltp", 0) or 0)
            oi_chg = int(leg.get("oi_change", 0) or 0)

            if is_call:
                total_cv += vol
                oi_call_delta += oi_chg
            else:
                total_pv += vol
                oi_put_delta += oi_chg

            # Unusual volume
            if oi >= 100 and vol > 0 and vol / oi > UNUSUAL_VOL_RATIO:
                entry = dict(strike=strike, oi=oi, vol=vol,
                             ratio=round(vol / oi, 1), ltp=ltp)
                (unusual_calls if is_call else unusual_puts).append(entry)

            # Block trades
            if vol >= BLOCK_LOT_MIN:
                blocks.append(dict(strike=strike, side=side, lots=vol,
                                   ltp=ltp, iv=round(iv * 100, 1)))

            # IV spike
            hkey = f"{strike}_{side}"
            hist = sym_hist.setdefault(hkey, [])
            hist.append(iv)
            if len(hist) > 20:
                hist.pop(0)
            if len(hist) >= 4:
                avg = sum(hist[:-1]) / (len(hist) - 1)
                if avg > 0 and abs(iv - avg) / avg > IV_SPIKE_THRESHOLD:
                    iv_spikes.append(dict(strike=strike, side=side,
                                          iv_now=round(iv * 100, 1),
                                          iv_avg=round(avg * 100, 1),
                                          change_pct=round((iv - avg) / avg * 100, 1)))

    # Sweep: 3+ unusual strikes in same direction
    sweep_detected = len(unusual_calls) >= 3 or len(unusual_puts) >= 3
    sweep_direction = None
    if sweep_detected:
        sweep_direction = "BULLISH" if len(unusual_calls) >= len(unusual_puts) else "BEARISH"

    # Direction & score
    cpr = total_cv / max(total_pv, 1)
    score = 50
    if cpr > 1.35:     score += 15; direction = "BULLISH"
    elif cpr < 0.75:   score -= 15; direction = "BEARISH"
    else:               direction = "NEUTRAL"

    if oi_call_delta > oi_put_delta:   score += 8
    else:                               score -= 8

    if sweep_detected:
        score += 10
        if sweep_direction:
            direction = sweep_direction

    score = max(0, min(100, score))

    # Smart money bias from blocks
    blk_calls = sum(1 for b in blocks if b["side"] == "CE")
    blk_puts  = sum(1 for b in blocks if b["side"] == "PE")
    if blk_calls > blk_puts:     smart_bias = "BULLISH"
    elif blk_puts > blk_calls:   smart_bias = "BEARISH"
    else:                         smart_bias = "NEUTRAL"

    oi_mom = round((oi_call_delta - oi_put_delta) / max(abs(oi_call_delta) + abs(oi_put_delta), 1), 3)

    flow = OptionsFlow(
        symbol=symbol, direction=direction, score=score,
        total_call_vol=total_cv, total_put_vol=total_pv,
        call_put_ratio=round(cpr, 3),
        unusual_calls=sorted(unusual_calls, key=lambda x: x["ratio"], reverse=True)[:4],
        unusual_puts=sorted(unusual_puts, key=lambda x: x["ratio"], reverse=True)[:4],
        block_trades=sorted(blocks, key=lambda x: x["lots"], reverse=True)[:5],
        sweep_detected=sweep_detected, sweep_direction=sweep_direction,
        iv_spikes=iv_spikes[:4], smart_bias=smart_bias, oi_momentum=oi_mom,
    )
    _cache[symbol] = flow
    return flow


def flow_context(symbol: str) -> str:
    """One-line summary for Claude context."""
    f = get_cached_flow(symbol)
    if not f:
        return ""
    return (
        f"flow={f.direction}({f.score}) C/P={f.call_put_ratio:.2f} "
        f"smart={f.smart_bias} blocks={len(f.block_trades)} "
        f"{'SWEEP_' + (f.sweep_direction or '') + ' ' if f.sweep_detected else ''}"
        f"IV_spikes={len(f.iv_spikes)}"
    )
