"""
opportunity_scanner.py — where is the money today?

Every AI cycle starts with this scan instead of staring at the 20-symbol watchlist. It walks
the NIFTY 100 (plus anything already streaming) and the two index underlyings and ranks concrete,
tradeable set-ups by quality, in any direction and any instrument the account allows:

  gap_and_go        opened ≥1.5% away from yesterday's close and is still running that way on volume
  breakout_20d      trading through the 20-day high (or low) on volume — intraday momentum or a swing
  trend_pullback    daily uptrend resting on the 20-day EMA (buy) / downtrend failing at it (swing put)
  mean_reversion    daily RSI washed out (≤28) or stretched (≥75) and far from the 20-day EMA
  relative_strength leading / lagging the index by a wide margin today while the index trends
  index_premium     NIFTY/BANKNIFTY implied vol rich and the market ranging → credit spread
  index_momentum    index trending hard with cheap options → buy the option, or trade the future

Each row carries the play, the direction, the tool the AI should use, a score, the reasons, and
stop/target ideas from ATR and structure. Filters: results/RBI event risk, the NIFTY 200-day trend
filter, pruned strategies, instruments switched off in settings, and dead or wild volatility.

The scan is a shortlist, not a signal: the AI still calls get_symbol_detail / get_market_context
and the bear-case review before anything is placed. Results are cached for 90 s.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

_cache: dict = {"at": 0.0, "result": None}
_lock = threading.Lock()
CACHE_S = 90
TIME_BUDGET_S = 25          # first scan of the day fetches daily history; stop early rather than hang


def universe() -> list[str]:
    from nifty100 import NIFTY_100
    from tick_engine import tick_engine
    syms = list(dict.fromkeys(NIFTY_100 + [s for s in tick_engine.symbols() if " " not in s]))
    return syms


def _live_snapshot(symbols: list[str]) -> dict[str, dict]:
    """ltp/open/high/low/prev_close/volume per symbol: Kite quote() batches, or the tick engine."""
    from tick_engine import tick_engine
    out: dict[str, dict] = {}
    from kite_client import kite_client
    if settings.uses_kite_market_data and kite_client.has_session():
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i + 100]
            try:
                raw = kite_client.quote_kite([f"NSE:{s}" for s in chunk])
            except Exception as exc:
                logger.warning("[scan] quote batch failed: {}", exc)
                raw = {}
            for s in chunk:
                d = raw.get(f"NSE:{s}")
                if not d or not d.get("last_price"):
                    continue
                o = d.get("ohlc") or {}
                out[s] = {"ltp": float(d["last_price"]), "open": float(o.get("open") or d["last_price"]),
                          "high": float(o.get("high") or d["last_price"]), "low": float(o.get("low") or d["last_price"]),
                          "prev_close": float(o.get("close") or d["last_price"]), "volume": int(d.get("volume_traded") or 0)}
    for s in symbols:
        if s in out:
            continue
        tick, _ = tick_engine.latest(s)
        if tick and tick.ltp > 0:
            pc = tick.ltp / (1 + tick.change_pct / 100) if tick.change_pct not in (None, -100) else tick.ltp
            out[s] = {"ltp": tick.ltp, "open": tick.open or tick.ltp, "high": tick.high or tick.ltp,
                      "low": tick.low or tick.ltp, "prev_close": pc, "volume": int(tick.volume or 0)}
    return out


def _index_change() -> Optional[float]:
    from tick_engine import tick_engine
    t, _ = tick_engine.latest("NIFTY 50")
    return round(t.change_pct, 2) if t else None


def _regime() -> str:
    try:
        from market_regime import regime_detector
        return regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _pruned() -> set[str]:
    try:
        from performance import performance
        return set(performance.pruned)
    except Exception:
        return set()


def _event_penalty(symbol: str) -> tuple[Optional[int], str]:
    """(penalty or None to skip, note)."""
    try:
        from event_calendar import get_event_risk
        ev = get_event_risk(symbol)
    except Exception:
        return 0, ""
    if ev.get("size_factor", 1.0) == 0.0:
        return None, f"event block: {ev.get('event_type')}"
    if ev.get("risk_level") in ("HIGH", "MEDIUM"):
        return -1, f"{ev.get('risk_level', '').lower()} event risk ({ev.get('event_type')})"
    return 0, ""


def _stock_plays(sym: str, q: dict, dv: Optional[dict], nifty_chg: Optional[float], risk_off: bool,
                 pruned: set[str], regime: str) -> list[dict]:
    plays: list[dict] = []
    ltp, pc = q["ltp"], q["prev_close"]
    if ltp <= 0 or pc <= 0:
        return plays
    chg = (ltp - pc) / pc * 100
    gap = (q["open"] - pc) / pc * 100
    day_range = (q["high"] - q["low"]) / pc * 100
    volx = (dv or {}).get("volume_vs_20d_avg") or None
    atr_pct = (dv or {}).get("atr_pct")
    rsi = (dv or {}).get("rsi14")
    trend = (dv or {}).get("trend", "UNKNOWN")
    e20 = (dv or {}).get("ema20")
    hi20, lo20 = (dv or {}).get("high_20d"), (dv or {}).get("low_20d")
    if atr_pct is not None and (atr_pct < 0.7 or atr_pct > 7):
        return plays                                      # too dull to pay costs, or too wild to stop
    pen, pen_note = _event_penalty(sym)
    if pen is None:
        return plays
    vol_bonus = 2 if (volx or 0) >= 2.5 else 1 if (volx or 0) >= 1.5 else 0

    def row(play, direction, instrument, tool, base, why, stop_pct, target_pct, strategy):
        if strategy in pruned:
            return
        score = base + vol_bonus + pen
        if direction == "LONG" and risk_off:
            score -= 2; why = why + "; NIFTY below 200-day EMA (risk-off): longs half size"
        if direction == "SHORT" and regime == "BULL_TREND":
            score -= 1
        if direction == "LONG" and regime == "BULL_TREND":
            score += 1
        if direction == "SHORT" and regime == "BEAR_TREND":
            score += 1
        if pen_note:
            why += "; " + pen_note
        plays.append({"symbol": sym, "play": play, "direction": direction, "instrument": instrument, "tool": tool,
                      "score": round(score, 1), "ltp": round(ltp, 2), "change_pct": round(chg, 2),
                      "volume_x_20d": round(volx, 2) if volx else None, "atr_pct": atr_pct, "rsi14": rsi,
                      "daily_trend": trend, "stop_idea": round(ltp * (1 - stop_pct / 100), 2) if direction == "LONG" else round(ltp * (1 + stop_pct / 100), 2),
                      "target_idea": round(ltp * (1 + target_pct / 100), 2) if direction == "LONG" else round(ltp * (1 - target_pct / 100), 2),
                      "why": why})

    a = max(atr_pct or 1.5, 0.8)
    # gap and go: opened away from yesterday, still running that way, with volume
    if abs(gap) >= 1.5 and vol_bonus >= 1 and day_range >= 0.6:
        if gap > 0 and ltp >= q["open"] and ltp >= q["high"] * 0.995:
            row("gap_and_go", "LONG", "equity intraday", "place_trade BUY", 3,
                f"gapped up {gap:.1f}% and holding near the high on {volx:.1f}x volume", a * 0.6, a * 1.2, "ai")
        elif gap < 0 and ltp <= q["open"] and ltp <= q["low"] * 1.005:
            row("gap_and_go", "SHORT", "equity intraday", "place_trade SELL", 3,
                f"gapped down {gap:.1f}% and pressing the low on {volx:.1f}x volume", a * 0.6, a * 1.2, "ai")
    # 20-day breakout / breakdown
    if hi20 and ltp > hi20 and vol_bonus >= 1:
        row("breakout_20d", "LONG", "equity intraday or swing", "place_trade BUY (intraday) / place_swing_trade (hold days)", 3,
            f"through the 20-day high {hi20:g} on {volx:.1f}x volume, daily trend {trend}", a * 0.8, a * 1.6, "ai_swing")
    if lo20 and ltp < lo20 and vol_bonus >= 1:
        row("breakdown_20d", "SHORT", "equity intraday or swing put", "place_trade SELL (intraday) / place_swing_put (F&O stock, hold days)", 3,
            f"through the 20-day low {lo20:g} on {volx:.1f}x volume, daily trend {trend}", a * 0.8, a * 1.6, "ai_swing_opt")
    # trend pullback to the 20-day EMA
    if e20 and trend == "UP" and abs(ltp - e20) / e20 * 100 <= 1.0 and rsi and 42 <= rsi <= 65:
        row("trend_pullback", "LONG", "equity swing", "place_swing_trade", 2,
            f"daily uptrend resting on EMA20 {e20:g}, RSI {rsi:.0f}", a * 1.5, a * 3.0, "ai_swing")
    if e20 and trend == "DOWN" and abs(ltp - e20) / e20 * 100 <= 1.0 and rsi and 35 <= rsi <= 58:
        row("trend_pullback", "SHORT", "swing put", "place_swing_put", 2,
            f"daily downtrend failing at EMA20 {e20:g}, RSI {rsi:.0f}", a * 1.5, a * 3.0, "ai_swing_opt")
    # mean reversion
    if rsi is not None and e20:
        dist = (ltp - e20) / e20 * 100
        if rsi <= 28 and dist <= -2 * a:
            row("mean_reversion", "LONG", "equity intraday", "place_trade BUY", 2,
                f"washed out: RSI {rsi:.0f}, {abs(dist):.1f}% below EMA20", a * 0.7, a * 1.1, "ai")
        if rsi >= 75 and dist >= 2 * a:
            row("mean_reversion", "SHORT", "equity intraday", "place_trade SELL", 2,
                f"stretched: RSI {rsi:.0f}, {dist:.1f}% above EMA20", a * 0.7, a * 1.1, "ai")
    # relative strength vs the index on a trending day
    if nifty_chg is not None and abs(nifty_chg) >= 0.4:
        rel = chg - nifty_chg
        if rel >= 1.5 and nifty_chg > 0 and vol_bonus >= 1:
            row("relative_strength", "LONG", "equity intraday", "place_trade BUY", 1,
                f"up {chg:.1f}% vs NIFTY {nifty_chg:+.1f}% — leader on a trending day", a * 0.6, a * 1.0, "ai")
        if rel <= -1.5 and nifty_chg < 0 and vol_bonus >= 1:
            row("relative_strength", "SHORT", "equity intraday", "place_trade SELL", 1,
                f"down {chg:.1f}% vs NIFTY {nifty_chg:+.1f}% — laggard on a down day", a * 0.6, a * 1.0, "ai")
    return plays


def _index_plays(regime: str, nifty_chg: Optional[float], pruned: set[str]) -> list[dict]:
    plays: list[dict] = []
    try:
        from options_intelligence import get_cached
    except Exception:
        return plays
    for und in ("NIFTY", "BANKNIFTY"):
        ctx = get_cached(und) or {}
        ivr = ctx.get("iv_rank")
        pcr = ctx.get("pcr")
        note_iv = f"IV rank {ivr:.0f}" if isinstance(ivr, (int, float)) else "IV rank unknown"
        if settings.ai_agent_option_selling_enabled and "ai_spread" not in pruned and isinstance(ivr, (int, float)) and ivr >= 55 \
                and regime in ("RANGING", "UNKNOWN", "BULL_TREND", "BEAR_TREND"):
            view = "BULLISH" if (regime == "BULL_TREND" or (nifty_chg or 0) > 0) else "BEARISH"
            plays.append({"symbol": und, "play": "index_premium", "direction": "NEUTRAL-" + view, "instrument": "credit spread",
                          "tool": f"place_option_spread {view}", "score": 2 + (1 if regime == "RANGING" else 0) + (1 if ivr >= 70 else 0),
                          "why": f"{note_iv} — premium is rich; sell a {('bull put' if view == 'BULLISH' else 'bear call')} spread 1-2 strikes OTM"
                                 + (f", PCR {pcr}" if pcr else ""), "regime": regime})
        if regime in ("BULL_TREND", "BEAR_TREND") and nifty_chg is not None and abs(nifty_chg) >= 0.6:
            side = "CE" if regime == "BULL_TREND" else "PE"
            if "ai_opt" not in pruned and (not isinstance(ivr, (int, float)) or ivr <= 45):
                plays.append({"symbol": und, "play": "index_momentum", "direction": "LONG" if side == "CE" else "SHORT",
                              "instrument": "index option (buy)", "tool": f"place_option_trade {side}",
                              "score": 2 + (1 if abs(nifty_chg) >= 1.0 else 0),
                              "why": f"{regime.replace('_', ' ').lower()}, NIFTY {nifty_chg:+.1f}% today, {note_iv} (options not expensive)", "regime": regime})
            if settings.ai_agent_futures_enabled and "ai_fut" not in pruned:
                plays.append({"symbol": und, "play": "index_momentum", "direction": "LONG" if side == "CE" else "SHORT",
                              "instrument": "index future", "tool": f"place_future_trade {'BUY' if side == 'CE' else 'SELL'}",
                              "score": 2 + (1 if abs(nifty_chg) >= 1.0 else 0),
                              "why": f"{regime.replace('_', ' ').lower()}, NIFTY {nifty_chg:+.1f}% — leveraged, tight stop", "regime": regime})
    return plays


def scan(top: int = 12, focus: str = "all", force: bool = False) -> dict:
    """Ranked opportunities. focus: all | intraday | swing | options."""
    with _lock:
        if not force and _cache["result"] and time.time() - _cache["at"] < CACHE_S:
            res = _cache["result"]
        else:
            res = _scan()
            if res["quoted"] > 0:                 # an empty scan (no data yet) must not be cached for 90 s
                _cache.update(at=time.time(), result=res)
    rows = res["opportunities"]
    if focus == "intraday":
        rows = [r for r in rows if "intraday" in r["instrument"] or r["instrument"] == "index future"]
    elif focus == "swing":
        rows = [r for r in rows if "swing" in r["instrument"] or "put" in r["instrument"]]
    elif focus == "options":
        rows = [r for r in rows if "option" in r["instrument"] or "spread" in r["instrument"]]
    return {**res, "opportunities": rows[:top], "total_found": len(res["opportunities"])}


def _scan() -> dict:
    from daily_data import daily_view
    from market_filter import state as trend_state
    t0 = time.time()
    syms = universe()
    snap = _live_snapshot(syms)
    nifty_chg = _index_change()
    regime = _regime()
    tf = trend_state()
    risk_off = tf.get("state") == "RISK_OFF"
    pruned = _pruned()
    plays: list[dict] = []
    scanned = 0
    truncated = False
    for s in syms:
        q = snap.get(s)
        if not q:
            continue
        if time.time() - t0 > TIME_BUDGET_S:
            truncated = True
            break
        try:
            dv = daily_view(s, candles=0)
        except Exception:
            dv = None
        scanned += 1
        plays.extend(_stock_plays(s, q, dv, nifty_chg, risk_off, pruned, regime))
    plays.extend(_index_plays(regime, nifty_chg, pruned))
    plays.sort(key=lambda r: -r["score"])
    best = {}
    for r in plays:                                   # one row per symbol+play; keep the best-scoring direction
        best.setdefault((r["symbol"], r["play"]), r)
    rows = list(best.values())
    rows.sort(key=lambda r: -r["score"])
    return {"at": now_ist().strftime("%H:%M:%S"), "regime": regime, "trend_filter": tf.get("state"),
            "nifty_change_pct": nifty_chg, "scanned": scanned, "universe": len(syms), "quoted": len(snap),
            "truncated": truncated, "took_s": round(time.time() - t0, 1), "opportunities": rows,
            "note": "shortlist only — confirm with get_symbol_detail / get_market_context; the bear-case review runs on every entry"}


def warm() -> None:
    """Prime the daily views for the universe (run in the morning so the first scan is fast)."""
    try:
        scan(force=True)
    except Exception as exc:
        logger.warning("[scan] warm-up failed: {}", exc)
