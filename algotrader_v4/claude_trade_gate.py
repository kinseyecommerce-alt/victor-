"""
claude_trade_gate.py — per-trade AI review of the strategy agents' signals.

Called for each generated signal before order placement when the gate is on (USE_CLAUDE_TRADE_GATE)
or when today's AI allocator marked the day as unusual. The reviewer sees the setup, the market and
the strategy's validated record, and approves (optionally at reduced size) or skips.

Design rules
- The signal comes from a mechanical strategy validated on real data; losses are normal for it
  (win rate ~40 %, profit from targets 2–2.5x the risk). The reviewer vetoes only on concrete red
  flags, never on "not sure".
- Calls go through llm_router (Claude → OpenAI failover) with a forced decision tool.
- The gate never raises and never blocks on infrastructure: a context error, timeout or API failure
  approves the trade and is counted as a fail-open, visible at /gate/log and /gate/stats.
"""
from __future__ import annotations
import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist as _now_ist, minutes_since_open, minutes_to_squareoff as _mts

# ── Gate decision log (ring buffer, read by /gate/log endpoint) ───────────────
_gate_log: deque[dict] = deque(maxlen=100)
_stats: dict[str, int] = {"reviewed": 0, "approved": 0, "vetoed": 0, "downsized": 0, "fail_open": 0}

STRATEGY_PLAYBOOK = {
    "intraday": "Opening-range (09:15-09:30) breakout taken 09:31-10:00 in the direction of the 20-day trend, "
                "VWAP and EMA9/21 agreeing; stop at the other side of the range; target 2.5x the risk.",
    "scalping": "First pullback that retests a broken opening range (within 0.2 %) 09:35-12:30, trend-aligned; "
                "stop at the range midpoint; target 1.5x the risk.",
    "swing":    "Multi-day breakout / pullback in a daily uptrend, delivery (CNC) with a GTT stop.",
    "fno":      "Index option buy on a directional index move; risk limited to the premium.",
}

_SYSTEM_PROMPT = """You review trade signals from mechanical NSE strategies before the order goes out. Each strategy was validated on 45 days of real 1-minute data after costs: they win about 40 % of trades and make their money from targets 1.5-2.5x the risk, so individual losers are expected and are NOT a reason to skip. Your job is to remove the few signals with a concrete, checkable red flag and let the rest through. Approve by default.

Skip (enter=false) only for one of these, named in red_flags:
- chasing: price already more than ~1x ATR beyond the trigger level, so the stop is far and the target near
- event: results / RBI / a scheduled event for this symbol within the session
- level: a strong level (prior day high/low, round number, recent swing) sits between entry and target within ~0.5x the planned reward, against the trade
- market: NIFTY moving hard (>0.5 %) against the trade direction right now
- time: too little session left for the target (intraday after ~14:15, scalping after ~13:00)
- exhaustion: RSI beyond 80 for a buy / below 20 for a sell on a vertical move with fading volume
Use size_factor 0.5 or 0.75 (instead of skipping) when a flag is present but weak, or when the portfolio is already carrying a loss today. Missing data is never a flag. Call decide exactly once."""

DECIDE_TOOL = {
    "name": "decide",
    "description": "Approve or skip this signal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "enter":       {"type": "boolean"},
            "size_factor": {"type": "number", "enum": [0.5, 0.75, 1.0]},
            "confidence":  {"type": "integer", "description": "0-100: how sure you are the trade should be taken"},
            "red_flags":   {"type": "array", "items": {"type": "string"}},
            "reason":      {"type": "string", "description": "one crisp sentence"},
        },
        "required": ["enter", "size_factor", "confidence", "red_flags", "reason"],
        "additionalProperties": False,
    },
}


@dataclass
class GateDecision:
    confidence: int
    enter: bool
    adjusted_sl_pct: Optional[float] = None
    adjusted_target_pct: Optional[float] = None
    size_factor: float = 1.0
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    latency_ms: int = 0
    fail_open: bool = False
    cost_usd: float = 0.0


def _allow(reason: str) -> GateDecision:
    return GateDecision(confidence=60, enter=True, reason=reason, fail_open=True)


def _validated(strategy: str) -> dict:
    """The strategy's last real-data replay (the evidence the reviewer is told to respect)."""
    try:
        import replay_backtest as rb
        r = rb._LAST.get(strategy) or {}
        return {k: r.get(k) for k in ("trades", "win_rate", "net_pnl", "profit_factor", "max_drawdown")} if r else {}
    except Exception:
        return {}


def _build_context(snap, action: str, signal: dict, strategy: str) -> dict:
    """Assemble the full trade context for Claude."""
    from market_regime import regime_detector
    from risk_manager import risk_manager
    from adaptive_engine import adaptive_engine

    ind = snap.indicators
    regime = regime_detector.current_regime
    sigs   = regime_detector.current_signals

    # Adaptive stats for this strategy×symbol
    params = adaptive_engine.get_params(strategy, snap.symbol)
    wr     = params.win_rate_20 * 100 if params.win_rate_20 else 50.0
    a_win  = params.target_pct  if hasattr(params, "target_pct")  else 2.0
    a_loss = params.sl_pct      if hasattr(params, "sl_pct")      else 1.0
    # Fractional Kelly (25% of full Kelly)
    b = a_win / max(a_loss, 0.01)
    kelly_raw = (wr / 100) - (1 - wr / 100) / max(b, 0.01)
    kelly_frac = round(max(0.0, kelly_raw) * 0.25, 3)

    risk_st = risk_manager.status()

    now_ist = _now_ist().strftime("%H:%M")
    minutes_open = minutes_since_open()

    # ── Intelligence modules (sync cache reads — never block) ─────────────────
    try:
        from levels_engine import level_context as _level_ctx
        level_ctx = _level_ctx(snap.symbol, snap.tick.ltp)
    except Exception:
        level_ctx = ""

    try:
        from event_calendar import get_event_risk
        _evt = get_event_risk(snap.symbol)
        event_risk = {
            "risk_level":  _evt.get("risk_level", "NONE"),
            "event_type":  _evt.get("event_type", ""),
            "hours_until": _evt.get("hours_until"),
            "size_factor": _evt.get("size_factor", 1.0),
            "description": _evt.get("description", ""),
        }
    except Exception:
        event_risk = {"risk_level": "NONE", "size_factor": 1.0, "description": ""}

    try:
        from options_intelligence import get_cached as _opts_cached
        _opts = _opts_cached(snap.symbol)
        options_iv = {
            "atm_iv":        _opts.get("atm_iv"),
            "pcr":           _opts.get("pcr"),
            "max_pain":      _opts.get("max_pain"),
            "iv_rank":       _opts.get("iv_rank"),
            "iv_percentile": _opts.get("iv_percentile"),
            "oi_buildup":    _opts.get("oi_buildup", [])[:2],
        } if _opts else {}
    except Exception:
        options_iv = {}

    try:
        from institutional_flow import get_cached_score as _inst_cached
        _inst = _inst_cached(snap.symbol)
        institutional = {
            "score":                round(_inst.get("institutional_score", 50.0), 1),
            "delivery_pct":         _inst.get("delivery_pct"),
            "has_block_deal":       _inst.get("has_block_deal", False),
            "block_deal_direction": _inst.get("block_deal_direction"),
            "is_default":           _inst.get("is_default", True),
        }
    except Exception:
        institutional = {}

    # ── Options-specific intelligence (only populated for fno strategy) ───────
    options_advanced: dict = {}
    if strategy == "fno":
        try:
            import iv_surface as _ivs
            _surf = _ivs.get_surface(snap.symbol)
            if _surf:
                options_advanced["iv_skew"] = {
                    "atm_iv":         round(_surf.atm_iv * 100, 2),
                    "put_skew":       round(_surf.put_skew * 100, 2),
                    "call_skew":      round(_surf.call_skew * 100, 2),
                    "risk_reversal":  round(_surf.risk_reversal, 4),
                    "butterfly":      round(_surf.butterfly, 4),
                    "skew_direction": _surf.skew_direction,
                    "pcr_oi":         _surf.pcr_oi,
                }
        except Exception:
            pass
        try:
            import gamma_scalp as _gex
            _gp = _gex.get_cached_gex(snap.symbol)
            if _gp:
                options_advanced["gex"] = {
                    "regime":        _gp.regime,
                    "net_gex":       round(_gp.net_gex, 0),
                    "pin_risk":      _gp.pin_risk,
                    "pin_strike":    _gp.pin_strike,
                    "call_wall":     _gp.top_call_wall.strike if _gp.top_call_wall else None,
                    "put_wall":      _gp.top_put_wall.strike  if _gp.top_put_wall  else None,
                    "flip_pct":      _gp.flip_pct,
                }
        except Exception:
            pass
        try:
            import options_flow as _of
            _fl = _of.get_cached_flow(snap.symbol)
            if _fl:
                options_advanced["options_flow"] = {
                    "direction":     _fl.direction,
                    "score":         _fl.score,
                    "call_put_ratio": _fl.call_put_ratio,
                    "smart_bias":    _fl.smart_bias,
                    "sweep":         _fl.sweep_detected,
                    "sweep_dir":     _fl.sweep_direction,
                    "blocks":        len(_fl.block_trades),
                    "iv_spikes":     len(_fl.iv_spikes),
                }
        except Exception:
            pass

    return {
        "symbol":   snap.symbol,
        "strategy": strategy,
        "signal":   action,
        "ltp":      snap.tick.ltp,
        "proposed_sl_pct":     signal.get("stop_loss_pct",  settings.stop_loss_pct),
        "proposed_target_pct": signal.get("target_pct",     settings.target_pct),
        "indicators": {
            "rsi_14":       round(ind.rsi_14, 2),
            "ema9":         round(ind.ema9,  2),
            "ema21":        round(ind.ema21, 2),
            "ema50":        round(ind.ema50, 2),
            "macd_hist":    round(ind.macd_hist, 4),
            "vwap":         round(ind.vwap, 2) if ind.vwap else None,
            "atr_14":       round(ind.atr_14, 4),
            "bb_upper":     round(ind.bb_upper, 2),
            "bb_lower":     round(ind.bb_lower, 2),
            "volume_ratio": round(ind.volume_ratio, 2),
            "momentum":     ind.momentum,
            "trend":        ind.trend,
            "price_vs_vwap": "above" if ind.vwap and snap.tick.ltp > ind.vwap else "below",
            "ema_trend":    "bullish" if ind.ema9 > ind.ema21 > 0 else "bearish",
        },
        "multi_timeframe": snap.mtf_alignment if hasattr(snap, "mtf_alignment") else {},
        "regime": {
            "current":          regime.value,
            "vix":              round(sigs.india_vix, 1) if sigs and sigs.india_vix else None,
            "pcr":              round(sigs.pcr, 2) if sigs and sigs.pcr else None,
            "nifty_change_pct": round(sigs.nifty_1d_chg_pct, 2) if sigs and sigs.nifty_ltp else None,
            "advance_decline":  round(sigs.advance_decline, 2) if sigs and (sigs.advance_count or sigs.decline_count) else None,
        },
        "strategy_rules":     STRATEGY_PLAYBOOK.get(strategy, ""),
        "strategy_validated": _validated(strategy),
        "proposed_stop":      signal.get("stop_loss"),
        "proposed_target":    signal.get("target"),
        "trigger":            str(signal.get("trigger") or signal.get("pattern") or "")[:120],
        "portfolio": {
            "open_positions":          risk_st.get("open_positions", 0),
            "daily_pnl":               risk_st.get("daily_pnl", 0.0),
            "max_daily_loss":          settings.max_daily_loss,
            "daily_loss_used_pct":     round(
                abs(min(risk_st.get("daily_pnl", 0), 0)) / settings.max_daily_loss * 100, 1
            ),
        },
        "edge_stats": {
            "win_rate_pct":   round(wr, 1),
            "avg_win_pct":    round(a_win, 2),
            "avg_loss_pct":   round(a_loss, 2),
            "sample_size":    getattr(params, "total_trades", 0),
            "kelly_fraction": kelly_frac,
        },
        "time_context": {
            "time_ist":             now_ist,
            "minutes_since_open":   minutes_open,
            "minutes_to_squareoff": max(0, _minutes_to_squareoff()),
        },
        "key_levels":        level_ctx,
        "event_risk":        event_risk,
        "options_iv":        options_iv,
        "institutional":     institutional,
        "options_advanced":  options_advanced,
    }


def _minutes_to_squareoff() -> int:
    return _mts(settings.squareoff_time)


async def decide(ctx: dict, strategy: str, symbol: str = "", action: str = "") -> GateDecision:
    """The model call on an already-built context (live or replayed). Raises on API errors."""
    from llm_provider import llm_router
    t0 = asyncio.get_event_loop().time()
    resp = await asyncio.wait_for(
        llm_router.create(messages=[{"role": "user", "content": json.dumps(ctx, default=str)}],
                          system=_SYSTEM_PROMPT, tools=[DECIDE_TOOL], max_tokens=3000, effort="low",
                          tool_choice={"type": "tool", "name": "decide"}),
        timeout=settings.claude_gate_timeout_s)
    call = next((b for b in resp.content if b.get("type") == "tool_use" and b.get("name") == "decide"), None)
    if call is None:
        raise ValueError("reviewer returned no decision")
    d = dict(call.input or {})
    cost = 0.0
    try:
        from llm_provider import price_for
        pr, u = price_for(getattr(resp, "model", "") or ""), resp.usage
        cost = ((u.input_tokens or 0) * pr["in"] + (u.output_tokens or 0) * pr["out"]
                + (getattr(u, "cache_read_input_tokens", 0) or 0) * pr["cache_read"]) / 1_000_000
    except Exception:
        pass
    sf = d.get("size_factor", 1.0)
    sf = 1.0 if sf not in (0.5, 0.75, 1.0) else float(sf)
    return GateDecision(confidence=int(d.get("confidence", 60) or 60), enter=bool(d.get("enter", True)),
                        size_factor=sf, reason=str(d.get("reason", ""))[:240],
                        warnings=[str(x)[:80] for x in (d.get("red_flags") or [])][:5],
                        latency_ms=int((asyncio.get_event_loop().time() - t0) * 1000), cost_usd=round(cost, 5))


async def assess(snap, action: str, signal: dict, strategy: str, force: bool = False) -> GateDecision:
    """Review one live signal. `force` = the allocator marked today unusual. Never raises."""
    if not (settings.use_claude_trade_gate or force):
        return GateDecision(confidence=70, enter=True, reason="Gate disabled — rule-based approval")
    if not (settings.anthropic_api_key or settings.openai_api_key):
        return GateDecision(confidence=70, enter=True, reason="No AI key — rule-based approval")
    sym = getattr(snap, "symbol", "?")
    try:
        ctx = _build_context(snap, action, signal, strategy)
    except Exception as exc:
        logger.warning("[gate] {} context failed ({}) — allowing trade", sym, exc)
        d = _allow(f"context error: {str(exc)[:80]}")
        _record(sym, strategy, action, d); return d
    try:
        d = await decide(ctx, strategy, sym, action)
    except asyncio.TimeoutError:
        logger.warning("[gate] {} timeout after {}s — allowing trade", sym, settings.claude_gate_timeout_s)
        d = _allow("timeout — approved unreviewed")
    except Exception as exc:
        logger.warning("[gate] {} error ({}) — allowing trade", sym, exc)
        d = _allow(f"AI error — approved unreviewed: {str(exc)[:80]}")
    _record(sym, strategy, action, d)
    return d


def _record(symbol: str, strategy: str, action: str, d: GateDecision) -> None:
    if d.fail_open:
        _stats["fail_open"] += 1
    else:
        _stats["reviewed"] += 1
        _stats["approved" if d.enter else "vetoed"] += 1
        if d.enter and d.size_factor < 1.0:
            _stats["downsized"] += 1
    verdict = "ENTER" if d.enter else "SKIP"
    warn = f" ⚠ {d.warnings[0]}" if d.warnings else ""
    logger.info("[gate] {} {} {} {} | conf={} size={} {}{}  ({}ms)", "✅" if d.enter else "🚫", verdict, action,
                symbol, d.confidence, d.size_factor, d.reason, warn, d.latency_ms)
    _gate_log.appendleft({
        "time":        _now_ist().strftime("%H:%M:%S"),
        "symbol":      symbol,
        "strategy":    strategy,
        "signal":      action,
        "confidence":  d.confidence,
        "decision":    verdict,
        "size_factor": d.size_factor,
        "reason":      d.reason,
        "warnings":    d.warnings,
        "latency_ms":  d.latency_ms,
        "fail_open":   d.fail_open,
    })


def get_gate_stats() -> dict:
    return dict(_stats)


def get_gate_log(n: int = 50) -> list[dict]:
    """Return last n gate decisions (newest first)."""
    return list(_gate_log)[:n]
