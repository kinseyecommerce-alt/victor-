"""
claude_trade_gate.py
Per-trade Claude intelligence gate using Sonnet.

Called for EVERY generated signal before order placement.
Claude assesses setup quality, can approve/veto/modify trade parameters.
Design principle: never block a genuinely good setup, never let a bad one through.

Fallback on any API error → allow the trade (never block due to infra issues).
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from loguru import logger

from config import settings
from ist_clock import now_ist as _now_ist, minutes_since_open, minutes_to_squareoff as _mts

# ── Gate decision log (ring buffer, read by /gate/log endpoint) ───────────────
_gate_log: deque[dict] = deque(maxlen=100)


_SYSTEM_PROMPT = """You are an elite NSE/BSE quantitative trader with decades of experience.
You assess individual trade setups and decide: execute as-is, execute with adjustments, or skip.

GOAL: capture every high-quality opportunity, protect capital from every weak setup.

RULES:
1. "enter": true  → execute the trade (optionally with your adjustments)
2. "enter": false → skip this trade
3. Never block a trade purely on missing data — if data is incomplete, approve with tighter SL
4. Adjust SL/target/size_factor only when clearly justified by the data
5. Penalise: regime mismatch, low volume, overbought/oversold extremes, approaching key levels
6. Reward: multi-TF alignment, high volume, clean trend, good R:R, healthy portfolio heat

Output ONLY valid JSON — no markdown, no explanation outside the JSON:
{
  "confidence": 0-100,
  "enter": true|false,
  "adjusted_sl_pct": <float or null>,
  "adjusted_target_pct": <float or null>,
  "size_factor": <0.25|0.5|0.75|1.0 — default 1.0>,
  "reason": "<one crisp sentence>",
  "warnings": ["<string>", ...]
}"""


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


_ALLOW_ON_ERROR = GateDecision(confidence=60, enter=True, reason="API fallback — rule-based approval")

_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


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
            "current":         regime.value,
            "vix":             round(sigs.vix, 1) if sigs else None,
            "pcr":             round(sigs.pcr, 2) if sigs else None,
            "nifty_direction": sigs.nifty_direction if sigs and hasattr(sigs, "nifty_direction") else None,
            "breadth":         round(sigs.breadth, 2) if sigs else None,
        },
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


async def assess(snap, action: str, signal: dict, strategy: str) -> GateDecision:
    """
    Ask Claude to assess this trade setup.
    Always returns a GateDecision — never raises.
    """
    if not settings.anthropic_api_key or not settings.use_claude_trade_gate:
        return GateDecision(confidence=70, enter=True, reason="Gate disabled — rule-based approval")

    t0 = asyncio.get_event_loop().time()
    ctx = _build_context(snap, action, signal, strategy)

    try:
        resp = await asyncio.wait_for(
            _get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": json.dumps(ctx)}],
            ),
            timeout=4.0,
        )
        raw = resp.content[0].text.strip().lstrip("```json").rstrip("```").strip()
        d   = json.loads(raw)
        latency = int((asyncio.get_event_loop().time() - t0) * 1000)

        decision = GateDecision(
            confidence=int(d.get("confidence", 60)),
            enter=bool(d.get("enter", True)),
            adjusted_sl_pct=d.get("adjusted_sl_pct"),
            adjusted_target_pct=d.get("adjusted_target_pct"),
            size_factor=float(d.get("size_factor", 1.0)),
            reason=d.get("reason", ""),
            warnings=d.get("warnings", []),
            latency_ms=latency,
        )

        _log(snap.symbol, strategy, action, decision, ctx["indicators"]["rsi_14"])
        return decision

    except asyncio.TimeoutError:
        logger.warning("[gate] {} timeout — allowing trade", snap.symbol)
        return _ALLOW_ON_ERROR
    except Exception as exc:
        logger.warning("[gate] {} error ({}) — allowing trade", snap.symbol, exc)
        return _ALLOW_ON_ERROR


def _log(symbol: str, strategy: str, action: str, d: GateDecision, rsi: float) -> None:
    verdict = "ENTER" if d.enter else "SKIP"
    warn = f" ⚠ {d.warnings[0]}" if d.warnings else ""
    logger.info(
        "[gate] {} {} {} {} | conf={} size={} {}{}  ({}ms)",
        "✅" if d.enter else "🚫", verdict, action, symbol,
        d.confidence, d.size_factor, d.reason, warn, d.latency_ms,
    )
    _gate_log.appendleft({
        "time":        datetime.now().strftime("%H:%M:%S"),
        "symbol":      symbol,
        "strategy":    strategy,
        "signal":      action,
        "confidence":  d.confidence,
        "decision":    verdict,
        "size_factor": d.size_factor,
        "reason":      d.reason,
        "warnings":    d.warnings,
        "latency_ms":  d.latency_ms,
    })


def get_gate_log(n: int = 50) -> list[dict]:
    """Return last n gate decisions (newest first)."""
    return list(_gate_log)[:n]
