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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import anthropic
from loguru import logger

from config import settings


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

    now_ist = datetime.now().strftime("%H:%M")
    minutes_open = _minutes_since_open()

    return {
        "symbol":   snap.symbol,
        "strategy": strategy,
        "signal":   action,
        "ltp":      snap.tick.ltp,
        "proposed_sl_pct":     signal.get("stop_loss_pct",  settings.stop_loss_pct),
        "proposed_target_pct": signal.get("target_pct",     settings.target_pct),
        "indicators": {
            "rsi_14":       round(ind.rsi_14, 2),
            "ema_20":       round(ind.ema_20, 2),
            "ema_50":       round(ind.ema_50, 2),
            "macd_hist":    round(ind.macd_hist, 4),
            "vwap":         round(ind.vwap, 2) if ind.vwap else None,
            "adx_14":       round(ind.adx_14, 2),
            "atr_14":       round(ind.atr_14, 4),
            "volume_ratio": round(ind.volume_ratio, 2),
            "bb_position":  round(ind.bb_position, 3) if hasattr(ind, "bb_position") else None,
            "price_vs_vwap": "above" if ind.vwap and snap.tick.ltp > ind.vwap else "below",
            "ema_trend":    "bullish" if ind.ema_20 > ind.ema_50 else "bearish",
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
            "time_ist":           now_ist,
            "minutes_since_open": minutes_open,
            "minutes_to_squareoff": max(0, _minutes_to_squareoff()),
        },
    }


def _minutes_since_open() -> int:
    now = datetime.now()
    open_h, open_m = 9, 15
    return max(0, (now.hour - open_h) * 60 + (now.minute - open_m))


def _minutes_to_squareoff() -> int:
    h, m = [int(x) for x in settings.squareoff_time.split(":")]
    now = datetime.now()
    sq_mins = h * 60 + m
    now_mins = now.hour * 60 + now.minute
    return sq_mins - now_mins


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
                system=_SYSTEM_PROMPT,
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
    verdict = "✅ ENTER" if d.enter else "🚫 SKIP"
    warn = f" ⚠ {d.warnings[0]}" if d.warnings else ""
    logger.info(
        "[gate] {} {} {} | conf={} size={} {}{}  {}",
        verdict, action, symbol, d.confidence, d.size_factor, d.reason, warn,
        f"({d.latency_ms}ms)",
    )
