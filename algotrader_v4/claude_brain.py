"""
claude_brain.py — the central Claude "brain" powering the MCX trading agents.

A single async Anthropic-API client that the master agent consults on its
periodic regime review (OFF the order hot path — never per-tick, never per-order).
Given a full market/portfolio snapshot it returns a *risk posture*: the prevailing
regime, a global size_factor, per-agent directives (run / reduce_size / pause) and
an optional halt. The master publishes this to the agent bus (TOPIC_REGIME); the
coordinator applies the size_factor to every entry. This keeps LLM intelligence in
the loop without adding latency to order placement.

Design principles
-----------------
* Off the hot path. Consulted ~once/minute, with a generous timeout.
* Graceful degradation. With no API key (or on any error/timeout) it returns a
  deterministic rule-based posture so the whole app runs fully offline.
* One shared AsyncAnthropic client, lazily created.
* Model: claude-opus-4-8 with adaptive thinking (settings-configurable).
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import anthropic
from loguru import logger

from config import settings

# Stable agent registry keys the brain issues directives for.
AGENT_KEYS = ("intraday", "scalping", "swing", "fno")
_VALID_ACTIONS = ("run", "reduce_size", "pause")

# Ring buffer of recent postures (read by /brain/log).
_brain_log: deque[dict] = deque(maxlen=60)


SYSTEM_PROMPT = """You are the MASTER TRADING INTELLIGENCE ("the brain") for an MCX
(Multi Commodity Exchange of India) commodity-futures algorithmic trading system.
You supervise four specialised agents that trade bullion, energy and base-metal futures:

  • intraday  — MIS intraday trend/breakout
  • scalping  — MIS fast mean-reversion / momentum bursts
  • swing     — NRML positional / overnight
  • fno       — MCX options & calendar/inter-commodity spreads (NRML)

You are consulted about once a minute — OFF the order hot path. You do NOT approve
individual trades. You set the RISK POSTURE the whole fleet trades under until the
next review: the regime, a global size multiplier, and per-agent directives.

Your dual mandate: CAPTURE EVERY GENUINE OPPORTUNITY. PREVENT EVERY AVOIDABLE LOSS.

Return ONLY valid JSON — no markdown, no code fences, no prose outside the JSON:
{
  "regime": "trending_up|trending_down|ranging|volatile",
  "regime_confidence": 0-100,
  "size_factor": 0.25|0.5|0.75|1.0,
  "agent_directives": {
    "intraday": {"action": "run|reduce_size|pause", "reason": "<short>"},
    "scalping": {"action": "run|reduce_size|pause", "reason": "<short>"},
    "swing":    {"action": "run|reduce_size|pause", "reason": "<short>"},
    "fno":      {"action": "run|reduce_size|pause", "reason": "<short>"}
  },
  "halt_new_trades": false,
  "opportunity_alert": "<null or one sentence about a specific opportunity window>",
  "reasoning": "<one crisp sentence on the current commodity market state and edge>"
}

RULES
- size_factor 1.0 only in a clean, well-confirmed regime; 0.25–0.5 when volatile,
  news-heavy, or when daily P&L is deep red.
- VOLATILE session (energy gap risk / high ATR) → favour scalping, reduce_size or
  pause swing (overnight exposure is dangerous into volatility).
- Strong TREND (trending_up/down) → favour intraday + swing; scalping still ok.
- RANGING / low ATR → reduce_size across the board; pause swing.
- If daily_pnl breaches half of max_daily_loss → halt_new_trades true.
- If a single agent is bleeding (deep negative pnl or a loss streak) → pause it.
- Respect MCX session risk: approaching session close, prefer reduce_size for MIS."""


@dataclass
class BrainPosture:
    """The fleet-wide risk posture returned by the brain."""
    regime: str = "unknown"
    regime_confidence: int = 50
    size_factor: float = 1.0
    agent_directives: dict = field(default_factory=dict)
    halt_new_trades: bool = False
    opportunity_alert: Optional[str] = None
    reasoning: str = ""
    source: str = "rule_based"        # "claude" | "rule_based" | "fallback"
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ── Client (lazily created, shared) ──────────────────────────────────────────
_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def is_enabled() -> bool:
    """True when the brain will actually consult Claude (key present + flag on)."""
    return bool(settings.anthropic_api_key) and bool(settings.use_claude_brain)


# ── Rule-based fallback posture (used offline / on error) ─────────────────────

def _rule_based_posture(snapshot: dict, source: str = "rule_based") -> BrainPosture:
    """Deterministic posture derived from the regime plan already in the snapshot.

    The master's regime_detector has already produced a size_factor and an
    active/paused split; we mirror it so the fleet still runs sensibly with no LLM.
    """
    plan = snapshot.get("regime_plan", {}) or {}
    regime = snapshot.get("regime", "unknown")
    size_factor = float(plan.get("size_factor", 1.0) or 1.0)
    active = set(plan.get("active", []) or [])
    paused = set(plan.get("paused", []) or [])

    directives: dict = {}
    for key in AGENT_KEYS:
        if key in paused:
            directives[key] = {"action": "pause", "reason": "regime plan paused"}
        elif size_factor < 0.75:
            directives[key] = {"action": "reduce_size", "reason": "regime plan reduced size"}
        else:
            directives[key] = {"action": "run", "reason": "regime plan active"}

    # Halt if the daily loss is already deep.
    risk = snapshot.get("risk", {}) or {}
    daily_pnl = float(risk.get("daily_pnl", 0.0) or 0.0)
    max_loss = float(getattr(settings, "max_daily_loss", 0) or 0)
    halt = bool(max_loss) and daily_pnl <= -0.5 * max_loss

    return BrainPosture(
        regime=regime,
        regime_confidence=int(plan.get("regime_confidence", 55) or 55),
        size_factor=size_factor,
        agent_directives=directives,
        halt_new_trades=halt,
        opportunity_alert=None,
        reasoning=(plan.get("reasoning", "") or f"Rule-based posture for {regime}")[:160],
        source=source,
    )


# ── Response validation / coercion ───────────────────────────────────────────

def _coerce_posture(d: dict, latency_ms: int) -> BrainPosture:
    """Validate + clamp the raw JSON Claude returned into a BrainPosture."""
    try:
        sf = float(d.get("size_factor", 1.0))
    except (TypeError, ValueError):
        sf = 1.0
    # Snap size_factor into the allowed ladder [0.25, 1.0].
    sf = min(1.0, max(0.25, sf))

    raw_dirs = d.get("agent_directives", {}) or {}
    directives: dict = {}
    for key in AGENT_KEYS:
        entry = raw_dirs.get(key, {}) or {}
        action = str(entry.get("action", "run")).lower()
        if action not in _VALID_ACTIONS:
            action = "run"
        directives[key] = {"action": action, "reason": str(entry.get("reason", ""))[:160]}

    alert = d.get("opportunity_alert")
    if alert in (None, "null", "", "none"):
        alert = None
    else:
        alert = str(alert)[:200]

    try:
        conf = int(d.get("regime_confidence", 55))
    except (TypeError, ValueError):
        conf = 55

    return BrainPosture(
        regime=str(d.get("regime", "unknown")),
        regime_confidence=max(0, min(100, conf)),
        size_factor=sf,
        agent_directives=directives,
        halt_new_trades=bool(d.get("halt_new_trades", False)),
        opportunity_alert=alert,
        reasoning=str(d.get("reasoning", ""))[:240],
        source="claude",
        latency_ms=latency_ms,
    )


def _extract_text(resp) -> str:
    """Pull the text block out of a Messages response (skips thinking blocks)."""
    parts = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def _parse_json(raw: str) -> dict:
    """Tolerant JSON extraction — strips code fences, finds the outer object."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip().rstrip("`").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


# ── Public entry point ───────────────────────────────────────────────────────

async def assess(snapshot: dict) -> BrainPosture:
    """Consult the brain for the current fleet risk posture.

    `snapshot` is the master review report (regime, regime_plan, risk, strategy
    P&L, market context, …). Always returns a BrainPosture — never raises.
    """
    if not is_enabled():
        return _rule_based_posture(snapshot, source="rule_based")

    t0 = asyncio.get_event_loop().time()
    try:
        resp = await asyncio.wait_for(
            _get_client().messages.create(
                model=settings.claude_brain_model,
                max_tokens=settings.claude_brain_max_tokens,
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": json.dumps(snapshot, default=str)}],
            ),
            timeout=settings.claude_brain_timeout_sec,
        )
        latency = int((asyncio.get_event_loop().time() - t0) * 1000)
        posture = _coerce_posture(_parse_json(_extract_text(resp)), latency)
        _log(posture)
        return posture
    except asyncio.TimeoutError:
        logger.warning("[brain] timeout ({}s) — falling back to rule-based posture",
                       settings.claude_brain_timeout_sec)
        return _rule_based_posture(snapshot, source="fallback")
    except Exception as exc:
        logger.warning("[brain] error ({}) — falling back to rule-based posture", exc)
        return _rule_based_posture(snapshot, source="fallback")


def _log(p: BrainPosture) -> None:
    logger.info(
        "[brain] regime={} size={} halt={} conf={} src={} ({}ms) — {}",
        p.regime, p.size_factor, p.halt_new_trades, p.regime_confidence,
        p.source, p.latency_ms, p.reasoning,
    )
    entry = p.to_dict()
    _brain_log.appendleft(entry)


def get_brain_log(n: int = 30) -> list[dict]:
    """Return the last n brain postures (newest first)."""
    return list(_brain_log)[:n]


def status() -> dict:
    """Health/summary for the /brain endpoint."""
    last = _brain_log[0] if _brain_log else None
    return {
        "enabled":       is_enabled(),
        "has_api_key":   bool(settings.anthropic_api_key),
        "flag_on":       bool(settings.use_claude_brain),
        "model":         settings.claude_brain_model,
        "timeout_sec":   settings.claude_brain_timeout_sec,
        "min_interval_sec": settings.claude_brain_min_interval_sec,
        "assessments":   len(_brain_log),
        "last_posture":  last,
    }
