"""
debate_gate.py — a second opinion before every AI entry (the useful idea from TradingAgents).

The AI trader decides to enter. Before the order goes out, one more Claude call is made in
a different role: argue AGAINST the trade (the bear case), let the bull answer, then rule.
The verdict comes back through a strict tool — proceed or not, a confidence, a size factor
and the strongest objection — and code enforces it: below DEBATE_MIN_CONFIDENCE the entry
is rejected and the AI is told why. The trade is already past every hard limit when this
runs, so it costs one small call per would-be entry, not per alert.

Why it helps: an agent that has just built a bullish case is anchored on it. A fresh call
whose job is to find the hole catches the "trading into results", "fading a strong trend",
"third correlated bank" mistakes that the proposer glossed over. It never adds risk — it can
only shrink or stop a trade.
"""
from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

VERDICT_TOOL = {
    "name": "debate_verdict",
    "description": "Rule on the proposed trade after arguing both sides.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "bear_case": {"type": "string", "description": "The strongest case AGAINST this trade, in 2-3 sentences, using the data given"},
            "bull_rebuttal": {"type": "string", "description": "The best answer to the bear case, 1-2 sentences"},
            "proceed": {"type": "boolean", "description": "true only if the bull case clearly survives"},
            "confidence": {"type": "number", "description": "0-1: how confident you are that this trade has positive expectancy after costs"},
            "size_factor": {"type": "number", "description": "0.25-1.0: scale the size if you proceed with reservations (1.0 = full)"},
            "key_risk": {"type": "string", "description": "One phrase: the single thing most likely to make this trade lose"},
        },
        "required": ["bear_case", "bull_rebuttal", "proceed", "confidence", "size_factor", "key_risk"],
        "additionalProperties": False,
    },
}

SYSTEM = """You are the risk reviewer for an algorithmic trading desk on NSE (India). Another \
agent has proposed a trade and it has already passed the hard limits. Your job is adversarial: \
find the reason it loses. Argue the bear case first from the data (levels, event risk, trend \
filter, multi-timeframe, correlation, options data, VIX, the agent's own reasoning), then give \
the bull side its best rebuttal, then rule.

Be strict but fair: most reasonable setups should proceed at full size; a trade should fail \
when the data contradicts its premise — fading a strong trend without a reversal signal, an \
entry right at resistance with the target beyond it, results or an RBI decision inside the \
holding period, a third correlated position, a stop inside the noise band (< 1 ATR), a target \
that needs a move larger than the day's typical range, option buying with IV rank high, \
late-session intraday entries. Trades the proposer could not justify from the data fail. \
Never invent facts; if data is missing, say so and judge on what there is."""


def _summ(proposal: dict, context: dict) -> str:
    ctx = {k: v for k, v in (context or {}).items() if k not in ("recent_lessons",)}
    return json.dumps({"proposed_trade": proposal, "market_context": ctx}, default=str)


async def review(proposal: dict, context: Optional[dict] = None, client=None) -> dict:
    """proposal: {kind, symbol, side, quantity, price, stop, target, reasoning, ...}.
    Returns {"proceed", "confidence", "size_factor", "bear_case", "bull_rebuttal", "key_risk",
    "enforced": bool, "reason": str}. A failed call never blocks: enforced=False, proceed=True."""
    if not settings.ai_agent_debate_enabled:
        return {"proceed": True, "confidence": None, "size_factor": 1.0, "enforced": False, "reason": "debate disabled"}
    from ai_trader import ai_trader, _FALLBACK_MODELS
    try:
        client = client or ai_trader._get_client()
        kwargs = dict(model=settings.ai_agent_model, max_tokens=1200, system=SYSTEM, tools=[VERDICT_TOOL],
                      tool_choice={"type": "tool", "name": "debate_verdict"},
                      messages=[{"role": "user", "content": "PROPOSED TRADE AND EVIDENCE\n" + _summ(proposal, context)}],
                      output_config={"effort": settings.ai_agent_debate_effort})
        if settings.ai_agent_model in _FALLBACK_MODELS:
            resp = await client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = await client.messages.create(**kwargs)
        try:
            ai_trader._track_usage(resp)
        except Exception:
            pass
        v = None
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "debate_verdict":
                v = dict(block.input)
        if v is None:
            raise ValueError("no verdict returned")
    except Exception as exc:                      # the gate must never add risk by failing closed on a stuck API
        logger.warning("[debate] review failed ({}); entry proceeds unreviewed", exc)
        return {"proceed": True, "confidence": None, "size_factor": 1.0, "enforced": False,
                "reason": f"debate unavailable: {str(exc)[:120]}"}
    conf = max(0.0, min(1.0, float(v.get("confidence", 0))))
    sf = max(0.25, min(1.0, float(v.get("size_factor", 1.0))))
    proceed = bool(v.get("proceed")) and conf >= settings.ai_agent_debate_min_confidence
    reason = ("" if proceed else
              f"debate: {'bear case won' if not v.get('proceed') else f'confidence {conf:.2f} < {settings.ai_agent_debate_min_confidence}'}"
              f" — {v.get('bear_case', '')[:240]} (key risk: {v.get('key_risk', '')})")
    out = {"proceed": proceed, "confidence": conf, "size_factor": sf if proceed else 0.0, "enforced": True,
           "bear_case": v.get("bear_case", ""), "bull_rebuttal": v.get("bull_rebuttal", ""),
           "key_risk": v.get("key_risk", ""), "reason": reason, "at": now_ist().strftime("%H:%M:%S")}
    logger.info("[debate] {} {} {} → {} (conf {:.2f}, size ×{})", proposal.get("side"), proposal.get("symbol"),
                proposal.get("kind"), "PROCEED" if proceed else "BLOCK", conf, out["size_factor"])
    return out
