"""
allocator.py — the AI as capital manager, not stock picker.

Twice on each trading day (09:08, before the bot auto-starts, and 12:30) the model receives the
regime read, today's event risk, and an evidence card per strategy (replay results on real data,
live results after costs, yesterday's P&L, pruned/enabled state) and returns one thing: a size
factor of 0–1.5 per strategy for the rest of the day, plus whether today is unusual enough that
every entry should pass the bear-case gate. The agents multiply their risk-based quantity by that
factor; everything else (stops, limits, kill switch, daily loss cap) is untouched and cannot be
overridden from here.

Four days of replay showed the model is a cautious trader but a sensible judge; this uses it for
the judgement at two calls a day instead of a decision every 15 minutes. If no AI key is set or
the call fails, every factor is 1.0 — the system never depends on the model to trade.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist, is_trading_day

STRATEGIES = ["intraday", "scalping", "swing", "fno"]
MAX_FACTOR = 1.5

_FILE = Path(os.getenv("ALLOCATION_FILE")
             or Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
             .parent / "allocation.json")

_state: dict = {}

SYSTEM_PROMPT = """You are the capital allocator of AlgoTrader Pro, an algorithmic trading system on NSE \
(India) with four mechanical strategies: intraday (opening-range breakout, trend-aligned, 2.5R), scalping \
(opening-range retest, 1.5R), swing (multi-day breakout, delivery) and fno (index option buys / spreads). \
You do not pick trades. You decide, from evidence, how much of its normal size each strategy should trade \
today, and whether today is unusual enough that every entry must pass the independent bear-case review.

Rules
- Each factor is 0 to 1.5 of normal size; 1.0 means "as validated". Default to 1.0 unless the evidence says \
otherwise. Use 0 only for a strategy whose recent live results are clearly losing after costs or whose \
regime is wrong today (e.g. breakout strategies in a ranging, low-volume day; option buying when VIX is \
collapsing). Go above 1.0 only when both the replay and the recent live results are positive and the \
regime fits; never above 1.5.
- Unusual day = major scheduled event (RBI policy, budget, index expiry with extreme VIX, results in many \
watchlist stocks), VIX move over 10 %, or a gap over 1 % on NIFTY: set unusual_day true, gate_all_entries \
true and shade the factors down.
- Missing data is not evidence. Before the open the regime detector reports UNKNOWN, VIX may be null and the platform's own size_factor / paused list reflects the previous session or a closed market: ignore them in that case and keep 1.0. Only cut a strategy on positive evidence against it (losing live results after costs, a wrong regime that is actually measured, an event). A strategy with no live history yet trades at its replay-validated size, 1.0.
- Be brief and concrete in the note (what you saw, what you changed, why). Call set_allocation exactly once."""

TOOL = {"name": "set_allocation",
        "description": "Set today's size factors (0–1.5 of normal size) per strategy and the day's risk posture.",
        "input_schema": {"type": "object",
                         "properties": {"intraday": {"type": "number"}, "scalping": {"type": "number"},
                                        "swing": {"type": "number"}, "fno": {"type": "number"},
                                        "unusual_day": {"type": "boolean"}, "gate_all_entries": {"type": "boolean"},
                                        "note": {"type": "string", "description": "2-4 sentences for the trade log"}},
                         "required": ["intraday", "scalping", "swing", "fno", "unusual_day", "gate_all_entries", "note"],
                         "additionalProperties": False}}


def _default(reason: str = "default") -> dict:
    return {"date": now_ist().date().isoformat(), "factors": {s: 1.0 for s in STRATEGIES}, "unusual_day": False,
            "gate_all_entries": False, "note": "", "source": reason, "at": None, "model": None}


def _load() -> dict:
    try:
        if _FILE.exists():
            return json.loads(_FILE.read_text())
    except Exception as exc:
        logger.debug("[allocator] cannot read {}: {}", _FILE, exc)
    return {}


def _save(st: dict) -> None:
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, default=str))
        tmp.replace(_FILE)
    except Exception as exc:
        logger.debug("[allocator] not saved: {}", exc)


def today() -> dict:
    """Today's allocation (persisted across restarts); defaults on a day without a decision."""
    global _state
    if not _state:
        _state = _load()
    if _state.get("date") != now_ist().date().isoformat():
        return _default()
    return _state


def size_factor(agent: str) -> float:
    """The AI allocator's factor for today × the regime factor (regime_v2: this strategy's tested result on
    today's predicted market type), capped at MAX_FACTOR. A 0 from the allocator stays 0."""
    try:
        f = float(today()["factors"].get(agent, 1.0))
    except Exception:
        f = 1.0
    try:
        import regime_v2
        f *= regime_v2.factor(agent)
    except Exception as exc:
        logger.debug("[allocator] regime factor unavailable: {}", exc)
    return round(min(MAX_FACTOR, max(0.0, f)), 3)


def gate_required() -> bool:
    return bool(today().get("gate_all_entries"))


def reset() -> None:
    global _state
    _state = {}


# ── evidence ─────────────────────────────────────────────────────────────
def _yesterday_pnl() -> dict[str, float]:
    from performance import performance
    d = now_ist().date() - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    out: dict[str, float] = {}
    for t in performance.trades:
        if str(t.get("t", ""))[:10] == d.isoformat():
            out[t["strategy"]] = round(out.get(t["strategy"], 0.0) + float(t.get("net", 0.0)), 2)
    return {"date": d.isoformat(), **out}


def evidence() -> dict:
    """Everything the allocator sees. Each block fails soft so one broken source never blocks the day."""
    ev: dict = {"date": now_ist().date().isoformat(), "time": now_ist().strftime("%H:%M")}
    try:
        import regime_v2
        ev["market_type"] = {"today": regime_v2.current(),
                             "note": "learned day-type read; its per-strategy size factors are applied on top of yours"}
    except Exception as exc:
        ev["market_type"] = {"error": str(exc)[:80]}
    try:
        from market_context import regime_block
        ev["regime"] = regime_block()
    except Exception as exc:
        ev["regime"] = {"error": str(exc)[:80]}
    try:
        from tick_engine import tick_engine
        from event_calendar import get_event_risk
        risks = []
        for sym in list(getattr(tick_engine, "_symbols", []))[:60]:
            e = get_event_risk(sym)
            if e.get("risk_level", "NONE") not in ("NONE", None):
                risks.append({"symbol": sym, "risk": e.get("risk_level"), "event": e.get("event_type"),
                              "hours_until": None if e.get("hours_until") in (None, float("inf")) else round(e["hours_until"], 1)})
        ev["events"] = risks[:12]
        idx, _ = tick_engine.latest("NIFTY 50")
        ev["nifty"] = {"ltp": idx.ltp, "change_pct": round(idx.change_pct, 2)} if idx else None
    except Exception as exc:
        ev["events"] = []; ev["nifty"] = None; ev["events_error"] = str(exc)[:80]
    cards = {}
    try:
        from performance import performance
        live = performance.strategy_stats()
    except Exception:
        live = {}
    try:
        import replay_backtest as rb
        replay = rb._LAST
    except Exception:
        replay = {}
    try:
        ypnl = _yesterday_pnl()
    except Exception:
        ypnl = {}
    for s in STRATEGIES:
        l = live.get(s) or {}
        r = replay.get(s) or {}
        cards[s] = {"replay": {k: r.get(k) for k in ("trades", "net_pnl", "profit_factor", "max_drawdown", "win_rate", "period")} if r else None,
                    "live": {k: l.get(k) for k in ("trades", "net_pnl", "win_rate", "profit_factor", "max_drawdown")} if l else None,
                    "live_recent": l.get("recent"), "pruned": l.get("pruned"), "enabled": l.get("enabled", True),
                    "yesterday_net": ypnl.get(s)}
    ev["strategies"] = cards
    ev["yesterday"] = ypnl.get("date")
    ev["limits"] = {"total_capital": settings.total_capital, "max_daily_loss": settings.max_daily_loss,
                    "risk_per_trade_pct": settings.risk_per_trade_pct}
    return ev


# ── decision ─────────────────────────────────────────────────────────────
async def _create(messages: list):
    from llm_provider import llm_router
    return await llm_router.create(messages=messages, system=SYSTEM_PROMPT, tools=[TOOL], max_tokens=6000,
                                   effort=settings.ai_agent_effort, tool_choice={"type": "tool", "name": "set_allocation"})


def _thin_evidence(ev: dict, strategy: str) -> bool:
    """No measured regime and no live trades for the strategy: nothing to justify switching it off."""
    regime = (ev.get("regime") or {}).get("regime")
    live = ((ev.get("strategies") or {}).get(strategy) or {}).get("live") or {}
    return regime in (None, "UNKNOWN") and not (live.get("trades") or 0)


def _clamp(x, lo=0.0, hi=MAX_FACTOR) -> float:
    try:
        return round(min(hi, max(lo, float(x))), 2)
    except (TypeError, ValueError):
        return 1.0


async def run(reason: str = "manual") -> dict:
    """Ask the model for today's allocation; on any failure keep (or fall back to) factors of 1.0."""
    global _state
    if not (settings.anthropic_api_key or settings.openai_api_key):
        st = _default("no_ai_key"); st["note"] = "No AI key set — all strategies at normal size."
        _state = st; _save(st); return st
    ev = await asyncio.to_thread(evidence)
    user = (f"{reason.capitalize()} allocation call at {ev['time']} IST on {ev['date']}.\n\nEvidence (JSON):\n"
            f"{json.dumps(ev, default=str)}\n\nSet today's allocation.")
    try:
        resp = await _create([{"role": "user", "content": user}])
    except Exception as exc:
        logger.warning("[allocator] model call failed: {}", exc)
        st = _state if _state.get("date") == ev["date"] else _default("fallback")
        st["note"] = (st.get("note") or "") + f" | allocator call failed ({str(exc)[:80]}): factors unchanged"
        _state = st; _save(st); return st
    call = next((b for b in resp.content if b.get("type") == "tool_use" and b.get("name") == "set_allocation"), None)
    if call is None:
        logger.warning("[allocator] model returned no allocation")
        st = _state if _state.get("date") == ev["date"] else _default("fallback")
        st["note"] = (st.get("note") or "") + " | model returned no allocation: factors unchanged"
        _state = st; _save(st); return st
    a = dict(call.input or {})
    factors = {s: _clamp(a.get(s, 1.0)) for s in STRATEGIES}
    floored = [s for s in STRATEGIES if factors[s] < 0.5 and _thin_evidence(ev, s)]
    for s in floored:                      # the model may not switch a strategy off on missing data
        factors[s] = 0.5
    st = {"date": ev["date"], "factors": factors,
          "unusual_day": bool(a.get("unusual_day")), "gate_all_entries": bool(a.get("gate_all_entries")),
          "note": str(a.get("note", ""))[:600] + (f" | floored to 0.5 on thin evidence: {', '.join(floored)}" if floored else ""),
          "source": "ai", "reason": reason,
          "at": now_ist().isoformat(timespec="seconds"), "model": getattr(resp, "model", None),
          "evidence": {k: ev.get(k) for k in ("regime", "events", "nifty", "yesterday")}}
    _state = st
    _save(st)
    logger.info("[allocator] {} → {} unusual={} gate_all={} | {}", reason, st["factors"], st["unusual_day"],
                st["gate_all_entries"], st["note"][:120])
    try:
        from agents.base_agent import send_telegram
        f = " ".join(f"{k} {v:g}×" for k, v in st["factors"].items())
        await send_telegram(f"🧭 <b>Allocation ({reason})</b> {f}{' · unusual day, all entries gated' if st['gate_all_entries'] else ''}\n{st['note'][:300]}")
    except Exception:
        pass
    return st
