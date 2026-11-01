"""
research_loop.py — the weekly improvement cycle, with evidence as the gate.

Every weekend:
  1. Collect the week: per-strategy results after costs, execution slippage, the scoreboard
     vs NIFTY, pruning state, the trade memory's lessons, the optimiser's latest report.
  2. Ask Claude for ONE concrete change — a parameter of a tunable agent, inside its allowed
     grid — with a reason. Nothing else is accepted: no new strategies, no limit changes.
  3. A/B it: replay the last REPLAY_DAYS on the current watchlist with the current parameters
     and with the proposal (same bars, same slippage, same costs).
  4. Adopt only if the proposal beats the current parameters on net P&L after costs, has
     at least MIN_TRADES trades and no worse drawdown than 1.2×. Otherwise it is logged as
     rejected, with the numbers, and nothing changes.
  5. Write the whole cycle to data/research_log.json and tell the owner.

Claude proposes; the replay decides. No change reaches the live agents on an opinion.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

LOG_FILE = Path(os.getenv("RESEARCH_LOG_FILE") or Path(__file__).resolve().parent / "data" / "research_log.json")
REPLAY_DAYS = 30
MIN_TRADES = 10
MAX_DD_WORSENING = 1.2
MAX_LOG = 200

PROPOSAL_TOOL = {
    "name": "propose_change",
    "description": "Propose exactly one parameter change for one tunable agent, or no change.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": ["intraday", "scalping", "none"],
                      "description": "'none' if nothing in the evidence justifies a change"},
            "param": {"type": "string", "description": "A parameter from the agent's allowed grid, or '' for none"},
            "value": {"type": "number", "description": "One of the allowed values for that parameter, or 0 for none"},
            "rationale": {"type": "string", "description": "Two sentences grounded in the numbers you were given"},
        },
        "required": ["agent", "param", "value", "rationale"],
        "additionalProperties": False,
    },
}

SYSTEM = """You are the research analyst for AlgoTrader Pro, an algorithmic trading system on \
NSE (India). Once a week you are shown what happened — results per strategy after costs, \
slippage, the scoreboard against NIFTY, which strategies were pruned, lessons from closed \
trades, and the walk-forward optimiser's report — and you may propose ONE parameter change \
for one tunable agent, chosen from its allowed grid. Your proposal is then tested by replaying \
the last month's real bars with and without it; it is adopted only if it wins out of sample.

Rules: ground every proposal in the numbers (e.g. "stops hit 70% of the time with TGT 0.8% — \
widen SL_MIN_PCT"); prefer no change when the sample is small (< 30 trades) or the evidence is \
mixed; never propose a value outside the grid; one change only."""


def _entry(kind: str, **kw) -> dict:
    return {"at": now_ist().isoformat(timespec="seconds"), "kind": kind, **kw}


def _load_log() -> list[dict]:
    try:
        return json.loads(LOG_FILE.read_text())
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.error("[research] log unreadable: {}", exc)
        return []


def _append_log(entry: dict) -> None:
    log = _load_log()
    log.append(entry)
    log = log[-MAX_LOG:]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=LOG_FILE.parent, prefix=".research.")
    with os.fdopen(fd, "w") as f:
        json.dump(log, f, indent=1)
    os.replace(tmp, LOG_FILE)


def gather_evidence(days: int = 7) -> dict:
    """Everything the analyst gets to see. Cache/ledger reads only."""
    from performance import performance
    from datetime import timedelta
    import optimizer
    import market_filter
    since = (now_ist().date() - timedelta(days=days)).isoformat()
    week = [t for t in performance.trades if t["t"][:10] >= since]
    by: dict[str, list] = {}
    for t in week:
        by.setdefault(t["strategy"], []).append(t)
    per_strategy = {}
    for s, rows in by.items():
        st = performance._stats(rows)
        reasons: dict[str, int] = {}
        for r in rows:
            k = "stop" if "SL" in r.get("note", "") or "TSL" in r.get("note", "") else ("target" if "T2" in r.get("note", "") or "TARGET" in r.get("note", "").upper() else "other")
            reasons[k] = reasons.get(k, 0) + 1
        per_strategy[s] = {**st, "exit_mix": reasons,
                           "avg_slip_bps": round(sum(r["slip_bps"] for r in rows if r.get("slip_bps") is not None) /
                                                 max(1, sum(1 for r in rows if r.get("slip_bps") is not None)), 1)}
    try:
        from trade_memory import get_recent_insights
        lessons = [{"symbol": e.get("symbol"), "strategy": e.get("strategy"), "pnl": e.get("pnl"), "lesson": e.get("lesson")}
                   for e in get_recent_insights(12) if e.get("lesson")]
    except Exception:
        lessons = []
    return {
        "week_from": since, "trades_this_week": len(week), "per_strategy": per_strategy,
        "scoreboard": {k: v for k, v in performance.scoreboard().items() if k in ("periods", "verdict", "days_tracked")},
        "pruned": {k: v["reason"] for k, v in performance.pruned.items()},
        "execution": performance.execution_stats().get("per_strategy", {}),
        "trend_filter": market_filter.state().get("state"),
        "current_params": {a: optimizer.current_params(a) for a in optimizer.GRIDS},
        "allowed_grid": optimizer.GRIDS,
        "optimizer_last": {a: {k: v for k, v in (r or {}).items() if k in ("adopted", "current", "at")}
                           for a, r in ((optimizer._state.get("report") or {}).items())},
        "lessons": lessons,
    }


async def propose(evidence: dict, client=None) -> dict:
    """One tool call to Claude. Returns the proposal dict (agent may be 'none')."""
    import anthropic
    from ai_trader import ai_trader, _FALLBACK_MODELS
    client = client or ai_trader._get_client()
    kwargs = dict(model=settings.ai_agent_model, max_tokens=2000, system=SYSTEM, tools=[PROPOSAL_TOOL],
                  tool_choice={"type": "tool", "name": "propose_change"},
                  messages=[{"role": "user", "content": "WEEKLY EVIDENCE\n" + json.dumps(evidence, default=str)}],
                  output_config={"effort": "medium"})
    if settings.ai_agent_model in _FALLBACK_MODELS:
        resp = await client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = await client.messages.create(**kwargs)
    try:
        ai_trader._track_usage(resp)
    except Exception:
        pass
    for block in resp.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "propose_change":
            return dict(block.input)
    return {"agent": "none", "param": "", "value": 0, "rationale": "no proposal returned"}


def validate(p: dict) -> Optional[str]:
    import optimizer
    if p.get("agent") in (None, "none", ""):
        return None
    grid = optimizer.GRIDS.get(p["agent"])
    if not grid:
        return f"{p['agent']} is not tunable"
    if p.get("param") not in grid:
        return f"{p.get('param')} is not in {p['agent']}'s grid {list(grid)}"
    allowed = grid[p["param"]]
    if not any(abs(float(p.get("value", 0)) - float(v)) < 1e-9 for v in allowed):
        return f"{p['value']} is not an allowed value for {p['param']} ({allowed})"
    if abs(float(p["value"]) - float(optimizer.current_params(p["agent"])[p["param"]])) < 1e-9:
        return "that is already the current value"
    return None


async def ab_test(agent: str, param: str, value: float, symbols: list[str], days: int = REPLAY_DAYS,
                  bars_by_symbol: Optional[dict] = None) -> dict:
    import optimizer
    import replay_backtest as rb
    cur = optimizer.current_params(agent)
    cand = {**cur, param: value}
    if bars_by_symbol is None:
        import asyncio
        bars_by_symbol = {}
        for s in symbols:
            bars, _ = await asyncio.to_thread(rb.load_bars, s, days)
            if bars:
                bars_by_symbol[s] = bars
    if not bars_by_symbol:
        return {"error": "no bar data"}
    a = (await rb.replay_async(agent, list(bars_by_symbol), days, params=cur, bars_by_symbol=bars_by_symbol)).metrics()
    b = (await rb.replay_async(agent, list(bars_by_symbol), days, params=cand, bars_by_symbol=bars_by_symbol)).metrics()
    verdict = "adopt"
    if b["trades"] < MIN_TRADES:
        verdict = f"reject: only {b['trades']} trades (< {MIN_TRADES})"
    elif b["net_pnl"] <= a["net_pnl"]:
        verdict = f"reject: net ₹{b['net_pnl']:,.0f} does not beat current ₹{a['net_pnl']:,.0f}"
    elif b["net_pnl"] <= 0:
        verdict = "reject: still loses money"
    elif a["max_drawdown"] > 0 and b["max_drawdown"] > a["max_drawdown"] * MAX_DD_WORSENING:
        verdict = f"reject: drawdown ₹{b['max_drawdown']:,.0f} vs ₹{a['max_drawdown']:,.0f}"
    return {"agent": agent, "current": {"params": cur, **a}, "proposed": {"params": cand, **b}, "verdict": verdict,
            "symbols": list(bars_by_symbol), "days": days}


async def run_cycle(symbols: Optional[list[str]] = None, client=None, bars_by_symbol: Optional[dict] = None) -> dict:
    """The whole weekly cycle. Returns the log entry that was written."""
    import optimizer
    if not settings.anthropic_api_key and client is None:
        entry = _entry("skipped", reason="no Anthropic key")
        _append_log(entry)
        return entry
    evidence = gather_evidence()
    try:
        proposal = await propose(evidence, client)
    except Exception as exc:
        entry = _entry("error", stage="propose", error=str(exc)[:300])
        _append_log(entry)
        return entry
    why_not = validate(proposal)
    if proposal.get("agent") in (None, "none", ""):
        entry = _entry("no_change", proposal=proposal, evidence_summary=_summ(evidence))
        _append_log(entry)
        return entry
    if why_not:
        entry = _entry("invalid", proposal=proposal, reason=why_not, evidence_summary=_summ(evidence))
        _append_log(entry)
        return entry
    if not symbols and not bars_by_symbol:
        from tick_engine import tick_engine
        symbols = [s for s in tick_engine.symbols() if " " not in s and tick_engine._exchange.get(s, "NSE") == "NSE"][:6]
        if not symbols:
            from symbol_scanner import NIFTY_50
            symbols = NIFTY_50[:6]
    ab = await ab_test(proposal["agent"], proposal["param"], float(proposal["value"]), symbols or [], bars_by_symbol=bars_by_symbol)
    if "error" in ab:
        entry = _entry("error", stage="ab_test", proposal=proposal, error=ab["error"])
        _append_log(entry)
        return entry
    if ab["verdict"] == "adopt":
        data = optimizer.load_params()
        data[proposal["agent"]] = {"params": ab["proposed"]["params"], "chosen_at": now_ist().isoformat(timespec="seconds"),
                                  "test_net": ab["proposed"]["net_pnl"], "current_test_net": ab["current"]["net_pnl"],
                                  "symbols": ab["symbols"], "days": ab["days"], "source": "research_loop"}
        optimizer.save_params(data)
        optimizer.apply_params()
    entry = _entry("adopted" if ab["verdict"] == "adopt" else "rejected", proposal=proposal, ab=ab,
                   evidence_summary=_summ(evidence))
    _append_log(entry)
    return entry


def _summ(ev: dict) -> dict:
    return {"trades_this_week": ev.get("trades_this_week"),
            "per_strategy": {k: {kk: v.get(kk) for kk in ("trades", "win_rate", "net_pnl", "profit_factor")} for k, v in ev.get("per_strategy", {}).items()},
            "verdict": (ev.get("scoreboard") or {}).get("verdict")}


def log(n: int = 20) -> list[dict]:
    return _load_log()[-n:]


def telegram_text(entry: dict) -> str:
    k = entry["kind"]
    p = entry.get("proposal") or {}
    if k == "adopted":
        ab = entry["ab"]
        return (f"<b>Research: change adopted</b>\n{p['agent']}.{p['param']} → {p['value']}\n"
                f"Replay {ab['days']}d: ₹{ab['proposed']['net_pnl']:,.0f} vs ₹{ab['current']['net_pnl']:,.0f} current\n{p.get('rationale', '')[:200]}")
    if k == "rejected":
        return f"<b>Research: proposal rejected</b>\n{p['agent']}.{p['param']} → {p['value']}\n{entry['ab']['verdict']}"
    if k == "no_change":
        return f"<b>Research: no change proposed</b>\n{p.get('rationale', '')[:200]}"
    return f"<b>Research: {k}</b> {entry.get('reason') or entry.get('error') or ''}"
