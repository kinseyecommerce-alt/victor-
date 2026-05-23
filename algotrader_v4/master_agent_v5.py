"""
master_agent_v5.py — AlgoTrader Pro v5
Tick-driven master agent with:
  • Market regime detection + automatic strategy gating
  • Claude-powered 5-minute review cycle
  • Adaptive engine integration (nightly nightly_review)
  • SEBI compliance hooks on every trade decision
  • Atomic bracket order orchestration
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import anthropic
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from ist_clock import now_ist, minutes_to_squareoff as _mts

from config import settings
from kite_client import kite_client
from risk_manager import risk_manager
from order_guard import order_guard
from backtest_engine import backtest_engine
from tick_engine import tick_engine
from market_regime import regime_detector, Regime, REGIME_PLANS
from adaptive_engine import adaptive_engine
from agents.base_agent import send_telegram
from agents.strategy_agents import ALL_AGENTS
from bot_state import is_agent_enabled


MASTER_PROMPT = """You are the MASTER TRADING INTELLIGENCE for an NSE/BSE algorithmic trading system.
You have FULL situational awareness: regime, market breadth, sector rotation, FII/DII flows,
PCR, VIX term structure, intraday P&L curve, and per-strategy adaptive performance.

Your dual mandate: CAPTURE EVERY GENUINE OPPORTUNITY. PREVENT EVERY AVOIDABLE LOSS.

Return ONLY valid JSON — no markdown, no code fences:
{
  "market_regime": "trending_up|trending_down|ranging|volatile",
  "regime_confidence": 0-100,
  "agent_directives": {
    "intraday":  {"action": "run|pause|reduce_size", "reason": "<specific reason>"},
    "fno":       {"action": "run|pause|reduce_size", "reason": "<specific reason>"},
    "swing":     {"action": "run|pause|reduce_size", "reason": "<specific reason>"},
    "scalping":  {"action": "run|pause|reduce_size", "reason": "<specific reason>"}
  },
  "capital_allocation": {"intraday": 0-100, "fno": 0-100, "swing": 0-100, "scalping": 0-100},
  "trade_gate_threshold": 55-85,
  "risk_override": {"halt_new_trades": false, "reason": ""},
  "opportunity_alert": "<null or 1-sentence alert about a specific opportunity window>",
  "summary": "<one crisp sentence on current market state and primary edge>"
}

MANDATORY RULES:
- capital_allocation must sum to 100. Never >40% to a single strategy.
- Pause any strategy: pnl_today < -2000, OR consecutive_errors > 3.
- VOLATILE + VIX>20 → favour scalping (up to 35%), reduce/pause swing.
- TRENDING + ADX>25 → favour intraday + swing, scalping max 20%.
- RANGING (ADX<18) → reduce all sizes 30%, no swing entries.
- If adaptive_status is CAUTIOUS → reduce_size. If RETIRED → pause.
- trade_gate_threshold: raise to 75 in ranging/volatile; lower to 60 in strong trend.
- If daily_pnl < -50% of max_daily_loss → halt_new_trades for 30 min.
- If PCR > 1.5 → bearish pressure on calls, warn FNO agent.
- If VIX spikes > 18 intraday → immediately reduce all size_factors 50%.
- opportunity_alert: flag if a sector is showing unusual breakout strength not yet captured."""


# ── Context helpers ────────────────────────────────────────────────────────────

def _minutes_to_squareoff() -> int:
    from config import settings as _s
    return _mts(_s.squareoff_time)


def _nifty_snapshot(live: dict) -> dict:
    """Pull NIFTY + BANKNIFTY from live tick data if available."""
    result = {}
    for key in ("NIFTY 50", "NIFTY50", "BANKNIFTY", "NIFTY BANK"):
        if key in live:
            snap = live[key]
            result[key] = {
                "ltp":   snap.get("ltp"),
                "change_pct": snap.get("change_pct"),
            }
    return result


def _sector_summary(live: dict) -> dict:
    """
    Summarise sector leadership from live data.
    Returns top 3 gainers and top 3 losers by change_pct.
    """
    changes = {
        sym: snap.get("change_pct", 0)
        for sym, snap in live.items()
        if snap.get("change_pct") is not None
    }
    if not changes:
        return {}
    sorted_syms = sorted(changes, key=changes.get, reverse=True)
    return {
        "top_gainers": sorted_syms[:3],
        "top_losers":  sorted_syms[-3:],
    }


_gate_veto_count = 0
_gate_allow_count = 0


def record_gate_decision(entered: bool) -> None:
    global _gate_veto_count, _gate_allow_count
    if entered:
        _gate_allow_count += 1
    else:
        _gate_veto_count += 1


def _gate_stats() -> dict:
    total = _gate_veto_count + _gate_allow_count
    veto_rate = round(_gate_veto_count / total * 100, 1) if total > 0 else 0
    return {
        "vetoed": _gate_veto_count,
        "allowed": _gate_allow_count,
        "veto_rate_pct": veto_rate,
    }


class MasterAgent:

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
        self.running = False
        self._agent_watchlists: dict[str, list[dict]] = {}
        self.last_directives: dict = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, strategies: list[str], watchlist: list[dict]) -> dict:
        self.running = True
        report: dict[str, dict] = {}

        for strat in strategies:
            agent = ALL_AGENTS.get(strat)
            if not agent:
                continue
            if not is_agent_enabled(strat):
                continue
            approved = agent.filter_watchlist(watchlist)
            self._agent_watchlists[strat] = approved
            report[strat] = {
                "total": len(watchlist),
                "approved": len(approved),
                "symbols": [a["symbol"] for a in approved],
            }

        tick_engine.subscribe(watchlist)

        for strat in strategies:
            agent = ALL_AGENTS.get(strat)
            if not agent:
                continue
            if not is_agent_enabled(strat):
                continue
            if self._agent_watchlists.get(strat):
                q = tick_engine.add_subscriber(f"agent_{strat}")
                agent.start(q)

        sq_h, sq_m = [int(x) for x in settings.squareoff_time.split(":")]
        self._scheduler.add_job(self._master_review,   "interval", seconds=60,  id="master_review")
        self._scheduler.add_job(self._auto_squareoff,  "cron", hour=sq_h, minute=sq_m,
                                 day_of_week="mon-fri", id="squareoff")
        self._scheduler.add_job(self._daily_reset,     "cron", hour=9, minute=15,
                                 day_of_week="mon-fri", id="daily_reset")
        self._scheduler.add_job(self._nightly_adaptive,"cron", hour=21, minute=0,
                                 day_of_week="mon-fri", id="nightly_adaptive")
        self._scheduler.add_job(self._weekly_backtest, "cron", hour=20, minute=0,
                                 day_of_week="sun", id="weekly_backtest")
        self._scheduler.add_job(self._weekly_memory_synthesis, "cron", hour=21, minute=0,
                                 day_of_week="sun", id="weekly_memory")
        self._scheduler.start()
        logger.info("[master_v5] started — tick-driven 1s")
        asyncio.create_task(send_telegram(
            f"<b>AlgoTrader Pro v5</b> started\nMode: {settings.trading_mode} | Tick: 1s\n"
            + "\n".join(f"  {s}: {r['approved']}/{r['total']} symbols" for s, r in report.items())
        ))
        from n8n_bridge import notify as _n8n
        asyncio.create_task(_n8n("system", {"type": "bot_started", "mode": settings.trading_mode}))
        return report

    async def stop(self) -> None:
        self.running = False
        tick_engine.stop()
        for a in ALL_AGENTS.values():
            a.stop()
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            pass
        await send_telegram("<b>AlgoTrader Pro v5 stopped</b>")
        from n8n_bridge import notify as _n8n
        asyncio.create_task(_n8n("system", {"type": "bot_stopped"}))

    # ── Scheduled jobs ─────────────────────────────────────────────────────────

    async def _master_review(self) -> None:
        if not self.running:
            return
        try:
            regime, plan = await regime_detector.update()
        except Exception as exc:
            logger.error("[master] Regime detection failed: {}", exc)
            regime = regime_detector.current_regime
            plan   = regime_detector.current_plan

        self._apply_regime_plan(regime, plan)

        sigs = regime_detector.current_signals
        live = tick_engine.all_latest()
        risk_st = risk_manager.status()

        # ── Gate stats: how aggressive/conservative the gate has been ────────
        gate_stats = _gate_stats()

        # ── Intraday P&L trajectory (last 5 data points from agents) ─────────
        pnl_trajectory = [
            {"strategy": n, "pnl": a.get_status().get("pnl_today", 0),
             "trades": a.get_status().get("trades_today", 0),
             "win_rate": a.get_status().get("win_rate_today", 0),
             "consecutive_losses": a.get_status().get("consecutive_losses", 0)}
            for n, a in ALL_AGENTS.items()
        ]

        report = {
            "timestamp":       now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
            "mode":            settings.trading_mode,
            "minutes_to_close": _minutes_to_squareoff(),

            # Regime
            "regime":          regime.value,
            "regime_plan": {
                "active":      plan.active,
                "paused":      plan.paused,
                "allocation":  plan.allocation,
                "size_factor": plan.size_factor,
                "reasoning":   plan.reasoning,
            },
            "regime_signals": (sigs.to_dict() if sigs else {}),

            # Market context
            "nifty_snapshot": _nifty_snapshot(live),
            "sector_leaders": _sector_summary(live),
            "gate_stats":     gate_stats,

            # Strategy performance
            "strategy_pnl":   pnl_trajectory,
            "adaptive_summary": adaptive_engine.summary(),

            # Risk
            "risk":  risk_st,
            "guard": order_guard.status(),
        }

        try:
            msg = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                system=MASTER_PROMPT,
                messages=[{"role": "user", "content": json.dumps(report, indent=2, default=str)}],
            )
            raw = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            d   = json.loads(raw)
            self.last_directives = {
                **d,
                "regime": regime.value,
                "regime_reasoning": plan.reasoning,
            }
            self._apply_directives(d)
        except Exception as exc:
            logger.error("[master] Claude review error: {}", exc)
            self.last_directives = {
                "regime":            regime.value,
                "regime_reasoning":  plan.reasoning,
                "strategy_plan": {
                    "active":      plan.active,
                    "paused":      plan.paused,
                    "allocation":  plan.allocation,
                },
                "summary": f"Regime {regime.value}. {plan.reasoning[:80]}",
            }

        summary = self.last_directives.get("summary", "")
        if summary:
            asyncio.create_task(send_telegram(
                f"<b>Regime: {regime.value}</b>\n"
                f"Active: {', '.join(plan.active)}\n"
                f"Paused: {', '.join(plan.paused) or 'none'}\n"
                f"Size:   {int(plan.size_factor * 100)}%\n{summary}"
            ))
            from n8n_bridge import notify as _n8n
            asyncio.create_task(_n8n("regime_change", {
                "regime":      regime.value,
                "active":      plan.active,
                "paused":      plan.paused,
                "size_factor": plan.size_factor,
                "reasoning":   plan.reasoning[:120],
                "signals":     sigs.to_dict() if sigs else {},
            }))

    async def _auto_squareoff(self) -> None:
        ids = kite_client.squareoff_all_positions()
        if ids:
            await send_telegram(f"<b>Auto square-off</b>\n{len(ids)} positions closed")
            from n8n_bridge import notify as _n8n
            asyncio.create_task(_n8n("system", {"type": "squareoff", "positions_closed": len(ids)}))

    async def _daily_reset(self) -> None:
        risk_manager.reset_daily()
        order_guard.reset_daily()
        for a in ALL_AGENTS.values():
            a.reset_daily()
        await send_telegram("<b>New trading day</b> — counters reset")
        from n8n_bridge import notify as _n8n
        asyncio.create_task(_n8n("system", {"type": "daily_reset"}))

    async def _nightly_adaptive(self) -> None:
        try:
            current_vix    = regime_detector.current_signals.vix if regime_detector.current_signals else 14.0
            regime_changed = regime_detector.history and len(regime_detector.history) >= 2 and \
                             regime_detector.history[-1] != regime_detector.history[-2]
            report = await adaptive_engine.nightly_review(current_vix, regime_changed)
            logger.info("[master] Nightly adaptive review: {} ok, {} adapt, {} retire",
                        len(report["strategies_ok"]),
                        len(report["strategies_adapt"]),
                        len(report["strategies_retire"]))
        except Exception as exc:
            logger.error("[master] Nightly adaptive review failed: {}", exc)

    async def _weekly_memory_synthesis(self) -> None:
        try:
            from trade_memory import weekly_synthesis
            await weekly_synthesis()
        except Exception as exc:
            logger.error("[master] Weekly memory synthesis failed: {}", exc)

    async def _weekly_backtest(self) -> None:
        """Every Sunday 8 PM — re-backtest the full symbol universe, refresh approved cache."""
        try:
            logger.info("[master] Weekly backtest starting…")
            summary = await asyncio.to_thread(backtest_engine.weekly_auto_backtest)
            lines = ["<b>Weekly Backtest Complete</b>"]
            for strat, data in summary.items():
                lines.append(
                    f"  {strat}: {data['pass_count']} pass / {data['fail_count']} fail"
                )
            await send_telegram("\n".join(lines))
        except Exception as exc:
            logger.error("[master] Weekly backtest failed: {}", exc)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _apply_regime_plan(self, regime: Regime, plan) -> None:
        for strat in plan.paused:
            agent = ALL_AGENTS.get(strat)
            if agent and agent.state.running:
                agent.stop()
                logger.info("[master] Regime {} → paused {}", regime.value, strat)

        for strat in plan.active:
            agent = ALL_AGENTS.get(strat)
            if agent and not agent.state.running:
                if not is_agent_enabled(strat):
                    continue
                if self._agent_watchlists.get(strat):
                    q = tick_engine.add_subscriber(f"agent_{strat}")
                    agent.start(q)
                    logger.info("[master] Regime {} → started {}", regime.value, strat)

    def _apply_directives(self, d: dict) -> None:
        for strat, directive in d.get("agent_directives", {}).items():
            agent = ALL_AGENTS.get(strat)
            if not agent:
                continue
            action = directive.get("action", "run")
            if action == "pause" and agent.state.running:
                agent.stop()
            elif action in ("run", "reduce_size") and not agent.state.running:
                if not is_agent_enabled(strat):
                    continue
                if self._agent_watchlists.get(strat):
                    q = tick_engine.add_subscriber(f"agent_{strat}")
                    agent.start(q)

        if d.get("risk_override", {}).get("halt_new_trades"):
            risk_manager.is_trading_halted = True
            logger.warning("[master] Claude halted new trades: {}",
                           d.get("risk_override", {}).get("reason", ""))

        # Apply dynamic trade gate threshold from master
        threshold = d.get("trade_gate_threshold")
        if threshold and settings.use_claude_trade_gate:
            settings.claude_gate_threshold = int(threshold)
            logger.info("[master] Gate threshold → {}", threshold)

        # Log opportunity alert if present
        alert = d.get("opportunity_alert")
        if alert and alert not in (None, "null", ""):
            logger.info("[master] 💡 Opportunity: {}", alert)
            asyncio.create_task(send_telegram(f"💡 <b>Opportunity Alert</b>\n{alert}"))

    def get_status(self) -> dict:
        return {
            "master_running":    self.running,
            "mode":              settings.trading_mode,
            "architecture":      "tick-driven 1s",
            "regime":            regime_detector.status(),
            "adaptive":          adaptive_engine.summary(),
            "live_market":       tick_engine.all_latest(),
            "last_directives":   self.last_directives,
            "agents":            {n: a.get_status() for n, a in ALL_AGENTS.items()},
            "risk":              risk_manager.status(),
            "guard":             order_guard.status(),
            "backtest_approved": {n: backtest_engine.get_approved_symbols(n) for n in ALL_AGENTS},
        }


master_agent = MasterAgent()
