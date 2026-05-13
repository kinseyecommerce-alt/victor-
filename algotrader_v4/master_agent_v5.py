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
from datetime import datetime
from typing import Optional

import anthropic
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

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


MASTER_PROMPT = """You are the MASTER TRADING AGENT for an NSE/BSE algo trading system.
You receive live indicator snapshots, regime data, adaptive learning metrics, and agent statuses.
Return ONLY valid JSON (no markdown, no code fences):
{
  "market_regime": "trending_up|trending_down|ranging|volatile",
  "regime_confidence": 0-100,
  "agent_directives": {
    "intraday":  {"action": "run|pause|reduce_size", "reason": "..."},
    "fno":       {"action": "run|pause|reduce_size", "reason": "..."},
    "swing":     {"action": "run|pause|reduce_size", "reason": "..."},
    "scalping":  {"action": "run|pause|reduce_size", "reason": "..."}
  },
  "capital_allocation": {"intraday": 0-100, "fno": 0-100, "swing": 0-100, "scalping": 0-100},
  "risk_override": {"halt_new_trades": false, "reason": ""},
  "summary": "one sentence"
}
Rules:
- capital_allocation must sum to 100. Never >40% to one strategy.
- Pause agents with pnl < -2000 or >3 consecutive errors.
- VOLATILE regime → favour scalping, reduce swing.
- TRENDING regime → favour swing + intraday.
- If adaptive status is CAUTIOUS or RETIRED for a strategy, reduce_size or pause.
- Respect SEBI kill-switch — if sebi_state != ACTIVE, halt_new_trades = true."""


class MasterAgent:

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._scheduler = AsyncIOScheduler()
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
            if self._agent_watchlists.get(strat):
                q = tick_engine.add_subscriber(f"agent_{strat}")
                agent.start(q)

        sq_h, sq_m = [int(x) for x in settings.squareoff_time.split(":")]
        self._scheduler.add_job(self._master_review,   "interval", seconds=300, id="master_review")
        self._scheduler.add_job(self._auto_squareoff,  "cron", hour=sq_h, minute=sq_m,
                                 day_of_week="mon-fri", id="squareoff")
        self._scheduler.add_job(self._daily_reset,     "cron", hour=9, minute=15,
                                 day_of_week="mon-fri", id="daily_reset")
        self._scheduler.add_job(self._nightly_adaptive,"cron", hour=21, minute=0,
                                 day_of_week="mon-fri", id="nightly_adaptive")
        self._scheduler.start()
        logger.info("[master_v5] started — tick-driven 1s")
        asyncio.create_task(send_telegram(
            f"<b>AlgoTrader Pro v5</b> started\nMode: {settings.trading_mode} | Tick: 1s\n"
            + "\n".join(f"  {s}: {r['approved']}/{r['total']} symbols" for s, r in report.items())
        ))
        return report

    def stop(self) -> None:
        self.running = False
        tick_engine.stop()
        for a in ALL_AGENTS.values():
            a.stop()
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            pass
        asyncio.create_task(send_telegram("<b>AlgoTrader Pro v5 stopped</b>"))

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

        report = {
            "timestamp":  datetime.now().isoformat(),
            "mode":       settings.trading_mode,
            "regime":     regime.value,
            "regime_plan": {
                "active":      plan.active,
                "paused":      plan.paused,
                "allocation":  plan.allocation,
                "size_factor": plan.size_factor,
            },
            "regime_signals": (regime_detector.current_signals.to_dict()
                               if regime_detector.current_signals else {}),
            "adaptive_summary": adaptive_engine.summary(),
            "live_market":  tick_engine.all_latest(),
            "risk":         risk_manager.status(),
            "guard":        order_guard.status(),
            "agents":       {n: a.get_status() for n, a in ALL_AGENTS.items()},
        }

        try:
            msg = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
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

    async def _auto_squareoff(self) -> None:
        ids = kite_client.squareoff_all_positions()
        if ids:
            await send_telegram(f"<b>Auto square-off</b>\n{len(ids)} positions closed")

    async def _daily_reset(self) -> None:
        risk_manager.reset_daily()
        order_guard.reset_daily()
        for a in ALL_AGENTS.values():
            a.reset_daily()
        await send_telegram("<b>New trading day</b> — counters reset")

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
                if self._agent_watchlists.get(strat):
                    q = tick_engine.add_subscriber(f"agent_{strat}")
                    agent.start(q)

        if d.get("risk_override", {}).get("halt_new_trades"):
            risk_manager.is_trading_halted = True
            logger.warning("[master] Claude halted new trades: {}",
                           d.get("risk_override", {}).get("reason", ""))

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
