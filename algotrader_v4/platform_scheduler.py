"""
platform_scheduler.py
Server-level scheduler — starts with FastAPI, independent of the trading bot.

Jobs (all IST, Mon–Fri):
  08:45  Kite daily login (user ID + password + TOTP from the vault), retried 08:55 and 09:05
         if it failed; also attempted at startup when today's token is missing
  09:16  Auto-start trading bot (1 min after daily_reset fires at 09:15)
"""
from __future__ import annotations

import asyncio
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from agents.base_agent import send_telegram
from config import settings
from kite_client import kite_client
from ist_clock import now_ist as _now_ist   # tz-aware IST: the container clock is UTC


# master_agent_v5 jobs (same schedule as master.start()) that must run whether or not the bot is started.
# Not the trading jobs (square-off, daily reset, momentum rotation): those belong to a running bot.
MASTER_HOUSEKEEPING: dict[str, dict] = {
    "_daily_scoreboard":        dict(hour=15, minute=35, day_of_week="mon-fri"),
    "_nightly_adaptive":        dict(hour=21, minute=0,  day_of_week="mon-fri"),
    "_weekly_backtest":         dict(hour=20, minute=0,  day_of_week="sun"),
    "_weekly_optimise":         dict(hour=21, minute=30, day_of_week="sat"),
    "_weekly_research":         dict(hour=21, minute=30, day_of_week="sun"),
    "_weekly_memory_synthesis": dict(hour=21, minute=0,  day_of_week="sun"),
}


VALIDATED_DOW = "mon-sun"    # validated jobs gate on is_trading_day (weekend Budget sessions included)


class PlatformScheduler:
    def __init__(self) -> None:
        self._sched = AsyncIOScheduler(timezone="Asia/Kolkata")
        self._token_ok = False

    def start(self) -> None:
        # Keys may arrive later through the dashboard vault, so the scheduler always runs and
        # each job checks what it needs when it fires.
        self._sched.add_job(
            self._kite_token_refresh, "cron",
            hour=8, minute="45,55", day_of_week="mon-fri", id="kite_refresh",
        )
        self._sched.add_job(
            self._kite_token_refresh, "cron",
            hour=9, minute=5, day_of_week="mon-fri", id="kite_refresh_last",
        )
        self._sched.add_job(self._kite_login_at_startup, "date", id="kite_login_startup")
        # a restart after 09:16 (deploy, crash) would otherwise leave the bot stopped for the day
        self._sched.add_job(self._auto_start_catch_up, "date", id="auto_start_catch_up",
                            run_date=_now_ist() + __import__("datetime").timedelta(seconds=45))
        self._sched.add_job(
            self._pre_market_report, "cron",
            hour=9, minute=0, day_of_week="mon-fri", id="pre_market",
        )
        self._sched.add_job(
            self._morning_data_refresh, "cron",
            hour=9, minute=10, day_of_week="mon-fri", id="morning_data",
        )
        self._sched.add_job(
            self._allocate, "cron", args=["morning"],
            hour=9, minute=8, day_of_week="mon-fri", id="allocate_morning",
        )
        self._sched.add_job(
            self._auto_start_bot, "cron",
            hour=9, minute=16, day_of_week="mon-fri", id="auto_start",
        )
        # market-type reads (regime_v2): at the open (the gap is known) and after the first hour
        self._sched.add_job(
            self._regime_read, "cron", args=["preopen"],
            hour=9, minute=16, second=30, day_of_week="mon-fri", id="regime_preopen",
        )
        self._sched.add_job(
            self._regime_read, "cron", args=["first_hour"],
            hour=10, minute=15, second=30, day_of_week="mon-fri", id="regime_first_hour",
        )
        self._sched.add_job(
            self._allocate, "cron", args=["midday"],
            hour=12, minute=30, day_of_week="mon-fri", id="allocate_midday",
        )
        # live paper forward test of the opening-auction fade (forward_test.py): pre-open indicative,
        # post-match auction, continuous open, 09:20 exit
        for _stage, (_h, _m, _s) in (("indicative_early", (9, 6, 45)), ("indicative", (9, 7, 30)), ("auction", (9, 8, 45)),
                                     ("open", (9, 15, 5)), ("exit", (9, 20, 0))):
            self._sched.add_job(
                self._forward_test, "cron", args=[_stage],
                hour=_h, minute=_m, second=_s, day_of_week="mon-fri", id=f"forward_{_stage}",
                misfire_grace_time=20,
            )
        # lab-validated PAPER agents (agents/validated_*.py): fixed IST times of the tested rules. Every day of the
        # week: each job gates on ist_clock.is_trading_day, which knows the full weekend sessions (Budget days) the
        # lab's calendar traded
        for _fn, _id, _t, _grace in ((self._validated_preflight, "validated_preflight", (8, 55, 0), 300),
                                     (self._validated_open, "validated_open", (9, 15, 5), 50),
                                     (self._validated_auc_exit, "validated_auc_exit", (9, 20, 0), 60),
                                     (self._validated_attn_entry, "validated_attn_entry", (15, 21, 0), 150),
                                     (self._validated_store, "validated_store", (15, 40, 0), 600)):
            self._sched.add_job(_fn, "cron", hour=_t[0], minute=_t[1], second=_t[2], day_of_week=VALIDATED_DOW,
                                id=_id, misfire_grace_time=_grace)
        # VRP-STRANGLE-016-WINGED: 10:20 entry; settlement from 15:31 on (retried every 10 min to 16:51);
        # read-only leg marks every 5 min for the status
        self._sched.add_job(self._validated_vrp_entry, "cron", hour=10, minute=20, second=0, day_of_week=VALIDATED_DOW,
                            id="validated_vrp_entry", misfire_grace_time=50)
        self._sched.add_job(self._validated_vrp_settle, "cron", hour="15-16", minute="1,11,21,31,41,51",
                            day_of_week=VALIDATED_DOW, id="validated_vrp_settle", misfire_grace_time=300)
        self._sched.add_job(self._validated_vrp_mark, "cron", hour="9-15", minute="*/5", day_of_week=VALIDATED_DOW,
                            id="validated_vrp_mark", misfire_grace_time=120)
        self._sched.add_job(self._validated_catch_up, "date", id="validated_catch_up",
                            run_date=_now_ist() + __import__("datetime").timedelta(seconds=60))
        self._sched.add_job(
            self._refresh_universe, "cron",
            hour=16, minute=30, day_of_week="mon-fri", id="universe_refresh",
        )
        self._sched.add_job(
            self._options_cache_refresh, "interval",
            minutes=5, id="options_cache",
        )
        # master's date-based housekeeping (scoreboard, nightly and weekend reviews) lives in master's own
        # scheduler, which only exists once master.start() ran. After a restart from 15:00 to the next 09:16,
        # or on a weekend/holiday, master is not running: these run them in its place (never twice).
        for _name, _cron in MASTER_HOUSEKEEPING.items():
            self._sched.add_job(self._master_job, "cron", args=[_name], id=f"master_{_name}", **_cron)
        self._sched.start()
        logger.info("[platform] scheduler started (Kite login@08:45/08:55/09:05, Report@09:00, Data@09:10, Start@09:16 IST)")

    async def stop(self) -> None:
        try:
            self._sched.shutdown(wait=False)
        except Exception:
            pass

    # ── Jobs ──────────────────────────────────────────────────────────────────

    async def _master_job(self, name: str) -> None:
        """Run one of master's housekeeping jobs when master is not running (its own scheduler runs it otherwise)."""
        from master_agent_v5 import master_agent as master
        if master.running:
            return
        logger.info("[platform] bot not running — running master's {} in its place", name)
        try:
            await getattr(master, name)()
        except Exception as exc:
            logger.error("[platform] {} failed: {}", name, exc)

    # ── lab-validated agents (PAPER) ──────────────────────────────────────────
    @staticmethod
    def _validated_agents() -> dict:
        from agents.strategy_agents import ALL_AGENTS
        return {n: a for n, a in ALL_AGENTS.items() if getattr(a, "scheduled", False)}

    async def _validated_call(self, name: str, method: str, *args):
        """Run one agent job; an exception is logged and counted in the agent's state.errors."""
        a = self._validated_agents().get(name)
        if a is None or not hasattr(a, method):
            return None
        try:
            r = getattr(a, method)(*args)
            return await r if asyncio.iscoroutine(r) else r
        except Exception as exc:
            a.state.errors.append(f"{method}: {str(exc)[:200]}")
            logger.exception("[validated] {}.{} failed: {}", name, method, exc)
            return None

    async def _validated_preflight(self) -> None:
        from ist_clock import is_trading_day
        if not is_trading_day():
            return
        import validated_data as VD
        try:
            if not VD.store_fresh(_now_ist().date()):
                await asyncio.to_thread(VD.refresh_store)
        except Exception as exc:
            logger.error("[validated] session store refresh failed: {}", exc)
        try:
            import event_calendar
            await event_calendar.refresh_calendar()
        except Exception as exc:
            logger.error("[validated] corporate-action calendar refresh failed: {}", exc)
        for name, a in self._validated_agents().items():
            a.auto_start()                       # not after an operator Stop (persisted until the next Start)
            logger.info("[validated] {} preflight: entries {}", name,
                        a.entry_block_reason_readonly() or "allowed")

    async def _validated_open(self) -> None:
        """09:15:05: one quote of the universe; attention exits at the open, then the auction fade enters."""
        from ist_clock import is_trading_day
        if not is_trading_day():
            return
        import validated_data as VD
        q = await asyncio.to_thread(VD.quotes, [f"NSE:{s}" for s in VD.UNIVERSE])
        await self._validated_call("attn_on", "exit_at_open", q)
        await self._validated_call("auc_fade", "enter", q)

    async def _validated_auc_exit(self) -> None:
        """09:20:00: the auction fade's exit (it retries inside until 09:29:50)."""
        from ist_clock import is_trading_day
        if is_trading_day():
            await self._validated_call("auc_fade", "exit")

    async def _validated_attn_entry(self) -> None:
        from ist_clock import is_trading_day
        if is_trading_day():
            await self._validated_call("attn_on", "decide_and_enter")

    async def _validated_store(self) -> None:
        from ist_clock import is_trading_day
        if not is_trading_day():
            return
        import validated_data as VD
        try:
            await asyncio.to_thread(VD.refresh_store)
        except Exception as exc:
            logger.error("[validated] session store refresh failed: {}", exc)
            return
        for name, a in self._validated_agents().items():
            if hasattr(a, "parity_audit"):
                await self._validated_call(name, "parity_audit")

    async def _validated_vrp_entry(self) -> None:
        from ist_clock import is_trading_day
        if is_trading_day():
            await self._validated_call("vrp_winged", "decide_and_enter")

    async def _validated_vrp_settle(self) -> None:
        # not gated on is_trading_day: an expiry's settlement is booked whenever its closes are available
        await self._validated_call("vrp_winged", "settle_due")

    async def _validated_vrp_mark(self) -> None:
        from ist_clock import is_trading_day
        if is_trading_day():
            await self._validated_call("vrp_winged", "mark")

    async def _validated_catch_up(self) -> None:
        """A process started after 08:55 lacks what the preflight prepared: the session store (refreshed when it
        misses the previous trading day) and the in-memory corporate-action calendar; then each agent's catch_up
        (serialised with the scheduled jobs by the agent's stage lock)."""
        from ist_clock import is_trading_day
        import validated_data as VD
        today = _now_ist().date()
        if is_trading_day(today):
            try:
                if not VD.store_fresh(today):
                    await asyncio.to_thread(VD.refresh_store)
            except Exception as exc:
                logger.error("[validated] catch-up store refresh failed: {}", exc)
            if not await VD.ensure_ca_calendar():
                logger.warning("[validated] catch-up: corporate-action calendar unavailable")
        for name in self._validated_agents():
            await self._validated_call(name, "catch_up", _now_ist())

    async def _refresh_universe(self) -> None:
        """After the close: replay the opening-range agents on the NIFTY 100 for the next session's universe."""
        from ist_clock import is_trading_day
        if not (settings.use_rolling_universe and is_trading_day()):
            return
        try:
            import universe
            st = await universe.refresh()
            await send_telegram(f"🌐 <b>Universe for {st['for_date']}</b>: {len(st['symbols'])} stocks"
                                + (" (too few — fixed list)" if st["fallback"] else f" — {', '.join(st['symbols'][:12])}"))
        except Exception as exc:
            logger.error("[platform] universe refresh failed: {}", exc)

    async def _regime_read(self, stage: str) -> None:
        """Today's market type for the regime-aware sizing (regime_v2)."""
        from ist_clock import is_trading_day
        if not is_trading_day():
            return
        try:
            import asyncio as _aio
            import regime_v2
            r = await _aio.to_thread(regime_v2.refresh, stage)
            if r.get("regime"):
                from agents.base_agent import send_telegram
                st = regime_v2.status()
                f = " ".join(f"{k} {v[0]:g}×" for k, v in st["factors"].items())
                await send_telegram(f"🧭 <b>Market type ({stage})</b>: {r['regime']} → {f}")
        except Exception as exc:
            logger.error("[platform] regime read ({}) failed: {}", stage, exc)

    async def _forward_test(self, stage: str) -> None:
        """One stage of the opening-auction forward test; the 09:20 stage builds the day's paper trades."""
        from ist_clock import is_trading_day
        if not is_trading_day():
            return
        try:
            import asyncio as _aio
            import forward_test
            r = await _aio.to_thread(forward_test.record, stage)
            # under VALIDATED_ONLY the auc_fade agent is the official record: no forward-test Telegram summary
            if stage == "exit" and r.get("net") and not getattr(settings, "validated_only", False):
                lines = "\n".join(f"{k}: {r['trades'][k]} trades, ₹{v:+,.0f}" for k, v in r["net"].items())
                await send_telegram(f"🧪 <b>Auction-fade forward test</b>\n{lines}")
        except Exception as exc:
            logger.error("[platform] forward test ({}) failed: {}", stage, exc)

    async def _allocate(self, reason: str) -> None:
        """The AI capital manager's two daily calls (09:08 before auto-start, 12:30)."""
        from ist_clock import is_trading_day
        if not is_trading_day():
            return
        try:
            import allocator
            await allocator.run(reason)
        except Exception as exc:
            logger.error("[platform] allocation ({}) failed: {}", reason, exc)

    async def _pre_market_report(self) -> None:
        from ist_clock import is_trading_day
        if not is_trading_day():
            return
        try:
            from pre_market_report import generate_pre_market_report
            await generate_pre_market_report()
        except Exception as exc:
            logger.error("[platform] Pre-market report failed: {}", exc)

    async def _morning_data_refresh(self) -> None:
        """Refresh all static daily data: levels, events, institutional flow, correlation matrix."""
        from ist_clock import is_trading_day
        if not is_trading_day():
            logger.info("[platform] Morning data refresh skipped — not a trading day")
            return
        from tick_engine import tick_engine
        symbols = list(tick_engine._subscribers.keys()) if hasattr(tick_engine, "_subscribers") else []
        symbols = sorted(set(symbols) | set(getattr(tick_engine, "_symbols", [])))   # warm watchlist too
        if not symbols:
            logger.info("[platform] Morning data refresh: no symbols yet — will retry after bot start")
            return
        results = []
        try:
            from daily_data import prefetch as daily_prefetch
            n = await daily_prefetch(symbols)
            results.append(f"daily history {n} ✓")   # the agents' 20-day trend filter reads this cache
        except Exception as exc:
            logger.warning("[platform] Daily history prefetch failed: {}", exc)
        try:
            from levels_engine import refresh_daily as levels_refresh
            await levels_refresh(symbols)
            results.append("levels ✓")
        except Exception as exc:
            logger.warning("[platform] Levels refresh failed: {}", exc)

        try:
            from event_calendar import refresh_calendar
            await refresh_calendar()
            results.append("events ✓")
        except Exception as exc:
            logger.warning("[platform] Event calendar refresh failed: {}", exc)

        try:
            from institutional_flow import refresh_daily
            await refresh_daily(symbols)
            results.append("institutional ✓")
        except Exception as exc:
            logger.warning("[platform] Institutional flow refresh failed: {}", exc)

        try:
            from correlation_guard import refresh_matrix
            await refresh_matrix(symbols)
            results.append("correlation ✓")
        except Exception as exc:
            logger.warning("[platform] Correlation matrix refresh failed: {}", exc)

        try:
            import opportunity_scanner
            await asyncio.to_thread(opportunity_scanner.warm)
            results.append("opportunity scan ✓")
        except Exception as exc:
            logger.warning("[platform] Opportunity scan warm-up failed: {}", exc)

        logger.info("[platform] Morning data refresh: {}", ", ".join(results) or "all failed")

    async def _options_cache_refresh(self) -> None:
        """Refresh options IV cache every 5 minutes during market hours."""
        from market_data import is_market_open
        if not is_market_open():
            return
        try:
            from tick_engine import tick_engine
            from options_intelligence import update_cache
            symbols = [s for s in (tick_engine.all_latest() or {}).keys()]
            if symbols:
                await update_cache(symbols[:20])  # limit to top 20 to avoid rate limits
        except Exception as exc:
            logger.debug("[platform] Options cache refresh: {}", exc)

    async def _kite_login_at_startup(self) -> None:
        """Server (re)started on a trading day without a working token → log in now."""
        import kite_auto_login
        try:
            if await asyncio.to_thread(kite_auto_login.needed_now):
                await self._kite_token_refresh()
        except Exception as exc:
            logger.debug("[platform] startup Kite login check: {}", exc)

    async def _kite_token_refresh(self) -> None:
        import kite_auto_login
        if not kite_auto_login.credentials_ready():
            logger.info("[platform] Kite auto-login skipped: {} not set (log in from the dashboard)",
                        ", ".join(kite_auto_login.missing_credentials()))
            return
        if self._token_ok and await asyncio.to_thread(kite_auto_login.token_valid_today):
            return                                   # an earlier attempt today already succeeded
        logger.info("[platform] Kite daily login starting…")
        try:
            from kite_auto_login import refresh_kite_token_async
            token = await refresh_kite_token_async()
            self._token_ok = True
            await send_telegram(
                f"✅ <b>Kite token refreshed</b>\n"
                f"Token: <code>{token[:8]}…</code>\n"
                f"Market opens in 25 minutes."
            )
        except Exception as exc:
            self._token_ok = False
            logger.error("[platform] Kite token refresh failed: {}", exc)
            login_url = f"https://{settings.kite_redirect_url.split('/')[2]}/login" \
                        if settings.kite_redirect_url else "/login"
            await send_telegram(
                f"⚠️ <b>Kite auto-login failed</b>\n"
                f"Reason: {exc}\n\n"
                f"Renew manually before 09:15:\n{login_url}"
            )

    async def _auto_start_catch_up(self) -> None:
        """At startup inside market hours (09:16-15:00 IST) run the 09:16 auto-start that was missed."""
        from ist_clock import now_ist
        t = now_ist().time()
        if t.replace(microsecond=0) < __import__("datetime").time(9, 16) or t >= __import__("datetime").time(15, 0):
            return
        logger.info("[platform] started inside market hours — running the missed 09:16 auto-start")
        await self._auto_start_bot()

    async def _auto_start_bot(self) -> None:
        if not settings.auto_start_strategies:
            logger.info("[platform] AUTO_START_STRATEGIES not set — skipping auto-start")
            return
        from ist_clock import is_trading_day, holiday_name, next_trading_day
        if not is_trading_day():
            logger.info("[platform] {} — exchange holiday, auto-start skipped; next session {}",
                        holiday_name() or "weekend", next_trading_day())
            await send_telegram(f"📅 <b>Exchange holiday</b> — {holiday_name() or 'no session'}. "
                                f"Bot not started; next session {next_trading_day():%a %d %b}.")
            return

        from master_agent_v5 import master_agent as master
        if master.running:
            logger.info("[platform] Bot already running — auto-start skipped")
            return

        # Verify Kite is authenticated
        try:
            kite_client.profile()
        except Exception as exc:
            logger.error("[platform] Kite not authenticated — auto-start aborted: {}", exc)
            await send_telegram(
                "⚠️ <b>Auto-start aborted</b>\n"
                "Kite session not valid. Please connect Kite and start the bot manually."
            )
            return

        strategies = [s.strip() for s in settings.auto_start_strategies.split(",") if s.strip()]

        # Build watchlist — Nifty 100 flag takes priority, then explicit list, then scanner
        if settings.use_nifty100_watchlist:
            from nifty100 import get_strategy_watchlist, NIFTY_100, as_watchlist
            if len(strategies) == 1:
                watchlist = get_strategy_watchlist(strategies[0])
            else:
                watchlist = as_watchlist(NIFTY_100)
            logger.info("[platform] Using full Nifty 100 watchlist ({} symbols)", len(watchlist))
        elif settings.use_rolling_universe and __import__("universe").today_watchlist():
            import universe
            watchlist = [{"symbol": s, "exchange": "NSE"} for s in universe.today_watchlist()]
            logger.info("[platform] Using the rolling universe ({} stocks)", len(watchlist))
        elif settings.auto_start_watchlist:
            watchlist = [
                {"symbol": s.strip(), "exchange": "NSE"}
                for s in settings.auto_start_watchlist.split(",")
                if s.strip()
            ]
        else:
            from symbol_scanner import symbol_scanner
            watchlist = symbol_scanner.last_scan or await asyncio.to_thread(symbol_scanner.scan)

        if not watchlist:
            await send_telegram("⚠️ <b>Auto-start skipped</b> — watchlist is empty")
            return

        try:
            report = master.start(strategies, watchlist)
            lines = [f"🚀 <b>Bot auto-started</b> | Mode: {settings.trading_mode}"]
            for strat, data in report.items():
                lines.append(f"  {strat}: {data['approved']}/{data['total']} symbols approved")
            await send_telegram("\n".join(lines))
            logger.info("[platform] bot auto-started: {}", {s: r["approved"] for s, r in report.items()})
        except Exception as exc:
            logger.error("[platform] auto-start failed: {}", exc)
            await send_telegram(f"⚠️ <b>Auto-start failed</b>\n{exc}")
            return
        if settings.ai_agent_auto_start and (settings.anthropic_api_key or settings.openai_api_key):
            try:
                from ai_trader import ai_trader
                if not ai_trader.enabled:
                    ai_trader.start()
                    logger.info("[platform] AI trader auto-started")
                    await send_telegram("🧠 <b>AI trader auto-started</b>")
            except Exception as exc:
                logger.error("[platform] AI auto-start failed: {}", exc)


platform_scheduler = PlatformScheduler()
