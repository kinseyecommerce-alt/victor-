"""
platform_scheduler.py — the server's only scheduler (APScheduler, IST), started with FastAPI.

Jobs (IST):
  08:45 / 08:55 / 09:05   Kite daily login (user id + password + TOTP from the vault); also at startup when today's
                          token is missing
  08:55                   validated preflight: session store + corporate-action calendar, agents set running
  09:06:45 … 09:20        forward test of the opening-auction fade (forward_test.py; AUC readiness reads it)
  09:15:00                daily reset of the agents' day counters (and the paper book's session)
  09:15:05                attention exits at the open, auction-fade entries          (validated_open)
  09:15:10                PEAD entries (only with VALIDATED_PEAD_ENABLED)
  09:20:00                auction-fade exit
  10:20:00                strangle entry; 15:01-16:51 every 10 min: strangle settlement; every 5 min: leg marks
  15:15 (SQUAREOFF_TIME)  paper square-off of leftover MIS / NRML rows (validated overnight holdings untouched)
  15:21:00                attention entry
  15:35                   daily scoreboard (ledger snapshot vs NIFTY 50, Telegram)
  15:40                   session-store refresh + parity audits
  16:10                   option-bar archive (OPTION_BAR_ARCHIVER_ENABLED)
  17:05                   shadow registry (SHADOW_REGISTRY_ENABLED)
  17:15                   drift & fill-quality monitor (warnings only)
  19:35-22:35, 08:20      PEAD evening / morning input refresh (VALIDATED_PEAD_ENABLED)
  every minute            portfolio drawdown brake (risk_budget.py)

Validated jobs run every day of the week and gate on ist_clock.is_trading_day (weekend Budget sessions included).
The legacy jobs (AI allocator, regime reads, universe refresh, pre-market report, morning data refresh, master's
nightly / weekly reviews, momentum rotation, bot auto-start of the tick agents) were removed in the core rebuild.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from agents.base_agent import send_telegram
from config import settings
from kite_client import kite_client  # noqa: F401  (tests patch it here)
from ist_clock import now_ist as _now_ist   # tz-aware IST: the container clock is UTC

VALIDATED_DOW = "mon-sun"    # validated jobs gate on is_trading_day (weekend Budget sessions included)


def _trading_today() -> bool:
    from ist_clock import is_trading_day
    return is_trading_day(_now_ist().date())


class PlatformScheduler:
    def __init__(self) -> None:
        self._sched = AsyncIOScheduler(timezone="Asia/Kolkata")
        self._token_ok = False

    # ── job table ──
    def start(self) -> None:
        # Keys may arrive later through the dashboard vault, so the scheduler always runs and each job checks
        # what it needs when it fires.
        kr = self._kite_token_refresh
        self._sched.add_job(kr, "cron", hour=8, minute="45,55", day_of_week="*", id="kite_refresh")
        self._sched.add_job(kr, "cron", hour=9, minute=5, day_of_week="*", id="kite_refresh_last")
        self._sched.add_job(self._kite_login_at_startup, "date", id="kite_login_startup")
        # live paper forward test of the opening-auction fade (forward_test.py)
        for _stage, (_h, _m, _s) in (("indicative_early", (9, 6, 45)), ("indicative", (9, 7, 30)),
                                     ("auction", (9, 8, 45)), ("open", (9, 15, 5)), ("exit", (9, 20, 0))):
            self._sched.add_job(self._forward_test, "cron", args=[_stage], hour=_h, minute=_m, second=_s,
                                day_of_week="mon-fri", id=f"forward_{_stage}", misfire_grace_time=20)
        for _fn, _id, _t, _grace in ((self._validated_preflight, "validated_preflight", (8, 55, 0), 300),
                                     (self._validated_open, "validated_open", (9, 15, 5), 50),
                                     (self._validated_auc_exit, "validated_auc_exit", (9, 20, 0), 60),
                                     (self._validated_attn_entry, "validated_attn_entry", (15, 21, 0), 150),
                                     (self._validated_store, "validated_store", (15, 40, 0), 600)):
            self._sched.add_job(_fn, "cron", hour=_t[0], minute=_t[1], second=_t[2], day_of_week=VALIDATED_DOW,
                                id=_id, misfire_grace_time=_grace)
        self._sched.add_job(self._validated_vrp_entry, "cron", hour=10, minute=20, second=0, day_of_week=VALIDATED_DOW,
                            id="validated_vrp_entry", misfire_grace_time=50)
        self._sched.add_job(self._validated_vrp_settle, "cron", hour="15-16", minute="1,11,21,31,41,51",
                            day_of_week=VALIDATED_DOW, id="validated_vrp_settle", misfire_grace_time=300)
        self._sched.add_job(self._validated_vrp_mark, "cron", hour="9-15", minute="*/5", day_of_week=VALIDATED_DOW,
                            id="validated_vrp_mark", misfire_grace_time=120)
        # EV-PEAD-LONG-H60 candidate: ONLY with VALIDATED_PEAD_ENABLED=true (default off) or while it still holds
        # positions / a plan from before the flag went off; PAPER only
        if self._pead_wanted():
            self._sched.add_job(self._validated_pead_evening, "cron", hour="19-22", minute=35, second=0,
                                day_of_week=VALIDATED_DOW, id="validated_pead_evening", misfire_grace_time=1800)
            self._sched.add_job(self._validated_pead_evening, "cron", hour=8, minute=20, second=0,
                                day_of_week=VALIDATED_DOW, id="validated_pead_morning", misfire_grace_time=1200)
            self._sched.add_job(self._validated_pead_entry, "cron", hour=9, minute=15, second=10,
                                day_of_week=VALIDATED_DOW, id="validated_pead_entry", misfire_grace_time=60)
        self._sched.add_job(self._validated_catch_up, "date", id="validated_catch_up",
                            run_date=_now_ist() + timedelta(seconds=60))
        # day counters and the paper square-off (formerly master's jobs; master existed only after a bot start)
        self._sched.add_job(self._daily_reset, "cron", hour=9, minute=15, second=0, day_of_week=VALIDATED_DOW,
                            id="daily_reset", misfire_grace_time=600)
        sq_h, sq_m = (int(x) for x in settings.squareoff_time.split(":"))
        self._sched.add_job(self._squareoff, "cron", hour=sq_h, minute=sq_m, day_of_week=VALIDATED_DOW,
                            id="squareoff", misfire_grace_time=300)
        self._sched.add_job(self._daily_scoreboard, "cron", hour=15, minute=35, day_of_week="mon-fri",
                            id="daily_scoreboard", misfire_grace_time=3600)
        if getattr(settings, "shadow_registry_enabled", False) and settings.trading_mode == "PAPER":
            self._sched.add_job(self._shadow_registry, "cron", hour=17, minute=5, day_of_week=VALIDATED_DOW,
                                id="shadow_registry", misfire_grace_time=3600)
        if self._option_archiver_on():
            self._sched.add_job(self._option_bar_archive, "cron", hour=16, minute=10, day_of_week=VALIDATED_DOW,
                                id="option_bar_archive", misfire_grace_time=3 * 3600)
        # engine upgrades: drawdown brake (every minute), drift monitor (after the last settlement retry)
        self._sched.add_job(self._risk_brake, "interval", minutes=1, id="risk_brake", misfire_grace_time=60)
        self._sched.add_job(self._drift_monitor, "cron", hour=17, minute=15, day_of_week=VALIDATED_DOW,
                            id="drift_monitor", misfire_grace_time=3600)
        self._sched.start()
        try:
            n_jobs = len(self._sched.get_jobs())
        except Exception:
            n_jobs = "?"
        logger.info("[platform] scheduler started ({} jobs)", n_jobs)

    async def stop(self) -> None:
        try:
            self._sched.shutdown(wait=False)
        except Exception:
            pass

    def jobs(self) -> list[dict]:
        """The scheduled jobs with their next run time (IST), soonest first."""
        out = []
        try:
            for j in self._sched.get_jobs():
                nrt = getattr(j, "next_run_time", None)
                out.append({"id": j.id, "next_run": nrt.isoformat() if nrt else None})
        except Exception:
            return []
        return sorted(out, key=lambda r: r["next_run"] or "9999")

    # ── bot control (formerly master_agent.start / stop) ──
    @staticmethod
    def bot_running() -> bool:
        import bot_state
        return not bot_state.operator_stopped()

    def bot_start(self) -> dict:
        """Operator Start: clears the persisted Stop and sets every enabled validated agent running."""
        import bot_state
        bot_state.set_operator_stopped(False)
        try:
            import paper_book
            paper_book.roll_day_if_due()
        except Exception as exc:
            logger.error("[platform] paper book day rollover failed: {}", exc)
        out = {}
        for name, a in self._validated_agents().items():
            if bot_state.is_agent_enabled(name):
                a.start()
            out[name] = {"running": a.state.running, "enabled": bot_state.is_agent_enabled(name)}
        return out

    async def bot_stop(self) -> dict:
        """Operator Stop (persisted until the next Start): no new entries; open positions still exit by rule."""
        import bot_state
        bot_state.set_operator_stopped(True)
        for a in self._validated_agents().values():
            a.stop()
        await send_telegram("<b>AlgoTrader Pro stopped</b> — no new entries until Start (open positions exit by rule)")
        return {n: {"running": a.state.running} for n, a in self._validated_agents().items()}

    async def _daily_reset(self) -> None:
        if not _trading_today():
            return
        for a in self._validated_agents().values():
            a.reset_daily()
        try:
            import paper_book
            paper_book.mark_day_reset()
        except Exception as exc:
            logger.error("[platform] paper book day mark failed: {}", exc)

    @staticmethod
    def squareoff_book() -> list[str]:
        """Flatten leftover intraday (MIS) / NRML paper rows; validated overnight holdings stay."""
        from agents.base_agent import overnight_hold_qty
        try:
            hold = overnight_hold_qty()
        except Exception as exc:
            logger.error("[platform] overnight holdings unavailable: {}", exc)
            hold = {}
        return kite_client.squareoff_all_positions(hold=hold)

    async def _squareoff(self) -> None:
        if not _trading_today() or settings.trading_mode != "PAPER":
            return
        ids = self.squareoff_book()
        if ids:
            logger.warning("[platform] square-off closed {} leftover row(s)", len(ids))
            await send_telegram(f"<b>Auto square-off</b>\n{len(ids)} positions closed")

    async def _daily_scoreboard(self) -> None:
        """After the close: store today's net P&L next to the NIFTY close, tell the owner."""
        try:
            from performance import performance
            snap = await asyncio.to_thread(performance.snapshot_today)
            sb = performance.scoreboard()
            p = sb["periods"]
            await send_telegram(
                f"<b>Scoreboard</b> {snap['date']}\n"
                f"Today: ₹{snap['net']:,.0f} net ({p['today']['account_pct']:+.2f}% vs NIFTY "
                f"{p['today'].get('nifty50_pct') if p['today'].get('nifty50_pct') is not None else '?'}%)\n"
                f"Month: {p['month']['account_pct']:+.2f}% vs NIFTY {p['month'].get('nifty50_pct')}%\n"
                f"{sb['verdict']}")
        except Exception as exc:
            logger.error("[platform] daily scoreboard failed: {}", exc)

    async def _risk_brake(self) -> None:
        try:
            import risk_budget
            risk_budget.apply_brake()
        except Exception as exc:
            logger.error("[platform] risk brake check failed: {}", exc)

    async def _drift_monitor(self) -> None:
        try:
            import drift_monitor
            await drift_monitor.daily_job()
        except Exception as exc:
            logger.error("[platform] drift monitor failed: {}", exc)

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
        if not _trading_today():
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
        if not _trading_today():
            return
        import validated_data as VD
        q = await asyncio.to_thread(VD.quotes, [f"NSE:{s}" for s in VD.UNIVERSE])
        await self._validated_call("attn_on", "exit_at_open", q)
        await self._validated_call("auc_fade", "enter", q)

    async def _validated_auc_exit(self) -> None:
        """09:20:00: the auction fade's exit (it retries inside until 09:29:50)."""
        if _trading_today():
            await self._validated_call("auc_fade", "exit")

    async def _validated_attn_entry(self) -> None:
        if _trading_today():
            await self._validated_call("attn_on", "decide_and_enter")

    async def _validated_store(self) -> None:
        if not _trading_today():
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
        if _trading_today():
            await self._validated_call("vrp_winged", "decide_and_enter")

    async def _validated_vrp_settle(self) -> None:
        # not gated on is_trading_day: an expiry's settlement is booked whenever its closes are available
        await self._validated_call("vrp_winged", "settle_due")

    async def _validated_vrp_mark(self) -> None:
        if _trading_today():
            await self._validated_call("vrp_winged", "mark")

    def _pead_wanted(self) -> bool:
        a = self._validated_agents().get("pead_h60")
        return bool(a is not None and settings.trading_mode == "PAPER" and a.wanted())

    async def _validated_pead_evening(self) -> None:
        # not gated on is_trading_day: the agent works out the last complete session (and any it missed) itself
        await self._validated_call("pead_h60", "evening")

    async def _validated_pead_entry(self) -> None:
        if _trading_today():
            await self._validated_call("pead_h60", "enter_at_open")

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

    async def _forward_test(self, stage: str) -> None:
        """One stage of the opening-auction forward test; the 09:20 stage builds the day's paper trades."""
        if not _trading_today():
            return
        try:
            import forward_test
            r = await asyncio.to_thread(forward_test.record, stage)
            # under VALIDATED_ONLY the auc_fade agent is the official record: no forward-test Telegram summary
            if stage == "exit" and r.get("net") and not getattr(settings, "validated_only", False):
                lines = "\n".join(f"{k}: {r['trades'][k]} trades, ₹{v:+,.0f}" for k, v in r["net"].items())
                await send_telegram(f"🧪 <b>Auction-fade forward test</b>\n{lines}")
        except Exception as exc:
            logger.error("[platform] forward test ({}) failed: {}", stage, exc)

    async def _shadow_registry(self) -> None:
        """Daily shadow-registry sync + evaluation (shadow_registry.daily_job re-checks the flag and PAPER)."""
        try:
            import shadow_registry
            await asyncio.to_thread(shadow_registry.daily_job)
        except Exception as exc:
            logger.error("[platform] shadow registry job failed: {}", exc)

    @staticmethod
    def _option_archiver_on() -> bool:
        try:
            import option_bar_archiver
            return option_bar_archiver.enabled()
        except Exception:
            return False

    async def _option_bar_archive(self) -> None:
        """Daily option / future minute-bar archive (option_bar_archiver.daily_job re-checks the flag)."""
        try:
            import option_bar_archiver
            await asyncio.to_thread(option_bar_archiver.daily_job)
        except Exception as exc:
            logger.error("[platform] option bar archive job failed: {}", exc)

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
                + ("Market opens in 25 minutes." if _now_ist().weekday() < 5 else "Weekend login (research/data only).")
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


platform_scheduler = PlatformScheduler()
