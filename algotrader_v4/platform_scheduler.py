"""
platform_scheduler.py
Server-level scheduler — starts with FastAPI, independent of the trading bot.

Jobs (all IST, Mon–Fri):
  08:50  Kite token auto-refresh via Playwright
  09:16  Auto-start trading bot (1 min after daily_reset fires at 09:15)
"""
from __future__ import annotations

import asyncio
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from agents.base_agent import send_telegram
from config import settings
from kite_client import kite_client


class PlatformScheduler:
    def __init__(self) -> None:
        self._sched = AsyncIOScheduler(timezone="Asia/Kolkata")
        self._token_ok = False

    def start(self) -> None:
        if not settings.kite_api_key:
            logger.warning("[platform] KITE_API_KEY not set — platform scheduler skipped")
            return

        self._sched.add_job(
            self._kite_token_refresh, "cron",
            hour=8, minute=50, day_of_week="mon-fri", id="kite_refresh",
        )
        self._sched.add_job(
            self._pre_market_report, "cron",
            hour=9, minute=0, day_of_week="mon-fri", id="pre_market",
        )
        self._sched.add_job(
            self._morning_data_refresh, "cron",
            hour=9, minute=10, day_of_week="mon-fri", id="morning_data",
        )
        self._sched.add_job(
            self._auto_start_bot, "cron",
            hour=9, minute=16, day_of_week="mon-fri", id="auto_start",
        )
        self._sched.add_job(
            self._options_cache_refresh, "interval",
            minutes=5, id="options_cache",
        )
        self._sched.start()
        logger.info("[platform] scheduler started (Kite@08:50, Report@09:00, Data@09:10, Start@09:16 IST)")

    async def stop(self) -> None:
        try:
            self._sched.shutdown(wait=False)
        except Exception:
            pass

    # ── Jobs ──────────────────────────────────────────────────────────────────

    async def _pre_market_report(self) -> None:
        try:
            from pre_market_report import generate_pre_market_report
            await generate_pre_market_report()
        except Exception as exc:
            logger.error("[platform] Pre-market report failed: {}", exc)

    async def _morning_data_refresh(self) -> None:
        """Refresh all static daily data: levels, events, institutional flow, correlation matrix."""
        from tick_engine import tick_engine
        symbols = list(tick_engine._subscribers.keys()) if hasattr(tick_engine, "_subscribers") else []
        if not symbols:
            logger.info("[platform] Morning data refresh: no symbols yet — will retry after bot start")
            return
        results = []
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

    async def _kite_token_refresh(self) -> None:
        logger.info("[platform] Kite token refresh starting…")
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

    async def _auto_start_bot(self) -> None:
        if not settings.auto_start_strategies:
            logger.info("[platform] AUTO_START_STRATEGIES not set — skipping auto-start")
            return

        from master_agent_v5 import master
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


platform_scheduler = PlatformScheduler()
