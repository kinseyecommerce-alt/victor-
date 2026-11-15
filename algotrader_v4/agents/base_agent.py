"""
agents/base_agent.py — the minimal agent base shared by the scheduler-driven validated agents.

The tick-driven legacy machinery (tick queues, trade gates, trailing stops, brackets, order guard, risk-manager
buckets) was removed in the core rebuild: every kept strategy is a validated, PAPER-only agent driven by
platform_scheduler jobs (agents/validated_base.py). What remains here:
  * AgentState / BaseAgent: running flag, day counters, skip counts, errors, get_status();
  * send_telegram(): the notifier the kept parts use (no-op without TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID);
  * overnight_hold_qty(): positions held past the close (the 15:15 square-off leaves them alone).
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from loguru import logger

from config import settings


@dataclass
class AgentState:
    name: str
    running: bool = False
    trades_today: int = 0
    pnl_today: float = 0.0
    ticks_processed: int = 0
    signals_fired: int = 0
    approved_symbols: list = field(default_factory=list)
    last_signal: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    skip_counts: dict = field(default_factory=dict)    # why signals did not become orders
    last_skip: str = ""


async def send_telegram(text: str) -> None:
    """One HTML message to the configured chat; silently does nothing without a bot token / chat id."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return
    try:
        from telegram import Bot
        bot = Bot(token=settings.telegram_bot_token)
        async with bot:
            await bot.send_message(chat_id=settings.telegram_chat_id, text=text, parse_mode="HTML")
    except Exception:
        pass


def overnight_hold_qty() -> dict[tuple[str, str], int]:
    """Signed quantity of every validated position held overnight (attention CNC longs, strangle NRML legs, PEAD
    holdings), keyed by (symbol, product). The broker-level square-off must leave these alone."""
    held: dict[tuple[str, str], int] = {}
    try:
        from agents.strategy_agents import ALL_AGENTS
        for a in ALL_AGENTS.values():
            if getattr(a, "scheduled", False):
                for key, q in a.overnight_holdings().items():
                    held[key] = held.get(key, 0) + int(q)
    except Exception as exc:
        logger.error("[squareoff] validated overnight holdings unavailable: {}", exc)
    return held


def reapply_overnight() -> int:
    """Kept for callers of the old engine: nothing to re-apply (no order guard / risk counters any more)."""
    return 0


class BaseAgent(ABC):
    name: str = "base"
    product: str = "MIS"
    scheduled: bool = False

    def __init__(self) -> None:
        self.state = AgentState(name=self.name)

    def start(self, queue=None) -> None:
        self.state.running = True

    def stop(self) -> None:
        self.state.running = False

    def _skip(self, sym: str, reason: str) -> None:
        self.state.skip_counts[reason] = self.state.skip_counts.get(reason, 0) + 1
        self.state.last_skip = f"{sym}: {reason}"[:160]

    def reset_daily(self) -> None:
        self.state.trades_today = self.state.pnl_today = 0
        self.state.ticks_processed = self.state.signals_fired = 0
        self.state.errors.clear()

    # day state across a restart (paper_book.py); plain JSON values only
    def export_day_state(self) -> dict:
        return {}

    def import_day_state(self, state: dict) -> None:
        return None

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "running": self.state.running,
            "trades_today": self.state.trades_today,
            "pnl_today": round(self.state.pnl_today, 0),
            "ticks_processed": self.state.ticks_processed,
            "signals_fired": self.state.signals_fired,
            "approved_symbols": self.state.approved_symbols,
            "last_signal": self.state.last_signal,
            "errors": self.state.errors[-5:],
            "skip_counts": self.state.skip_counts,
            "last_skip": self.state.last_skip,
        }
