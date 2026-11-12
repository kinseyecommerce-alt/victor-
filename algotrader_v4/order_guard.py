"""
order_guard.py
Central gate between every agent and the broker.
  1. Block duplicate orders
  2. Enforce per-strategy trade limits per day
  3. Enforce cooldown period after a losing trade
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

from loguru import logger
from config import settings


@dataclass
class ActiveOrder:
    symbol: str
    strategy: str
    side: str
    order_id: str
    placed_at: float


# The AI trader's order-guard keys (ai_trader.AGENT / *_STRATEGY). Its daily budget is AI_AGENT_MAX_TRADES_PER_DAY,
# which the AI enforces (and reports to the model) across all of them; the guard fell back to 10/day here and
# hard-blocked the AI long before its own limit. The per-strategy post-loss cooldown is for the mechanical agents:
# the AI picks unrelated names on its own evidence, and the model was never told of the block.
AI_GUARD_KEYS = frozenset({"ai", "ai_opt", "ai_fut", "ai_swing", "ai_swing_opt", "ai_spread", "momentum"})


class OrderGuard:

    def __init__(self) -> None:
        self._lock = Lock()
        self._active: dict[tuple[str, str, str], ActiveOrder] = {}
        self._trade_count: dict[str, int] = defaultdict(int)
        self._cooldown_until: dict[str, float] = defaultdict(float)

    @staticmethod
    def _max_trades(strategy: str) -> int:
        if strategy in AI_GUARD_KEYS:
            return settings.ai_agent_max_trades_per_day
        return {
            "intraday": settings.max_trades_intraday,
            "fno":      settings.max_trades_fno,
            "swing":    settings.max_trades_swing,
            "scalping": settings.max_trades_scalping,
        }.get(strategy, 10)

    def can_place(self, symbol: str, strategy: str, side: str) -> tuple[bool, str]:
        with self._lock:
            key = (symbol, strategy, side)
            if key in self._active:
                return False, f"Duplicate blocked: {side} {symbol} already active for {strategy}"
            reverse_key = (symbol, strategy, "SELL" if side == "BUY" else "BUY")
            if reverse_key in self._active:
                return False, f"Conflicting position: {self._active[reverse_key].side} {symbol} already active"
            limit = self._max_trades(strategy)
            if self._trade_count[strategy] >= limit:
                return False, f"Overtrade blocked: {strategy} already at {limit} trades today"
            now = time.time()
            if strategy not in AI_GUARD_KEYS and now < self._cooldown_until[strategy]:
                remaining = int(self._cooldown_until[strategy] - now)
                return False, f"Cooldown active for {strategy}: {remaining}s remaining"
            return True, "OK"

    def register_order(self, symbol: str, strategy: str, side: str, order_id: str) -> None:
        with self._lock:
            key = (symbol, strategy, side)
            self._active[key] = ActiveOrder(symbol=symbol, strategy=strategy, side=side,
                                             order_id=order_id, placed_at=time.time())
            self._trade_count[strategy] += 1

    def restore_active(self, symbol: str, strategy: str, side: str, order_id: str) -> None:
        """Re-mark a position held from an earlier day (after reset_daily or a restart)
        without counting it as a new trade today."""
        with self._lock:
            self._active.setdefault((symbol, strategy, side), ActiveOrder(
                symbol=symbol, strategy=strategy, side=side, order_id=order_id, placed_at=time.time()))

    def release_order(self, symbol: str, strategy: str, side: str, pnl: float = 0.0) -> None:
        with self._lock:
            key = (symbol, strategy, side)
            self._active.pop(key, None)
            if pnl < 0 and settings.cooldown_after_loss_sec > 0 and strategy not in AI_GUARD_KEYS:
                self._cooldown_until[strategy] = time.time() + settings.cooldown_after_loss_sec

    def is_symbol_active_anywhere(self, symbol: str) -> list[str]:
        with self._lock:
            return [ao.strategy for (sym, strat, side), ao in self._active.items() if sym == symbol]

    def reset_daily(self) -> None:
        with self._lock:
            self._active.clear()
            self._trade_count.clear()
            self._cooldown_until.clear()

    def status(self) -> dict:
        with self._lock:
            return {
                "active_orders": {
                    f"{k[0]}|{k[1]}|{k[2]}": {"order_id": v.order_id, "placed_at": v.placed_at}
                    for k, v in self._active.items()
                },
                "trades_today": dict(self._trade_count),
                "cooldowns": {
                    k: max(0, int(v - time.time()))
                    for k, v in self._cooldown_until.items() if v > time.time()
                },
            }


order_guard = OrderGuard()