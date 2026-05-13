"""
agents/base_agent.py  (v3 — tick-driven)
Every agent runs a continuous asyncio loop consuming MarketSnapshots
from the TickEngine queue. Strategies evaluate on every live tick.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import httpx
from loguru import logger

from config import settings
from kite_client import kite_client
from risk_manager import risk_manager
from order_guard import order_guard
from backtest_engine import backtest_engine
from tick_engine import MarketSnapshot, LiveIndicators
from trailing_sl_engine import trailing_sl_engine, TrailingSLEngine
from atomic_bracket import atomic_bracket_engine


@dataclass
class AgentState:
    name:             str
    running:          bool  = False
    trades_today:     int   = 0
    pnl_today:        float = 0.0
    ticks_processed:  int   = 0
    signals_fired:    int   = 0
    approved_symbols: list  = field(default_factory=list)
    last_signal:      dict  = field(default_factory=dict)
    errors:           list  = field(default_factory=list)


async def send_telegram(text: str) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            await client.post(url, json={
                "chat_id": settings.telegram_chat_id,
                "text": text, "parse_mode": "HTML"
            })
        except Exception:
            pass


class BaseAgent(ABC):
    name:    str = "base"
    product: str = "MIS"
    min_candles_1min: int = 20

    def __init__(self) -> None:
        self.state   = AgentState(name=self.name)
        self._queue: Optional[asyncio.Queue] = None
        self._task:  Optional[asyncio.Task]  = None
        self._approved: set[str] = set()

    @abstractmethod
    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        """Return ("BUY"|"SELL"|"EXIT"|"HOLD", signal_dict|None)."""
        ...

    @abstractmethod
    def should_exit_position(self, position: dict, ind: LiveIndicators) -> tuple[bool, str]:
        ...

    # ── Backtest filter ───────────────────────────────────────────────

    def filter_watchlist(self, watchlist: list[dict]) -> list[dict]:
        approved = []
        for item in watchlist:
            sym, exch = item["symbol"], item.get("exchange", "NSE")
            res = backtest_engine.run(sym, exch, self.name)
            if res.passed:
                approved.append(item)
                self._approved.add(sym)
                logger.info("[{}] {} PASS (win={:.0f}% sharpe={:.2f})",
                            self.name, sym, res.win_rate, res.sharpe_ratio)
            else:
                logger.info("[{}] {} FAIL: {}", self.name, sym,
                            ", ".join(res.fail_reasons))
        self.state.approved_symbols = [a["symbol"] for a in approved]
        return approved

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self, queue: asyncio.Queue) -> None:
        self._queue = queue
        self.state.running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[{}] started (tick-driven)", self.name)

    def stop(self) -> None:
        self.state.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("[{}] stopped", self.name)

    # ── Main tick loop ────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while self.state.running:
            try:
                snap: MarketSnapshot = await asyncio.wait_for(
                    self._queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if snap.symbol not in self._approved:
                continue
            if len(snap.candles_1min) < self.min_candles_1min:
                continue

            self.state.ticks_processed += 1
            try:
                # Check trailing SL engine first (highest priority)
                await trailing_sl_engine.on_tick(
                    snap.symbol, snap.tick.ltp, snap.indicators.atr_14
                )
                await self._check_exits_on_tick(snap)
                action, signal = self.evaluate_tick(snap)
                if action in ("BUY", "SELL") and signal:
                    await self._try_enter(snap, action, signal)
            except Exception as exc:
                err = f"{snap.symbol}: {str(exc)[:100]}"
                self.state.errors.append(err)

    # ── Entry ─────────────────────────────────────────────────────────

    async def _try_enter(self, snap: MarketSnapshot, action: str, signal: dict) -> None:
        sym  = snap.symbol
        ltp  = snap.tick.ltp
        exch = signal.get("exchange", "NSE")
        qty  = risk_manager.calculate_quantity(ltp)

        allowed, reason = order_guard.can_place(sym, self.name, action)
        if not allowed:
            return
        if order_guard.is_symbol_active_anywhere(sym):
            return
        allowed, _ = risk_manager.check_before_order(sym, qty, ltp, action)
        if not allowed:
            return

        order_id = kite_client.place_order(
            tradingsymbol=sym, exchange=exch,
            transaction_type=action, quantity=qty,
            order_type="MARKET", product=signal.get("product", self.product),
            tag=f"Agent-{self.name}",
        )
        order_guard.register_order(sym, self.name, action, order_id)
        risk_manager.position_opened()
        self.state.trades_today  += 1
        self.state.signals_fired += 1
        self.state.last_signal    = signal

        # Register with trailing SL engine (monitors every tick)
        trailing_sl_engine.register(
            symbol=sym, strategy=self.name, side=action,
            entry_price=ltp, quantity=qty, order_id=order_id,
            atr=snap.indicators.atr_14,
        )

        sl = signal.get("stop_loss", risk_manager.sl_price(ltp, action))
        kite_client.place_order(
            tradingsymbol=sym, exchange=exch,
            transaction_type="SELL" if action == "BUY" else "BUY",
            quantity=qty, order_type="SL-M",
            product=signal.get("product", self.product),
            trigger_price=sl, tag=f"Agent-{self.name}-SL",
        )

        ind = snap.indicators
        await send_telegram(
            f"<b>[{self.name.upper()}]</b> {action} {sym} @ ₹{ltp:.2f}\n"
            f"Qty: {qty} | SL: ₹{sl:.2f} | Target: ₹{signal.get('target', 0):.2f}\n"
            f"RSI: {ind.rsi_14:.1f} | Trend: {ind.trend} | Vol: {ind.volume_ratio:.1f}x\n"
            f"Order: {order_id}"
        )

    # ── Exit ──────────────────────────────────────────────────────────

    async def _check_exits_on_tick(self, snap: MarketSnapshot) -> None:
        # Symbols managed by atomic bracket are handled by TSL engine callbacks
        # Only handle exits for positions NOT in atomic bracket
        sym = snap.symbol
        ind = snap.indicators
        for pos in kite_client.positions().get("net", []):
            if pos.get("tradingsymbol") != sym or pos.get("quantity", 0) == 0:
                continue
            should, reason = self.should_exit_position(pos, ind)
            if not should:
                continue
            side = "SELL" if pos["quantity"] > 0 else "BUY"
            qty  = abs(pos["quantity"])
            pnl  = pos.get("pnl", 0)
            oid  = kite_client.place_order(
                tradingsymbol=sym, exchange=pos.get("exchange", "NSE"),
                transaction_type=side, quantity=qty, order_type="MARKET",
                product=pos.get("product", self.product), tag=f"Agent-{self.name}-EXIT",
            )
            order_guard.release_order(sym, self.name, "BUY" if side == "SELL" else "SELL", pnl)
            risk_manager.record_trade(pnl)
            risk_manager.position_closed()
            self.state.pnl_today += pnl
            trailing_sl_engine.deregister(oid)
            await send_telegram(
                f"{'🔴' if pnl<0 else '🟢'} <b>[{self.name.upper()}]</b> EXIT {sym}\n"
                f"Reason: {reason} | P&L: ₹{pnl:.0f}"
            )
            break

    # ── Utils ─────────────────────────────────────────────────────────

    def reset_daily(self) -> None:
        self.state.trades_today = self.state.pnl_today = 0
        self.state.ticks_processed = self.state.signals_fired = 0
        self.state.errors.clear()

    def get_status(self) -> dict:
        return {
            "name":             self.name,
            "running":          self.state.running,
            "trades_today":     self.state.trades_today,
            "pnl_today":        round(self.state.pnl_today, 0),
            "ticks_processed":  self.state.ticks_processed,
            "signals_fired":    self.state.signals_fired,
            "approved_symbols": self.state.approved_symbols,
            "last_signal":      self.state.last_signal,
            "errors":           self.state.errors[-5:],
        }