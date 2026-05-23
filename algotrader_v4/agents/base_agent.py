"""
agents/base_agent.py  (v3 — tick-driven)
Every agent runs a continuous asyncio loop consuming MarketSnapshots
from the TickEngine queue. Strategies evaluate on every live tick.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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


# ── TSL callback registry ────────────────────────────────────────────────────
# Maps main order_id → {sl_order_id, product, exchange}
_tsl_sl_orders: dict[str, dict] = {}
_tsl_callbacks_installed: bool = False


def _setup_tsl_callbacks() -> None:
    """Wire global TSL callbacks once (idempotent). Must be called before first register()."""
    global _tsl_callbacks_installed
    if _tsl_callbacks_installed:
        return
    _tsl_callbacks_installed = True

    async def _on_sl_moved(pos, old_sl: float, move_type: str) -> None:
        entry = _tsl_sl_orders.get(pos.order_id)
        if not entry:
            return
        sl_oid = entry["sl_order_id"]
        try:
            kite_client.modify_order(order_id=sl_oid, trigger_price=pos.current_sl)
        except Exception:
            try:
                kite_client.cancel_order(sl_oid)
            except Exception:
                pass
            new_sl_oid = kite_client.place_order(
                tradingsymbol=pos.symbol,
                exchange=entry.get("exchange", "NSE"),
                transaction_type="SELL" if pos.side == "BUY" else "BUY",
                quantity=pos.quantity, order_type="SL-M",
                product=entry.get("product", "MIS"),
                trigger_price=pos.current_sl,
                tag=f"TSL-{pos.strategy}",
            )
            entry["sl_order_id"] = new_sl_oid

    async def _on_sl_hit(pos, ltp: float, pnl: float) -> None:
        entry = _tsl_sl_orders.pop(pos.order_id, None)
        if entry and entry.get("sl_order_id"):
            try:
                kite_client.cancel_order(entry["sl_order_id"])
            except Exception:
                pass
        kite_client.place_order(
            tradingsymbol=pos.symbol,
            exchange=entry.get("exchange", "NSE") if entry else "NSE",
            transaction_type="SELL" if pos.side == "BUY" else "BUY",
            quantity=pos.quantity, order_type="MARKET",
            product=entry.get("product", "MIS") if entry else "MIS",
            tag=f"TSL-HIT-{pos.strategy}",
        )
        order_guard.release_order(pos.symbol, pos.strategy, pos.side, pnl)
        risk_manager.record_trade(pnl)
        risk_manager.position_closed()
        trailing_sl_engine.deregister(pos.order_id)

    async def _on_target_hit(pos, ltp: float, level: int) -> None:
        if level == 2:
            entry = _tsl_sl_orders.pop(pos.order_id, None)
            kite_client.place_order(
                tradingsymbol=pos.symbol,
                exchange=entry.get("exchange", "NSE") if entry else "NSE",
                transaction_type="SELL" if pos.side == "BUY" else "BUY",
                quantity=pos.quantity, order_type="MARKET",
                product=entry.get("product", "MIS") if entry else "MIS",
                tag=f"TSL-T{level}-{pos.strategy}",
            )

    trailing_sl_engine.on_sl_moved   = _on_sl_moved
    trailing_sl_engine.on_sl_hit     = _on_sl_hit
    trailing_sl_engine.on_target_hit = _on_target_hit


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
    # HIGH-4: use python-telegram-bot library — keeps token out of URL paths/logs
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return
    try:
        from telegram import Bot
        bot = Bot(token=settings.telegram_bot_token)
        async with bot:
            await bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=text, parse_mode="HTML",
            )
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

    # ── Backtest filter ───────────────────────────────────────────────────

    def filter_watchlist(self, watchlist: list[dict]) -> list[dict]:
        # Fast path: pre-learned system — skip per-symbol backtests
        if settings.skip_startup_backtest:
            approved_path = Path("logs/approved_symbols.json")
            if approved_path.exists():
                import json
                pre = json.loads(approved_path.read_text()).get(self.name, [])
                pre_set = set(pre)
                approved = [i for i in watchlist if i["symbol"] in pre_set] if pre_set \
                           else list(watchlist)
                label = f"pre-learned ({len(pre_set)} approved symbols on file)"
            else:
                # No file yet — approve everything (user trusts their watchlist)
                approved = list(watchlist)
                label = "skip_backtest=true, no seed file — approving all"

            for item in approved:
                self._approved.add(item["symbol"])
            self.state.approved_symbols = [a["symbol"] for a in approved]
            logger.info("[{}] {} | {} symbols ready", self.name, label, len(approved))
            return approved

        # Normal path: run backtest per symbol (first-time setup)
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

    # ── Lifecycle ──────────────────────────────────────────────────────

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

    # ── Main tick loop ──────────────────────────────────────────────────

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
                    # ── Multi-timeframe alignment ──────────────────────
                    if settings.use_multi_timeframe:
                        from multi_timeframe import check as mtf_check
                        mtf = mtf_check(snap, action)
                        if not mtf.aligned:
                            logger.debug("[{}] {} MTF skip {}/3 TFs aligned",
                                         self.name, snap.symbol, mtf.score)
                            continue

                    # ── Event calendar hard block (CRITICAL = <1h to results/RBI) ──
                    from event_calendar import get_event_risk
                    _evt = get_event_risk(snap.symbol)
                    if _evt["size_factor"] == 0.0:
                        logger.debug("[{}] {} event BLOCK: {}",
                                     self.name, snap.symbol, _evt["description"])
                        continue

                    # ── Claude per-trade intelligence gate ────────────────────
                    if settings.use_claude_trade_gate:
                        from claude_trade_gate import assess as gate_assess
                        from master_agent_v5 import record_gate_decision
                        gate = await gate_assess(snap, action, signal, self.name)
                        record_gate_decision(gate.enter)
                        if not gate.enter:
                            continue
                        # Apply Claude's SL/target/size adjustments
                        if gate.adjusted_sl_pct:
                            signal["stop_loss_pct"]  = gate.adjusted_sl_pct
                        if gate.adjusted_target_pct:
                            signal["target_pct"]     = gate.adjusted_target_pct
                        signal["_gate_size_factor"]  = gate.size_factor
                        signal["_gate_confidence"]   = gate.confidence

                    # ── Compound size factors: event elevation + correlation ──
                    _sf = signal.get("_gate_size_factor", 1.0)
                    if _evt["size_factor"] < 1.0:
                        _sf = round(_sf * _evt["size_factor"], 3)

                    from correlation_guard import check as _corr_check
                    _open_syms = [
                        p["tradingsymbol"] for p in kite_client.positions().get("net", [])
                        if p.get("quantity", 0) != 0
                    ]
                    _corr = _corr_check(snap.symbol, _open_syms)
                    if not _corr["allowed"]:
                        logger.debug("[{}] {} corr BLOCK: {}",
                                     self.name, snap.symbol, _corr["reason"])
                        continue
                    if _corr["size_factor"] < 1.0:
                        _sf = round(_sf * _corr["size_factor"], 3)
                    signal["_gate_size_factor"] = _sf

                    await self._try_enter(snap, action, signal)
            except Exception as exc:
                err = f"{snap.symbol}: {str(exc)[:100]}"
                self.state.errors.append(err)

    # ── Entry ───────────────────────────────────────────────────────────

    async def _try_enter(self, snap: MarketSnapshot, action: str, signal: dict) -> None:
        sym  = snap.symbol
        ltp  = snap.tick.ltp
        exch = signal.get("exchange", "NSE")
        qty  = risk_manager.calculate_quantity(ltp, agent=self.name)

        # Apply Kelly-adjusted size (gate may have set a size_factor)
        size_factor = signal.pop("_gate_size_factor", 1.0)
        if settings.use_kelly_sizing and size_factor < 1.0:
            qty = max(1, int(qty * size_factor))

        allowed, reason = order_guard.can_place(sym, self.name, action)
        if not allowed:
            return
        if order_guard.is_symbol_active_anywhere(sym):
            return
        allowed, _ = risk_manager.check_before_order(sym, qty, ltp, action)
        if not allowed:
            return

        # LOW-2: SEBI pre-order compliance check
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector
        sebi_ok, algo_id, sebi_reason = sebi_compliance.pre_order_check(
            strategy=self.name, symbol=sym, exchange=exch,
            transaction_type=action, quantity=qty,
            order_type="MARKET", price_at_signal=ltp,
            signal_source=f"agent_{self.name}",
            regime=regime_detector.current_regime.value,
        )
        if not sebi_ok:
            logger.warning("[{}] SEBI blocked {} {}: {}", self.name, action, sym, sebi_reason)
            return

        order_id = kite_client.place_order(
            tradingsymbol=sym, exchange=exch,
            transaction_type=action, quantity=qty,
            order_type="MARKET", product=signal.get("product", self.product),
            tag=f"Agent-{self.name}",
        )
        sebi_compliance.record_order_id(self.name, sym, order_id)
        order_guard.register_order(sym, self.name, action, order_id)
        risk_manager.position_opened()
        self.state.trades_today  += 1
        self.state.signals_fired += 1
        self.state.last_signal    = signal

        # Wire TSL callbacks (idempotent — only installs once globally)
        _setup_tsl_callbacks()

        # Register with trailing SL engine (monitors every tick)
        trailing_sl_engine.register(
            symbol=sym, strategy=self.name, side=action,
            entry_price=ltp, quantity=qty, order_id=order_id,
            atr=snap.indicators.atr_14,
        )

        sl = signal.get("stop_loss", risk_manager.sl_price(ltp, action))
        product = signal.get("product", self.product)
        sl_order_id = kite_client.place_order(
            tradingsymbol=sym, exchange=exch,
            transaction_type="SELL" if action == "BUY" else "BUY",
            quantity=qty, order_type="SL-M",
            product=product,
            trigger_price=sl, tag=f"Agent-{self.name}-SL",
        )

        # Register SL-M order so TSL callbacks can modify/cancel it
        _tsl_sl_orders[order_id] = {
            "sl_order_id": sl_order_id,
            "product":     product,
            "exchange":    exch,
        }

        ind = snap.indicators
        await send_telegram(
            f"<b>[{self.name.upper()}]</b> {action} {sym} @ ₹{ltp:.2f}\n"
            f"Qty: {qty} | SL: ₹{sl:.2f} | Target: ₹{signal.get('target', 0):.2f}\n"
            f"RSI: {ind.rsi_14:.1f} | Trend: {ind.trend} | Vol: {ind.volume_ratio:.1f}x\n"
            f"Order: {order_id}"
        )
        from n8n_bridge import notify as _n8n
        asyncio.create_task(_n8n("trade_entry", {
            "agent":     self.name,
            "symbol":    sym,
            "action":    action,
            "price":     ltp,
            "quantity":  qty,
            "stop_loss": sl,
            "target":    signal.get("target", 0),
            "order_id":  order_id,
            "pattern":   signal.get("trigger", ""),
            "rsi":       ind.rsi_14,
            "trend":     ind.trend,
            "vol_ratio": ind.volume_ratio,
        }))

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
            _dot = "\U0001f534" if pnl < 0 else "\U0001f7e2"
            await send_telegram(
                f"{_dot} <b>[{self.name.upper()}]</b> EXIT {sym}\n"
                f"Reason: {reason} | P&L: ₹{pnl:.0f}"
            )
            from n8n_bridge import notify as _n8n
            asyncio.create_task(_n8n("trade_exit", {
                "agent":  self.name,
                "symbol": sym,
                "reason": reason,
                "pnl":    pnl,
                "side":   side,
            }))
            try:
                from trade_memory import record_trade as _record_trade
                from market_regime import regime_detector
                asyncio.create_task(_record_trade(
                    {"symbol": sym, "strategy": self.name, "side": side,
                     "pnl": pnl, "exit_reason": reason},
                    market_context={
                        "regime": regime_detector.current_regime.value
                        if regime_detector.current_regime else "unknown",
                    },
                ))
            except Exception:
                pass
            break

    # ── Utils ───────────────────────────────────────────────────────────

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
