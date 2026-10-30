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
# Maps entry order_id → {sl_order_id, product, exchange, symbol, guard_symbol, agent, side}
#   symbol       — the traded instrument (equity, or the option leg for F&O)
#   guard_symbol — the key order_guard tracks (the underlying for F&O)
# Options add:  sl_type="SL" + sl_buffer_pct (stop-loss LIMIT, not SL-M, which
#   brokers reject on options) and exit_limit_buffer_pct (exits as marketable LIMIT)
_tsl_sl_orders: dict[str, dict] = {}
_tsl_callbacks_installed: bool = False


def sl_limit_price(trigger: float, side: str, buffer_pct: float) -> float:
    """Limit price for a stop-loss LIMIT order protecting a `side` position: worse than the
    trigger by buffer_pct so it still fills on a fast move."""
    from options_resolver import round_tick
    return round_tick(trigger * (1 - buffer_pct / 100) if side == "BUY" else trigger * (1 + buffer_pct / 100))


def _exit_order(entry: dict, symbol: str, side: str, fallback_px: float = 0.0) -> dict:
    """MARKET exit, or a marketable LIMIT for instruments that need one (options).
    Reference price: last tick → live Kite quote → fallback_px (the position's own price)."""
    buf = entry.get("exit_limit_buffer_pct")
    if not buf:
        return {"order_type": "MARKET"}
    from tick_engine import tick_engine
    from options_resolver import round_tick
    ref = 0.0
    tick, _ = tick_engine.latest(symbol)
    if tick and tick.ltp > 0:
        ref = (tick.bid or tick.ltp) if side == "BUY" else (tick.ask or tick.ltp)
    if ref <= 0:
        try:
            q = kite_client.quote_kite([f"{entry.get('exchange', 'NFO')}:{symbol}"])
            ref = float(next(iter(q.values()), {}).get("last_price") or 0)
        except Exception:
            ref = 0.0
    ref = ref or fallback_px
    if ref <= 0:
        return {"order_type": "MARKET"}
    px = ref * (1 - buf / 100) if side == "BUY" else ref * (1 + buf / 100)
    return {"order_type": "LIMIT", "price": round_tick(max(px, 0.05))}


def _close_tracked(entry_oid: str, symbol: str, side: str, quantity: int,
                   pnl: float, tag: str, strategy: str) -> None:
    """Fully close a tracked position: cancel its resting SL-M, exit at market,
    and release every piece of bookkeeping. Safe to call once per position."""
    entry = _tsl_sl_orders.pop(entry_oid, None) or {}
    tracked = trailing_sl_engine._positions.get(entry_oid)
    fallback_px = tracked.entry_price if tracked else 0.0
    sl_oid = entry.get("sl_order_id")
    if sl_oid:
        # Must go before the exit: a leftover SL-M would later fire on a flat
        # position and open an unintended reverse position
        try:
            kite_client.cancel_order(sl_oid)
        except Exception as exc:
            logger.error("[{}] could not cancel SL-M {} for {}: {}", strategy, sl_oid, symbol, exc)
    try:
        kite_client.place_order(
            tradingsymbol=symbol,
            exchange=entry.get("exchange", "NSE"),
            transaction_type="SELL" if side == "BUY" else "BUY",
            quantity=quantity,
            product=entry.get("product", "MIS"),
            tag=tag,
            **_exit_order(entry, symbol, side, fallback_px),
        )
    except Exception as exc:
        logger.critical("[{}] exit order for {} qty={} FAILED: {}", strategy, symbol, quantity, exc)
    order_guard.release_order(entry.get("guard_symbol", symbol), strategy, side, pnl)
    risk_manager.record_trade(pnl)
    risk_manager.position_closed()
    trailing_sl_engine.deregister(entry_oid)


def _setup_tsl_callbacks() -> None:
    """Wire global TSL callbacks once (idempotent). Must be called before first register()."""
    global _tsl_callbacks_installed
    if _tsl_callbacks_installed:
        return
    _tsl_callbacks_installed = True

    async def _on_sl_moved(pos, old_sl: float, move_type: str) -> None:
        entry = _tsl_sl_orders.get(pos.order_id)
        if not entry or not entry.get("sl_order_id"):
            return
        sl_oid = entry["sl_order_id"]
        sl_type = entry.get("sl_type", "SL-M")
        limit = (sl_limit_price(pos.current_sl, pos.side, entry.get("sl_buffer_pct", 5.0))
                 if sl_type == "SL" else 0.0)
        try:
            kite_client.modify_order(order_id=sl_oid, trigger_price=pos.current_sl, price=limit)
        except Exception:
            try:
                kite_client.cancel_order(sl_oid)
            except Exception:
                pass
            new_sl_oid = kite_client.place_order(
                tradingsymbol=pos.symbol,
                exchange=entry.get("exchange", "NSE"),
                transaction_type="SELL" if pos.side == "BUY" else "BUY",
                quantity=pos.quantity, order_type=sl_type,
                product=entry.get("product", "MIS"),
                trigger_price=pos.current_sl, price=limit,
                tag=f"TSL-{pos.strategy}",
            )
            entry["sl_order_id"] = new_sl_oid

    async def _on_sl_hit(pos, ltp: float, pnl: float) -> None:
        _close_tracked(pos.order_id, pos.symbol, pos.side, pos.quantity, pnl,
                       f"TSL-HIT-{pos.strategy}", pos.strategy)

    async def _on_target_hit(pos, ltp: float, level: int) -> None:
        if level == 2:
            pnl = (ltp - pos.entry_price) * pos.quantity * (1 if pos.side == "BUY" else -1)
            _close_tracked(pos.order_id, pos.symbol, pos.side, pos.quantity, pnl,
                           f"TSL-T{level}-{pos.strategy}", pos.strategy)

    trailing_sl_engine.on_sl_moved   = _on_sl_moved
    trailing_sl_engine.on_sl_hit     = _on_sl_hit
    trailing_sl_engine.on_target_hit = _on_target_hit


# ── Entry protection ─────────────────────────────────────────────────────────
SL_PLACE_RETRIES = 3
SL_RETRY_DELAY_S = 0.2


async def place_sl_with_retry(
    agent: str, symbol: str, exchange: str, side: str,
    quantity: int, product: str, trigger_price: float,
    order_type: str = "SL-M", limit_buffer_pct: float = 5.0,
) -> Optional[str]:
    """Place the stop for a just-entered position — SL-M, or a stop-loss LIMIT ("SL") for
    options. Returns the SL order id, or None if every attempt failed."""
    price = sl_limit_price(trigger_price, side, limit_buffer_pct) if order_type == "SL" else 0.0
    for attempt in range(1, SL_PLACE_RETRIES + 1):
        try:
            return kite_client.place_order(
                tradingsymbol=symbol, exchange=exchange,
                transaction_type="SELL" if side == "BUY" else "BUY",
                quantity=quantity, order_type=order_type, product=product,
                trigger_price=trigger_price, price=price, tag=f"Agent-{agent}-SL",
            )
        except Exception as exc:
            logger.error("[{}] SL-M for {} failed (attempt {}/{}): {}",
                         agent, symbol, attempt, SL_PLACE_RETRIES, exc)
            if attempt < SL_PLACE_RETRIES:
                await asyncio.sleep(SL_RETRY_DELAY_S)
    return None


async def flatten_unprotected(
    agent: str, symbol: str, exchange: str, side: str,
    quantity: int, product: str, exit_limit_price: float = 0.0,
) -> bool:
    """Close a position whose SL-M could not be placed. Returns True if the exit order went through.
    exit_limit_price > 0 exits with a LIMIT order (options) instead of MARKET."""
    try:
        kite_client.place_order(
            tradingsymbol=symbol, exchange=exchange,
            transaction_type="SELL" if side == "BUY" else "BUY",
            quantity=quantity, product=product,
            order_type="LIMIT" if exit_limit_price > 0 else "MARKET", price=exit_limit_price,
            tag=f"Agent-{agent}-NOSL-EXIT",
        )
        logger.critical("[{}] SL-M rejected for {} — position closed at market", agent, symbol)
        await send_telegram(f"⚠️ <b>[{agent.upper()}]</b> SL-M rejected for {symbol} — "
                            f"entry closed at market")
        return True
    except Exception as exc:
        logger.critical("[{}] SL-M AND exit both failed for {} qty={}: {} — "
                        "position is OPEN without exchange SL, software TSL only",
                        agent, symbol, quantity, exc)
        await send_telegram(f"🚨 <b>[{agent.upper()}]</b> {symbol} qty={quantity} is OPEN "
                            f"with NO exchange stop-loss. Check Kite now.")
        return False


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
        # Instruments this agent holds but doesn't trade signals on (F&O option legs):
        # their ticks only drive exits
        self._exit_only: set[str] = set()

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

            if snap.symbol in self._exit_only:
                try:
                    await trailing_sl_engine.on_tick(
                        snap.symbol, snap.tick.ltp, snap.indicators.atr_14)
                    await self._check_exits_on_tick(snap)
                except Exception as exc:
                    logger.error("[{}] exit tick error {}: {}", self.name, snap.symbol, exc)
                continue
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

        product = signal.get("product", self.product)
        order_id = kite_client.place_order(
            tradingsymbol=sym, exchange=exch,
            transaction_type=action, quantity=qty,
            order_type="MARKET", product=product,
            tag=f"Agent-{self.name}",
        )
        sebi_compliance.record_order_id(self.name, sym, order_id)

        # Protective SL-M goes in immediately; no position is kept without one
        sl = signal.get("stop_loss", risk_manager.sl_price(ltp, action))
        sl_order_id = await place_sl_with_retry(
            self.name, sym, exch, action, qty, product, sl)
        if sl_order_id is None and await flatten_unprotected(
                self.name, sym, exch, action, qty, product):
            return

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

        # Track the position so every exit path can cancel its SL-M and release it
        _tsl_sl_orders[order_id] = {
            "sl_order_id":  sl_order_id,
            "product":      product,
            "exchange":     exch,
            "symbol":       sym,
            "guard_symbol": sym,
            "agent":        self.name,
            "side":         action,
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
        """Strategy-rule exits (should_exit_position) for positions this agent opened."""
        sym = snap.symbol
        ind = snap.indicators
        tracked = next(((oid, e) for oid, e in _tsl_sl_orders.items()
                        if e.get("symbol") == sym and e.get("agent") == self.name), None)
        if tracked is None:
            return   # not ours — another agent's or a manual position
        entry_oid, entry = tracked
        for pos in kite_client.positions().get("net", []):
            if pos.get("tradingsymbol") != sym or pos.get("quantity", 0) == 0:
                continue
            should, reason = self.should_exit_position(pos, ind)
            if not should:
                continue
            side = "SELL" if pos["quantity"] > 0 else "BUY"
            qty  = abs(pos["quantity"])
            if settings.trading_mode == "LIVE" and "pnl" in pos:
                pnl = pos["pnl"]
            else:
                pnl = (snap.tick.ltp - pos.get("average_price", 0.0)) * pos["quantity"]
            _close_tracked(entry_oid, sym, entry.get("side", "BUY" if side == "SELL" else "SELL"),
                           qty, pnl, f"Agent-{self.name}-EXIT", self.name)
            self.state.pnl_today += pnl
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
