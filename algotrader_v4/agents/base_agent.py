"""
agents/base_agent.py  (v3 — tick-driven)
Every agent runs a continuous asyncio loop consuming MarketSnapshots
from the TickEngine queue. Strategies evaluate on every live tick.
"""
from __future__ import annotations

import asyncio
import time
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
# Delivery (CNC) swing positions add: gtt_id + gtt_buffer_pct — the stop is a GTT at
#   Zerodha instead of a day SL order, so it survives the close and server restarts
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
                   pnl: float, tag: str, strategy: str, send_exit: bool = True) -> bool:
    """Fully close a tracked position: cancel its resting SL-M, exit at market,
    and release every piece of bookkeeping. Safe to call once per position.
    send_exit=False: the broker position is already flat (its stop filled), only release the bookkeeping.
    Returns False when the exit order could not be placed (the position may still be open at the broker)."""
    entry = _tsl_sl_orders.pop(entry_oid, None) or {}
    tracked = trailing_sl_engine._positions.get(entry_oid)
    fallback_px = tracked.entry_price if tracked else 0.0
    gtt_id = entry.get("gtt_id")
    if gtt_id:
        # Same reason as the SL-M below: a GTT left behind would sell holdings later
        try:
            kite_client.delete_gtt(gtt_id)
        except Exception as exc:
            logger.error("[{}] could not delete GTT {} for {}: {}", strategy, gtt_id, symbol, exc)
    sl_oid = entry.get("sl_order_id")
    if sl_oid:
        # Must go before the exit: a leftover SL-M would later fire on a flat
        # position and open an unintended reverse position
        try:
            kite_client.cancel_order(sl_oid)
        except Exception as exc:
            logger.error("[{}] could not cancel SL-M {} for {}: {}", strategy, sl_oid, symbol, exc)
    exited = True
    if send_exit:
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
            exited = False
            logger.critical("[{}] exit order for {} qty={} FAILED: {}", strategy, symbol, quantity, exc)
    order_guard.release_order(entry.get("guard_symbol", symbol), strategy, side, pnl)
    risk_manager.record_trade(pnl)
    risk_manager.position_closed()
    trailing_sl_engine.deregister(entry_oid)
    try:
        from performance import performance
        performance.record_trade(strategy, symbol, side, quantity, fallback_px, pnl,
                                 entry.get("exchange", "NSE"), entry.get("product", "MIS"), note=tag,
                                 signal_price=entry.get("signal_price"), order_id=entry_oid)
    except Exception as exc:
        logger.error("[perf] trade not recorded for {}: {}", symbol, exc)
    _notify_agent_result(strategy, symbol, pnl)
    _book_changed()
    return exited


def close_tracked_for_squareoff(reason: str = "SquareOff", only: Optional[set] = None) -> list[str]:
    """The scheduled square-off: close every tracked intraday position (not delivery / swing holdings) through
    _close_tracked, so each lands in the ledger and releases its bookkeeping. kite_client.squareoff_all_positions
    alone flattened the book behind the agents' backs: the trade never reached the ledger and its tracking
    (stop order, trailing stop, order guard, open-position count) was left behind.

    The broker's net position is read once and bounds every exit: a tracked position whose symbol is already
    flat (its exchange stop filled between polls) is only released, never sold again (that would open a fresh
    reverse position), and an exit never exceeds the open net quantity. Returns the symbols whose exit order
    actually went out — the caller excludes only those from the broker-level flatten, so a failed exit is
    still flattened there. `only`: restrict to these entry order ids."""
    from tick_engine import tick_engine
    try:
        net_rows = kite_client.positions().get("net", []) or []
    except Exception as exc:
        logger.error("[squareoff] net positions unavailable ({}) — leaving every position to the broker flatten", exc)
        return []
    # the broker keeps one net row per (symbol, product); the paper book one per symbol
    by_product = settings.trading_mode != "PAPER"
    open_qty: dict[tuple, int] = {}
    for p in net_rows:
        key = (p.get("tradingsymbol"), (p.get("product") or "MIS") if by_product else "")
        open_qty[key] = open_qty.get(key, 0) + int(p.get("quantity") or 0)
    closed = []
    for oid, e in list(_tsl_sl_orders.items()):
        if only is not None and oid not in only:
            continue
        if e.get("product", "MIS") == "CNC" or e.get("kind") == "swing":
            continue
        pos = trailing_sl_engine._positions.get(oid)
        if pos is None:
            continue
        sign = 1 if pos.side == "BUY" else -1
        key = (pos.symbol, e.get("product", "MIS") if by_product else "")
        qty = min(int(pos.quantity), max(0, open_qty.get(key, 0) * sign))
        tick, _ = tick_engine.latest(pos.symbol)
        px = tick.ltp if tick and tick.ltp > 0 else pos.entry_price
        tag = f"SquareOff-{pos.strategy}"[:20]
        try:
            if qty <= 0:
                # flat at the broker already: the stop filled. Release the tracking at the stop price.
                stop_px = pos.current_sl if pos.current_sl and pos.current_sl > 0 else px
                pnl = (stop_px - pos.entry_price) * pos.quantity * sign
                logger.warning("[squareoff] {} {} is already flat at the broker — releasing its tracking, "
                               "no exit order", pos.strategy, pos.symbol)
                _close_tracked(oid, pos.symbol, pos.side, pos.quantity, pnl, tag, pos.strategy, send_exit=False)
                exited = False
            else:
                if qty < pos.quantity:
                    logger.warning("[squareoff] {} {} tracked qty {} but only {} open at the broker — exiting {}",
                                   pos.strategy, pos.symbol, pos.quantity, qty, qty)
                pnl = (px - pos.entry_price) * qty * sign
                open_qty[key] = open_qty.get(key, 0) - qty * sign
                exited = _close_tracked(oid, pos.symbol, pos.side, qty, pnl, tag, pos.strategy)
        except Exception as exc:
            logger.error("[squareoff] {} close failed: {}", pos.symbol, exc)
            continue
        if exited:
            closed.append(pos.symbol)            # its exit went out: the broker flatten must not send a second one
        try:
            from agents.strategy_agents import ALL_AGENTS
            a = ALL_AGENTS.get(e.get("agent"))
            if a is not None:
                a.state.pnl_today += pnl
        except Exception:
            pass
        logger.info("[squareoff] {} {} closed ({}) P&L ₹{:.0f}", pos.strategy, pos.symbol, reason, pnl)
    return closed


def reapply_overnight() -> int:
    """After the daily counter reset (master._daily_reset, paper_book.roll_day_if_due): positions held from
    earlier days (CNC delivery / swing, any agent) still exist — re-mark them in the order guard (a second
    entry on the same symbol stays blocked) and in the open-position count. The AI trader's own entries are
    re-applied by ai_trader.reapply_overnight. Returns the number of strategy-agent positions re-applied."""
    n = 0
    for oid, e in list(_tsl_sl_orders.items()):
        if e.get("agent") == "ai" or not (e.get("product", "MIS") == "CNC" or e.get("kind") == "swing"):
            continue
        pos = trailing_sl_engine._positions.get(oid)
        strategy = pos.strategy if pos else e.get("agent", "")
        order_guard.restore_active(e.get("guard_symbol", e.get("symbol")), strategy,
                                   e.get("side", pos.side if pos else "BUY"), oid)
        risk_manager.position_opened()
        n += 1
    try:
        from ai_trader import ai_trader
        ai_trader.reapply_overnight()
    except Exception as exc:
        logger.warning("[reset] AI overnight positions not re-applied: {}", exc)
    return n


def _book_changed() -> None:
    """A position closed: write the PAPER book now, so a restart a moment later cannot restore a trade
    the ledger already holds (no-op in LIVE, or before the startup restore ran)."""
    try:
        import paper_book
        paper_book.save()
    except Exception as exc:
        logger.debug("[paper_book] save after close failed: {}", exc)


_ENTRY_LOCK = asyncio.Lock()


def allocate_quantity(agent: str, qty: int, ltp: float) -> tuple[int, str]:
    """Scale a risk-based quantity by today's allocator factor (0–1.5) and cap it by the ₹ still
    free in the agent's capital bucket. Returns (quantity, reason) — quantity 0 means skip."""
    import allocator
    f = allocator.size_factor(agent)
    if f <= 0:
        return 0, f"allocator set {agent} to 0× today"
    if f != 1.0:
        qty = int(qty * f)
    if ltp <= 0:
        return max(0, qty), "no price"
    room = risk_manager.bucket_room(agent)
    cap = int(room // ltp)
    if cap < 1:
        return 0, f"capital bucket full (₹{risk_manager.bucket_exposure(agent):,.0f} deployed of ₹{risk_manager.bucket_total(agent):,.0f})"
    why = "ok" if f == 1.0 else f"allocator {f:g}×"
    if qty > cap:
        qty, why = cap, f"capped by bucket room ₹{room:,.0f}"
        # A remnant of the bucket funds a position whose flat costs (~₹40 a round trip) eat the edge:
        # skip a bucket-capped position under min_position_value_pct of the agent's normal size. Only
        # here: deliberate cuts (allocator, AI review, correlation) multiply together — 0.6 × 0.75 × 0.75
        # left scalping at a third of its size, under the floor, and it could not trade at all (11 Nov).
        floor = settings.min_position_value_pct / 100 * risk_manager.max_capital_for_agent(agent)
        if qty * ltp < floor:
            return 0, f"position ₹{qty * ltp:,.0f} under the ₹{floor:,.0f} minimum ({why})"
    return max(0, qty), why


def _notify_agent_result(strategy: str, symbol: str, pnl: float) -> None:
    """Agents that adapt within the day (ScalpV2Agent stops after a loss) learn the result here."""
    try:
        from agents.strategy_agents import ALL_AGENTS      # lazy: strategy_agents imports this module
        hook = getattr(ALL_AGENTS.get(strategy), "record_result", None)
        if hook:
            hook(symbol, pnl)
    except Exception as exc:
        logger.debug("[{}] record_result failed for {}: {}", strategy, symbol, exc)


def _setup_tsl_callbacks() -> None:
    """Wire global TSL callbacks once (idempotent). Must be called before first register()."""
    global _tsl_callbacks_installed
    if _tsl_callbacks_installed:
        return
    _tsl_callbacks_installed = True

    async def _on_sl_moved(pos, old_sl: float, move_type: str) -> None:
        entry = _tsl_sl_orders.get(pos.order_id)
        if entry and entry.get("gtt_id"):
            off = entry.get("gtt_trigger_offset_pct", 0.0) / 100
            trigger = round(pos.current_sl * (1 - off if pos.side == "BUY" else 1 + off), 2)
            try:
                kite_client.modify_gtt_stop(
                    entry["gtt_id"], pos.symbol, entry.get("exchange", "NSE"), pos.quantity,
                    trigger, sl_limit_price(trigger, pos.side, entry.get("gtt_buffer_pct", 1.0)),
                    pos.best_price or pos.entry_price, product=entry.get("product", "CNC"))
            except Exception as exc:
                logger.error("[{}] GTT {} update for {} failed: {} — software stop still active",
                             pos.strategy, entry["gtt_id"], pos.symbol, exc)
            return
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
    skip_counts:      dict  = field(default_factory=dict)    # why signals did not become orders
    last_skip:        str   = ""


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
        self._review_tasks: set = set()           # in-flight AI reviews (keep references)
        self._reviewing: set = set()              # symbols with a review in flight
        self._reviewed_at: dict = {}              # symbol → monotonic time of its last review attempt
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

    # Strategy-rule exits (should_exit_position) on every tick. The validated agents' edge comes from
    # holding to the structural stop or the 1.5-2.5R target; their "trend failed" rules were never part of
    # the validation and, replayed on 45 days of real bars, turn every one of them into a loser
    # (intraday +₹13,260 → −₹17,187, scalping +₹3,561 → −₹10,404, swing +₹3,527 → −₹8,836). With False,
    # only the intraday 15:00 square-off branch is consulted; stops, targets and the trailing engine apply.
    strategy_rule_exits: bool = True

    def rule_exit(self, pos: dict, ind) -> tuple[bool, str]:
        """The exit-rule policy shared by the live loop and the replay."""
        if not self.strategy_rule_exits:
            from datetime import time as _dtime
            if self.product != "MIS":
                return False, ""
            import agents.strategy_agents as _sa
            t = _sa.now_ist().time().replace(tzinfo=None)
            h, m = (int(x) for x in settings.squareoff_time.split(":"))
            if t >= _dtime(h, m):            # same time as the replay that validated these agents
                return True, f"Auto square-off {settings.squareoff_time}"
            return False, ""
        return self.should_exit_position(pos, ind)

    # Agents validated at strategy level on real 1-minute data (replay_backtest) trade their whole
    # watchlist; the legacy per-symbol backtest tests a different, older rule set and cannot fetch history
    # on the server ("Insufficient data (0 bars)" for every symbol), which left them with nothing to trade.
    validated_by_replay: bool = False

    def filter_watchlist(self, watchlist: list[dict]) -> list[dict]:
        try:
            from tick_engine import canonical_symbol           # INFOSYS → INFY etc.: ticks arrive canonical
            watchlist = [{**i, "symbol": canonical_symbol(i["symbol"])} for i in watchlist]
        except Exception:
            pass
        if self.validated_by_replay:
            approved = list(watchlist)
            self._approved.update(i["symbol"] for i in approved)
            self.state.approved_symbols = [a["symbol"] for a in approved]
            logger.info("[{}] validated by real-data replay — all {} watchlist symbols ready", self.name, len(approved))
            return approved
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
            no_data = not res.passed and res.fail_reasons and all("Insufficient data" in r for r in res.fail_reasons)
            if no_data:                      # missing history is not evidence against the symbol
                approved.append(item)
                self._approved.add(sym)
                logger.warning("[{}] {} no history for the approval backtest — approved", self.name, sym)
                continue
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

                    # ── AI review of the signal (claude_trade_gate) ─────────────
                    # Runs in its own task so a 5-25 s model call never stalls this loop
                    # (a burst of opening-range signals would otherwise be reviewed one by
                    # one and entered stale); the entry re-checks price drift afterwards.
                    if self._gate_applies():
                        sym = snap.symbol
                        # one review per symbol at a time, and none again within gate_review_cooldown_s:
                        # agents re-signal every tick/minute, and each review is a paid model call
                        if sym in self._reviewing or (time.monotonic() - self._reviewed_at.get(sym, -1e9)
                                                      < settings.gate_review_cooldown_s):
                            continue
                        self._reviewed_at[sym] = time.monotonic()
                        # never pay for a review of a trade the sizing/risk checks would refuse anyway
                        why = self._entry_precheck(snap, action, signal)
                        if why:
                            self._skip(sym, f"{why} (no AI review)")
                            continue
                        self._reviewing.add(sym)
                        task = asyncio.create_task(self._review_then_enter(snap, action, signal, _evt))
                        self._review_tasks.add(task)
                        task.add_done_callback(self._review_tasks.discard)
                        task.add_done_callback(lambda _t, _s=sym: self._reviewing.discard(_s))
                        continue
                    await self._finish_entry(snap, action, signal, _evt)
            except Exception as exc:
                err = f"{snap.symbol}: {str(exc)[:100]}"
                self.state.errors.append(err)
                # a crash here silently drops a validated signal: always make it visible
                logger.opt(exception=exc).warning("[{}] entry path error on {} — signal dropped", self.name, snap.symbol)

    # ── Entry ───────────────────────────────────────────────────────────

    def _gate_applies(self) -> bool:
        """The AI review runs for the strategies where replay showed it adds value (gate_strategies),
        and for every strategy on a day the allocator marked unusual."""
        import allocator as _alloc
        if _alloc.gate_required():
            return True
        if not settings.use_claude_trade_gate:
            return False
        allowed = {x.strip() for x in settings.gate_strategies.split(",") if x.strip()}
        return not allowed or self.name in allowed

    async def _review_then_enter(self, snap: MarketSnapshot, action: str, signal: dict, evt: dict) -> None:
        sym = snap.symbol
        try:
            import allocator as _alloc
            from claude_trade_gate import assess as gate_assess
            from master_agent_v5 import record_gate_decision
            gate = await gate_assess(snap, action, signal, self.name, force=_alloc.gate_required())
            record_gate_decision(gate.enter)
            if not gate.enter:
                return self._skip(sym, f"AI review: {gate.reason[:100]}")
            signal["_gate_size_factor"] = gate.size_factor
            signal["_gate_confidence"] = gate.confidence
            # the review took time: enter only if price has not run away or through the stop meanwhile
            from tick_engine import tick_engine as _te
            latest, _ = _te.latest(sym)
            if latest and latest.ltp > 0 and snap.tick.ltp > 0:
                drift = (latest.ltp / snap.tick.ltp - 1) * 100 * (1 if action == "BUY" else -1)
                stop = signal.get("stop_loss")
                through = stop and ((action == "BUY" and latest.ltp <= stop) or (action == "SELL" and latest.ltp >= stop))
                if drift > settings.gate_max_drift_pct or through:
                    return self._skip(sym, f"price moved {drift:+.2f}% during the AI review"
                                           + (" through the stop" if through else ""))
                import dataclasses as _dc
                snap = _dc.replace(snap, tick=latest)
            await self._finish_entry(snap, action, signal, evt)
        except Exception as exc:
            self.state.errors.append(f"{sym}: {str(exc)[:100]}")
            logger.opt(exception=exc).warning("[{}] review/entry error on {} — signal dropped", self.name, sym)

    def _entry_precheck(self, snap: MarketSnapshot, action: str, signal: dict) -> Optional[str]:
        """The side-effect-free part of _try_enter's sizing and risk checks for an equity entry: the reason
        it would be refused, or None. Run before the AI review so refused trades cost no model call."""
        try:                                         # trade limits / cooldowns apply to every entry path
            ok, reason = order_guard.can_place(snap.symbol, self.name, action)
            if not ok:
                return f"order guard: {reason}"
        except Exception as exc:
            logger.debug("[{}] order-guard pre-check failed on {}: {}", self.name, snap.symbol, exc)
        try:                                         # per-agent open-position cap (FNO_MAX_OPEN_POSITIONS)
            cap_reason = risk_manager.open_cap_reason(self.name)
            if cap_reason:
                return cap_reason
        except Exception as exc:
            logger.debug("[{}] open-cap pre-check failed on {}: {}", self.name, snap.symbol, exc)
        if signal.get("exchange", "NSE") != "NSE" or type(self)._try_enter is not BaseAgent._try_enter:
            return None                              # options/overridden entry paths size differently
        try:
            sym, ltp = snap.symbol, snap.tick.ltp
            qty, why = allocate_quantity(self.name, risk_manager.calculate_quantity(ltp, agent=self.name), ltp)
            if qty < 1:
                return why
            import portfolio_risk
            pct = signal.get("stop_loss_pct", settings.stop_loss_pct) / 100
            stop_px = signal.get("stop_loss") or (ltp * (1 - pct) if action == "BUY" else ltp * (1 + pct))
            pr = portfolio_risk.pre_entry(sym, action, qty, ltp, stop_px, kind="equity",
                                          underlying=signal.get("underlying"), agent=self.name)
            if not pr["ok"]:
                return f"portfolio risk: {pr['reason']}"
            if min(qty, pr["max_quantity"]) * ltp < settings.min_position_value_abs:
                return f"portfolio risk cut it to ₹{min(qty, pr['max_quantity']) * ltp:,.0f}, under the ₹{settings.min_position_value_abs:,.0f} minimum"
            if order_guard.is_symbol_active_anywhere(sym):
                return "symbol already active in another agent"
            ok, reason = order_guard.can_place(sym, self.name, action)
            if not ok:
                return f"order guard: {reason}"
        except Exception as exc:                     # a broken pre-check must not block the review
            logger.debug("[{}] entry pre-check failed on {}: {}", self.name, snap.symbol, exc)
        return None

    async def _finish_entry(self, snap: MarketSnapshot, action: str, signal: dict, evt: dict) -> None:
        """Compound the size factors (review, event elevation, correlation) and enter."""
        _sf = signal.get("_gate_size_factor", 1.0)
        if signal.pop("_extended", False):       # outside the replay-validated window (intraday_all_day)
            _sf = round(_sf * settings.extended_window_size_factor, 3)
        if evt["size_factor"] < 1.0:
            _sf = round(_sf * evt["size_factor"], 3)
        from correlation_guard import check as _corr_check
        _open_syms = [p["tradingsymbol"] for p in kite_client.positions().get("net", []) if p.get("quantity", 0) != 0]
        _corr = _corr_check(snap.symbol, _open_syms)
        if not _corr["allowed"]:
            return self._skip(snap.symbol, f"correlation: {_corr['reason']}")
        if _corr["size_factor"] < 1.0:
            _sf = round(_sf * _corr["size_factor"], 3)
        signal["_gate_size_factor"] = _sf
        async with _ENTRY_LOCK:                 # concurrent reviews must not race for the same bucket room
            await self._try_enter(snap, action, signal)

    def _skip(self, sym: str, reason: str) -> None:
        """Every signal that does not become an order says why (log + per-reason count in get_status)."""
        key = reason.split(":")[0][:40]
        self.state.skip_counts[key] = self.state.skip_counts.get(key, 0) + 1
        self.state.last_skip = f"{sym}: {reason}"[:160]
        logger.info("[{}] {} signal skipped — {}", self.name, sym, reason)

    async def _try_enter(self, snap: MarketSnapshot, action: str, signal: dict) -> None:
        sym  = snap.symbol
        ltp  = snap.tick.ltp
        exch = signal.get("exchange", "NSE")
        qty  = risk_manager.calculate_quantity(ltp, agent=self.name)

        # Apply Kelly-adjusted size (gate may have set a size_factor)
        size_factor = signal.pop("_gate_size_factor", 1.0)
        if settings.use_kelly_sizing and size_factor < 1.0:
            qty = max(1, int(qty * size_factor))

        # Today's allocation (AI capital manager) and the shared-bucket cap
        qty, why = allocate_quantity(self.name, qty, ltp)
        if qty < 1:
            return self._skip(sym, why)

        from performance import performance
        allowed, reason = performance.is_allowed(self.name)
        if not allowed:
            return self._skip(sym, f"pruned: {reason}")   # auto-pruned: losing after costs
        import portfolio_risk
        stop_px = signal.get("stop_loss") or (ltp * (1 - signal.get("stop_loss_pct", settings.stop_loss_pct) / 100)
                                              if action == "BUY" else ltp * (1 + signal.get("stop_loss_pct", settings.stop_loss_pct) / 100))
        pr = portfolio_risk.pre_entry(sym, action, qty, ltp, stop_px,
                                      kind="option" if exch == "NFO" else "equity",
                                      underlying=signal.get("underlying"), agent=self.name)
        if not pr["ok"]:
            return self._skip(sym, f"portfolio risk: {pr['reason']}")
        if qty > pr["max_quantity"]:
            logger.info("[{}] {} size {} → {} ({})", self.name, sym, qty, pr["max_quantity"], pr["reason"])
            qty = pr["max_quantity"]
            # Risk sizing is deliberate (₹ at risk per trade ÷ stop distance: a 3 % swing stop caps a position
            # at a third of the 1 %-risk budget's ₹10L); only a position too small to cover its flat costs is
            # skipped. The %-of-normal-size floor stopped every swing entry on 11 Nov.
            if exch == "NSE" and qty * ltp < settings.min_position_value_abs:
                return self._skip(sym, f"portfolio risk cut it to ₹{qty * ltp:,.0f}, under the ₹{settings.min_position_value_abs:,.0f} minimum")
        allowed, reason = order_guard.can_place(sym, self.name, action)
        if not allowed:
            return self._skip(sym, f"order guard: {reason}")
        if order_guard.is_symbol_active_anywhere(sym):
            return self._skip(sym, "symbol already active in another agent")
        allowed, reason = risk_manager.check_before_order(sym, qty, ltp, action)
        if not allowed:
            return self._skip(sym, f"risk: {reason}")

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
            return self._skip(sym, f"SEBI: {sebi_reason}")

        product = signal.get("product", self.product)
        from execution import entry_order, fill_after
        order_id = kite_client.place_order(
            tradingsymbol=sym, exchange=exch,
            transaction_type=action, quantity=qty,
            product=product, tag=f"Agent-{self.name}",
            **(entry_order(action, snap.tick, ltp) if exch == "NSE" else {"order_type": "MARKET"}),
        )
        sebi_compliance.record_order_id(self.name, sym, order_id)
        fill = fill_after(order_id, ltp)          # the real average price, for the stop, P&L and slippage

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
        tsl_pos = trailing_sl_engine.register(
            symbol=sym, strategy=self.name, side=action,
            entry_price=fill, quantity=qty, order_id=order_id,
            atr=snap.indicators.atr_14,
        )
        # Start the software stop where the exchange stop actually is. Otherwise the engine
        # tracks its own default % stop, and its first trail move could LOOSEN the real stop.
        tsl_pos.current_sl = round(sl, 2)

        # Track the position so every exit path can cancel its SL-M and release it
        _tsl_sl_orders[order_id] = {
            "sl_order_id":  sl_order_id,
            "product":      product,
            "exchange":     exch,
            "symbol":       sym,
            "guard_symbol": sym,
            "agent":        self.name,
            "side":         action,
            "signal_price": ltp,
        }
        _book_changed()                     # write the paper book now, not up to 3 s later

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
            should, reason = self.rule_exit(pos, ind)
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

    # ── Day state across a restart (paper_book.py) ─────────────────────────
    # Agents that remember what they did today (one ORB per symbol, losses per symbol) export it so a
    # mid-session restart does not let them repeat today's trades. Plain JSON values only.

    def export_day_state(self) -> dict:
        return {}

    def import_day_state(self, state: dict) -> None:
        return None

    @staticmethod
    def _dump_day_keyed(d: dict, day) -> list:
        """{(symbol, date): value} -> [[symbol, "YYYY-MM-DD", value]] for `day` only."""
        return [[k[0], k[1].isoformat(), v] for k, v in d.items()
                if isinstance(k, tuple) and len(k) == 2 and k[1] == day]

    @staticmethod
    def _load_day_keyed(rows) -> dict:
        from datetime import date as _date
        return {(r[0], _date.fromisoformat(r[1])): r[2] for r in rows or []}

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
            "skip_counts":      self.state.skip_counts,
            "last_skip":        self.state.last_skip,
        }
