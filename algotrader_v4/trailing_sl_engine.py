"""
trailing_sl_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-time Automated & Trailing Stop Loss engine.

Runs on every live tick (1-second cadence) for every open position.

Modes per position:
  FIXED      → initial hard SL, never moves
  BREAKEVEN  → SL moves to entry once profit > breakeven_pct
  TRAILING   → SL follows highest-profit price at trail_pct behind
  ATR_TRAIL  → SL trails by N × ATR (volatility-adaptive)

Per-strategy defaults:
  Scalping  → trail 0.15% behind, activates at 0.3% profit, ATR×0.8
  Intraday  → trail 0.50% behind, activates at 1.0% profit, ATR×1.2
  F&O       → trail 20%  of premium, activates at 30% premium gain
  Swing     → trail 1.0% behind, activates at 2.0% profit, ATR×1.5

Events emitted (logged + Telegram):
  • SL moved to breakeven      → "{sym} SL → breakeven ₹X (profit locked)"
  • Trailing SL tightened      → "{sym} TSL ₹old → ₹new (+₹Xk locked)"
  • SL hit, position closed    → "{sym} SL triggered ₹X | P&L ₹Y"
  • Target 1 hit, TSL activated→ "{sym} T1 ₹X hit — TSL now active"
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Optional, Callable

from loguru import logger


# ── Types ─────────────────────────────────────────────────────────────────────

class SLMode(str, Enum):
    FIXED      = "FIXED"       # static SL, never moves
    BREAKEVEN  = "BREAKEVEN"   # moves to entry once trigger_pct reached
    TRAILING   = "TRAILING"    # % trail behind best price
    ATR_TRAIL  = "ATR_TRAIL"   # ATR × multiplier trail


class SLStatus(str, Enum):
    ACTIVE    = "ACTIVE"       # watching
    HIT       = "HIT"          # SL triggered — position closed
    CANCELLED = "CANCELLED"    # position closed by target / manual


@dataclass
class TrailConfig:
    """Strategy-specific trailing SL configuration."""
    initial_sl_pct:    float       # % below entry for initial SL
    trail_pct:         float       # % behind best price when trailing
    breakeven_pct:     float       # profit % to move SL to breakeven
    activation_pct:    float       # profit % to start trailing
    target1_pct:       float       # first target %
    target2_pct:       float       # second target %
    mode:              SLMode = SLMode.TRAILING
    atr_multiplier:    float = 1.5  # used if mode=ATR_TRAIL


# Strategy trail configs
TRAIL_CONFIGS: dict[str, TrailConfig] = {
    "scalping": TrailConfig(
        initial_sl_pct  = 0.25,
        trail_pct       = 0.15,
        breakeven_pct   = 0.15,
        activation_pct  = 0.30,
        target1_pct     = 0.50,
        target2_pct     = 0.80,
        mode            = SLMode.ATR_TRAIL,
        atr_multiplier  = 0.8,
    ),
    "intraday": TrailConfig(
        initial_sl_pct  = 1.50,
        trail_pct       = 0.50,
        breakeven_pct   = 0.50,
        activation_pct  = 1.00,
        target1_pct     = 2.00,
        target2_pct     = 3.50,
        mode            = SLMode.TRAILING,
        atr_multiplier  = 1.2,
    ),
    "fno": TrailConfig(
        initial_sl_pct  = 35.0,   # % of premium
        trail_pct       = 20.0,
        breakeven_pct   = 15.0,
        activation_pct  = 30.0,
        target1_pct     = 50.0,
        target2_pct     = 80.0,
        mode            = SLMode.TRAILING,
        atr_multiplier  = 2.0,
    ),
    "swing": TrailConfig(
        initial_sl_pct  = 3.00,
        trail_pct       = 1.00,
        breakeven_pct   = 1.00,
        activation_pct  = 2.00,
        target1_pct     = 4.00,
        target2_pct     = 8.00,
        mode            = SLMode.ATR_TRAIL,
        atr_multiplier  = 1.5,
    ),
}


# ── Position SL tracker ────────────────────────────────────────────────────────

@dataclass
class PositionSL:
    """Live SL tracker for one open position."""
    symbol:         str
    strategy:       str
    side:           str           # BUY / SELL
    entry_price:    float
    quantity:       int
    order_id:       str

    # Dynamic state
    current_sl:     float         # current active SL level
    best_price:     float         # highest (BUY) or lowest (SELL) price seen
    trail_active:   bool  = False
    breakeven_hit:  bool  = False
    target1_hit:    bool  = False
    target2_hit:    bool  = False

    # Stats
    sl_moves:       int   = 0     # how many times TSL has tightened
    max_profit:     float = 0.0
    locked_profit:  float = 0.0   # profit locked by current SL level

    status:         SLStatus = SLStatus.ACTIVE
    opened_at:      float = field(default_factory=time.time)
    last_updated:   float = field(default_factory=time.time)

    # ATR snapshot (updated from tick indicators)
    atr:            float = 0.0

    @property
    def cfg(self) -> TrailConfig:
        return TRAIL_CONFIGS.get(self.strategy, TRAIL_CONFIGS["intraday"])

    @property
    def current_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == "BUY":
            return (self.best_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.best_price) / self.entry_price * 100

    @property
    def locked_profit_pct(self) -> float:
        """% profit locked by current SL."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "BUY":
            return (self.current_sl - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_sl) / self.entry_price * 100

    def to_dict(self) -> dict:
        return {
            "symbol":         self.symbol,
            "strategy":       self.strategy,
            "side":           self.side,
            "entry_price":    round(self.entry_price, 2),
            "quantity":       self.quantity,
            "current_sl":     round(self.current_sl, 2),
            "best_price":     round(self.best_price, 2),
            "trail_active":   self.trail_active,
            "breakeven_hit":  self.breakeven_hit,
            "target1_hit":    self.target1_hit,
            "target2_hit":    self.target2_hit,
            "sl_moves":       self.sl_moves,
            "max_profit_pct": round(self.current_pnl_pct, 2),
            "locked_profit_pct": round(self.locked_profit_pct, 2),
            "status":         self.status.value,
            "cfg": {
                "initial_sl_pct":  self.cfg.initial_sl_pct,
                "trail_pct":       self.cfg.trail_pct,
                "activation_pct":  self.cfg.activation_pct,
                "target1_pct":     self.cfg.target1_pct,
                "target2_pct":     self.cfg.target2_pct,
                "mode":            self.cfg.mode.value,
            },
        }


# ── Engine ─────────────────────────────────────────────────────────────────────

class TrailingSLEngine:
    """
    Monitors all open positions on every tick.
    Emits close callbacks when SL or target is hit.
    Updates current_sl in real time as price moves.
    """

    def __init__(self) -> None:
        self._positions: dict[str, PositionSL] = {}  # key = order_id
        self._lock = Lock()

        # Callbacks — set by BotEngine / agents
        self.on_sl_hit:      Optional[Callable] = None   # async (pos: PositionSL)
        self.on_target_hit:  Optional[Callable] = None   # async (pos: PositionSL, level: int)
        self.on_sl_moved:    Optional[Callable] = None   # async (pos: PositionSL, old_sl: float)

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        symbol:      str,
        strategy:    str,
        side:        str,
        entry_price: float,
        quantity:    int,
        order_id:    str,
        atr:         float = 0.0,
    ) -> PositionSL:
        """
        Register a new open position for trailing SL monitoring.
        Called immediately after order is confirmed.
        """
        cfg = TRAIL_CONFIGS.get(strategy, TRAIL_CONFIGS["intraday"])

        # Initial SL
        if side == "BUY":
            init_sl = round(entry_price * (1 - cfg.initial_sl_pct / 100), 2)
        else:
            init_sl = round(entry_price * (1 + cfg.initial_sl_pct / 100), 2)

        pos = PositionSL(
            symbol=symbol, strategy=strategy, side=side,
            entry_price=entry_price, quantity=quantity,
            order_id=order_id, current_sl=init_sl,
            best_price=entry_price, atr=atr,
        )

        with self._lock:
            self._positions[order_id] = pos

        logger.info(
            "TSL registered: {} {} {} entry=₹{:.2f} initial_sl=₹{:.2f} mode={}",
            strategy, side, symbol, entry_price, init_sl, cfg.mode.value
        )
        return pos

    def deregister(self, order_id: str) -> None:
        with self._lock:
            pos = self._positions.pop(order_id, None)
        if pos:
            logger.info("TSL deregistered: {} {}", pos.symbol, order_id)

    # ── Main tick handler ─────────────────────────────────────────────

    async def on_tick(
        self,
        symbol:  str,
        ltp:     float,
        atr_14:  float = 0.0,
    ) -> None:
        """
        Called on every 1-second tick for a symbol.
        Evaluates all active positions for that symbol.
        """
        positions_for_symbol = []
        with self._lock:
            positions_for_symbol = [
                p for p in self._positions.values()
                if p.symbol == symbol and p.status == SLStatus.ACTIVE
            ]

        for pos in positions_for_symbol:
            await self._evaluate(pos, ltp, atr_14)

    async def _evaluate(
        self, pos: PositionSL, ltp: float, atr_14: float
    ) -> None:
        cfg = pos.cfg
        old_sl = pos.current_sl

        # Update ATR if available
        if atr_14 > 0:
            pos.atr = atr_14

        # ── 1. Update best price ───────────────────────────────────────
        if pos.side == "BUY":
            if ltp > pos.best_price:
                pos.best_price = ltp
        else:
            if ltp < pos.best_price:
                pos.best_price = ltp

        profit_pct = pos.current_pnl_pct

        # ── 2. Check SL hit ────────────────────────────────────────────
        sl_hit = (pos.side == "BUY" and ltp <= pos.current_sl) or \
                 (pos.side == "SELL" and ltp >= pos.current_sl)

        if sl_hit:
            pos.status = SLStatus.HIT
            pnl = (ltp - pos.entry_price) * pos.quantity * (1 if pos.side == "BUY" else -1)
            logger.warning(
                "🔴 SL HIT: {} {} @ ₹{:.2f} | SL was ₹{:.2f} | P&L ₹{:.0f}",
                pos.symbol, pos.side, ltp, pos.current_sl, pnl
            )
            if self.on_sl_hit:
                await self.on_sl_hit(pos, ltp, pnl)
            return

        # ── 3. Target 2 hit ────────────────────────────────────────────
        if not pos.target2_hit:
            t2 = (cfg.target2_pct / 100)
            t2_price = pos.entry_price * (1 + t2 if pos.side == "BUY" else 1 - t2)
            if (pos.side == "BUY" and ltp >= t2_price) or \
               (pos.side == "SELL" and ltp <= t2_price):
                pos.target2_hit = True
                logger.info("🎯 TARGET 2 hit: {} @ ₹{:.2f}", pos.symbol, ltp)
                if self.on_target_hit:
                    await self.on_target_hit(pos, ltp, 2)

        # ── 4. Target 1 hit → tighten trail ───────────────────────────
        if not pos.target1_hit:
            t1 = (cfg.target1_pct / 100)
            t1_price = pos.entry_price * (1 + t1 if pos.side == "BUY" else 1 - t1)
            if (pos.side == "BUY" and ltp >= t1_price) or \
               (pos.side == "SELL" and ltp <= t1_price):
                pos.target1_hit = True
                # On T1 hit: tighten trail to half the normal trail
                if pos.side == "BUY":
                    tighter_sl = round(ltp * (1 - cfg.trail_pct / 200), 2)  # half trail
                    if tighter_sl > pos.current_sl:
                        pos.current_sl = tighter_sl
                        pos.sl_moves  += 1
                else:
                    tighter_sl = round(ltp * (1 + cfg.trail_pct / 200), 2)
                    if tighter_sl < pos.current_sl:
                        pos.current_sl = tighter_sl
                        pos.sl_moves  += 1
                logger.info(
                    "🎯 T1 hit: {} @ ₹{:.2f} → TSL tightened to ₹{:.2f}",
                    pos.symbol, ltp, pos.current_sl
                )
                if self.on_target_hit:
                    await self.on_target_hit(pos, ltp, 1)

        # ── 5. Breakeven ───────────────────────────────────────────────
        if not pos.breakeven_hit and profit_pct >= cfg.breakeven_pct:
            be_sl = pos.entry_price  # exactly entry = breakeven
            if pos.side == "BUY" and be_sl > pos.current_sl:
                pos.current_sl   = round(be_sl, 2)
                pos.breakeven_hit = True
                pos.sl_moves     += 1
                logger.info("✅ Breakeven locked: {} SL → ₹{:.2f}", pos.symbol, pos.current_sl)
                if self.on_sl_moved:
                    await self.on_sl_moved(pos, old_sl, "BREAKEVEN")
            elif pos.side == "SELL" and be_sl < pos.current_sl:
                pos.current_sl   = round(be_sl, 2)
                pos.breakeven_hit = True
                pos.sl_moves     += 1
                logger.info("✅ Breakeven locked: {} SL → ₹{:.2f}", pos.symbol, pos.current_sl)
                if self.on_sl_moved:
                    await self.on_sl_moved(pos, old_sl, "BREAKEVEN")

        # ── 6. Activate trailing ───────────────────────────────────────
        if not pos.trail_active and profit_pct >= cfg.activation_pct:
            pos.trail_active = True
            logger.info("🔄 Trailing SL activated: {} profit={:.1f}%", pos.symbol, profit_pct)

        # ── 7. Move trailing SL ────────────────────────────────────────
        if pos.trail_active:
            new_sl = self._compute_trail_sl(pos, ltp, atr_14)

            if new_sl is not None:
                # Only move SL in profit direction (never widen it)
                if pos.side == "BUY" and new_sl > pos.current_sl:
                    old = pos.current_sl
                    pos.current_sl  = new_sl
                    pos.sl_moves   += 1
                    locked = (new_sl - pos.entry_price) * pos.quantity
                    pos.locked_profit = locked
                    logger.info(
                        "📈 TSL moved: {} ₹{:.2f} → ₹{:.2f} | locked ₹{:.0f}",
                        pos.symbol, old, new_sl, locked
                    )
                    if self.on_sl_moved:
                        await self.on_sl_moved(pos, old, "TRAIL")

                elif pos.side == "SELL" and new_sl < pos.current_sl:
                    old = pos.current_sl
                    pos.current_sl = new_sl
                    pos.sl_moves  += 1
                    locked = (pos.entry_price - new_sl) * pos.quantity
                    pos.locked_profit = locked
                    logger.info(
                        "📈 TSL moved: {} ₹{:.2f} → ₹{:.2f} | locked ₹{:.0f}",
                        pos.symbol, old, new_sl, locked
                    )
                    if self.on_sl_moved:
                        await self.on_sl_moved(pos, old, "TRAIL")

        pos.last_updated = time.time()

    def _compute_trail_sl(
        self, pos: PositionSL, ltp: float, atr: float
    ) -> Optional[float]:
        cfg = pos.cfg

        if cfg.mode == SLMode.ATR_TRAIL and atr > 0:
            trail_dist = atr * cfg.atr_multiplier
        else:
            # Percentage trail — tighter after T1 hit
            pct = cfg.trail_pct / 200 if pos.target1_hit else cfg.trail_pct / 100
            trail_dist = pos.best_price * pct

        if pos.side == "BUY":
            return round(pos.best_price - trail_dist, 2)
        else:
            return round(pos.best_price + trail_dist, 2)

    # ── Queries ────────────────────────────────────────────────────────

    def get_position(self, order_id: str) -> Optional[PositionSL]:
        return self._positions.get(order_id)

    def all_positions(self) -> list[dict]:
        with self._lock:
            return [p.to_dict() for p in self._positions.values()
                    if p.status == SLStatus.ACTIVE]

    def status_summary(self) -> dict:
        with self._lock:
            active = [p for p in self._positions.values() if p.status == SLStatus.ACTIVE]
            return {
                "active_count":    len(active),
                "total_locked":    round(sum(p.locked_profit for p in active), 0),
                "positions":       [p.to_dict() for p in active],
                "sl_moves_today":  sum(p.sl_moves for p in self._positions.values()),
            }


# ── Singleton ──────────────────────────────────────────────────────────────────
trailing_sl_engine = TrailingSLEngine()
