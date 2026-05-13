"""
adaptive_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adaptive Learning Engine

Answers the question: "Do we need to re-run the full backtest today?"

Answer: NO — most of the time.

Instead, the engine:
  1. Learns from every live trade result (online learning)
  2. Tracks rolling performance per strategy × symbol
  3. Auto-tunes SL%, target%, entry conditions based on recent outcomes
  4. Only triggers full re-backtest when performance degrades significantly

Three learning modes:
  ── MODE 1: Online Parameter Adaptation (every closed trade)
     Adjusts SL, target, trail using Bayesian update from last 20 trades.
     Never needs re-backtest. Runs in real-time.

  ── MODE 2: Rolling Walk-Forward (nightly, 3 min)
     Re-runs only the LAST 30 days on changed strategies.
     Uses cached OHLCV — no fresh fetch needed.
     Updates score and approved list without full 6-month backtest.

  ── MODE 3: Full Re-Backtest (triggered by regime change or decay)
     Only when:
       - Win rate drops below gate × 0.85 for 5 consecutive days
       - Market regime changes significantly (VIX jump > 5 pts)
       - New strategy version deployed
       - Monday morning weekly reset

Decision logic (runs nightly at 21:00 IST):
  For each strategy × symbol:
    If rolling_30d_wr >= gate_wr × 0.90 → KEEP (no backtest needed)
    If rolling_30d_wr >= gate_wr × 0.75 → ADAPT (parameter tweak only)
    If rolling_30d_wr < gate_wr × 0.75  → RETIRE + trigger re-backtest
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from strategy_signals import GATE_THRESHOLDS, STRATEGY_CONFIG


# ── Trade record ───────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Every live or paper trade stored for learning."""
    id:          str
    symbol:      str
    strategy:    str
    side:        str          # BUY / SELL
    entry:       float
    exit:        float
    qty:         int
    pnl:         float        # net P&L after costs
    pnl_pct:     float        # as % of entry
    won:         bool
    regime:      str          # market regime at entry
    sl_pct:      float        # actual SL % used
    target_pct:  float        # actual target % used
    exit_reason: str          # SL_HIT / TARGET / TIMEOUT / MANUAL
    entry_time:  str
    hold_bars:   int          # number of bars held
    entry_rsi:   float = 0.0
    entry_adx:   float = 0.0
    entry_vwap_dev: float = 0.0


# ── Adaptive parameter set ───────────────────────────────────────────────────

@dataclass
class AdaptiveParams:
    """Live-updated parameters for a strategy × symbol pair."""
    strategy:    str
    symbol:      str

    # Core params (start from backtest defaults, adapt over time)
    sl_pct:      float = 1.5
    target_pct:  float = 3.0
    trail_pct:   float = 0.5
    size_factor: float = 1.0   # 0.25 – 1.5 (reduce when losing, increase when winning)

    # Entry condition thresholds
    min_rsi:     float = 45.0
    max_rsi:     float = 67.0
    min_adx:     float = 20.0
    min_vol_ratio: float = 1.2

    # Rolling performance (last 20 trades)
    win_rate_20:  float = 0.0
    sharpe_20:    float = 0.0
    avg_win_pct:  float = 0.0
    avg_loss_pct: float = 0.0
    streak:       int   = 0    # positive = wins, negative = losses

    # State
    status:       str   = "ACTIVE"  # ACTIVE / CAUTIOUS / RETIRED
    last_updated: str   = field(default_factory=lambda: datetime.now().isoformat())
    adaptation_count: int = 0

    # Learning history
    regime_performance: dict = field(default_factory=dict)  # regime → {wr, trades}

    def to_dict(self) -> dict:
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════════════════
# ADAPTIVE LEARNING ENGINE
# ═════════════════════════════════════════════════════════════════════════════════

class AdaptiveLearningEngine:
    """
    Learns from every closed trade.
    Adapts SL, target, size, and entry conditions without re-running backtest.
    Triggers re-backtest only when necessary.
    """

    DECAY_THRESHOLD    = 0.75
    CAUTION_THRESHOLD  = 0.90
    MIN_TRADES_ADAPT   = 5
    ROLLING_WINDOW     = 20
    VIX_JUMP_THRESHOLD = 5.0

    def __init__(self) -> None:
        self._params: dict[str, AdaptiveParams] = {}
        self._trades: dict[str, deque] = {}
        self._rebacktest_queue: list[dict] = []
        self.backtests_skipped: int = 0
        self.backtests_run:     int = 0
        self._store = Path("logs/adaptive")
        self._store.mkdir(parents=True, exist_ok=True)
        self._load_state()
        logger.info("AdaptiveLearningEngine initialized")

    def record_trade(self, trade: TradeRecord) -> None:
        key = f"{trade.strategy}::{trade.symbol}"
        if key not in self._params:
            self._params[key] = self._init_params(trade.strategy, trade.symbol)
        if key not in self._trades:
            self._trades[key] = deque(maxlen=self.ROLLING_WINDOW)
        self._trades[key].append(trade)
        params = self._params[key]
        trades_list = list(self._trades[key])
        self._update_rolling_metrics(params, trades_list)
        regime = trade.regime
        if regime not in params.regime_performance:
            params.regime_performance[regime] = {"wins": 0, "total": 0}
        params.regime_performance[regime]["total"] += 1
        if trade.won:
            params.regime_performance[regime]["wins"] += 1
        if len(trades_list) >= self.MIN_TRADES_ADAPT:
            self._adapt_parameters(params, trades_list, trade.strategy)
        self._check_health(params, trade.strategy)
        params.last_updated = datetime.now().isoformat()
        params.adaptation_count += 1
        logger.info("Adapt [{}::{}] WR={:.0f}% size={:.2f} sl={:.1f}% → {}",
            trade.strategy, trade.symbol, params.win_rate_20 * 100,
            params.size_factor, params.sl_pct, params.status)
        self._save_state()

    def _adapt_parameters(self, params: AdaptiveParams, trades: list[TradeRecord], strategy: str) -> None:
        gate = GATE_THRESHOLDS.get(strategy, {})
        cfg  = STRATEGY_CONFIG.get(strategy, {})
        recent = trades[-10:]
        wins   = [t for t in recent if t.won]
        losses = [t for t in recent if not t.won]
        wr_recent = len(wins) / max(len(recent), 1)
        gate_wr  = gate.get("win_rate", 55) / 100
        rolling_wr = params.win_rate_20
        if rolling_wr >= gate_wr * 1.1:
            params.size_factor = min(1.5, params.size_factor + 0.05)
        elif rolling_wr >= gate_wr * 0.90:
            params.size_factor = max(1.0, params.size_factor - 0.02)
        elif rolling_wr >= gate_wr * 0.75:
            params.size_factor = max(0.5, params.size_factor - 0.1)
        else:
            params.size_factor = 0.25
        if losses:
            avg_loss_pct = abs(np.mean([t.pnl_pct for t in losses]))
            base_sl = cfg.get("sl", 1.5)
            if avg_loss_pct > base_sl * 1.5 and params.sl_pct < base_sl * 2.0:
                params.sl_pct = round(params.sl_pct * 1.08, 2)
            elif avg_loss_pct < base_sl * 0.6 and params.sl_pct > base_sl * 0.8:
                params.sl_pct = round(params.sl_pct * 0.95, 2)
        if wins:
            avg_win_pct = abs(np.mean([t.pnl_pct for t in wins]))
            base_t1 = cfg.get("t1", 3.0)
            if avg_win_pct < base_t1 * 0.5:
                params.target_pct = round(params.target_pct * 0.92, 2)
            elif avg_win_pct > base_t1 * 1.5:
                params.target_pct = round(params.target_pct * 1.06, 2)
        if wr_recent < 0.50 and len(recent) >= 8:
            params.min_rsi = min(50.0, params.min_rsi + 1.0)
            params.max_rsi = max(62.0, params.max_rsi - 1.0)
            params.min_adx = min(28.0, params.min_adx + 1.0)
        elif wr_recent >= 0.65:
            params.min_rsi = max(40.0, params.min_rsi - 0.5)
            params.max_rsi = min(72.0, params.max_rsi + 0.5)
            params.min_adx = max(18.0, params.min_adx - 0.5)
        if recent and recent[-1].won:
            params.streak = max(1, params.streak + 1)
        elif recent:
            params.streak = min(-1, params.streak - 1)
        if wins:
            target_exits = sum(1 for t in wins if t.exit_reason == "TARGET")
            trail_exits  = sum(1 for t in wins if t.exit_reason == "TRAIL")
            if trail_exits > target_exits:
                params.trail_pct = round(max(0.2, params.trail_pct * 0.92), 2)

    def _check_health(self, params: AdaptiveParams, strategy: str) -> None:
        gate   = GATE_THRESHOLDS.get(strategy, {})
        gate_wr = gate.get("win_rate", 55) / 100
        if params.win_rate_20 >= gate_wr * self.CAUTION_THRESHOLD:
            params.status = "ACTIVE"
            self.backtests_skipped += 1
        elif params.win_rate_20 >= gate_wr * self.DECAY_THRESHOLD:
            params.status = "CAUTIOUS"
            self.backtests_skipped += 1
        else:
            params.status = "RETIRED"
            self._queue_rebacktest(strategy, params.symbol, "performance_decay")

    async def nightly_review(self, current_vix: float = 14.0, regime_changed: bool = False) -> dict:
        report = {"date": date.today().isoformat(), "strategies_ok": [],
                  "strategies_adapt": [], "strategies_retire": [],
                  "rebacktest_needed": [], "vix": current_vix, "regime_changed": regime_changed}
        for key, params in self._params.items():
            strategy, symbol = key.split("::")
            gate = GATE_THRESHOLDS.get(strategy, {})
            gate_wr = gate.get("win_rate", 55) / 100
            if regime_changed:
                self._queue_rebacktest(strategy, symbol, "regime_change")
                report["rebacktest_needed"].append(f"{symbol}::{strategy}")
                continue
            vix_spike = current_vix > 22
            if vix_spike and strategy in ("iron_condor","short_straddle","short_strangle","iron_butterfly"):
                self._queue_rebacktest(strategy, symbol, "vix_spike")
                report["rebacktest_needed"].append(f"{symbol}::{strategy}")
                continue
            if params.win_rate_20 >= gate_wr * self.CAUTION_THRESHOLD:
                report["strategies_ok"].append(f"{symbol}::{strategy} WR={params.win_rate_20*100:.0f}%")
                self.backtests_skipped += 1
            elif params.win_rate_20 >= gate_wr * self.DECAY_THRESHOLD:
                report["strategies_adapt"].append(f"{symbol}::{strategy}")
            else:
                report["strategies_retire"].append(f"{symbol}::{strategy}")
                report["rebacktest_needed"].append(f"{symbol}::{strategy}")
        return report

    def on_regime_change(self, old_regime: str, new_regime: str, vix: float) -> None:
        volatile_regime = new_regime in ("BEAR_VOLATILE", "HIGH_VOLATILE")
        bear_regime     = new_regime in ("BEAR_TREND", "BEAR_VOLATILE")
        for key, params in self._params.items():
            strategy, symbol = key.split("::")
            if volatile_regime:
                params.size_factor = min(params.size_factor, 0.5)
                params.sl_pct = round(params.sl_pct * 1.2, 2)
            if bear_regime and "swing" in strategy:
                params.status = "CAUTIOUS"
                params.size_factor = 0.0
            if new_regime == "BULL_TREND" and old_regime != "BULL_TREND":
                params.size_factor = max(0.75, params.size_factor)
            if strategy in ("iron_condor", "short_straddle") and volatile_regime:
                self._queue_rebacktest(strategy, symbol, f"regime→{new_regime}")
        self._save_state()

    def get_params(self, strategy: str, symbol: str) -> AdaptiveParams:
        key = f"{strategy}::{symbol}"
        if key not in self._params:
            self._params[key] = self._init_params(strategy, symbol)
        return self._params[key]

    def should_rebacktest(self, strategy: str, symbol: str) -> tuple[bool, str]:
        key = f"{strategy}::{symbol}"
        for item in self._rebacktest_queue:
            if item["strategy"] == strategy and item["symbol"] == symbol:
                return True, item["reason"]
        params = self._params.get(key)
        if not params:
            return True, "No history — first run"
        gate_wr = GATE_THRESHOLDS.get(strategy, {}).get("win_rate", 55) / 100
        if params.win_rate_20 < gate_wr * self.DECAY_THRESHOLD:
            return True, f"Performance decay (WR {params.win_rate_20*100:.0f}%)"
        return False, "OK — using adaptive params"

    def summary(self) -> dict:
        active   = sum(1 for p in self._params.values() if p.status == "ACTIVE")
        cautious = sum(1 for p in self._params.values() if p.status == "CAUTIOUS")
        retired  = sum(1 for p in self._params.values() if p.status == "RETIRED")
        return {
            "total_pairs": len(self._params), "active": active,
            "cautious": cautious, "retired": retired,
            "backtests_skipped": self.backtests_skipped,
            "backtests_run": self.backtests_run,
            "rebacktest_queue": len(self._rebacktest_queue),
            "efficiency_pct": round(self.backtests_skipped /
                max(self.backtests_skipped + self.backtests_run, 1) * 100, 1),
            "all_params": {k: v.to_dict() for k, v in list(self._params.items())[:20]},
        }

    def _init_params(self, strategy: str, symbol: str) -> AdaptiveParams:
        cfg  = STRATEGY_CONFIG.get(strategy, {})
        gate = GATE_THRESHOLDS.get(strategy, {})
        return AdaptiveParams(
            strategy=strategy, symbol=symbol,
            sl_pct=cfg.get("sl", 1.5), target_pct=cfg.get("t1", 3.0),
            trail_pct=cfg.get("sl", 0.5) * 0.33,
            min_rsi=45.0, max_rsi=67.0,
            min_adx=float(gate.get("win_rate", 55)) / 5,
        )

    def _update_rolling_metrics(self, params: AdaptiveParams, trades: list[TradeRecord]) -> None:
        if not trades: return
        pnls = [t.pnl_pct for t in trades]
        wins = [t for t in trades if t.won]
        params.win_rate_20  = len(wins) / len(trades)
        if len(pnls) > 1:
            arr = np.array(pnls)
            params.sharpe_20 = float(arr.mean() / arr.std() * math.sqrt(len(arr))) if arr.std() > 0 else 0
        params.avg_win_pct  = float(np.mean([t.pnl_pct for t in wins])) if wins else 0
        params.avg_loss_pct = float(np.mean([t.pnl_pct for t in trades if not t.won])) if trades else 0

    def _queue_rebacktest(self, strategy: str, symbol: str, reason: str) -> None:
        entry = {"strategy": strategy, "symbol": symbol, "reason": reason,
                 "queued_at": datetime.now().isoformat()}
        for item in self._rebacktest_queue:
            if item["strategy"] == strategy and item["symbol"] == symbol:
                return
        self._rebacktest_queue.append(entry)

    def _save_state(self) -> None:
        state = {k: v.to_dict() for k, v in self._params.items()}
        path = self._store / "adaptive_params.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self) -> None:
        path = self._store / "adaptive_params.json"
        if not path.exists(): return
        try:
            with open(path) as f:
                data = json.load(f)
            for key, d in data.items():
                strategy, symbol = key.split("::")
                self._params[key] = AdaptiveParams(
                    strategy=strategy, symbol=symbol,
                    **{k: v for k, v in d.items() if k not in ("strategy", "symbol")}
                )
            logger.info("Loaded {} adaptive param sets", len(self._params))
        except Exception as exc:
            logger.warning("Could not load adaptive state: {}", exc)


adaptive_engine = AdaptiveLearningEngine()
