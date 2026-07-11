"""
agents/mcx_agents.py — MCX commodity trading agents.

Four trading-type agents, one per style, all tick-driven subclasses of
BaseAgent. Each runs a registry of 20 named strategies (agents/mcx_strategies.py)
on every tick and takes the best-scoring signal; the BaseAgent pipeline then
publishes it to the agent bus and asks the coordinator to arbitrate before any
order is placed.

Registry keys are kept stable (intraday / scalping / swing / fno) so the rest
of the platform keeps working; the `label` attribute carries the MCX name.

  intraday → MCX Intraday (MIS)          20 trend/momentum/breakout strategies
  scalping → MCX Scalping (MIS)          20 fast micro-momentum / mean-reversion
  swing    → MCX Positional (NRML)       20 multi-day trend-following strategies
  fno      → MCX Options / Spread (NRML) 20 volatility/directional + spread overlay
"""
from __future__ import annotations

from datetime import time as dtime, timedelta
from typing import Optional

from ist_clock import now_ist
from agents.base_agent import BaseAgent
from tick_engine import MarketSnapshot, LiveIndicators
import mcx_universe
from agents.mcx_strategies import (
    SCtx, update_prev, STRATEGY_REGISTRY,
    INTRADAY_STRATEGIES, SCALPING_STRATEGIES,
    POSITIONAL_STRATEGIES, OPTIONS_STRATEGIES,
)


class MCXAgentBase(BaseAgent):
    """Shared MCX helpers: strategy-registry runner, lot-aware tick-rounded signals."""
    exchange:   str = "MCX"
    label:      str = "MCX"
    strategies: list = []       # set per agent (list of (name, fn))

    SL_ATR:    float = 1.5
    TGT_ATR:   float = 2.5
    MIN_SCORE: int   = 3

    def __init__(self) -> None:
        super().__init__()
        self._pstate: dict[str, dict] = {}   # symbol → rolling previous-tick state
        self._last_strategy: str = ""

    # ── Universe filter ───────────────────────────────────────────────────────
    def filter_watchlist(self, watchlist: list[dict]) -> list[dict]:
        """Approve only the contracts in this agent's MCX universe (no yfinance backtest)."""
        universe = {i["symbol"] for i in mcx_universe.get_strategy_watchlist(self.name)}
        approved = [i for i in watchlist if i["symbol"] in universe] or list(watchlist)
        for i in approved:
            self._approved.add(i["symbol"])
        self.state.approved_symbols = [i["symbol"] for i in approved]
        return approved

    # ── Session / opening-range helpers ───────────────────────────────────────
    def _session_close_guard(self, sym: str, buffer_min: int = 15) -> bool:
        c = mcx_universe.contract(sym)
        if not c:
            return False
        close_dt = now_ist().replace(hour=c.session_close.hour,
                                     minute=c.session_close.minute, second=0, microsecond=0)
        return now_ist() >= (close_dt - timedelta(minutes=buffer_min))

    def _update_orb(self, snap: MarketSnapshot, ps: dict, t: dtime) -> None:
        """Track the 09:00–09:30 opening range for the ORB strategy."""
        if dtime(9, 0) <= t <= dtime(9, 30):
            ltp = snap.tick.ltp
            ps["orb_high"] = max(ps.get("orb_high", ltp), snap.tick.high or ltp)
            ps["orb_low"]  = min(ps.get("orb_low", ltp),  snap.tick.low  or ltp)
            ps["orb_ready"] = False
        elif t > dtime(9, 30) and "orb_high" in ps:
            ps["orb_ready"] = True

    # ── Strategy-registry runner ──────────────────────────────────────────────
    def _run_strategies(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind, sym, ltp = snap.indicators, snap.symbol, snap.tick.ltp
        ps = self._pstate.setdefault(sym, {})
        t  = now_ist().time().replace(tzinfo=None)
        self._update_orb(snap, ps, t)

        ctx = SCtx(sym=sym, ltp=ltp, ind=ind, prev=ps,
                   candles=snap.candles_1min, t=t)

        best: Optional[tuple[int, str, str]] = None   # (score, action, name)
        for name, fn in self.strategies:
            try:
                res = fn(ctx)
            except Exception:
                res = None
            if not res:
                continue
            action, score = res
            if action in ("BUY", "SELL") and (best is None or score > best[0]):
                best = (score, action, name)

        update_prev(ps, ctx)

        if best is None or best[0] < self.MIN_SCORE:
            return "HOLD", None
        score, action, name = best
        self._last_strategy = name
        sf = 1.0 if score >= 5 else (0.85 if score >= 4 else 0.7)
        return self._mk_signal(sym, action, ltp, ind, name, score, sf)

    # ── Signal construction ───────────────────────────────────────────────────
    def _mk_signal(self, sym, action, ltp, ind, trigger, score, sf) -> tuple[str, dict]:
        atr      = ind.atr_14 or ltp * 0.005
        sl_dist  = atr * self.SL_ATR
        tgt_dist = atr * self.TGT_ATR
        if action == "BUY":
            sl  = mcx_universe.round_to_tick(sym, ltp - sl_dist)
            tgt = mcx_universe.round_to_tick(sym, ltp + tgt_dist)
        else:
            sl  = mcx_universe.round_to_tick(sym, ltp + sl_dist)
            tgt = mcx_universe.round_to_tick(sym, ltp - tgt_dist)
        return action, {
            "symbol":            sym,
            "exchange":          "MCX",
            "side":              action,
            "price":             ltp,
            "stop_loss":         sl,
            "target":            tgt,
            "stop_loss_pct":     round(sl_dist  / ltp * 100, 3),
            "target_pct":        round(tgt_dist / ltp * 100, 3),
            "product":           self.product,
            "lot_size":          mcx_universe.lot_size(sym),
            "strategy":          trigger,
            "_gate_size_factor": sf,
            "trigger": (
                f"{self.name.upper()}-{action} [{trigger}] score={score} "
                f"sf={sf} rsi={ind.rsi_14:.0f} trend={ind.trend}"
            ),
        }

    # ── Generic exit ──────────────────────────────────────────────────────────
    def should_exit_position(self, position: dict, ind: LiveIndicators) -> tuple[bool, str]:
        avg = position.get("average_price") or 0.0
        qty = position.get("quantity", 0)
        if not avg or qty == 0:
            return False, ""
        ltp   = ind.ltp
        long_ = qty > 0
        atr   = ind.atr_14 or avg * 0.005
        sl_dist  = atr * self.SL_ATR
        tgt_dist = atr * self.TGT_ATR
        if long_:
            if ltp <= avg - sl_dist:  return True, "SL hit"
            if ltp >= avg + tgt_dist: return True, "target reached"
            if ind.rsi_14 >= 78:      return True, "RSI exhaustion"
        else:
            if ltp >= avg + sl_dist:  return True, "SL hit"
            if ltp <= avg - tgt_dist: return True, "target reached"
            if ind.rsi_14 <= 22:      return True, "RSI exhaustion"
        return False, ""

    # ── Status ────────────────────────────────────────────────────────────────
    def strategy_names(self) -> list[str]:
        return [n for n, _ in self.strategies]

    def get_status(self) -> dict:
        st = super().get_status()
        st["label"]          = self.label
        st["strategy_count"] = len(self.strategies)
        st["strategies"]     = self.strategy_names()
        st["last_strategy"]  = self._last_strategy
        return st


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MCX INTRADAY (MIS)
# ═══════════════════════════════════════════════════════════════════════════════
class MCXIntradayAgent(MCXAgentBase):
    name    = "intraday"
    label   = "MCX Intraday (MIS)"
    product = "MIS"
    min_candles_1min = 21
    strategies = INTRADAY_STRATEGIES
    SL_ATR, TGT_ATR, MIN_SCORE = 1.5, 2.5, 3

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        if self._session_close_guard(snap.symbol, buffer_min=15):
            return "HOLD", None
        return self._run_strategies(snap)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MCX SCALPING (MIS) — with loss-streak cooldown
# ═══════════════════════════════════════════════════════════════════════════════
class MCXScalpingAgent(MCXAgentBase):
    name    = "scalping"
    label   = "MCX Scalping (MIS)"
    product = "MIS"
    min_candles_1min = 15
    strategies = SCALPING_STRATEGIES
    SL_ATR, TGT_ATR, MIN_SCORE = 0.8, 1.2, 3
    COOLDOWN_MIN = 5

    def __init__(self) -> None:
        super().__init__()
        self._loss_streak:    dict[str, int] = {}
        self._cooldown_until: dict[str, object] = {}

    def _in_cooldown(self, sym: str) -> bool:
        until = self._cooldown_until.get(sym)
        return bool(until and until > now_ist())

    def _record_outcome(self, sym: str, win: bool) -> None:
        if win:
            self._loss_streak[sym] = 0
            return
        self._loss_streak[sym] = self._loss_streak.get(sym, 0) + 1
        if self._loss_streak[sym] >= 3:
            self._cooldown_until[sym] = now_ist() + timedelta(minutes=self.COOLDOWN_MIN)
            self._loss_streak[sym] = 0

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        if self._session_close_guard(snap.symbol, buffer_min=10) or self._in_cooldown(snap.symbol):
            return "HOLD", None
        return self._run_strategies(snap)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MCX POSITIONAL (NRML) — carries overnight, no session-close square-off
# ═══════════════════════════════════════════════════════════════════════════════
class MCXPositionalAgent(MCXAgentBase):
    name    = "swing"
    label   = "MCX Positional (NRML)"
    product = "NRML"
    min_candles_1min = 30
    strategies = POSITIONAL_STRATEGIES
    SL_ATR, TGT_ATR, MIN_SCORE = 3.0, 6.0, 3

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        return self._run_strategies(snap)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MCX OPTIONS / SPREAD (NRML) — 20 vol/directional strategies + spread overlay
# ═══════════════════════════════════════════════════════════════════════════════
class MCXOptionsSpreadAgent(MCXAgentBase):
    name    = "fno"
    label   = "MCX Options / Spread (NRML)"
    product = "NRML"
    min_candles_1min = 20
    strategies = OPTIONS_STRATEGIES
    SL_ATR, TGT_ATR, MIN_SCORE = 2.0, 4.0, 3

    SPREAD_BAND = 0.015   # ratio deviation (fraction of mean) to trigger a spread leg

    def __init__(self) -> None:
        super().__init__()
        self._last_ltp:   dict[str, float] = {}
        self._ratio_mean: dict[tuple[str, str], float] = {}
        self._pair_of:    dict[str, tuple[str, str]] = {}
        for a, b in mcx_universe.SPREAD_PAIRS:
            self._pair_of[a] = (a, b)
            self._pair_of[b] = (a, b)

    def _spread_signal(self, sym: str, ltp: float) -> tuple[str, str]:
        """Inter-commodity spread leg on ratio dislocation. Returns (action, trigger) or ('','')."""
        self._last_ltp[sym] = ltp
        pair = self._pair_of.get(sym)
        if not pair:
            return "", ""
        a, b = pair
        pa, pb = self._last_ltp.get(a), self._last_ltp.get(b)
        if not pa or not pb or pb == 0:
            return "", ""
        ratio = pa / pb
        mean  = self._ratio_mean.get(pair)
        self._ratio_mean[pair] = ratio if mean is None else round(mean * 0.97 + ratio * 0.03, 6)
        if mean is None:
            return "", ""
        dev = (ratio - mean) / mean
        if dev <= -self.SPREAD_BAND:
            return ("BUY", "SPREAD_LONG_A") if sym == a else ("SELL", "SPREAD_SHORT_B")
        if dev >= self.SPREAD_BAND:
            return ("SELL", "SPREAD_SHORT_A") if sym == a else ("BUY", "SPREAD_LONG_B")
        return "", ""

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        # Spread overlay takes priority over the directional/vol strategy registry
        s_action, s_trig = self._spread_signal(snap.symbol, snap.tick.ltp)
        if s_action:
            self._last_strategy = s_trig
            return self._mk_signal(snap.symbol, s_action, snap.tick.ltp,
                                   snap.indicators, s_trig, 4, 0.85)
        return self._run_strategies(snap)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — the live MCX agent set (replaces the equity ALL_AGENTS)
# ═══════════════════════════════════════════════════════════════════════════════
MCX_AGENTS: dict[str, BaseAgent] = {
    "intraday": MCXIntradayAgent(),
    "scalping": MCXScalpingAgent(),
    "swing":    MCXPositionalAgent(),
    "fno":      MCXOptionsSpreadAgent(),
}
