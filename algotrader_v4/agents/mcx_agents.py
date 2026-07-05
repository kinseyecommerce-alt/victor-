"""
agents/mcx_agents.py — MCX commodity trading agents.

Four trading-type agents, one per style, all tick-driven subclasses of
BaseAgent. They replace the retired NSE/BSE equity agents as the live registry
(ALL_AGENTS). Each returns ("BUY"|"SELL"|"HOLD"|"EXIT", signal|None); the
BaseAgent pipeline then publishes the signal to the agent bus and asks the
coordinator to arbitrate before any order is placed.

Registry keys are kept stable (intraday / scalping / swing / fno) so the rest
of the platform — risk buckets, scheduler, dashboard — keeps working; the
`label` attribute carries the human-facing MCX name.

  intraday → MCX Intraday (MIS)          fast momentum, square off intraday
  scalping → MCX Scalping (MIS)          high-frequency small moves
  swing    → MCX Positional (NRML)       multi-day trend, carries overnight
  fno      → MCX Options / Spread (NRML) directional options + spread legs
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from ist_clock import now_ist
from agents.base_agent import BaseAgent
from tick_engine import MarketSnapshot, LiveIndicators
import mcx_universe


class MCXAgentBase(BaseAgent):
    """Shared MCX helpers: lot-aware, tick-rounded, session-aware signal building."""
    exchange: str = "MCX"
    label:    str = "MCX"

    # SL / target as ATR multiples — overridden per style
    SL_ATR:  float = 1.5
    TGT_ATR: float = 2.5

    def filter_watchlist(self, watchlist: list[dict]) -> list[dict]:
        """Approve only the contracts in this agent's MCX universe (no yfinance backtest)."""
        universe = {i["symbol"] for i in mcx_universe.get_strategy_watchlist(self.name)}
        approved = [i for i in watchlist if i["symbol"] in universe] or list(watchlist)
        for i in approved:
            self._approved.add(i["symbol"])
        self.state.approved_symbols = [i["symbol"] for i in approved]
        return approved

    def _session_close_guard(self, sym: str, buffer_min: int = 15) -> bool:
        """True if within `buffer_min` of this contract's session close (block new intraday entries)."""
        c = mcx_universe.contract(sym)
        if not c:
            return False
        close_dt = now_ist().replace(
            hour=c.session_close.hour, minute=c.session_close.minute,
            second=0, microsecond=0,
        )
        return now_ist() >= (close_dt - timedelta(minutes=buffer_min))

    def _mk_signal(
        self, sym: str, action: str, ltp: float, ind: LiveIndicators,
        trigger: str, score: float, sf: float,
    ) -> tuple[str, dict]:
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
            "_gate_size_factor": sf,
            "trigger": (
                f"{self.name.upper()}-{action} [{trigger}] score={score} "
                f"sf={sf} rsi={ind.rsi_14:.0f} trend={ind.trend}"
            ),
        }

    # Generic exit shared by all MCX agents: SL / target vs average price, or RSI flip.
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
            if ltp <= avg - sl_dist:
                return True, "SL hit"
            if ltp >= avg + tgt_dist:
                return True, "target reached"
            if ind.rsi_14 >= 78:
                return True, "RSI exhaustion"
        else:
            if ltp >= avg + sl_dist:
                return True, "SL hit"
            if ltp <= avg - tgt_dist:
                return True, "target reached"
            if ind.rsi_14 <= 22:
                return True, "RSI exhaustion"
        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MCX INTRADAY  (MIS) — fast momentum, square off before session close
# ═══════════════════════════════════════════════════════════════════════════════
class MCXIntradayAgent(MCXAgentBase):
    name    = "intraday"
    label   = "MCX Intraday (MIS)"
    product = "MIS"
    min_candles_1min = 21

    SL_ATR    = 1.5
    TGT_ATR   = 2.5
    MIN_SCORE = 3

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind, sym, ltp = snap.indicators, snap.symbol, snap.tick.ltp
        if self._session_close_guard(sym, buffer_min=15):
            return "HOLD", None
        if not ind.vwap or ind.vwap <= 0:
            return "HOLD", None

        score, action, trig = 0, "", ""
        # Trend-continuation long
        if (ltp > ind.vwap and ind.ema9 > ind.ema21 > 0
                and 48 <= ind.rsi_14 <= 72 and ind.macd_hist > 0
                and ind.volume_ratio >= 1.3):
            score = 3 + (1 if ind.ema21 > ind.ema50 > 0 else 0) + (1 if ind.volume_ratio >= 1.8 else 0)
            action, trig = "BUY", "VWAP_TREND"
        # Trend-continuation short
        elif (ltp < ind.vwap and ind.ema9 < ind.ema21 and ind.ema21 > 0
                and 28 <= ind.rsi_14 <= 52 and ind.macd_hist < 0
                and ind.volume_ratio >= 1.3):
            score = 3 + (1 if 0 < ind.ema50 and ind.ema21 < ind.ema50 else 0) + (1 if ind.volume_ratio >= 1.8 else 0)
            action, trig = "SELL", "VWAP_TREND"

        if not action or score < self.MIN_SCORE:
            return "HOLD", None
        sf = 1.0 if score >= 5 else 0.75
        return self._mk_signal(sym, action, ltp, ind, trig, score, sf)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MCX SCALPING  (MIS) — high frequency, tight SL/target, loss-streak cooldown
# ═══════════════════════════════════════════════════════════════════════════════
class MCXScalpingAgent(MCXAgentBase):
    name    = "scalping"
    label   = "MCX Scalping (MIS)"
    product = "MIS"
    min_candles_1min = 15

    SL_ATR    = 0.8
    TGT_ATR   = 1.2
    MIN_SCORE = 3
    COOLDOWN_MIN = 5

    def __init__(self) -> None:
        super().__init__()
        self._loss_streak:   dict[str, int] = {}
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
        ind, sym, ltp = snap.indicators, snap.symbol, snap.tick.ltp
        if self._session_close_guard(sym, buffer_min=10) or self._in_cooldown(sym):
            return "HOLD", None

        score, action, trig = 0, "", ""
        # Short-burst momentum with volume — long
        if (ind.rsi_7 > 55 and ind.macd_hist > 0 and ind.volume_ratio >= 1.5
                and ltp >= ind.vwap):
            score = 3 + (1 if ind.volume_ratio >= 2.2 else 0)
            action, trig = "BUY", "SCALP_MOMO"
        # Short-burst momentum — short
        elif (ind.rsi_7 < 45 and ind.macd_hist < 0 and ind.volume_ratio >= 1.5
                and ltp <= ind.vwap):
            score = 3 + (1 if ind.volume_ratio >= 2.2 else 0)
            action, trig = "SELL", "SCALP_MOMO"

        if not action or score < self.MIN_SCORE:
            return "HOLD", None
        sf = 0.75 if score < 4 else 1.0
        return self._mk_signal(sym, action, ltp, ind, trig, score, sf)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MCX POSITIONAL  (NRML) — multi-day trend, carries overnight, wide stops
# ═══════════════════════════════════════════════════════════════════════════════
class MCXPositionalAgent(MCXAgentBase):
    name    = "swing"
    label   = "MCX Positional (NRML)"
    product = "NRML"
    min_candles_1min = 30

    SL_ATR    = 3.0
    TGT_ATR   = 6.0
    MIN_SCORE = 4

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        # No session-close square-off: positional carries overnight (NRML).
        ind, sym, ltp = snap.indicators, snap.symbol, snap.tick.ltp

        ema_bull = ind.ema9 > ind.ema21 > 0 and ind.ema21 > ind.ema50 > 0 and ind.ema50 > ind.ema200 > 0
        ema_bear = ind.ema9 < ind.ema21 and ind.ema21 < ind.ema50 and 0 < ind.ema200 and ind.ema50 < ind.ema200

        score, action, trig = 0, "", ""
        if ema_bull and 50 <= ind.rsi_14 <= 70 and ind.macd_hist > 0:
            score = 4 + (1 if ltp > ind.vwap else 0)
            action, trig = "BUY", "TREND_STACK"
        elif ema_bear and 30 <= ind.rsi_14 <= 50 and ind.macd_hist < 0:
            score = 4 + (1 if ltp < ind.vwap else 0)
            action, trig = "SELL", "TREND_STACK"

        if not action or score < self.MIN_SCORE:
            return "HOLD", None
        sf = 1.0 if score >= 5 else 0.75
        return self._mk_signal(sym, action, ltp, ind, trig, score, sf)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MCX OPTIONS / SPREAD  (NRML) — directional options + inter-commodity spreads
# ═══════════════════════════════════════════════════════════════════════════════
class MCXOptionsSpreadAgent(MCXAgentBase):
    name    = "fno"
    label   = "MCX Options / Spread (NRML)"
    product = "NRML"
    min_candles_1min = 20

    SL_ATR    = 2.0
    TGT_ATR   = 4.0
    MIN_SCORE = 3

    # Ratio band for inter-commodity spread deviation (fraction of mean)
    SPREAD_BAND = 0.015

    def __init__(self) -> None:
        super().__init__()
        self._last_ltp:    dict[str, float] = {}
        self._ratio_mean:  dict[tuple[str, str], float] = {}
        self._pair_of: dict[str, tuple[str, str]] = {}
        for a, b in mcx_universe.SPREAD_PAIRS:
            self._pair_of[a] = (a, b)
            self._pair_of[b] = (a, b)

    def _spread_signal(self, sym: str, ltp: float) -> tuple[str, str]:
        """Detect ratio dislocation on a spread pair. Returns (action, trigger) or ('','')."""
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
        # EWMA of the ratio
        self._ratio_mean[pair] = ratio if mean is None else round(mean * 0.97 + ratio * 0.03, 6)
        if mean is None:
            return "", ""
        dev = (ratio - mean) / mean
        if dev <= -self.SPREAD_BAND:
            # leg A cheap relative to B → buy A
            return ("BUY", "SPREAD_LONG_A") if sym == a else ("SELL", "SPREAD_SHORT_B")
        if dev >= self.SPREAD_BAND:
            return ("SELL", "SPREAD_SHORT_A") if sym == a else ("BUY", "SPREAD_LONG_B")
        return "", ""

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind, sym, ltp = snap.indicators, snap.symbol, snap.tick.ltp

        # ── Spread leg dislocation (highest priority) ─────────────────────────
        s_action, s_trig = self._spread_signal(sym, ltp)
        if s_action:
            return self._mk_signal(sym, s_action, ltp, ind, s_trig, 4, 0.75)

        # ── Directional options-buy on strong momentum, avoiding extreme vol ──
        if ind.volatility == "HIGH":
            return "HOLD", None
        score, action, trig = 0, "", ""
        if (ind.ema9 > ind.ema21 > 0 and 52 <= ind.rsi_14 <= 70
                and ind.macd_hist > 0 and ind.volume_ratio >= 1.4):
            score = 3 + (1 if ltp > ind.vwap else 0)
            action, trig = "BUY", "OPT_CE"   # buy call proxy
        elif (ind.ema9 < ind.ema21 and ind.ema21 > 0 and 30 <= ind.rsi_14 <= 48
                and ind.macd_hist < 0 and ind.volume_ratio >= 1.4):
            score = 3 + (1 if ltp < ind.vwap else 0)
            action, trig = "SELL", "OPT_PE"  # buy put proxy

        if not action or score < self.MIN_SCORE:
            return "HOLD", None
        sf = 1.0 if score >= 4 else 0.75
        return self._mk_signal(sym, action, ltp, ind, trig, score, sf)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — the live MCX agent set (replaces the equity ALL_AGENTS)
# ═══════════════════════════════════════════════════════════════════════════════
MCX_AGENTS: dict[str, BaseAgent] = {
    "intraday": MCXIntradayAgent(),
    "scalping": MCXScalpingAgent(),
    "swing":    MCXPositionalAgent(),
    "fno":      MCXOptionsSpreadAgent(),
}
