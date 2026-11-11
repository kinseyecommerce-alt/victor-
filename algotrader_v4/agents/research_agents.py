"""
research_agents.py — research-grade agents, replay only (NOT in the live ALL_AGENTS registry).

Both were written to fill gaps a 30-day breakdown by NIFTY day type showed: swing is long-only and lost on
trend-down and volatile days, and the breakout agents lost on range days. They are registered in
replay_backtest._agent() ("swing_short", "meanrev_v2") so they can be measured on real minute bars; neither
trades live, and nothing here changes a live agent. Treat any result on 30 days as anecdotal.

SwingShortAgent  (name "swing_short")
  The mirror of SwingAgent for down-trends, modelled as a short in stock futures (NRML, held across days,
  futures cost model). Same structure, same 3 % stop / 8 % target, same once-a-minute throttle; every rule is
  the logical inverse of the long one, and no filter was added or tuned:
    1. EMA50_REJECT   — rally to EMA50 (within 1.5 %) inside an EMA200 downtrend, RSI 40-60, volatility not HIGH
    2. ICHIMOKU_DOWN  — fresh Tenkan cross below Kijun with price below the cloud and EMA200, −DI in control,
                        RSI > 30
  Simplifications to remember: no lot sizes (sized like swing's per-symbol capital), no expiry roll, no
  margin model; the holding limit is the replay's swing limit.

MeanRevV2Agent  (name "meanrev_v2")
  Fades large intraday stretches back to the SESSION VWAP, built so that costs are a small share of each trade:
    • session VWAP and its volume-weighted σ are accumulated by the agent from today's 1-minute bars (the
      indicator `vwap` is a 14-bar rolling VWAP and the `vwap_*` bands span ~500 bars, i.e. several days,
      so neither is a session VWAP); no trade unless the agent saw the session from its first minutes
    • entries 10:30-14:30 only, at most one per symbol per day
    • setup: a bar that reached VWAP ± K_SIGMA·σ and closed back inside the band in the fade direction
      (rejection bar: red for a short, green for a long)
    • target = the session VWAP at signal time, and it must be ≥ MIN_TGT_PCT away (0.5 % → ~₹80 round-trip
      costs on ₹1L are ~16 % of a full win)
    • stop = STOP_SIGMA·σ beyond the entry (a fixed multiple); if the rejection bar already traded beyond
      that stop, the setup is skipped
  Three tunables (K_SIGMA, STOP_SIGMA, MIN_TGT_PCT) plus the time window. MIS, squared off like intraday.
"""
from __future__ import annotations

import math
from datetime import datetime, time
from typing import Optional

from agents.base_agent import BaseAgent
from agents.strategy_agents import SwingAgent
from tick_engine import LiveIndicators, MarketSnapshot


def now_ist():
    """Resolved at call time through strategy_agents so the replay's clock patch applies."""
    import agents.strategy_agents as _sa
    return _sa.now_ist()


# ═══════════════════════════════════════════════════════════════════════════════
# SWING SHORT — stock futures, mirror of SwingAgent
# ═══════════════════════════════════════════════════════════════════════════════

class SwingShortAgent(SwingAgent):
    validated_by_replay = False      # research only
    strategy_rule_exits = False      # like swing: stop, target, trailing, holding period
    VERSION = "v1"
    name    = "swing_short"
    product = "NRML"                 # stock futures, carried overnight
    exchange = "NFO"

    SL_PCT  = 3.0                    # swing's 3 % stop / 8 % target, mirrored
    TGT_PCT = 8.0

    _last_eval: dict[str, float] = {}     # own throttle state (not shared with SwingAgent)

    def __init__(self) -> None:
        super().__init__()
        self._prev_tk_below: dict[str, bool] = {}   # sym → tenkan < kijun at last eval

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        import time as _time
        sym = snap.symbol
        now = _time.time()
        if now - self._last_eval.get(sym, 0) < 60:      # same once-a-minute throttle as SwingAgent
            return "HOLD", None
        self._last_eval[sym] = now

        ind, ltp = snap.indicators, snap.tick.ltp
        if not ind.ema200 or ltp <= 0:
            return "HOLD", None

        pattern = ""
        for pat_fn in (self._pat_ema50_reject, self._pat_ichimoku_down):
            pattern = pat_fn(sym, ind, ltp)
            if pattern:
                break
        if ind.ichimoku_tenkan and ind.ichimoku_kijun:
            self._prev_tk_below[sym] = ind.ichimoku_tenkan < ind.ichimoku_kijun
        if not pattern:
            return "HOLD", None

        sl  = round(ltp * (1 + self.SL_PCT / 100), 2)
        tgt = round(ltp * (1 - self.TGT_PCT / 100), 2)
        return "SELL", {
            "symbol":    sym,
            "exchange":  self.exchange,
            "side":      "SELL",
            "price":     ltp,
            "stop_loss": sl,
            "target":    tgt,
            "stop_loss_pct": self.SL_PCT, "target_pct": self.TGT_PCT,
            "product":   self.product,
            "instrument": "FUT",
            "trigger":   f"SWSHORT [{pattern}] trend=DOWN rsi={ind.rsi_14:.0f}",
        }

    def _pat_ema50_reject(self, sym: str, ind: LiveIndicators, ltp: float) -> str:
        trend_ok   = ltp < ind.ema200                                    # long-term downtrend
        ema50_near = ind.ema50 > 0 and abs(ltp - ind.ema50) / ind.ema50 < 0.015
        rsi_ok     = 40 < ind.rsi_14 < 60                                # distribution zone
        low_vol    = ind.volatility != "HIGH"                            # calm rally
        return "EMA50-REJECT" if trend_ok and ema50_near and rsi_ok and low_vol else ""

    def _pat_ichimoku_down(self, sym: str, ind: LiveIndicators, ltp: float) -> str:
        tk, kj, bot = ind.ichimoku_tenkan, ind.ichimoku_kijun, ind.ichimoku_cloud_bot
        if not (tk and kj and bot):
            return ""
        was_below = self._prev_tk_below.get(sym, tk < kj)
        fresh_cross = not was_below and tk < kj
        if (fresh_cross and ltp < bot and ltp < ind.ema200
                and ind.minus_di > ind.plus_di and ind.rsi_14 > 30):
            return "ICHIMOKU-DOWN"
        return ""

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", ind.ltp)
        ltp   = ind.ltp
        if not entry:
            return False, ""
        if ltp >= entry * (1 + self.SL_PCT / 100):  return True, f"Swing-short SL ₹{ltp:.2f}"
        if ltp <= entry * (1 - self.TGT_PCT / 100): return True, f"Swing-short target ₹{ltp:.2f}"
        if ind.trend == "UP" and ind.ema9 > ind.ema21:
            return True, "Trend reversal exit"
        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# MEAN REVERSION v2 — fade K-σ stretches from the session VWAP
# ═══════════════════════════════════════════════════════════════════════════════

class MeanRevV2Agent(BaseAgent):
    name = "meanrev_v2"
    validated_by_replay = False      # research only
    strategy_rule_exits = False
    VERSION = "v1"
    product = "MIS"
    min_candles_1min = 30

    START_MIN   = 630      # 10:30 — after the opening drive has set the day's range
    END_MIN     = 870      # 14:30 — leave time for the reversion before the square-off
    K_SIGMA     = 3.0      # stretch: the bar reached session VWAP ± K σ
    STOP_SIGMA  = 1.5      # stop distance = this × σ beyond the entry
    MIN_TGT_PCT = 0.5      # VWAP must be at least this % away (costs stay a small share)
    OPEN_SEEN_BY = time(9, 20)   # the session must have been seen from here, or VWAP is not the session's

    def __init__(self) -> None:
        super().__init__()
        self._sess: dict = {}      # (sym, day) → session accumulator

    @staticmethod
    def _t(minutes) -> time:
        m = int(minutes)
        return time(m // 60, m % 60)

    # ── session VWAP ──────────────────────────────────────────────────────
    def _session(self, sym: str, snap: MarketSnapshot, day) -> dict:
        k = (sym, day)
        s = self._sess.get(k)
        if s is None:
            s = self._sess[k] = {"bars": {}, "first": None, "last": None, "sv": 0.0, "spv": 0.0, "spv2": 0.0,
                                 "fired": False}
            for old in [x for x in self._sess if x[0] == sym and x != k]:
                self._sess.pop(old, None)
        new = []
        for c in reversed(snap.candles_1min):              # only the newest candles change between calls
            ts = getattr(c, "ts", None)
            if ts is None or ts.date() != day:
                break
            if s["last"] is not None and ts < s["last"]:
                break
            new.append(c)
        for c in reversed(new):
            tp, v = (c.high + c.low + c.close) / 3.0, max(float(c.volume or 0), 0.0)
            old = s["bars"].get(c.ts)
            if old:                                         # an in-progress candle was updated
                otp, ov = old
                s["sv"] -= ov; s["spv"] -= otp * ov; s["spv2"] -= otp * otp * ov
            s["bars"][c.ts] = (tp, v)
            s["sv"] += v; s["spv"] += tp * v; s["spv2"] += tp * tp * v
            s["first"] = c.ts if s["first"] is None or c.ts < s["first"] else s["first"]
            s["last"] = c.ts if s["last"] is None or c.ts > s["last"] else s["last"]
        return s

    @staticmethod
    def session_vwap(s: dict) -> tuple[float, float]:
        """(vwap, σ) of the typical price, volume-weighted, over the session so far."""
        if s["sv"] <= 0:
            return 0.0, 0.0
        vw = s["spv"] / s["sv"]
        var = max(s["spv2"] / s["sv"] - vw * vw, 0.0)
        return vw, math.sqrt(var)

    # ── entry ─────────────────────────────────────────────────────────────
    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        sym, ltp = snap.symbol, snap.tick.ltp
        now = now_ist().replace(tzinfo=None)
        t = now.time()
        if ltp <= 0 or not snap.candles_1min:
            return "HOLD", None
        s = self._session(sym, snap, now.date())
        if s["fired"] or not (self._t(self.START_MIN) <= t <= self._t(self.END_MIN)):
            return "HOLD", None
        if s["first"] is None or s["first"].time() > self.OPEN_SEEN_BY:
            return "HOLD", None                             # joined mid-session: no honest session VWAP
        vw, sd = self.session_vwap(s)
        if vw <= 0 or sd <= 0:
            return "HOLD", None
        c = snap.candles_1min[-1]
        if getattr(c, "ts", None) is None or c.ts.date() != now.date():
            return "HOLD", None
        dist = self.STOP_SIGMA * sd
        upper, lower = vw + self.K_SIGMA * sd, vw - self.K_SIGMA * sd
        tgt_pct = abs(ltp - vw) / ltp * 100
        if tgt_pct < self.MIN_TGT_PCT:
            return "HOLD", None
        # stretched up to the band and rejected (red bar closing back inside) → short to VWAP
        if c.high >= upper and c.close < c.open and ltp < upper and ltp > vw:
            if c.high >= ltp + dist:                        # the bar already traded through our stop
                return "HOLD", None
            s["fired"] = True
            return self._sig(sym, "SELL", ltp, ltp + dist, vw, sd)
        if c.low <= lower and c.close > c.open and ltp > lower and ltp < vw:
            if c.low <= ltp - dist:
                return "HOLD", None
            s["fired"] = True
            return self._sig(sym, "BUY", ltp, ltp - dist, vw, sd)
        return "HOLD", None

    def _sig(self, sym, side, ltp, stop, target, sd):
        risk = abs(ltp - stop)
        return side, {
            "symbol": sym, "exchange": "NSE", "side": side, "price": ltp,
            "stop_loss": round(stop, 2), "target": round(target, 2),
            "stop_loss_pct": round(risk / ltp * 100, 3), "target_pct": round(abs(target - ltp) / ltp * 100, 3),
            "product": self.product, "_gate_size_factor": 1.0,
            "trigger": f"MR2-{side} [VWAP_FADE] {self.K_SIGMA:g}σ rr {abs(target - ltp) / risk:.1f}",
        }

    def should_exit_position(self, pos: dict, ind) -> tuple[bool, str]:
        """Stops, targets and the square-off are handled by the bracket / replay; no rule exits."""
        return False, ""
