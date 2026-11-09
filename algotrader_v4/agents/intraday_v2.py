"""
intraday_v2.py — intraday and scalping rebuilt around NSE costs.

The first-generation agents fired on ten overlapping 1-minute patterns, hundreds of times a
day, and paid ~₹41 per round trip on ₹40k positions: on 45 days of real Kite data both lost
money in every parameter combination the optimiser tried. These two agents take the opposite
stance: very few, well-defined trades per symbol per day, each with a structural stop and a
target at least 2× the risk, so one winner pays for two losers after costs.

IntradayV2Agent  (name "intraday")
  ORB        opening-range (09:15–09:30) breakout, 09:31–12:30, fresh break of the range high/low on
             ≥1.3× volume, on the VWAP side of the move with EMA9/21 agreeing; stop at the other side
             of the range (its midpoint when the range is wide); target 2R. One per symbol per day.
  VWAP_TREND 10:00–13:30, a fresh VWAP cross in the direction of ADX>20 with +DI/−DI agreeing and
             volume ≥1.2×; stop below/above the last 10 bars (≥0.35%); target 2R. One per day.

ScalpV2Agent  (name "scalping")
  PULLBACK   09:45–14:30 in a trending stock (ADX ≥ 22, EMA9>EMA21>EMA50 above VWAP, or the mirror):
             price pulled back to EMA21 within the last 6 bars and the current bar closes back through
             EMA9 in the trend direction, RSI 42–68; stop under the pullback low (0.3–0.6%),
             target 1.8R. At most 3 per symbol per day, 10-minute cooldown, stop after 2 losses.

Both are MIS, square off by code at 15:00 like every intraday trade, and go through the same risk
checks, cost model and trailing-stop engine as before.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Optional

from agents.base_agent import BaseAgent
import agents.strategy_agents as _sa   # the replay patches _sa.now_ist to follow the replayed bar


def now_ist():
    return _sa.now_ist()
from tick_engine import MarketSnapshot


def _day_key(sym: str, now: datetime) -> tuple:
    return (sym, now.date())


class IntradayV2Agent(BaseAgent):
    name = "intraday"
    product = "MIS"
    min_candles_1min = 21

    # tunables (the optimiser's grid can sweep these); times are minutes after midnight IST
    ORB_END_MIN        = 570    # 09:30 — the opening range is 09:15 to here
    ORB_WINDOW_END_MIN = 600    # 10:00 — breakouts are taken until here
    USE_VWAP_TREND     = 0      # the VWAP-cross pattern lost money on real data; off by default
    VWAP_START_MIN     = 600
    VWAP_WINDOW_END_MIN = 810
    VOL_ORB        = 1.3
    VOL_VWAP       = 1.2
    ADX_MIN        = 20.0
    RANGE_MIN_PCT  = 0.25     # opening range too small → noise
    RANGE_MAX_PCT  = 1.5      # too wide → stop too far
    SL_MIN_PCT     = 0.35
    SL_MAX_PCT     = 1.2
    RR             = 2.0
    STOP_AT_MID_ABOVE_PCT = 0.8   # a range wider than this uses its midpoint as the stop
    MAX_PER_DAY    = 2        # per symbol: at most one ORB and one VWAP trade

    @staticmethod
    def _t(minutes) -> time:
        m = int(minutes)
        return time(m // 60, m % 60)

    def __init__(self) -> None:
        super().__init__()
        self._orb: dict = {}          # (sym, day) → {"h","l","done"}
        self._fired: dict = {}        # (sym, day) → {"ORB": bool, "VWAP_TREND": bool}
        self._prev_close: dict = {}   # sym → last close
        self._prev_vwap_side: dict = {}

    # ── helpers ───────────────────────────────────────────────────────────
    def _range(self, sym, snap, now):
        k = _day_key(sym, now)
        r = self._orb.get(k)
        if r is None:
            r = self._orb[k] = {"h": None, "l": None, "done": False}
            # a new day: forget older keys for this symbol
            for old in [x for x in self._orb if x[0] == sym and x != k]:
                self._orb.pop(old, None); self._fired.pop(old, None)
        if not r["done"]:
            for c in snap.candles_1min:
                ts = getattr(c, "ts", None)
                if ts is None or ts.date() != now.date():
                    continue
                if time(9, 15) <= ts.time() < self._t(self.ORB_END_MIN):
                    r["h"] = c.high if r["h"] is None else max(r["h"], c.high)
                    r["l"] = c.low if r["l"] is None else min(r["l"], c.low)
            if now.time() >= self._t(self.ORB_END_MIN) and r["h"] is not None:
                r["done"] = True
        return r

    def _signal(self, sym, side, ltp, stop, pattern, note, extra=None):
        dist = abs(ltp - stop)
        dist = min(max(dist, ltp * self.SL_MIN_PCT / 100), ltp * self.SL_MAX_PCT / 100)
        stop = ltp - dist if side == "BUY" else ltp + dist
        target = ltp + dist * self.RR if side == "BUY" else ltp - dist * self.RR
        return side, {
            "symbol": sym, "exchange": "NSE", "side": side, "price": ltp,
            "stop_loss": round(stop, 2), "target": round(target, 2),
            "stop_loss_pct": round(dist / ltp * 100, 3), "target_pct": round(dist * self.RR / ltp * 100, 3),
            "product": self.product, "_gate_size_factor": 1.0,
            "trigger": f"INTRA2-{side} [{pattern}] {note}", **(extra or {}),
        }


    def should_exit_position(self, pos: dict, ind) -> tuple[bool, str]:
        """Stops and targets are enforced by the bracket / trailing engine; this is the belt-and-braces
        check the base agent calls on each tick: square-off at 15:00 and a clear trend failure."""
        entry = pos.get("average_price", 0.0) or 0.0
        if entry <= 0:
            return False, ""
        side = "BUY" if pos.get("quantity", 0) > 0 else "SELL"
        if now_ist().time().replace(tzinfo=None) >= time(15, 0):
            return True, "Auto square-off 3:00 PM"
        if side == "BUY" and ind.ema9 and ind.ema21 and ind.ema9 < ind.ema21 and ind.ltp < ind.vwap:
            return True, "trend failed (EMA9<EMA21 below VWAP)"
        if side == "SELL" and ind.ema9 and ind.ema21 and ind.ema9 > ind.ema21 and ind.ltp > ind.vwap:
            return True, "trend failed (EMA9>EMA21 above VWAP)"
        return False, ""

    # ── entry ─────────────────────────────────────────────────────────────
    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind, sym, ltp = snap.indicators, snap.symbol, snap.tick.ltp
        now = now_ist().replace(tzinfo=None)
        t = now.time()
        prev_close = self._prev_close.get(sym, ltp)
        prev_side = self._prev_vwap_side.get(sym)
        self._prev_close[sym] = ltp
        if ind.vwap:
            self._prev_vwap_side[sym] = ltp > ind.vwap
        if not ind.ema9 or not ind.ema21 or not ind.vwap or ltp <= 0:
            return "HOLD", None
        if t >= time(14, 45):
            return "HOLD", None
        r = self._range(sym, snap, now)
        fired = self._fired.setdefault(_day_key(sym, now), {})
        if sum(1 for v in fired.values() if v) >= self.MAX_PER_DAY:
            return "HOLD", None
        atr = ind.atr_14 or ltp * 0.004

        # 1. opening-range breakout
        if not fired.get("ORB") and r["done"] and self._t(self.ORB_END_MIN) < t <= self._t(self.ORB_WINDOW_END_MIN):
            h, l = r["h"], r["l"]
            rng_pct = (h - l) / ltp * 100
            if self.RANGE_MIN_PCT <= rng_pct <= self.RANGE_MAX_PCT and ind.volume_ratio >= self.VOL_ORB:
                mid = (h + l) / 2
                if prev_close <= h < ltp and ltp > ind.vwap and ind.ema9 > ind.ema21:
                    fired["ORB"] = True
                    stop = l if rng_pct <= self.STOP_AT_MID_ABOVE_PCT else mid
                    return self._signal(sym, "BUY", ltp, stop, "ORB", f"range {l:.1f}-{h:.1f} vol {ind.volume_ratio:.1f}x")
                if prev_close >= l > ltp and ltp < ind.vwap and ind.ema9 < ind.ema21:
                    fired["ORB"] = True
                    stop = h if rng_pct <= self.STOP_AT_MID_ABOVE_PCT else mid
                    return self._signal(sym, "SELL", ltp, stop, "ORB", f"range {l:.1f}-{h:.1f} vol {ind.volume_ratio:.1f}x")

        # 2. VWAP cross with the trend
        if self.USE_VWAP_TREND and not fired.get("VWAP_TREND") and self._t(self.VWAP_START_MIN) <= t <= self._t(self.VWAP_WINDOW_END_MIN) and prev_side is not None \
                and ind.adx_14 >= self.ADX_MIN and ind.volume_ratio >= self.VOL_VWAP and len(snap.candles_1min) >= 10:
            recent = snap.candles_1min[-10:]
            lo10, hi10 = min(c.low for c in recent), max(c.high for c in recent)
            if (not prev_side) and ltp > ind.vwap and ind.plus_di > ind.minus_di and ind.ema9 > ind.ema21:
                fired["VWAP_TREND"] = True
                return self._signal(sym, "BUY", ltp, min(lo10, ltp - atr), "VWAP_TREND", f"adx {ind.adx_14:.0f} vol {ind.volume_ratio:.1f}x")
            if prev_side and ltp < ind.vwap and ind.minus_di > ind.plus_di and ind.ema9 < ind.ema21:
                fired["VWAP_TREND"] = True
                return self._signal(sym, "SELL", ltp, max(hi10, ltp + atr), "VWAP_TREND", f"adx {ind.adx_14:.0f} vol {ind.volume_ratio:.1f}x")
        return "HOLD", None


class ScalpV2Agent(BaseAgent):
    name = "scalping"
    product = "MIS"
    min_candles_1min = 50

    START, END     = time(9, 45), time(14, 30)
    ADX_MIN        = 22.0
    RSI_LO, RSI_HI = 42.0, 68.0
    PULLBACK_BARS  = 6
    TOUCH_PCT      = 0.05     # how close to EMA21 counts as a touch (% of price)
    SL_MIN_PCT     = 0.30
    SL_MAX_PCT     = 0.60
    RR             = 1.8
    MAX_PER_DAY    = 3
    COOLDOWN_S     = 600
    MAX_LOSSES     = 2        # per symbol per day (the agent is told via record_result)

    def __init__(self) -> None:
        super().__init__()
        self._count: dict = {}       # (sym, day) → trades
        self._last: dict = {}        # sym → datetime of last signal
        self._losses: dict = {}      # (sym, day) → losses

    def record_result(self, sym: str, pnl: float, now: Optional[datetime] = None) -> None:
        """Called by the executor/replay when a trade closes, so a losing day stops early."""
        now = now or now_ist().replace(tzinfo=None)
        if pnl < 0:
            k = _day_key(sym, now)
            self._losses[k] = self._losses.get(k, 0) + 1

    def _signal(self, sym, side, ltp, stop, note):
        dist = abs(ltp - stop)
        dist = min(max(dist, ltp * self.SL_MIN_PCT / 100), ltp * self.SL_MAX_PCT / 100)
        stop = ltp - dist if side == "BUY" else ltp + dist
        target = ltp + dist * self.RR if side == "BUY" else ltp - dist * self.RR
        return side, {
            "symbol": sym, "exchange": "NSE", "side": side, "price": ltp,
            "stop_loss": round(stop, 2), "target": round(target, 2),
            "stop_loss_pct": round(dist / ltp * 100, 3), "target_pct": round(dist * self.RR / ltp * 100, 3),
            "product": self.product, "_gate_size_factor": 1.0,
            "trigger": f"SCALP2-{side} [PULLBACK] {note}",
        }

    def should_exit_position(self, pos: dict, ind) -> tuple[bool, str]:
        """Stops and targets are enforced by the bracket / trailing engine; this is the belt-and-braces
        check the base agent calls on each tick: square-off at 15:00 and a clear trend failure."""
        entry = pos.get("average_price", 0.0) or 0.0
        if entry <= 0:
            return False, ""
        side = "BUY" if pos.get("quantity", 0) > 0 else "SELL"
        if now_ist().time().replace(tzinfo=None) >= time(15, 0):
            return True, "Auto square-off 3:00 PM"
        if side == "BUY" and ind.ema9 and ind.ema21 and ind.ema9 < ind.ema21 and ind.ltp < ind.vwap:
            return True, "trend failed (EMA9<EMA21 below VWAP)"
        if side == "SELL" and ind.ema9 and ind.ema21 and ind.ema9 > ind.ema21 and ind.ltp > ind.vwap:
            return True, "trend failed (EMA9>EMA21 above VWAP)"
        return False, ""

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind, sym, ltp = snap.indicators, snap.symbol, snap.tick.ltp
        now = now_ist().replace(tzinfo=None)
        t = now.time()
        if not (self.START <= t <= self.END) or not ind.ema9 or not ind.ema21 or not ind.ema50 or not ind.vwap:
            return "HOLD", None
        k = _day_key(sym, now)
        if self._count.get(k, 0) >= self.MAX_PER_DAY or self._losses.get(k, 0) >= self.MAX_LOSSES:
            return "HOLD", None
        last = self._last.get(sym)
        if last and (now - last).total_seconds() < self.COOLDOWN_S:
            return "HOLD", None
        if ind.adx_14 < self.ADX_MIN or len(snap.candles_1min) < self.PULLBACK_BARS + 1:
            return "HOLD", None
        cur = snap.candles_1min[-1]
        recent = snap.candles_1min[-(self.PULLBACK_BARS + 1):-1]
        touch = ltp * self.TOUCH_PCT / 100
        up = ind.ema9 > ind.ema21 > ind.ema50 and ltp > ind.vwap and ind.plus_di > ind.minus_di
        dn = ind.ema9 < ind.ema21 < ind.ema50 and ltp < ind.vwap and ind.minus_di > ind.plus_di
        if up and self.RSI_LO <= ind.rsi_14 <= self.RSI_HI:
            pulled = any(c.low <= ind.ema21 + touch for c in recent)
            if pulled and cur.close > cur.open and cur.close > ind.ema9:
                stop = min(c.low for c in recent)
                self._count[k] = self._count.get(k, 0) + 1; self._last[sym] = now
                return self._signal(sym, "BUY", ltp, stop, f"adx {ind.adx_14:.0f} rsi {ind.rsi_14:.0f}")
        if dn and (100 - self.RSI_HI) <= ind.rsi_14 <= (100 - self.RSI_LO):
            pulled = any(c.high >= ind.ema21 - touch for c in recent)
            if pulled and cur.close < cur.open and cur.close < ind.ema9:
                stop = max(c.high for c in recent)
                self._count[k] = self._count.get(k, 0) + 1; self._last[sym] = now
                return self._signal(sym, "SELL", ltp, stop, f"adx {ind.adx_14:.0f} rsi {ind.rsi_14:.0f}")
        return "HOLD", None
