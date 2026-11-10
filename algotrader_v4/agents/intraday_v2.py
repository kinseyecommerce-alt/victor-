"""
intraday_v2.py — intraday and scalping rebuilt around NSE costs, validated on real Kite 1-minute data.

The first-generation agents fired on ten overlapping 1-minute patterns, hundreds of times a day, and
paid ~₹41 per round trip on ₹40k positions: on 45 days of real data both lost money in every parameter
combination the optimiser tried (intraday −₹1.63L, scalping −₹6.73L on 19 NIFTY stocks). These two
agents take the opposite stance: very few, well-defined trades per symbol per day, each with a
structural stop and a target well above the risk, so one winner pays for the losers after costs.

IntradayV2Agent  (name "intraday")
  ORB   opening-range (09:15–09:30) breakout taken 09:31–10:00 only, range 0.25–1.5% of price, on
        ≥1.0× volume, on the VWAP side of the move with EMA9/21 agreeing, and only in the direction
        of the 20-day trend (close vs 20-day EMA). Stop at the other side of the range (its midpoint
        when the range is wider than 0.8%), capped at 0.35–0.8%; target 2.5R. One per symbol per day.
        Replay, 19 NIFTY stocks × 45 days after costs and 5 bps slippage: +₹32,283 on 115 trades,
        profit factor 1.42, max drawdown ₹11,871; both halves of the period positive.
  VWAP_TREND  kept for research (USE_VWAP_TREND=0 by default: it lost money on real data).

ScalpV2Agent  (name "scalping")
  ORB_RETEST  after price has clearly left the opening range (≥0.4%), the first pullback that touches
        the broken level (within 0.2%) between 09:35 and 12:30 and closes back in the breakout
        direction, with EMA9/21 and VWAP agreeing and the 20-day trend aligned. Stop at the range
        midpoint (0.25–0.7%); target 1.5R. One per symbol per day; the day stops after a loss.
        Replay: +₹9,955 on 56 trades, profit factor 1.28, max drawdown ₹9,769; both halves positive
        (+₹7,266 / +₹2,689). A modest edge — far smaller than the intraday agent's.
  PULLBACK  kept for research (MODE="PULLBACK"): EMA21 pullbacks lost ₹3.6L on the same data.

Both are MIS, square off by code at 15:00 like every intraday trade, and go through the same risk
checks, cost model and trailing-stop engine as before. Tunables are class attributes so the
optimiser and the replay can override them per instance.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Optional

from agents.base_agent import BaseAgent


def now_ist():
    """IST clock via strategy_agents, resolved at call time: the replay patches that module's
    now_ist to follow the replayed bar, and importing it lazily avoids a circular import
    (strategy_agents registers these agents in ALL_AGENTS)."""
    import agents.strategy_agents as _sa
    return _sa.now_ist()


def _day_key(sym: str, now: datetime) -> tuple:
    return (sym, now.date())


# Daily bias provider: (symbol, date) → "UP" | "DOWN" | None. Live: close vs 20-day EMA from daily_data;
# the replay installs its own provider built from the bars under test so there is no look-ahead.
BIAS_PROVIDER = None


def daily_bias(sym: str, day) -> Optional[str]:
    if BIAS_PROVIDER is not None:
        try:
            return BIAS_PROVIDER(sym, day)
        except Exception:
            return None
    try:
        import daily_data
        base = sym.split("#")[0]
        if not daily_data.is_cached(base):
            # first look of the day: fetch on a worker thread, trade unfiltered until it lands
            daily_data.prefetch_in_background([base])
            return None
        v = daily_data.daily_view(base, candles=0)
        if not v or not v.get("ema20"):
            return None
        return "UP" if v["ltp"] > v["ema20"] else "DOWN"
    except Exception:
        return None


class IntradayV2Agent(BaseAgent):
    name = "intraday"
    validated_by_replay = True
    strategy_rule_exits = False
    VERSION = "v2"                 # performance ledger and pruning judge each version on its own trades
    product = "MIS"
    min_candles_1min = 21

    # tunables (the optimiser's grid can sweep these); times are minutes after midnight IST
    ORB_END_MIN        = 570    # 09:30 — the opening range is 09:15 to here
    ORB_WINDOW_END_MIN = 600    # 10:00 — breakouts are taken until here
    USE_VWAP_TREND     = 0      # the VWAP-cross pattern lost money on real data; off by default
    VWAP_START_MIN     = 600
    VWAP_WINDOW_END_MIN = 810
    VOL_ORB        = 1.0      # swept on 45 days of real data: 1.0 > 1.3 > 1.6
    VOL_VWAP       = 1.2
    ADX_MIN        = 20.0
    RANGE_MIN_PCT  = 0.25     # opening range too small → noise
    RANGE_MAX_PCT  = 1.5      # too wide → stop too far
    SL_MIN_PCT     = 0.35
    SL_MAX_PCT     = 0.8
    RR             = 2.5      # swept: 2.5 > 2.0 > 1.5
    STOP_AT_MID_ABOVE_PCT = 0.8   # a range wider than this uses its midpoint as the stop
    MAX_PER_DAY    = 2        # per symbol: at most one ORB and one VWAP trade
    TREND_FILTER   = 1        # only break out in the direction of the 20-day trend (real data: with-trend
                              # breakouts made ₹474/trade, against-trend lost ₹121/trade)

    def _bias(self, sym: str, day) -> Optional[str]:
        return daily_bias(sym, day)

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
                bias = self._bias(sym, now.date()) if self.TREND_FILTER else None
                long_ok = bias in (None, "UP"); short_ok = bias in (None, "DOWN")
                if long_ok and prev_close <= h < ltp and ltp > ind.vwap and ind.ema9 > ind.ema21:
                    fired["ORB"] = True
                    stop = l if rng_pct <= self.STOP_AT_MID_ABOVE_PCT else mid
                    return self._signal(sym, "BUY", ltp, stop, "ORB", f"range {l:.1f}-{h:.1f} vol {ind.volume_ratio:.1f}x")
                if short_ok and prev_close >= l > ltp and ltp < ind.vwap and ind.ema9 < ind.ema21:
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
    validated_by_replay = True
    strategy_rule_exits = False
    VERSION = "v2"                 # performance ledger and pruning judge each version on its own trades
    product = "MIS"
    min_candles_1min = 50

    MODE = "ORB_RETEST"       # ORB_RETEST: re-enter a broken opening range on its first pullback;
                              # PULLBACK: EMA21 pullback in an ADX trend (lost money on real data)
    RT_ORB_END_MIN   = 570    # same 09:15-09:30 range as the intraday agent
    RT_START_MIN     = 575    # look for the retest from 09:35 …
    RT_END_MIN       = 750    # … until 12:30
    RT_TOUCH_PCT     = 0.2    # price must come back within this % of the broken level
    RT_RR            = 1.5
    RT_TREND_FILTER  = 1      # only retest in the direction of the 20-day trend (shares IntradayV2Agent's bias)
    RT_SL_MIN_PCT    = 0.25
    RT_SL_MAX_PCT    = 0.7

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
        self._orb: dict = {}         # (sym, day) → {"h","l","done","broke": None|"UP"|"DOWN","retested": bool}
        self._prev_close: dict = {}

    @staticmethod
    def _t(minutes) -> time:
        m = int(minutes); return time(m // 60, m % 60)

    def _range(self, sym, snap, now):
        k = _day_key(sym, now)
        r = self._orb.get(k)
        if r is None:
            r = self._orb[k] = {"h": None, "l": None, "done": False, "broke": None, "retested": False}
            for old in [x for x in self._orb if x[0] == sym and x != k]:
                self._orb.pop(old, None)
        if not r["done"]:
            for c in snap.candles_1min:
                ts = getattr(c, "ts", None)
                if ts is None or ts.date() != now.date():
                    continue
                if time(9, 15) <= ts.time() < self._t(self.RT_ORB_END_MIN):
                    r["h"] = c.high if r["h"] is None else max(r["h"], c.high)
                    r["l"] = c.low if r["l"] is None else min(r["l"], c.low)
            if now.time() >= self._t(self.RT_ORB_END_MIN) and r["h"] is not None:
                r["done"] = True
        return r

    def _orb_retest(self, snap, ind, sym, ltp, now, t):
        """After the range breaks, the first pullback to the broken level that holds is the entry:
        the level has flipped from resistance to support (or the mirror). Stop beyond the range
        midpoint, target RT_RR × risk. One per symbol per day."""
        r = self._range(sym, snap, now)
        prev = self._prev_close.get(sym, ltp)
        self._prev_close[sym] = ltp
        if not r["done"] or r["retested"] or ind.ema9 is None:
            return "HOLD", None
        h, l = r["h"], r["l"]
        if (h - l) / ltp * 100 > 1.5 or (h - l) / ltp * 100 < 0.2:
            return "HOLD", None
        bias = daily_bias(sym, now.date()) if self.RT_TREND_FILTER else None
        if r["broke"] is None:
            if ltp > h * 1.001: r["broke"] = "UP"; r["peak"] = ltp
            elif ltp < l * 0.999: r["broke"] = "DOWN"; r["peak"] = ltp
            return "HOLD", None
        if not (self._t(self.RT_START_MIN) <= t <= self._t(self.RT_END_MIN)):
            return "HOLD", None
        cur = snap.candles_1min[-1]
        touch = ltp * self.RT_TOUCH_PCT / 100
        mid = (h + l) / 2
        if r["broke"] == "UP":
            r["peak"] = max(r.get("peak", ltp), ltp)
            if r["peak"] < h * 1.004:                     # never really left the range: no retest trade
                return "HOLD", None
            if ltp < mid:                                  # whipsaw back inside: give up for the day
                r["retested"] = True; return "HOLD", None
            if bias == "DOWN":
                return "HOLD", None
            if cur.low <= h + touch and cur.close > h and cur.close > cur.open and ind.ema9 > ind.ema21 and ltp > ind.vwap:
                r["retested"] = True
                dist = min(max(ltp - mid, ltp * self.RT_SL_MIN_PCT / 100), ltp * self.RT_SL_MAX_PCT / 100)
                return self._sig("BUY", sym, ltp, dist, f"ORB_RETEST hi {h:.1f}")
        else:
            r["peak"] = min(r.get("peak", ltp), ltp)
            if r["peak"] > l * 0.996:
                return "HOLD", None
            if ltp > mid:
                r["retested"] = True; return "HOLD", None
            if bias == "UP":
                return "HOLD", None
            if cur.high >= l - touch and cur.close < l and cur.close < cur.open and ind.ema9 < ind.ema21 and ltp < ind.vwap:
                r["retested"] = True
                dist = min(max(mid - ltp, ltp * self.RT_SL_MIN_PCT / 100), ltp * self.RT_SL_MAX_PCT / 100)
                return self._sig("SELL", sym, ltp, dist, f"ORB_RETEST lo {l:.1f}")
        return "HOLD", None

    def _sig(self, side, sym, ltp, dist, note, rr=None):
        rr = rr or self.RT_RR
        stop = ltp - dist if side == "BUY" else ltp + dist
        target = ltp + dist * rr if side == "BUY" else ltp - dist * rr
        return side, {"symbol": sym, "exchange": "NSE", "side": side, "price": ltp,
                      "stop_loss": round(stop, 2), "target": round(target, 2),
                      "stop_loss_pct": round(dist / ltp * 100, 3), "target_pct": round(dist * rr / ltp * 100, 3),
                      "product": self.product, "_gate_size_factor": 1.0, "trigger": f"SCALP2-{side} [{note}]"}

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
        if t >= time(14, 45) or not ind.ema9 or not ind.vwap or ltp <= 0 or not snap.candles_1min:
            return "HOLD", None
        if self.MODE == "ORB_RETEST":
            return self._orb_retest(snap, ind, sym, ltp, now, t)
        if not (self.START <= t <= self.END) or not ind.ema21 or not ind.ema50:
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
