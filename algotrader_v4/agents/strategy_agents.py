"""
agents/strategy_agents.py  (v3 — tick-driven)
All four agents now call evaluate_tick() on every 1-second market update.
Entry logic reads from live LiveIndicators (EMA, RSI, VWAP, MACD, BB, ATR).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

from agents.base_agent import BaseAgent
from tick_engine import MarketSnapshot, LiveIndicators
from risk_manager import risk_manager


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  INTRADAY  —  MIS, VWAP + EMA + RSI + volume confirmation
# ═══════════════════════════════════════════════════════════════════════════════

class IntradayAgent(BaseAgent):
    name    = "intraday"
    product = "MIS"
    min_candles_1min = 21

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind = snap.indicators
        ltp = snap.tick.ltp

        # ── Entry conditions (ALL must hold) ────────────────────────────
        vwap_above  = ltp > ind.vwap > 0          # price above VWAP
        ema_bullish = ind.ema9 > ind.ema21 > 0    # short EMA above long EMA
        rsi_ok      = 45 < ind.rsi_14 < 67        # momentum not exhausted
        macd_bull   = ind.macd_hist > 0            # MACD histogram positive
        vol_spike   = ind.volume_ratio >= 1.3      # above-average volume

        # ── Trend filter ────────────────────────────────────────────
        trend_up    = ind.trend == "UP"

        if vwap_above and ema_bullish and rsi_ok and macd_bull and vol_spike and trend_up:
            sl  = round(max(ind.vwap, risk_manager.sl_price(ltp, "BUY")), 2)
            tgt = risk_manager.target_price(ltp, "BUY")
            return "BUY", {
                "symbol":     snap.symbol,
                "exchange":   "NSE",
                "side":       "BUY",
                "price":      ltp,
                "stop_loss":  sl,
                "target":     tgt,
                "product":    self.product,
                "trigger":    "VWAP+EMA+MACD+VOL",
            }

        # ── Short side: price breaks below VWAP with momentum ─────────
        vwap_below  = ltp < ind.vwap > 0
        ema_bear    = ind.ema9 < ind.ema21
        rsi_bear    = 33 < ind.rsi_14 < 55
        macd_bear   = ind.macd_hist < 0

        if vwap_below and ema_bear and rsi_bear and macd_bear and vol_spike:
            sl  = round(min(ind.vwap, risk_manager.sl_price(ltp, "SELL")), 2)
            tgt = risk_manager.target_price(ltp, "SELL")
            return "SELL", {
                "symbol":    snap.symbol,
                "exchange":  "NSE",
                "side":      "SELL",
                "price":     ltp,
                "stop_loss": sl,
                "target":    tgt,
                "product":   self.product,
                "trigger":   "VWAP-BREAK+EMA+MACD",
            }

        return "HOLD", None

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", ind.ltp)
        ltp   = ind.ltp
        side  = "BUY" if pos.get("quantity", 0) > 0 else "SELL"

        sl  = risk_manager.sl_price(entry, side)
        tgt = risk_manager.target_price(entry, side)

        if side == "BUY":
            if ltp <= sl:              return True, f"SL hit ₹{ltp:.2f}"
            if ltp >= tgt:             return True, f"Target hit ₹{ltp:.2f}"
            if ind.trend == "DOWN" and ind.macd_hist < 0:
                                       return True, "Trend reversal exit"
        else:
            if ltp >= sl:              return True, f"SL hit ₹{ltp:.2f}"
            if ltp <= tgt:             return True, f"Target hit ₹{ltp:.2f}"
            if ind.trend == "UP" and ind.macd_hist > 0:
                                       return True, "Trend reversal exit"

        now = datetime.now().time()
        if now.hour >= 15:             return True, "Auto square-off 3:00 PM"
        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  F&O  —  NRML, IV proxy + OI + RSI extremes + Bollinger breakout
# ═══════════════════════════════════════════════════════════════════════════════

class FnOAgent(BaseAgent):
    name    = "fno"
    product = "NRML"
    min_candles_1min = 26
    IV_THRESHOLD = 40

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind = snap.indicators
        ltp = snap.tick.ltp

        # IV proxy: ATR relative to recent average (computed in indicators)
        # We use BB width as IV proxy here (available in LiveIndicators)
        bb_width = 0.0
        if ind.bb_upper and ind.bb_lower and ind.bb_mid:
            bb_width = (ind.bb_upper - ind.bb_lower) / ind.bb_mid * 100

        low_iv   = bb_width < 2.0             # IV relatively low → buy premium
        rsi_ext  = ind.rsi_14 < 38 or ind.rsi_14 > 62

        if not (low_iv and rsi_ext):
            return "HOLD", None

        # Buy Call (bullish)
        if ind.rsi_14 > 62 and ind.trend == "UP" and ltp > ind.bb_upper:
            tgt = round(ltp * 1.5, 2)   # 50% gain on premium
            sl  = round(ltp * 0.65, 2)  # 35% loss limit
            return "BUY", {
                "symbol":      snap.symbol,
                "exchange":    "NFO",
                "side":        "BUY",
                "option_type": "CE",
                "price":       ltp,
                "stop_loss":   sl,
                "target":      tgt,
                "product":     self.product,
                "trigger":     f"BB-BREAKOUT-CALL rsi={ind.rsi_14:.0f}",
            }

        # Buy Put (bearish)
        if ind.rsi_14 < 38 and ind.trend == "DOWN" and ltp < ind.bb_lower:
            tgt = round(ltp * 1.5, 2)
            sl  = round(ltp * 0.65, 2)
            return "BUY", {
                "symbol":      snap.symbol,
                "exchange":    "NFO",
                "side":        "BUY",
                "option_type": "PE",
                "price":       ltp,
                "stop_loss":   sl,
                "target":      tgt,
                "product":     self.product,
                "trigger":     f"BB-BREAKDOWN-PUT rsi={ind.rsi_14:.0f}",
            }

        return "HOLD", None

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", ind.ltp)
        ltp   = ind.ltp
        if not entry:
            return False, ""
        change = (ltp - entry) / entry * 100
        if change <= -35:  return True, f"Option loss 35% ₹{ltp:.2f}"
        if change >= 80:   return True, f"Option profit 80% ₹{ltp:.2f}"
        # Exit if mean-reversion: RSI returns to neutral
        if 45 < ind.rsi_14 < 55:
            return True, "RSI neutral — exit option"
        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SWING  —  CNC, EMA200 trend + EMA50 bounce + RSI + ATR filter
# ═══════════════════════════════════════════════════════════════════════════════

class SwingAgent(BaseAgent):
    name    = "swing"
    product = "CNC"
    min_candles_1min = 200   # needs EMA200

    # Swing only evaluates once per minute (not every tick — saves noise)
    _last_eval: dict[str, float] = {}

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        import time as _time
        sym = snap.symbol
        now = _time.time()
        # Throttle: evaluate at most once every 60 seconds per symbol
        if now - self._last_eval.get(sym, 0) < 60:
            return "HOLD", None
        self._last_eval[sym] = now

        ind = snap.indicators
        ltp = snap.tick.ltp

        # Needs EMA200 for long-term trend
        if not ind.ema200:
            return "HOLD", None

        # Long-term uptrend
        trend_ok   = ltp > ind.ema200
        # Price pulling back to EMA50 (within 1.5%)
        ema50_near = ind.ema50 > 0 and abs(ltp - ind.ema50) / ind.ema50 < 0.015
        # Short EMA above long EMA
        ema_up     = ind.ema21 > 0 and ind.ema50 > 0 and ind.ema21 > ind.ema50
        # RSI in accumulation zone
        rsi_ok     = 40 < ind.rsi_14 < 60
        # Low volatility pullback (ATR not spiking)
        low_vol    = ind.volatility != "HIGH"

        if trend_ok and ema50_near and rsi_ok and low_vol:
            sl  = round(ltp * 0.97, 2)   # 3% SL for swing
            tgt = round(ltp * 1.08, 2)   # 8% target
            return "BUY", {
                "symbol":    sym,
                "exchange":  "NSE",
                "side":      "BUY",
                "price":     ltp,
                "stop_loss": sl,
                "target":    tgt,
                "product":   self.product,
                "trigger":   f"EMA50-BOUNCE trend=UP rsi={ind.rsi_14:.0f}",
            }

        return "HOLD", None

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", ind.ltp)
        ltp   = ind.ltp
        if not entry:
            return False, ""
        if ltp <= entry * 0.97:   return True, f"Swing SL ₹{ltp:.2f}"
        if ltp >= entry * 1.08:   return True, f"Swing target ₹{ltp:.2f}"
        # Trend broke down
        if ind.trend == "DOWN" and ind.ema9 < ind.ema21:
            return True, "Trend breakdown exit"
        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SCALPING  —  MIS, EMA9 micro-cross + RSI + bid-ask spread + volume
# ═══════════════════════════════════════════════════════════════════════════════

class ScalpingAgent(BaseAgent):
    """
    High-frequency intraday scalper.

    Entry: EMA9 micro-cross confirmed by 8-factor score (≥5 required).
    SL/Target: ATR-based (0.6×ATR / 1.2×ATR) for dynamic risk sizing.
    Filters: time-of-day, level proximity, volatility regime, loss-streak cooldown.
    """
    name    = "scalping"
    product = "MIS"
    min_candles_1min = 15

    # ATR multipliers — 1:2 R:R
    SL_ATR   = 0.6
    TGT_ATR  = 1.2

    # Fixed fallbacks when ATR is unavailable
    SL_PCT   = 0.25
    TGT_PCT  = 0.50

    # Minimum score (out of 8 factors) required to enter
    MIN_SCORE = 5

    # Per-symbol rolling state (class-level — shared across instances)
    _prev_ema9:      dict[str, float]    = {}
    _prev_ltp:       dict[str, float]    = {}
    _loss_streak:    dict[str, int]      = {}
    _cooldown_until: dict[str, datetime] = {}

    # ── Entry ─────────────────────────────────────────────────────────────────

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        sym = snap.symbol
        ind = snap.indicators
        ltp = snap.tick.ltp

        if not ind.ema9:
            return "HOLD", None

        # ── Guard 1: time-of-day ────────────────────────────────────────────
        t = datetime.now().time()
        if time(9, 15) <= t < time(9, 30):   # chaotic open — skip
            return "HOLD", None
        if t >= time(14, 40):                 # wind-down — no new scalps
            return "HOLD", None

        # ── Guard 2: loss-streak cooldown ───────────────────────────────────
        cd = self._cooldown_until.get(sym)
        if cd and datetime.now() < cd:
            return "HOLD", None

        # ── Guard 3: spread filter ───────────────────────────────────────────
        spread = snap.tick.ask - snap.tick.bid
        if spread > ltp * 0.0004:             # 0.04% max spread
            return "HOLD", None

        # ── Guard 4: volatility regime ──────────────────────────────────────
        atr = ind.atr_14 or 0.0
        atr_ratio = atr / ltp if ltp > 0 else 0.0
        if atr_ratio > 0.005:                 # too volatile — wide stops required
            return "HOLD", None
        if atr_ratio < 0.0003:                # dead market — no movement
            return "HOLD", None

        # ── EMA9 micro-cross detection ───────────────────────────────────────
        prev_ema9 = self._prev_ema9.get(sym, ind.ema9)
        prev_ltp  = self._prev_ltp.get(sym, ltp)
        self._prev_ema9[sym] = ind.ema9
        self._prev_ltp[sym]  = ltp

        bull_cross = prev_ltp < prev_ema9 and ltp > ind.ema9
        bear_cross = prev_ltp > prev_ema9 and ltp < ind.ema9
        if not (bull_cross or bear_cross):
            return "HOLD", None

        action = "BUY" if bull_cross else "SELL"

        # ── Multi-factor scoring ─────────────────────────────────────────────
        score, reasons = self._score_setup(snap, ind, ltp, action)
        if score < self.MIN_SCORE:
            return "HOLD", None

        # ── Level proximity guard ────────────────────────────────────────────
        if not self._level_ok(sym, ltp, action):
            return "HOLD", None

        # ── ATR-based SL & target ────────────────────────────────────────────
        sl_dist  = max(atr * self.SL_ATR,  ltp * self.SL_PCT  / 100)
        tgt_dist = max(atr * self.TGT_ATR, ltp * self.TGT_PCT / 100)

        if action == "BUY":
            sl  = round(ltp - sl_dist, 2)
            tgt = round(ltp + tgt_dist, 2)
        else:
            sl  = round(ltp + sl_dist, 2)
            tgt = round(ltp - tgt_dist, 2)

        return action, {
            "symbol":       sym,
            "exchange":     "NSE",
            "side":         action,
            "price":        ltp,
            "stop_loss":    sl,
            "target":       tgt,
            "stop_loss_pct": round(sl_dist / ltp * 100, 3),
            "target_pct":   round(tgt_dist / ltp * 100, 3),
            "product":      self.product,
            "trigger":      f"SCALP-{action} score={score}/{self.MIN_SCORE}min "
                            f"{' '.join(reasons[:4])}",
        }

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_setup(
        self, snap: MarketSnapshot, ind: LiveIndicators, ltp: float, action: str
    ) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        is_buy = action == "BUY"

        # 1. VWAP alignment (price on correct side of VWAP)
        if ind.vwap and ind.vwap > 0:
            if (is_buy and ltp > ind.vwap) or (not is_buy and ltp < ind.vwap):
                score += 1; reasons.append("VWAP✓")

        # 2. RSI-7 in healthy zone — not extended
        rsi = ind.rsi_7
        if is_buy and 50 < rsi < 70:
            score += 1; reasons.append(f"RSI{rsi:.0f}")
        elif not is_buy and 30 < rsi < 50:
            score += 1; reasons.append(f"RSI{rsi:.0f}")

        # 3. Volume surge ≥ 1.5×
        if ind.volume_ratio >= 1.5:
            score += 1; reasons.append(f"VOL{ind.volume_ratio:.1f}x")
        elif ind.volume_ratio >= 1.3:
            score += 1  # partial — count but don't annotate

        # 4. ADX confirms trend is established (not choppy)
        if ind.adx_14 >= 22:
            score += 1; reasons.append(f"ADX{ind.adx_14:.0f}")

        # 5. MACD histogram confirms direction
        if (is_buy and ind.macd_hist > 0) or (not is_buy and ind.macd_hist < 0):
            score += 1; reasons.append("MACD✓")

        # 6. Candle microstructure — ≥2 of last 3 candles confirm direction
        if len(snap.candles_1min) >= 3:
            last3 = snap.candles_1min[-3:]
            if is_buy:
                green = sum(1 for c in last3 if c.close >= c.open)
                if green >= 2:
                    score += 1; reasons.append(f"{green}G")
            else:
                red = sum(1 for c in last3 if c.close <= c.open)
                if red >= 2:
                    score += 1; reasons.append(f"{red}R")

        # 7. Price velocity — last 5 closes moving in signal direction
        if len(snap.candles_1min) >= 5:
            closes = [c.close for c in snap.candles_1min[-5:]]
            if (is_buy and closes[-1] > closes[0]) or (not is_buy and closes[-1] < closes[0]):
                score += 1; reasons.append("VEL✓")

        # 8. EMA21 macro-trend alignment (trade with the bigger trend)
        if ind.ema21 and ind.ema21 > 0:
            if (is_buy and ltp > ind.ema21) or (not is_buy and ltp < ind.ema21):
                score += 1; reasons.append("EMA21✓")

        return score, reasons

    # ── Level proximity guard ─────────────────────────────────────────────────

    def _level_ok(self, sym: str, ltp: float, side: str) -> bool:
        try:
            from levels_engine import get_levels
            lvls = get_levels(sym)
            if not lvls:
                return True
            threshold = ltp * 0.0015   # block if within 0.15% of opposing level
            resistance_keys = ("r1", "r2", "pdh", "weekly_high", "vwap_upper_1")
            support_keys    = ("s1", "s2", "pdl", "weekly_low",  "vwap_lower_1")
            if side == "BUY":
                for k in resistance_keys:
                    v = lvls.get(k)
                    if v and 0 < v - ltp < threshold:
                        return False   # buying into resistance wall
            else:
                for k in support_keys:
                    v = lvls.get(k)
                    if v and 0 < ltp - v < threshold:
                        return False   # selling into support wall
        except Exception:
            pass
        return True

    # ── Loss-streak tracking ──────────────────────────────────────────────────

    def _record_outcome(self, sym: str, won: bool) -> None:
        if won:
            self._loss_streak[sym] = 0
        else:
            streak = self._loss_streak.get(sym, 0) + 1
            self._loss_streak[sym] = streak
            if streak >= 3:
                self._cooldown_until[sym] = datetime.now() + timedelta(minutes=20)
                from loguru import logger
                logger.warning("[scalping] {} 3-loss streak — 20-min cooldown", sym)
            elif streak >= 2:
                self._cooldown_until[sym] = datetime.now() + timedelta(minutes=5)

    # ── Exit ──────────────────────────────────────────────────────────────────

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", ind.ltp)
        ltp   = ind.ltp
        sym   = pos.get("tradingsymbol", "")
        side  = "BUY" if pos.get("quantity", 0) > 0 else "SELL"
        if not entry or not ltp:
            return False, ""

        atr      = ind.atr_14 or 0.0
        sl_dist  = max(atr * self.SL_ATR,  entry * self.SL_PCT  / 100)
        tgt_dist = max(atr * self.TGT_ATR, entry * self.TGT_PCT / 100)

        if side == "BUY":
            sl, tgt = entry - sl_dist, entry + tgt_dist
            if ltp <= sl:
                self._record_outcome(sym, False)
                return True, f"Scalp SL ₹{ltp:.2f}"
            if ltp >= tgt:
                self._record_outcome(sym, True)
                return True, f"Scalp target ₹{ltp:.2f}"
            # Early exit: strong reversal confirmed by MACD flip
            if ind.momentum == "STRONG_DOWN" and ind.macd_hist < 0:
                self._record_outcome(sym, ltp > entry)
                return True, "Strong momentum reversal"
            # VWAP breakdown — losing VWAP support on a long is a bad sign
            if ind.vwap and ltp < ind.vwap * 0.9985:
                self._record_outcome(sym, ltp > entry)
                return True, "VWAP breakdown exit"
        else:
            sl, tgt = entry + sl_dist, entry - tgt_dist
            if ltp >= sl:
                self._record_outcome(sym, False)
                return True, f"Scalp SL ₹{ltp:.2f}"
            if ltp <= tgt:
                self._record_outcome(sym, True)
                return True, f"Scalp target ₹{ltp:.2f}"
            if ind.momentum == "STRONG_UP" and ind.macd_hist > 0:
                self._record_outcome(sym, ltp < entry)
                return True, "Strong momentum reversal"
            if ind.vwap and ltp > ind.vwap * 1.0015:
                self._record_outcome(sym, ltp < entry)
                return True, "VWAP breakout exit"

        # Hard auto-exit well before close (leave 15 min for TSL to close)
        if datetime.now().time() >= time(14, 55):
            return True, "Auto square-off 2:55 PM"

        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

ALL_AGENTS: dict[str, BaseAgent] = {
    "intraday": IntradayAgent(),
    "fno":      FnOAgent(),
    "swing":    SwingAgent(),
    "scalping": ScalpingAgent(),
}