"""
agents/strategy_agents.py  (v3 — tick-driven)
All four agents now call evaluate_tick() on every 1-second market update.
Entry logic reads from live LiveIndicators (EMA, RSI, VWAP, MACD, BB, ATR).
"""
from __future__ import annotations

from datetime import datetime
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
    name    = "scalping"
    product = "MIS"
    min_candles_1min = 10
    SL_PCT  = 0.25    # tight SL for scalping
    TGT_PCT = 0.50

    # Track previous EMA9 for cross detection
    _prev_ema9: dict[str, float] = {}
    _prev_ltp:  dict[str, float] = {}

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        sym = snap.symbol
        ind = snap.indicators
        ltp = snap.tick.ltp

        if not ind.ema9:
            return "HOLD", None

        prev_ema9 = self._prev_ema9.get(sym, ind.ema9)
        prev_ltp  = self._prev_ltp.get(sym, ltp)

        self._prev_ema9[sym] = ind.ema9
        self._prev_ltp[sym]  = ltp

        # ── Spread filter (avoid wide spreads on scalp) ──────────────
        max_spread = ltp * 0.0003   # 0.03% max allowed spread
        spread = snap.tick.ask - snap.tick.bid
        if spread > max_spread:
            return "HOLD", None

        # ── Bullish micro-cross ──────────────────────────────────
        bull_cross = prev_ltp < prev_ema9 and ltp > ind.ema9
        rsi_ok     = 52 < ind.rsi_7 < 72
        vol_spike  = ind.volume_ratio >= 1.4
        momentum   = ind.momentum in ("STRONG_UP", "WEAK_UP")

        if bull_cross and rsi_ok and vol_spike and momentum:
            sl  = round(ltp * (1 - self.SL_PCT  / 100), 2)
            tgt = round(ltp * (1 + self.TGT_PCT / 100), 2)
            return "BUY", {
                "symbol":    sym,
                "exchange":  "NSE",
                "side":      "BUY",
                "price":     ltp,
                "stop_loss": sl,
                "target":    tgt,
                "product":   self.product,
                "trigger":   f"EMA9-CROSS vol={ind.volume_ratio:.1f}x rsi={ind.rsi_7:.0f}",
            }

        # ── Bearish micro-cross ──────────────────────────────────
        bear_cross  = prev_ltp > prev_ema9 and ltp < ind.ema9
        rsi_bear    = 28 < ind.rsi_7 < 48
        mom_down    = ind.momentum in ("STRONG_DOWN", "WEAK_DOWN")

        if bear_cross and rsi_bear and vol_spike and mom_down:
            sl  = round(ltp * (1 + self.SL_PCT  / 100), 2)
            tgt = round(ltp * (1 - self.TGT_PCT / 100), 2)
            return "SELL", {
                "symbol":    sym,
                "exchange":  "NSE",
                "side":      "SELL",
                "price":     ltp,
                "stop_loss": sl,
                "target":    tgt,
                "product":   self.product,
                "trigger":   f"EMA9-CROSS-SHORT vol={ind.volume_ratio:.1f}x",
            }

        return "HOLD", None

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", ind.ltp)
        ltp   = ind.ltp
        side  = "BUY" if pos.get("quantity", 0) > 0 else "SELL"
        if not entry:
            return False, ""

        if side == "BUY":
            sl  = entry * (1 - self.SL_PCT  / 100)
            tgt = entry * (1 + self.TGT_PCT / 100)
            if ltp <= sl:   return True, f"Scalp SL ₹{ltp:.2f}"
            if ltp >= tgt:  return True, f"Scalp target ₹{ltp:.2f}"
            if ind.momentum in ("STRONG_DOWN", "WEAK_DOWN"):
                            return True, "Momentum reversal exit"
        else:
            sl  = entry * (1 + self.SL_PCT  / 100)
            tgt = entry * (1 - self.TGT_PCT / 100)
            if ltp >= sl:   return True, f"Scalp SL ₹{ltp:.2f}"
            if ltp <= tgt:  return True, f"Scalp target ₹{ltp:.2f}"
            if ind.momentum in ("STRONG_UP", "WEAK_UP"):
                            return True, "Momentum reversal exit"

        now = datetime.now().time()
        if now.hour >= 15:  return True, "Auto square-off 3:00 PM"
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