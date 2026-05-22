"""
agents/strategy_agents.py  (v3 — tick-driven)
All four agents now call evaluate_tick() on every 1-second market update.
Entry logic reads from live LiveIndicators (EMA, RSI, VWAP, MACD, BB, ATR).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

from ist_clock import now_ist
from agents.base_agent import BaseAgent
from tick_engine import MarketSnapshot, LiveIndicators
from risk_manager import risk_manager


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  INTRADAY  —  MIS, 5-pattern never-miss architecture
# ═══════════════════════════════════════════════════════════════════════════════

class IntradayAgent(BaseAgent):
    """
    Never-miss NSE intraday agent — 5 entry patterns, all market sessions covered.

    Patterns (fire independently, best score wins each tick):
      1. VWAP_TREND   — price+EMA above VWAP with RSI/MACD/volume (trend continuation)
      2. EMA_PULLBACK — pullback into RSI 45-62 zone in a strong 3-EMA trend
      3. ORB_BREAK    — opening range breakout (9:30-10:30 execution window)
      4. BREAKOUT     — 15-bar high/low break with heavy volume (≥1.5×)
      5. VWAP_RECLAIM — fresh VWAP cross with volume (regime change entry)

    Context bonuses (added to every pattern base score):
      EMA full align (0-2), VWAP side (0-1), RSI zone (0-1),
      volume (0-1), MACD direction (0-1), institutional flow (0-1)

    Sizing tiers:  score 4 → 0.5×  |  5-6 → 0.75×  |  7+ → 1.0×
    Cooldown:      180s per direction (BUY/SELL tracked independently)
    SL/TGT:        ATR-based — SL=1.5×ATR14, TGT=2.5×ATR14
    """
    name    = "intraday"
    product = "MIS"
    min_candles_1min = 21

    SL_ATR      = 1.5
    TGT_ATR     = 2.5
    SL_MIN_PCT  = 0.5
    TGT_MIN_PCT = 0.8
    MIN_SCORE   = 4
    COOL_S      = 180   # 3-min per direction

    def __init__(self) -> None:
        super().__init__()
        # Per-symbol rolling state — instance-level so each agent is independent
        self._prev_above_vwap: dict = {}
        self._prev_ltp:        dict = {}
        self._prev_rsi:        dict = {}
        self._prev_squeeze:    dict = {}   # sym → squeeze_on last tick
        self._orb_high:        dict = {}
        self._orb_low:         dict = {}
        self._orb_fired:       dict = {}
        self._cool_ts:         dict = {}   # sym → {"BUY": datetime, "SELL": datetime}

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind = snap.indicators
        sym = snap.symbol
        ltp = snap.tick.ltp
        now = now_ist()
        t   = now.time().replace(tzinfo=None)

        if time(14, 50) <= t:
            return "HOLD", None

        self._update_orb(sym, snap, t)

        best_score, best_action, best_pattern = -1, "", ""
        for pat_fn in (self._pat_vwap_trend, self._pat_ema_pullback,
                       self._pat_orb_break, self._pat_breakout, self._pat_vwap_reclaim,
                       self._pat_ttm_squeeze, self._pat_vwap_band_revert):
            try:
                action, base, pname = pat_fn(sym, snap, ind, ltp, t)
            except Exception:
                continue
            if not action:
                continue
            total = base + self._ctx_bonus(action, sym, ind, ltp)
            if total > best_score:
                best_score, best_action, best_pattern = total, action, pname

        self._update_state(sym, ind, ltp)

        if best_score < self.MIN_SCORE or not best_action:
            return "HOLD", None

        cools = self._cool_ts.setdefault(sym, {})
        last  = cools.get(best_action)
        if last and (now - last).total_seconds() < self.COOL_S:
            return "HOLD", None
        cools[best_action] = now

        atr      = ind.atr_14 or ltp * 0.005
        sl_dist  = max(atr * self.SL_ATR,  ltp * self.SL_MIN_PCT  / 100)
        tgt_dist = max(atr * self.TGT_ATR, ltp * self.TGT_MIN_PCT / 100)
        sf       = 1.0 if best_score >= 7 else (0.75 if best_score >= 5 else 0.5)

        if best_action == "BUY":
            sl  = round(ltp - sl_dist, 2)
            tgt = round(ltp + tgt_dist, 2)
        else:
            sl  = round(ltp + sl_dist, 2)
            tgt = round(ltp - tgt_dist, 2)

        return best_action, {
            "symbol":            sym,
            "exchange":          "NSE",
            "side":              best_action,
            "price":             ltp,
            "stop_loss":         sl,
            "target":            tgt,
            "stop_loss_pct":     round(sl_dist  / ltp * 100, 3),
            "target_pct":        round(tgt_dist / ltp * 100, 3),
            "product":           self.product,
            "_gate_size_factor": sf,
            "trigger": (
                f"INTRA-{best_action} [{best_pattern}] score={best_score} "
                f"sf={sf} rsi={ind.rsi_14:.0f} trend={ind.trend}"
            ),
        }

    # ── Pattern 1: VWAP_TREND ─────────────────────────────────────────────────

    def _pat_vwap_trend(self, sym, snap, ind, ltp, t):
        if not ind.vwap or ind.vwap <= 0:
            return "", 0, ""
        if (ltp > ind.vwap and ind.ema9 > ind.ema21 > 0
                and 45 <= ind.rsi_14 <= 72
                and ind.macd_hist > 0 and ind.volume_ratio >= 1.3):
            return "BUY", 3, "VWAP_TREND"
        if (ltp < ind.vwap and ind.ema9 < ind.ema21 > 0
                and 28 <= ind.rsi_14 <= 55
                and ind.macd_hist < 0 and ind.volume_ratio >= 1.3):
            return "SELL", 3, "VWAP_TREND"
        return "", 0, ""

    # ── Pattern 2: EMA_PULLBACK ────────────────────────────────────────────────

    def _pat_ema_pullback(self, sym, snap, ind, ltp, t):
        prev_rsi = self._prev_rsi.get(sym, ind.rsi_14)
        ema_bull = ind.ema9 > ind.ema21 > 0 and ind.ema21 > ind.ema50 > 0
        ema_bear = ind.ema9 < ind.ema21 > 0 and ind.ema21 < ind.ema50 > 0
        # Uptrend pullback: RSI cooled from extended (>63) into 45-62 zone
        if ema_bull and prev_rsi > 63 and 45 <= ind.rsi_14 <= 62:
            return "BUY", 4, "EMA_PULLBACK"
        # Downtrend pullback: RSI recovered from oversold (<37) into 38-55 zone
        if ema_bear and prev_rsi < 37 and 38 <= ind.rsi_14 <= 55:
            return "SELL", 4, "EMA_PULLBACK"
        return "", 0, ""

    # ── Pattern 3: ORB_BREAK ──────────────────────────────────────────────────

    def _pat_orb_break(self, sym, snap, ind, ltp, t):
        if not (time(9, 30) <= t <= time(10, 30)):
            return "", 0, ""
        orb_h = self._orb_high.get(sym)
        orb_l = self._orb_low.get(sym)
        if not (orb_h and orb_l and orb_h > orb_l):
            return "", 0, ""
        if self._orb_fired.get(sym):
            return "", 0, ""
        prev = self._prev_ltp.get(sym, ltp)
        if prev <= orb_h and ltp > orb_h * 1.001 and ind.volume_ratio >= 1.2:
            self._orb_fired[sym] = True
            return "BUY", 5, "ORB_BREAK"
        if prev >= orb_l and ltp < orb_l * 0.999 and ind.volume_ratio >= 1.2:
            self._orb_fired[sym] = True
            return "SELL", 5, "ORB_BREAK"
        return "", 0, ""

    # ── Pattern 4: BREAKOUT ────────────────────────────────────────────────────

    def _pat_breakout(self, sym, snap, ind, ltp, t):
        n = 15
        if len(snap.candles_1min) < n or ind.volume_ratio < 1.5:
            return "", 0, ""
        last_n = snap.candles_1min[-n:]
        n_high = max(c.high for c in last_n)
        n_low  = min(c.low  for c in last_n)
        prev   = self._prev_ltp.get(sym, ltp)
        if prev < n_high and ltp > n_high:
            return "BUY",  3, "BREAKOUT"
        if prev > n_low  and ltp < n_low:
            return "SELL", 3, "BREAKOUT"
        return "", 0, ""

    # ── Pattern 5: VWAP_RECLAIM ────────────────────────────────────────────────

    def _pat_vwap_reclaim(self, sym, snap, ind, ltp, t):
        if not ind.vwap or ind.vwap <= 0:
            return "", 0, ""
        was_above = self._prev_above_vwap.get(sym, ltp >= ind.vwap)
        now_above = ltp > ind.vwap
        if was_above == now_above or ind.volume_ratio < 1.2:
            return "", 0, ""
        return ("BUY", 3, "VWAP_RECLAIM") if now_above else ("SELL", 3, "VWAP_RECLAIM")

    # ── Pattern 6: TTM_SQUEEZE ────────────────────────────────────────────────
    # Fire when squeeze releases (bands expand) with momentum aligned to direction.
    # Squeeze builds energy; the breakout bar is the entry signal.

    def _pat_ttm_squeeze(self, sym, snap, ind, ltp, t):
        if ind.squeeze_on:
            return "", 0, ""
        prev_squeeze = self._prev_squeeze.get(sym, True)
        if not prev_squeeze:
            return "", 0, ""
        mom = ind.squeeze_momentum
        if mom > 0 and 40 <= ind.rsi_14 <= 70 and ind.volume_ratio >= 1.2:
            return "BUY", 4, "TTM_SQUEEZE"
        if mom < 0 and 30 <= ind.rsi_14 <= 60 and ind.volume_ratio >= 1.2:
            return "SELL", 4, "TTM_SQUEEZE"
        return "", 0, ""

    def _pat_vwap_band_revert(self, sym, snap, ind, ltp, t):
        """Mean-reversion from VWAP 3σ band extremes — top NSE 2026 pattern."""
        u3, l3 = ind.vwap_upper3, ind.vwap_lower3
        u2, l2 = ind.vwap_upper2, ind.vwap_lower2
        if not (u3 > 0 and l3 > 0):
            return "", 0, ""
        prev_ltp = self._prev_ltp.get(sym, ltp)
        # Price touched 3σ upper band last tick and now pulling back below 2σ
        if prev_ltp >= u3 and ltp < u2 and ind.rsi_14 > 65 and ind.volume_ratio >= 1.0:
            return "SELL", 4, "VWAP_BAND_REVERT"
        # Price touched 3σ lower band last tick and now bouncing above 2σ
        if prev_ltp <= l3 and ltp > l2 and ind.rsi_14 < 35 and ind.volume_ratio >= 1.0:
            return "BUY", 4, "VWAP_BAND_REVERT"
        return "", 0, ""

    # ── Context bonus (+0 to +6 points added to every pattern) ───────────────

    def _ctx_bonus(self, action: str, sym: str, ind: LiveIndicators, ltp: float) -> int:
        b = 0
        is_buy = action == "BUY"

        # EMA alignment (0-2): full 3-EMA stack = +2, 2-EMA only = +1
        if is_buy:
            if ind.ema9 > ind.ema21 > 0 and ind.ema21 > ind.ema50 > 0:
                b += 2
            elif ind.ema9 > ind.ema21 > 0:
                b += 1
        else:
            if ind.ema9 < ind.ema21 > 0 and ind.ema21 < ind.ema50 > 0:
                b += 2
            elif ind.ema9 < ind.ema21 > 0:
                b += 1

        # VWAP side (0-1)
        if ind.vwap and ind.vwap > 0:
            if (is_buy and ltp > ind.vwap) or (not is_buy and ltp < ind.vwap):
                b += 1

        # RSI zone (0-1)
        if (is_buy and 44 <= ind.rsi_14 <= 72) or (not is_buy and 28 <= ind.rsi_14 <= 56):
            b += 1

        # Volume (0-1)
        if ind.volume_ratio >= 1.3:
            b += 1

        # MACD direction (0-1)
        if (is_buy and ind.macd_hist > 0) or (not is_buy and ind.macd_hist < 0):
            b += 1

        # Institutional flow (0-1) — sync cache, fails silently
        try:
            from institutional_flow import get_cached_score
            inst = get_cached_score(sym)
            if inst:
                score_val = inst.get("institutional_score", 50.0)
                if (is_buy and score_val > 55) or (not is_buy and score_val < 45):
                    b += 1
        except Exception:
            pass

        return b

    # ── ORB builder (called every tick 9:15-9:30) ─────────────────────────────

    def _update_orb(self, sym: str, snap: MarketSnapshot, t: time) -> None:
        if not (time(9, 15) <= t <= time(9, 30)):
            return
        if sym not in self._orb_high:
            self._orb_high[sym]  = snap.tick.ltp
            self._orb_low[sym]   = snap.tick.ltp
            self._orb_fired[sym] = False
        for c in snap.candles_1min:
            c_t = getattr(c, "ts", None)
            if c_t and time(9, 15) <= c_t.time() <= time(9, 30):
                self._orb_high[sym] = max(self._orb_high[sym], c.high)
                self._orb_low[sym]  = min(self._orb_low[sym],  c.low)

    # ── State updater (called at end of every tick) ───────────────────────────

    def _update_state(self, sym: str, ind: LiveIndicators, ltp: float) -> None:
        if ind.vwap and ind.vwap > 0:
            self._prev_above_vwap[sym] = ltp > ind.vwap
        self._prev_ltp[sym]    = ltp
        self._prev_rsi[sym]    = ind.rsi_14
        self._prev_squeeze[sym] = ind.squeeze_on

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", ind.ltp)
        ltp   = ind.ltp
        side  = "BUY" if pos.get("quantity", 0) > 0 else "SELL"
        if not entry or entry <= 0:
            return False, ""

        atr      = ind.atr_14 or entry * 0.005
        sl_dist  = max(atr * self.SL_ATR,  entry * self.SL_MIN_PCT  / 100)
        tgt_dist = max(atr * self.TGT_ATR, entry * self.TGT_MIN_PCT / 100)

        if side == "BUY":
            if ltp <= entry - sl_dist:  return True, f"SL hit ₹{ltp:.2f}"
            if ltp >= entry + tgt_dist: return True, f"Target ₹{ltp:.2f}"
            if ind.trend == "DOWN" and ind.macd_hist < 0:
                return True, "Trend reversal exit"
        else:
            if ltp >= entry + sl_dist:  return True, f"SL hit ₹{ltp:.2f}"
            if ltp <= entry - tgt_dist: return True, f"Target ₹{ltp:.2f}"
            if ind.trend == "UP" and ind.macd_hist > 0:
                return True, "Trend reversal exit"

        now = now_ist().time().replace(tzinfo=None)
        if now.hour >= 15:
            return True, "Auto square-off 3:00 PM"
        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  F&O  —  NRML, IV proxy + OI + RSI extremes + Bollinger breakout
# ═══════════════════════════════════════════════════════════════════════════════

class FnOAgent(BaseAgent):
    """
    Never-miss NSE/NFO options agent — 7 entry patterns, all market conditions covered.

    Patterns (fire independently, best score wins each tick):
      1. EMA_CROSS    — 9/21/50 EMA full alignment (highest conviction trend)
      2. TREND_PULL   — Pullback into 48-58 RSI zone in a strong EMA trend
      3. ORB          — Opening range breakout (9:15-9:30 range, execute 9:30-10:00)
      4. VWAP_RECLAIM — Price crosses above/below VWAP with volume
      5. BB_SQUEEZE   — Bollinger Band squeeze expanding → volatility breakout
      6. RSI_EXTREME  — RSI > 72 / < 28 with MACD confirmation (momentum)
      7. SURGE        — Large candle body (>0.4%) + heavy volume (>1.8×)

    Context bonuses (added to every pattern score):
      IV rank, options flow, GEX regime, volume, MACD, skew

    Sizing tiers:   score 4 → 0.25×  |  5 → 0.5×  |  6-7 → 0.75×  |  8+ → 1.0×
    Cooldown:       120s per symbol per direction (CE and PE tracked independently)
    IV gate:        hard block if IV rank > 72% (never buy expensive premium)
    """
    name    = "fno"
    product = "NRML"
    min_candles_1min = 10

    LOT_SIZES: dict = {"NIFTY": 75, "BANKNIFTY": 15, "MIDCPNIFTY": 75,
                       "FINNIFTY": 40, "SENSEX": 10}
    MIN_SCORE    = 4        # minimum score to fire at 0.25× size
    MAX_IV_BUY   = 72       # hard block above this IV rank
    COOL_S       = 120      # 2-min per symbol per direction

    # ── Per-symbol state ──────────────────────────────────────────────────────
    _orb_high:        dict = {}   # sym → float (ORB high built 9:15-9:30)
    _orb_low:         dict = {}   # sym → float
    _orb_fired:       dict = {}   # sym → bool  (prevent ORB retrigger)
    _last_candle_ts:  dict = {}   # sym → candle ts (SURGE dedup)
    _prev_above_vwap: dict = {}   # sym → bool (VWAP cross state)
    _prev_bb_width:   dict = {}   # sym → float (squeeze detection)
    _prev_ltp:        dict = {}   # sym → float (generic prev price)
    _prev_rsi:        dict = {}   # sym → float (pullback detection)
    _cool_ts:         dict = {}   # sym → {"CE": datetime, "PE": datetime}

    # ── Main entry loop ───────────────────────────────────────────────────────

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        ind = snap.indicators
        sym = snap.symbol
        ltp = snap.tick.ltp
        now = now_ist()
        t   = now.time().replace(tzinfo=None)

        # Exit-only window
        if time(14, 50) <= t:
            return "HOLD", None

        # Cached intelligence (sync — zero latency)
        from options_intelligence import get_cached
        import iv_surface, gamma_scalp, options_flow

        opts = get_cached(sym)
        surf = iv_surface.get_surface(sym)
        gex  = gamma_scalp.get_cached_gex(sym)
        flow = options_flow.get_cached_flow(sym)

        iv_rank = float(opts.get("iv_rank", 50.0)) if opts else 50.0
        atm_iv  = float(opts.get("atm_iv",  25.0)) if opts else 25.0

        # Hard IV gate — never buy expensive premium
        if iv_rank > self.MAX_IV_BUY:
            self._update_state(sym, ind, ltp)
            return "HOLD", None

        # Update ORB range (9:15-9:30 window)
        self._update_orb(sym, snap, t)

        # Run all 7 patterns — collect the best signal
        best_score, best_opt, best_pattern = -1, "", ""
        patterns = [
            self._pat_ema_cross,
            self._pat_trend_pull,
            self._pat_orb,
            self._pat_vwap_reclaim,
            self._pat_bb_squeeze,
            self._pat_rsi_extreme,
            self._pat_surge,
        ]
        for pat_fn in patterns:
            try:
                opt_type, base, pname = pat_fn(sym, snap, ind, ltp, t)
            except Exception:
                continue
            if not opt_type:
                continue
            # Add context bonuses
            total = base + self._ctx_bonus(opt_type, ind, ltp, iv_rank, surf, gex, flow)
            if total > best_score:
                best_score, best_opt, best_pattern = total, opt_type, pname

        if best_score < self.MIN_SCORE:
            self._update_state(sym, ind, ltp)
            return "HOLD", None

        # Per-direction cooldown
        cools = self._cool_ts.setdefault(sym, {})
        last  = cools.get(best_opt)
        if last and (now - last).total_seconds() < self.COOL_S:
            self._update_state(sym, ind, ltp)
            return "HOLD", None
        cools[best_opt] = now

        # SL / TGT from IV regime
        sl_pct, tgt_pct = self._iv_sl_tgt(iv_rank)

        # Size factor: 4 tiers
        sf = (1.0  if best_score >= 8 else
              0.75 if best_score >= 6 else
              0.5  if best_score >= 5 else 0.25)

        # Strike and NFO symbol
        strike  = self._pick_strike(ltp, best_opt, atm_iv)
        opt_sym = self._nfo_symbol(sym, strike, best_opt)
        lot_sz  = self.LOT_SIZES.get(sym, 1)

        self._update_state(sym, ind, ltp)
        return "BUY", {
            "exchange":          "NFO",
            "option_symbol":     opt_sym,
            "option_type":       best_opt,
            "strike":            strike,
            "lot_size":          lot_sz,
            "stop_loss_pct":     sl_pct,
            "target_pct":        tgt_pct,
            "iv_rank":           round(iv_rank, 1),
            "atm_iv":            round(atm_iv, 2),
            "score":             best_score,
            "pattern":           best_pattern,
            "_gate_size_factor": sf,
            "trigger": (
                f"FNO-{best_opt} [{best_pattern}] score={best_score}/14 "
                f"IVr={iv_rank:.0f}% sf={sf} rsi={ind.rsi_14:.0f} "
                f"trend={ind.trend}"
            ),
        }

    # ── Pattern 1: EMA_CROSS — full 9/21/50 alignment ────────────────────────

    def _pat_ema_cross(self, sym, snap, ind, ltp, t):
        ema_bull = ind.ema9 > ind.ema21 > 0 and ind.ema21 > ind.ema50 > 0
        ema_bear = ind.ema9 < ind.ema21 > 0 and ind.ema21 < ind.ema50 > 0
        if ema_bull and ind.rsi_14 > 50:
            return "CE", 5, "EMA_CROSS"
        if ema_bear and ind.rsi_14 < 50:
            return "PE", 5, "EMA_CROSS"
        return "", 0, ""

    # ── Pattern 2: TREND_PULL — pullback entry in strong trend ───────────────

    def _pat_trend_pull(self, sym, snap, ind, ltp, t):
        prev_rsi = self._prev_rsi.get(sym, ind.rsi_14)
        ema_bull  = ind.ema9 > ind.ema21 > 0 and ind.ema21 > ind.ema50 > 0
        ema_bear  = ind.ema9 < ind.ema21 > 0 and ind.ema21 < ind.ema50 > 0
        # Pullback: was extended (>62), now cooled to 48-60 → re-entry
        if ema_bull and prev_rsi > 60 and 48 <= ind.rsi_14 <= 60:
            return "CE", 5, "TREND_PULL"
        # Pullback: was oversold (<38), now recovered to 40-52 → re-short
        if ema_bear and prev_rsi < 40 and 40 <= ind.rsi_14 <= 52:
            return "PE", 5, "TREND_PULL"
        return "", 0, ""

    # ── Pattern 3: ORB — opening range breakout ───────────────────────────────

    def _pat_orb(self, sym, snap, ind, ltp, t):
        if not (time(9, 30) <= t <= time(10, 0)):
            return "", 0, ""
        orb_h = self._orb_high.get(sym)
        orb_l = self._orb_low.get(sym)
        if not (orb_h and orb_l and orb_h > orb_l):
            return "", 0, ""
        if self._orb_fired.get(sym):
            return "", 0, ""
        prev = self._prev_ltp.get(sym, ltp)
        # Break above ORB high
        if prev <= orb_h and ltp > orb_h * 1.001:
            self._orb_fired[sym] = True
            return "CE", 5, "ORB"
        # Break below ORB low
        if prev >= orb_l and ltp < orb_l * 0.999:
            self._orb_fired[sym] = True
            return "PE", 5, "ORB"
        return "", 0, ""

    # ── Pattern 4: VWAP_RECLAIM — cross above/below VWAP with volume ─────────

    def _pat_vwap_reclaim(self, sym, snap, ind, ltp, t):
        if not ind.vwap or ind.vwap <= 0:
            return "", 0, ""
        was_above = self._prev_above_vwap.get(sym, ltp >= ind.vwap)
        now_above = ltp > ind.vwap
        if was_above == now_above:
            return "", 0, ""
        if ind.volume_ratio < 1.2:
            return "", 0, ""
        if now_above:   # crossed above → CE
            return "CE", 3, "VWAP_RECLAIM"
        return "PE", 3, "VWAP_RECLAIM"    # crossed below → PE

    # ── Pattern 5: BB_SQUEEZE — Bollinger squeeze breakout ───────────────────

    def _pat_bb_squeeze(self, sym, snap, ind, ltp, t):
        if not (ind.bb_upper and ind.bb_lower and ind.bb_mid and ind.bb_mid > 0):
            return "", 0, ""
        bw = (ind.bb_upper - ind.bb_lower) / ind.bb_mid * 100
        prev_bw = self._prev_bb_width.get(sym, bw)
        # Squeeze was tight and is now expanding
        if prev_bw < 1.8 and bw > prev_bw * 1.15:
            direction = "CE" if ltp > ind.bb_mid else "PE"
            return direction, 3, "BB_SQUEEZE"
        return "", 0, ""

    # ── Pattern 6: RSI_EXTREME — extreme RSI momentum continuation ────────────

    def _pat_rsi_extreme(self, sym, snap, ind, ltp, t):
        # Overbought with positive MACD → strong upward momentum, buy CE
        if ind.rsi_14 > 72 and ind.macd_hist > 0 and ind.volume_ratio > 1.3:
            return "CE", 3, "RSI_EXTREME"
        # Oversold with negative MACD → strong downward momentum, buy PE
        if ind.rsi_14 < 28 and ind.macd_hist < 0 and ind.volume_ratio > 1.3:
            return "PE", 3, "RSI_EXTREME"
        return "", 0, ""

    # ── Pattern 7: SURGE — large candle body + volume ─────────────────────────

    def _pat_surge(self, sym, snap, ind, ltp, t):
        if len(snap.candles_1min) < 2:
            return "", 0, ""
        last_c = snap.candles_1min[-1]
        c_ts   = getattr(last_c, "ts", None)
        if not c_ts or c_ts == self._last_candle_ts.get(sym):
            return "", 0, ""
        if last_c.open <= 0:
            return "", 0, ""
        body_pct = abs(last_c.close - last_c.open) / last_c.open
        if body_pct > 0.004 and ind.volume_ratio > 1.8:
            self._last_candle_ts[sym] = c_ts
            direction = "CE" if last_c.close > last_c.open else "PE"
            return direction, 3, "SURGE"
        return "", 0, ""

    # ── Context bonus (+0 to +6 points added to every pattern) ───────────────

    def _ctx_bonus(self, opt_type, ind, ltp, iv_rank, surf, gex, flow) -> int:
        b = 0
        is_call = (opt_type == "CE")

        # IV rank (0-2)
        if   iv_rank <= 28: b += 2
        elif iv_rank <= 55: b += 1
        elif iv_rank >  65: b -= 1

        # Flow (0-1)
        if flow:
            if is_call  and flow.call_put_ratio > 1.1:   b += 1
            if not is_call and flow.call_put_ratio < 0.9: b += 1

        # GEX (0-1)
        if gex:
            if not gex.pin_risk and gex.regime != "SHORT_GAMMA": b += 1
        else:
            b += 1

        # Volume (0-1)
        if ind.volume_ratio > 1.3: b += 1

        # MACD (0-1)
        if is_call  and ind.macd_hist > 0:   b += 1
        if not is_call and ind.macd_hist < 0: b += 1

        # IV skew (0-1)
        if surf:
            if is_call  and surf.risk_reversal > -0.005: b += 1
            if not is_call and surf.put_skew > 0.005:     b += 1

        return b

    # ── ORB builder (called every tick 9:15-9:30) ─────────────────────────────

    def _update_orb(self, sym: str, snap: MarketSnapshot, t: time) -> None:
        if not (time(9, 15) <= t <= time(9, 30)):
            return
        if sym not in self._orb_high:
            self._orb_high[sym] = snap.tick.ltp
            self._orb_low[sym]  = snap.tick.ltp
            self._orb_fired[sym] = False
        for c in snap.candles_1min:
            c_t = getattr(c, "ts", None)
            if c_t and time(9, 15) <= c_t.time() <= time(9, 30):
                self._orb_high[sym] = max(self._orb_high[sym], c.high)
                self._orb_low[sym]  = min(self._orb_low[sym],  c.low)

    # ── State updater (called at end of every tick) ───────────────────────────

    def _update_state(self, sym: str, ind: LiveIndicators, ltp: float) -> None:
        if ind.vwap and ind.vwap > 0:
            self._prev_above_vwap[sym] = ltp > ind.vwap
        if ind.bb_upper and ind.bb_lower and ind.bb_mid and ind.bb_mid > 0:
            self._prev_bb_width[sym] = (ind.bb_upper - ind.bb_lower) / ind.bb_mid * 100
        self._prev_ltp[sym] = ltp
        self._prev_rsi[sym] = ind.rsi_14

    # ── IV-adaptive SL / TGT ─────────────────────────────────────────────────

    def _iv_sl_tgt(self, iv_rank: float) -> tuple[float, float]:
        if   iv_rank < 25: return 35.0, 100.0   # cheap vol → vol expansion expected
        elif iv_rank < 50: return 30.0,  65.0
        elif iv_rank < 65: return 25.0,  48.0
        else:              return 20.0,  35.0

    # ── Strike selection (delta ~0.40 proxy) ─────────────────────────────────

    def _pick_strike(self, spot: float, opt_type: str, atm_iv: float) -> int:
        import math
        step = 100 if spot > 30000 else 50
        iv   = max((atm_iv / 100.0) if atm_iv > 1.0 else atm_iv, 0.12)
        T    = 7.0 / 365.0
        offset = 0.25 * spot * iv * math.sqrt(T)
        raw    = (spot + offset) if opt_type == "CE" else (spot - offset)
        return max(int(round(raw / step) * step), step)

    # ── NFO symbol builder ────────────────────────────────────────────────────

    def _nfo_symbol(self, underlying: str, strike: int, opt_type: str) -> str:
        from datetime import date, timedelta
        today    = date.today()
        target   = 2 if underlying in ("BANKNIFTY", "MIDCPNIFTY") else 3
        expiry   = today + timedelta(days=1)
        while expiry.weekday() != target:
            expiry += timedelta(days=1)
        if (expiry + timedelta(days=7)).month != expiry.month:
            return f"{underlying}{expiry.strftime('%y')}{expiry.strftime('%b').upper()}{strike}{opt_type}"
        return f"{underlying}{expiry.strftime('%y%m%d')}{strike}{opt_type}"

    # ── Option-aware _try_enter override ─────────────────────────────────────

    async def _try_enter(self, snap: MarketSnapshot, action: str, signal: dict) -> None:
        import math
        from agents.base_agent import send_telegram
        from kite_client import kite_client
        from risk_manager import risk_manager
        from order_guard import order_guard
        from trailing_sl_engine import trailing_sl_engine
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector
        from config import settings

        underlying = snap.symbol
        opt_sym    = signal.get("option_symbol", underlying)
        exch       = signal.get("exchange", "NFO")
        lot_size   = signal.get("lot_size", 1)
        iv_rank    = signal.get("iv_rank", 50.0)
        atm_iv     = signal.get("atm_iv", 25.0)
        sf         = signal.pop("_gate_size_factor", 1.0)

        # BS-approximate ATM premium for sizing
        S         = snap.tick.ltp
        iv        = max((atm_iv / 100.0) if atm_iv > 1.0 else atm_iv, 0.10)
        opt_price = max(round(S * iv * math.sqrt(7.0 / 365.0) / math.sqrt(2 * math.pi), 2), 5.0)

        qty = lot_size
        if settings.use_kelly_sizing and sf < 1.0:
            qty = max(lot_size, int(lot_size * sf))

        allowed, _ = order_guard.can_place(underlying, self.name, action)
        if not allowed:
            return
        if order_guard.is_symbol_active_anywhere(underlying):
            return
        allowed, _ = risk_manager.check_before_order(opt_sym, qty, opt_price, action)
        if not allowed:
            return

        sebi_ok, _aid, sebi_reason = sebi_compliance.pre_order_check(
            strategy=self.name, symbol=opt_sym, exchange=exch,
            transaction_type=action, quantity=qty,
            order_type="MARKET", price_at_signal=opt_price,
            signal_source=f"agent_{self.name}",
            regime=regime_detector.current_regime.value
                   if regime_detector.current_regime else "UNKNOWN",
        )
        if not sebi_ok:
            from loguru import logger
            logger.warning("[fno] SEBI blocked {} {}: {}", action, opt_sym, sebi_reason)
            return

        order_id = kite_client.place_order(
            tradingsymbol=opt_sym, exchange=exch,
            transaction_type=action, quantity=qty,
            order_type="MARKET", product=self.product,
            tag="Agent-fno",
        )
        sebi_compliance.record_order_id(self.name, opt_sym, order_id)
        order_guard.register_order(underlying, self.name, action, order_id)
        risk_manager.position_opened()
        self.state.trades_today  += 1
        self.state.signals_fired += 1
        self.state.last_signal    = signal

        trailing_sl_engine.register(
            symbol=underlying, strategy=self.name, side=action,
            entry_price=opt_price, quantity=qty, order_id=order_id,
            atr=snap.indicators.atr_14,
        )

        sl_pct  = signal.get("stop_loss_pct", 30)
        tgt_pct = signal.get("target_pct", 65)
        sl_px   = round(opt_price * (1 - sl_pct / 100), 2)
        tgt_px  = round(opt_price * (1 + tgt_pct / 100), 2)

        kite_client.place_order(
            tradingsymbol=opt_sym, exchange=exch,
            transaction_type="SELL", quantity=qty,
            order_type="SL-M", product=self.product,
            trigger_price=sl_px, tag="Agent-fno-SL",
        )

        await send_telegram(
            f"<b>[FNO]</b> {action} {opt_sym} ≈₹{opt_price:.1f}\n"
            f"Pattern: {signal.get('pattern')} | Score: {signal.get('score')}/14\n"
            f"{signal.get('option_type')} {signal.get('strike')} | IVr={iv_rank:.0f}% sf={sf}\n"
            f"SL: ₹{sl_px:.1f} | TGT: ₹{tgt_px:.1f} | Ord: {order_id}"
        )

    # ── Exit conditions ───────────────────────────────────────────────────────

    def should_exit_position(self, pos: dict, ind: LiveIndicators) -> tuple[bool, str]:
        entry = pos.get("average_price", 0.0)
        ltp   = ind.ltp
        if not entry or entry <= 0:
            return False, ""

        chg = (ltp - entry) / entry * 100
        qty = pos.get("quantity", 0)

        # Near-zero protection
        if ltp < entry * 0.10:
            return True, f"Option near-zero ₹{ltp:.1f} ({chg:.0f}%)"

        # Hard stop 30%
        if chg <= -30:
            return True, f"Option SL -30% ₹{ltp:.1f}"

        # Progressive profit exits
        if chg >= 100:
            return True, f"Option +100% ₹{ltp:.1f}"
        if chg >= 60 and ind.rsi_14 > 73:
            return True, f"Option +60% + overbought RSI={ind.rsi_14:.0f}"
        if chg >= 50 and ind.momentum in ("WEAK_UP", "NEUTRAL", "WEAK_DOWN"):
            return True, f"Option +50% momentum fading"

        # Theta decay protection: direction lost, RSI neutral
        if 44 < ind.rsi_14 < 56 and ind.momentum == "NEUTRAL":
            return True, "RSI+momentum neutral — exit before theta decay"

        # Trend reversal while not deeply profitable
        if qty > 0 and ind.trend == "DOWN" and ind.ema9 < ind.ema21 and chg < 30:
            return True, "Trend reversed DOWN — exit call"
        if qty < 0 and ind.trend == "UP" and ind.ema9 > ind.ema21 and chg < 30:
            return True, "Trend reversed UP — exit put"

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
    Maximum-opportunity scalper — 5 entry patterns, 3-tier confidence sizing.

    Patterns detected every tick:
      1. EMA9_CROSS    — LTP micro-cross over/under EMA9
      2. EMA921_CROSS  — EMA9 crosses EMA21 (stronger, less frequent)
      3. VWAP_BOUNCE   — price touches VWAP then reverses with volume
      4. SURGE         — explosive candle ≥0.3% body + 2× volume
      5. ORB           — opening-range breakout (09:30-09:45 execution window)

    Score 8 confirmation factors → adaptive size:
      3-4/8 = 0.5×  |  5-6/8 = 0.75×  |  7-8/8 = 1.0×
    Claude gate further refines; can still veto entirely.

    Hard guards (cannot be overridden):
      spread, dead-market, level wall, loss-streak cooldown, 90s deduplication.
    """
    name    = "scalping"
    product = "MIS"
    min_candles_1min = 10

    SL_ATR  = 0.6
    TGT_ATR = 1.2
    SL_PCT  = 0.25
    TGT_PCT = 0.50

    MIN_SCORE = 3     # minimum to fire; Claude gate handles further filtering

    # Per-symbol rolling state
    _prev_ema9:       dict[str, float]    = {}
    _prev_ema21:      dict[str, float]    = {}
    _prev_ltp:        dict[str, float]    = {}
    _prev_near_vwap:  dict[str, bool]     = {}
    _prev_st_dir:     dict[str, str]      = {}   # Supertrend direction last tick
    _orb_high:        dict[str, float]    = {}
    _orb_low:         dict[str, float]    = {}
    _last_candle_ts:  dict[str, object]   = {}   # last candle that triggered SURGE
    _last_signal_ts:  dict[str, datetime] = {}   # deduplication timestamp
    _last_signal_dir: dict[str, str]      = {}   # deduplication direction
    _loss_streak:     dict[str, int]      = {}
    _cooldown_until:  dict[str, datetime] = {}

    # ── Entry ─────────────────────────────────────────────────────────────────

    def evaluate_tick(self, snap: MarketSnapshot) -> tuple[str, Optional[dict]]:
        sym = snap.symbol
        ind = snap.indicators
        ltp = snap.tick.ltp
        now = datetime.now()
        t   = now.time()

        if not ind.ema9:
            return "HOLD", None

        # ── Hard guard 1: loss-streak cooldown ──────────────────────────────
        cd = self._cooldown_until.get(sym)
        if cd and now < cd:
            return "HOLD", None

        # ── Hard guard 2: spread (0.05% — slightly wider than before) ───────
        spread = snap.tick.ask - snap.tick.bid
        if spread > ltp * 0.0005:
            return "HOLD", None

        # ── Hard guard 3: dead market (ATR/price < 0.02%) ───────────────────
        atr = ind.atr_14 or 0.0
        if ltp > 0 and atr / ltp < 0.0002:
            return "HOLD", None

        # ── Hard guard 4: no trading last 10 min before squareoff ───────────
        if t >= time(14, 50):
            return "HOLD", None

        # ── Update rolling state ─────────────────────────────────────────────
        prev_ema9  = self._prev_ema9.get(sym, ind.ema9)
        prev_ema21 = self._prev_ema21.get(sym, ind.ema21 or ind.ema9)
        prev_ltp   = self._prev_ltp.get(sym, ltp)
        self._prev_ema9[sym]   = ind.ema9
        self._prev_ema21[sym]  = ind.ema21 or ind.ema9
        self._prev_ltp[sym]    = ltp
        self._prev_st_dir[sym] = ind.supertrend_dir

        # ── Build / update opening-range high/low ────────────────────────────
        self._update_orb(sym, snap, t)

        # ── Pattern detection (first match wins — ordered by priority) ───────
        action, pattern = self._detect_pattern(
            sym, snap, ind, ltp, t, now,
            prev_ema9, prev_ema21, prev_ltp,
        )
        if action == "HOLD":
            return "HOLD", None

        # ── Signal deduplication: same symbol+direction within 90s → skip ───
        last_ts  = self._last_signal_ts.get(sym)
        last_dir = self._last_signal_dir.get(sym)
        if last_ts and last_dir == action and (now - last_ts).total_seconds() < 90:
            return "HOLD", None

        # ── Scoring for confidence/size ──────────────────────────────────────
        score, reasons = self._score_setup(snap, ind, ltp, action)
        if score < self.MIN_SCORE:
            return "HOLD", None

        # ── Level proximity guard ─────────────────────────────────────────────
        if not self._level_ok(sym, ltp, action):
            return "HOLD", None

        # ── Adaptive size from score ──────────────────────────────────────────
        sf = 0.5 if score <= 4 else (0.75 if score <= 6 else 1.0)

        # ── ATR-based SL & target ────────────────────────────────────────────
        sl_dist  = max(atr * self.SL_ATR,  ltp * self.SL_PCT  / 100)
        tgt_dist = max(atr * self.TGT_ATR, ltp * self.TGT_PCT / 100)

        # ── Record signal timestamp for dedup ────────────────────────────────
        self._last_signal_ts[sym]  = now
        self._last_signal_dir[sym] = action

        if action == "BUY":
            sl  = round(ltp - sl_dist, 2)
            tgt = round(ltp + tgt_dist, 2)
        else:
            sl  = round(ltp + sl_dist, 2)
            tgt = round(ltp - tgt_dist, 2)

        return action, {
            "symbol":            sym,
            "exchange":          "NSE",
            "side":              action,
            "price":             ltp,
            "stop_loss":         sl,
            "target":            tgt,
            "stop_loss_pct":     round(sl_dist  / ltp * 100, 3),
            "target_pct":        round(tgt_dist / ltp * 100, 3),
            "product":           self.product,
            "_gate_size_factor": sf,
            "trigger": f"{pattern} score={score}/8 sf={sf} {' '.join(reasons[:4])}",
        }

    # ── Pattern detection ─────────────────────────────────────────────────────

    def _detect_pattern(
        self,
        sym: str, snap: MarketSnapshot, ind: LiveIndicators,
        ltp: float, t: time, now: datetime,
        prev_ema9: float, prev_ema21: float, prev_ltp: float,
    ) -> tuple[str, str]:
        """Return (action, pattern_name) or ('HOLD', '')."""

        # Pattern 1: EMA9 micro-cross (fastest, tick-resolution)
        if prev_ltp < prev_ema9 and ltp > ind.ema9:
            return "BUY",  "EMA9X"
        if prev_ltp > prev_ema9 and ltp < ind.ema9:
            return "SELL", "EMA9X"

        # Pattern 2: EMA9/EMA21 cross (higher conviction)
        if ind.ema21 and ind.ema21 > 0:
            prev_diff = prev_ema9 - prev_ema21
            curr_diff = ind.ema9  - ind.ema21
            if prev_diff <= 0 < curr_diff:
                return "BUY",  "EMA921X"
            if prev_diff >= 0 > curr_diff:
                return "SELL", "EMA921X"

        # Pattern 3: VWAP bounce (price touches VWAP band then reverses)
        if ind.vwap and ind.vwap > 0:
            was_near = self._prev_near_vwap.get(sym, False)
            near_now = abs(ltp - ind.vwap) / ind.vwap < 0.0008
            self._prev_near_vwap[sym] = near_now
            if was_near and not near_now and ind.volume_ratio >= 1.3:
                if ltp > ind.vwap:
                    return "BUY",  "VWAP_BOUNCE"
                return "SELL", "VWAP_BOUNCE"

        # Pattern 4: Momentum surge (explosive candle ≥0.3% body + 2× volume)
        if len(snap.candles_1min) >= 2:
            last_c = snap.candles_1min[-1]
            c_ts   = getattr(last_c, "ts", None)
            if c_ts and c_ts != self._last_candle_ts.get(sym):
                body_pct = (
                    abs(last_c.close - last_c.open) / last_c.open
                    if last_c.open > 0 else 0.0
                )
                if body_pct > 0.003 and ind.volume_ratio > 2.0:
                    self._last_candle_ts[sym] = c_ts
                    if last_c.close > last_c.open:
                        return "BUY",  "SURGE"
                    return "SELL", "SURGE"

        # Pattern 5: Opening range breakout (execute 09:30-09:45)
        if time(9, 30) <= t <= time(9, 45):
            orb_h = self._orb_high.get(sym)
            orb_l = self._orb_low.get(sym)
            if orb_h and orb_l and orb_h > orb_l:
                breakout_up   = ltp > orb_h * 1.001 and prev_ltp <= orb_h * 1.001
                breakout_down = ltp < orb_l * 0.999 and prev_ltp >= orb_l * 0.999
                if breakout_up:
                    return "BUY",  "ORB"
                if breakout_down:
                    return "SELL", "ORB"

        # Pattern 6: Supertrend flip — direction changed this tick
        prev_st_dir = self._prev_st_dir.get(sym, ind.supertrend_dir)
        curr_st_dir = ind.supertrend_dir
        if curr_st_dir != prev_st_dir and ind.volume_ratio >= 1.2:
            if curr_st_dir == "UP":
                return "BUY",  "SUPERTREND_FLIP"
            if curr_st_dir == "DOWN":
                return "SELL", "SUPERTREND_FLIP"

        return "HOLD", ""

    # ── ORB builder ───────────────────────────────────────────────────────────

    def _update_orb(self, sym: str, snap: MarketSnapshot, t: time) -> None:
        """Track the 09:15-09:30 opening range high and low from 1-min candles."""
        if not (time(9, 15) <= t <= time(9, 30)):
            return
        orb_candles = [
            c for c in snap.candles_1min
            if hasattr(c, "ts") and time(9, 15) <= c.ts.time() <= time(9, 30)
        ]
        if orb_candles:
            self._orb_high[sym] = max(c.high for c in orb_candles)
            self._orb_low[sym]  = min(c.low  for c in orb_candles)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_setup(
        self, snap: MarketSnapshot, ind: LiveIndicators, ltp: float, action: str
    ) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        is_buy = action == "BUY"

        # 1. VWAP alignment
        if ind.vwap and ind.vwap > 0:
            if (is_buy and ltp > ind.vwap) or (not is_buy and ltp < ind.vwap):
                score += 1; reasons.append("VWAP✓")

        # 2. RSI-7 in tradeable zone (widened: 44-76 / 24-56)
        rsi = ind.rsi_7
        if is_buy and 44 < rsi < 76:
            score += 1; reasons.append(f"RSI{rsi:.0f}")
        elif not is_buy and 24 < rsi < 56:
            score += 1; reasons.append(f"RSI{rsi:.0f}")

        # 3. Volume confirmation (≥1.2× for partial, ≥1.5× for full)
        if ind.volume_ratio >= 1.5:
            score += 1; reasons.append(f"VOL{ind.volume_ratio:.1f}x")
        elif ind.volume_ratio >= 1.2:
            score += 1

        # 4. ADX ≥20 (some trend present)
        if ind.adx_14 >= 20:
            score += 1; reasons.append(f"ADX{ind.adx_14:.0f}")

        # 5. MACD histogram direction
        if (is_buy and ind.macd_hist > 0) or (not is_buy and ind.macd_hist < 0):
            score += 1; reasons.append("MACD✓")

        # 6. Candle microstructure (≥2 of last 3 confirm direction)
        if len(snap.candles_1min) >= 3:
            last3 = snap.candles_1min[-3:]
            if is_buy:
                g = sum(1 for c in last3 if c.close >= c.open)
                if g >= 2: score += 1; reasons.append(f"{g}G")
            else:
                r = sum(1 for c in last3 if c.close <= c.open)
                if r >= 2: score += 1; reasons.append(f"{r}R")

        # 7. Price velocity (last 5 closes trending)
        if len(snap.candles_1min) >= 5:
            closes = [c.close for c in snap.candles_1min[-5:]]
            if (is_buy and closes[-1] > closes[0]) or (not is_buy and closes[-1] < closes[0]):
                score += 1; reasons.append("VEL✓")

        # 8. EMA21 macro-trend alignment
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
            threshold = ltp * 0.0015
            keys_r = ("r1", "r2", "pdh", "weekly_high", "vwap_upper_1")
            keys_s = ("s1", "s2", "pdl", "weekly_low",  "vwap_lower_1")
            if side == "BUY":
                for k in keys_r:
                    v = lvls.get(k)
                    if v and 0 < v - ltp < threshold:
                        return False
            else:
                for k in keys_s:
                    v = lvls.get(k)
                    if v and 0 < ltp - v < threshold:
                        return False
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
                self._record_outcome(sym, False); return True, f"Scalp SL ₹{ltp:.2f}"
            if ltp >= tgt:
                self._record_outcome(sym, True);  return True, f"Scalp target ₹{ltp:.2f}"
            if ind.momentum == "STRONG_DOWN" and ind.macd_hist < 0:
                self._record_outcome(sym, ltp > entry); return True, "Strong reversal"
            if ind.vwap and ltp < ind.vwap * 0.9985:
                self._record_outcome(sym, ltp > entry); return True, "VWAP breakdown"
        else:
            sl, tgt = entry + sl_dist, entry - tgt_dist
            if ltp >= sl:
                self._record_outcome(sym, False); return True, f"Scalp SL ₹{ltp:.2f}"
            if ltp <= tgt:
                self._record_outcome(sym, True);  return True, f"Scalp target ₹{ltp:.2f}"
            if ind.momentum == "STRONG_UP" and ind.macd_hist > 0:
                self._record_outcome(sym, ltp < entry); return True, "Strong reversal"
            if ind.vwap and ltp > ind.vwap * 1.0015:
                self._record_outcome(sym, ltp < entry); return True, "VWAP breakout"

        if datetime.now().time() >= time(14, 55):
            return True, "Auto square-off 2:55 PM"

        return False, ""

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