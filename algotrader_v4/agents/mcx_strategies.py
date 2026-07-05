"""
agents/mcx_strategies.py — 80 MCX signal strategies (20 per trading-type agent).

Each MCX agent runs a registry of 20 named strategies on every tick; the
best-scoring one wins (see agents/mcx_agents.py). Strategies are small pure
functions over an `SCtx` (the tick's indicators plus the symbol's rolling
previous-tick state), returning `(action, score)` or `None`.

The 20 per agent are tuned to that agent's style and reflect the way MCX
commodity futures actually trade today:

  intraday  → trend-continuation, momentum & intraday breakout
  scalping  → fast micro-momentum & mean-reversion around VWAP/bands
  positional→ multi-day trend-following, EMA/Supertrend/HMA regimes
  options   → volatility (squeeze/expansion) + directional option-buy proxies

Scores (2-5) express conviction and drive the agent's size factor.
Indicators available on `SCtx.ind` (LiveIndicators): EMA 9/21/50/200, VWAP +
2σ/3σ bands, RSI-14/7, MACD, Bollinger, ATR, volume_ratio, OBV, Supertrend,
HMA, TTM squeeze, Stochastic-RSI, Williams %R, plus trend/momentum/volatility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Callable, Optional

from tick_engine import LiveIndicators, Candle

# A strategy: (context) -> (action, score) | None ; action ∈ {"BUY","SELL"}
Signal   = Optional[tuple[str, int]]
Strategy = Callable[["SCtx"], Signal]


@dataclass
class SCtx:
    """Per-tick context: indicators + this symbol's previous-tick rolling state."""
    sym:     str
    ltp:     float
    ind:     LiveIndicators
    prev:    dict
    candles: list        # candles_1min
    t:       dtime       # IST wall-clock time

    def __post_init__(self) -> None:
        i, p, ltp = self.ind, self.prev, self.ltp
        # RSI / stoch / williams
        self.rsi   = i.rsi_14
        self.rsi7  = i.rsi_7
        self.stk   = i.stoch_rsi_k
        self.std_  = i.stoch_rsi_d
        self.pstk  = p.get("stk", self.stk)
        self.wr    = i.williams_r
        self.vol   = i.volume_ratio
        # VWAP
        self.vwap  = i.vwap
        self.above_vwap = i.vwap > 0 and ltp > i.vwap
        self.pabove_vwap = p.get("above_vwap", self.above_vwap)
        self.vwap_x_up = (not self.pabove_vwap) and self.above_vwap
        self.vwap_x_dn = self.pabove_vwap and (not self.above_vwap)
        # EMA stacks
        self.ema_bull = i.ema9 > i.ema21 > 0
        self.ema_bear = 0 < i.ema21 and i.ema9 < i.ema21
        self.ema_full_bull = i.ema9 > i.ema21 > i.ema50 > i.ema200 > 0
        self.ema_full_bear = 0 < i.ema200 and i.ema9 < i.ema21 < i.ema50 < i.ema200
        self.pema9 = p.get("ema9", i.ema9)
        self.pltp  = p.get("ltp", ltp)
        self.ema9_x_up = self.pltp <= self.pema9 and ltp > i.ema9 and i.ema9 > 0
        self.ema9_x_dn = self.pltp >= self.pema9 and ltp < i.ema9 and i.ema9 > 0
        # MACD
        self.macd_up = i.macd_hist > 0
        self.macd_dn = i.macd_hist < 0
        self.pmacd   = p.get("macd_hist", i.macd_hist)
        self.macd_x_up = self.pmacd <= 0 < i.macd_hist
        self.macd_x_dn = self.pmacd >= 0 > i.macd_hist
        self.macd_sig_up = i.macd > i.macd_signal
        # Supertrend / HMA
        self.st_up = i.supertrend_dir == "UP"
        self.st_dn = i.supertrend_dir == "DOWN"
        self.pst   = p.get("st", i.supertrend_dir)
        self.st_flip_up = self.st_up and self.pst != "UP"
        self.st_flip_dn = self.st_dn and self.pst != "DOWN"
        self.hma_up = i.hma_dir == "UP"
        self.hma_dn = i.hma_dir == "DOWN"
        self.phma   = p.get("hma", i.hma_dir)
        self.hma_flip_up = self.hma_up and self.phma != "UP"
        self.hma_flip_dn = self.hma_dn and self.phma != "DOWN"
        # Bollinger
        self.bb_break_up = i.bb_upper > 0 and ltp > i.bb_upper
        self.bb_break_dn = i.bb_lower > 0 and ltp < i.bb_lower
        self.bb_width = (i.bb_upper - i.bb_lower) if i.bb_upper > 0 else 0.0
        self.pbb_width = p.get("bb_width", self.bb_width)
        # Squeeze
        self.squeeze  = i.squeeze_on
        self.psqueeze = p.get("squeeze", i.squeeze_on)
        self.sq_fire  = (not self.squeeze) and self.psqueeze
        self.sq_mom   = i.squeeze_momentum
        # OBV / ATR
        self.obv  = i.obv
        self.pobv = p.get("obv", i.obv)
        self.obv_up = i.obv > self.pobv
        self.atr  = i.atr_14
        self.patr = p.get("atr", i.atr_14)
        self.atr_expand = i.atr_14 > self.patr > 0
        # regime strings
        self.trend_up   = i.trend == "UP"
        self.trend_dn   = i.trend == "DOWN"
        self.mom_up     = i.momentum == "UP"
        self.mom_dn     = i.momentum == "DOWN"
        self.high_vol   = i.volatility == "HIGH"
        # session structure
        self.day_open = i.day_open

    def nbar(self, n: int) -> tuple[float, float]:
        """(highest high, lowest low) over the last n 1-min candles (excl. current)."""
        cs = self.candles[-(n + 1):-1] if len(self.candles) > n else self.candles[:-1]
        if not cs:
            return self.ltp, self.ltp
        return max(c.high for c in cs), min(c.low for c in cs)


# ══════════════════════════════════════════════════════════════════════════════
# 1. INTRADAY (MIS) — trend continuation, momentum & intraday breakout
# ══════════════════════════════════════════════════════════════════════════════
def i_vwap_trend(c):
    if c.above_vwap and c.ema_bull and 48 <= c.rsi <= 72 and c.macd_up and c.vol >= 1.3:
        return "BUY", 3
    if (not c.above_vwap) and c.ema_bear and 28 <= c.rsi <= 52 and c.macd_dn and c.vol >= 1.3:
        return "SELL", 3

def i_vwap_reclaim(c):
    if c.vwap_x_up and c.vol >= 1.2: return "BUY", 3
    if c.vwap_x_dn and c.vol >= 1.2: return "SELL", 3

def i_ema_cross(c):
    if c.ema9_x_up and c.above_vwap: return "BUY", 3
    if c.ema9_x_dn and not c.above_vwap: return "SELL", 3

def i_ema_ribbon(c):
    if c.ema_full_bull and 50 <= c.rsi <= 70: return "BUY", 4
    if c.ema_full_bear and 30 <= c.rsi <= 50: return "SELL", 4

def i_ema_pullback(c):
    if c.ema_bull and c.prev.get("rsi", c.rsi) > 63 and 45 <= c.rsi <= 60: return "BUY", 4
    if c.ema_bear and c.prev.get("rsi", c.rsi) < 37 and 40 <= c.rsi <= 55: return "SELL", 4

def i_supertrend_flip(c):
    if c.st_flip_up and c.vol >= 1.2: return "BUY", 4
    if c.st_flip_dn and c.vol >= 1.2: return "SELL", 4

def i_supertrend_align(c):
    if c.st_up and c.trend_up and c.macd_up: return "BUY", 3
    if c.st_dn and c.trend_dn and c.macd_dn: return "SELL", 3

def i_macd_zero_cross(c):
    if c.macd_x_up and c.above_vwap: return "BUY", 3
    if c.macd_x_dn and not c.above_vwap: return "SELL", 3

def i_macd_signal_cross(c):
    if c.macd_sig_up and c.macd_up and c.ema_bull and c.rsi >= 50: return "BUY", 3
    if (not c.macd_sig_up) and c.macd_dn and c.ema_bear and c.rsi <= 50: return "SELL", 3

def i_rsi_momentum(c):
    if c.rsi >= 60 and c.rsi > c.prev.get("rsi", c.rsi) and c.trend_up: return "BUY", 3
    if c.rsi <= 40 and c.rsi < c.prev.get("rsi", c.rsi) and c.trend_dn: return "SELL", 3

def i_bollinger_breakout(c):
    if c.bb_break_up and c.vol >= 1.5 and c.macd_up: return "BUY", 4
    if c.bb_break_dn and c.vol >= 1.5 and c.macd_dn: return "SELL", 4

def i_squeeze_fire(c):
    if c.sq_fire and c.sq_mom > 0 and 45 <= c.rsi <= 72: return "BUY", 4
    if c.sq_fire and c.sq_mom < 0 and 28 <= c.rsi <= 55: return "SELL", 4

def i_orb_break(c):
    oh, ol = c.prev.get("orb_high"), c.prev.get("orb_low")
    if not (c.prev.get("orb_ready") and oh and ol):
        return None
    if c.pltp <= oh and c.ltp > oh * 1.0008 and c.vol >= 1.2: return "BUY", 5
    if c.pltp >= ol and c.ltp < ol * 0.9992 and c.vol >= 1.2: return "SELL", 5

def i_range_breakout(c):
    hh, ll = c.nbar(15)
    if c.pltp < hh <= c.ltp and c.vol >= 1.4: return "BUY", 3
    if c.pltp > ll >= c.ltp and c.vol >= 1.4: return "SELL", 3

def i_volume_thrust(c):
    if c.vol >= 2.0 and c.macd_up and c.above_vwap: return "BUY", 4
    if c.vol >= 2.0 and c.macd_dn and not c.above_vwap: return "SELL", 4

def i_obv_trend(c):
    if c.obv_up and c.ema_bull and c.rsi >= 52: return "BUY", 3
    if (not c.obv_up) and c.ema_bear and c.rsi <= 48: return "SELL", 3

def i_hma_flip(c):
    if c.hma_flip_up and c.above_vwap: return "BUY", 3
    if c.hma_flip_dn and not c.above_vwap: return "SELL", 3

def i_vwap_band_ride(c):
    i = c.ind
    if i.vwap_upper2 > 0 and c.vwap < c.ltp < i.vwap_upper2 and c.ema_bull and c.macd_up:
        return "BUY", 3
    if i.vwap_lower2 > 0 and i.vwap_lower2 < c.ltp < c.vwap and c.ema_bear and c.macd_dn:
        return "SELL", 3

def i_stoch_cross(c):
    if c.pstk <= 20 < c.stk and c.trend_up: return "BUY", 3
    if c.pstk >= 80 > c.stk and c.trend_dn: return "SELL", 3

def i_gap_go(c):
    if not c.day_open: return None
    if c.ltp > c.day_open * 1.003 and c.above_vwap and c.vol >= 1.3: return "BUY", 3
    if c.ltp < c.day_open * 0.997 and not c.above_vwap and c.vol >= 1.3: return "SELL", 3


INTRADAY_STRATEGIES: list[tuple[str, Strategy]] = [
    ("VWAP_TREND", i_vwap_trend), ("VWAP_RECLAIM", i_vwap_reclaim),
    ("EMA_CROSS", i_ema_cross), ("EMA_RIBBON", i_ema_ribbon),
    ("EMA_PULLBACK", i_ema_pullback), ("SUPERTREND_FLIP", i_supertrend_flip),
    ("SUPERTREND_ALIGN", i_supertrend_align), ("MACD_ZERO_CROSS", i_macd_zero_cross),
    ("MACD_SIGNAL_CROSS", i_macd_signal_cross), ("RSI_MOMENTUM", i_rsi_momentum),
    ("BOLLINGER_BREAKOUT", i_bollinger_breakout), ("SQUEEZE_FIRE", i_squeeze_fire),
    ("ORB_BREAK", i_orb_break), ("RANGE_BREAKOUT", i_range_breakout),
    ("VOLUME_THRUST", i_volume_thrust), ("OBV_TREND", i_obv_trend),
    ("HMA_FLIP", i_hma_flip), ("VWAP_BAND_RIDE", i_vwap_band_ride),
    ("STOCH_CROSS", i_stoch_cross), ("GAP_GO", i_gap_go),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCALPING (MIS) — fast micro-momentum & mean-reversion around VWAP/bands
# ══════════════════════════════════════════════════════════════════════════════
def sc_rsi7_momo(c):
    if c.rsi7 > 55 and c.macd_up and c.vol >= 1.5 and c.above_vwap: return "BUY", 3
    if c.rsi7 < 45 and c.macd_dn and c.vol >= 1.5 and not c.above_vwap: return "SELL", 3

def sc_vwap_scalp(c):
    if c.vwap_x_up and c.vol >= 1.3: return "BUY", 3
    if c.vwap_x_dn and c.vol >= 1.3: return "SELL", 3

def sc_stoch_reversal(c):
    if c.pstk < 15 and c.stk > c.pstk and c.ltp >= c.vwap: return "BUY", 3
    if c.pstk > 85 and c.stk < c.pstk and c.ltp <= c.vwap: return "SELL", 3

def sc_williams_reversal(c):
    if c.wr <= -80 and c.rsi7 > c.prev.get("rsi7", c.rsi7): return "BUY", 3
    if c.wr >= -20 and c.rsi7 < c.prev.get("rsi7", c.rsi7): return "SELL", 3

def sc_bb_fade_lower(c):
    if c.bb_break_dn and c.rsi7 < 30: return "BUY", 3   # mean revert up
def sc_bb_fade_upper(c):
    if c.bb_break_up and c.rsi7 > 70: return "SELL", 3  # mean revert down

def sc_vwap_lower2_bounce(c):
    i = c.ind
    if i.vwap_lower2 > 0 and c.ltp <= i.vwap_lower2 and c.rsi7 < 35: return "BUY", 4
def sc_vwap_upper2_fade(c):
    i = c.ind
    if i.vwap_upper2 > 0 and c.ltp >= i.vwap_upper2 and c.rsi7 > 65: return "SELL", 4

def sc_micro_macd(c):
    if c.macd_x_up and c.vol >= 1.4: return "BUY", 3
    if c.macd_x_dn and c.vol >= 1.4: return "SELL", 3

def sc_volume_spike(c):
    if c.vol >= 2.5 and c.ltp > c.pltp and c.above_vwap: return "BUY", 4
    if c.vol >= 2.5 and c.ltp < c.pltp and not c.above_vwap: return "SELL", 4

def sc_ema9_reclaim(c):
    if c.ema9_x_up and c.rsi7 >= 50: return "BUY", 3
    if c.ema9_x_dn and c.rsi7 <= 50: return "SELL", 3

def sc_supertrend_scalp(c):
    if c.st_flip_up: return "BUY", 3
    if c.st_flip_dn: return "SELL", 3

def sc_momentum_burst(c):
    if c.ind.change_pct >= 0.3 and c.vol >= 1.8 and c.macd_up: return "BUY", 3
    if c.ind.change_pct <= -0.3 and c.vol >= 1.8 and c.macd_dn: return "SELL", 3

def sc_stoch_cross_fast(c):
    if c.pstk <= c.std_ < c.stk and c.stk < 60: return "BUY", 3
    if c.pstk >= c.std_ > c.stk and c.stk > 40: return "SELL", 3

def sc_rsi7_meanrevert(c):
    if c.rsi7 < 25 and c.ltp > c.pltp: return "BUY", 3
    if c.rsi7 > 75 and c.ltp < c.pltp: return "SELL", 3

def sc_range_fade(c):
    if c.ind.day_low and c.ltp <= c.ind.day_low * 1.001 and c.rsi7 < 35: return "BUY", 3
    if c.ind.day_high and c.ltp >= c.ind.day_high * 0.999 and c.rsi7 > 65: return "SELL", 3

def sc_hma_micro(c):
    if c.hma_flip_up and c.vol >= 1.3: return "BUY", 3
    if c.hma_flip_dn and c.vol >= 1.3: return "SELL", 3

def sc_squeeze_pop(c):
    if c.sq_fire and c.sq_mom > 0: return "BUY", 3
    if c.sq_fire and c.sq_mom < 0: return "SELL", 3

def sc_obv_tick(c):
    if c.obv_up and c.rsi7 > 55 and c.above_vwap: return "BUY", 2
    if (not c.obv_up) and c.rsi7 < 45 and not c.above_vwap: return "SELL", 2

def sc_vwap_snap(c):
    i = c.ind
    if i.vwap_upper2 > 0 and c.pltp >= i.vwap_upper2 and c.ltp < i.vwap_upper2: return "SELL", 3
    if i.vwap_lower2 > 0 and c.pltp <= i.vwap_lower2 and c.ltp > i.vwap_lower2: return "BUY", 3


SCALPING_STRATEGIES: list[tuple[str, Strategy]] = [
    ("RSI7_MOMO", sc_rsi7_momo), ("VWAP_SCALP", sc_vwap_scalp),
    ("STOCH_REVERSAL", sc_stoch_reversal), ("WILLIAMS_REVERSAL", sc_williams_reversal),
    ("BB_FADE_LOWER", sc_bb_fade_lower), ("BB_FADE_UPPER", sc_bb_fade_upper),
    ("VWAP_L2_BOUNCE", sc_vwap_lower2_bounce), ("VWAP_U2_FADE", sc_vwap_upper2_fade),
    ("MICRO_MACD", sc_micro_macd), ("VOLUME_SPIKE", sc_volume_spike),
    ("EMA9_RECLAIM", sc_ema9_reclaim), ("SUPERTREND_SCALP", sc_supertrend_scalp),
    ("MOMENTUM_BURST", sc_momentum_burst), ("STOCH_CROSS_FAST", sc_stoch_cross_fast),
    ("RSI7_MEANREVERT", sc_rsi7_meanrevert), ("RANGE_FADE", sc_range_fade),
    ("HMA_MICRO", sc_hma_micro), ("SQUEEZE_POP", sc_squeeze_pop),
    ("OBV_TICK", sc_obv_tick), ("VWAP_SNAP", sc_vwap_snap),
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. POSITIONAL (NRML) — multi-day trend following, regime alignment
# ══════════════════════════════════════════════════════════════════════════════
def po_full_ribbon(c):
    if c.ema_full_bull and 50 <= c.rsi <= 70 and c.macd_up: return "BUY", 5
    if c.ema_full_bear and 30 <= c.rsi <= 50 and c.macd_dn: return "SELL", 5

def po_supertrend_trend(c):
    if c.st_up and c.ema_bull and c.rsi >= 50: return "BUY", 4
    if c.st_dn and c.ema_bear and c.rsi <= 50: return "SELL", 4

def po_hma_trend(c):
    if c.hma_up and c.ema_full_bull: return "BUY", 4
    if c.hma_dn and c.ema_full_bear: return "SELL", 4

def po_ema50_pullback(c):
    if c.ema_full_bull and c.ltp <= c.ind.ema50 * 1.004 and c.rsi >= 45: return "BUY", 4
    if c.ema_full_bear and c.ltp >= c.ind.ema50 * 0.996 and c.rsi <= 55: return "SELL", 4

def po_ema200_trend(c):
    if c.ind.ema200 > 0 and c.ltp > c.ind.ema200 and c.macd_up and c.trend_up: return "BUY", 4
    if c.ind.ema200 > 0 and c.ltp < c.ind.ema200 and c.macd_dn and c.trend_dn: return "SELL", 4

def po_macd_trend(c):
    if c.macd_x_up and c.ema_bull and c.ltp > c.vwap: return "BUY", 3
    if c.macd_x_dn and c.ema_bear and c.ltp < c.vwap: return "SELL", 3

def po_rsi_trend_zone(c):
    if 50 <= c.rsi <= 65 and c.trend_up and c.ema_bull: return "BUY", 3
    if 35 <= c.rsi <= 50 and c.trend_dn and c.ema_bear: return "SELL", 3

def po_breakout_day_high(c):
    if c.ind.day_high and c.pltp < c.ind.day_high <= c.ltp and c.ema_bull: return "BUY", 4
    if c.ind.day_low and c.pltp > c.ind.day_low >= c.ltp and c.ema_bear: return "SELL", 4

def po_donchian(c):
    hh, ll = c.nbar(40)
    if c.pltp < hh <= c.ltp and c.trend_up: return "BUY", 4
    if c.pltp > ll >= c.ltp and c.trend_dn: return "SELL", 4

def po_vwap_trend_daily(c):
    if c.above_vwap and c.ema_full_bull and c.macd_up: return "BUY", 3
    if (not c.above_vwap) and c.ema_full_bear and c.macd_dn: return "SELL", 3

def po_obv_accumulation(c):
    if c.obv_up and c.ema_full_bull and c.rsi >= 50: return "BUY", 3
    if (not c.obv_up) and c.ema_full_bear and c.rsi <= 50: return "SELL", 3

def po_supertrend_hma(c):
    if c.st_up and c.hma_up and c.rsi >= 50: return "BUY", 4
    if c.st_dn and c.hma_dn and c.rsi <= 50: return "SELL", 4

def po_momentum_string(c):
    if c.trend_up and c.mom_up and c.macd_up and c.ema_bull: return "BUY", 3
    if c.trend_dn and c.mom_dn and c.macd_dn and c.ema_bear: return "SELL", 3

def po_pullback_ema21(c):
    if c.ema_full_bull and c.ltp <= c.ind.ema21 * 1.002 and c.macd_up: return "BUY", 4
    if c.ema_full_bear and c.ltp >= c.ind.ema21 * 0.998 and c.macd_dn: return "SELL", 4

def po_bollinger_ride(c):
    if c.ind.bb_upper > 0 and c.ltp > c.ind.bb_mid and c.ema_bull and c.rsi >= 55: return "BUY", 3
    if c.ind.bb_lower > 0 and c.ltp < c.ind.bb_mid and c.ema_bear and c.rsi <= 45: return "SELL", 3

def po_volatility_expansion(c):
    if c.atr_expand and c.ema_full_bull and c.macd_up: return "BUY", 3
    if c.atr_expand and c.ema_full_bear and c.macd_dn: return "SELL", 3

def po_higher_high(c):
    hh, ll = c.nbar(30)
    if c.ltp >= hh and c.ema_bull and c.rsi >= 55: return "BUY", 3
    if c.ltp <= ll and c.ema_bear and c.rsi <= 45: return "SELL", 3

def po_macd_zero_trend(c):
    if c.macd_up and c.ema_full_bull and c.above_vwap: return "BUY", 3
    if c.macd_dn and c.ema_full_bear and not c.above_vwap: return "SELL", 3

def po_stoch_pullback(c):
    if c.ema_full_bull and c.pstk <= 25 < c.stk: return "BUY", 4
    if c.ema_full_bear and c.pstk >= 75 > c.stk: return "SELL", 4

def po_supertrend_flip_swing(c):
    if c.st_flip_up and c.ema_bull and c.rsi >= 50: return "BUY", 5
    if c.st_flip_dn and c.ema_bear and c.rsi <= 50: return "SELL", 5


POSITIONAL_STRATEGIES: list[tuple[str, Strategy]] = [
    ("FULL_RIBBON", po_full_ribbon), ("SUPERTREND_TREND", po_supertrend_trend),
    ("HMA_TREND", po_hma_trend), ("EMA50_PULLBACK", po_ema50_pullback),
    ("EMA200_TREND", po_ema200_trend), ("MACD_TREND", po_macd_trend),
    ("RSI_TREND_ZONE", po_rsi_trend_zone), ("BREAKOUT_DAY_HIGH", po_breakout_day_high),
    ("DONCHIAN", po_donchian), ("VWAP_TREND_DAILY", po_vwap_trend_daily),
    ("OBV_ACCUMULATION", po_obv_accumulation), ("SUPERTREND_HMA", po_supertrend_hma),
    ("MOMENTUM_STRING", po_momentum_string), ("PULLBACK_EMA21", po_pullback_ema21),
    ("BOLLINGER_RIDE", po_bollinger_ride), ("VOLATILITY_EXPANSION", po_volatility_expansion),
    ("HIGHER_HIGH", po_higher_high), ("MACD_ZERO_TREND", po_macd_zero_trend),
    ("STOCH_PULLBACK", po_stoch_pullback), ("SUPERTREND_FLIP_SWING", po_supertrend_flip_swing),
]


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPTIONS / SPREAD (NRML) — volatility + directional option-buy proxies
#    (BUY ≈ buy CE / long the future; SELL ≈ buy PE / short the future)
#    High realised volatility is avoided (premiums too rich).
# ══════════════════════════════════════════════════════════════════════════════
def op_squeeze_fire_ce(c):
    if c.sq_fire and c.sq_mom > 0 and 45 <= c.rsi <= 70 and not c.high_vol: return "BUY", 5
def op_squeeze_fire_pe(c):
    if c.sq_fire and c.sq_mom < 0 and 30 <= c.rsi <= 55 and not c.high_vol: return "SELL", 5

def op_bb_expansion_ce(c):
    if c.bb_width > c.pbb_width > 0 and c.ltp > c.ind.bb_mid and c.macd_up and not c.high_vol:
        return "BUY", 4
def op_bb_expansion_pe(c):
    if c.bb_width > c.pbb_width > 0 and c.ltp < c.ind.bb_mid and c.macd_dn and not c.high_vol:
        return "SELL", 4

def op_atr_expansion(c):
    if c.atr_expand and c.ema_bull and c.macd_up and not c.high_vol: return "BUY", 3
    if c.atr_expand and c.ema_bear and c.macd_dn and not c.high_vol: return "SELL", 3

def op_directional_ce(c):
    if c.ema_bull and 52 <= c.rsi <= 70 and c.macd_up and c.vol >= 1.4 and not c.high_vol:
        return "BUY", 3
def op_directional_pe(c):
    if c.ema_bear and 30 <= c.rsi <= 48 and c.macd_dn and c.vol >= 1.4 and not c.high_vol:
        return "SELL", 3

def op_low_vol_trend(c):
    if c.ind.volatility in ("LOW", "NORMAL") and c.ema_full_bull and c.macd_up: return "BUY", 3
    if c.ind.volatility in ("LOW", "NORMAL") and c.ema_full_bear and c.macd_dn: return "SELL", 3

def op_vwap_trend_opt(c):
    if c.above_vwap and c.ema_bull and c.macd_up and not c.high_vol: return "BUY", 3
    if (not c.above_vwap) and c.ema_bear and c.macd_dn and not c.high_vol: return "SELL", 3

def op_supertrend_opt(c):
    if c.st_flip_up and not c.high_vol: return "BUY", 4
    if c.st_flip_dn and not c.high_vol: return "SELL", 4

def op_macd_trend_opt(c):
    if c.macd_x_up and c.ema_bull and not c.high_vol: return "BUY", 3
    if c.macd_x_dn and c.ema_bear and not c.high_vol: return "SELL", 3

def op_rsi_breakout_opt(c):
    if c.prev.get("rsi", c.rsi) <= 60 < c.rsi and c.trend_up and not c.high_vol: return "BUY", 3
    if c.prev.get("rsi", c.rsi) >= 40 > c.rsi and c.trend_dn and not c.high_vol: return "SELL", 3

def op_bollinger_break_opt(c):
    if c.bb_break_up and c.vol >= 1.4 and not c.high_vol: return "BUY", 3
    if c.bb_break_dn and c.vol >= 1.4 and not c.high_vol: return "SELL", 3

def op_momentum_burst_opt(c):
    if c.ind.change_pct >= 0.4 and c.vol >= 1.6 and not c.high_vol: return "BUY", 3
    if c.ind.change_pct <= -0.4 and c.vol >= 1.6 and not c.high_vol: return "SELL", 3

def op_hma_opt(c):
    if c.hma_flip_up and c.ema_bull and not c.high_vol: return "BUY", 3
    if c.hma_flip_dn and c.ema_bear and not c.high_vol: return "SELL", 3

def op_stoch_opt(c):
    if c.pstk <= 20 < c.stk and c.ema_bull and not c.high_vol: return "BUY", 3
    if c.pstk >= 80 > c.stk and c.ema_bear and not c.high_vol: return "SELL", 3

def op_range_break_opt(c):
    hh, ll = c.nbar(20)
    if c.pltp < hh <= c.ltp and c.vol >= 1.4 and not c.high_vol: return "BUY", 3
    if c.pltp > ll >= c.ltp and c.vol >= 1.4 and not c.high_vol: return "SELL", 3

def op_volume_thrust_opt(c):
    if c.vol >= 2.0 and c.macd_up and c.above_vwap and not c.high_vol: return "BUY", 4
    if c.vol >= 2.0 and c.macd_dn and not c.above_vwap and not c.high_vol: return "SELL", 4

def op_trend_align_opt(c):
    if c.trend_up and c.mom_up and c.st_up and not c.high_vol: return "BUY", 4
    if c.trend_dn and c.mom_dn and c.st_dn and not c.high_vol: return "SELL", 4

def op_pullback_opt(c):
    if c.ema_bull and c.prev.get("rsi", c.rsi) > 62 and 46 <= c.rsi <= 58 and not c.high_vol:
        return "BUY", 4
    if c.ema_bear and c.prev.get("rsi", c.rsi) < 38 and 42 <= c.rsi <= 54 and not c.high_vol:
        return "SELL", 4


OPTIONS_STRATEGIES: list[tuple[str, Strategy]] = [
    ("SQUEEZE_FIRE_CE", op_squeeze_fire_ce), ("SQUEEZE_FIRE_PE", op_squeeze_fire_pe),
    ("BB_EXPANSION_CE", op_bb_expansion_ce), ("BB_EXPANSION_PE", op_bb_expansion_pe),
    ("ATR_EXPANSION", op_atr_expansion), ("DIRECTIONAL_CE", op_directional_ce),
    ("DIRECTIONAL_PE", op_directional_pe), ("LOW_VOL_TREND", op_low_vol_trend),
    ("VWAP_TREND_OPT", op_vwap_trend_opt), ("SUPERTREND_OPT", op_supertrend_opt),
    ("MACD_TREND_OPT", op_macd_trend_opt), ("RSI_BREAKOUT_OPT", op_rsi_breakout_opt),
    ("BOLLINGER_BREAK_OPT", op_bollinger_break_opt), ("MOMENTUM_BURST_OPT", op_momentum_burst_opt),
    ("HMA_OPT", op_hma_opt), ("STOCH_OPT", op_stoch_opt),
    ("RANGE_BREAK_OPT", op_range_break_opt), ("VOLUME_THRUST_OPT", op_volume_thrust_opt),
    ("TREND_ALIGN_OPT", op_trend_align_opt), ("PULLBACK_OPT", op_pullback_opt),
]


# Registry keyed by agent name
STRATEGY_REGISTRY: dict[str, list[tuple[str, Strategy]]] = {
    "intraday": INTRADAY_STRATEGIES,
    "scalping": SCALPING_STRATEGIES,
    "swing":    POSITIONAL_STRATEGIES,
    "fno":      OPTIONS_STRATEGIES,
}


def update_prev(prev: dict, c: SCtx) -> None:
    """Persist the fields the next tick's crossover/slope detection needs."""
    i = c.ind
    prev.update({
        "ltp": c.ltp, "rsi": i.rsi_14, "rsi7": i.rsi_7, "stk": i.stoch_rsi_k,
        "above_vwap": c.above_vwap, "ema9": i.ema9, "macd_hist": i.macd_hist,
        "st": i.supertrend_dir, "hma": i.hma_dir, "obv": i.obv, "atr": i.atr_14,
        "squeeze": i.squeeze_on, "bb_width": c.bb_width,
    })
