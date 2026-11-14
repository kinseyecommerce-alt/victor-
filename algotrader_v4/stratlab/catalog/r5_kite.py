"""
Round 5 — strategies published in the Zerodha / Kite ecosystem (Zerodha Varsity, Streak examples, AlgoTest Pine
templates, Tradetron blog builds, AlgoBulls pyalgostrategypool), tested EXACTLY AS PUBLISHED, with no tuning.

Pre-registration: stratlab/prereg/kite_ecosystem_v1.json (byte-identical to <scratchpad>/PREREGISTERED_KITE.json),
written before any run. It lists every id below with its source URL, the published rule, the ONE plain default
used for each field the source leaves unspecified, and the eras it runs on. Changing anything here makes it a new,
unregistered strategy.

Conventions shared by all entries (stated in the pre-registration):
  * Indicators run on continuous candle series (stratlab.kite_tf: Kite-style 09:15 alignment, short last
    candle kept) with the platform's own formulas (stratlab.kite_ind). Decisions at candle close, fills at the
    next 1-minute bar's open (the lab's standard), market exits on signal at the next bar's open.
  * Intraday entries end at the lab's 15:15 square-off (MIS) when the source gives no holding rule.
  * Stock templates run on the whole lab universe (~97 NIFTY-100 names); AlgoTest's point brackets (50 / 100 pts)
    run on NIFTY index futures (underlying NIFTY, 1 lot = 65, spot index as the future).
  * "No stop" = a placeholder stop 50 % away (kite_tf.NOSTOP), i.e. never hit.
  * max_per_day = 75 for signal-driven intraday entries (the platforms re-enter on every new signal while flat).
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Entry, Swing, strategy
from .. import kite_ind as K
from ..kite_tf import machine, series, today, minute_series

MANY = 75
FOREVER = 100_000                   # positional "no max hold": closed by its exit rule or at the period end


def _G(s, tf, key, fn):
    """Per-candle array `key` over the symbol's whole tf-minute series (cached)."""
    base = getattr(s, "_s", s)
    S = series(base._sd, tf)
    return S, S.get(key, fn)


# ══════════════════════════════════════════════════════════════════════════
# Streak examples (help centre / Z-Connect staff replies), stocks, intraday MIS
# ══════════════════════════════════════════════════════════════════════════
def _rsi30(S):
    r = K.rsi(S.c, 14)
    return K.cross_dn(r, 70), K.cross_up(r, 30)


@strategy("r5_kite_streak_rsi30m_short", "intraday", "kite_streak",
          "Streak help example: 30m RSI(14) crosses below 70 -> short; exit RSI crosses above 30, SL 1 %, TP 2 %, 15:15",
          {"tf": 30, "rsi": 14, "ob": 70, "os": 30, "sl_pct": 1.0, "tp_pct": 2.0}, max_per_day=MANY)
def r5_kite_streak_rsi30m_short(s, p):
    S, (si, so) = _G(s, p.tf, "rsi30_short", _rsi30)
    return machine(s, p.tf, short_in=si, short_out=so, bracket=("pct", p.sl_pct, p.tp_pct))


def _orb5(S):
    lo = S.sess_lo[S.sess]
    orh, orl = S.h[lo], S.l[lo]
    same = ~S.first
    return K.cross_up(S.c, orh) & same, K.cross_dn(S.c, orl) & same


@strategy("r5_kite_streak_orb5_long", "intraday", "kite_streak",
          "Streak staff reply: 5m close crosses above the 5-minute opening-range high -> long; exit 5m close crosses below its low",
          {"tf": 5}, max_per_day=MANY)
def r5_kite_streak_orb5_long(s, p):
    S, (li, lo_) = _G(s, p.tf, "orb5", _orb5)
    return machine(s, p.tf, long_in=li, long_out=lo_)


def _nth30(S, nth=4):
    st4 = 9 * 60 + 15 + 30 * (nth - 1)
    idx = S.sess_lo[S.sess] + (nth - 1)
    idx = np.minimum(idx, len(S) - 1)
    ok = (S.start[idx] == st4) & (S.sess[idx] == S.sess) & (S.start > st4)
    h4 = np.where(ok, S.h[idx], np.nan)
    l4 = np.where(ok, S.l[idx], np.nan)
    with np.errstate(invalid="ignore"):
        return S.c > h4, S.c < l4


@strategy("r5_kite_streak_4th30m_long", "intraday", "kite_streak",
          "Streak staff reply: 30m close above the high of the day's 4th 30m candle -> long; exit 30m close below its low",
          {"tf": 30, "nth": 4}, max_per_day=MANY)
def r5_kite_streak_4th30m_long(s, p):
    S, (li, lo_) = _G(s, p.tf, "nth30_4", _nth30)
    return machine(s, p.tf, long_in=li, long_out=lo_)


def _st7(S):
    _d, line = K.supertrend(S.h, S.l, S.c, 7, 0.5)
    with np.errstate(invalid="ignore"):
        return (S.c > S.o) & (S.c > line), (S.c < S.o) & (S.c < line)


@strategy("r5_kite_streak_st7_long", "intraday", "kite_streak",
          "Streak staff reply: 5m green candle closing above Supertrend(7,0.5) -> long; exit red candle closing below it",
          {"tf": 5, "atr": 7, "mult": 0.5}, max_per_day=MANY)
def r5_kite_streak_st7_long(s, p):
    S, (li, lo_) = _G(s, p.tf, "st7_05", _st7)
    return machine(s, p.tf, long_in=li, long_out=lo_)


# ══════════════════════════════════════════════════════════════════════════
# AlgoTest Pine templates (docs.algotest.in/signals/pinescripts), NIFTY futures 5m, 50-pt SL / 100-pt TP,
# an opposite signal reverses the position (Pine strategy.entry, pyramiding 0)
# ══════════════════════════════════════════════════════════════════════════
def _at(s, p, key, fn, short_wins=False):
    S, (L, Sh) = _G(s, p.tf, key, fn)
    if short_wins:                                  # both entry calls on one bar: the later (short) decides
        L = L & ~Sh
    return machine(s, p.tf, long_in=L, short_in=Sh, long_out=Sh, short_out=L,
                   bracket=("pts", p.sl_pts, p.tp_pts))


_ATP = {"tf": 5, "sl_pts": 50, "tp_pts": 100}


def _at_rsi(S):
    r = K.rsi(S.c, 14)
    return K.cross_up(r, 30), K.cross_dn(r, 70)


@strategy("r5_kite_at_rsi", "futures", "kite_algotest",
          "AlgoTest Pine RSI: NIFTY 5m RSI(14) crosses above 30 long / below 70 short; 50-pt SL, 100-pt TP, reversal, 15:15",
          dict(_ATP, rsi=14, os=30, ob=70), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_rsi(s, p):
    return _at(s, p, "at_rsi", _at_rsi)


def _at_st(S):
    return K.supertrend_simplified_algotest(S.h, S.l, S.c, 10, 3.0)


@strategy("r5_kite_at_supertrend", "futures", "kite_algotest",
          "AlgoTest Pine Supertrend (non-ratcheting hlc3+-3ATR10, as coded): flip up long / down short; 50/100 pts, reversal",
          dict(_ATP, atr=10, mult=3.0), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_supertrend(s, p):
    return _at(s, p, "at_st", _at_st, short_wins=True)


def _at_sma(S):
    f, sl = K.sma(S.c, 9), K.sma(S.c, 21)
    return K.cross_up(f, sl), K.cross_dn(f, sl)


@strategy("r5_kite_at_sma9_21", "futures", "kite_algotest",
          "AlgoTest Pine MA crossover: NIFTY 5m SMA9 crosses above SMA21 long / below short; 50/100 pts, reversal",
          dict(_ATP, fast=9, slow=21), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_sma9_21(s, p):
    return _at(s, p, "at_sma", _at_sma)


def _at_macd(S):
    line, sig = K.macd(S.c, 12, 26, 9)
    return K.cross_up(line, sig), K.cross_dn(line, sig)


@strategy("r5_kite_at_macd", "futures", "kite_algotest",
          "AlgoTest Pine MACD: NIFTY 5m MACD(12,26,9) crosses above signal long / below short; 50/100 pts, reversal",
          dict(_ATP, fast=12, slow=26, sig=9), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_macd(s, p):
    return _at(s, p, "at_macd", _at_macd)


def _at_bb(S):
    _m, up, lo = K.bollinger(S.c, 20, 2.0)
    return K.cross_up(S.c, lo), K.cross_dn(S.c, up)


@strategy("r5_kite_at_bollinger", "futures", "kite_algotest",
          "AlgoTest Pine BB(20,2): close crosses above the lower band long / below the upper band short; 50/100 pts, reversal",
          dict(_ATP, n=20, k=2.0), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_bollinger(s, p):
    return _at(s, p, "at_bb", _at_bb)


def _at_adx(S):
    pdi, mdi, adx = K.pine_dmi(S.h, S.l, S.c, 14, 25)
    with np.errstate(invalid="ignore"):
        strong = adx > 25
    return strong & K.cross_up(pdi, mdi), strong & K.cross_dn(pdi, mdi)


@strategy("r5_kite_at_adx", "futures", "kite_algotest",
          "AlgoTest Pine ADX: ta.dmi(14, 25) ADX > 25 and +DI crosses above -DI long / below short; 50/100 pts, reversal",
          dict(_ATP, di=14, adx_smooth=25, adx_min=25), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_adx(s, p):
    return _at(s, p, "at_adx", _at_adx)


def _at_psar(S):
    sar = K.pine_sar(S.h, S.l, S.c, 0.02, 0.02, 0.2)
    return K.cross_up(S.c, sar), K.cross_dn(S.c, sar)


@strategy("r5_kite_at_psar", "futures", "kite_algotest",
          "AlgoTest Pine PSAR(0.02,0.02,0.2): close crosses above SAR long / below short; 50/100 pts, reversal",
          dict(_ATP, start=0.02, inc=0.02, max=0.2), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_psar(s, p):
    return _at(s, p, "at_psar", _at_psar)


def _at_stoch(S):
    k = K.pine_stoch(S.c, S.h, S.l, 14)
    d = K.sma(k, 3)
    with np.errstate(invalid="ignore"):
        return K.cross_up(k, d) & (k < 20), K.cross_dn(k, d) & (k > 80)


@strategy("r5_kite_at_stoch", "futures", "kite_algotest",
          "AlgoTest Pine Stochastic(14,3): %K crosses above %D with %K<20 long / below with %K>80 short; 50/100 pts",
          dict(_ATP, k=14, d=3, os=20, ob=80), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_stoch(s, p):
    return _at(s, p, "at_stoch", _at_stoch)


def _at_emr(S):
    e9, e21 = K.ema(S.c, 9), K.ema(S.c, 21)
    line, sig = K.macd(S.c, 12, 26, 9)
    r = K.rsi(S.c, 7)
    with np.errstate(invalid="ignore"):
        return (K.cross_up(e9, e21) & (sig > line) & (r > 60)), (K.cross_dn(e9, e21) & (sig < line) & (r < 40))


@strategy("r5_kite_at_ema_macd_rsi", "futures", "kite_algotest",
          "AlgoTest Pine template as coded: EMA9 x EMA21 up AND MACD signal > line AND RSI(7) > 60 long (mirror); 50/100 pts",
          dict(_ATP, fast=9, slow=21, rsi=7, hi=60, lo=40), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_ema_macd_rsi(s, p):
    return _at(s, p, "at_emr", _at_emr)


def _at_wr(S):
    w = K.williams_r(S.h, S.l, S.c, 14)
    with np.errstate(invalid="ignore"):
        return w < -80, w > -20


@strategy("r5_kite_at_williams", "futures", "kite_algotest",
          "AlgoTest Pine Williams %R(14): %R < -80 long, %R > -20 short (level test); 50/100 pts, reversal",
          dict(_ATP, n=14, os=-80, ob=-20), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_williams(s, p):
    return _at(s, p, "at_wr", _at_wr)


def _at_ichi(S):
    a, b = K.ichimoku_algotest(S.h, S.l, 9, 26, 52, 26)
    with np.errstate(invalid="ignore"):
        return (S.c > a) & (S.c > b), (S.c < a) & (S.c < b)


@strategy("r5_kite_at_ichimoku", "futures", "kite_algotest",
          "AlgoTest Pine Ichimoku (SMA-of-hl2 lines, as coded): close above both spans long / below both short; 50/100 pts",
          dict(_ATP, conv=9, base=26, span_b=52, disp=26), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r5_kite_at_ichimoku(s, p):
    return _at(s, p, "at_ichi", _at_ichi)


# ══════════════════════════════════════════════════════════════════════════
# AlgoBulls pyalgostrategypool (talib, crossover on the last two candles rounded to 2 dp), stocks, intraday,
# exit on the opposite signal; the same candle's entry check then opens the opposite position
# ══════════════════════════════════════════════════════════════════════════
def _ab_cross(a, b):
    x = K.algobulls_cross(a, b)
    return x == 1, x == -1


def _ab(s, p, key, fn):
    S, (L, Sh, XL, XS) = _G(s, p.tf, key, fn)
    return machine(s, p.tf, long_in=L, short_in=Sh, long_out=XL, short_out=XS)


def _ab_any(L, Sh):
    anyx = L | Sh
    return L, Sh, anyx, anyx                        # exit on the next crossover of either direction


@strategy("r5_kite_ab_ema12_20", "intraday", "kite_algobulls",
          "AlgoBulls EMA regular order: 60m EMA12 crosses above EMA20 buy / below sell; exit on the next crossover",
          {"tf": 60, "fast": 12, "slow": 20}, max_per_day=MANY)
def r5_kite_ab_ema12_20(s, p):
    return _ab(s, p, "ab_ema", lambda S: _ab_any(*_ab_cross(K.ema(S.c, 12), K.ema(S.c, 20))))


@strategy("r5_kite_ab_macd_5m", "intraday", "kite_algobulls",
          "AlgoBulls MACD crossover: 5m MACD(12,26,9) crosses its signal up buy / down sell; exit on the next crossover",
          {"tf": 5, "fast": 12, "slow": 26, "sig": 9}, max_per_day=MANY)
def r5_kite_ab_macd_5m(s, p):
    return _ab(s, p, "ab_macd", lambda S: _ab_any(*_ab_cross(*K.macd(S.c, 12, 26, 9))))


def _ab_aroon(S):
    dn, up = K.talib_aroon(S.h, S.l, 12)
    return _ab_any(*_ab_cross(up, dn))


@strategy("r5_kite_ab_aroon_15m", "intraday", "kite_algobulls",
          "AlgoBulls Aroon crossover: 15m Aroon(12) up crosses above down buy / below sell; exit on the next crossover",
          {"tf": 15, "n": 12}, max_per_day=MANY)
def r5_kite_ab_aroon_15m(s, p):
    return _ab(s, p, "ab_aroon", _ab_aroon)


def _ab_bb(S):
    _m, up, lo = K.bollinger(S.c, 7, 2.0)
    pl, ph, pc = K.shift(S.l), K.shift(S.h), K.shift(S.c)
    with np.errstate(invalid="ignore"):
        buy = (pl <= lo) & (S.c > pc)                # any of prev O/H/L/C <= lower  <=>  prev low <= lower
        sell = (ph >= up) & (S.c < pc) & ~buy        # elif
    return buy, sell, sell, buy


@strategy("r5_kite_ab_bollinger_60m", "intraday", "kite_algobulls",
          "AlgoBulls Bollinger rebound (code defaults 7, 2): prev 60m candle at/below lower band and close up -> buy (mirror)",
          {"tf": 60, "n": 7, "k": 2.0}, max_per_day=MANY)
def r5_kite_ab_bollinger_60m(s, p):
    return _ab(s, p, "ab_bb", _ab_bb)


def _ab_rrsi(S):
    r = K.rsi(S.c, 7)
    os_, ob = K.algobulls_cross(r, 32), K.algobulls_cross(r, 70)
    buy = os_ == 1
    sell = (ob == -1) & ~buy
    return buy, sell, os_ == -1, ob == 1


@strategy("r5_kite_ab_reverse_rsi_60m", "intraday", "kite_algobulls",
          "AlgoBulls reverse RSI (as coded): 60m RSI(7) up through 32 buy, down through 70 sell; exit back through the same level",
          {"tf": 60, "rsi": 7, "os": 32, "ob": 70}, max_per_day=MANY)
def r5_kite_ab_reverse_rsi_60m(s, p):
    return _ab(s, p, "ab_rrsi", _ab_rrsi)


def _ab_stoch(S):
    k, d = K.talib_stoch(S.h, S.l, S.c, 7, 2, 2)
    return _ab_any(*_ab_cross(k, d))


@strategy("r5_kite_ab_stoch_60m", "intraday", "kite_algobulls",
          "AlgoBulls stochastic crossover: 60m STOCH(7,2,2) slowK crosses slowD up buy / down sell; exit on the next crossover",
          {"tf": 60, "fastk": 7, "slowk": 2, "slowd": 2}, max_per_day=MANY)
def r5_kite_ab_stoch_60m(s, p):
    return _ab(s, p, "ab_stoch", _ab_stoch)


def _ab_vwap(S):
    vw = K.session_vwap((S.h + S.l + S.c) / 3.0, S.v, S.first)
    return _ab_any(*_ab_cross(S.c, vw))


@strategy("r5_kite_ab_vwap_60m", "intraday", "kite_algobulls",
          "AlgoBulls VWAP crossover: 60m close crosses the day's VWAP up buy / down sell; exit on the next crossover",
          {"tf": 60}, max_per_day=MANY)
def r5_kite_ab_vwap_60m(s, p):
    return _ab(s, p, "ab_vwap", _ab_vwap)


def _ab_atr(S):
    a = K.atr(S.h, S.l, S.c, 20)
    tr = np.sign(a - K.shift(a, 39))                 # atr.iloc[-40] = 39 candles before the latest
    tr = np.nan_to_num(tr, nan=0.0)
    return tr == 1, tr == -1, tr == -1, tr == 1


@strategy("r5_kite_ab_atr_trend_60m", "intraday", "kite_algobulls",
          "AlgoBulls volatility trend: 60m ATR(20) above its value 40 candles back -> buy, below -> sell; exit when it flips",
          {"tf": 60, "atr": 20, "back": 40}, max_per_day=MANY)
def r5_kite_ab_atr_trend_60m(s, p):
    return _ab(s, p, "ab_atr", _ab_atr)


def _ab_iema(S):
    up, dn = _ab_cross(K.ema(S.c, 2), K.ema(S.c, 7))
    return _ab_any(dn, up)                            # contrarian: EMA2 crosses BELOW EMA7 -> buy


@strategy("r5_kite_ab_inverse_ema_60m", "intraday", "kite_algobulls",
          "AlgoBulls inverse EMA scalping: 60m EMA2 crosses below EMA7 buy / above sell; exit on the next crossover",
          {"tf": 60, "fast": 2, "slow": 7}, max_per_day=MANY)
def r5_kite_ab_inverse_ema_60m(s, p):
    return _ab(s, p, "ab_iema", _ab_iema)


def _ab_rsimacd(S):
    _line, sig = K.macd(S.c, 12, 26, 9)
    r = K.rsi(sig, 3)
    os_, ob = K.algobulls_cross(r, 15), K.algobulls_cross(r, 85)
    buy = os_ == 1
    sell = (ob == -1) & ~buy
    return buy, sell, ob == 1, os_ == -1


@strategy("r5_kite_ab_rsi_macd_60m", "intraday", "kite_algobulls",
          "AlgoBulls RSI-MACD: RSI(3) of the 60m MACD signal up through 15 buy / down through 85 sell; exit at the other level",
          {"tf": 60, "rsi": 3, "os": 15, "ob": 85}, max_per_day=MANY)
def r5_kite_ab_rsi_macd_60m(s, p):
    return _ab(s, p, "ab_rsimacd", _ab_rsimacd)


# ══════════════════════════════════════════════════════════════════════════
# Varsity intraday EMA crossovers ("15 x 30 minutes", "5 x 10-minute": EMA periods on the 1-minute chart, per the
# author's reply), long only, exit on the cross back, 15:15
# ══════════════════════════════════════════════════════════════════════════
def _v_ema1m(s, p):
    base = getattr(s, "_s", s)
    key = f"vema_{p.fast}_{p.slow}"

    def fn(S):
        f, sl = K.ema(S.c, p.fast), K.ema(S.c, p.slow)
        return K.cross_up(f, sl), K.cross_dn(f, sl)
    li, lo_ = minute_series(base._sd, key, fn)
    return machine(s, 1, long_in=li, long_out=lo_)


@strategy("r5_kite_vars_ema15x30_1m", "intraday", "kite_varsity",
          "Varsity MA chapter: 1-minute EMA15 crosses above EMA30 -> long; exit on the cross back; 15:15 square-off",
          {"fast": 15, "slow": 30}, max_per_day=MANY)
def r5_kite_vars_ema15x30_1m(s, p):
    return _v_ema1m(s, p)


@strategy("r5_kite_vars_ema5x10_1m", "intraday", "kite_varsity",
          "Varsity MA chapter: 1-minute EMA5 crosses above EMA10 -> long; exit on the cross back; 15:15 square-off",
          {"fast": 5, "slow": 10}, max_per_day=MANY)
def r5_kite_vars_ema5x10_1m(s, p):
    return _v_ema1m(s, p)


# ══════════════════════════════════════════════════════════════════════════
# Tradetron ORB builds, stocks (1-minute closes stand in for LTP)
# ══════════════════════════════════════════════════════════════════════════
def _first(mask, k):
    m = np.asarray(mask)
    idx = np.flatnonzero(m[k:])
    return k + int(idx[0]) if len(idx) else -1


@strategy("r5_kite_tt_orb15_stock", "intraday", "kite_tradetron",
          "Tradetron Nifty50 stock ORB (stock as the signal/proxy for its CE/PE): 09:15-09:30 range break; stop other edge, target +range",
          {"minutes": 15}, max_per_day=2)
def r5_kite_tt_orb15_stock(s, p):
    hi, lo, k = s.orb(p.minutes)
    if k >= len(s) or not hi > lo:
        return []
    out = []
    i = _first(s.c > hi, k)
    if i >= 0:
        out.append(Entry(i, LONG, stop=lo, target=hi + (hi - lo)))
    i = _first(s.c < lo, k)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=hi, target=lo - (hi - lo)))
    return out


@strategy("r5_kite_tt_orb30_list", "intraday", "kite_tradetron",
          "Tradetron list ORB (US template mapped to NSE): first 30-min range, first break after it; stop other edge, target fill +-2 ranges",
          {"minutes": 30, "target_ranges": 2.0}, max_per_day=1)
def r5_kite_tt_orb30_list(s, p):
    hi, lo, k = s.orb(p.minutes)
    if k >= len(s) or not hi > lo:
        return []
    iu, idn = _first(s.c > hi, k), _first(s.c < lo, k)
    rng = p.target_ranges * (hi - lo)
    if iu >= 0 and (idn < 0 or iu < idn):
        return [Entry(iu, LONG, stop=lo, bracket=("pts", None, rng))]
    if idn >= 0:
        return [Entry(idn, SHORT, stop=hi, bracket=("pts", None, rng))]
    return []


# ══════════════════════════════════════════════════════════════════════════
# Varsity daily (EOD) rules: decision at the daily close, fill at the next open, exit by rule at the next open
# ══════════════════════════════════════════════════════════════════════════
def _ema_d(c, n):
    return K.ema_sma_seed(c, n)


def _px_ema_exit(n):
    def f(v):
        e = _ema_d(v.c, n)
        return bool(np.isfinite(e[-1]) and v.c[-1] < e[-1])
    return f


@strategy("r5_kite_vars_px_ema50", "swing", "kite_varsity",
          "Varsity: daily close turns above its 50-day EMA -> buy (delivery); exit when the close turns below it",
          {"n": 50})
def r5_kite_vars_px_ema50(d, p):
    if len(d) < p.n + 1:
        return None
    e = _ema_d(d.c, p.n)
    if np.isfinite(e[-2]) and d.c[-1] > e[-1] and d.c[-2] <= e[-2]:
        return Swing(LONG, max_hold=FOREVER, exit_fn=_px_ema_exit(p.n))
    return None


def _x_exit(f, sl):
    def g(v):
        a, b = _ema_d(v.c, f), _ema_d(v.c, sl)
        return bool(np.isfinite(b[-1]) and a[-1] < b[-1])
    return g


def _ema_x(d, p):
    if len(d) < p.slow + 1:
        return None
    a, b = _ema_d(d.c, p.fast), _ema_d(d.c, p.slow)
    if np.isfinite(b[-2]) and a[-1] > b[-1] and a[-2] <= b[-2]:
        return Swing(LONG, max_hold=FOREVER, exit_fn=_x_exit(p.fast, p.slow))
    return None


@strategy("r5_kite_vars_ema9_21", "swing", "kite_varsity",
          "Varsity: daily EMA9 crosses above EMA21 -> buy (delivery); exit when EMA9 crosses below EMA21", {"fast": 9, "slow": 21})
def r5_kite_vars_ema9_21(d, p):
    return _ema_x(d, p)


@strategy("r5_kite_vars_ema25_50", "swing", "kite_varsity",
          "Varsity: daily EMA25 crosses above EMA50 -> buy (delivery); exit when EMA25 crosses below EMA50", {"fast": 25, "slow": 50})
def r5_kite_vars_ema25_50(d, p):
    return _ema_x(d, p)


@strategy("r5_kite_vars_ema50_100", "swing", "kite_varsity",
          "Varsity: daily EMA50 crosses above EMA100 -> buy (delivery); exit when EMA50 crosses below EMA100",
          {"fast": 50, "slow": 100})
def r5_kite_vars_ema50_100(d, p):
    return _ema_x(d, p)


@strategy("r5_kite_vars_ema100_200", "swing", "kite_varsity",
          "Varsity: daily EMA100 crosses above EMA200 -> buy (delivery); exit when EMA100 crosses below EMA200",
          {"fast": 100, "slow": 200})
def r5_kite_vars_ema100_200(d, p):
    return _ema_x(d, p)


def _macd_d(c, fast=12, slow=26, sig=9):
    """Daily MACD with SMA-seeded EMAs (TA-Lib style): NaN until slow + sig - 1 sessions exist."""
    line = _ema_d(c, fast) - _ema_d(c, slow)
    sg = np.full(len(c), np.nan)
    if len(c) >= slow:
        sg[slow - 1:] = K.ema_sma_seed(line[slow - 1:], sig)
    return line, sg


def _macd_exit(side):
    def f(v):
        line, sig = _macd_d(v.c)
        if not np.isfinite(sig[-1]):
            return False
        return bool(line[-1] < sig[-1]) if side > 0 else bool(line[-1] > sig[-1])
    return f


@strategy("r5_kite_vars_macd_d", "futures", "kite_varsity",
          "Varsity: daily MACD(12,26,9) crosses above signal -> long, below -> short (stock futures); opposite cross exits and reverses",
          {"fast": 12, "slow": 26, "sig": 9}, horizon="positional")
def r5_kite_vars_macd_d(d, p):
    if len(d) < p.slow + p.sig:
        return None
    line, sig = _macd_d(d.c, p.fast, p.slow, p.sig)
    if not (np.isfinite(sig[-1]) and np.isfinite(sig[-2])):
        return None
    if line[-1] > sig[-1] and line[-2] <= sig[-2]:
        return Swing(LONG, max_hold=FOREVER, exit_fn=_macd_exit(LONG))
    if line[-1] < sig[-1] and line[-2] >= sig[-2]:
        return Swing(SHORT, max_hold=FOREVER, exit_fn=_macd_exit(SHORT))
    return None


def _bb_exit(side, n):
    def f(v):
        m = K.sma(v.c, n)
        return bool(v.c[-1] >= m[-1]) if side > 0 else bool(v.c[-1] <= m[-1])
    return f


@strategy("r5_kite_vars_bb20_d", "futures", "kite_varsity",
          "Varsity: daily low touches the lower BB(20,2) -> long, high touches the upper -> short (futures); exit at the 20-day SMA",
          {"n": 20, "k": 2.0}, horizon="positional")
def r5_kite_vars_bb20_d(d, p):
    if len(d) < p.n:
        return None
    _m, up, lo = K.bollinger(d.c, p.n, p.k)
    tl, th = d.l[-1] <= lo[-1], d.h[-1] >= up[-1]
    if tl and not th:
        return Swing(LONG, max_hold=FOREVER, exit_fn=_bb_exit(LONG, p.n))
    if th and not tl:
        return Swing(SHORT, max_hold=FOREVER, exit_fn=_bb_exit(SHORT, p.n))
    return None


# ── Varsity candlesticks (risk-taker entry: pattern checked at 15:20 on the pattern day, filled at the 15:20 bar
#    open; stop at the pattern extreme; target = 1.5 R (Varsity checklist "minimum RRR 1.5"); stock futures).
#    Prior trend = the close of the session before the pattern's first candle vs its 10-day SMA. ─────────────────
SHADOW_TOL = 0.002        # Varsity author: shadows "0.2 % or lesser" (marubozu); also used for "tiny" shadows
HAMMER_BODY = 0.02        # Varsity hammer 3:20 check: open and close "within 1-2 %" -> |C - O| <= 2 % of O
RR = 1.5


def _trend(d, k):
    """+1 / -1 / 0: close of session index k (negative = from the end) above / below its 10-day SMA."""
    c = d.c
    j = len(c) + k if k < 0 else k
    if j < 9:
        return 0
    m = float(np.mean(c[j - 9:j + 1]))
    return 1 if c[j] > m else (-1 if c[j] < m else 0)


def _avg_body(d, k, n=10):
    """Mean |C-O| of the n sessions before index k (lab candles.py 'long' / 'short' body reference)."""
    j = len(d.c) + k
    if j < n:
        return np.nan
    return float(np.mean(np.abs(d.c[j - n:j] - d.o[j - n:j])))


def _cs(side, stop):
    return Swing(side, stop=float(stop), rr=RR, max_hold=FOREVER)


def cs_bull_marubozu(d):
    o, h, l, c = d.o[-1], d.h[-1], d.l[-1], d.c[-1]
    if c > o and (o - l) <= SHADOW_TOL * c and (h - c) <= SHADOW_TOL * c and 0.01 <= (h - l) / c <= 0.10:
        return _cs(LONG, l)
    return None


def cs_bear_marubozu(d):
    o, h, l, c = d.o[-1], d.h[-1], d.l[-1], d.c[-1]
    if c < o and (h - o) <= SHADOW_TOL * c and (c - l) <= SHADOW_TOL * c and 0.01 <= (h - l) / c <= 0.10:
        return _cs(SHORT, h)
    return None


def _umbrella(d):
    o, h, l, c = d.o[-1], d.h[-1], d.l[-1], d.c[-1]
    body = abs(c - o)
    return (body <= HAMMER_BODY * o and (min(o, c) - l) >= 2 * body and (h - max(o, c)) <= SHADOW_TOL * c
            and h > l)


def cs_hammer(d):
    if len(d) >= 11 and _trend(d, -2) < 0 and _umbrella(d):
        return _cs(LONG, d.l[-1])
    return None


def cs_hanging_man(d):
    if len(d) >= 11 and _trend(d, -2) > 0 and _umbrella(d):
        return _cs(SHORT, d.h[-1])
    return None


def cs_shooting_star(d):
    if len(d) < 11 or _trend(d, -2) <= 0:
        return None
    o, h, l, c = d.o[-1], d.h[-1], d.l[-1], d.c[-1]
    body = abs(c - o)
    if (h - max(o, c)) >= 2 * body and (min(o, c) - l) <= SHADOW_TOL * c and h > l:
        return _cs(SHORT, h)
    return None


def cs_bull_engulfing(d):
    if len(d) < 12 or _trend(d, -3) >= 0:
        return None
    o1, c1, o2, c2 = d.o[-2], d.c[-2], d.o[-1], d.c[-1]
    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 > o1:
        return _cs(LONG, min(d.l[-2], d.l[-1]))
    return None


def cs_bear_engulfing(d):
    if len(d) < 12 or _trend(d, -3) <= 0:
        return None
    o1, c1, o2, c2 = d.o[-2], d.c[-2], d.o[-1], d.c[-1]
    if c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
        return _cs(SHORT, max(d.h[-2], d.h[-1]))
    return None


def cs_piercing(d):
    if len(d) < 12 or _trend(d, -3) >= 0:
        return None
    o1, c1, o2, c2 = d.o[-2], d.c[-2], d.o[-1], d.c[-1]
    b1, b2 = o1 - c1, c2 - o2
    if b1 > 0 and b2 > 0 and o2 <= c1 and 0.5 * b1 <= b2 < b1:
        return _cs(LONG, min(d.l[-2], d.l[-1]))
    return None


def cs_dark_cloud(d):
    if len(d) < 12 or _trend(d, -3) <= 0:
        return None
    o1, c1, o2, c2 = d.o[-2], d.c[-2], d.o[-1], d.c[-1]
    b1, b2 = c1 - o1, o2 - c2
    if b1 > 0 and b2 > 0 and o2 > c1 and 0.5 * b1 <= b2 <= b1:
        return _cs(SHORT, max(d.h[-2], d.h[-1]))
    return None


def cs_morning_star(d):
    if len(d) < 23 or _trend(d, -4) >= 0:
        return None
    o1, c1, o2, c2, o3, c3 = d.o[-3], d.c[-3], d.o[-2], d.c[-2], d.o[-1], d.c[-1]
    ab1, ab2 = _avg_body(d, -3), _avg_body(d, -2)
    if not (np.isfinite(ab1) and np.isfinite(ab2)):
        return None
    if (c1 < o1 and (o1 - c1) > ab1 and abs(c2 - o2) < ab2 and max(o2, c2) < c1
            and o3 > max(o2, c2) and c3 > o1 and c3 > o3):
        return _cs(LONG, min(d.l[-3], d.l[-2], d.l[-1]))
    return None


def cs_evening_star(d):
    if len(d) < 23 or _trend(d, -4) <= 0:
        return None
    o1, c1, o2, c2, o3, c3 = d.o[-3], d.c[-3], d.o[-2], d.c[-2], d.o[-1], d.c[-1]
    ab1, ab2 = _avg_body(d, -3), _avg_body(d, -2)
    if not (np.isfinite(ab1) and np.isfinite(ab2)):
        return None
    if (c1 > o1 and (c1 - o1) > ab1 and abs(c2 - o2) < ab2 and min(o2, c2) > c1
            and o3 < min(o2, c2) and c3 < o1 and c3 < o3):
        return _cs(SHORT, max(d.h[-3], d.h[-2], d.h[-1]))
    return None


CANDLES = {
    "bull_marubozu": (cs_bull_marubozu, "bullish marubozu (shadows <= 0.2 %, range 1-10 %), stop at its low"),
    "bear_marubozu": (cs_bear_marubozu, "bearish marubozu (shadows <= 0.2 %, range 1-10 %), stop at its high"),
    "hammer": (cs_hammer, "hammer after a downtrend (lower shadow >= 2x body, body <= 2 %), stop at its low"),
    "hanging_man": (cs_hanging_man, "hanging man after an uptrend (hammer shape), short, stop at its high"),
    "shooting_star": (cs_shooting_star, "shooting star after an uptrend (upper shadow >= 2x body), short, stop high"),
    "bull_engulfing": (cs_bull_engulfing, "bullish engulfing after a downtrend, stop at the P1-P2 low"),
    "bear_engulfing": (cs_bear_engulfing, "bearish engulfing after an uptrend, short, stop at the P1-P2 high"),
    "piercing": (cs_piercing, "piercing (P2 body 50-100 % of P1, open <= P1 close) after a downtrend, stop low"),
    "dark_cloud": (cs_dark_cloud, "dark cloud cover (P2 body 50-100 % of P1) after an uptrend, short, stop high"),
    "morning_star": (cs_morning_star, "morning star (long red, gapped small body, P3 gaps up and closes > P1 open)"),
    "evening_star": (cs_evening_star, "evening star (long blue, gapped small body, P3 gaps down, closes < P1 open)"),
}


def _register_candles():
    for name, (fn, what) in CANDLES.items():
        def run(d, p, _fn=fn):
            return _fn(d)
        run.__name__ = f"r5_kite_vars_cs_{name}"
        strategy(f"r5_kite_vars_cs_{name}", "futures", "kite_varsity_cs",
                 f"Varsity {what}; checked at 15:20, filled at the 15:20 open, target 1.5R (stock futures)",
                 {"pattern": name, "rr": RR, "shadow_tol": SHADOW_TOL, "hammer_body": HAMMER_BODY, "trend_sma": 10},
                 horizon="positional", decision_time="15:20")(run)

        def run_nxo(d, p, _fn=fn):
            return _fn(d)
        run_nxo.__name__ = f"r5_kite_vars_cs_{name}_nxo"
        strategy(f"r5_kite_vars_cs_{name}_nxo", "futures", "kite_varsity_cs",
                 f"2012-23 daily-data twin of the Varsity {name}: same rule on the full daily candle, filled at the NEXT open",
                 {"pattern": name, "rr": RR, "shadow_tol": SHADOW_TOL, "hammer_body": HAMMER_BODY, "trend_sma": 10,
                  "fill": "next_open"}, horizon="positional")(run_nxo)


_register_candles()


# ══════════════════════════════════════════════════════════════════════════
# Tradetron Double Supertrend on NIFTY futures (daily): both green -> long 2 lots, both red -> short 1 lot,
# exit as soon as they disagree (decided at the daily close, executed at the next open)
# ══════════════════════════════════════════════════════════════════════════
def _dst_state(v):
    return int(K.double_supertrend(v.h, v.l, v.c)[-1])


def _dst_exit(side):
    def f(v):
        return _dst_state(v) != side
    return f


@strategy("r5_kite_tt_dst_nifty", "futures", "kite_tradetron",
          "Tradetron Double Supertrend: NIFTY daily ST(10,3) and ST(20,2) both green long 2 lots / both red short 1 lot; exit on disagreement",
          {"st1": (10, 3.0), "st2": (20, 2.0), "long_lots": 2, "short_lots": 1}, horizon="positional", underlying="NIFTY")
def r5_kite_tt_dst_nifty(d, p):
    if len(d) < 25:
        return None
    st = _dst_state(d)
    if st == 1:
        return Swing(LONG, max_hold=FOREVER, exit_fn=_dst_exit(1), lots=p.long_lots)
    if st == -1:
        return Swing(SHORT, max_hold=FOREVER, exit_fn=_dst_exit(-1), lots=p.short_lots)
    return None
