"""
stratlab/tfgrid.py — every timeframe-portable rule on every timeframe, traded like a person with Rs 10 lakh per
agent (capital.py), judged by the lab's pre-registered protocols. Research only (PAPER/research, never live).

What it runs (all fixed in the pre-registration written by `--preregister` BEFORE any result):
  PORTABLE GRID   RULES (textbook parameters, no tuning) x TIMEFRAMES (1, 3, 5, 15, 30, 60 minute, daily, weekly)
                  on the NIFTY-100 stocks (agent by timeframe: 1-5 min scalping, 15-60 min intraday, daily / weekly
                  swing = CNC longs + stock-futures shorts) and, for daily / weekly, again as stock futures both ways
                  (futures agent), and on NIFTY / BANKNIFTY index futures at every timeframe (futures agent; rules
                  that need volume are not run on the indices, whose minute bars carry no volume).
  NATIVE          every catalogue @strategy (stratlab/catalog) on its own timeframe through the UNCHANGED sim.py
                  (wrapped only to apply the 1:3 bracket and to read its initial stop), and the live PAPER
                  strategies + AUC-FADE-2PCT through their own module runners (report only).
  EXIT RULE       user instruction "Risk 1:3", amended into the plan before any result: every grid combo trades a
                  1:3 bracket - stop = the rule's own initial stop (rules without one: 1.5 x ATR(14) of its own
                  timeframe), target = fill +- 3 x |fill - stop|. The rule's own exit signal, its max hold and the
                  15:15 intraday square-off still apply; whichever comes first ends the trade. The original
                  exits are NOT run as extra variants.
  ENGINE          the lab's fill model at minute resolution for every timeframe: a decision at the close of a candle
                  fills at the OPEN of the next 1-minute bar, slippage worse (2 bps stocks, 1 bps index); stop and
                  target are checked on every later minute's low / high (the fill minute included), stop first if
                  both touch, a gap through the stop fills at that open, a target needs a trade one tick through it;
                  rule exits fill at the next minute's open; intraday timeframes take no fill at/after 15:15 and are
                  squared off at the 15:15 open; daily / weekly hold overnight and are closed at the period's last
                  close. Indicators run on the continuous candle series (Kite-style: candles aligned at 09:15, the
                  short last candle kept; daily = one session; weekly = the sessions of one ISO week); the value at
                  candle j uses candles <= j only, and the data are CUT at the period end before anything is computed.
  CANDIDATES      every entry event (the rule's condition turning true) with its own trade outcome; capital.simulate
                  then takes or skips each one as the account allows (one position per symbol, slots, margin, daily
                  loss stop, trades per day, time order, ties by previous-session traded value).
  PROTOCOL        stage 1 = protocol.py's DISCOVERY screen (2023-11-10 .. 2025-12-16, H1/H2, PF >= 1.15, net > 0 in
                  both halves, >= 150 intraday / 60 positional trades, top symbol <= 35 %) on the account's trades;
                  passers get ONE FINAL look (2025-12-17 .. 2026-11-09, PF >= 1.05, net > 0; tfgrid final ledger);
                  stage 2 = protocol_min_oos's gates on 2016-01-01 .. 2023-11-09 (H1 2016-19, H2 2020-23), once, for
                  every stage-1 passer, dirty-link rule applied before the account sees a candidate; DSR (trials.py)
                  of the stage-2 daily account P&L with N = the full trial count and with the grid's own N.
                  PROVEN = stage-1 pass AND FINAL confirmed AND stage-2 pass AND DSR(full N) >= 0.95.

    cd algotrader_v4
    python -m stratlab.tfgrid --list
    python -m stratlab.tfgrid --preregister            # writes <scratch>/tfgrid/PREREGISTERED_TFGRID.json + repo copy
    python -m stratlab.tfgrid --stage1 [--jobs 4]      # DISCOVERY, every combo
    python -m stratlab.tfgrid --final                  # one FINAL look for stage-1 passers
    python -m stratlab.tfgrid --stage2                 # 2016-23 for stage-1 passers
    python -m stratlab.tfgrid --controls               # random-entry / planted-edge controls through the account
    python -m stratlab.tfgrid --live                   # the 4 named strategies, Rs 10 lakh accounts, both eras
    python -m stratlab.tfgrid --report                 # REPORT.md, trials ledger rows, DSR
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import namedtuple

import numpy as np

from . import capital as K
from . import data as D
from . import kite_ind as KI
from . import protocol as P
from .data import SESSION_OPEN

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SCRATCH = os.environ.get("STRATLAB_SCRATCH", _SCRATCH)
ROOT = os.path.join(SCRATCH, "tfgrid")
PREREG = os.path.join(ROOT, "PREREGISTERED_TFGRID.json")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "tfgrid_v1.json")
BARS3Y = os.path.join(SCRATCH, "bars3y")
CACHE3Y = os.path.join(SCRATCH, "stratlab_cache", "bars3y_fb3d4d")
CACHE_OLD = os.path.join(SCRATCH, "oldera_min", "cache")
FINAL_LEDGER = os.path.join(ROOT, "final_ledger.json")
PROGRESS = os.path.join(ROOT, "progress.txt")

TFS = ("1m", "3m", "5m", "15m", "30m", "60m", "1d", "1w")
TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "60m": 60}
POSITIONAL_TFS = ("1d", "1w")
STOCK_AGENT = {"1m": "scalping", "3m": "scalping", "5m": "scalping", "15m": "intraday", "30m": "intraday",
               "60m": "intraday", "1d": "swing", "1w": "swing"}
SQ_OFF = 15 * 60 + 15
RR = 3.0                      # the user's 1:3 risk-reward bracket
ATR_STOP = 1.5                # rules without their own stop: 1.5 x ATR(14) of their own timeframe
ATR_N = 14
SLIP_STOCK = 2.0 / 1e4        # sim.DEFAULT_SLIPPAGE_BPS
SLIP_INDEX = 1.0 / 1e4        # sim.INDEX_SLIPPAGE_BPS
INDEX_SYMS = {"_NIFTY50_min": "NIFTY", "_NIFTYBANK_min": "BANKNIFTY"}
INDEX_LOT = {"NIFTY": 65, "BANKNIFTY": 30}      # sim.LOT_SIZES (2026 lots, every date)
SPLIT3 = {"h1": (20231110, 20241129), "h2": (20241202, 20251216), "final": (20251217, 20261109)}
DISC = (SPLIT3["h1"][0], SPLIT3["h2"][1])
OLD_ERA = (20160101, 20231109)
CONTROL_SYMBOLS = 25          # controls: the first 25 stock symbols (alphabetical), as controls/run_controls.py
CONTROL_SEEDS = 10            # random-entry controls per timeframe (luck rate of the account screen)
REASONS = ("stop", "target", "signal", "squareoff", "max_hold", "period_end")


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        os.makedirs(ROOT, exist_ok=True)
        with open(PROGRESS, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# minute bars and candles
# ══════════════════════════════════════════════════════════════════════════
class Bars:
    """One instrument's minute bars CUT at `hi` (nothing after the period end is ever loaded into an indicator)."""

    def __init__(self, sd, hi: int | None = None, lo_data: int | None = None):
        keep = np.ones(len(sd.date), bool)
        if hi is not None:
            keep &= sd.date <= hi
        if lo_data is not None:
            keep &= sd.date >= lo_data
        self.symbol = sd.symbol
        self.date = np.asarray(sd.date[keep], np.int64)
        self.t = np.asarray(sd.t[keep], np.int64)
        self.o, self.h, self.l, self.c, self.v = (np.asarray(x[keep], float) for x in (sd.o, sd.h, sd.l, sd.c, sd.v))
        n = len(self.date)
        if n:
            cut = np.flatnonzero(self.date[1:] != self.date[:-1]) + 1
            self.starts = np.concatenate(([0], cut)).astype(np.int64)
            self.ends = np.concatenate((cut, [n])).astype(np.int64)
        else:
            self.starts = self.ends = np.zeros(0, np.int64)
        self.days = self.date[self.starts] if n else np.zeros(0, np.int64)
        self.sess = np.repeat(np.arange(len(self.starts)), self.ends - self.starts)
        # first minute at/after 15:15 per session (== session end if the data stop earlier)
        self.sq = np.array([a + int(np.searchsorted(self.t[a:b], SQ_OFF)) for a, b in zip(self.starts, self.ends)],
                           np.int64)
        tv = np.add.reduceat(self.c * self.v, self.starts) if n else np.zeros(0)
        self.prio = np.concatenate(([0.0], tv[:-1])) if n else np.zeros(0)     # previous session's traded value
        self.key = self.date * 10000 + self.t

    def __len__(self):
        return len(self.date)


Candles = namedtuple("Candles", "o h l c v first last sess")


def candles(B: Bars, tf: str) -> Candles:
    """Continuous candles of one timeframe: o h l c v, first / last minute index, session index."""
    n = len(B)
    if n == 0:
        z = np.zeros(0)
        return Candles(z, z, z, z, z, z.astype(np.int64), z.astype(np.int64), z.astype(np.int64))
    if tf in TF_MIN:
        k = TF_MIN[tf]
        bid = B.sess * 1000 + (B.t - SESSION_OPEN) // k
        cut = np.flatnonzero(bid[1:] != bid[:-1]) + 1
        st = np.concatenate(([0], cut)).astype(np.int64)
    elif tf == "1d":
        st = B.starts.copy()
    elif tf == "1w":
        wk = np.array([_isoweek(int(d)) for d in B.days], np.int64)
        cs = np.concatenate(([0], np.flatnonzero(wk[1:] != wk[:-1]) + 1))
        st = B.starts[cs]
    else:
        raise ValueError(tf)
    en = np.concatenate((st[1:], [n])) - 1
    return Candles(B.o[st], np.maximum.reduceat(B.h, st), np.minimum.reduceat(B.l, st), B.c[en],
                   np.add.reduceat(B.v, st), st, en, B.sess[st])


def _isoweek(d: int) -> int:
    y, w, _ = _dt.date(d // 10000, d // 100 % 100, d % 100).isocalendar()
    return y * 100 + w


# ══════════════════════════════════════════════════════════════════════════
# rules (textbook parameters; signals at candle close; masks over candles)
# ══════════════════════════════════════════════════════════════════════════
Sig = namedtuple("Sig", "li si lo so stl sts max_hold")
Rule = namedtuple("Rule", "id family desc stop_desc needs_volume fn")
RULES: dict = {}


def rule(rid, family, desc, stop_desc="1.5 x ATR(14) (no own stop)", needs_volume=False):
    def deco(fn):
        RULES[rid] = Rule(rid, family, desc, stop_desc, needs_volume, fn)
        return fn
    return deco


def _sh(x, k=1):
    return KI.shift(x, k)


def _b(x):
    return np.asarray(np.nan_to_num(np.asarray(x, float), nan=0.0) != 0, bool) if np.asarray(x).dtype != bool \
        else np.asarray(x, bool)


def _onset(m):
    """Entry EVENT: the condition turning true on this candle (a condition that stays true is one event)."""
    m = np.asarray(m, bool)
    out = m.copy()
    out[1:] = m[1:] & ~m[:-1]
    return out


def _sig(li, si, lo=None, so=None, stl=None, sts=None, max_hold=None):
    n = len(li)
    z = np.zeros(n, bool)
    return Sig(_onset(li), _onset(si), z if lo is None else np.asarray(lo, bool),
               z if so is None else np.asarray(so, bool), stl, sts, max_hold)


def _cross(a, b):
    return KI.cross_up(a, b), KI.cross_dn(a, b)


def _ma_cross(C, f, s, kind="ema"):
    fa = KI.ema(C.c, f) if kind == "ema" else KI.sma(C.c, f)
    sa = KI.ema(C.c, s) if kind == "ema" else KI.sma(C.c, s)
    if kind == "ema":
        fa[:f - 1] = np.nan
        sa[:s - 1] = np.nan
    up, dn = _cross(fa, sa)
    return _sig(up, dn, dn, up)


@rule("ema_9_21", "ma_cross", "EMA9 crosses EMA21: long on the up-cross, short on the down-cross; exit on the opposite cross")
def r_ema_9_21(C):
    return _ma_cross(C, 9, 21)


@rule("ema_20_50", "ma_cross", "EMA20 crosses EMA50; exit on the opposite cross")
def r_ema_20_50(C):
    return _ma_cross(C, 20, 50)


@rule("sma_50_200", "ma_cross", "SMA50 crosses SMA200 (golden / death cross); exit on the opposite cross")
def r_sma_50_200(C):
    return _ma_cross(C, 50, 200, "sma")


@rule("px_ema50", "ma_cross", "Close crosses EMA50 (Varsity); exit on the opposite cross")
def r_px_ema50(C):
    e = KI.ema(C.c, 50)
    e[:49] = np.nan
    up, dn = _cross(C.c, e)
    return _sig(up, dn, dn, up)


@rule("macd_signal", "macd", "MACD(12,26,9) line crosses its signal line; exit on the opposite cross")
def r_macd_signal(C):
    line, sg = KI.macd(C.c)
    up, dn = _cross(line, sg)
    return _sig(up, dn, dn, up)


@rule("macd_zero", "macd", "MACD(12,26) line crosses zero; exit on the opposite cross")
def r_macd_zero(C):
    line, _ = KI.macd(C.c)
    up, dn = _cross(line, 0.0)
    return _sig(up, dn, dn, up)


@rule("supertrend_10_3", "supertrend", "Supertrend(10, 3) turns green / red; exit on the flip",
      "the Supertrend line at the signal candle")
def r_supertrend(C):
    d, line = KI.supertrend(C.h, C.l, C.c, 10, 3.0)
    pd_ = _sh(d.astype(float))
    up = (d == 1) & (pd_ == -1)
    dn = (d == -1) & (pd_ == 1)
    return _sig(up, dn, dn, up, line, line)


@rule("psar", "psar", "Parabolic SAR(0.02, 0.02, 0.2) flips below / above the close; exit on the flip",
      "the SAR value at the signal candle")
def r_psar(C):
    sar = KI.pine_sar(C.h, C.l, C.c)
    with np.errstate(invalid="ignore"):
        upst = np.isfinite(sar) & (C.c > sar)
        dnst = np.isfinite(sar) & (C.c < sar)
    up = upst & _b(_sh(dnst.astype(float)))
    dn = dnst & _b(_sh(upst.astype(float)))
    return _sig(up, dn, dn, up, sar, sar)


@rule("adx_di", "adx", "ADX(14) > 25 and +DI crosses above -DI (long) / -DI above +DI (short); exit on the DI cross back")
def r_adx(C):
    pdi, mdi, adx = KI.pine_dmi(C.h, C.l, C.c, 14, 14)
    up, dn = _cross(pdi, mdi)
    with np.errstate(invalid="ignore"):
        strong = adx > 25
    return _sig(up & strong, dn & strong, dn, up)


@rule("ichimoku_tk", "ichimoku", "Ichimoku(9,26,52): Tenkan crosses Kijun above the cloud (long) / below it (short); "
      "exit on the opposite Tenkan-Kijun cross")
def r_ichimoku(C):
    mid = lambda n: (KI.highest(C.h, n) + KI.lowest(C.l, n)) / 2.0
    ten, kij = mid(9), mid(26)
    sa, sb = _sh((ten + kij) / 2.0, 26), _sh(mid(52), 26)
    up, dn = _cross(ten, kij)
    with np.errstate(invalid="ignore"):
        above = C.c > np.fmax(sa, sb)
        below = C.c < np.fmin(sa, sb)
    return _sig(up & above, dn & below, dn, up)


@rule("aroon_25", "aroon", "Aroon(25): Aroon-up crosses above Aroon-down (long) / below (short); exit on the opposite cross")
def r_aroon(C):
    dn_, up_ = KI.talib_aroon(C.h, C.l, 25)
    up, dn = _cross(up_, dn_)
    return _sig(up, dn, dn, up)


@rule("heikin_ashi", "heikin_ashi", "Heikin-Ashi candle turns green (long) / red (short); exit on the colour flip")
def r_heikin(C):
    hc = (C.o + C.h + C.l + C.c) / 4.0
    n = len(hc)
    if n == 0:
        return _sig(np.zeros(0, bool), np.zeros(0, bool))
    seq = np.concatenate(([(C.o[0] + C.c[0]) / 2.0], hc[:-1]))
    import pandas as pd
    ho = pd.Series(seq).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    g, r = hc > ho, hc < ho
    return _sig(g, r, r, g)


def _donchian(C, n_in, n_out):
    hh, ll = _sh(KI.highest(C.h, n_in)), _sh(KI.lowest(C.l, n_in))
    xh, xl = _sh(KI.highest(C.h, n_out)), _sh(KI.lowest(C.l, n_out))
    with np.errstate(invalid="ignore"):
        li, si = C.c > hh, C.c < ll
        lo, so = C.c < xl, C.c > xh
    nat = 2.0 * KI.atr(C.h, C.l, C.c, 20)                  # Turtle N = ATR(20); the Turtle stop is 2N
    return _sig(li, si, lo, so, C.c - nat, C.c + nat)


@rule("donchian_20_10", "donchian", "Turtle system 1: close above the prior 20-candle high (long) / below the low "
      "(short); exit on a close beyond the prior 10-candle opposite extreme", "Turtle 2N = 2 x ATR(20)")
def r_don20(C):
    return _donchian(C, 20, 10)


@rule("donchian_55_20", "donchian", "Turtle system 2: 55-candle breakout, 20-candle opposite exit", "Turtle 2N = 2 x ATR(20)")
def r_don55(C):
    return _donchian(C, 55, 20)


@rule("keltner_20_2", "keltner", "Close crosses above EMA20 + 2 x ATR(10) (long) / below EMA20 - 2 x ATR(10) (short); "
      "exit on a close back through EMA20")
def r_keltner(C):
    m = KI.ema(C.c, 20)
    m[:19] = np.nan
    a = KI.atr(C.h, C.l, C.c, 10)
    up1, _ = _cross(C.c, m + 2 * a)
    _, dn1 = _cross(C.c, m - 2 * a)
    up_m, dn_m = _cross(C.c, m)
    return _sig(up1, dn1, dn_m, up_m)


@rule("bb_breakout", "bollinger", "Close crosses above the upper Bollinger band(20, 2) (long) / below the lower "
      "(short); exit on a close back through the middle band")
def r_bb_break(C):
    m, u, lw = KI.bollinger(C.c, 20, 2.0)
    up, _ = _cross(C.c, u)
    _, dn = _cross(C.c, lw)
    um, dm = _cross(C.c, m)
    return _sig(up, dn, dm, um)


@rule("roc_12", "momentum", "Rate of change(12) crosses zero; exit on the opposite cross")
def r_roc(C):
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = C.c / _sh(C.c, 12) - 1.0
    up, dn = _cross(roc, 0.0)
    return _sig(up, dn, dn, up)


@rule("vol_breakout_20", "volume", "Close above the prior 20-candle high on volume > 2 x its 20-candle average (long; "
      "mirror short); exit on a close beyond the prior 10-candle opposite extreme", needs_volume=True)
def r_volbrk(C):
    hh, ll = _sh(KI.highest(C.h, 20)), _sh(KI.lowest(C.l, 20))
    xh, xl = _sh(KI.highest(C.h, 10)), _sh(KI.lowest(C.l, 10))
    va = _sh(KI.sma(C.v, 20))
    with np.errstate(invalid="ignore"):
        big = C.v > 2.0 * va
        return _sig((C.c > hh) & big, (C.c < ll) & big, C.c < xl, C.c > xh)


@rule("rsi_30_70", "rsi", "RSI(14) crosses up through 30 (long, exit when it crosses up through 70) / down through 70 "
      "(short, exit when it crosses down through 30)")
def r_rsi(C):
    r = KI.rsi(C.c, 14)
    u30, d30 = _cross(r, 30.0)
    u70, d70 = _cross(r, 70.0)
    return _sig(u30, d70, u70, d30)


@rule("rsi2_connors", "rsi", "Connors RSI(2): RSI2 < 10 above SMA200 (long, exit close > SMA5) / RSI2 > 90 below "
      "SMA200 (short, exit close < SMA5)")
def r_rsi2(C):
    r2 = KI.rsi(C.c, 2)
    s200, s5 = KI.sma(C.c, 200), KI.sma(C.c, 5)
    with np.errstate(invalid="ignore"):
        return _sig((r2 < 10) & (C.c > s200), (r2 > 90) & (C.c < s200), C.c > s5, C.c < s5)


@rule("bb_reversion", "bollinger", "Close crosses below the lower Bollinger band(20, 2) (long) / above the upper "
      "(short); exit on a close back through the middle band")
def r_bb_rev(C):
    m, u, lw = KI.bollinger(C.c, 20, 2.0)
    _, dn = _cross(C.c, lw)
    up, _ = _cross(C.c, u)
    um, dm = _cross(C.c, m)
    return _sig(dn, up, um, dm)


@rule("stoch_14_3_3", "stochastic", "Slow stochastic(14,3,3): %K crosses above %D below 20 (long, exit %K crosses "
      "up through 80) / below %D above 80 (short, exit %K crosses down through 20)")
def r_stoch(C):
    k = KI.sma(KI.pine_stoch(C.c, C.h, C.l, 14), 3)
    d = KI.sma(k, 3)
    up, dn = _cross(k, d)
    u80, _ = _cross(k, 80.0)
    _, d20 = _cross(k, 20.0)
    with np.errstate(invalid="ignore"):
        return _sig(up & (k < 20), dn & (k > 80), u80, d20)


@rule("williams_r_14", "williams", "Williams %R(14) crosses up through -80 (long, exit up through -20) / down through "
      "-20 (short, exit down through -80)")
def r_wr(C):
    w = KI.williams_r(C.h, C.l, C.c, 14)
    u80, d80 = _cross(w, -80.0)
    u20, d20 = _cross(w, -20.0)
    return _sig(u80, d20, u20, d80)


@rule("cci_20", "cci", "Lambert CCI(20): crosses above +100 (long) / below -100 (short); exit when it crosses back "
      "through 0")
def r_cci(C):
    tp = (C.h + C.l + C.c) / 3.0
    n = 20
    cci = np.full(len(tp), np.nan)
    if len(tp) >= n:
        w = np.lib.stride_tricks.sliding_window_view(tp, n)
        m = w.mean(axis=1)
        md = np.abs(w - m[:, None]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cci[n - 1:] = (tp[n - 1:] - m) / (0.015 * md)
    up, _ = _cross(cci, 100.0)
    _, dn = _cross(cci, -100.0)
    u0, d0 = _cross(cci, 0.0)
    return _sig(up, dn, d0, u0)


@rule("mfi_14", "volume", "Money flow index(14) crosses up through 20 (long, exit up through 80) / down through 80 "
      "(short, exit down through 20)", needs_volume=True)
def r_mfi(C):
    tp = (C.h + C.l + C.c) / 3.0
    mf = tp * C.v
    ptp = _sh(tp)
    with np.errstate(invalid="ignore"):
        pos = np.where(tp > ptp, mf, 0.0)
        neg = np.where(tp < ptp, mf, 0.0)
    import pandas as pd
    sp_ = pd.Series(pos).rolling(14).sum().to_numpy()
    sn = pd.Series(neg).rolling(14).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mfi = np.where(sn > 0, 100.0 - 100.0 / (1.0 + sp_ / sn), np.where(sp_ > 0, 100.0, 50.0))
    mfi[:14] = np.nan
    u20, d20 = _cross(mfi, 20.0)
    u80, d80 = _cross(mfi, 80.0)
    return _sig(u20, d80, u80, d20)


@rule("inside_bar_brk", "price_pattern", "Inside bar: the candle before is inside its mother candle; a close above the "
      "mother's high (long) / below its low (short); max hold 10 candles", "the other side of the mother candle")
def r_inside(C):
    ins = np.zeros(len(C.c), bool)
    ins[1:] = (C.h[1:] <= C.h[:-1]) & (C.l[1:] >= C.l[:-1])
    pins = _b(_sh(ins.astype(float)))
    mh, ml = _sh(C.h, 2), _sh(C.l, 2)
    with np.errstate(invalid="ignore"):
        li, si = pins & (C.c > mh), pins & (C.c < ml)
    return Sig(li, si, np.zeros(len(li), bool), np.zeros(len(li), bool), ml, mh, 10)


@rule("nr7_brk", "price_pattern", "NR7: the candle before has the narrowest range of the last 7; a close above its high "
      "(long) / below its low (short); max hold 10 candles", "the other side of the NR7 candle")
def r_nr7(C):
    rg = C.h - C.l
    nr = np.zeros(len(rg), bool)
    if len(rg) >= 7:
        w = np.lib.stride_tricks.sliding_window_view(rg, 7)
        nr[6:] = rg[6:] <= w.min(axis=1)
    pn = _b(_sh(nr.astype(float)))
    ph, pl = _sh(C.h), _sh(C.l)
    with np.errstate(invalid="ignore"):
        li, si = pn & (C.c > ph), pn & (C.c < pl)
    return Sig(li, si, np.zeros(len(li), bool), np.zeros(len(li), bool), pl, ph, 10)


@rule("td9_setup", "td_sequential", "TD Sequential setup: 9 consecutive closes below (above) the close 4 candles earlier "
      "-> long (short) at the 9th; max hold 10 candles", "the lowest low (highest high) of the 9 setup candles")
def r_td9(C):
    c4 = _sh(C.c, 4)
    with np.errstate(invalid="ignore"):
        dn, up = C.c < c4, C.c > c4

    def runlen(m):
        out = np.zeros(len(m), np.int64)
        r = 0
        for i, x in enumerate(m):
            r = r + 1 if x else 0
            out[i] = r
        return out
    rd, ru = runlen(dn), runlen(up)
    lo9, hi9 = KI.lowest(C.l, 9), KI.highest(C.h, 9)
    return Sig(rd == 9, ru == 9, np.zeros(len(C.c), bool), np.zeros(len(C.c), bool), lo9, hi9, 10)


CS_PAIRS = {"cs_engulfing": ("bull_engulfing", "bear_engulfing"), "cs_hammer_star": ("hammer", "shooting_star"),
            "cs_harami": ("bull_harami", "bear_harami"), "cs_piercing_dark": ("piercing_line", "dark_cloud"),
            "cs_stars": ("morning_star", "evening_star"), "cs_soldiers_crows": ("three_white_soldiers", "three_black_crows"),
            "cs_marubozu": ("bull_marubozu", "bear_marubozu"), "cs_three_inside": ("three_inside_up", "three_inside_down"),
            "cs_three_outside": ("three_outside_up", "three_outside_down"), "cs_tweezer": ("tweezer_bottom", "tweezer_top")}


def _cs_rule(rid, pair):
    def fn(C):
        from . import candles as CS
        sg = CS.signals(C.o, C.h, C.l, C.c, list(pair))
        li, _ = sg[pair[0]]
        _, si = sg[pair[1]]
        nl, ns = CS.PATTERNS[pair[0]].n, CS.PATTERNS[pair[1]].n
        lo_l, _ = CS.pattern_extremes(C.l, C.h, nl)
        _, hi_s = CS.pattern_extremes(C.l, C.h, ns)
        a = KI.atr(C.h, C.l, C.c, ATR_N)
        stl = np.fmin(lo_l, C.c - 0.3 * a)
        sts = np.fmax(hi_s, C.c + 0.3 * a)
        return Sig(np.asarray(li, bool), np.asarray(si, bool), np.zeros(len(C.c), bool), np.zeros(len(C.c), bool),
                   stl, sts, 10)
    rule(rid, "candlestick", f"{pair[0]} (long) / {pair[1]} (short), candles.py textbook detectors with the lab's "
         f"prior-trend rule; max hold 10 candles", "beyond the pattern extreme, at least 0.3 x ATR(14) (the lab's plan)")(fn)


for _rid, _pair in CS_PAIRS.items():
    _cs_rule(_rid, _pair)


# ── controls (not trials): random entries and planted edges through the same engine + account ──
CONTROL_RULES: dict = {}
PLANTED = (0.05, 0.10, 0.20, 1.00)
PLANT_H = 10                   # planted side = the sign of the move over the next 10 candles (a deliberate peek)


def _control(rid, seed, planted):
    def fn(C):
        n = len(C.c)
        z = np.zeros(n, bool)
        if n == 0:
            return Sig(z, z, z, z, None, None, 10)
        per_sess = n / max(1, len(np.unique(C.sess)))
        p = 2.0 / per_sess if per_sess > 1.5 else (0.10 if n > 400 else 0.20)
        key = int(abs(float(C.c[0])) * 100) % 1_000_003
        rng = np.random.default_rng([seed, key, n])
        ev = rng.random(n) < min(p, 1.0)
        side = np.where(rng.random(n) < 0.5, 1, -1)
        if planted:
            fut = np.full(n, np.nan)
            fut[:n - PLANT_H] = C.c[PLANT_H:] - C.c[:n - PLANT_H]
            use = (rng.random(n) < planted) & np.isfinite(fut) & (fut != 0)
            side = np.where(use, np.sign(np.nan_to_num(fut)).astype(int), side)
        return Sig(ev & (side > 0), ev & (side < 0), z, z, None, None, 10)
    CONTROL_RULES[rid] = Rule(rid, "control", f"control seed {seed} planted {planted}", "1.5 x ATR(14)", False, fn)


for _s in range(1, CONTROL_SEEDS + 1):
    _control(f"CTRL_RANDOM_{_s}", _s, 0.0)
for _x in PLANTED:
    _control(f"CTRL_PLANTED_{int(round(_x * 100))}", 1, _x)


# ══════════════════════════════════════════════════════════════════════════
# candidate engine (the lab's fill rules at minute resolution, 1:3 bracket)
# ══════════════════════════════════════════════════════════════════════════
def _tick(price, date, index):
    if index:
        return 0.05
    return 0.01 if (price < 250.0 and int(date) >= 20250616) else 0.05


def _target_fill(side, target, o, extreme, tick):
    from .sim import target_fill
    return target_fill(side, target, o, extreme, tick)


def gen(B: Bars, C: Candles, S: Sig, *, positional: bool, index: bool, lo: int, hi: int, rr: float = RR,
        slip: float | None = None) -> dict:
    """All entry events of one rule on one instrument with their trade outcomes. Returns column arrays:
    side, f (fill minute index), fill, stop, x (exit minute index), xclose (exit at that minute's close), xpx, reason."""
    slip = (SLIP_INDEX if index else SLIP_STOCK) if slip is None else slip
    n = len(B)
    nc = len(C.c)
    cols = {k: [] for k in ("side", "f", "fill", "stop", "x", "xclose", "xpx", "reason")}
    if n == 0 or nc == 0:
        return {k: np.asarray(v) for k, v in cols.items()}
    a14 = KI.atr(C.h, C.l, C.c, ATR_N)
    o, h, l = B.o, B.h, B.l
    for side, ent, ext, own in ((1, S.li, S.lo, S.stl), (-1, S.si, S.so, S.sts)):
        stop_arr = (C.c - side * ATR_STOP * a14) if own is None else np.asarray(own, float)
        E = np.flatnonzero(np.asarray(ext, bool))
        J = np.flatnonzero(np.asarray(ent, bool) & np.isfinite(stop_arr))
        for j in J:
            f = int(C.last[j]) + 1
            if f >= n:
                continue
            s_f = int(B.sess[f])
            if not positional:
                if s_f != int(C.sess[j]) or B.t[f] >= SQ_OFF:
                    continue
            d = int(B.date[f])
            if d < lo or d > hi:
                continue
            raw = float(o[f])
            stop = float(stop_arr[j])
            if side * (raw - stop) <= 0:
                continue                                    # already through the stop at the open: not taken
            fill = raw * (1.0 + side * slip)
            risk = abs(fill - stop)
            target = fill + side * rr * risk if rr else None
            # the latest exit: square-off / period end, then the rule's exit signal, then max hold
            if positional:
                xm, xc, why = n - 1, True, "period_end"
            else:
                sq = int(B.sq[s_f])
                if sq < int(B.ends[s_f]):
                    xm, xc, why = sq, False, "squareoff"
                else:
                    xm, xc, why = int(B.ends[s_f]) - 1, True, "squareoff"
            if len(E):
                p = int(np.searchsorted(E, j, side="right"))
                if p < len(E):
                    m = int(C.last[E[p]]) + 1
                    if m < n and (m * 2 < xm * 2 + xc):
                        xm, xc, why = m, False, "signal"
            if S.max_hold:
                kk = j + int(S.max_hold)
                if kk < nc:
                    m = int(C.last[kk])
                    if m * 2 + 1 < xm * 2 + xc:
                        xm, xc, why = m, True, "max_hold"
            e = xm + 1 if xc else xm                        # minutes [f, e) are checked for stop / target
            px = None
            if e > f:
                if side > 0:
                    hs = l[f:e] <= stop
                    ht = h[f:e] >= target + _tick(target, d, index) - 1e-9 if target is not None else None
                else:
                    hs = h[f:e] >= stop
                    ht = l[f:e] <= target - _tick(target, d, index) + 1e-9 if target is not None else None
                hit = hs | ht if ht is not None else hs
                if hit.any():
                    r = int(hit.argmax())
                    k = f + r
                    if hs[r]:
                        q = min(stop, float(o[k])) if side > 0 else max(stop, float(o[k]))
                        px, xm, xc, why = q * (1.0 - side * slip), k, False, "stop"
                    else:
                        tk = _tick(target, int(B.date[k]), index)
                        px = _target_fill(side, target, float(o[k]), float(h[k] if side > 0 else l[k]), tk)
                        xm, xc, why = k, False, "target"
            if px is None:
                px = (float(B.c[xm]) if xc else float(o[xm])) * (1.0 - side * slip)
            cols["side"].append(side)
            cols["f"].append(f)
            cols["fill"].append(fill)
            cols["stop"].append(stop)
            cols["x"].append(xm)
            cols["xclose"].append(xc)
            cols["xpx"].append(px)
            cols["reason"].append(REASONS.index(why))
    out = {k: np.asarray(v) for k, v in cols.items()}
    return out


def to_columns(B: Bars, g: dict, sym_code: int) -> dict:
    """Engine output -> the account's candidate columns (keys, prices, priority)."""
    f = np.asarray(g["f"]).astype(np.int64)
    if len(f) == 0:
        return {"sym": np.zeros(0, np.int32), "side": np.zeros(0, np.int8), "ekey": np.zeros(0, np.int64),
                "fill": np.zeros(0), "stop": np.zeros(0), "xkey": np.zeros(0, np.int64), "xpx": np.zeros(0),
                "reason": np.zeros(0, np.int8), "prio": np.zeros(0)}
    x = np.asarray(g["x"]).astype(np.int64)
    return {"sym": np.full(len(f), sym_code, np.int32), "side": g["side"].astype(np.int8),
            "ekey": B.key[f], "fill": g["fill"].astype(float), "stop": g["stop"].astype(float),
            "xkey": B.key[x] + g["xclose"].astype(np.int64), "xpx": g["xpx"].astype(float),
            "reason": g["reason"].astype(np.int8), "prio": B.prio[B.sess[f]]}


def concat(parts: list) -> dict:
    keys = ("sym", "side", "ekey", "fill", "stop", "xkey", "xpx", "reason", "prio")
    parts = [p for p in parts if p is not None and len(p["ekey"])]
    if not parts:
        return to_columns(None, {"f": np.zeros(0)}, 0)
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


# ══════════════════════════════════════════════════════════════════════════
# combos
# ══════════════════════════════════════════════════════════════════════════
def combos() -> list:
    """Every portable combo: (combo id, rule, tf, universe, agent). universe: stocks | stockfut | index."""
    out = []
    for rid, r in RULES.items():
        for tf in TFS:
            out.append((f"tfg_{rid}@{tf}:stocks", rid, tf, "stocks", STOCK_AGENT[tf]))
            if tf in POSITIONAL_TFS:
                out.append((f"tfg_{rid}@{tf}:stockfut", rid, tf, "stockfut", "futures"))
            if not r.needs_volume:
                out.append((f"tfg_{rid}@{tf}:index", rid, tf, "index", "futures"))
    return out


def instrument(universe: str, agent: str, side: int) -> str:
    if universe in ("stockfut", "index"):
        return "fut"
    if agent == "swing":
        return "cnc" if side > 0 else "fut"
    return "mis"


def horizon_of(tf: str) -> str:
    return "positional" if tf in POSITIONAL_TFS else "intraday"


# ══════════════════════════════════════════════════════════════════════════
# per-symbol work
# ══════════════════════════════════════════════════════════════════════════
_W: dict = {}


def _sym_work(args):
    """One instrument, one timeframe, every rule: candidate columns per rule (lo..hi, data cut at hi)."""
    cache, sym, code, tf, lo, hi, rids = args
    sd = D.load_symbol(cache, sym)
    out = {}
    if sd is None:
        return sym, out, 0.0
    t0 = time.time()
    B = Bars(sd, hi=hi)
    del sd
    if len(B) == 0:
        return sym, out, 0.0
    C = candles(B, tf)
    index = sym in INDEX_SYMS
    for rid in rids:
        r = RULES.get(rid) or CONTROL_RULES.get(rid)
        if r is None or (index and getattr(r, "needs_volume", False)):
            continue
        try:
            with np.errstate(all="ignore"):
                S = r.fn(C)
            g = gen(B, C, S, positional=tf in POSITIONAL_TFS, index=index, lo=lo, hi=hi)
            out[rid] = to_columns(B, g, code)
        except Exception as exc:                     # a rule error makes the combo INVALID, never silently empty
            out[rid] = {"error": f"{sym}: {type(exc).__name__}: {str(exc)[:160]}"}
    return sym, out, time.time() - t0


def stock_symbols(cache: str) -> list:
    with open(os.path.join(cache, "_manifest.json")) as fh:
        return sorted(s for s in json.load(fh)["symbols"] if not s.startswith("_"))


def lot_sizes(symbols, day: int = 20261109) -> dict:
    """Stock-futures lots: nsedata's bhavcopy lot in force on `day` (2026 lots, used for every date; labelled)."""
    from . import nsedata as N
    out = {}
    for s in symbols:
        try:
            out[s] = int(N.lot_size(s, day))
        except Exception:
            out[s] = None
    return out


def build_candidates(cache, tf, lo, hi, rids, jobs=4, symbols=None, with_index=True) -> tuple:
    """{rid: {"stocks": cols, "index": cols, "errors": [...]}} for one timeframe, and the symbol table."""
    syms = list(symbols if symbols is not None else stock_symbols(cache))
    if with_index:
        syms += list(INDEX_SYMS)
    table = {s: i for i, s in enumerate(syms)}
    args = [(cache, s, table[s], tf, lo, hi, list(rids)) for s in syms]
    parts: dict = {rid: {"stocks": [], "index": [], "errors": []} for rid in rids}
    cpu = 0.0
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs, maxtasksperchild=4) as pool:
        for sym, out, secs in pool.imap_unordered(_sym_work, args):
            cpu += secs
            for rid, cols in out.items():
                if "error" in cols:
                    parts[rid]["errors"].append(cols["error"])
                    continue
                parts[rid]["index" if sym in INDEX_SYMS else "stocks"].append(cols)
    res = {rid: {"stocks": concat(p["stocks"]), "index": concat(p["index"]), "errors": p["errors"]}
           for rid, p in parts.items()}
    return res, syms, cpu


def cand_iter(cols: dict, syms: list, universe: str, agent: str, lots: dict, cid: str, dirty=None, nxt=None,
              fb_lots: int = 1):
    """Candidate tuples in the account's priority order (time, then previous-session traded value)."""
    n = len(cols["ekey"])
    if n == 0:
        return []
    order = np.lexsort((cols["side"], cols["sym"], -cols["prio"], cols["ekey"]))
    out = []
    excluded = 0
    for i in order:
        sym = syms[int(cols["sym"][i])]
        side = int(cols["side"][i])
        inst = instrument(universe, agent, side)
        if universe == "index":
            lot = INDEX_LOT[INDEX_SYMS[sym]]
        elif inst == "fut":
            lot = lots.get(sym)
            if not lot:
                excluded += 1
                continue
        else:
            lot = 1
        ek, xk = int(cols["ekey"][i]), int(cols["xkey"][i])
        if dirty is not None:
            from .protocol_min_oos import _dirty_hit
            if _dirty_hit(dirty, sym, ek // 10000, xk // 10000, nxt or {}):
                excluded += 1
                continue
        out.append((sym, side, ek, float(cols["fill"][i]), float(cols["stop"][i]), xk, float(cols["xpx"][i]),
                    REASONS[int(cols["reason"][i])], float(cols["prio"][i]), inst, int(lot), int(fb_lots), cid))
    cand_iter.excluded = excluded
    return out


def run_account(cols, syms, universe, agent, lots, cid, agcfg=None, dirty=None, nxt=None, compound=False,
                fb_lots: int = 1) -> dict:
    ag = agcfg or K.agents()[agent]
    if compound:
        from dataclasses import replace
        ag = replace(ag, compound=True)
    cs = cand_iter(cols, syms, universe, agent, lots, cid, dirty, nxt, fb_lots)
    res = K.simulate(cs, ag, cid, presorted=True)
    res["excluded"] = getattr(cand_iter, "excluded", 0)
    return res


# ══════════════════════════════════════════════════════════════════════════
# native catalogue specs (unchanged sim.py; 1:3 bracket applied by a wrapper)
# ══════════════════════════════════════════════════════════════════════════
NATIVE_EXCLUDED = {}           # id -> reason (filled by native_specs)


def native_agent(sp) -> tuple:
    """(agent, universe) of a catalogue spec on its native timeframe."""
    if sp.kind == "intraday":
        return "intraday", "stocks"
    if sp.kind == "scalping":
        return "scalping", "stocks"
    if sp.kind == "swing":
        return "swing", "stocks"
    if getattr(sp, "underlying", None):
        return "futures", "index"
    return "futures", "stockfut"


def native_class(sp) -> str:
    """Why a catalogue id is run on its native timeframe only (informational, from its id / description)."""
    txt = (sp.id + " " + sp.desc).lower()
    for key, why in (("auction", "clock time (09:15 auction)"), ("gap", "event (overnight gap)"),
                     ("orb", "clock time (opening range)"), ("opening", "clock time (opening range)"),
                     ("expiry", "event (expiry day)"), ("delivery", "data source (NSE delivery %)"),
                     ("oi", "data source (F&O open interest)"), ("participant", "data source (participant OI)"),
                     ("budget", "event (calendar)"), ("mpc", "event (calendar)"), ("holiday", "event (calendar)"),
                     ("vwap", "session anchor (VWAP)"), ("pdh", "session anchor (previous day levels)"),
                     ("15:", "clock time"), ("14:", "clock time"), ("09:", "clock time"), ("10:", "clock time"),
                     ("nifty", "index relation / index futures"), ("month", "calendar"), ("week", "calendar")):
        if key in txt:
            return "native (" + why + ")"
    return "native timeframe (catalogue parameters; its family is also in the portable grid where it is one)"


def _atr_daily(d) -> float:
    a = KI.atr(d.h, d.l, d.c, ATR_N)
    return float(a[-1]) if len(a) and np.isfinite(a[-1]) else float("nan")


_STOPS: dict = {}              # (spec id, symbol) -> {...} recorded by the wrappers


def wrap_rr(sp):
    """A copy of a catalogue spec whose entries carry the 1:3 bracket: same stop (none -> 1.5 x ATR(14) of its own
    timeframe: the 5-minute ATR the lab keeps for intraday sessions, the daily ATR for positional specs), target
    removed and rr = 3; its own exit signal, time stop, trailing rule and max hold unchanged."""
    from dataclasses import replace
    from .api import Swing
    fn0 = sp.fn
    sid = sp.id

    if sp.horizon == "intraday":
        def fn(s, p):
            ents = fn0(s, p) or []
            if not isinstance(ents, (list, tuple)):
                ents = [ents]
            out = []
            for e in ents:
                if e is None:
                    continue
                br = getattr(e, "bracket", None)
                if br is not None:
                    e.bracket = (br[0], br[1], None)
                if e.stop is None and br is None:
                    a = float(s.atr5[e.i]) if np.isfinite(s.atr5[e.i]) else float("nan")
                    if not np.isfinite(a):
                        continue
                    e.stop = float(s.c[e.i]) - e.side * ATR_STOP * a
                e.target = None
                e.rr = RR
                out.append(e)
            return out if sp.style == "vector" else (out[0] if out else None)
    else:
        def fn(d, p):
            sig = fn0(d, p)
            if sig is None or not isinstance(sig, Swing):
                return sig
            if sig.stop is None and sig.stop_pct is None and sig.entry_stop_ticks is None:
                a = _atr_daily(d)
                if not np.isfinite(a):
                    return None
                sig.stop = float(d.c[-1]) - sig.side * ATR_STOP * a
            sig.target = None
            sig.target_pct = None
            sig.rr = RR
            _STOPS.setdefault((sid, d.symbol), []).append((int(d.date[-1]), sig.stop, sig.stop_pct, sig.side))
            return sig
    fn.__wrapped__ = fn0
    return replace(sp, fn=fn)


def native_specs() -> dict:
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    return dict(sorted(REGISTRY.items()))


def _native_stop_of(sp, tr, sd_days) -> float:
    """Initial stop of a lab trade: intraday from the recorded entry, positional from the recorded Swing."""
    fill, side = float(tr[5]), int(tr[3])
    if sp.horizon == "intraday":
        rec = _STOPS.get((sp.id, tr[1], int(tr[2]), str(tr[4]), side))
        return float(rec) if rec is not None else float("nan")
    recs = _STOPS.get((sp.id, tr[1]), [])
    e = int(tr[2])
    best = None
    for (dd, st, sp_pct, sd_) in recs:
        if sd_ == side and (dd < e or (sp.decision_time is not None and dd == e)):
            if best is None or dd > best[0]:
                best = (dd, st, sp_pct)
    if best is None:
        return float("nan")
    if best[1] is not None:
        return float(best[1])
    if best[2]:
        return fill * (1 - side * best[2] / 100.0)
    return float("nan")


def _native_work(args):
    """One symbol through ONE sim.run_symbol call (per year for intraday specs when `years`) for a chunk of wrapped
    catalogue specs; candidate columns per spec. sim._take is wrapped only to record each taken entry's stop."""
    cache, sym, code, ids, lo, hi, years = args
    from . import sim
    from .api import REGISTRY
    from . import catalog  # noqa: F401
    specs = [wrap_rr(REGISTRY[i]) for i in ids]
    specs = [sp for sp in specs if sim.applies(sp, sym)]
    out = {}
    if not specs:
        return sym, out
    orig_take = sim._take

    def rec_take(spec, s, e, trades, slip=None):
        n0 = len(trades)
        k = orig_take(spec, s, e, trades, slip)
        if len(trades) > n0:
            fill = float(trades[-1][5])
            br = getattr(e, "bracket", None)
            if br is not None and br[1] is not None:
                dist = fill * float(br[1]) / 100.0 if br[0] == "pct" else float(br[1])
                st = fill - e.side * dist
            else:
                st = float(e.stop)
            _STOPS[(spec.id, s.symbol, int(s.date), str(trades[-1][4]), int(e.side))] = st
        return k
    sim._take = rec_take
    try:
        sd = D.load_symbol(cache, sym)
        if sd is None:
            return sym, out
        idx = D.load_symbol(cache, "_NIFTY50_min")
        res = {sp.id: {"trades": [], "error": None} for sp in specs}

        def fold(r):
            for i, v in r.items():
                res[i]["trades"] += v["trades"]
                res[i]["error"] = res[i]["error"] or v["error"]
        pos = [sp for sp in specs if sp.horizon == "positional"]
        intr = [sp for sp in specs if sp.horizon == "intraday"]
        if pos:
            fold(sim.run_symbol(cache, sym, pos, lo, hi, audit=False, sd=sd, index=idx))
        if intr:
            spans = [(max(lo, y * 10000 + 101), min(hi, y * 10000 + 1231)) for y in range(lo // 10000, hi // 10000 + 1)] \
                if years else [(lo, hi)]
            for a, b in spans:
                fold(sim.run_symbol(cache, sym, intr, a, b, audit=False, sd=sd, index=idx))
        by = {sp.id: sp for sp in specs}
        for i, r in res.items():
            if r["error"]:
                out[i] = {"error": r["error"]}
            elif r["trades"]:
                stops = [_native_stop_of(by[i], t, sd.days) for t in r["trades"]]
                out[i] = _native_cols(r["trades"], stops, code, sd)
    finally:
        sim._take = orig_take
        _STOPS.clear()
    return sym, out


_CUR = [""]


def _hm2m(s: str, default: int) -> int:
    try:
        return int(s[:2]) * 60 + int(s[3:5])
    except Exception:
        return default


def _native_cols(trades, stops, code, sd) -> dict:
    days = [int(x) for x in sd.days]
    d = sd.daily()
    tv = d["c"] * d["v"]
    prio_of = {days[i]: (float(tv[i - 1]) if i > 0 else 0.0) for i in range(len(days))}
    ek, xk = [], []
    for t in trades:
        ek.append(int(t[2]) * 10000 + _hm2m(str(t[4]), 555))
        xt = str(t[7])
        xk.append(int(t[6]) * 10000 + (_hm2m(xt, 929) + (1 if xt in ("", "15:29") else 0)))
    rs = [REASONS.index(t[13]) if t[13] in REASONS else (REASONS.index("period_end") if t[13] in ("data_end",)
          else REASONS.index("signal") if t[13] in ("time", "signal") else REASONS.index("squareoff")) for t in trades]
    return {"sym": np.full(len(trades), code, np.int32), "side": np.array([int(t[3]) for t in trades], np.int8),
            "ekey": np.array(ek, np.int64), "fill": np.array([float(t[5]) for t in trades]),
            "stop": np.array(stops, float), "xkey": np.array(xk, np.int64),
            "xpx": np.array([float(t[8]) for t in trades]), "reason": np.array(rs, np.int8),
            "prio": np.array([prio_of.get(int(t[2]), 0.0) for t in trades])}


def build_native(cache, ids, lo, hi, jobs=4, years=False, symbols=None, log_fn=log) -> tuple:
    """{id: {"cols": candidate columns, "errors": [...]}} and the symbol table; fresh worker per symbol."""
    syms = list(symbols if symbols is not None else stock_symbols(cache)) + list(INDEX_SYMS)
    table = {s: i for i, s in enumerate(syms)}
    parts = {i: {"cols": [], "errors": []} for i in ids}
    args = [(cache, s, table[s], list(ids), lo, hi, years) for s in syms]
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
        for k, (sym, out) in enumerate(pool.imap_unordered(_native_work, args), 1):
            for i, c in out.items():
                if "error" in c:
                    parts[i]["errors"].append(c["error"])
                else:
                    parts[i]["cols"].append(c)
            if k % 20 == 0 or k == len(args):
                log_fn(f"    native {len(ids)} ids: {k}/{len(args)} symbols ({time.time() - t0:.0f}s)")
    return {i: {"cols": concat(p["cols"]), "errors": p["errors"]} for i, p in parts.items()}, syms


# ══════════════════════════════════════════════════════════════════════════
# judging
# ══════════════════════════════════════════════════════════════════════════
def split3():
    return P.Split(SPLIT3["h1"], SPLIT3["h2"], SPLIT3["final"], 726, 508, 218)


def judge_stage1(res: dict, tf_or_horizon: str, single: bool) -> dict:
    hz = tf_or_horizon if tf_or_horizon in ("intraday", "positional") else horizon_of(tf_or_horizon)
    return P.screen(res["trades"], split3(), hz, single_underlying=single)


def summary_row(cid, meta, res, met, scr) -> dict:
    return {"id": cid, **meta, "candidates": res["candidates"], "taken": res["taken"], "skipped": res["skipped"],
            "excluded": res.get("excluded", 0), "metrics": met,
            "screen": {"passed": scr["passed"], "reasons": scr["reasons"], "h1": scr["h1"], "h2": scr["h2"],
                       "disc": scr["disc"], "top_symbol": scr["top_symbol"],
                       "top_share": None if scr["top_share"] != scr["top_share"] else round(scr["top_share"], 4)},
            "mean_gross": round(float(np.mean([t[10] for t in res["trades"]])), 2) if res["trades"] else None,
            "mean_cost": round(float(np.mean([t[11] for t in res["trades"]])), 2) if res["trades"] else None}


def calendar3() -> list:
    from . import trials as T
    return T.calendar_3y(SCRATCH)


# ══════════════════════════════════════════════════════════════════════════
# pre-registration
# ══════════════════════════════════════════════════════════════════════════
CODE_FILES = ("tfgrid.py", "tfgrid_run.py", "tests_tfgrid.py", "capital.py", "sim.py", "protocol.py", "protocol_min_oos.py", "protocol_oos.py",
              "kite_ind.py", "candles.py", "data.py", "trials.py", "auction.py", "attention.py", "options_vrp.py")


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def code_sha() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {f: _sha(os.path.join(here, f)) for f in CODE_FILES if os.path.exists(os.path.join(here, f))}
    root = os.path.dirname(here)
    out["../costs.py"] = _sha(os.path.join(root, "costs.py"))
    out["../config.py"] = _sha(os.path.join(root, "config.py"))
    return out


def rule_fingerprint(rid) -> str:
    r = RULES[rid]
    try:
        src = inspect.getsource(r.fn)
    except Exception:
        src = repr(r.fn)
    if rid in CS_PAIRS:
        src += repr(CS_PAIRS[rid]) + inspect.getsource(_cs_rule)
    return hashlib.sha1((src + repr((RR, ATR_STOP, ATR_N))).encode()).hexdigest()[:12]


LIVE4 = ("AUC-FADE-3PCT", "AUC-FADE-2PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED")


def preregister(path: str = PREREG, repo_copy: str = REPO_PREREG, amendment: str = "") -> dict:
    from .sim import COST_MODEL
    cv = K.config_values()
    ags = {k: v.as_dict() for k, v in K.agents(cv).items()}
    nat = native_specs()
    stocks = stock_symbols(CACHE3Y)
    lots = lot_sizes(stocks)
    grid = [{"id": c[0], "rule": c[1], "tf": c[2], "universe": c[3], "agent": c[4], "horizon": horizon_of(c[2]),
             "fingerprint": rule_fingerprint(c[1])} for c in combos()]
    natives = []
    for i, sp in nat.items():
        ag, uni = native_agent(sp)
        natives.append({"id": f"nat:{i}:rr3", "catalogue_id": i, "kind": sp.kind, "horizon": sp.horizon,
                        "agent": ag, "universe": uni, "class": native_class(sp), "fingerprint": sp.fingerprint})
    doc = {
        "title": "tfgrid v1: every timeframe-portable rule on every timeframe, Rs 10 lakh per agent, 1:3 bracket",
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statement": "Written BEFORE any tfgrid result was computed. Code + this file are committed before the run.",
        "amendment": amendment,
        "periods": {"stage1_discovery": DISC, "h1": SPLIT3["h1"], "h2": SPLIT3["h2"], "final_one_look": SPLIT3["final"],
                    "stage2_era": OLD_ERA, "stage2_h1": [20160101, 20191231], "stage2_h2": [20200101, 20231109]},
        "data": {"2023-26": CACHE3Y, "2016-23": CACHE_OLD, "universe": "today's NIFTY-100 stocks (survivorship-biased) "
                 "+ NIFTY 50 / NIFTY BANK spot minute bars standing in for index futures",
                 "stocks": stocks},
        "timeframes": list(TFS), "stock_agent_by_tf": STOCK_AGENT,
        "exit_rule": {"rr": RR, "atr_stop_mult": ATR_STOP, "atr_n": ATR_N,
                      "text": "1:3 bracket on every combo: stop = the rule's own initial stop (none: 1.5 x ATR(14) of "
                              "its own timeframe), target = fill +- 3 x |fill - stop|; the rule's exit signal, its max "
                              "hold and the 15:15 intraday square-off still apply, whichever comes first. The textbook "
                              "exits are NOT run as extra variants. Natives: the same wrapper on their own entries "
                              "(no stop: 1.5 x the lab's 5-minute ATR intraday / daily ATR(14) positional). "
                              "Exception: the live strategies + AUC-FADE-2PCT keep their own exits; a 1:3 version of "
                              "them is report-only (no trial)."},
        "engine": {"fills": "decision at candle close, fill at the next 1-minute open +- slippage (2 bps stocks, 1 bps "
                            "index); stops/targets on every later minute's low/high incl. the fill minute; stop first; "
                            "gap through the stop fills at the open; target needs one tick through; rule exits at the "
                            "next minute's open; intraday tfs: no fill at/after 15:15, square-off at the 15:15 open; "
                            "daily/weekly: overnight, closed at the period's last close; data cut at the period end",
                   "entry_events": "an entry is the rule's condition turning true on a candle; every event is a "
                                   "candidate with its own outcome; the account picks",
                   "index_futures": "spot index minute bars stand in for the future (no basis / roll; lab convention), "
                                    "2026 lots NIFTY 65 / BANKNIFTY 30 on every date",
                   "stock_futures": "simulated on the cash price with futures costs; lot = nsedata bhavcopy lot of "
                                    "2026-11-09 used for EVERY date incl. 2016-23 (labelled); no basis/roll",
                   "cost_model": COST_MODEL},
        "account": {"capital": K.CAPITAL, "compounded": False, "compounded_variant": "report only",
                    "risk_pct": K.RISK_PCT, "mis_leverage": K.MIS_LEVERAGE,
                    "futures_margin": "lots x lot x price x config.ai_agent_future_margin_pct (15 %) ~ SPAN+exposure; "
                                      "index orders capped at config.max_futures_order_value",
                    "cnc": "1x", "agents": ags, "config_values": cv,
                    "daily_loss": "config.max_daily_loss / config.total_capital x Rs 10 lakh (realised net, new "
                                  "entries only)",
                    "priority": "fill time; ties by previous-session traded value (desc); exits before entries at the "
                                "same minute",
                    "no_stop_fallback": "Rs 1 lakh notional (stocks), spec lots (index futures), 1 lot (stock futures)",
                    "stock_futures_lots_2026": lots},
        "protocol": {"stage1": "protocol.screen on the account's DISCOVERY trades (gates unchanged: H1 & H2 net > 0, "
                               "PF >= 1.15, >= 150 intraday / 60 positional trades, top symbol <= 35 %; index combos "
                               "exempt from the symbol gate as the lab's index futures)",
                     "final": "one look per passer (net > 0 and PF >= 1.05), tfgrid final ledger; never for failures",
                     "stage2": "protocol_min_oos.judge gates on 2016-01-01..2023-11-09 (H1 2016-19, H2 2020-23, PF >= "
                               "1.15, >= 150 / 60 trades per half, top symbol <= 35 %); dirty-link rule applied to "
                               "candidates before the account; every stage-1 passer, once",
                     "dsr": "trials.dsr on the stage-2 daily account P&L (non-annualised daily SR, skew, kurtosis): "
                            "N = full trial count of trials_v2.csv after appending this grid (all eras), and the fair "
                            "N = combos in this grid; V[SR] = robust (1.4826 MAD)^2 of the grid's stage-1 daily "
                            "Sharpes, floored at 1/(T-1)",
                     "proven": "stage-1 pass AND FINAL confirmed AND stage-2 pass AND DSR(full N) >= 0.95",
                     "excess": "long-biased survivors: net minus the same notional's NIFTY move over each trade, "
                               "and account return vs NIFTY buy-and-hold",
                     "trials_ledger": "every combo evaluated is appended (era 2023-26 stage tfgrid-screen / "
                                      "tfgrid-final; era 2016-23 stage tfgrid-oos); backup taken first"},
        "controls": {"what": "RANDOM entry (seeds 1..%d per timeframe, coin side, 1.5 ATR stop, 1:3) and PLANTED edges "
                             "(5 / 10 / 20 / 100 %% of entries take the side of the next 10 candles' move) through the "
                             "same engine + account on DISCOVERY, the first %d stocks" % (CONTROL_SEEDS, CONTROL_SYMBOLS),
                     "use": "engine bias check and the empirical luck rate of the account screen; not trials"},
        "live_strategies": {"ids": list(LIVE4), "accounts": "auction agent (AUC-FADE, MIS), attention agent "
                            "(SWING-ATTN-ON-3X, stock futures on cash), strangle agent (VRP winged, 2 lots); each "
                            "Rs 10 lakh with its config.validated_* limits (Rs 1 lakh per stock trade, max gross Rs 10 "
                            "lakh, its daily loss stop), own exits; both eras; report only (already judged)",
                            "rr_variant": "report-only 1:3 bracket on the same trades (AUC-FADE: 1.5 x 5-minute "
                                          "ATR(14) stop; ATTN: 1.5 x daily ATR(14); strangle: the defined risk is the "
                                          "wing loss, a 3x target exceeds the maximum profit (the credit): not "
                                          "applicable, identical)"},
        "portable_rules": {rid: {"family": r.family, "desc": r.desc, "own_stop": r.stop_desc,
                                 "needs_volume": r.needs_volume, "fingerprint": rule_fingerprint(rid)}
                           for rid, r in RULES.items()},
        "not_portable_modules": {
            "AUC-FADE-2PCT / AUC-FADE-3PCT / SCALP-AUC-FADE-1TO3-0925": "clock time: 09:15 call auction",
            "SWING-ATTN-ON-3X": "clock time + event (15:20 volume shock, overnight)",
            "VRP-STRANGLE-016-* / r3_options_* / r4_realopt_*": "expiry-cycle option structures on modelled / EOD premiums",
            "ltr / r3_crosssec / factor_momentum / r6_portfolio / r7_portfolio / r7_tps": "portfolio books with their own "
                "capital model (already judged); not re-run",
        },
        "grid_combos": grid,
        "native_combos": natives,
        "counts": {"rules": len(RULES), "grid_combos": len(grid), "native_combos": len(natives),
                   "by_tf": {tf: sum(1 for g in grid if g["tf"] == tf) for tf in TFS},
                   "by_agent": {a: sum(1 for g in grid if g["agent"] == a) + sum(1 for x in natives if x["agent"] == a)
                                for a in ("scalping", "intraday", "swing", "futures", "options")}},
        "code_sha256": code_sha(),
    }
    blob = json.dumps(doc, indent=1, sort_keys=True, default=str).encode()
    for p in (path, repo_copy):
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(blob)
    with open(path + ".sha256", "w") as fh:
        fh.write(hashlib.sha256(blob).hexdigest() + "\n")
    return doc


def load_prereg(path: str = PREREG) -> dict:
    with open(path, "rb") as fh:
        blob = fh.read()
    want = open(path + ".sha256").read().strip()
    if hashlib.sha256(blob).hexdigest() != want:
        raise SystemExit("pre-registration sha256 mismatch: refusing to run")
    doc = json.loads(blob)
    for g in doc["grid_combos"]:
        if g["fingerprint"] != rule_fingerprint(g["rule"]):
            raise SystemExit(f"{g['id']}: rule changed after the pre-registration: refusing to run")
    return doc


# ══════════════════════════════════════════════════════════════════════════
# stage 1 (DISCOVERY), FINAL, stage 2 (2016-23)
# ══════════════════════════════════════════════════════════════════════════
def _sim_task(args):
    cid, rid, tf, uni, agent = args
    rr = _W["cand"][rid]
    cols = rr["index"] if uni == "index" else rr["stocks"]
    if rr["errors"]:
        return cid, {"id": cid, "rule": rid, "tf": tf, "universe": uni, "agent": agent, "invalid": rr["errors"][:3]}, None
    res = run_account(cols, _W["syms"], uni, agent, _W["lots"], cid, dirty=_W.get("dirty"), nxt=_W.get("nxt"))
    lo, hi = _W["period"]
    met = K.metrics(res, _W["cal"], lo, hi)
    meta = {"rule": rid, "tf": tf, "universe": uni, "agent": agent}
    if _W["stage"] == "stage1":
        scr = judge_stage1(res, tf, uni == "index")
        row = summary_row(cid, meta, res, met, scr)
        row["daily"] = [round(x, 2) for x in K.daily_series(res, _W["cal"], lo, hi)]
        comp = run_account(cols, _W["syms"], uni, agent, _W["lots"], cid, compound=True)
        row["compounded"] = K.metrics(comp, _W["cal"], lo, hi)
        keep = res["trades"] if scr["passed"] else None
        return cid, row, keep
    return cid, {"meta": meta, "res": res, "met": met}, res["trades"]


def run_grid(stage: str, lo: int, hi: int, cache: str, combos_sel: list, jobs: int, out_dir: str, cal: list,
             dirty=None, nxt=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stocks = stock_symbols(cache)
    lots = lot_sizes(stocks)
    results = {}
    for tf in TFS:
        sel = [c for c in combos_sel if c[2] == tf]
        if not sel:
            continue
        rids = sorted({c[1] for c in sel})
        t0 = time.time()
        cand, syms, cpu = build_candidates(cache, tf, lo, hi, rids, jobs=jobs,
                                           with_index=any(c[3] == "index" for c in sel))
        ncand = sum(len(v["stocks"]["ekey"]) + len(v["index"]["ekey"]) for v in cand.values())
        log(f"  {stage} {tf}: {len(rids)} rules, {ncand:,} candidates built in {time.time() - t0:.0f}s (cpu {cpu:.0f}s)")
        _W.clear()
        _W.update({"cand": cand, "syms": syms, "lots": lots, "period": (lo, hi), "cal": cal, "stage": stage,
                   "dirty": dirty, "nxt": nxt})
        t1 = time.time()
        os.makedirs(os.path.join(out_dir, "trades"), exist_ok=True)
        with mp.get_context("fork").Pool(jobs) as pool:
            for cid, row, trades in pool.imap_unordered(_sim_task, [tuple(c) for c in sel]):
                results[cid] = row
                if trades is not None:
                    save_trades(os.path.join(out_dir, "trades", _fn(cid) + ".pkl.gz"), trades)
        log(f"  {stage} {tf}: {len(sel)} accounts simulated in {time.time() - t1:.0f}s")
        with gzip.open(os.path.join(out_dir, f"results_{tf}.json.gz"), "wt") as fh:
            json.dump({k: v for k, v in results.items() if k in {c[0] for c in sel}}, fh, default=_js)
        _W.clear()
        del cand
    return results


def save_trades(path: str, trades) -> None:
    with gzip.open(path, "wb") as fh:
        pickle.dump(list(trades), fh)


def load_trades(path: str) -> list:
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _fn(cid: str) -> str:
    return cid.replace("@", "_at_").replace(":", "__").replace("/", "_")


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return str(o)
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--amendment", default="")
    ap.add_argument("--jobs", type=int, default=4)
    a, _ = ap.parse_known_args(argv)
    if a.list:
        cs = combos()
        print(f"{len(RULES)} portable rules, {len(cs)} grid combos")
        for tf in TFS:
            print(f"  {tf}: {sum(1 for c in cs if c[2] == tf)}")
        for rid, r in RULES.items():
            print(f"  {rid:18} {r.family:14} stop: {r.stop_desc}")
        return 0
    if a.preregister:
        doc = preregister(amendment=a.amendment)
        print(json.dumps(doc["counts"], indent=1))
        print("written", PREREG, REPO_PREREG)
        return 0
    from . import tfgrid_run
    return tfgrid_run.main(argv)


if __name__ == "__main__":
    sys.exit(main())


# ══════════════════════════════════════════════════════════════════════════
# trials ledger rows (every combo evaluated is a trial)
# ══════════════════════════════════════════════════════════════════════════
TRIALS_PROTOCOL = "tfgrid capital 1:3 v1"


def trial_row(cid: str, *, era: str, stage: str, dataset: str, period, trades, daily, verdict: str, note: str = "",
              group: str = "", fp: str = "", source: str = "", trades_file: str = "") -> dict:
    """One trials.py ledger row for one account evaluation (daily = the account's daily realised P&L series)."""
    from . import trials as T
    from .sim import COST_MODEL
    nets = [float(t[12]) for t in trades]
    mt = T.moments(nets)
    md = T.moments(daily) if daily else T.moments([])
    gp, gl = sum(x for x in nets if x > 0), -sum(x for x in nets if x < 0)
    return {"strategy_id": cid, "family": "tfgrid-native" if cid.startswith("nat:") else "tfgrid", "group": group,
            "params_fp": fp, "era": era, "protocol": TRIALS_PROTOCOL, "stage": stage, "dataset": dataset,
            "period": f"{period[0]}-{period[1]}", "cost_model": COST_MODEL, "trades": len(nets),
            "net": round(sum(nets), 2), "pf": (round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0)),
            "mean_net": mt["mean"], "std_net": mt["std"], "sr_trade": mt["sr"],
            "mean_gross": (sum(float(t[10]) for t in trades) / len(trades)) if trades else None,
            "mean_cost": (sum(float(t[11]) for t in trades) / len(trades)) if trades else None,
            "n_days": md["n"], "sr_daily": md["sr"], "skew_daily": md["skew"], "kurt_daily": md["kurt"],
            "verdict": verdict, "source": source, "trades_file": trades_file, "note": note}


def append_trials(rows: list, ledger_path: str, backup: bool = True) -> int:
    from . import trials as T
    if backup and os.path.exists(ledger_path):
        b = ledger_path.replace(".csv", ".before_tfgrid.csv")
        if not os.path.exists(b):
            import shutil
            shutil.copy2(ledger_path, b)
    return T.TrialsLedger(ledger_path).append(rows)
