"""
Round-3 scalping candidates (pre-registered in scratchpad/r3/scalping/PREREGISTERED.txt before any run).

Large, fast dislocations that might clear the ~0.12 % MIS round trip: VWAP sigma snap-backs, volume climaxes with
rejection wicks, opening over-extension fades, stop runs around yesterday's extremes / opening range / round numbers,
NIFTY-shock divergence, squeeze breakouts and fake-outs, large-gap reversals, exhaustion runs, absorption, the MIS
square-off unwind and expiry-day reversion. Parameters are round textbook values fixed before any result.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from ..api import LONG, SHORT, Entry, between, cross_dn, cross_up, hhmm, prev_high, prev_low, runlen, shift, strategy

PAD = 0.0005          # stops 0.05 % beyond the stated extreme


def _each(mask, make):
    out = []
    for i in np.flatnonzero(mask):
        e = make(int(i))
        if e is not None:
            out.append(e)
    return out


def _roll_max(x, n):
    """max of x[i-n+1 .. i] (inclusive of i)."""
    from ..indicators import highest
    return highest(x, n)


def _roll_min(x, n):
    from ..indicators import lowest
    return lowest(x, n)


def _ok(x) -> bool:
    return x is not None and np.isfinite(x)


# ── vwap_snap ────────────────────────────────────────────────────────────────
def _vwap3_reentry(s, p, extra=None):
    win = between(s, "10:00", "14:45") & (s.vsd > 0)
    up_b = s.vwap + p.k * s.vsd
    dn_b = s.vwap - p.k * s.vsd
    far = p.k * s.vsd >= p.min_dist_pct / 100.0 * s.c
    pc = shift(s.c)
    pub, pdb = shift(up_b), shift(dn_b)
    sh = win & far & (pc > pub) & (s.c < up_b) & (s.c < s.o)
    lg = win & far & (pc < pdb) & (s.c > dn_b) & (s.c > s.o)
    if extra is not None:
        sh &= extra
        lg &= extra
    hh, ll = _roll_max(s.h, p.look), _roll_min(s.l, p.look)
    return (_each(sh, lambda i: Entry(i, SHORT, stop=float(hh[i]) * (1 + PAD),
                                      target=float(s.vwap[i] + p.target_k * s.vsd[i]), time_stop=p.hold))
            + _each(lg, lambda i: Entry(i, LONG, stop=float(ll[i]) * (1 - PAD),
                                        target=float(s.vwap[i] - p.target_k * s.vsd[i]), time_stop=p.hold)))


@strategy("r3_scalping_vwap3sd_reentry", "scalping", "vwap_snap",
          "After a close beyond VWAP+-3 sigma (>= 0.75 % away), first bearish/bullish close back inside: fade, stop 10-bar extreme, target VWAP+-1 sigma, 30 min",
          {"k": 3.0, "min_dist_pct": 0.75, "target_k": 1.0, "look": 10, "hold": 30})
def r3_scalping_vwap3sd_reentry(s, p):
    return _vwap3_reentry(s, p)


@strategy("r3_scalping_vwap25sd_climax", "scalping", "vwap_snap",
          "Close beyond VWAP+-2.5 sigma, >= 1 % from VWAP, on 3x volume: fade at once, stop bar extreme + 0.5 sigma, target VWAP+-1 sigma, 30 min",
          {"k": 2.5, "min_dist_pct": 1.0, "vr": 3.0, "stop_k": 0.5, "target_k": 1.0, "hold": 30})
def r3_scalping_vwap25sd_climax(s, p):
    win = between(s, "10:00", "14:45") & (s.vsd > 0) & (s.vr >= p.vr)
    dist = np.abs(s.c / s.vwap - 1.0) >= p.min_dist_pct / 100.0
    sh = win & dist & (s.c > s.vwap + p.k * s.vsd)
    lg = win & dist & (s.c < s.vwap - p.k * s.vsd)
    return (_each(sh, lambda i: Entry(i, SHORT, stop=float(s.h[i] + p.stop_k * s.vsd[i]),
                                      target=float(s.vwap[i] + p.target_k * s.vsd[i]), time_stop=p.hold))
            + _each(lg, lambda i: Entry(i, LONG, stop=float(s.l[i] - p.stop_k * s.vsd[i]),
                                        target=float(s.vwap[i] - p.target_k * s.vsd[i]), time_stop=p.hold)))


@strategy("r3_scalping_vwap3sd_idio", "scalping", "vwap_snap",
          "vwap3sd_reentry only when NIFTY moved < 0.20 % over the last 30 bars (stock-specific dislocation)",
          {"k": 3.0, "min_dist_pct": 0.75, "target_k": 1.0, "look": 10, "hold": 30, "nifty_n": 30,
           "nifty_max_pct": 0.20}, needs_index=True)
def r3_scalping_vwap3sd_idio(s, p):
    if s.nifty is None:
        return []
    nf = np.asarray(s.nifty, float)
    calm = np.abs(nf / shift(nf, p.nifty_n) - 1.0) < p.nifty_max_pct / 100.0
    return _vwap3_reentry(s, p, extra=calm)


# ── climax ───────────────────────────────────────────────────────────────────
def _climax_base(s, p):
    rng = s.h - s.l
    ap = shift(s.atr)
    return (between(s, "09:30", "14:45") & (s.vr >= p.vr) & (rng >= p.range_atr * ap)
            & (rng >= p.min_range_pct / 100.0 * s.c) & (rng > 0)), rng


def _hammer_masks(s, p):
    big, rng = _climax_base(s, p)
    pdl, pdh = shift(s.day_lo), shift(s.day_hi)
    rs = np.where(rng > 0, rng, 1.0)
    lw = (np.minimum(s.o, s.c) - s.l) / rs
    uw = (s.h - np.maximum(s.o, s.c)) / rs
    lg = big & (s.l < pdl) & (lw >= p.wick)
    sh = big & (s.h > pdh) & (uw >= p.wick)
    return lg, sh


@strategy("r3_scalping_climax_hammer", "scalping", "climax",
          "5x-volume, 2x-ATR (>= 0.4 %) bar making a new day low with a >= 60 % lower wick: buy (mirror), stop past the bar, 1.5R, 20 min",
          {"vr": 5.0, "range_atr": 2.0, "min_range_pct": 0.40, "wick": 0.60, "rr": 1.5, "hold": 20})
def r3_scalping_climax_hammer(s, p):
    lg, sh = _hammer_masks(s, p)
    return (_each(lg, lambda i: Entry(i, LONG, stop=float(s.l[i]) * (1 - PAD), rr=p.rr, time_stop=p.hold))
            + _each(sh, lambda i: Entry(i, SHORT, stop=float(s.h[i]) * (1 + PAD), rr=p.rr, time_stop=p.hold)))


@strategy("r3_scalping_climax_hammer_stretched", "scalping", "climax",
          "climax_hammer only when the close is beyond VWAP+-2 sigma",
          {"vr": 5.0, "range_atr": 2.0, "min_range_pct": 0.40, "wick": 0.60, "rr": 1.5, "hold": 20, "k": 2.0})
def r3_scalping_climax_hammer_stretched(s, p):
    lg, sh = _hammer_masks(s, p)
    lg &= (s.vsd > 0) & (s.c <= s.vwap - p.k * s.vsd)
    sh &= (s.vsd > 0) & (s.c >= s.vwap + p.k * s.vsd)
    return (_each(lg, lambda i: Entry(i, LONG, stop=float(s.l[i]) * (1 - PAD), rr=p.rr, time_stop=p.hold))
            + _each(sh, lambda i: Entry(i, SHORT, stop=float(s.h[i]) * (1 + PAD), rr=p.rr, time_stop=p.hold)))


@strategy("r3_scalping_climax_confirm", "scalping", "climax",
          "Selling climax (5x vol, 2x ATR, new day low, close in bottom third) then within 3 bars a close above its high: buy (mirror), stop lowest low since, 1.5R, 20 min",
          {"vr": 5.0, "range_atr": 2.0, "min_range_pct": 0.40, "within": 3, "rr": 1.5, "hold": 20})
def r3_scalping_climax_confirm(s, p):
    big, rng = _climax_base(s, p)
    pdl, pdh = shift(s.day_lo), shift(s.day_hi)
    rs = np.where(rng > 0, rng, 1.0)
    pos = (s.c - s.l) / rs
    cl_dn = np.flatnonzero(big & (s.l < pdl) & (pos <= 1.0 / 3.0))
    cl_up = np.flatnonzero(big & (s.h > pdh) & (pos >= 2.0 / 3.0))
    n = len(s)
    out = []
    for j in cl_dn:
        for i in range(j + 1, min(n, j + 1 + p.within)):
            if s.c[i] > s.h[j]:
                out.append(Entry(i, LONG, stop=float(s.l[j:i + 1].min()) * (1 - PAD), rr=p.rr, time_stop=p.hold))
                break
    for j in cl_up:
        for i in range(j + 1, min(n, j + 1 + p.within)):
            if s.c[i] < s.l[j]:
                out.append(Entry(i, SHORT, stop=float(s.h[j:i + 1].max()) * (1 + PAD), rr=p.rr, time_stop=p.hold))
                break
    return out


# ── open_ext ─────────────────────────────────────────────────────────────────
def _open_fade(s, p, at: str, need_reversal_bar: bool, nifty_ok=None):
    i = s.idx(hhmm(at))
    if i >= len(s) or s.t[i] != hhmm(at):
        return []
    o0, c = float(s.o[0]), float(s.c[i])
    r = c / o0 - 1.0
    if abs(r) < p.min_move_pct / 100.0:
        return []
    if nifty_ok is not None and not nifty_ok(i):
        return []
    tgt = c - p.retrace * (c - o0)
    if r > 0:
        if need_reversal_bar and not s.c[i] < s.o[i]:
            return []
        return [Entry(i, SHORT, stop=float(s.day_hi[i]) * (1 + p.stop_buf_pct / 100.0), target=tgt,
                      time_stop=p.hold)]
    if need_reversal_bar and not s.c[i] > s.o[i]:
        return []
    return [Entry(i, LONG, stop=float(s.day_lo[i]) * (1 - p.stop_buf_pct / 100.0), target=tgt, time_stop=p.hold)]


@strategy("r3_scalping_open5_fade", "scalping", "open_ext",
          "At 09:19 a move of >= 1.5 % from the 09:15 open is faded; stop day extreme +-0.25 %, target half retrace, 30 min",
          {"min_move_pct": 1.5, "stop_buf_pct": 0.25, "retrace": 0.5, "hold": 30}, max_per_day=1)
def r3_scalping_open5_fade(s, p):
    return _open_fade(s, p, "09:19", False)


@strategy("r3_scalping_open5_fade_idio", "scalping", "open_ext",
          "At 09:19 a >= 1.0 % move from the open while NIFTY moved < 0.30 % since 09:15 is faded; stop day extreme +-0.25 %, half retrace, 30 min",
          {"min_move_pct": 1.0, "nifty_max_pct": 0.30, "stop_buf_pct": 0.25, "retrace": 0.5, "hold": 30},
          max_per_day=1, needs_index=True)
def r3_scalping_open5_fade_idio(s, p):
    if s.nifty is None:
        return []
    nf = s.nifty

    def calm(i):
        return _ok(nf[0]) and _ok(nf[i]) and abs(nf[i] / nf[0] - 1.0) < p.nifty_max_pct / 100.0
    return _open_fade(s, p, "09:19", False, calm)


@strategy("r3_scalping_open15_fade", "scalping", "open_ext",
          "At 09:29 a >= 2 % move from the open whose 09:29 bar closes against it is faded; stop day extreme +-0.25 %, half retrace, 30 min",
          {"min_move_pct": 2.0, "stop_buf_pct": 0.25, "retrace": 0.5, "hold": 30}, max_per_day=1)
def r3_scalping_open15_fade(s, p):
    return _open_fade(s, p, "09:29", True)


# ── stoprun ──────────────────────────────────────────────────────────────────
@strategy("r3_scalping_pdl_sweep", "scalping", "stoprun",
          "Open inside yesterday's range; low sweeps >= 0.2 % under PDL then a close back above PDL on 1.5x vol: buy (mirror PDH), stop 10-bar low, 2R, 30 min",
          {"sweep_pct": 0.20, "look": 10, "vr": 1.5, "rr": 2.0, "hold": 30})
def r3_scalping_pdl_sweep(s, p):
    if not (_ok(s.pdl) and _ok(s.pdh)) or not (s.pdl < s.o[0] < s.pdh):
        return []
    win = between(s, "09:30", "14:45") & (s.vr >= p.vr)
    ll, hh = _roll_min(s.l, p.look), _roll_max(s.h, p.look)
    pc = shift(s.c)
    lg = win & (ll <= s.pdl * (1 - p.sweep_pct / 100.0)) & (s.c > s.pdl) & (pc <= s.pdl)
    sh = win & (hh >= s.pdh * (1 + p.sweep_pct / 100.0)) & (s.c < s.pdh) & (pc >= s.pdh)
    return (_each(lg, lambda i: Entry(i, LONG, stop=float(ll[i]) * (1 - PAD), rr=p.rr, time_stop=p.hold))
            + _each(sh, lambda i: Entry(i, SHORT, stop=float(hh[i]) * (1 + PAD), rr=p.rr, time_stop=p.hold)))


@strategy("r3_scalping_orb30_sweep", "scalping", "stoprun",
          "After 10:00 a poke >= 0.15 % beyond the 30-minute range that closes back inside within 5 bars: fade, stop 5-bar extreme, 2R, 30 min",
          {"orb": 30, "poke_pct": 0.15, "look": 5, "rr": 2.0, "hold": 30})
def r3_scalping_orb30_sweep(s, p):
    hi, lo, k = s.orb(p.orb)
    if k >= len(s) or not hi > lo:
        return []
    win = between(s, "10:00", "14:45")
    hh, ll = _roll_max(s.h, p.look), _roll_min(s.l, p.look)
    pc = shift(s.c)
    up_poke, dn_poke = hi * (1 + p.poke_pct / 100.0), lo * (1 - p.poke_pct / 100.0)
    sh = win & (hh >= up_poke) & (s.c < hi) & ((pc >= hi) | (s.h >= up_poke))
    lg = win & (ll <= dn_poke) & (s.c > lo) & ((pc <= lo) | (s.l <= dn_poke))
    return (_each(sh, lambda i: Entry(i, SHORT, stop=float(hh[i]) * (1 + PAD), rr=p.rr, time_stop=p.hold))
            + _each(lg, lambda i: Entry(i, LONG, stop=float(ll[i]) * (1 - PAD), rr=p.rr, time_stop=p.hold)))


def _round_step(price: float, grid_pct: float) -> float:
    want = price * grid_pct / 100.0
    if not want > 0:
        return np.nan
    e = np.floor(np.log10(want))
    cands = [m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1.0, 2.0, 5.0)]
    return float(min(cands, key=lambda x: abs(np.log(x / want))))


@strategy("r3_scalping_round_sweep", "scalping", "stoprun",
          "Bar pokes >= 0.10 % through the nearest round level (1-2-5 grid ~1 % of price) and closes back on the other side on 2x vol: fade, stop bar extreme, 2R, 20 min",
          {"grid_pct": 1.0, "poke_pct": 0.10, "vr": 2.0, "rr": 2.0, "hold": 20})
def r3_scalping_round_sweep(s, p):
    if not _ok(s.pdc):
        return []
    step = _round_step(s.pdc, p.grid_pct)
    if not _ok(step):
        return []
    L = np.round(s.c / step) * step
    win = between(s, "09:30", "14:45") & (s.vr >= p.vr)
    lg = win & (s.l <= L * (1 - p.poke_pct / 100.0)) & (s.c > L) & (s.c > s.o)
    sh = win & (s.h >= L * (1 + p.poke_pct / 100.0)) & (s.c < L) & (s.c < s.o)
    return (_each(lg, lambda i: Entry(i, LONG, stop=float(s.l[i]) * (1 - PAD), rr=p.rr, time_stop=p.hold))
            + _each(sh, lambda i: Entry(i, SHORT, stop=float(s.h[i]) * (1 + PAD), rr=p.rr, time_stop=p.hold)))


# ── nifty_div ────────────────────────────────────────────────────────────────
def _shock(s, p):
    nf = np.asarray(s.nifty, float)
    nr = nf / shift(nf, p.n) - 1.0
    sr = s.c / shift(s.c, p.n) - 1.0
    win = between(s, "09:30", "14:45")
    th = p.shock_pct / 100.0
    return win, nr, sr, th


def _pct_entry(i, side, c, p):
    return Entry(i, side, stop=c * (1 - side * p.stop_pct / 100.0), target=c * (1 + side * p.target_pct / 100.0),
                 time_stop=p.hold)


@strategy("r3_scalping_nifty_shock_resist", "scalping", "nifty_div",
          "NIFTY falls >= 0.35 % in 5 bars while the stock is flat/up and above VWAP: buy (mirror), stop 0.5 %, target 0.75 %, 30 min",
          {"n": 5, "shock_pct": 0.35, "stop_pct": 0.50, "target_pct": 0.75, "hold": 30}, needs_index=True)
def r3_scalping_nifty_shock_resist(s, p):
    if s.nifty is None:
        return []
    win, nr, sr, th = _shock(s, p)
    lg = win & (nr <= -th) & (sr >= 0) & (s.c > s.vwap)
    sh = win & (nr >= th) & (sr <= 0) & (s.c < s.vwap)
    return (_each(lg, lambda i: _pct_entry(i, LONG, float(s.c[i]), p))
            + _each(sh, lambda i: _pct_entry(i, SHORT, float(s.c[i]), p)))


@strategy("r3_scalping_nifty_shock_catchup", "scalping", "nifty_div",
          "NIFTY falls >= 0.35 % in 5 bars while the stock is flat/up: short it to catch up (mirror), stop 0.5 %, target 0.75 %, 15 min",
          {"n": 5, "shock_pct": 0.35, "stop_pct": 0.50, "target_pct": 0.75, "hold": 15}, needs_index=True)
def r3_scalping_nifty_shock_catchup(s, p):
    if s.nifty is None:
        return []
    win, nr, sr, th = _shock(s, p)
    sh = win & (nr <= -th) & (sr >= 0)
    lg = win & (nr >= th) & (sr <= 0)
    return (_each(lg, lambda i: _pct_entry(i, LONG, float(s.c[i]), p))
            + _each(sh, lambda i: _pct_entry(i, SHORT, float(s.c[i]), p)))


@strategy("r3_scalping_nifty_shock_overshoot", "scalping", "nifty_div",
          "NIFTY falls >= 0.35 % in 5 bars and the stock falls >= 3x as much: buy (mirror), stop 5-bar low - 0.1 %, target half retrace, 20 min",
          {"n": 5, "shock_pct": 0.35, "mult": 3.0, "stop_buf_pct": 0.10, "retrace": 0.5, "hold": 20},
          needs_index=True)
def r3_scalping_nifty_shock_overshoot(s, p):
    if s.nifty is None:
        return []
    win, nr, sr, th = _shock(s, p)
    lg = win & (nr <= -th) & (sr <= p.mult * nr)
    sh = win & (nr >= th) & (sr >= p.mult * nr)
    ll, hh = _roll_min(s.l, p.n), _roll_max(s.h, p.n)
    c0 = shift(s.c, p.n)
    b = p.stop_buf_pct / 100.0
    return (_each(lg, lambda i: Entry(i, LONG, stop=float(ll[i]) * (1 - b),
                                      target=float(s.c[i] + p.retrace * (c0[i] - s.c[i])), time_stop=p.hold))
            + _each(sh, lambda i: Entry(i, SHORT, stop=float(hh[i]) * (1 + b),
                                        target=float(s.c[i] - p.retrace * (s.c[i] - c0[i])), time_stop=p.hold)))


# ── squeeze ──────────────────────────────────────────────────────────────────
def _box(s, p):
    bh, bl = prev_high(s.h, p.n), prev_low(s.l, p.n)
    tight = (bh - bl) <= p.max_width_pct / 100.0 * s.c
    return bh, bl, tight


def _squeeze_break(s, p, day_pct=None):
    bh, bl, tight = _box(s, p)
    win = between(s, "10:15", "14:30") & tight & (s.vr >= p.vr)
    lg = win & (s.c > bh)
    sh = win & (s.c < bl)
    if day_pct is not None:
        r = s.c / s.o[0] - 1.0
        lg &= r >= day_pct / 100.0
        sh &= r <= -day_pct / 100.0
    return (_each(lg, lambda i: Entry(i, LONG, stop=float(bl[i]), rr=p.rr, time_stop=p.hold))
            + _each(sh, lambda i: Entry(i, SHORT, stop=float(bh[i]), rr=p.rr, time_stop=p.hold)))


@strategy("r3_scalping_squeeze_break", "scalping", "squeeze",
          "30-bar box narrower than 0.4 % then a close beyond it on 3x volume: follow, stop other side of the box, 2R, 30 min",
          {"n": 30, "max_width_pct": 0.40, "vr": 3.0, "rr": 2.0, "hold": 30})
def r3_scalping_squeeze_break(s, p):
    return _squeeze_break(s, p)


@strategy("r3_scalping_squeeze_break_daydir", "scalping", "squeeze",
          "squeeze_break only in the direction of the day (stock >= +0.5 % from open for longs, <= -0.5 % for shorts)",
          {"n": 30, "max_width_pct": 0.40, "vr": 3.0, "rr": 2.0, "hold": 30, "day_pct": 0.50})
def r3_scalping_squeeze_break_daydir(s, p):
    return _squeeze_break(s, p, p.day_pct)


@strategy("r3_scalping_squeeze_fakeout", "scalping", "squeeze",
          "Close beyond a < 0.4 % 30-bar box that returns inside within 3 bars: fade, stop extreme since the break, target the other side, 30 min",
          {"n": 30, "max_width_pct": 0.40, "within": 3, "hold": 30})
def r3_scalping_squeeze_fakeout(s, p):
    bh, bl, tight = _box(s, p)
    win = between(s, "10:15", "14:30")
    n = len(s)
    out = []
    for j in np.flatnonzero(win & tight & (s.c > bh)):
        for i in range(j + 1, min(n, j + 1 + p.within)):
            if s.c[i] < bh[j]:
                out.append(Entry(i, SHORT, stop=float(s.h[j:i + 1].max()) * (1 + PAD), target=float(bl[j]),
                                 time_stop=p.hold))
                break
    for j in np.flatnonzero(win & tight & (s.c < bl)):
        for i in range(j + 1, min(n, j + 1 + p.within)):
            if s.c[i] > bl[j]:
                out.append(Entry(i, LONG, stop=float(s.l[j:i + 1].min()) * (1 - PAD), target=float(bh[j]),
                                 time_stop=p.hold))
                break
    return out


# ── gap ──────────────────────────────────────────────────────────────────────
def _gap_dir(s, p):
    g = s.gap
    if not _ok(g):
        return 0
    a = abs(g) * 100.0
    if p.gap_min_pct <= a <= p.gap_max_pct:
        return 1 if g > 0 else -1
    return 0


@strategy("r3_scalping_gap2_lowerlow", "scalping", "gap",
          "Gap of 2-10 %: 09:20-10:30 first close beyond the previous 5-bar extreme against the gap: fade, stop day extreme, target half gap fill, 30 min",
          {"gap_min_pct": 2.0, "gap_max_pct": 10.0, "n": 5, "hold": 30}, max_per_day=1)
def r3_scalping_gap2_lowerlow(s, p):
    d = _gap_dir(s, p)
    if d == 0:
        return []
    win = between(s, "09:20", "10:30")
    half = (float(s.o[0]) + s.pdc) / 2.0
    if d > 0:
        i = int(np.argmax(win & (s.c < prev_low(s.l, p.n)))) if (win & (s.c < prev_low(s.l, p.n))).any() else -1
        return [] if i < 0 else [Entry(i, SHORT, stop=float(s.day_hi[i]) * (1 + PAD), target=half, time_stop=p.hold)]
    m = win & (s.c > prev_high(s.h, p.n))
    i = int(np.argmax(m)) if m.any() else -1
    return [] if i < 0 else [Entry(i, LONG, stop=float(s.day_lo[i]) * (1 - PAD), target=half, time_stop=p.hold)]


@strategy("r3_scalping_gap2_vwap_loss", "scalping", "gap",
          "Gap of 2-10 %: 09:30-11:00 first close crossing VWAP against the gap: fade, stop 10-bar extreme, 2R, 30 min",
          {"gap_min_pct": 2.0, "gap_max_pct": 10.0, "look": 10, "rr": 2.0, "hold": 30}, max_per_day=1)
def r3_scalping_gap2_vwap_loss(s, p):
    d = _gap_dir(s, p)
    if d == 0:
        return []
    win = between(s, "09:30", "11:00")
    if d > 0:
        m = win & cross_dn(s.c, s.vwap)
        if not m.any():
            return []
        i = int(np.argmax(m))
        return [Entry(i, SHORT, stop=float(s.h[max(0, i - p.look + 1):i + 1].max()) * (1 + PAD), rr=p.rr,
                      time_stop=p.hold)]
    m = win & cross_up(s.c, s.vwap)
    if not m.any():
        return []
    i = int(np.argmax(m))
    return [Entry(i, LONG, stop=float(s.l[max(0, i - p.look + 1):i + 1].min()) * (1 - PAD), rr=p.rr,
                  time_stop=p.hold)]


# ── time_of_day ──────────────────────────────────────────────────────────────
@strategy("r3_scalping_late_vwap_revert", "scalping", "time_of_day",
          "At 14:55 a stock >= 1 % and >= 2 sigma from VWAP is faded toward VWAP ahead of the MIS square-off; stop 0.75 %, 20 min",
          {"min_dist_pct": 1.0, "k": 2.0, "stop_pct": 0.75, "hold": 20}, max_per_day=1)
def r3_scalping_late_vwap_revert(s, p):
    i = s.idx(hhmm("14:55"))
    if i >= len(s) or s.t[i] != hhmm("14:55"):
        return []
    c, vw, sd = float(s.c[i]), float(s.vwap[i]), float(s.vsd[i])
    if not sd > 0 or abs(c / vw - 1.0) < p.min_dist_pct / 100.0 or abs(c - vw) < p.k * sd:
        return []
    side = SHORT if c > vw else LONG
    return [Entry(i, side, stop=c * (1 - side * p.stop_pct / 100.0), target=vw, time_stop=p.hold)]


@strategy("r3_scalping_lunch_spike_fade", "scalping", "time_of_day",
          "12:00-13:30 a 5x-volume, 3x-ATR (>= 0.4 %) bar with a >= 60 % body is faded; stop past the bar, target its midpoint, 20 min",
          {"vr": 5.0, "range_atr": 3.0, "min_range_pct": 0.40, "body": 0.60, "hold": 20})
def r3_scalping_lunch_spike_fade(s, p):
    rng = s.h - s.l
    rs = np.where(rng > 0, rng, 1.0)
    big = (between(s, "12:00", "13:30") & (s.vr >= p.vr) & (rng >= p.range_atr * shift(s.atr))
           & (rng >= p.min_range_pct / 100.0 * s.c) & (np.abs(s.c - s.o) / rs >= p.body) & (rng > 0))
    mid = (s.h + s.l) / 2.0
    return (_each(big & (s.c > s.o), lambda i: Entry(i, SHORT, stop=float(s.h[i]) * (1 + PAD), target=float(mid[i]),
                                                     time_stop=p.hold))
            + _each(big & (s.c < s.o), lambda i: Entry(i, LONG, stop=float(s.l[i]) * (1 - PAD), target=float(mid[i]),
                                                       time_stop=p.hold)))


# ── exhaustion ───────────────────────────────────────────────────────────────
@strategy("r3_scalping_run7_fade", "scalping", "exhaustion",
          "7 consecutive higher 1-minute closes adding >= 1 %: short (mirror), stop 7-bar high + 0.2 %, target half retrace, 20 min",
          {"n": 7, "min_move_pct": 1.0, "stop_buf_pct": 0.20, "retrace": 0.5, "hold": 20})
def r3_scalping_run7_fade(s, p):
    pc = shift(s.c)
    up_run, dn_run = runlen(s.c > pc), runlen(s.c < pc)
    c0 = shift(s.c, p.n)
    mv = s.c / c0 - 1.0
    win = between(s, "09:30", "14:45")
    sh = win & (up_run >= p.n) & (mv >= p.min_move_pct / 100.0)
    lg = win & (dn_run >= p.n) & (mv <= -p.min_move_pct / 100.0)
    hh, ll = _roll_max(s.h, p.n), _roll_min(s.l, p.n)
    b = p.stop_buf_pct / 100.0
    return (_each(sh, lambda i: Entry(i, SHORT, stop=float(hh[i]) * (1 + b),
                                      target=float(s.c[i] - p.retrace * (s.c[i] - c0[i])), time_stop=p.hold))
            + _each(lg, lambda i: Entry(i, LONG, stop=float(ll[i]) * (1 - b),
                                        target=float(s.c[i] + p.retrace * (c0[i] - s.c[i])), time_stop=p.hold)))


@strategy("r3_scalping_5m_bigbar_fade", "scalping", "exhaustion",
          "Completed 5-minute bar with range >= 3x prior 5m ATR, body >= 1 % and >= 70 % of range: fade, stop extreme +-0.1 %, target half the body, 30 min",
          {"range_atr": 3.0, "min_body_pct": 1.0, "body": 0.70, "stop_buf_pct": 0.10, "retrace": 0.5, "hold": 30})
def r3_scalping_5m_bigbar_fade(s, p):
    b = s.bars5()
    out = []
    lo_t, hi_t = hhmm("09:30"), hhmm("14:45")
    bf = p.stop_buf_pct / 100.0
    for k in range(len(b.end)):
        e = int(b.end[k])
        if e < 1 or not (lo_t <= s.t[e] < hi_t):
            continue
        o5, h5, l5, c5 = float(b.o[k]), float(b.h[k]), float(b.l[k]), float(b.c[k])
        rng, body = h5 - l5, c5 - o5
        a5 = float(s.atr5[e - 1])        # ATR of the 5-minute bars completed BEFORE this one
        if not (rng > 0 and _ok(a5) and a5 > 0):
            continue
        if rng < p.range_atr * a5 or abs(body) < p.min_body_pct / 100.0 * o5 or abs(body) < p.body * rng:
            continue
        tgt = c5 - p.retrace * body
        if body > 0:
            out.append(Entry(e, SHORT, stop=h5 * (1 + bf), target=tgt, time_stop=p.hold))
        else:
            out.append(Entry(e, LONG, stop=l5 * (1 - bf), target=tgt, time_stop=p.hold))
    return out


@strategy("r3_scalping_datr_exhaust", "scalping", "exhaustion",
          "After 10:30 with the day range >= 2x daily ATR: previous bar made the day high and this bar closes under its low in the upper half: short (mirror), stop day high, 1.5R, 30 min",
          {"datr_mult": 2.0, "rr": 1.5, "hold": 30})
def r3_scalping_datr_exhaust(s, p):
    if not _ok(s.datr) or not s.datr > 0:
        return []
    win = between(s, "10:30", "14:45") & ((s.day_hi - s.day_lo) >= p.datr_mult * s.datr)
    mid = (s.day_hi + s.day_lo) / 2.0
    ph, pl = shift(s.h), shift(s.l)
    pdh_run, pdl_run = shift(s.day_hi), shift(s.day_lo)
    sh = win & (ph >= pdh_run) & (s.c < pl) & (s.c > mid)
    lg = win & (pl <= pdl_run) & (s.c > ph) & (s.c < mid)
    return (_each(sh, lambda i: Entry(i, SHORT, stop=float(s.day_hi[i]) * (1 + PAD), rr=p.rr, time_stop=p.hold))
            + _each(lg, lambda i: Entry(i, LONG, stop=float(s.day_lo[i]) * (1 - PAD), rr=p.rr, time_stop=p.hold)))


@strategy("r3_scalping_rsi5_extreme", "scalping", "exhaustion_osc",
          "5-minute RSI <= 15 with the stock >= 1 % below its open, then a close above the previous bar's high: buy (mirror), stop day low, 1.5R, 30 min",
          {"lo": 15, "hi": 85, "day_pct": 1.0, "rr": 1.5, "hold": 30})
def r3_scalping_rsi5_extreme(s, p):
    win = between(s, "10:00", "14:45")
    r = s.c / s.o[0] - 1.0
    lg = win & (s.rsi5 <= p.lo) & (r <= -p.day_pct / 100.0) & (s.c > shift(s.h))
    sh = win & (s.rsi5 >= p.hi) & (r >= p.day_pct / 100.0) & (s.c < shift(s.l))
    return (_each(lg, lambda i: Entry(i, LONG, stop=float(s.day_lo[i]) * (1 - PAD), rr=p.rr, time_stop=p.hold))
            + _each(sh, lambda i: Entry(i, SHORT, stop=float(s.day_hi[i]) * (1 + PAD), rr=p.rr, time_stop=p.hold)))


# ── gap_fill ─────────────────────────────────────────────────────────────────
@strategy("r3_scalping_pdc_gapfill_bounce", "scalping", "gap_fill",
          "Gap of 1-4 %: the first touch of yesterday's close (09:30-13:00) that closes back on the gap side: trade the bounce, stop 0.4 % past PDC, 2R, 30 min",
          {"gap_min_pct": 1.0, "gap_max_pct": 4.0, "stop_pct": 0.40, "rr": 2.0, "hold": 30}, max_per_day=1)
def r3_scalping_pdc_gapfill_bounce(s, p):
    d = _gap_dir(s, p)
    if d == 0:
        return []
    win = between(s, "09:30", "13:00")
    pc = s.pdc
    if d > 0:
        first = (s.l <= pc) & (shift(s.day_lo) > pc)
        m = win & first & (s.c > pc)
        if not m.any():
            return []
        i = int(np.argmax(m))
        return [Entry(i, LONG, stop=pc * (1 - p.stop_pct / 100.0), rr=p.rr, time_stop=p.hold)]
    first = (s.h >= pc) & (shift(s.day_hi) < pc)
    m = win & first & (s.c < pc)
    if not m.any():
        return []
    i = int(np.argmax(m))
    return [Entry(i, SHORT, stop=pc * (1 + p.stop_pct / 100.0), rr=p.rr, time_stop=p.hold)]


# ── volume ───────────────────────────────────────────────────────────────────
@strategy("r3_scalping_absorption", "scalping", "volume",
          "8x-volume bar with a small body and no range expansion at a new day high, stock >= 1 % up on the day: short (mirror), stop day high + 0.1 %, 2R, 20 min",
          {"vr": 8.0, "body_max": 0.25, "range_atr_max": 1.5, "day_pct": 1.0, "stop_buf_pct": 0.10, "rr": 2.0,
           "hold": 20})
def r3_scalping_absorption(s, p):
    rng = s.h - s.l
    rs = np.where(rng > 0, rng, 1.0)
    r = s.c / s.o[0] - 1.0
    base = (between(s, "09:45", "14:45") & (s.vr >= p.vr) & (rng > 0) & (np.abs(s.c - s.o) / rs <= p.body_max)
            & (rng <= p.range_atr_max * shift(s.atr)))
    sh = base & (s.h > shift(s.day_hi)) & (r >= p.day_pct / 100.0)
    lg = base & (s.l < shift(s.day_lo)) & (r <= -p.day_pct / 100.0)
    b = p.stop_buf_pct / 100.0
    return (_each(sh, lambda i: Entry(i, SHORT, stop=float(s.day_hi[i]) * (1 + b), rr=p.rr, time_stop=p.hold))
            + _each(lg, lambda i: Entry(i, LONG, stop=float(s.day_lo[i]) * (1 - b), rr=p.rr, time_stop=p.hold)))


# ── calendar ─────────────────────────────────────────────────────────────────
def _is_monthly_expiry(yyyymmdd: int) -> bool:
    """Last Thursday of the month before 2025-09-01, last Tuesday from then (holiday shifts not detected)."""
    d = _dt.date(yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100)
    wd = 3 if yyyymmdd < 20250901 else 1
    if d.weekday() != wd:
        return False
    return (d + _dt.timedelta(days=7)).month != d.month


@strategy("r3_scalping_expiry_vwap_fade", "scalping", "calendar",
          "Monthly F&O expiry day 13:30-15:00: a close beyond VWAP+-2 sigma and >= 0.75 % from VWAP is faded; stop 0.6 %, target VWAP, 30 min",
          {"k": 2.0, "min_dist_pct": 0.75, "stop_pct": 0.60, "hold": 30})
def r3_scalping_expiry_vwap_fade(s, p):
    if not _is_monthly_expiry(int(s.date)):
        return []
    win = between(s, "13:30", "15:00") & (s.vsd > 0) & (np.abs(s.c / s.vwap - 1.0) >= p.min_dist_pct / 100.0)
    sh = win & (s.c > s.vwap + p.k * s.vsd)
    lg = win & (s.c < s.vwap - p.k * s.vsd)
    st = p.stop_pct / 100.0
    return (_each(sh, lambda i: Entry(i, SHORT, stop=float(s.c[i]) * (1 + st), target=float(s.vwap[i]),
                                      time_stop=p.hold))
            + _each(lg, lambda i: Entry(i, LONG, stop=float(s.c[i]) * (1 - st), target=float(s.vwap[i]),
                                        time_stop=p.hold)))
