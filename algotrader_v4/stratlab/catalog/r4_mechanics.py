"""
Round-4 NSE-specific candidates, family NSE MECHANICS AND CALENDAR (24 strategies, ids r4_mechanics_*).
Pre-registered BEFORE any result in scratchpad/r4/mechanics/PREREGISTERED.txt - parameters are round / textbook
values fixed there; do not tune them. Every variant is one more candidate that can pass by luck.

NSE EOD data comes ONLY through stratlab.nsedata's as-of helpers:
  * intraday: N.asof_session(s) -> rows dated <= previous F&O session; nothing dated s.date is read.
  * next-open positional: N.asof_swing(d) -> rows dated <= d.date[-1] (published that evening, fill next open).
  * decision_time="15:20" positional strategies read no NSE EOD file (calendar only).
_guard_rows() re-checks every Rows / Chain read and raises LookAheadError if anything is dated at/after the date
it must precede. Calendars (holidays, Muhurat, Budget, RBI MPC) are hard-coded public schedules (see PREREGISTERED).
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from .. import nsedata as N
from ..api import LONG, SHORT, Entry, Swing, hhmm, strategy

# ── public calendars (sources in PREREGISTERED.txt) ─────────────────────────
HOLIDAYS = frozenset((
    20231114, 20231127, 20231225, 20240122, 20240126, 20240308, 20240325, 20240329, 20240411, 20240417, 20240501,
    20240520, 20240617, 20240717, 20240815, 20241002, 20241101, 20241115, 20241120, 20241225, 20250226, 20250314,
    20250331, 20250410, 20250414, 20250418, 20250501, 20250815, 20250827, 20251002, 20251021, 20251022, 20251105,
    20251225, 20260115, 20260126, 20260303, 20260326, 20260331, 20260403, 20260414, 20260501, 20260528, 20260626,
    20260914, 20261002, 20261020, 20261110))
MUHURAT = (20231112, 20241101, 20251021, 20261108)
BUDGET = frozenset((20240201, 20240723, 20250201, 20260201))
MPC = frozenset((20231208, 20240208, 20240405, 20240607, 20240808, 20241009, 20241206, 20250207, 20250409,
                 20250606, 20250806, 20251001, 20251205, 20260206, 20260408, 20260605, 20260805, 20261007))
RATE_SENS = frozenset(("AXISBANK", "BANKBARODA", "CANBK", "HDFCBANK", "ICICIBANK", "INDUSINDBK", "KOTAKBANK", "SBIN",
                       "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "RECLTD", "MARUTI", "M&M", "BAJAJ-AUTO",
                       "HEROMOTOCO", "EICHERMOT", "TVSMOTOR", "DLF", "LODHA"))
CAL_END = 20261112        # last date the hard-coded holiday list covers

HYG_N, HYG_JUMP = 130, 0.25
_CA_GAPS = {("SIEMENS", 20250407), ("VEDL", 20260430)}


def _d(x: int) -> _dt.date:
    x = int(x)
    return _dt.date(x // 10000, x // 100 % 100, x % 100)


def _i(x: _dt.date) -> int:
    return x.year * 10000 + x.month * 100 + x.day


def _regular(x: _dt.date) -> bool:
    return x.weekday() < 5 and _i(x) not in HOLIDAYS


def _next_regular(day: int) -> int | None:
    x = _d(day) + _dt.timedelta(1)
    for _ in range(15):
        if _i(x) > CAL_END:
            return None
        if _regular(x):
            return _i(x)
        x += _dt.timedelta(1)
    return None


def _prev_regular(day: int) -> int:
    x = _d(day) - _dt.timedelta(1)
    while not _regular(x):
        x -= _dt.timedelta(1)
    return _i(x)


def _regular_between(a: int, b: int) -> int:
    """Regular sessions strictly after a and up to and including b."""
    x, n, end = _d(a) + _dt.timedelta(1), 0, _d(b)
    while x <= end:
        n += _regular(x)
        x += _dt.timedelta(1)
    return n


def _is_preholiday(day: int) -> bool:
    """The next weekday after `day` is a weekday trading holiday."""
    x = _d(day) + _dt.timedelta(1)
    while x.weekday() >= 5:
        x += _dt.timedelta(1)
    return _i(x) <= CAL_END and _i(x) in HOLIDAYS


def _is_month_last(day: int) -> bool:
    nx = _next_regular(day)
    return nx is not None and nx // 100 != int(day) // 100


# ── guards ──────────────────────────────────────────────────────────────────
def _guard_rows(rows, must_precede: int, what: str):
    """Explicit look-ahead guard: every row must be dated strictly before `must_precede`."""
    if rows is None:
        return None
    dates = np.atleast_1d(rows["date"])
    if len(dates) and int(np.max(dates)) >= int(must_precede):
        raise N.LookAheadError(f"r4_mechanics: {what} returned data dated {int(np.max(dates))} "
                               f">= {int(must_precede)} (decision/fill date)")
    return rows


def _prev_fo(day: int) -> int | None:
    f = N.fo_sessions(before=int(day))
    return int(f[-1]) if len(f) else None


def _clean(d) -> bool:
    c, o = d.c, d.o
    n = min(len(c), HYG_N + 1)
    if n >= 2:
        cc, oo = c[-n:], o[-n:]
        pc = cc[:-1]
        if np.any(np.abs(cc[1:] / pc - 1.0) > HYG_JUMP) or np.any(np.abs(oo[1:] / pc - 1.0) > HYG_JUMP):
            return False
    recent = d.date[-250:]
    for s_, day in _CA_GAPS:
        if s_ == d.symbol and len(recent) and recent[0] <= day <= recent[-1]:
            return False
    return True


def _at(s, t: str) -> int:
    """Index of the bar starting at minute t, or -1 if it is not in the session."""
    i = s.idx(hhmm(t))
    if i >= len(s) or int(s.t[i]) != hhmm(t):
        return -1
    return int(i)


# ════════════════════════════════════════════════════════════════════════════
# A. Closing-session VWAP
# ════════════════════════════════════════════════════════════════════════════
def _close_vwap(s, p):
    i = _at(s, p.at)
    if i < 0:
        return []
    c, vw, sd = float(s.c[i]), float(s.vwap[i]), float(s.vsd[i])
    if not (np.isfinite(vw) and vw > 0):
        return []
    if p.mode == "sd":
        if not (np.isfinite(sd) and sd > 0):
            return []
        if c > vw + p.k * sd:
            return [Entry(i, SHORT, stop=c + p.stop_sd * sd, target=vw)]
        if c < vw - p.k * sd:
            return [Entry(i, LONG, stop=c - p.stop_sd * sd, target=vw)]
        return []
    dev = c / vw - 1.0
    if dev > p.dev / 100.0:
        return [Entry(i, SHORT, stop=c * (1 + p.stop / 100.0), target=vw)]
    if dev < -p.dev / 100.0:
        return [Entry(i, LONG, stop=c * (1 - p.stop / 100.0), target=vw)]
    return []


strategy("r4_mechanics_close_vwap_2sd", "intraday", "mech_close_vwap",
         "14:59 close beyond day VWAP +-2 sigma -> fade toward VWAP into the official-close window; stop 1 sigma, sq-off",
         {"mode": "sd", "k": 2.0, "stop_sd": 1.0, "at": "14:59"})(_close_vwap)
strategy("r4_mechanics_close_vwap_1pct", "intraday", "mech_close_vwap",
         "14:59 close > 1 % from day VWAP -> fade toward VWAP into the official-close window; stop 0.5 %, sq-off",
         {"mode": "pct", "dev": 1.0, "stop": 0.5, "at": "14:59"})(_close_vwap)


@strategy("r4_mechanics_prev_mark_fade", "intraday", "mech_close_vwap",
          "Yesterday's last trade vs its 15:00-15:29 VWAP (official-close proxy) > +0.5 % -> SHORT at 09:20 (< -0.5 % LONG); 1 % stop, 60 min",
          {"mark": 0.5, "stop": 1.0, "at": "09:19", "hold": 60, "min_bars": 25})
def r4_mechanics_prev_mark_fade(s, p):
    k = int(s.day_index)
    if k < 1:
        return []
    sd = s._sd
    a, b = int(sd.starts[k - 1]), int(sd.ends[k - 1])          # previous session only (never today's bars)
    t = sd.t[a:b]
    m = (t >= 900) & (t < 930)
    if int(m.sum()) < p.min_bars:
        return []
    h, l, c, v = sd.h[a:b][m], sd.l[a:b][m], sd.c[a:b][m], sd.v[a:b][m]
    tp = (h + l + c) / 3.0
    off = float(np.sum(tp * v) / np.sum(v)) if np.sum(v) > 0 else float(np.mean(tp))
    last = float(sd.c[b - 1])
    if not (off > 0):
        return []
    mark = last / off - 1.0
    i = _at(s, p.at)
    if i < 0:
        return []
    px = float(s.c[i])
    if mark > p.mark / 100.0:
        return [Entry(i, SHORT, stop=px * (1 + p.stop / 100.0), time_stop=p.hold)]
    if mark < -p.mark / 100.0:
        return [Entry(i, LONG, stop=px * (1 - p.stop / 100.0), time_stop=p.hold)]
    return []


# ════════════════════════════════════════════════════════════════════════════
# B. F&O ban list
# ════════════════════════════════════════════════════════════════════════════
def _ret(c, n):
    return float(c[-1] / c[-1 - n] - 1.0) if len(c) > n else np.nan


@strategy("r4_mechanics_ban_entry_long", "swing", "mech_ban",
          "Stock enters F&O ban next session after a > 5 % 10-session fall (short build-up) -> buy next open, 8 % stop, hold 5",
          {"n": 10, "fall": 5.0, "stop_pct": 8.0, "hold": 5})
def r4_mechanics_ban_entry_long(d, p):
    if len(d) < p.n + 2 or not _clean(d):
        return None
    D = int(d.date[-1])
    nx = N.next_fo_session(D)
    if nx is None:
        return None
    if not N.asof_swing(d).banned() or N.banned(d.symbol, D):        # first ban day = next session
        return None
    r = _ret(d.c, p.n)
    if np.isfinite(r) and r < -p.fall / 100.0:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r4_mechanics_ban_entry_short_id", "intraday", "mech_ban",
          "First F&O-ban day after a > 5 % 10-session rise (long build-up): short cash at 09:20, 2 % stop, square-off",
          {"n": 10, "rise": 5.0, "stop": 2.0, "at": "09:19"})
def r4_mechanics_ban_entry_short_id(s, p):
    T = int(s.date)
    if len(s.prev.c) < p.n + 1:
        return []
    if not N.asof_session(s).banned():
        return []
    pf = _prev_fo(T)
    if pf is None or N.banned(s.symbol, pf):
        return []
    r = _ret(s.prev.c, p.n)
    if not (np.isfinite(r) and r > p.rise / 100.0):
        return []
    i = _at(s, p.at)
    if i < 0:
        return []
    return [Entry(i, SHORT, stop=float(s.c[i]) * (1 + p.stop / 100.0))]


@strategy("r4_mechanics_ban_exit_long", "swing", "mech_ban",
          "F&O ban lifts next session and the stock fell during the ban -> buy next open, 8 % stop, hold 5",
          {"stop_pct": 8.0, "hold": 5})
def r4_mechanics_ban_exit_long(d, p):
    if len(d) < 30 or not _clean(d):
        return None
    D = int(d.date[-1])
    if N.next_fo_session(D) is None or not N.banned(d.symbol, D) or N.asof_swing(d).banned():
        return None
    fo = N.fo_sessions(before=D + 1)                      # F&O dates <= D
    k = len(fo) - 1
    while k > 0 and N.banned(d.symbol, int(fo[k - 1])):
        k -= 1
    first = int(fo[k])                                     # first ban day of the current streak
    j = int(np.searchsorted(d.date, first, side="left")) - 1   # last bars session before it
    if j < 0:
        return None
    if d.c[-1] < d.c[j]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# C. Stock option expiry pin at the max-OI strike
# ════════════════════════════════════════════════════════════════════════════
def _pin(s, p):
    T = int(s.date)
    a = N.asof_session(s)
    f = _guard_rows(a.fut_oi(n=1), T, "asof_session.fut_oi")
    pf = _prev_fo(T)
    if len(f) == 0 or pf is None or int(f.date[-1]) != pf or int(f.near_expiry[-1]) != T:
        return []
    ch = a.chain("near")
    if ch is None or int(ch.expiry) != T:
        return []
    if int(ch.date) >= T:
        raise N.LookAheadError(f"r4_mechanics: chain dated {ch.date} >= {T}")
    if int(ch.date) != pf or not (np.isfinite(ch.spot) and np.isfinite(s.pdc)) or abs(ch.spot / s.pdc - 1.0) > 0.02:
        return []
    tot = np.nan_to_num(ch.ce_oi) + np.nan_to_num(ch.pe_oi)
    if not len(tot) or tot.max() <= 0:
        return []
    K = float(ch.strike[int(np.argmax(tot))])
    i = _at(s, p.at)
    if i < 0:
        return []
    c = float(s.c[i])
    if c > K * (1 + p.band / 100.0):
        return [Entry(i, SHORT, stop=c + (c - K), target=K)]
    if c < K * (1 - p.band / 100.0):
        return [Entry(i, LONG, stop=c - (K - c), target=K)]
    return []


strategy("r4_mechanics_expiry_pin_1300", "intraday", "mech_expiry_pin",
         "Monthly stock expiry day 12:59: > 1 % from yesterday's max-OI strike -> trade toward it (target strike, 1:1 stop)",
         {"band": 1.0, "at": "12:59"})(_pin)
strategy("r4_mechanics_expiry_pin_1400", "intraday", "mech_expiry_pin",
         "Monthly stock expiry day 13:59: > 1 % from yesterday's max-OI strike -> trade toward it (target strike, 1:1 stop)",
         {"band": 1.0, "at": "13:59"})(_pin)


# ════════════════════════════════════════════════════════════════════════════
# D. Index monthly expiry / new series
# ════════════════════════════════════════════════════════════════════════════
def _idx_row(d, sym):
    D = int(d.date[-1])
    r = _guard_rows(N.asof_swing(d).fut_oi(n=1, sym=sym), D + 1, f"asof_swing.fut_oi({sym})")
    if len(r) == 0 or int(r.date[-1]) != D:
        return None
    return r


def _new_series(sym):
    def fn(d, p):
        r = _idx_row(d, sym)
        if r is None or int(r.near_expiry[-1]) != int(d.date[-1]):
            return None
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return fn


strategy("r4_mechanics_nifty_new_series", "futures", "mech_index_expiry",
         "Evening of the NIFTY monthly expiry -> LONG NIFTY next open (first day of the new series), hold 5, 3 % stop",
         {"stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="NIFTY")(_new_series("NIFTY"))
strategy("r4_mechanics_bnf_new_series", "futures", "mech_index_expiry",
         "Evening of the BANKNIFTY monthly expiry -> LONG BANKNIFTY next open, hold 5, 3 % stop",
         {"stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="BANKNIFTY")(_new_series("BANKNIFTY"))


@strategy("r4_mechanics_nifty_pre_expiry", "futures", "mech_index_expiry",
          "3 regular sessions left to the NIFTY monthly expiry -> SHORT NIFTY next open, hold 3 (to expiry close), 3 % stop",
          {"left": 3, "stop_pct": 3.0, "hold": 3}, horizon="positional", underlying="NIFTY")
def r4_mechanics_nifty_pre_expiry(d, p):
    r = _idx_row(d, "NIFTY")
    if r is None:
        return None
    D, E = int(d.date[-1]), int(r.near_expiry[-1])
    if E <= D or E > CAL_END or _regular_between(D, E) != p.left:
        return None
    return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)


# ════════════════════════════════════════════════════════════════════════════
# E. Month-start SIP inflows (calendar only)
# ════════════════════════════════════════════════════════════════════════════
def _sip(d, p):
    if not _is_month_last(int(d.date[-1])):
        return None
    return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)


strategy("r4_mechanics_sip_nifty_1520", "futures", "mech_sip",
         "Last regular session of the month 15:20 -> LONG NIFTY (month-start SIP inflows), hold 5, 3 % stop",
         {"stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="NIFTY", decision_time="15:20")(_sip)
strategy("r4_mechanics_sip_bnf_1520", "futures", "mech_sip",
         "Last regular session of the month 15:20 -> LONG BANKNIFTY (month-start SIP inflows), hold 5, 3 % stop",
         {"stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="BANKNIFTY", decision_time="15:20")(_sip)
strategy("r4_mechanics_sip_nifty_d1to10", "futures", "mech_sip",
         "Evening of the month's last regular session -> LONG NIFTY next open (day 1), hold 7 (~days 1-10), 3 % stop",
         {"stop_pct": 3.0, "hold": 7}, horizon="positional", underlying="NIFTY")(_sip)


# ════════════════════════════════════════════════════════════════════════════
# F. Pre-holiday sessions
# ════════════════════════════════════════════════════════════════════════════
def _preh_id(s, p):
    if not _is_preholiday(int(s.date)):
        return []
    i = _at(s, "09:15")
    if i < 0:
        return []
    return [Entry(i, LONG, stop=float(s.c[i]) * (1 - p.stop / 100.0))]


strategy("r4_mechanics_preholiday_nifty_id", "futures", "mech_preholiday",
         "Pre-holiday session: LONG NIFTY future at 09:16, 1 % stop, square-off 15:15",
         {"stop": 1.0}, horizon="intraday", underlying="NIFTY")(_preh_id)
strategy("r4_mechanics_preholiday_stocks_id", "intraday", "mech_preholiday",
         "Pre-holiday session: LONG every stock at 09:16, 1.5 % stop, square-off 15:15",
         {"stop": 1.5})(_preh_id)


@strategy("r4_mechanics_preholiday_nifty_1520", "futures", "mech_preholiday",
          "15:20 of the session before a pre-holiday session -> LONG NIFTY, exit at the pre-holiday close (hold 1), 2 % stop",
          {"stop_pct": 2.0, "hold": 1}, horizon="positional", underlying="NIFTY", decision_time="15:20")
def r4_mechanics_preholiday_nifty_1520(d, p):
    nx = _next_regular(int(d.date[-1]))
    if nx is None or not _is_preholiday(nx):
        return None
    return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)


# ════════════════════════════════════════════════════════════════════════════
# G. Diwali / Muhurat window
# ════════════════════════════════════════════════════════════════════════════
@strategy("r4_mechanics_diwali_window", "swing", "mech_diwali",
          "5 regular sessions before the Muhurat date -> buy every stock next open, hold 10, 8 % stop",
          {"before": 5, "stop_pct": 8.0, "hold": 10})
def r4_mechanics_diwali_window(d, p):
    if len(d) < 30 or not _clean(d):
        return None
    D = int(d.date[-1])
    for M in MUHURAT:
        if M > D and M <= CAL_END:
            # regular sessions strictly between D and M (M itself is a weekend or a holiday for regular trading)
            if _regular_between(D, _prev_regular(M)) == p.before:
                return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
            return None
    return None


# ════════════════════════════════════════════════════════════════════════════
# H. Scheduled macro events
# ════════════════════════════════════════════════════════════════════════════
def _event_mom(days, universe=None):
    def fn(s, p):
        if int(s.date) not in days or (universe is not None and s.symbol not in universe):
            return []
        r, i = _at(s, p.ref), _at(s, p.at)
        if r < 0 or i < 0 or i <= r:
            return []
        ref, c = float(s.c[r]), float(s.c[i])
        mv = c / ref - 1.0
        if mv > p.move / 100.0:
            return [Entry(i, LONG, stop=ref)]
        if mv < -p.move / 100.0:
            return [Entry(i, SHORT, stop=ref)]
        return []
    return fn


strategy("r4_mechanics_budget_mom", "intraday", "mech_events",
         "Union Budget day: move 10:59->12:59 > 1 % -> follow it (stop at the 10:59 price), square-off",
         {"ref": "10:59", "at": "12:59", "move": 1.0})(_event_mom(BUDGET))
strategy("r4_mechanics_mpc_ratesens_mom", "intraday", "mech_events",
         "RBI MPC decision day, rate-sensitive stocks: move 09:59->10:29 > 0.5 % -> follow (stop 09:59 price), sq-off",
         {"ref": "09:59", "at": "10:29", "move": 0.5})(_event_mom(MPC, RATE_SENS))
strategy("r4_mechanics_mpc_all_mom", "intraday", "mech_events",
         "RBI MPC decision day, all stocks: move 09:59->10:29 > 0.5 % -> follow (stop 09:59 price), square-off",
         {"ref": "09:59", "at": "10:29", "move": 0.5})(_event_mom(MPC))


# ════════════════════════════════════════════════════════════════════════════
# I. 52-week high with delivery > 60 %
# ════════════════════════════════════════════════════════════════════════════
def _hi52(d, p):
    if len(d) < p.n + 1 or not _clean(d):
        return None
    if not d.c[-1] > d.c[-p.n - 1:-1].max():
        return None
    if p.vol:
        m = float(np.mean(d.v[-21:-1]))
        if not (m > 0 and d.v[-1] > 1.5 * m):
            return None
    D = int(d.date[-1])
    r = _guard_rows(N.asof_swing(d).delivery(n=1), D + 1, "asof_swing.delivery")
    if len(r) == 0 or int(r.date[-1]) != D:
        return None
    x = float(r.deliv_pct[-1])
    if np.isfinite(x) and x > p.deliv:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


strategy("r4_mechanics_52wh_deliv60_h10", "swing", "mech_52wh_deliv",
         "Close at a 250-session closing high with deliv_pct(D) > 60 -> buy next open, 8 % stop, hold 10",
         {"n": 250, "deliv": 60.0, "vol": False, "stop_pct": 8.0, "hold": 10})(_hi52)
strategy("r4_mechanics_52wh_deliv60_h20", "swing", "mech_52wh_deliv",
         "Close at a 250-session closing high with deliv_pct(D) > 60 -> buy next open, 8 % stop, hold 20",
         {"n": 250, "deliv": 60.0, "vol": False, "stop_pct": 8.0, "hold": 20})(_hi52)
strategy("r4_mechanics_52wh_deliv60_vol", "swing", "mech_52wh_deliv",
         "250-session closing high, deliv_pct(D) > 60, volume > 1.5x 20d -> buy next open, 8 % stop, hold 20",
         {"n": 250, "deliv": 60.0, "vol": True, "stop_pct": 8.0, "hold": 20})(_hi52)
