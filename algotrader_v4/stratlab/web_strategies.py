"""
stratlab/web_strategies.py — the pre-registered WEB-SOURCED strategies (prereg/web_v1.json), chosen from the
external-evidence shortlist (<scratchpad>/webstrats/SHORTLIST.md) BEFORE any of our data was looked at. Four
variants, each defined by its published rule with round / textbook parameters:

  A  web_lowvol30_nifty100   NSE Indices "NIFTY100 Low Volatility 30" (methodology Nov 2026 pp. 190-192): every
                             quarter, from the point-in-time NIFTY 100 members that trade in F&O and have >= 1 year
                             of history, rank on the standard deviation of 250 daily log returns ending the last
                             session of Feb / May / Aug / Nov; keep the 30 least volatile, an existing member stays
                             while ranked <= 60. Buy every selected name at the first open of Apr / Jul / Oct / Jan
                             (the index changes take effect at the end of Mar / Jun / Sep / Dec) and sell at the close
                             of the quarter's last session. Equal Rs 1 lakh per name (the index weights by 1/vol).
  B  web_divm_pred_exmonth   Hartzmark & Solomon (JFE 2013) dividend-month premium; India: Hameed, Xie & Zhong (2024)
                             +1.01 %/month, t = 8.35. At the close of a month's last session, buy (next open) every
                             top-500 member that went ex-dividend (cash dividend) in the same calendar month one year
                             earlier; sell at the close of the new month's last session. Dividends going ex while held
                             are credited (gross) to the trade.
  C  web_ea_straddle_m3_0    Gao, Xing & Zhang (JFQA 2018) [-3, 0]: day 0 = the first session on / after the notified
                             results board-meeting date; buy one lot of the at-the-money call + put (10-60 days to
                             expiry) at the EOD close of day -3, sell at the EOD close of day 0. Real NSE EOD prices.
  D  web_ea_straddle_m1_p1   the same, [-1, +1]: buy at the close of day -1, sell at the close of day +1 (holds
                             through the announcement whatever its hour).

A and B are ordinary positional lab specs (api.Spec, Swing, run by sim.sim_positional on the events_daily point-in-time
cache); C and D are nsedata.OptionSpec (real-option-price EOD simulator). None of them is in the catalogue REGISTRY:
they need tables built by web_run.setup and are run only by web_run.py. Do not tune: the parameters are the papers'.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from . import events_data as E
from . import flowdata as F
from . import nsedata as N
from .api import Spec, Swing, LONG

# ══════════════════════════════════════════════════════════════════════════
# shared tables (set by web_run.setup BEFORE workers fork; tests set them by hand)
# ══════════════════════════════════════════════════════════════════════════
LV_SELECTION: dict = {}          # {first session of the holding quarter (yyyymmdd): frozenset(symbols)}
DIV_CREDIT: dict = {}            # {symbol: (ex_dates int64 array sorted, Rs per share array)} — REAL dividends


def month_of(day: int) -> int:
    return int(day) // 100


def sessions_in_month(yyyymm: int) -> int:
    return sum(1 for d in E.calendar() if d // 100 == yyyymm)


def quarter_start_months() -> tuple:
    return (1, 4, 7, 10)


def sessions_in_quarter(first_day: int) -> int:
    """Number of calendar sessions from `first_day` (first session of a quarter) to the quarter's last session."""
    y, m = first_day // 10000, first_day // 100 % 100
    q_months = {y * 100 + m, y * 100 + m + 1, y * 100 + m + 2}
    return sum(1 for d in E.calendar() if d // 100 in q_months)


# ══════════════════════════════════════════════════════════════════════════
# A. NIFTY100 Low Volatility 30
# ══════════════════════════════════════════════════════════════════════════
LV_N, LV_BUFFER, LV_WINDOW, LV_MIN_HISTORY = 30, 60, 250, 250
CUTOFF_MONTHS = (2, 5, 8, 11)


def lv_cutoffs(calendar: list) -> list:
    """(cutoff session, first session of the holding quarter): cutoff = last session of Feb / May / Aug / Nov,
    holding quarter starts at the first session of the month after the NEXT month (Apr / Jul / Oct / Jan)."""
    cal = sorted(calendar)
    out = []
    for i, d in enumerate(cal[:-1]):
        if d // 100 % 100 in CUTOFF_MONTHS and cal[i + 1] // 100 != d // 100:
            y, m = d // 10000, d // 100 % 100
            tm = m + 2
            ty = y + (1 if tm > 12 else 0)
            tm = tm - 12 if tm > 12 else tm
            first = next((x for x in cal[i + 1:] if x // 100 == ty * 100 + tm), None)
            if first is not None:
                out.append((d, first))
    return out


def lv_volatility(dates: np.ndarray, close: np.ndarray, cutoff: int, excluded=None) -> float:
    """Standard deviation (ddof 1) of the last LV_WINDOW daily log returns ending at `cutoff` (dates <= cutoff
    only). NaN when fewer than LV_MIN_HISTORY returns exist. `excluded(d0, d1)` -> True drops that overnight link
    (unadjusted corporate-action gaps)."""
    k = int(np.searchsorted(dates, int(cutoff), side="right"))
    if k < LV_WINDOW + 1:
        return float("nan")
    d, c = dates[k - LV_WINDOW - 1:k], close[k - LV_WINDOW - 1:k]
    ok = (c[1:] > 0) & (c[:-1] > 0)
    r = np.log(np.where(ok, c[1:], 1.0) / np.where(ok, c[:-1], 1.0))
    if excluded is not None:
        for j in range(len(r)):
            if ok[j] and excluded(int(d[j]), int(d[j + 1])):
                ok[j] = False
    if ok.sum() < LV_MIN_HISTORY * 0.9:
        return float("nan")
    return float(np.std(r[ok], ddof=1))


def lv_select(vols: dict, previous: frozenset, n: int = LV_N, buffer: int = LV_BUFFER) -> frozenset:
    """The NSE rule: rank ascending on volatility (ties by symbol); existing members ranked <= buffer stay; the rest
    are filled from the lowest ranks until n names."""
    ranked = sorted((v, s) for s, v in vols.items() if v == v)
    rank = {s: i + 1 for i, (_v, s) in enumerate(ranked)}
    keep = [s for s in sorted(previous) if rank.get(s, 10 ** 9) <= buffer]
    keep = sorted(keep, key=lambda s: rank[s])[:n]
    out = list(keep)
    for _v, s in ranked:
        if len(out) >= n:
            break
        if s not in out:
            out.append(s)
    return frozenset(out)


def build_lv_selection(calendar: list, eligible_fn, series_fn, excluded=None, cheat=None) -> dict:
    """{first session of holding quarter: selection}. eligible_fn(cutoff) -> set of symbols (NIFTY 100 members known
    at the cutoff's close that trade in F&O); series_fn(sym) -> (dates, close) or None. Every number used for the
    selection at a cutoff is dated <= the cutoff (lv_volatility slices by date). cheat(sym, first) -> score
    replaces the volatility (planted-edge control only)."""
    out, prev = {}, frozenset()
    for cutoff, first in lv_cutoffs(calendar):
        vols = {}
        for s in sorted(eligible_fn(cutoff)):
            ser = series_fn(s)
            if ser is None:
                continue
            v = lv_volatility(ser[0], ser[1], cutoff, None if excluded is None else
                              (lambda a, b, s=s: excluded(s, a, b)))
            if v == v and cheat is not None:
                v = cheat(s, first)
            vols[s] = v
        sel = lv_select(vols, prev)
        out[first] = sel
        prev = sel
    return out


def lowvol30(d, p):
    day = int(d.date[-1])
    nxt = E.next_session(day)
    if nxt is None or nxt // 100 == day // 100 or nxt // 100 % 100 not in quarter_start_months():
        return None
    sel = LV_SELECTION.get(nxt)
    if not sel or d.symbol not in sel:
        return None
    return Swing(LONG, stop_pct=None, max_hold=sessions_in_quarter(nxt))


# ══════════════════════════════════════════════════════════════════════════
# B. dividend-month premium
# ══════════════════════════════════════════════════════════════════════════
def predicted_ex_month(sym: str, day: int, target_month: int) -> bool:
    """A cash dividend of `sym` went ex in calendar month target_month - 12 months, as known by the close of `day`."""
    y, m = divmod(int(target_month), 100)
    want = (y - 1) * 100 + m
    r = F.corp_actions(sym, int(day) + 1, kinds=("dividend",), upcoming=False)
    if not len(r["ex_date"]):
        return False
    for ex, amt in zip(r["ex_date"], r["dividend"]):
        if int(ex) // 100 == want and amt == amt and float(amt) > 0:
            return True
    return False


def divm(d, p):
    day = int(d.date[-1])
    nxt = E.next_session(day)
    if nxt is None or nxt // 100 == day // 100:
        return None                                  # decide only at the close of a month's last session
    if not predicted_ex_month(d.symbol, day, nxt // 100):
        return None
    return Swing(LONG, stop_pct=None, max_hold=sessions_in_month(nxt // 100))


def dividend_credit(sym: str, entry: int, exit_: int) -> float:
    """Rs per share of REAL cash dividends with entry < ex-date <= exit (a buyer at the entry session's open is not
    entitled on an ex-date equal to the entry; a holder at the close before the ex-date is, even if he sells on the
    ex-date)."""
    t = DIV_CREDIT.get(sym)
    if t is None:
        return 0.0
    ex, amt = t
    a, b = np.searchsorted(ex, int(entry), "right"), np.searchsorted(ex, int(exit_), "right")
    return float(np.nansum(amt[a:b]))


def build_div_credit(symbols) -> dict:
    """{symbol: (ex_dates, Rs/share)} of every cash dividend in the corporate-actions table (P&L accounting only:
    a dividend is cash received by the holder, whatever was known at the decision)."""
    F._load_ca()
    out = {}
    for s in symbols:
        r = F.corp_actions(s, 99991231, kinds=("dividend",))
        if not len(r["ex_date"]):
            continue
        m = np.isfinite(r["dividend"]) & (r["dividend"] > 0)
        if m.any():
            ex = np.asarray(r["ex_date"][m], np.int64)
            o = np.argsort(ex, kind="stable")
            out[s] = (ex[o], np.asarray(r["dividend"][m], float)[o])
    return out


def credit_dividends(trades: list) -> tuple[list, float]:
    """Trades with the dividends received added to gross and net (qty x Rs/share). Returns (trades, total Rs)."""
    out, tot = [], 0.0
    for t in trades:
        x = int(t[6]) if t[6] else int(t[2])
        dv = dividend_credit(t[1], int(t[2]), x) * float(t[9]) * (1 if int(t[3]) > 0 else -1)
        if dv:
            t = tuple(t[:10]) + (round(float(t[10]) + dv, 2), t[11], round(float(t[12]) + dv, 2)) + tuple(t[13:])
            tot += dv
        out.append(t)
    return out, tot


# ══════════════════════════════════════════════════════════════════════════
# C / D. long ATM straddles into results (real NSE EOD option prices)
# ══════════════════════════════════════════════════════════════════════════
DECIDE_HHMM = 1500               # notices broadcast before 15:00 of the entry session; the order goes in at the close
MIN_DTE, MAX_DTE = 10, 60        # Gao-Xing-Zhang: 10-60 days to maturity
MONEYNESS = (0.95, 1.05)


def _cal_days(a: int, b: int) -> int:
    da = _dt.date(a // 10000, a // 100 % 100, a % 100)
    db = _dt.date(b // 10000, b // 100 % 100, b % 100)
    return (db - da).days


def day0_of(meeting: int) -> int | None:
    """First calendar session on / after the meeting date (a Saturday meeting reacts on Monday)."""
    cal = E.calendar()
    i = E.session_index(int(meeting))
    return cal[i] if i < len(cal) else None


def ea_meeting(sym: str, day: int, ahead: int):
    """(meeting, day0, notice ts) when the LATEST results-meeting notice broadcast before DECIDE_HHMM of `day` puts
    day 0 exactly `ahead` sessions after `day` and no results of `sym` were published since that notice; else None."""
    st = E.stamp(day, DECIDE_HHMM)
    with E.cutoff(st):
        rows = E.board_meetings(sym, st)
        if not len(rows):
            return None
        idx = np.flatnonzero(rows["results"])
        if not len(idx):
            return None
        j = int(idx[-1])                                   # the latest notice wins (reschedules supersede)
        meeting, nts = int(rows["meeting"][j]), int(rows["ts"][j])
        if meeting < day:
            return None
        d0 = day0_of(meeting)
        cal = E.calendar()
        i = E.session_index(day)
        if d0 is None or i >= len(cal) or cal[i] != day:
            return None
        if E.session_index(d0) - E.session_index(day) != ahead:
            return None
        if len(E.results(sym, st, since=nts)):
            return None                                    # numbers already out: nothing to anticipate
        return meeting, d0, nts


def _prev_fo(day: int) -> int | None:
    d = N.fo_sessions(before=day)
    return int(d[-1]) if len(d) else None


def ea_decide(ctx, p):
    m = ea_meeting(ctx.symbol, int(ctx.date), int(p.ahead))
    if m is None:
        return None
    meeting, d0, _nts = m
    day = int(ctx.date)
    prev = _prev_fo(day)
    chain = None
    for which in ("near", "next"):
        c = ctx.chain(which)
        if c is None or int(c.date) != prev:
            continue
        if int(c.date) >= day:
            raise N.LookAheadError(f"web_ea: chain dated {c.date} for a decision on {day}")
        dte = _cal_days(day, int(c.expiry))
        if MIN_DTE <= dte <= MAX_DTE:
            chain = c
            break
    if chain is None or not chain.spot > 0:
        return None
    k = chain.atm()
    if not k == k or not MONEYNESS[0] <= chain.spot / k <= MONEYNESS[1]:
        return None
    return N.Plan(int(chain.expiry), [N.Leg("CE", float(k), +1, 1), N.Leg("PE", float(k), +1, 1)],
                  exit=int(p.hold), note=f"meeting {meeting} day0 {d0}")


# ══════════════════════════════════════════════════════════════════════════
# specs
# ══════════════════════════════════════════════════════════════════════════
def _spec(id, family, desc, fn, params) -> Spec:
    return Spec(id=id, kind="swing", family=family, desc=desc, fn=fn, params=dict(params), style="vector",
                max_per_day=1, horizon="positional", decision_time=None, needs_index=False, module=__name__)


DAILY = [
    _spec("web_lowvol30_nifty100", "web_lowvol",
          "NSE NIFTY100 Low Volatility 30 rule: 30 least volatile (1-year daily log-return std, data to end Feb/May/"
          "Aug/Nov, retain if rank <= 60) of point-in-time NIFTY 100 F&O names; buy first open of Apr/Jul/Oct/Jan, "
          "sell at the quarter's last close; Rs 1 lakh each, no stop",
          lowvol30, {"n": LV_N, "buffer": LV_BUFFER, "window": LV_WINDOW, "min_history": LV_MIN_HISTORY,
                     "cutoff_months": list(CUTOFF_MONTHS), "stop_pct": None}),
    _spec("web_divm_pred_exmonth", "web_divm",
          "Dividend-month premium (Hartzmark-Solomon 2013): at a month's last close buy (next open) every top-500 "
          "member that went ex-dividend in the same calendar month a year earlier; sell at that month's last close; "
          "dividends received credited; no stop",
          divm, {"lag_months": 12, "credit_dividends": True, "stop_pct": None}),
]


def _ospec(id, desc, params, underlyings=("RELIANCE",)) -> N.OptionSpec:
    return N.OptionSpec(id, desc, dict(params), ea_decide, underlyings=tuple(underlyings), family="web_ea")


EA_PARAMS = {
    "web_ea_straddle_m3_0": ("Long 1-lot ATM straddle (10-60 DTE) at the EOD close 3 sessions before the results "
                             "day (day 0 = first session on/after the notified board-meeting date), sold at day 0's "
                             "close (Gao-Xing-Zhang 2018 [-3,0]); real NSE EOD prices",
                             {"ahead": 3, "hold": 3, "min_dte": MIN_DTE, "max_dte": MAX_DTE,
                              "decide_hhmm": DECIDE_HHMM, "moneyness": list(MONEYNESS)}),
    "web_ea_straddle_m1_p1": ("Long 1-lot ATM straddle (10-60 DTE) at the EOD close of the session before the "
                              "results day, sold at the close of the session after it (Gao-Xing-Zhang [-1,+1]); "
                              "real NSE EOD prices",
                              {"ahead": 1, "hold": 2, "min_dte": MIN_DTE, "max_dte": MAX_DTE,
                               "decide_hhmm": DECIDE_HHMM, "moneyness": list(MONEYNESS)}),
}


def option_specs(underlyings) -> list:
    return [_ospec(i, d, p, underlyings) for i, (d, p) in EA_PARAMS.items()]


OPTION_IDS = tuple(EA_PARAMS)
ALL_IDS = tuple(sp.id for sp in DAILY) + OPTION_IDS
BY_ID = {sp.id: sp for sp in DAILY}
LONG_ONLY_BENCH = {"web_lowvol30_nifty100": "NIFTY100", "web_divm_pred_exmonth": "NIFTY500"}


def option_fingerprint(sid: str) -> str:
    """Fingerprint of an option spec independent of its underlyings list (which depends on the era's data)."""
    return _ospec(sid, *EA_PARAMS[sid], underlyings=("X",)).fingerprint
