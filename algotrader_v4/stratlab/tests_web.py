"""
stratlab/tests_web.py — tests for the web-sourced study (web_strategies.py, web_run.py) on SYNTHETIC data only:
the NIFTY100 Low Volatility 30 cut-off calendar, volatility window and its causality (data after the cut-off never
changes a selection), the rank-60 retention buffer; the dividend-month decision (only at a month's last session, only
for a dividend in the same month a year earlier, held to the month's last close), the dividend entitlement rule
(entry < ex-date <= exit) and its credit; the earnings-straddle meeting timing (exactly `ahead` sessions before day 0,
Saturday meetings, notices after 15:00 invisible, the latest notice wins, results already out -> no trade, reads past
the cut-off raise), the expiry / moneyness choice and the plan; the frozen pre-registration matches the code.
Run by test_pipeline.py (section "STRATLAB: web-sourced strategies") or directly: python -m stratlab.tests_web
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import os
import tempfile

import numpy as np

from . import data as D
from . import events_data as E
from . import events_parse as EP
from . import flowdata as F
from . import nsedata as N
from . import sim
from . import web_strategies as W

CA_COLS = ("ex_date", "symbol", "symbol_then", "lab_symbol", "in_lab", "series", "isin", "face_value", "kind",
           "ratio_a", "ratio_b", "fv_from", "fv_to", "dividend", "rights_premium", "factor", "record_date",
           "first_listed", "known_date", "subject")


def _days(n, start=dt.date(2024, 1, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(int(d.strftime("%Y%m%d")))
        d += dt.timedelta(days=1)
    return out


def _sd(sym, days, close):
    c = np.asarray(close, float)
    o = np.concatenate(([c[0]], c[:-1]))
    h, l = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
    sd = D.from_arrays(sym, np.array(days), np.full(len(days), 555), o, h, l, c, np.full(len(c), 1000.0), min_bars=1)
    sd.member = np.ones(len(days), bool)
    return sd


def _ts(day, hhmm):
    return f"{day // 10000:04d}-{day // 100 % 100:02d}-{day % 100:02d} {hhmm // 100:02d}:{hhmm % 100:02d}:00"


def _iso(d):
    return f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"


class _Ctx:
    """Temporary corporate-action table (dividends: (sym, ex_date, Rs)), event tables (boardmeet: (sym, meeting iso,
    notice ts); results: (sym, ts)) and calendar; restores module state on exit."""

    def __init__(self, days, divs=(), boardmeet=(), results=()):
        self.days, self.divs, self.bm, self.res = days, divs, boardmeet, results

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.old_flow, self.old_ev, self.old_cal = F.FLOW_DIR, E.EVENTS_DIR, list(E.calendar())
        os.makedirs(os.path.join(t, "corpactions"))
        with gzip.open(os.path.join(t, "corpactions", "corpactions.csv.gz"), "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(CA_COLS), lineterminator="\n")
            w.writeheader()
            for s, ex, amt in self.divs:
                w.writerow({k: "" for k in CA_COLS} | {"ex_date": ex, "symbol": s, "symbol_then": s, "lab_symbol": s,
                                                       "kind": "dividend", "dividend": amt, "known_date": ex,
                                                       "first_listed": ex, "subject": "Dividend"})
        EP.write_csv(os.path.join(t, "insider.csv.gz"), EP.INS_COLS, [])
        EP.write_csv(os.path.join(t, "results.csv.gz"), EP.RES_COLS,
                     [{"symbol": s, "period_end": "2024-03-31", "first_ts": ts} for s, ts in self.res])
        EP.write_csv(os.path.join(t, "boardmeet.csv.gz"), EP.BM_COLS,
                     [{"symbol": s, "meeting_date": m, "purpose": "Financial Results", "is_results": 1, "notice_ts": ts}
                      for s, m, ts in self.bm])
        F.use(t)
        E.use(t)
        E.set_calendar(self.days)
        self.old_credit = dict(W.DIV_CREDIT)
        W.DIV_CREDIT.clear()
        return self

    def __exit__(self, *a):
        F.use(self.old_flow)
        E.use(self.old_ev)
        E.set_calendar(self.old_cal)
        W.DIV_CREDIT.clear()
        W.DIV_CREDIT.update(self.old_credit)
        self.tmp.cleanup()


def _run(spec, sd, lo=0, hi=99999999):
    from . import web_run as WR
    return sim.sim_positional(WR.guarded(spec, sd), sd, lo, hi, None, None, 0.0)


# ── A. low volatility ───────────────────────────────────────────────────────
def t_lv_cutoffs_and_quarters():
    """cut-offs = last sessions of Feb/May/Aug/Nov; holding quarter starts Apr/Jul/Oct/Jan (year roll for Nov)"""
    days = _days(520, dt.date(2024, 1, 1))
    cs = W.lv_cutoffs(days)
    assert cs and all(c // 100 % 100 in (2, 5, 8, 11) for c, _ in cs)
    for c, first in cs:
        i = days.index(c)
        assert days[i + 1] // 100 != c // 100                        # last session of its month
        assert first // 100 % 100 == {2: 4, 5: 7, 8: 10, 11: 1}[c // 100 % 100]
        assert first == min(d for d in days if d // 100 == first // 100)
    nov = [x for x in cs if x[0] // 100 % 100 == 11][0]
    assert nov[1] // 10000 == nov[0] // 10000 + 1
    with _Ctx(days):
        q = W.sessions_in_quarter(cs[0][1])
        y, m = cs[0][1] // 10000, cs[0][1] // 100 % 100
        assert q == sum(1 for d in days if d // 100 in (y * 100 + m, y * 100 + m + 1, y * 100 + m + 2))


def t_lv_volatility_window_and_causality():
    """std of the last 250 log returns ending at the cut-off; later data (even absurd) changes nothing"""
    days = np.array(_days(400), np.int64)
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    cut = int(days[300])
    v = W.lv_volatility(days, c, cut)
    r = np.diff(np.log(c[300 - 250:301]))
    assert len(r) == 250 and abs(v - np.std(r, ddof=1)) < 1e-12
    c2 = c.copy()
    c2[301:] *= 5.0
    assert W.lv_volatility(days, c2, cut) == v
    assert W.lv_volatility(days[:301], c[:301], cut) == v           # physically truncated
    assert np.isnan(W.lv_volatility(days, c, int(days[100])))       # < 1 year of history
    # excluded overnight links are dropped
    v3 = W.lv_volatility(days, c, cut, excluded=lambda a, b: b == int(days[299]))
    assert v3 != v


def t_lv_select_buffer_and_selection_causal():
    """30 lowest; an existing member stays while ranked <= 60; selection unchanged by data after each cut-off"""
    vols = {f"S{i:03d}": 0.01 + i * 0.001 for i in range(100)}
    sel = W.lv_select(vols, frozenset())
    assert sel == frozenset(f"S{i:03d}" for i in range(30))
    prev = frozenset([f"S{i:03d}" for i in range(29)] + ["S055"])      # S055 ranked 56 -> retained
    sel2 = W.lv_select(vols, prev)
    assert "S055" in sel2 and len(sel2) == 30 and "S029" not in sel2
    prev3 = frozenset([f"S{i:03d}" for i in range(29)] + ["S070"])      # ranked 71 -> dropped
    sel3 = W.lv_select(vols, prev3)
    assert "S070" not in sel3 and "S029" in sel3
    days = _days(700, dt.date(2023, 1, 2))
    rng = np.random.default_rng(2)
    ser = {f"X{i:02d}": (np.array(days, np.int64), 100 * np.exp(np.cumsum(rng.normal(0, 0.005 + 0.0005 * i, len(days)))))
           for i in range(40)}
    full = W.build_lv_selection(days, lambda c: set(ser), lambda s: ser[s])
    cut, first = W.lv_cutoffs(days)[6]
    k = days.index(cut) + 1
    trunc = {s: (v[0][:k], v[1][:k]) for s, v in ser.items()}
    part = W.build_lv_selection(days[:k + 60], lambda c: set(trunc), lambda s: trunc[s])
    assert part[first] == full[first] and len(full[first]) == 30


def t_lowvol_spec_enters_selected_names_only():
    """a selected name is bought at the quarter's first open and sold at its last close"""
    days = _days(200, dt.date(2025, 1, 1))
    first = min(d for d in days if d // 100 == 202504)
    last = max(d for d in days if d // 100 == 202506)
    sp = W.BY_ID["web_lowvol30_nifty100"]
    old = dict(W.LV_SELECTION)
    try:
        W.LV_SELECTION.clear()
        W.LV_SELECTION[first] = frozenset({"AAA"})
        with _Ctx(days):
            tr, _ = _run(sp, _sd("AAA", days, np.full(200, 100.0)))
            assert len(tr) == 1 and tr[0][2] == first and tr[0][6] == last, tr
            assert _run(sp, _sd("BBB", days, np.full(200, 100.0)))[0] == []
    finally:
        W.LV_SELECTION.clear()
        W.LV_SELECTION.update(old)


# ── B. dividend month ───────────────────────────────────────────────────────
def t_divm_decision_month_and_hold():
    """buy at the first open of month m only if a dividend went ex in month m-12; sell at m's last close"""
    days = _days(420, dt.date(2024, 1, 1))
    mar24 = min(d for d in days if d // 100 == 202403) + 5
    first = min(d for d in days if d // 100 == 202503)
    last = max(d for d in days if d // 100 == 202503)
    sp = W.BY_ID["web_divm_pred_exmonth"]
    with _Ctx(days, divs=[("AAA", mar24, 5.0)]):
        tr, _ = _run(sp, _sd("AAA", days, np.full(420, 100.0)))
        assert len(tr) == 1 and tr[0][2] == first and tr[0][6] == last, tr
        assert tr[0][13] == "max_hold"
    with _Ctx(days, divs=[("AAA", mar24, float("nan"))]):              # no amount: not a cash dividend
        assert _run(sp, _sd("AAA", days, np.full(420, 100.0)))[0] == []
    feb25 = min(d for d in days if d // 100 == 202502) + 3
    with _Ctx(days, divs=[("AAA", feb25, 5.0)]):                        # this year's dividend, not last year's
        tr, _ = _run(sp, _sd("AAA", days, np.full(420, 100.0)))
        assert all(t[2] // 100 != 202503 for t in tr)


def t_dividend_credit_entitlement():
    """entry < ex-date <= exit: buying ON the ex-date gets nothing, selling ON the ex-date keeps it"""
    save = dict(W.DIV_CREDIT)
    try:
        W.DIV_CREDIT.clear()
        W.DIV_CREDIT["AAA"] = (np.array([20250310, 20250320], np.int64), np.array([2.0, 3.0]))
        assert W.dividend_credit("AAA", 20250310, 20250331) == 3.0
        assert W.dividend_credit("AAA", 20250309, 20250310) == 2.0
        assert W.dividend_credit("AAA", 20250301, 20250331) == 5.0
        assert W.dividend_credit("BBB", 20250301, 20250331) == 0.0
        t = ("x", "AAA", 20250303, 1, "09:15", 100.0, 20250331, "15:30", 99.0, 10, -10.0, 2.0, -12.0, "max_hold")
        out, tot = W.credit_dividends([t])
        assert tot == 50.0 and out[0][10] == 40.0 and out[0][12] == 38.0 and out[0][11] == 2.0
    finally:
        W.DIV_CREDIT.clear()
        W.DIV_CREDIT.update(save)


# ── C / D. earnings straddles ───────────────────────────────────────────────
def t_ea_meeting_timing():
    """entry exactly `ahead` sessions before day 0; notices after 15:00 of the entry day are not seen; Saturday
    meetings react on Monday; the latest notice wins; results already out -> no trade"""
    days = _days(60, dt.date(2024, 7, 1))
    m = days[30]
    with _Ctx(days, boardmeet=[("AAA", _iso(m), _ts(days[10], 1800))]):
        assert W.ea_meeting("AAA", days[27], 3)[1] == m
        assert W.ea_meeting("AAA", days[26], 3) is None and W.ea_meeting("AAA", days[28], 3) is None
        assert W.ea_meeting("AAA", days[29], 1)[1] == m
    with _Ctx(days, boardmeet=[("AAA", _iso(m), _ts(days[27], 1501))]):     # notified after 15:00 of the entry day
        assert W.ea_meeting("AAA", days[27], 3) is None
    with _Ctx(days, boardmeet=[("AAA", _iso(m), _ts(days[27], 1430))]):
        assert W.ea_meeting("AAA", days[27], 3) is not None
    sat = dt.date(days[30] // 10000, days[30] // 100 % 100, days[30] % 100)
    while sat.weekday() != 5:
        sat += dt.timedelta(days=1)
    mon = min(d for d in days if d > int(sat.strftime("%Y%m%d")))
    with _Ctx(days, boardmeet=[("AAA", sat.isoformat(), _ts(days[10], 1800))]):
        k = days.index(mon)
        assert W.ea_meeting("AAA", days[k - 3], 3)[1] == mon
    with _Ctx(days, boardmeet=[("AAA", _iso(m), _ts(days[10], 1800)), ("AAA", _iso(days[40]), _ts(days[20], 1800))]):
        assert W.ea_meeting("AAA", days[27], 3) is None                     # rescheduled: latest notice wins
        assert W.ea_meeting("AAA", days[37], 3)[1] == days[40]
    with _Ctx(days, boardmeet=[("AAA", _iso(m), _ts(days[10], 1800))], results=[("AAA", _ts(days[25], 1700))]):
        assert W.ea_meeting("AAA", days[27], 3) is None                     # numbers already public
    with _Ctx(days, boardmeet=[("AAA", _iso(m), _ts(days[10], 1800))]):
        with E.cutoff(E.stamp(days[20], 900)):
            try:
                W.ea_meeting("AAA", days[27], 3)
                raise AssertionError("a read past the cut-off must raise")
            except E.LookAheadError:
                pass


class _FakeCtx:
    def __init__(self, sym, date, chains):
        self.symbol, self.date, self._c = sym, date, chains

    def chain(self, which="near"):
        return self._c.get(which)


def _chain(day, expiry, spot, strikes):
    k = np.array(strikes, float)
    px = np.full(len(k), 10.0)
    nan = np.full(len(k), np.nan)
    return N.Chain("AAA", day, expiry, k, px, px, nan, nan, px, px, nan, nan, px, px, spot, spot)


def t_ea_decide_expiry_moneyness_and_plan():
    """10-60 days to expiry (near, else next), ATM strike from the previous session's chain, both legs bought, exit
    after `hold` sessions; a chain dated on the decision day raises"""
    days = _days(60, dt.date(2024, 7, 1))
    m = days[30]
    entry, prev = days[27], days[26]
    sp = W.option_specs(("AAA",))[0]
    old = W._prev_fo
    W._prev_fo = lambda day: prev
    try:
        with _Ctx(days, boardmeet=[("AAA", _iso(m), _ts(days[10], 1800))]):
            near = _chain(prev, days[29], 1003.0, [980, 1000, 1020])          # 2 sessions to expiry: < 10 days
            nxt = _chain(prev, days[50], 1003.0, [980, 1000, 1020])
            plan = W.ea_decide(_FakeCtx("AAA", entry, {"near": near, "next": nxt}), sp.p)
            assert plan is not None and plan.expiry == days[50] and plan.exit == 3
            assert sorted((l.typ, l.strike, l.side) for l in plan.legs) == [("CE", 1000.0, 1), ("PE", 1000.0, 1)]
            far = _chain(prev, 20250101, 1003.0, [980, 1000, 1020])          # > 60 days
            assert W.ea_decide(_FakeCtx("AAA", entry, {"near": near, "next": far}), sp.p) is None
            odd = _chain(prev, days[50], 1003.0, [1200, 1300])                # no strike within 5 %
            assert W.ea_decide(_FakeCtx("AAA", entry, {"near": odd}), sp.p) is None
            stale = _chain(days[20], days[50], 1003.0, [1000])                # not the previous F&O session
            assert W.ea_decide(_FakeCtx("AAA", entry, {"near": stale}), sp.p) is None
            W._prev_fo = lambda day: entry
            bad = _chain(entry, days[50], 1003.0, [1000])
            try:
                W.ea_decide(_FakeCtx("AAA", entry, {"near": bad}), sp.p)
                raise AssertionError("a same-day chain must raise")
            except N.LookAheadError:
                pass
    finally:
        W._prev_fo = old


def t_prereg_matches_code():
    """the committed pre-registration lists exactly these 4 variants with matching fingerprints and gates"""
    from . import web_run as WR
    assert len(W.ALL_IDS) == WR.N_VARIANTS == 4
    assert all(i not in __import__("stratlab.api", fromlist=["REGISTRY"]).REGISTRY for i in W.ALL_IDS)
    if not os.path.exists(WR.REPO_PREREG):
        return
    with open(WR.REPO_PREREG) as f:
        pre = json.load(f)
    assert pre["n_variants"] == 4 and len(pre["variants"]) == 4
    for k, v in WR.fingerprints().items():
        assert pre["strategies"][k]["fingerprint"] == v, k
    assert pre["gates"]["screen"]["min_pf"] == 1.15 and pre["gates"]["dsr_min"] == 0.95
    assert pre["code_sha256"]["web_strategies.py"] == WR.code_sha()["web_strategies.py"]


TESTS = [(f.__name__, f) for f in (
    t_lv_cutoffs_and_quarters, t_lv_volatility_window_and_causality, t_lv_select_buffer_and_selection_causal,
    t_lowvol_spec_enters_selected_names_only, t_divm_decision_month_and_hold, t_dividend_credit_entitlement,
    t_ea_meeting_timing, t_ea_decide_expiry_moneyness_and_plan, t_prereg_matches_code)]


if __name__ == "__main__":
    bad = 0
    for n, f in TESTS:
        try:
            f()
            print("PASS", n)
        except Exception as exc:
            bad += 1
            import traceback
            traceback.print_exc()
            print("FAIL", n, exc)
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
