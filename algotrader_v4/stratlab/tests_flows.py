"""
stratlab/tests_flows.py — tests for the flow study (flows_strategies.py, flows_run.py) on SYNTHETIC data only:
the client-class keyword classifier (fixed order, deterministic), per-client netting of a session's deals, the
next-open availability of deals (a deal dated D fills at D+1's open, never earlier; the guard clips later rows),
announcement timing -> first tradable session for index changes (after-close assumption, weekend releases), the
exclusion of rows without a real announcement date from the run-up variants (but not from the effective-date
variants), month-precision rows never used, index pooling / de-duplication, the 1:3 bracket maths, the placebo
(same stocks, same count), the Rs 10 lakh account, the trials-ledger append and the frozen pre-registration.
Run by test_pipeline.py (section "STRATLAB: flow study") or directly: python -m stratlab.tests_flows
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
from . import flowdata as F
from . import flows_strategies as FS
from . import nsedata as N
from . import sim


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


DEAL_COLS = ("date", "kind", "symbol", "lab_symbol", "in_lab", "security", "client", "side", "qty", "price",
             "value_cr", "remarks")
CH_COLS = ("index", "announce_date", "effective_date", "action", "symbol", "lab_symbol", "in_lab", "company",
           "source", "announce_source", "effective_precision", "note")


class _Flow:
    """Temporary flow tables (deals: (date, sym, client, side, value_cr); changes: (index, ann, eff, action, sym,
    announce_source, precision)) + calendar + F&O list; restores the module state on exit."""

    def __init__(self, days, deals=(), changes=(), fo=None):
        self.days, self.deals, self.changes, self.fo = days, deals, changes, fo

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_dir, self.old_cal = F.FLOW_DIR, list(E.calendar())
        os.makedirs(os.path.join(self.tmp.name, "deals"))
        os.makedirs(os.path.join(self.tmp.name, "index_membership"))
        with gzip.open(os.path.join(self.tmp.name, "deals", "deals.csv.gz"), "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(DEAL_COLS)
            for d, s, cl, side, v in self.deals:
                w.writerow([d, "bulk", s, s, 0, "", cl, side, round(v * 1e5), 100.0, v, ""])
        with open(os.path.join(self.tmp.name, "index_membership", "index_changes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CH_COLS)
            for ix, a, e, act, s, src, prec in self.changes:
                w.writerow([ix, a, e, act, s, s, 0, "", "x.pdf", src, prec, ""])
        F.use(self.tmp.name)
        E.set_calendar(self.days)
        E.set_fo(self.fo or {})
        return self

    def __exit__(self, *a):
        F.use(self.old_dir)
        E.set_calendar(self.old_cal)
        E.set_fo({})
        self.tmp.cleanup()


def _run(spec, sd, lo=0, hi=99999999):
    from . import flows_run as FR
    return sim.sim_positional(FR.guarded(spec, sd), sd, lo, hi, None, None, 0.0)


MF = "SBI MUTUAL FUND"


# ── classifier ──────────────────────────────────────────────────────────────
def t_client_classifier_fixed_and_deterministic():
    """fixed ordered keyword list: MF > INSURANCE > PROP_BROKER > FUND > OTHER; case / dots / spaces ignored"""
    want = {"SBI MUTUAL FUND": "MF", "hdfc  trustee company ltd a/c hdfc mid-cap": "MF",
            "ICICI PRUDENTIAL LIFE INSURANCE COMPANY LIMITED": "INSURANCE", "THE NEW INDIA ASSURANCE CO": "INSURANCE",
            "GRAVITON RESEARCH CAPITAL LLP": "PROP_BROKER", "BNP PARIBAS ARBITRAGE": "PROP_BROKER",
            "GOLDMAN SACHS BANK EUROPE SE - ODI": "PROP_BROKER", "SHARE INDIA SECURITIES LIMITED": "PROP_BROKER",
            "MANSI SHARE AND STOCK ADVISORS PVT LTD": "PROP_BROKER",
            "THE VANGUARD GROUP INC A/C VANGUARD EMERG. MKTS STOCK INDEX FUND": "FUND",
            "ISHARES CORE MSCI EMERGING MARKETS ETF": "FUND", "GOVERNMENT OF SINGAPORE": "FUND",
            "NORGES BANK ON ACCOUNT OF THE GOVERNMENT PENSION FUND GLOBAL": "FUND",
            "ABU DHABI INVESTMENT AUTHORITY": "FUND", "PI OPPORTUNITIES AIF V LLP": "FUND",
            "S.B. ADANI FAMILY TRUST": "OTHER", "SOCIETE GENERALE": "OTHER", "RADHAKISHAN DAMANI": "OTHER",
            "MORGAN STANLEY ASIA (SINGAPORE) PTE.": "OTHER", "": "OTHER"}
    for name, c in want.items():
        assert FS.classify(name) == c, (name, FS.classify(name), c)
        assert FS.classify(name.lower()) == c and FS.classify("  " + name.replace(" ", "   ") + " ") == c
    assert [c for c, _ in FS.CLIENT_CLASSES] == ["MF", "INSURANCE", "PROP_BROKER", "FUND"]
    assert FS.INSTITUTION == ("MF", "INSURANCE", "FUND")
    FS.classify.cache_clear()
    assert [FS.classify(n) for n in want] == list(want.values())              # no state: same answer again


def t_net_flows_round_trip():
    """a client's buys and sells of one session net; a same-day round trip does not qualify"""
    days = _days(5)
    with _Flow(days, deals=[(days[1], "ABC", MF, "BUY", 12.0), (days[1], "ABC", MF, "SELL", 11.5),
                            (days[1], "ABC", "HDFC MUTUAL FUND", "BUY", 6.0),
                            (days[1], "ABC", "HDFC  Mutual Fund", "BUY", 5.0)]):
        nf = FS.net_flows(F.deals("ABC", days[2]))
        assert [(c, round(n, 6)) for _d, c, n in nf] == [("HDFC MUTUAL FUND", 11.0), (MF, 0.5)]
        assert FS.inst_events("ABC", days[2], days[1], 10.0, +1) == [(days[1], "HDFC MUTUAL FUND")]
        assert FS.inst_events("ABC", days[1], days[0], 10.0, +1) == []         # not visible on its own date
    # intra-group transfer: equal quantity and price, same first word -> both legs dropped (not a flow)
    with _Flow(days, deals=[(days[1], "ABC", "ISHARES CORE EMERGING MARKETS MAURITIUS CO", "SELL", 40.0),
                            (days[1], "ABC", "ISHARES CORE MSCI EMERGING MARKETS ETF", "BUY", 40.0),
                            (days[1], "ABC", "THE VANGUARD GROUP EMERGING FUND", "BUY", 30.0),
                            (days[1], "ABC", "THE ABC PROMOTER HOLDINGS", "SELL", 30.0)]):
        r = F.deals("ABC", days[2])
        assert list(FS.group_transfer_mask(r)) == [True, True, False, False] or \
            sorted(FS.group_transfer_mask(r)) == [False, False, True, True]
        assert FS.inst_events("ABC", days[2], days[1], 10.0, +1) == [(days[1], "THE VANGUARD GROUP EMERGING FUND")]
        assert FS.inst_events("ABC", days[2], days[1], 10.0, -1) == []


# ── deals: availability ─────────────────────────────────────────────────────
def t_deal_next_open_availability():
    """a deal dated D fills at D+1's open (never D); smaller / prop / round-trip deals never trade; later rows
    never change an earlier trade; the guard clips any read past the decision evening"""
    days = _days(60)
    sd = _sd("ABC", days, 100 + np.arange(60) * 0.1)
    sp = FS.BY_ID["fl_inst_buy_h20"]
    for dd in (days[10], days[25]):
        with _Flow(days, deals=[(dd, "ABC", MF, "BUY", 15.0)]):
            tr, _ = _run(sp, sd)
            i = days.index(dd)
            assert len(tr) == 1 and tr[0][2] == days[i + 1] and tr[0][6] == days[i + 20], tr
            assert tr[0][13] == "max_hold" and tr[0][3] == 1
    for deals in ([(days[10], "ABC", MF, "BUY", 9.9)], [(days[10], "ABC", "GRAVITON RESEARCH CAPITAL LLP", "BUY", 90)],
                  [(days[10], "ABC", "RADHAKISHAN DAMANI", "BUY", 90)],
                  [(days[10], "ABC", MF, "BUY", 50.0), (days[10], "ABC", MF, "SELL", 45.0)],
                  [(days[10], "XYZ", MF, "BUY", 50.0)]):
        with _Flow(days, deals=deals):
            assert _run(sp, sd)[0] == [], deals
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 6.0), (days[10], "ABC", MF, "BUY", 6.0)]):
        assert [t[2] for t in _run(sp, sd)[0]] == [days[11]]                  # netted within the session
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 15.0)]):
        a = _run(sp, sd)[0]
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 15.0), (days[45], "ABC", MF, "BUY", 99.0)]):
        b = _run(sp, sd)[0]
    assert b[0] == a[0] and len(b) == 2
    from . import flows_run as FR
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 15.0)]):
        peek = FR._Guard(lambda d, p: len(F.deals("ABC", 99999999)))
        assert peek(_View("ABC", days[:10]), None) == 0                       # decision on D-1: the deal is unseen
        assert peek(_View("ABC", days[:11]), None) == 1
        with N.cutoff(days[10]):
            assert FS.inst_events("ABC", days[11], days[10], 10.0, +1) == []


class _View:
    def __init__(self, sym, dates):
        self.symbol, self.date = sym, np.array(dates)

    def __len__(self):
        return len(self.date)


def t_sell_short_needs_future_and_cluster():
    """sell -> short only with a listed stock future; cluster = 2 distinct names within 10 sessions, today adds one"""
    days = _days(60)
    sd = _sd("ABC", days, 100 + np.arange(60) * 0.1)
    sh = FS.BY_ID["fl_inst_sell_short_h20"]
    with _Flow(days, deals=[(days[10], "ABC", MF, "SELL", 20.0)]):
        assert _run(sh, sd)[0] == []
    with _Flow(days, deals=[(days[10], "ABC", MF, "SELL", 20.0)], fo={days[0]: frozenset({"ABC"})}):
        tr = _run(sh, sd)[0]
        assert len(tr) == 1 and tr[0][3] == -1 and tr[0][2] == days[11]
    cl = FS.BY_ID["fl_inst_cluster_h20"]
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 20.0), (days[15], "ABC", "LIFE INSURANCE CORPORATION OF INDIA",
                                                                 "BUY", 20.0)]):
        assert [t[2] for t in _run(cl, sd)[0]] == [days[16]]
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 20.0), (days[15], "ABC", MF, "BUY", 20.0)]):
        assert _run(cl, sd)[0] == []                                           # the same name twice
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 20.0), (days[20], "ABC", "HDFC MUTUAL FUND", "BUY", 20.0)]):
        assert _run(cl, sd)[0] == []                                           # 10 sessions apart: outside
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 20.0), (days[10], "ABC", "HDFC MUTUAL FUND", "BUY", 20.0)]):
        assert [t[2] for t in _run(cl, sd)[0]] == [days[11]]                   # same block, two buyers


# ── index changes ───────────────────────────────────────────────────────────
def t_index_announcement_timing():
    """run-up: buy at the open after the announcement session, sell at the close of the session before the
    effective date; a weekend release fills one session later; effective-date variants fill the open after it"""
    days = _days(80)
    sd = _sd("ABC", days, 100 + np.arange(80) * 0.1)
    ru, rv = FS.BY_ID["fl_idx_add_runup"], FS.BY_ID["fl_idx_add_reversal"]
    fo = {days[0]: frozenset({"ABC"})}
    with _Flow(days, changes=[("NIFTY50", days[10], days[30], "IN", "ABC", "press", "day")], fo=fo):
        tr = _run(ru, sd)[0]
        assert len(tr) == 1 and tr[0][2] == days[11] and tr[0][6] == days[29] and tr[0][13] == "max_hold", tr
        tv = _run(rv, sd)[0]
        assert len(tv) == 1 and tv[0][3] == -1 and tv[0][2] == days[31] and tv[0][6] == days[50], tv
    sat = int((dt.date(int(str(days[9])[:4]), int(str(days[9])[4:6]), int(str(days[9])[6:])) +
               dt.timedelta(days=1)).strftime("%Y%m%d"))
    fri = days[9]
    if dt.date(fri // 10000, fri // 100 % 100, fri % 100).weekday() == 4:   # a Friday: the release on Saturday
        with _Flow(days, changes=[("NIFTY50", sat, days[30], "IN", "ABC", "press", "day")], fo=fo):
            tr = _run(ru, sd)[0]
            assert tr[0][2] == days[11], tr                                    # decided Monday, filled Tuesday
    with _Flow(days, changes=[("NIFTY50", days[29], days[30], "IN", "ABC", "press", "day")], fo=fo):
        assert _run(ru, sd)[0] == []                                           # no session left before effective
    with _Flow(days, changes=[("NIFTY50", days[10], days[30], "IN", "ABC", "press", "day")]):
        assert _run(rv, sd)[0] == []                                           # short needs a stock future


def t_runup_excludes_rows_without_real_announcement():
    """announce_source != press: no run-up trade, but the effective-date variant trades; month precision: never"""
    days = _days(80)
    sd = _sd("ABC", days, 100 + np.arange(80) * 0.1)
    fo = {days[0]: frozenset({"ABC"})}
    ru, rv = FS.BY_ID["fl_idx_add_runup"], FS.BY_ID["fl_idx_add_reversal"]
    dn, rb = FS.BY_ID["fl_idx_del_rundown"], FS.BY_ID["fl_idx_del_rebound"]
    none = "none: set to the session before the effective date"
    with _Flow(days, changes=[("NIFTY50", days[29], days[30], "IN", "ABC", none, "inferred"),
                              ("NIFTY100", days[39], days[40], "OUT", "ABC", none, "day")], fo=fo):
        assert _run(ru, sd)[0] == [] and _run(dn, sd)[0] == []
        assert [t[2] for t in _run(rv, sd)[0]] == [days[31]]
        assert [t[2] for t in _run(rb, sd)[0]] == [days[41]]
    with _Flow(days, changes=[("NIFTY50", days[10], days[30], "IN", "ABC", "press", "month"),
                              ("NIFTY50", days[10], days[30], "OUT", "ABC", "press", "month")], fo=fo):
        for sp in (ru, rv, dn, rb):
            assert _run(sp, sd)[0] == [], sp.id
    with _Flow(days, changes=[("NIFTY100", days[10], days[30], "OUT", "ABC", "press", "day")], fo=fo):
        tr = _run(dn, sd)[0]
        assert len(tr) == 1 and tr[0][3] == -1 and tr[0][2] == days[11] and tr[0][6] == days[29]


def t_index_pooling_and_dedupe():
    """NIFTY 50 + NIFTY 100 pooled, one event per (stock, effective date); NIFTY Next 50 rows are not events"""
    days = _days(80)
    with _Flow(days, changes=[("NIFTY50", days[10], days[30], "IN", "ABC", "press", "day"),
                              ("NIFTY100", days[10], days[30], "IN", "ABC", "press", "day"),
                              ("NIFTYNEXT50", days[10], days[30], "IN", "XYZ", "press", "day"),
                              ("NIFTYNEXT50", days[10], days[30], "OUT", "ABC", "press", "day")]):
        assert FS.index_events("ABC", days[60], "add", 0, True) == [(days[10], days[30])]
        assert FS.index_events("XYZ", days[60], "add", 0, False) == []
        assert FS.index_events("ABC", days[60], "del", 0, False) == []
        assert FS.index_events("ABC", days[10], "add", 0, True) == []          # announced that evening


def t_bracket_maths_report_only_twins():
    """1:3 twins: target = 3 x stop distance from the fill; same entry, same time limit"""
    assert len(FS.SPECS) == 8 and len(FS.RR3) == 8 and len({s.id for s in FS.SPECS + FS.RR3}) == 16
    for sp, tw in zip(FS.SPECS, FS.RR3):
        assert tw.id == sp.id + "_rr3" and tw.fn is sp.fn and tw.fingerprint != sp.fingerprint
        assert tw.params["target_pct"] == 3 * tw.params["stop_pct"] and tw.params.get("hold") == sp.params.get("hold")
        assert sp.params["stop_pct"] == 20.0
    days = _days(60)
    up = np.concatenate((np.full(12, 100.0), 100.0 * 1.02 ** np.arange(1, 49)))
    dn = np.concatenate((np.full(12, 100.0), 100.0 * 0.98 ** np.arange(1, 49)))
    tw = next(t for t in FS.RR3 if t.id == "fl_inst_buy_h20_rr3")
    with _Flow(days, deals=[(days[10], "ABC", MF, "BUY", 20.0)]):
        a = _run(tw, _sd("ABC", days, up))[0][0]
        b = _run(tw, _sd("ABC", days, dn))[0][0]
    assert a[13] == "target" and abs(a[8] / a[5] - 1.15) < 1e-9
    assert b[13] == "stop" and b[8] / b[5] <= 0.95 + 1e-9 and a[2] == b[2] == days[11]


def t_placebo_same_stocks_same_count():
    """placebo: every (stock, session) deal group moves together to another session of the same stock and era;
    index groups keep their session gap"""
    from . import flows_run as FR
    days = _days(120)
    deals = [(days[10], "ABC", MF, "BUY", 20.0), (days[10], "ABC", MF, "SELL", 19.0), (days[50], "XYZ", MF, "BUY", 5)]
    changes = [("NIFTY50", days[10], days[30], "IN", "ABC", "press", "day"),
               ("NIFTY100", days[10], days[30], "IN", "ABC", "press", "day")]
    old_ctx, old_sess = dict(FR._CTX), FR._sessions_of
    try:
        with _Flow(days, deals=deals, changes=changes), tempfile.TemporaryDirectory() as td:
            FR._CTX["cal"] = days
            FR._sessions_of = lambda sym, cache: days
            n = FR.placebo_tables(td, F.FLOW_DIR, seed=7)
            assert n["deal_rows"] == 3 and n["deal_groups"] == 2 and n["index_rows"] == 2
            with gzip.open(os.path.join(td, "deals", "deals.csv.gz"), "rt") as f:
                rr = list(csv.DictReader(f))
            abc = [r for r in rr if r["lab_symbol"] == "ABC"]
            assert len(abc) == 2 and abc[0]["date"] == abc[1]["date"] and len(rr) == 3
            ch = list(csv.DictReader(open(os.path.join(td, "index_membership", "index_changes.csv"))))
            assert len({(r["announce_date"], r["effective_date"]) for r in ch}) == 1
            a, e = int(ch[0]["announce_date"]), int(ch[0]["effective_date"])
            assert days.index(e) - days.index(a) == 20
    finally:
        FR._CTX.clear()
        FR._CTX.update(old_ctx)
        FR._sessions_of = old_sess


def t_account_and_ledger_append():
    """the Rs 10 lakh account sizes by 1 % risk from the 20 % stop; a stock-future short does not fit; old-era rows
    append once to a (temporary) trials ledger"""
    from . import events_run as ER
    from . import trials as TR
    days = _days(60)
    sp = FS.BY_ID["fl_inst_buy_h20"]
    tr = [(sp.id, "ABC", days[5], 1, "09:15", 100.0, days[24], "15:29", 110.0, 1000, 10000.0, 200.0, 9800.0, "max_hold")]
    acc = ER.account(tr, sp, days, days[0], days[-1])
    assert acc["trades"] == 1 and 4500 < acc["net"] < 5000
    fut = FS.BY_ID["fl_idx_add_reversal"]
    accf = ER.account([(fut.id, "ABC", days[5], -1, "09:15", 100.0, days[24], "15:29", 90.0, 1, 0, 0, 0, "max_hold")],
                      fut, days, days[0], days[-1])
    assert accf["trades"] == 0 and accf["skipped"].get("below_one_lot") == 1
    with tempfile.TemporaryDirectory() as td:
        led = os.path.join(td, "trials.csv")
        res = [{"id": sp.id, "spec": sp, "verdict": "OOS-FAILED", "why": "x", "trades": tr}]
        rows = ER.ledger_rows_old(res, days, era=(days[0], days[-1]), source="flows/oos_ledger.json")
        assert len(rows) == 1 and rows[0]["stage"] == "oos" and rows[0]["source"].startswith("flows/")
        assert TR.TrialsLedger(led).append(rows) == 1 and TR.TrialsLedger(led).append(rows) == 0
        assert TR.count_trials(TR.TrialsLedger(led).read(), "2012-23") == 1


def t_prereg_matches_code():
    """the committed pre-registration lists exactly these 8 variants, fingerprints and the keyword list"""
    from . import flows_run as FR
    assert len(FS.SPECS) <= 8
    assert all(sp.id not in __import__("stratlab.api", fromlist=["REGISTRY"]).REGISTRY for sp in FS.SPECS)
    if not os.path.exists(FR.REPO_PREREG):
        return
    with open(FR.REPO_PREREG) as f:
        pre = json.load(f)
    assert pre["n_variants"] == len(FS.SPECS) == len(pre["variants"])
    for sp in list(FS.SPECS) + list(FS.RR3):
        assert pre["strategies"][sp.id]["fingerprint"] == sp.fingerprint, sp.id
    assert [(x["class"], x["regex"]) for x in pre["client_classes"]["ordered_keywords"]] == \
        [(c, list(p)) for c, p in FS.CLIENT_CLASSES]
    assert pre["gates"]["screen"]["min_pf"] == 1.15 and pre["gates"]["dsr_min"] == 0.95
    assert pre["index_pooling"]["pooled"] is True
    assert pre["code_sha256"]["flows_strategies.py"] == FR.code_sha()["flows_strategies.py"]


TESTS = [(f.__name__, f) for f in (
    t_client_classifier_fixed_and_deterministic, t_net_flows_round_trip, t_deal_next_open_availability,
    t_sell_short_needs_future_and_cluster, t_index_announcement_timing,
    t_runup_excludes_rows_without_real_announcement, t_index_pooling_and_dedupe, t_bracket_maths_report_only_twins,
    t_placebo_same_stocks_same_count, t_account_and_ledger_append, t_prereg_matches_code)]


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
