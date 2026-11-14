"""
stratlab/tests_events.py — tests for the event study (events_parse / events_data / events_daily / events_strategies /
events_run) on SYNTHETIC data only: timestamp parsing (no guessed times), symbol renames, insider rows from both
feed formats and their de-duplication, the first-public-time rule for results, the availability rule (strictly
before the order cut-off; LookAheadError under a cutoff), in-hours vs after-hours first tradable session, every
strategy's trigger and that a later broadcast moves the entry later (the transaction date is never used), the
pre-earnings calendar arithmetic, the intraday earnings-gap rule and its look-ahead audit, the 1:3 bracket maths,
the corporate-action rules (factor parsing, gap check, trade exclusion), the turnover universe, the event-study
bootstrap, the trials-ledger append and that the frozen pre-registration still matches the code.
Run by test_pipeline.py (section "STRATLAB: event study") or directly: python -m stratlab.tests_events
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import tempfile

import numpy as np

from . import data as D
from . import events_daily as ED
from . import events_data as E
from . import events_parse as EP
from . import events_strategies as ES
from . import sim


# ── helpers ─────────────────────────────────────────────────────────────────
def _days(n, start=dt.date(2024, 1, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(int(d.strftime("%Y%m%d")))
        d += dt.timedelta(days=1)
    return out


def _daily_sd(sym, days, close, vol=None, open_=None):
    c = np.asarray(close, float)
    o = np.asarray(open_, float) if open_ is not None else np.concatenate(([c[0]], c[:-1]))
    h, l = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
    v = np.asarray(vol, float) if vol is not None else np.full(len(c), 1000.0)
    sd = D.from_arrays(sym, np.array(days), np.full(len(days), 555), o, h, l, c, v, min_bars=1)
    sd.member = np.ones(len(days), bool)
    return sd


def _ts(day, hhmm):
    return f"{day // 10000:04d}-{day // 100 % 100:02d}-{day % 100:02d} {hhmm // 100:02d}:{hhmm % 100:02d}:00"


class _Tables:
    """Temporary event tables + calendar; restores the module state on exit."""

    def __init__(self, days, insider=(), results=(), boardmeet=(), fo=None, exclusions=()):
        self.days, self.ins, self.res, self.bm, self.fo, self.exc = days, insider, results, boardmeet, fo, exclusions

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_dir, self.old_cal = E.EVENTS_DIR, list(E.calendar())
        ins = [{"key": f"K{i}", "symbol": r[0], "person": r[5] if len(r) > 5 else "p", "person_cat": r[4] if len(r) > 4
                else "Promoters", "is_promoter": int("promoter" in (r[4] if len(r) > 4 else "Promoters").lower()),
                "mode": r[3] if len(r) > 3 else "Market Purchase", "txn": "Buy", "sec_type": "Equity Shares",
                "qty": 1000, "value": r[2], "from_date": "2020-01-01", "to_date": "2020-01-01", "broadcast": r[1]}
               for i, r in enumerate(self.ins)]
        EP.write_csv(os.path.join(self.tmp.name, "insider.csv.gz"), EP.INS_COLS, ins)
        EP.write_csv(os.path.join(self.tmp.name, "results.csv.gz"), EP.RES_COLS,
                     [{"symbol": s, "period_end": "2020-03-31", "first_ts": t} for s, t in self.res])
        EP.write_csv(os.path.join(self.tmp.name, "boardmeet.csv.gz"), EP.BM_COLS,
                     [{"symbol": s, "meeting_date": m, "purpose": "Financial Results", "is_results": 1, "notice_ts": t}
                      for s, m, t in self.bm])
        E.use(self.tmp.name)
        E.set_calendar(self.days)
        E.set_fo(self.fo or {})
        E.set_exclusions(self.exc)
        return self

    def __exit__(self, *a):
        E.use(self.old_dir)
        E.set_calendar(self.old_cal)
        E.set_fo({})
        E.set_exclusions([])
        self.tmp.cleanup()


def _run(spec, sd, lo=0, hi=99999999, nifty=None):
    from . import events_run as ER
    return sim.sim_positional(ER.guarded(spec, sd), sd, lo, hi, nifty, None, 0.0)


# ── parsing ─────────────────────────────────────────────────────────────────
def t_parse_timestamps_never_guessed():
    """broadcast times parse in both NSE formats; a bare date is never promoted to a time"""
    assert EP.parse_ts("10-Jan-2024 21:21") == dt.datetime(2024, 1, 10, 21, 21)
    assert EP.parse_ts("31-JAN-2013 18:18:03") == dt.datetime(2013, 1, 31, 18, 18, 3)
    assert EP.parse_ts("10-Jan-2024") is None and EP.parse_ts("-") is None and EP.parse_ts(None) is None
    assert EP.parse_ts("31-Feb-2024 10:00") is None
    assert EP.parse_date("30-JUN-2025") == dt.date(2025, 6, 30)
    assert EP.insider_legacy_row({"date": "10-Jan-2024", "symbol": "X"}, []) is None      # no time: dropped
    assert E.ts_stamp("2024-01-10 21:21:00") == 202401102121 and E.stamp(20240110, 900) == 202401100900


def t_symbol_renames():
    """a filing's symbol is followed to the current one; a rename dated before the filing is not applied"""
    ch = [(20200101, "AAA", "BBB"), (20220101, "BBB", "CCC")]
    assert EP.canon("AAA", 20190101, ch) == "CCC" and EP.canon("BBB", 20210101, ch) == "CCC"
    assert EP.canon("AAA", 20210101, ch) == "AAA"            # AAA seen after its rename = another company
    assert EP.canon(" zzz ", 20190101, ch) == "ZZZ"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "symbolchange.csv")
        with open(p, "w") as f:
            f.write("NAME,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE\nZom,ZOMATO,ETERNAL,20-MAR-2025\nbad,ONLY3\n")
        assert EP.load_changes(p) == [(20250320, "ZOMATO", "ETERNAL")]
        assert EP.canon("ZOMATO", 20240101, EP.load_changes(p)) == "ETERNAL"


def t_insider_rows_and_dedupe():
    """legacy rows: promoter flag, broadcast time; the same transaction re-filed keeps the EARLIEST broadcast"""
    base = {"did": "1", "symbol": "ABC", "company": "A", "acqName": "Mr P", "personCategory": "Promoter Group",
            "acqMode": "Market Purchase", "tdpTransactionType": "Buy", "secType": "Equity Shares", "secAcq": "1000",
            "secVal": "25000000", "acqfromDt": "08-Jan-2024", "acqtoDt": "08-Jan-2024", "intimDt": "09-Jan-2024",
            "date": "10-Jan-2024 21:21", "exchange": "NSE"}
    a = EP.insider_legacy_row(base, [])
    assert a["is_promoter"] == 1 and a["broadcast"] == "2024-01-10 21:21:00" and a["value"] == 25e6
    assert a["to_date"] == "2024-01-08"                       # the trade date is kept as content only
    b = EP.insider_legacy_row(dict(base, did="2", date="11-Jan-2024 09:30"), [])       # revised filing
    c = EP.insider_legacy_row(dict(base, did="3", personCategory="Employees/Designated Employees",
                                   acqName="Ms E"), [])
    out, n = EP.dedupe_insider([b, c, a])
    assert n == 1 and [r["key"] for r in out] == ["L1", "L3"] and out[1]["is_promoter"] == 0
    assert EP.is_promoter("Promoters") and EP.is_promoter("Member of Promoter Group") and not EP.is_promoter("Director")
    assert EP.is_equity("Equity Shares") and EP.is_equity("Equity") and not EP.is_equity("Warrants")


def t_insider_document_format():
    """the 2026 per-filing XBRL document: one row per disclosure context, broadcast taken from the INDEX row"""
    ns = 'xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:c="http://www.bseindia.com/xbrl/co"'
    fact = lambda n, v, ctx: f'<c:{n} contextRef="{ctx}">{v}</c:{n}>'
    body = "".join([
        fact("Symbol", "ABC", "MainI"),
        fact("NameOfThePerson", "Mr P", "D1"), fact("CategoryOfPerson", "Promoter", "D1"),
        fact("TypeOfInstrument", "Equity", "D1"), fact("SecuritiesAcquiredOrDisposedNumberOfSecurity", "500", "D1"),
        fact("SecuritiesAcquiredOrDisposedValueOfSecurity", "12000000", "D1"),
        fact("SecuritiesAcquiredOrDisposedTransactionType", "Buy", "D1"),
        fact("ModeOfAcquisitionOrDisposal", "Market Purchase", "D1"),
        fact("DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate", "2026-05-18", "D1"),
        fact("NameOfThePerson", "Ms E", "D2"), fact("CategoryOfPerson", "Employee", "D2"),
        fact("SecuritiesAcquiredOrDisposedNumberOfSecurity", "10", "D2"),
        fact("ModeOfAcquisitionOrDisposal", "Market Sale", "D2")])
    xml = f'<?xml version="1.0"?><xbrli:xbrl {ns}>{body}</xbrli:xbrl>'.encode()
    idx = {"appId": "77", "symbol": "ABC", "broadcastDateTime": "21-May-2026 19:02:11", "companyName": "A"}
    rows = EP.insider_doc_rows(xml, idx, [])
    assert len(rows) == 2 and rows[0]["broadcast"] == "2026-05-21 19:02:11"
    assert rows[0]["is_promoter"] == 1 and rows[0]["value"] == 12e6 and rows[0]["to_date"] == "2026-05-18"
    assert rows[1]["is_promoter"] == 0 and rows[1]["mode"] == "Market Sale"
    assert EP.insider_doc_rows(xml, {"symbol": "ABC"}, []) == []                     # no broadcast time: nothing
    assert EP.insider_doc_rows(b'<!DOCTYPE x [<!ENTITY a "b">]><x/>', idx, []) == []    # DTDs are refused
    assert EP.insider_doc_rows(b"not xml", idx, []) == []


def t_results_first_public_time():
    """first_ts = the earliest of the structured filing and the qualifying announcements after the period end"""
    fr = [{"symbol": "ABC", "toDate": "31-Dec-2023", "broadCastDate": "27-Jan-2024 11:00:00", "isin": "I"},
          {"symbol": "ABC", "toDate": "31-Dec-2023", "broadCastDate": "26-Jan-2024 18:00:00"},
          {"symbol": "LATE", "toDate": "31-Dec-2022", "broadCastDate": "26-Jan-2024 18:00:00"},   # re-filing: no event
          {"symbol": "NOT", "toDate": "31-Dec-2023", "broadCastDate": "bad"}]
    ifr = [{"symbol": "NEW", "qe_Date": "30-JUN-2025", "broadcast_Date": "05-Aug-2025 22:48:33"}]
    a_fr = [{"symbol": "ABC", "an_dt": "25-Jan-2024 15:45:10"},          # the results PDF: first public moment
            {"symbol": "ABC", "an_dt": "20-Dec-2023 10:00:00"},          # before the period end: ignored
            {"symbol": "ABC", "an_dt": "28-Jan-2024 10:00:00"}]          # after the structured filing: ignored
    a_obm = [{"symbol": "ABC", "an_dt": "10-Jan-2024 12:00:00"},         # not a results meeting date: ignored
             {"symbol": "NEW", "an_dt": "05-Aug-2025 14:10:00"}]         # results meeting day: counts
    meet = {"NEW": {dt.date(2025, 8, 5)}, "ABC": {dt.date(2024, 1, 25)}}
    out, rep = EP.build_results(fr, a_fr, a_obm, [], ifr, meet)
    by = {r["symbol"]: r for r in out}
    assert set(by) == {"ABC", "NEW"} and rep["fr_no_ts"] == 1 and rep["fr_no_period"] == 1
    assert by["ABC"]["first_ts"] == "2024-01-25 15:45:10" and by["ABC"]["fr_ts"] == "2024-01-26 18:00:00"
    assert by["ABC"]["src_first"] == "ann" and by["ABC"]["n_fr"] == 2
    assert by["NEW"]["first_ts"] == "2025-08-05 14:10:00" and by["NEW"]["period_end"] == "2025-06-30"
    out2, _ = EP.build_results(fr, [], a_obm, [], ifr, {})               # no meeting list: OBM never qualifies
    assert {r["symbol"]: r["src_first"] for r in out2} == {"ABC": "fr", "NEW": "fr"}


# ── availability ────────────────────────────────────────────────────────────
def t_availability_strictly_before():
    """a row is usable only if broadcast strictly before the stamp; the cutoff guard raises on a later request"""
    days = _days(10)
    d3, d4 = days[3], days[4]
    with _Tables(days, insider=[("ABC", _ts(d3, 1800), 2e7), ("ABC", _ts(d4, 859), 1e7), ("ABC", _ts(d4, 900), 5e6)]):
        assert len(E.insider("ABC", E.stamp(d4, E.ORDER_CUTOFF))) == 2          # 18:00 the evening before, 08:59
        assert len(E.insider("ABC", E.stamp(d4, 901))) == 3
        assert len(E.insider("ABC", E.stamp(d3, 1800))) == 0                     # strictly before
        assert len(E.insider("ABC", E.stamp(d4, 900), since=E.stamp(d4, 0))) == 1
        assert len(E.insider("NOPE", E.stamp(d4, 900))) == 0
        with E.cutoff(E.stamp(d4, E.ORDER_CUTOFF)):
            assert len(E.insider("ABC", E.stamp(d4, E.ORDER_CUTOFF))) == 2
            try:
                E.insider("ABC", E.stamp(d4, 901))
                raise AssertionError("a read past the cutoff must raise")
            except E.LookAheadError:
                pass
        assert E.next_session(d3) == d4 and E.prev_session(d4) == d3 and E.next_session(days[-1]) is None
        assert E.session_index(d4) == 4


def t_in_hours_vs_after_hours_reaction_session():
    """results before 15:30 belong to that session's reaction; at / after 15:30, or on a weekend, to the next"""
    days = _days(15)
    fri, mon = days[4], days[5]
    sat = int((dt.date(fri // 10000, fri // 100 % 100, fri % 100) + dt.timedelta(days=1)).strftime("%Y%m%d"))
    sd = _daily_sd("ABC", days, np.full(15, 100.0))
    cases = {"in_hours": (_ts(fri, 1400), fri), "last_minute": (_ts(fri, 1529), fri), "at_close": (_ts(fri, 1530), mon),
             "evening": (_ts(fri, 2130), mon), "saturday": (_ts(sat, 1100), mon), "pre_open": (_ts(mon, 815), mon)}
    for name, (ts, want) in cases.items():
        with _Tables(days, results=[("ABC", ts)]):
            hit = []
            for i in range(1, len(days)):
                v = sim.DailyView(sd.days[:i + 1], *[sd.daily()[k][:i + 1] for k in "ohlcv"])
                v.symbol = "ABC"
                if len(E.asof_swing(v).results_reacted_today()):
                    hit.append(days[i])
            assert hit == [want], (name, hit, want)
    # intraday: only results out after the previous close and before 09:00 count as "overnight"
    class S:
        symbol, date = "ABC", mon
    for ts, want in ((_ts(fri, 1400), 0), (_ts(fri, 1600), 1), (_ts(sat, 1100), 1), (_ts(mon, 859), 1),
                     (_ts(mon, 900), 0), (_ts(mon, 1100), 0)):
        with _Tables(days, results=[("ABC", ts)]):
            assert len(E.asof_session(S).results_before_open()) == want, ts


# ── strategies ──────────────────────────────────────────────────────────────
def t_promoter_buy_enters_after_broadcast_only():
    """entry = the first open after the broadcast (never the transaction date); a later broadcast moves it later"""
    days = _days(60)
    sd = _daily_sd("ABC", days, 100 + np.arange(60) * 0.1)
    sp = ES.BY_ID["ev_promoter_buy_h20"]
    for bc_day, hhmm, want_entry in ((days[10], 1800, days[11]), (days[11], 859, days[11]), (days[11], 905, days[12]),
                                     (days[20], 1200, days[21])):
        with _Tables(days, insider=[("ABC", _ts(bc_day, hhmm), 2e7)]):       # from_date / to_date are 2020-01-01
            tr, _ = _run(sp, sd)
            assert len(tr) == 1 and tr[0][2] == want_entry, (bc_day, hhmm, tr)
            assert tr[0][6] == days[days.index(want_entry) + 19] and tr[0][13] == "max_hold"
    with _Tables(days, insider=[("ABC", _ts(days[10], 1800), 6e6)]):
        assert _run(sp, sd)[0] == []                                           # below Rs 1 crore
    with _Tables(days, insider=[("ABC", _ts(days[10], 1800), 6e6), ("ABC", _ts(days[18], 1800), 6e6)]):
        tr, _ = _run(sp, sd)
        assert [t[2] for t in tr] == [days[19]]                                # the cluster reaches 1 cr on the 2nd
    with _Tables(days, insider=[("ABC", _ts(days[2], 1800), 6e6), ("ABC", _ts(days[30], 1800), 6e6)]):
        assert _run(sp, sd)[0] == []                                           # 28 sessions apart: > 30 calendar days
    for cat, mode in (("Director", "Market Purchase"), ("Promoters", "Pledge Creation"), ("Promoters", "ESOP")):
        with _Tables(days, insider=[("ABC", _ts(days[10], 1800), 5e7, mode, cat)]):
            assert _run(sp, sd)[0] == [], (cat, mode)
    # causality: adding a LATER disclosure never changes an earlier trade
    with _Tables(days, insider=[("ABC", _ts(days[10], 1800), 2e7)]):
        a = _run(sp, sd)[0]
    with _Tables(days, insider=[("ABC", _ts(days[10], 1800), 2e7), ("ABC", _ts(days[40], 1800), 9e7)]):
        b = _run(sp, sd)[0]
    assert b[0] == a[0] and len(b) == 2


def t_promoter_dip_and_short():
    """the dip variant needs the close >= 20 % below the 252-session high; the short needs a listed stock future"""
    days = _days(320)
    up = 100 + np.arange(320) * 0.05
    dip = up.copy()
    dip[280:] = up[279] * 0.75
    ev = [("ABC", _ts(days[300], 1800), 3e7)]
    sp = ES.BY_ID["ev_promoter_buy_dip_h60"]
    with _Tables(days, insider=ev):
        assert _run(sp, _daily_sd("ABC", days, up))[0] == []
        tr, _ = _run(sp, _daily_sd("ABC", days, dip))
        assert len(tr) == 1 and tr[0][2] == days[301]
    with _Tables(days, insider=ev, exclusions=[("ABC", days[200])]):
        assert _run(sp, _daily_sd("ABC", days, dip))[0] == []                  # unadjusted action inside the window
    sell = [("ABC", _ts(days[300], 1800), 3e7, "Market Sale")]
    ss = ES.BY_ID["ev_promoter_sell_short_h20"]
    with _Tables(days, insider=sell):
        assert _run(ss, _daily_sd("ABC", days, up))[0] == []                   # no future listed
    with _Tables(days, insider=sell, fo={days[299]: frozenset(["ABC"])}):
        tr, _ = _run(ss, _daily_sd("ABC", days, up))
        assert len(tr) == 1 and tr[0][3] == -1 and tr[0][2] == days[301]
    with _Tables(days, insider=sell, fo={days[299]: frozenset(["ABC"]), days[300]: frozenset(["XYZ"])}):
        assert _run(ss, _daily_sd("ABC", days, up))[0] == []                   # dropped from F&O by the decision day


def t_pead_long_and_short():
    """reaction-day abnormal return >= 4 % on above-average volume -> next open; the mirror shorts F&O names"""
    days = _days(80)
    c = np.full(80, 100.0)
    c[40:] = 106.0                                                             # +6 % on session 40
    v = np.full(80, 1000.0)
    v[40] = 3000.0
    nifty = {"date": np.array(days), "c": np.full(80, 20000.0)}
    res = [("ABC", _ts(days[39], 1700))]                                       # after the close of session 39
    lp, sp = ES.BY_ID["ev_pead_long_h20"], ES.BY_ID["ev_pead_short_h20"]
    with _Tables(days, results=res):
        tr, _ = _run(lp, _daily_sd("ABC", days, c, v), nifty=nifty)
        assert len(tr) == 1 and tr[0][2] == days[41] and tr[0][3] == 1 and tr[0][6] == days[60]
        assert _run(lp, _daily_sd("ABC", days, c), nifty=nifty)[0] == []       # volume not above average
        n2 = {"date": np.array(days), "c": np.where(np.arange(80) >= 40, 20600.0, 20000.0)}
        assert _run(lp, _daily_sd("ABC", days, c, v), nifty=n2)[0] == []       # the index rose 3 %: abnormal < 4 %
    with _Tables(days, results=[("ABC", _ts(days[30], 1700))]):
        assert _run(lp, _daily_sd("ABC", days, c, v), nifty=nifty)[0] == []    # the jump is not a results reaction
    dn = np.full(80, 100.0)
    dn[40:] = 94.0
    with _Tables(days, results=res, fo={days[38]: frozenset(["ABC"])}):
        tr, _ = _run(sp, _daily_sd("ABC", days, dn, v), nifty=nifty)
        assert len(tr) == 1 and tr[0][3] == -1 and tr[0][2] == days[41]
    with _Tables(days, results=res):
        assert _run(sp, _daily_sd("ABC", days, dn, v), nifty=nifty)[0] == []
    with _Tables(days, results=res, exclusions=[("ABC", days[40])]):
        assert _run(lp, _daily_sd("ABC", days, c, v), nifty=nifty)[0] == []    # the "reaction" is an ex-day


def t_pre_earnings_calendar():
    """entry 5 sessions before the notified meeting, exit at the close of the last session before it"""
    days = _days(40)
    sd = _daily_sd("ABC", days, np.full(40, 100.0))
    sp = ES.BY_ID["ev_pre_earnings_h5"]
    iso = lambda d: f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"
    with _Tables(days, boardmeet=[("ABC", iso(days[20]), _ts(days[5], 1800))]):
        tr, _ = _run(sp, sd)
        assert len(tr) == 1 and tr[0][2] == days[15] and tr[0][6] == days[19] and tr[0][13] == "max_hold"
    sat = dt.date(days[19] // 10000, days[19] // 100 % 100, days[19] % 100)
    while sat.weekday() != 5:
        sat += dt.timedelta(days=1)
    k = max(i for i, d in enumerate(days) if d < int(sat.strftime("%Y%m%d")))
    with _Tables(days, boardmeet=[("ABC", sat.isoformat(), _ts(days[5], 1800))]):     # a Saturday meeting
        tr, _ = _run(sp, sd)
        assert len(tr) == 1 and tr[0][6] == days[k] and tr[0][2] == days[k - 4]
    with _Tables(days, boardmeet=[("ABC", iso(days[20]), _ts(days[15], 905))]):       # notified too late
        assert _run(sp, sd)[0] == []
    with _Tables(days, boardmeet=[("ABC", "2030-01-10", _ts(days[5], 1800))]):        # beyond the calendar
        assert _run(sp, sd)[0] == []


def _minute_sd(days, gap_day, gap):
    T = np.arange(555, 930)
    date = np.repeat(days, len(T))
    t = np.tile(T, len(days))
    rng = np.random.default_rng(5)
    c = np.empty(len(date))
    px = 100.0
    for i, d in enumerate(days):
        o0 = px * (1 + (gap if d == gap_day else rng.normal(0, 0.004)))
        path = o0 * np.exp(np.cumsum(rng.normal(0, 0.0003, len(T))))
        c[i * len(T):(i + 1) * len(T)] = path
        px = path[-1]
    o = np.concatenate(([c[0]], c[:-1]))
    for i in range(len(days)):
        a = i * len(T)
        o[a] = c[a]
    return D.from_arrays("ABC", date, t, o, np.maximum(o, c) * 1.0002, np.minimum(o, c) * 0.9998, c,
                         np.full(len(c), 500.0), min_bars=200)


def t_egap_intraday_rule_and_audit():
    """results out overnight + a gap >= 0.5 sigma -> one trade with the gap from 09:16; causal under the audit"""
    from .indicators import build_sessions
    days = _days(120)
    sp = ES.BY_ID["ev_egap_intraday"]
    for gap, side in ((0.03, 1), (-0.03, -1)):
        sd = _minute_sd(days, days[100], gap)
        ss = [s for s in build_sessions(sd, None)]
        s100 = next(s for s in ss if s.date == days[100])
        with _Tables(days, results=[("ABC", _ts(days[99], 1900))]):
            ents = sp.fn(s100, sp.p)
            assert len(ents) == 1 and ents[0].side == side and ents[0].i == 0
            tr, _ = sim.sim_session(sp, s100, 0.0)
            assert len(tr) == 1 and tr[0][4] == "09:16" and tr[0][13] in ("squareoff", "stop")
            assert sim.audit_lookahead(sp, s100) is None
            assert all(sp.fn(s, sp.p) == [] for s in ss if s.date != days[100])
        with _Tables(days, results=[("ABC", _ts(days[99], 1400))]):          # out during yesterday's session
            assert sp.fn(s100, sp.p) == []
        with _Tables(days, results=[("ABC", _ts(days[100], 930))]):          # out after today's open
            assert sp.fn(s100, sp.p) == []
    sd = _minute_sd(days, days[100], 0.0002)                                   # a gap far below 0.5 sigma
    s100 = next(s for s in build_sessions(sd, None) if s.date == days[100])
    with _Tables(days, results=[("ABC", _ts(days[99], 1900))]):
        assert sp.fn(s100, sp.p) == []


def t_bracket_maths_report_only_twins():
    """1:3 twins: target distance = 3 x stop distance from the fill; same entry, same time limit"""
    assert len(ES.SPECS) == 9 and len(ES.RR3) == 9 and len({s.id for s in ES.SPECS + ES.RR3}) == 18
    for sp, tw in zip(ES.SPECS, ES.RR3):
        assert tw.id == sp.id + "_rr3" and tw.fn is sp.fn and tw.fingerprint != sp.fingerprint
        if sp.horizon == "positional":
            assert tw.params["target_pct"] == 3 * tw.params["stop_pct"] and tw.params["hold"] == sp.params["hold"]
        else:
            assert tw.params["bracket"][2] == 3 * tw.params["bracket"][1]
    days = _days(60)
    up = np.concatenate((np.full(12, 100.0), 100.0 * 1.02 ** np.arange(1, 49)))
    dn = np.concatenate((np.full(12, 100.0), 100.0 * 0.98 ** np.arange(1, 49)))
    tw = ES.BY_ID["ev_promoter_buy_h20"]
    tw = next(t for t in ES.RR3 if t.id == "ev_promoter_buy_h20_rr3")
    with _Tables(days, insider=[("ABC", _ts(days[10], 1800), 2e7)]):
        a = _run(tw, _daily_sd("ABC", days, up))[0][0]
        b = _run(tw, _daily_sd("ABC", days, dn))[0][0]
    assert a[13] == "target" and abs(a[8] / a[5] - 1.15) < 1e-9             # +15 % from the fill
    assert b[13] == "stop" and b[8] / a[5] <= 0.95 + 1e-9 and a[2] == b[2] == days[11]
    win, loss = a[8] / a[5] - 1, 1 - 0.95
    assert abs(win / loss - 3.0) < 1e-6


# ── data rules ──────────────────────────────────────────────────────────────
def t_corporate_action_rules():
    """factors from the action text must agree with the ex-day gap; trades across an excluded day are dropped"""
    assert ED.parse_action("Bonus 1:1")[0] == [("bonus", 0.5)]
    assert ED.parse_action("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share")[0] == \
        [("split", 0.2)]
    assert ED.parse_action("Interim Dividend - Rs 5 Per Share") == ([], None)
    assert ED.parse_action("Demerger")[1] == "scheme" and ED.parse_action("Rights 1:5 @ Premium Rs 10/-")[1] == "rights"
    assert ED.decide_factor([("bonus", 0.5)], None, 0.51, 0.49) == (0.5, "applied")
    assert ED.decide_factor([("bonus", 0.5)], None, 0.99, 1.01)[1] == "rejected_price_check"
    assert ED.decide_factor([], "scheme", 0.60, 0.62) == (0.60, "applied_gap_scheme")
    assert ED.decide_factor([], "scheme", 0.95, 0.96) == (None, "not_applied_scheme")
    assert ED.decide_factor([], None, 0.5, 0.5)[0] is None
    from . import events_run as ER
    days = _days(30)
    with _Tables(days, exclusions=[("ABC", days[12])]):
        mk = lambda e, x, s="ABC": ("id", s, days[e], 1, "09:15", 100.0, days[x], "15:29", 101.0, 10, 10.0, 1.0, 9.0, "x")
        keep, n = ER.drop_excluded([mk(5, 10), mk(11, 14), mk(12, 13), mk(13, 20), mk(11, 14, "OTH")])
        assert n == 2 and [(t[2], t[1]) for t in keep] == [(days[5], "ABC"), (days[13], "ABC"), (days[11], "OTH")]
        assert E.ca_excluded("ABC", days[11], days[12]) and not E.ca_excluded("ABC", days[12], days[20])


def t_universe_and_renames_fast():
    """the monthly universe uses only sessions before the month; the fast rename equals data_daily's"""
    import pandas as pd
    from . import data_daily as DD
    days = _days(90, dt.date(2024, 1, 1))
    rows = []
    for d in days:
        for s, tv in (("BIG", 9e7), ("MID", 5e7), ("SML", 1e7), ("ETF", 9e9)):
            if s == "SML" and d >= 20240301:
                tv = 9e9                                   # becomes the biggest only from March
            rows.append((d, s, "EQ", tv))
    eq = pd.DataFrame(rows, columns=["date", "symbol", "series", "turnover"])
    u = ED.monthly_universe(eq, {"BIG", "MID", "SML"}, top=2)
    by = {m: list(g["symbol"]) for m, g in u.groupby("month")}
    assert by[202402] == ["BIG", "MID"] and by[202403] == ["BIG", "MID"]          # March list: data to end Feb
    assert by[202404][0] == "SML" and "ETF" not in set(u["symbol"])
    df = pd.DataFrame({"date": [20190101, 20190102, 20200105, 20200106, 20190101, 20210101],
                       "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "AAA"], "series": "EQ"})
    ch = [(20200101, "AAA", "BBB"), (20190601, "CCC", "BBB")]
    a, ap = ED.apply_renames_fast(df, ch)
    b, bp = DD.apply_renames(df.copy(), pd.DataFrame(ch, columns=["date", "old_symbol", "new_symbol"]))
    assert list(a["symbol"]) == list(b["symbol"]) and [tuple(x) for x in ap] == [tuple(x) for x in bp]
    # CCC -> BBB (2019-06) applies; AAA -> BBB (2020-01) would then overlap BBB's rows, so it is NOT a pure rename
    assert list(a["symbol"]) == ["AAA", "AAA", "BBB", "BBB", "BBB", "AAA"]
    a2, ap2 = ED.apply_renames_fast(df, ch[:1])
    assert list(a2["symbol"]) == ["BBB", "BBB", "BBB", "BBB", "CCC", "AAA"] and len(ap2) == 1


def t_event_study_bootstrap():
    """abnormal return = side x stock return minus side x NIFTY over the same window; months are the blocks"""
    from . import events_run as ER
    days = _days(60)
    nifty = {"date": np.array(days), "o": np.full(60, 100.0), "c": np.linspace(100.0, 105.9, 60)}
    mk = lambda e, x, side, en, ex: ("i", "A", days[e], side, "09:15", en, days[x], "15:29", ex, 1, 0, 0, 0, "x")
    ar = ER.abnormal_returns([mk(0, 10, 1, 100.0, 110.0), mk(30, 40, -1, 100.0, 95.0)], nifty)
    assert abs(ar[0][1] - (0.10 - (nifty["c"][10] / 100.0 - 1))) < 1e-12
    assert abs(ar[1][1] - (0.05 + (nifty["c"][40] / 100.0 - 1))) < 1e-12
    data = [(202401 + i % 6, 0.01 * (1 + i % 3), 0.0) for i in range(120)]
    b1, b2 = ER.block_bootstrap(data), ER.block_bootstrap(data)
    assert b1 == b2 and b1["n"] == 120 and b1["months"] == 6
    assert b1["ci95_mean_ar_pct"][0] <= b1["mean_ar_pct"] <= b1["ci95_mean_ar_pct"][1]
    assert ER.block_bootstrap([]) == {"n": 0}


def t_account_and_ledger_append():
    """the Rs 10 lakh account sizes by 1 % risk from the stop; old-era rows append once to the trials ledger"""
    from . import events_run as ER
    from . import trials as TR
    days = _days(60)
    sp = ES.BY_ID["ev_promoter_buy_h20"]
    tr = [(sp.id, "ABC", days[5], 1, "09:15", 100.0, days[24], "15:29", 110.0, 1000, 10000.0, 200.0, 9800.0, "max_hold")]
    acc = ER.account(tr, sp, days, days[0], days[-1])
    assert acc["trades"] == 1 and acc["candidates"] == 1
    # 1 % of Rs 10 lakh = Rs 10,000 risk, 20 % stop on Rs 100 -> 500 shares -> gross Rs 5,000 before costs
    assert 4500 < acc["net"] < 5000
    fut = ES.BY_ID["ev_pead_short_h20"]
    accf = ER.account([(fut.id, "ABC", days[5], -1, "09:15", 100.0, days[24], "15:29", 90.0, 1, 0, 0, 0, "max_hold")],
                      fut, days, days[0], days[-1])
    assert accf["trades"] == 0 and accf["skipped"].get("below_one_lot") == 1     # one lot risks far more than 1 %
    with tempfile.TemporaryDirectory() as td:
        led = os.path.join(td, "trials.csv")
        res = [{"id": sp.id, "spec": sp, "verdict": "OOS-FAILED", "why": "x", "trades": tr},
               {"id": "skip", "spec": sp, "verdict": "NOT JUDGED", "why": "y"}]
        rows = ER.ledger_rows_old(res, days, era=(days[0], days[-1]))
        assert len(rows) == 1 and rows[0]["stage"] == "oos" and rows[0]["trades"] == 1
        assert TR.TrialsLedger(led).append(rows) == 1 and TR.TrialsLedger(led).append(rows) == 0
        assert TR.count_trials(TR.TrialsLedger(led).read(), "2012-23") == 1


def t_prereg_matches_code():
    """the committed pre-registration lists exactly these variants with these fingerprints (<= 10, never retuned)"""
    from . import events_run as ER
    assert len(ES.SPECS) <= 10
    assert all(sp.id not in __import__("stratlab.api", fromlist=["REGISTRY"]).REGISTRY for sp in ES.SPECS)
    if not os.path.exists(ER.REPO_PREREG):
        return
    with open(ER.REPO_PREREG) as f:
        pre = json.load(f)
    assert pre["n_variants"] == len(ES.SPECS) == len(pre["variants"])
    for sp in list(ES.SPECS) + list(ES.RR3):
        assert pre["strategies"][sp.id]["fingerprint"] == sp.fingerprint, sp.id
    assert pre["gates"]["screen"]["min_pf"] == 1.15 and pre["gates"]["final"]["min_pf"] == 1.05
    assert pre["gates"]["dsr_min"] == 0.95


TESTS = [(f.__name__, f) for f in (
    t_parse_timestamps_never_guessed, t_symbol_renames, t_insider_rows_and_dedupe, t_insider_document_format,
    t_results_first_public_time, t_availability_strictly_before, t_in_hours_vs_after_hours_reaction_session,
    t_promoter_buy_enters_after_broadcast_only, t_promoter_dip_and_short, t_pead_long_and_short,
    t_pre_earnings_calendar, t_egap_intraday_rule_and_audit, t_bracket_maths_report_only_twins,
    t_corporate_action_rules, t_universe_and_renames_fast, t_event_study_bootstrap, t_account_and_ledger_append,
    t_prereg_matches_code)]


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
