"""
stratlab/tests_flowdata.py — synthetic tests of stratlab/flowdata.py (bulk / block deals, corporate actions, index
membership, FII statistics) and stratlab/optmin.py (option / futures minute bars): the `before` rule, the cutoff
guards and the file contracts.  No strategy is run here.  Run by test_pipeline.py (section "STRATLAB: flow data").
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from . import flowdata as F
from . import nsedata as N
from . import optmin as O


def _gz(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    op = gzip.open(path, "wt", newline="") if path.endswith(".gz") else open(path, "w", newline="")
    with op as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _flow_dir(td):
    _gz(os.path.join(td, "deals", "deals.csv.gz"),
        ["date", "kind", "symbol", "lab_symbol", "in_lab", "security", "client", "side", "qty", "price", "value_cr", "remarks"],
        [[20240102, "bulk", "ZOMATO", "ETERNAL", 1, "Zomato", "FUND A", "BUY", 1000, 100.0, 0.01, ""],
         [20240102, "bulk", "ZOMATO", "ETERNAL", 1, "Zomato", "FUND B", "SELL", 1000, 100.0, 0.01, ""],
         [20240103, "block", "ZOMATO", "ETERNAL", 1, "Zomato", "FUND C", "BUY", 5000, 101.0, 0.0505, ""],
         [20250410, "bulk", "ETERNAL", "ETERNAL", 1, "Eternal", "FUND A", "SELL", 700, 220.0, 0.0154, ""],
         [20240103, "bulk", "OTHERCO", "OTHERCO", 0, "Other", "FUND Z", "BUY", 10, 5.0, 0.0, ""]])
    hdr = ["ex_date", "symbol", "symbol_then", "lab_symbol", "in_lab", "series", "isin", "face_value", "kind", "ratio_a",
           "ratio_b", "fv_from", "fv_to", "dividend", "rights_premium", "factor", "record_date", "first_listed",
           "known_date", "subject"]
    _gz(os.path.join(td, "corpactions", "corpactions.csv.gz"), hdr,
        [[20240315, "AAA", "AAA", "AAA", 1, "EQ", "INE1", 10, "split", "", "", 10, 2, "", "", 0.2, 20240315, 20240305, 20240305, "Split 10 to 2"],
         [20240620, "AAA", "AAA", "AAA", 1, "EQ", "INE1", 2, "bonus", 1, 1, "", "", "", "", 0.5, 20240620, 20240610, 20240610, "Bonus 1:1"],
         [20240620, "AAA", "AAA", "AAA", 1, "EQ", "INE1", 2, "dividend", "", "", "", "", 3.5, "", "", 20240620, 20240610, 20240610, "Dividend Rs 3.50"],
         [20240902, "AAA", "AAA", "AAA", 1, "EQ", "INE1", 2, "demerger", "", "", "", "", "", "", "", "", "", 20240830, "Demerger"],
         [20240401, "ETERNAL", "ZOMATO", "ETERNAL", 1, "EQ", "INE2", 1, "dividend", "", "", "", "", 1.0, "", "", "", 20240320, 20240320, "Dividend Re 1"]])
    with open(os.path.join(td, "corpactions", "bar_adjustment_issues.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "symbol", "date", "issue"])
        w.writerow(["min_2016_2023", "AAA", 20240902, "unadjusted_demerger"])
    os.makedirs(os.path.join(td, "index_membership"), exist_ok=True)
    with open(os.path.join(td, "index_membership", "index_changes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "announce_date", "effective_date", "action", "symbol", "lab_symbol", "in_lab", "company", "source"])
        w.writerow(["NIFTY50", 20240828, 20240930, "OUT", "AAA", "AAA", 1, "Aaa Ltd", "ind_prs28082024.pdf"])
        w.writerow(["NIFTY50", 20240828, 20240930, "IN", "BBB", "BBB", 1, "Bbb Ltd", "ind_prs28082024.pdf"])
        w.writerow(["NIFTY100", 20240828, 20240930, "IN", "CCC", "CCC", 0, "Ccc Ltd", "ind_prs28082024.pdf"])
    _gz(os.path.join(td, "index_membership", "index_members.csv.gz"),
        ["index", "symbol", "lab_symbol", "in_lab", "from_date", "to_date"],
        [["NIFTY50", "AAA", "AAA", 1, 20120102, 20240929], ["NIFTY50", "BBB", "BBB", 1, 20240930, ""],
         ["NIFTY50", "KEEP", "KEEP", 1, 20120102, ""], ["NIFTY100", "CCC", "CCC", 0, 20240930, ""],
         ["NIFTYNEXT50", "DDD", "DDD", 0, 20120102, ""]])
    json.dump({"built_through": 20261113, "NIFTYNEXT50": {"reliable_from": 20210401}},
              open(os.path.join(td, "index_membership", "data_check.json"), "w"))
    _gz(os.path.join(td, "index_membership", "index_weights.csv.gz"),
        ["month", "snapshot_date", "index", "symbol", "lab_symbol", "in_lab", "weight_pct", "ff_mcap_cr"],
        [[202408, 20240830, "NIFTY50", "AAA", "AAA", 1, 60.0, 600.0], [202408, 20240830, "NIFTY50", "KEEP", "KEEP", 1, 40.0, 400.0],
         [202409, 20240930, "NIFTY50", "BBB", "BBB", 1, 55.0, 550.0], [202409, 20240930, "NIFTY50", "KEEP", "KEEP", 1, 45.0, 450.0]])
    _gz(os.path.join(td, "fii_stats", "fii_stats.csv.gz"),
        ["date", "instrument", "buy_contracts", "buy_amt_cr", "sell_contracts", "sell_amt_cr", "oi_contracts", "oi_amt_cr"],
        [[f"2024-01-0{d}", ins, 100 * d, 1.0 * d, 50 * d, 0.5 * d, 1000 + d, 10.0]
         for d in (2, 3, 4, 5) for ins in ("INDEX FUTURES", "STOCK FUTURES")])


class _flow:
    def __enter__(self):
        self.td = tempfile.TemporaryDirectory()
        self.old = F.FLOW_DIR
        _flow_dir(self.td.name)
        F.use(self.td.name)
        return self.td.name

    def __exit__(self, *a):
        F.use(self.old)
        self.td.cleanup()


def t_deals_before_rule():
    with _flow():
        assert len(F.deals("ETERNAL", 20240102)) == 0                      # the 01-02 deals are not public on 01-02
        r = F.deals("ETERNAL", 20240103)
        assert list(r.date) == [20240102, 20240102] and sorted(r.side) == [-1, 1] and set(r.client) == {"FUND A", "FUND B"}
        r = F.deals("ETERNAL", 20240104)
        assert list(r.date) == [20240102, 20240102, 20240103] and r.kind[-1] == "block" and r.qty[-1] == 5000
        assert list(F.deals("ETERNAL", 20240104, kind="block").date) == [20240103]
        # the renamed company keeps ONE history under the lab name; the traded symbol stays in `symbol`
        r = F.deals("ETERNAL", 20991231)
        assert len(r) == 4 and list(r.symbol) == ["ZOMATO", "ZOMATO", "ZOMATO", "ETERNAL"]
        assert list(F.deals("ETERNAL", 20991231, since=20240103).date) == [20240103, 20250410]
        assert len(F.deals("ZOMATO", 20991231)) == 0 and len(F.deals("NOPE", 20991231)) == 0
        allr = F.deals(None, 20240104)
        assert len(allr) == 4 and "OTHERCO" in set(allr.lab) and list(allr.date) == sorted(allr.date)
        with N.cutoff(20240103):                                           # the global guard clips `before`
            assert list(F.deals("ETERNAL", 20991231).date) == [20240102, 20240102]
        # returned arrays are copies: a strategy cannot corrupt the cache
        r = F.deals("ETERNAL", 20991231)
        r.qty[:] = -1
        assert F.deals("ETERNAL", 20991231).qty[0] == 1000


def t_corp_actions_known_date_rule():
    with _flow():
        assert len(F.corp_actions("AAA", 20240305)) == 0                   # listed on 03-05 (evening): not yet known
        r = F.corp_actions("AAA", 20240306)
        assert list(r.ex_date) == [20240315] and r.kind[0] == "split" and abs(r.factor[0] - 0.2) < 1e-12
        assert len(F.corp_actions("AAA", 20240306, upcoming=False)) == 0   # known but not yet ex
        assert list(F.corp_actions("AAA", 20240316, upcoming=False).ex_date) == [20240315]
        # an action without a Bc listing is known from the session before its ex-date -> visible ON the ex-date
        assert "demerger" not in set(F.corp_actions("AAA", 20240830).kind)
        assert "demerger" in set(F.corp_actions("AAA", 20240902).kind)
        r = F.corp_actions("AAA", 20991231)
        assert list(r.ex_date) == [20240315, 20240620, 20240620, 20240902]
        d = r.dividend[r.kind == "dividend"]
        assert len(d) == 1 and d[0] == 3.5 and np.isnan(r.factor[r.kind == "demerger"][0])
        assert list(F.corp_actions("AAA", 20991231, kinds=("bonus", "split")).kind) == ["split", "bonus"]
        assert list(F.corp_actions("AAA", 20991231, since=20240601, kinds=("bonus",)).ex_date) == [20240620]
        # adjustment factor between two dates uses only actions already ex and known
        assert abs(F.adj_factor("AAA", 20240101, 20240701) - 0.1) < 1e-12
        assert abs(F.adj_factor("AAA", 20240101, 20240620) - 0.2) < 1e-12        # the bonus goes ex ON 06-20
        assert abs(F.adj_factor("AAA", 20240401, 20240701) - 0.5) < 1e-12
        assert F.adj_factor("AAA", 20240701, 20991231) == 1.0
        # renamed company: feed symbol is the current one, lookup is by lab symbol
        assert list(F.corp_actions("ETERNAL", 20240321).symbol) == ["ETERNAL"]
        with N.cutoff(20240306):
            assert list(F.corp_actions("AAA", 20991231).ex_date) == [20240315]
        assert [x["issue"] for x in F.bar_adjustment_issues("AAA")] == ["unadjusted_demerger"]
        assert F.bar_adjustment_issues("ZZZ") == []


def t_index_membership_point_in_time():
    with _flow():
        assert len(F.index_changes("NIFTY50", 20240828)) == 0              # announced that evening
        ch = F.index_changes("NIFTY50", 20240829)
        assert len(ch) == 2 and set(ch.action) == {"IN", "OUT"} and set(ch.effective_date) == {20240930}
        assert dict(zip(ch.action, ch.lab)) == {"IN": "BBB", "OUT": "AAA"}
        assert len(F.index_changes("NIFTY50", 20991231, since=20240901)) == 0
        assert len(F.index_changes("SENSEX", 20991231)) == 0
        # membership switches on the effective date: the day before 09-30 AAA is in, from 09-30 BBB is
        assert F.index_members("NIFTY50", 20240930) == {"AAA", "KEEP"}
        assert F.index_members("NIFTY50", 20241001) == {"BBB", "KEEP"}
        assert F.index_members("NIFTY50", 20120103) == {"AAA", "KEEP"} and F.index_members("NIFTY50", 20120102) == set()
        assert F.was_member("NIFTY50", "AAA", 20240101) and not F.was_member("NIFTY50", "BBB", 20240101)
        assert F.index_members("NIFTY100", 20241001) == {"CCC"} and F.index_members("NIFTY100", 20240930) == set()
        assert F.index_coverage("NIFTY50") == (20120102, 20261113) and F.index_coverage("NOPE") == (None, None)
        # an index whose chain of releases is broken is served only from its verified date on
        assert F.index_coverage("NIFTYNEXT50") == (20210401, 20261113)
        assert F.index_members("NIFTYNEXT50", 20210401) == set() and F.index_members("NIFTYNEXT50", 20150101) == set()
        assert F.index_members("NIFTYNEXT50", 20210402) == {"DDD"}
        with N.cutoff(20240829):                                           # the future composition stays hidden
            assert F.index_members("NIFTY50", 20991231) == {"AAA", "KEEP"}
            assert len(F.index_changes("NIFTY50", 20991231)) == 2
        with N.cutoff(20240828):
            assert len(F.index_changes("NIFTY50", 20991231)) == 0
        # month-end weights become visible WEIGHT_LAG_DAYS calendar days after the snapshot date
        assert len(F.index_weights("NIFTY50", 20240909)) == 0                         # 08-30 + 10 days = 09-09
        w = F.index_weights("NIFTY50", 20240910)
        assert set(w.lab) == {"AAA", "KEEP"} and abs(w.weight.sum() - 100.0) < 1e-9 and set(w.date) == {20240830}
        assert set(F.index_weights("NIFTY50", 20241010).lab) == {"AAA", "KEEP"}     # 09-30 file not yet posted
        assert set(F.index_weights("NIFTY50", 20241011).lab) == {"BBB", "KEEP"}
        with N.cutoff(20240915):
            assert set(F.index_weights("NIFTY50", 20991231).date) == {20240830}
        assert len(F.index_weights("NIFTY100", 20991231)) == 0


def t_fii_stats_and_asof():
    with _flow():
        r = F.fii_stats(20240104, n=5)
        assert list(r.date) == [20240102, 20240103] and list(r.buy_contracts) == [200.0, 300.0]
        assert list(F.fii_stats(20240104, n=1).date) == [20240103]
        assert len(F.fii_stats(20240102)) == 0 and len(F.fii_stats(20240104, instrument="NOPE")) == 0
        assert F.fii_instruments() == ["INDEX FUTURES", "STOCK FUTURES"]
        with N.cutoff(20240103):
            assert list(F.fii_stats(20991231, n=9).date) == [20240102]
        # Asof picks `before` like nsedata's helpers
        s = SimpleNamespace(symbol="ETERNAL", date=20240103)
        assert list(F.asof_session(s).deals().date) == [20240102, 20240102]
        d = SimpleNamespace(symbol="ETERNAL", date=np.array([20240102, 20240103]))
        assert list(F.asof_swing(d).deals().date) == [20240102, 20240102, 20240103]
        assert list(F.asof_swing_intraday(d).deals().date) == [20240102, 20240102]
        assert list(F.asof_swing(d).fii_stats(n=9).date) == [20240102, 20240103]
        assert F.asof_session(SimpleNamespace(symbol="AAA", date=20241001)).index_members() == {"BBB", "KEEP"}
        assert "deals" in F.coverage() and "NIFTY50" in F.coverage()


def t_missing_files_are_empty_not_errors():
    with tempfile.TemporaryDirectory() as td:
        old = F.FLOW_DIR
        try:
            F.use(td)
            assert len(F.deals("X", 20991231)) == 0 and len(F.corp_actions("X", 20991231)) == 0
            assert len(F.index_changes("NIFTY50", 20991231)) == 0 and F.index_members("NIFTY50", 20991231) == set()
            assert len(F.fii_stats(20991231)) == 0 and F.bar_adjustment_issues() == []
            assert "missing" in F.coverage()
        finally:
            F.use(old)


# ── optmin ──────────────────────────────────────────────────────────────────

def _optmin_dir(td):
    rows = []
    for sym, kind, strike, typ in (("NIFTY26N1725000CE", "OPT", 25000, "CE"), ("NIFTY26N1725000PE", "OPT", 25000, "PE"),
                                   ("NIFTY26N1725100CE", "OPT", 25100, "CE"), ("NIFTY26NOVFUT", "FUT", 0, "FUT")):
        exp = 20261123 if kind == "FUT" else 20261117
        rows.append([sym, sym, kind, "NIFTY", exp, strike, typ, 65, 0.05, 20261110, 20261112, 20261113, 2, 0])
    with open(os.path.join(td, "contracts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tradingsymbol", "file", "kind", "underlying", "expiry", "strike", "type", "lot", "tick", "first_day",
                    "first_bar_day", "last_bar_day", "bar_days", "bars"])
        w.writerows(rows)
    hdr = ["date", "minute", "open", "high", "low", "close", "volume"]

    def bars(base, skip=()):
        out = []
        for day in (20261112, 20261113):
            for m in range(555, 565):
                if (day, m) in skip:
                    continue
                p = base + (day - 20261112) * 10 + (m - 555)
                out.append([day, m, p, p + 0.5, p - 0.5, p + 0.25, 65 * (m - 554)])
        return out
    _gz(os.path.join(td, "bars", "NIFTY26N1725000CE.csv.gz"), hdr, bars(100.0))
    _gz(os.path.join(td, "bars", "NIFTY26N1725000PE.csv.gz"), hdr, bars(80.0, skip={(20261113, 558), (20261113, 559), (20261113, 560)}))
    _gz(os.path.join(td, "bars", "NIFTY26N1725100CE.csv.gz"), hdr, [[20261112, 555, 60, 60, 60, 60, 65]])
    _gz(os.path.join(td, "bars", "NIFTY26NOVFUT.csv.gz"), hdr, bars(25000.0))


class _om:
    def __enter__(self):
        self.td = tempfile.TemporaryDirectory()
        self.old = O.OPTMIN_DIR
        _optmin_dir(self.td.name)
        O.use(self.td.name)
        return self.td.name

    def __exit__(self, *a):
        O.use(self.old)
        self.td.cleanup()


def t_optmin_minute_before_rule():
    with _om():
        ce = O.symbol("NIFTY", 20261117, 25000, "CE")
        assert ce == "NIFTY26N1725000CE" and O.symbol("NIFTY", 20261117, 99999, "CE") is None
        assert O.symbol("NIFTY", 20261123) == "NIFTY26NOVFUT" and O.expiries("NIFTY") == [20261117]
        assert len(O.contracts("NIFTY", kind="OPT")) == 3 and O.info(ce)["lot"] == 65
        # a decision at 09:20 (minute 560) on 11-13 sees the bars that START before 09:20: 09:15..09:19
        b = O.bars(ce, 20261113, 560)
        assert len(b) == 15 and (b.date[-1], b.t[-1]) == (20261113, 559) and b.c[-1] == 100 + 10 + 4 + 0.25
        assert len(O.bars(ce, 20261113)) == 10 and O.bars(ce, 20261113).date[-1] == 20261112    # minute=None: earlier sessions
        assert len(O.bars(ce, 20261113, 555)) == 10                                              # nothing of today at the open
        assert list(O.bars(ce, 20261113, 560, n=2).t) == [558, 559]
        s = O.session(ce, 20261113, 558)
        assert list(s.t) == [555, 556, 557] and len(O.session(ce, 20261113)) == 10
        la = O.last(ce, 20261113, 560)
        assert la["t"] == 559 and la["age"] == 0
        # illiquid strike: last trade 09:17 -> stale by 2 minutes at 09:20; nothing today -> None
        pe = "NIFTY26N1725000PE"
        la = O.last(pe, 20261113, 560)
        assert la["t"] == 557 and la["age"] == 2
        assert O.last("NIFTY26N1725100CE", 20261113, 560) is None
        assert O.last("NIFTY26N1725100CE", 20261113, 560, same_day=False)["date"] == 20261112
        ch = O.chain("NIFTY", 20261117, 20261113, 560)
        assert set(ch) == {(25000.0, "CE"), (25000.0, "PE")} and ch[(25000.0, "PE")]["age"] == 2
        assert set(O.chain("NIFTY", 20261117, 20261113, 560, max_age=1)) == {(25000.0, "CE")}
        # fill = the bar that starts AT the minute (next-open fill after a decision at the previous close)
        f = O.fill(ce, 20261113, 560)
        assert f["o"] == 100 + 10 + 5 and O.fill(pe, 20261113, 560) is None                    # no trade that minute
        assert O.fill("NOPE", 20261113, 560) is None and len(O.bars("NOPE", 20261113, 560)) == 0
        assert O.sessions(ce) == [20261112, 20261113] and O.sessions(ce, 20261113) == [20261112]
        # copies
        b = O.bars(ce, 20261113, 560)
        b.c[:] = -1
        assert O.bars(ce, 20261113, 560).c[-1] > 0
        assert "contracts: 4" in O.coverage()


def t_optmin_cutoff_guards():
    with _om():
        ce = "NIFTY26N1725000CE"
        with O.cutoff(20261113, 558):
            assert list(O.session(ce, 20261113).t) == [555, 556, 557]              # readers are clipped
            assert O.bars(ce, 20991231).t[-1] == 557 and O.last(ce, 20261113, 600)["t"] == 557
            assert O.fill(ce, 20261113, 557)["t"] == 557
            for m in (558, 559):
                try:
                    O.fill(ce, 20261113, m)
                    raise AssertionError("fill at/after the cutoff must raise")
                except N.LookAheadError:
                    pass
            assert set(O.chain("NIFTY", 20261117, 20261113, 600)) == {(25000.0, "CE"), (25000.0, "PE")}
            assert O.chain("NIFTY", 20261117, 20261113, 600)[(25000.0, "CE")]["t"] == 557
        with N.cutoff(20261113):                                                   # nsedata's day cutoff: nothing of 11-13
            assert len(O.session(ce, 20261113)) == 0 and O.bars(ce, 20991231).date[-1] == 20261112
            assert O.sessions(ce) == [20261112]
            try:
                O.fill(ce, 20261113, 555)
                raise AssertionError("fill on the cutoff day must raise")
            except N.LookAheadError:
                pass
            assert O.fill(ce, 20261112, 564)["t"] == 564
        assert len(O.session(ce, 20261113)) == 10                                  # guards are released
        # changing FUTURE bars never changes what an earlier instant sees
        before = O.bars(ce, 20261113, 558)
        t = O._table(ce)
        t["c"][t["key"] >= 20261113 * 10000 + 558] += 1000.0
        after = O.bars(ce, 20261113, 558)
        assert np.array_equal(before.c, after.c) and np.array_equal(before.t, after.t)


TESTS = [
    ("flowdata: deals dated D are visible only from D+1 (before rule, cutoff, rename under one lab symbol)",
     t_deals_before_rule),
    ("flowdata: corporate actions visible from their known date (Bc listing, else ex-date); adj_factor uses past ex-dates",
     t_corp_actions_known_date_rule),
    ("flowdata: index changes visible after the announcement; membership switches on the effective date",
     t_index_membership_point_in_time),
    ("flowdata: FII statistics strictly before; asof_session / asof_swing / asof_swing_intraday pick `before`",
     t_fii_stats_and_asof),
    ("flowdata: missing files give empty results, not errors", t_missing_files_are_empty_not_errors),
    ("optmin: a decision at minute m sees only bars that start before m; last / chain / fill semantics",
     t_optmin_minute_before_rule),
    ("optmin: cutoff(date, minute) and nsedata.cutoff clip readers and make fill raise; future bars cannot leak",
     t_optmin_cutoff_guards),
]
