"""
test_validated_pead.py — tests of the EV-PEAD-LONG-H60 PAPER candidate (agents/validated_pead.py, pead_data.py).

Run: cd algotrader_v4 && python test_validated_pead.py        (test_pipeline.py runs the same functions)
     cd algotrader_v4 && python test_validated_pead.py --build-fixture   (rebuild fixtures/ from the lab scratch)
Never touches the network (every download is patched) and never writes into the repo.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import gzip
import json
import os
import sys
import tempfile
from datetime import date, datetime, time as dtime, timedelta
from unittest import mock

os.environ.setdefault("PERFORMANCE_FILE", os.path.join(tempfile.mkdtemp(), "performance.json"))
os.environ["SHADOW_TRIALS_CSV"] = os.path.join(tempfile.mkdtemp(), "absent_trials.csv")   # never the lab's file

import numpy as np

import pead_data as PD
import pead_parity as PP
import agents.validated_pead as V
from config import settings
from stratlab import events_strategies as ES
from stratlab import sim

D = 20261111                 # Wednesday; previous session Mon 11-09 (11-10 holiday), next Thu 11-12
D_PREV, D_NEXT = 20261109, 20261112


class Skip(Exception):
    """Lab data absent (scratch only)."""


def _ist(d: int, hh: int, mm: int, ss: int = 0) -> datetime:
    from ist_clock import now_ist
    return datetime.combine(PD.to_date(d), dtime(hh, mm, ss)).replace(tzinfo=now_ist().tzinfo)


def _calendar(day: int = D, n: int = 30) -> list[int]:
    d = PD.to_date(day)
    cal = PD.sessions_between(d - timedelta(days=70), d)[-n:]
    return cal + PD.sessions_between(d + timedelta(days=1), d + timedelta(days=10))[:1]


def _bars(day: int = D, last_close: float = 106.0, last_vol: float = 1500.0, n: int = 25, close: float = 100.0,
          vol: float = 1000.0, skip_prev: bool = False) -> PD.Bars:
    cal = [s for s in _calendar(day, n + 1) if s <= day][-n:]
    if skip_prev:
        cal = cal[:-2] + cal[-1:]
    c = [close] * (len(cal) - 1) + [last_close]
    v = [vol] * (len(cal) - 1) + [last_vol]
    return PD.Bars(tuple(cal), tuple(c), tuple(c), tuple(c), tuple(c), tuple(v))


def _nifty(day: int = D, last: float = 20000.0) -> dict:
    cal = [s for s in _calendar(day) if s <= day]
    return {s: (last if s == day else 20000.0) for s in cal}


def _plan(events, bars, members=None, holding=(), exclusions=(), nifty=None, day=D):
    members = set(bars) if members is None else members
    return V.plan_pead(day, _calendar(day), events, bars, nifty or _nifty(day), members, holding, exclusions)


# ══════════════════════════════════════════════════════════════════════════
# lab constants, fingerprint, parity
# ══════════════════════════════════════════════════════════════════════════
def test_lab_constants_and_fingerprint():
    sp = ES.BY_ID["ev_pead_long_h60"]
    assert sp.params == {"min_ar_pct": 4.0, "vol_n": 20, "stop_pct": 20.0, "hold": 60}, sp.params
    assert sp.kind == "swing" and sp.cost_kind == "equity_cnc"
    assert (V.NOTIONAL, sim.DEFAULT_SLIPPAGE_BPS) == (100_000.0, 2.0)
    # pinned: a change to any lab function / constant the agent imports must come with a VERSION bump
    assert V.LAB_FINGERPRINT == "5df1167e03965713", V.LAB_FINGERPRINT
    a = V.PeadCandidateAgent()
    assert (a.name, a.LAB_ID, a.VERSION) == ("pead_h60", "EV-PEAD-LONG-H60", "EV-PEAD-LONG-H60/ev1/v1")
    assert a.product == "CNC" and a.COST_KIND == "equity_cnc" and a.scheduled and a.CANDIDATE and not a.PROVEN
    assert V.ES.pead_long is ES.pead_long and V.ES._reaction is ES._reaction, "imported from the lab, not copied"
    exp = a.LAB_EXPECTATION
    assert exp["DISC_2023_26"]["trades"] == 529 and exp["FINAL"]["trades"] == 296 and exp["proven"] is False
    assert exp["OLD_ERA_2012_23"]["trades"] == 2736 and "live_only_guards" in exp


def test_fixture_parity():
    """~30 sampled lab days (both eras) from the committed fixture: the live decision equals the lab's entries."""
    days = PP.load_fixture()
    assert len(days) >= 25
    n_lab = 0
    for x in days:
        got = {e.symbol for e in PP.run_day(x)[0]}
        assert got == x["expected"], (x["day"], sorted(got), sorted(x["expected"]))
        n_lab += len(x["expected"])
    assert n_lab >= 40, n_lab
    eras = {x["day"] >= 20231110 for x in days}
    assert eras == {True, False}, "both eras sampled"


def test_lab_full_parity():
    """Scratch only: every session of both eras (2012-2026) through plan_pead() with the lab cache's inputs. Every
    lab entry is reproduced; the only extras are the trades the lab dropped afterwards for a corporate-action /
    gap day inside their holding window (events_run.drop_excluded: 3 in 2023-26, 8 in 2012-23)."""
    if not PP.lab_available():
        raise Skip("lab data set absent")
    lab = PP.Lab()
    r = PP.replay(lab)
    assert r["missing"] == [] and r["matched"] == r["lab_entries"] >= 3500, \
        {k: (v if not isinstance(v, list) else v[:10]) for k, v in r.items()}
    c = PP.classify_extra(lab, r["extra"])
    assert c["unexplained"] == [] and len(c["lab_dropped_ca"]) == 11, c
    print(f"      parity: {r['days']} sessions, lab entries {r['lab_entries']}, matched {r['matched']}, "
          f"extra {len(r['extra'])} (all lab drop_excluded)")


def test_live_data_path_parity():
    """Scratch only: the live data path (raw NSE bhavcopy files -> read_new -> new_factors -> monthly_universe)
    on 2024-02 .. 2026-11. Divergences are universe-boundary membership only: the replay ranks with TODAY's
    EQUITY_L (stocks delisted since are missing; live always uses the then-current list)."""
    if not PP.live_path_available():
        raise Skip("raw bhavcopy files absent")
    r = PP.live_path_replay()
    rate = r["matched"] / r["lab_entries"]
    assert rate >= 0.99 and len(r["extra"]) <= 15, (rate, r["missing"], r["extra"])
    worst = min(a / b for a, b in r["universe_match"].values())
    assert worst >= 0.98, r["universe_match"]
    print(f"      live path: {r['days']} sessions, matched {r['matched']}/{r['lab_entries']} ({rate:.1%}), "
          f"extra {len(r['extra'])}, universe overlap >= {worst:.1%}")


# ══════════════════════════════════════════════════════════════════════════
# the rule: reaction-day mapping and boundaries
# ══════════════════════════════════════════════════════════════════════════
def test_reaction_day_mapping():
    """Results in hours react the same session; after the close or before the open, the next session."""
    bars = {"AAA": _bars()}
    prev_d = PD.to_date(D_PREV).isoformat()
    day_d = PD.to_date(D).isoformat()
    cases = [
        (f"{day_d} 14:00:00", True, "in hours on D"),
        (f"{day_d} 15:29:59", True, "15:29 on D (strictly before the close)"),
        (f"{day_d} 15:30:00", False, "15:30 on D: reacts on the next session"),
        (f"{day_d} 16:10:00", False, "after the close on D"),
        (f"{day_d} 08:30:00", True, "pre-open on D"),
        (f"{prev_d} 16:10:00", True, "after the close of the previous session"),
        (f"{prev_d} 15:30:00", True, "exactly the previous close"),
        (f"{prev_d} 15:29:00", False, "in hours on the previous session (reacted then)"),
        ("2026-11-10 11:00:00", True, "holiday between the sessions"),
    ]
    for ts, want, why in cases:
        got, _ = _plan([("AAA", ts, "2026-09-30")], bars)
        assert bool(got) is want, (why, ts, got)
        if got:
            e = got[0]
            assert (e.symbol, e.decision_day, e.entry_day) == ("AAA", D, D_NEXT) and abs(e.ar - 0.06) < 1e-9, e


def test_rule_boundaries():
    ev = lambda s: (s, f"{PD.to_date(D).isoformat()} 12:00:00", "2026-09-30")
    # +4.01 % abnormal: traded; +3.99 %: not (NIFTY flat)
    got, sk = _plan([ev("UP"), ev("LOW")], {"UP": _bars(last_close=104.01), "LOW": _bars(last_close=103.99)})
    assert [e.symbol for e in got] == ["UP"] and sk == {"LOW": "below_threshold"}, (got, sk)
    # abnormal vs NIFTY: stock +6 %, NIFTY +3 % -> 3 %: not traded
    got, _ = _plan([ev("A")], {"A": _bars()}, nifty=_nifty(last=20600.0))
    assert got == []
    # volume equal to the 20-bar mean: not traded (strictly above)
    got, sk = _plan([ev("V")], {"V": _bars(last_vol=1000.0)})
    assert got == [] and sk["V"] == "volume_not_above_average", sk
    # universe gate, holding, no bar today, missing previous session, exclusion on D, short history
    got, sk = _plan([ev("N")], {"N": _bars()}, members=set())
    assert got == [] and sk["N"] == "not_member"
    got, sk = _plan([ev("H")], {"H": _bars()}, holding={"H"})
    assert got == [] and sk["H"] == "holding"
    got, sk = _plan([ev("X")], {}, members={"X"})
    assert got == [] and sk["X"] == "no_bar_today"
    got, sk = _plan([ev("P")], {"P": _bars(skip_prev=True)})
    assert got == [] and sk["P"] == "no_bar_prev_session", sk
    got, sk = _plan([ev("C")], {"C": _bars()}, exclusions=[("C", D)])
    assert got == [] and sk["C"] == "corporate_action", sk
    got, sk = _plan([ev("S")], {"S": _bars(n=21)})
    assert got == [] and sk["S"] == "short_history", sk
    # the lab module's state is restored after a decision
    from stratlab import events_data as E
    assert "results" not in E._T or E._T.get("results") is not None


def test_sizing_stop_and_clock():
    assert V.entry_qty(500.0) == int(100_000 // (500.0 * 1.0002)) == 199
    assert V.entry_qty(250_000.0) == 1
    assert abs(V.stop_price(500.0) - 500.0 * 1.0002 * 0.8) < 1e-9
    stop = V.stop_price(100.0)
    assert V.hold_step(stop, 1, (100, 101, 90, 95)) is None
    assert V.hold_step(stop, 5, (90, 92, 79.0, 85)) == ("stop", stop)                # touched intraday: at the stop
    assert V.hold_step(stop, 5, (70, 75, 65, 72)) == ("stop", 70.0)                  # gapped through: at the open
    assert V.hold_step(stop, 59, (100, 101, 99, 100.5)) is None
    assert V.hold_step(stop, 60, (100, 101, 99, 100.5)) == ("max_hold", 100.5)       # close of the 60th session
    assert V.hold_step(stop, 60, (100, 101, 79, 100.5))[0] == "stop", "the stop is checked before the clock"


def test_hold_rule_equals_lab_simulator():
    """hold_step walked bar by bar == sim.sim_positional (the lab's own simulator) on synthetic paths."""
    from stratlab import data as SD
    from stratlab.api import Spec, Swing, LONG

    def lab_trade(o, h, l, c, sig_day):
        n = len(c)
        dates = np.arange(20200101, 20200101 + n)          # plain increasing ints: the simulator only orders them
        sd = SD.from_arrays("SYN", dates, np.full(n, 555, np.int16), np.array(o, float), np.array(h, float),
                            np.array(l, float), np.array(c, float), np.full(n, 1e5), min_bars=1)
        fn = lambda d, p: Swing(LONG, stop_pct=20.0, max_hold=60) if int(d.date[-1]) == int(dates[sig_day]) else None
        sp = Spec(id="syn", kind="swing", family="t", desc="t", fn=fn, params={}, horizon="positional")
        tr, _ = sim.sim_positional(sp, sd, int(dates[0]), int(dates[-1]), None, None, V.SLIP)
        return tr[0]

    def live_trade(o, h, l, c, sig_day):
        e = sig_day + 1
        stop = V.stop_price(o[e])
        for k in range(e, len(c)):
            r = V.hold_step(stop, k - e + 1, (o[k], h[k], l[k], c[k]))
            if r:
                return k, r[0], r[1] * (1 - V.SLIP)
        return None

    rng = np.random.default_rng(7)
    for trial in range(12):
        n = 90
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02 if trial % 3 else 0.06, n))
        o = c * (1 + rng.normal(0, 0.005, n))
        h = np.maximum(o, c) * 1.01
        l = np.minimum(o, c) * 0.99
        if trial == 5:
            o[20], l[20] = c[19] * 0.7, c[19] * 0.65                     # a gap through the stop
        lt = lab_trade(o, h, l, c, 3)
        k, reason, px = live_trade(list(o), list(h), list(l), list(c), 3)
        assert int(lt[6]) == 20200101 + k and lt[13] == reason and abs(lt[8] - round(px, 4)) < 1e-3, (trial, lt, k, reason, px)
        assert lt[9] == V.entry_qty(o[4]), (lt[9], V.entry_qty(o[4]))


def test_caps():
    es = [V.PeadEntry(s, D, D_NEXT, ar) for s, ar in (("A", 0.05), ("B", 0.09), ("C", 0.09), ("D", 0.04))]
    px = {"A": 100.0, "B": 100.0, "C": 100.0, "D": 0.0}
    take, sk = V.apply_caps(es, px, n_open=8, gross_open=0.0, max_positions=10, max_gross=0.0, notional=1e5)
    assert [t[0].symbol for t in take] == ["B", "C"] and sk == {"A": "cap_positions", "D": "no_data:open"}, (take, sk)
    take, sk = V.apply_caps(es, px, n_open=0, gross_open=850_000.0, max_positions=100, max_gross=1_000_000.0,
                            notional=1e5)
    assert [t[0].symbol for t in take] == ["B"] and sk["C"] == "cap_gross" and sk["A"] == "cap_gross", sk
    assert take[0][1] == V.entry_qty(100.0) == 999


# ══════════════════════════════════════════════════════════════════════════
# pead_data (pure parts, no network)
# ══════════════════════════════════════════════════════════════════════════
def test_pead_data_parsers():
    idx = (b"Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value\n"
           b"Nifty 50,13-11-2026,25209.85,25246.35,25054.4,25144.3\nNifty Next 50,13-11-2026,1,2,3,4\n")
    assert PD.parse_index_file(idx) == (25209.85, 25144.3)
    eq = PD.parse_equity_list(b"SYMBOL,NAME OF COMPANY\nAAA,a\nNIFTYBEES,etf\nBBB,b\n",
                              b"Symbol,Underlying\nNIFTYBEES,Nifty\n")
    assert eq == {"AAA", "BBB"}
    bh = (b"SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE\nAAA, EQ, 12-Nov-2026, 1, 2\n")
    assert PD.bhav_date_inside(bh) == 20261112
    assert PD.quarter_end_before(date(2026, 11, 5)) == date(2026, 9, 30)
    assert PD.quarter_end_before(date(2026, 1, 20)) == date(2025, 12, 31)
    assert PD.quarter_end_before(date(2026, 4, 1)) == date(2026, 3, 31)


def _write_feed(raw: str, feed: str, rows: list, name: str = "20261101_20261111_p01.json.gz") -> None:
    os.makedirs(os.path.join(raw, feed), exist_ok=True)
    with gzip.open(os.path.join(raw, feed, name), "wt") as f:
        json.dump({"data": rows} if feed != "bm" else rows, f)


def test_results_events_lab_rule_and_provisional():
    """events_parse.build_results on stored feeds (first_ts = the results-day board-meeting outcome before the
    structured filing), plus a provisional row for an announcement whose filing is not out yet."""
    raw = tempfile.mkdtemp()
    _write_feed(raw, "bm", [
        {"bm_symbol": "AAA", "bm_date": "10-Nov-2026", "bm_purpose": "Financial Results", "bm_timestamp": "01-Nov-2026 10:00:00"},
        {"bm_symbol": "BBB", "bm_date": "11-Nov-2026", "bm_purpose": "Financial Results", "bm_timestamp": "02-Nov-2026 10:00:00"},
        {"bm_symbol": "CCC", "bm_date": "11-Nov-2026", "bm_purpose": "Board Meeting Intimation", "bm_timestamp": "02-Nov-2026 10:00:00"}])
    _write_feed(raw, "ann_obm", [
        {"symbol": "AAA", "an_dt": "10-Nov-2026 16:05:00"},          # outcome on the results day, filing at 18:00
        {"symbol": "BBB", "an_dt": "11-Nov-2026 13:10:00"},          # outcome, filing not out yet -> provisional
        {"symbol": "CCC", "an_dt": "11-Nov-2026 13:10:00"}])         # meeting not notified for results -> nothing
    _write_feed(raw, "ifr", [{"symbol": "AAA", "broadcast_Date": "10-Nov-2026 18:00:00", "qe_Date": "30-SEP-2026"}])
    for f in ("fr", "ann_fr"):
        _write_feed(raw, f, [])
    ev = {e["symbol"]: e for e in PD.results_events(raw)}
    assert set(ev) == {"AAA", "BBB"}, ev
    assert ev["AAA"]["first_ts"] == "2026-11-10 16:05:00" and ev["AAA"]["src"] == "ann", ev["AAA"]
    assert ev["BBB"]["first_ts"] == "2026-11-11 13:10:00" and ev["BBB"]["src"] == "provisional"
    assert ev["BBB"]["period_end"] == "2026-09-30"
    lo, hi = datetime(2026, 11, 9, 15, 30), datetime(2026, 11, 11, 15, 30)
    assert PD.window_has_announcements(raw, lo, hi) == 4
    assert PD.window_has_announcements(raw, datetime(2026, 11, 12, 15, 30), datetime(2026, 11, 13, 15, 30)) == 0


def test_feed_failure_stops_and_reports():
    calls = []

    class FakeClient:
        def fetch(self, feed, d0, d1):
            calls.append(feed)
            return ("refused", []) if feed == "fr" else ("ok", [])

    with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"VALIDATED_PEAD_DIR": td}):
        res = PD.fetch_feeds(date(2026, 11, 11), client=FakeClient())
    assert res["ok"] is False and res["feeds"]["fr"] == "refused" and res["feeds"]["ifr"] == "ok", res
    assert "ann_obm" not in calls, "stops at the first failure"


def test_build_bars_adjusts_like_the_lab():
    """A 1:1 bonus inside the window: earlier prices x 0.5, volumes / 0.5 (events_daily.new_factors); a demerger
    with a large gap: the ex-day is an exclusion; close = LAST traded price."""
    import pandas as pd
    cal = [20261102, 20261103, 20261104, 20261105, 20261106]
    rows = []
    for i, d in enumerate(cal):
        px = 200.0 if i < 3 else 100.0                                # bonus ex-date 20261105
        rows.append({"date": d, "symbol": "BON", "series": "EQ", "open": px, "high": px, "low": px, "close": px,
                     "last": px + 0.5, "prev_close": px, "volume": 1000.0 if i < 3 else 2000.0, "turnover": 1e7})
        dm = 100.0 if i < 2 else 50.0                                 # demerger-like gap on 20261104
        rows.append({"date": d, "symbol": "DEM", "series": "EQ", "open": dm, "high": dm, "low": dm, "close": dm,
                     "last": dm, "prev_close": dm, "volume": 500.0, "turnover": 1e6})
    eq = pd.DataFrame(rows)
    actions = [("BON", 20261105, "Bonus 1:1"), ("DEM", 20261104, "Demerger")]
    bars, excl, applied = PD.build_bars(eq, actions, cal)
    b = bars["BON"]
    assert b.c[:3] == (100.25, 100.25, 100.25) and b.c[3] == 100.5, b.c        # (200 + 0.5) x 0.5, LAST
    assert b.v[:3] == (2000.0, 2000.0, 2000.0) and b.v[3] == 2000.0, b.v
    assert ("BON", 20261105, 0.5, "applied") in applied, applied
    assert ("DEM", 20261104) in excl, excl


# ══════════════════════════════════════════════════════════════════════════
# the agent (paper flow, fail-safe, PAPER-only, flag)
# ══════════════════════════════════════════════════════════════════════════
@contextlib.contextmanager
def pead_env(now: datetime, flag: bool = True, quotes: dict | None = None):
    """A fresh agent at IST `now`: PAPER + Kite data, VALIDATED_PEAD_ENABLED=flag, private ledger / paper book /
    store, fake quotes, SEBI check approved, asyncio.sleep advancing the clock."""
    import bot_state
    import performance as perf_mod
    from kite_client import kite_client
    from sebi_compliance import sebi_compliance
    clk = [now]
    keys = ("trading_mode", "paper_data_source", "validated_only", "validated_pead_enabled", "validated_notional_scale")
    saved = {k: getattr(settings, k) for k in keys}
    saved_k = (list(kite_client._paper_orders), copy.deepcopy(kite_client._paper_positions))
    saved_en = bot_state._agent_enabled.get("pead_h60", True)
    td = tempfile.mkdtemp()
    with mock.patch.object(perf_mod, "FILE", __import__("pathlib").Path(td) / "perf.json"):
        p = perf_mod.Performance()

    async def _sleep(s):
        clk[0] = clk[0] + timedelta(seconds=s)
    try:
        kite_client._paper_orders.clear()
        kite_client._paper_positions.clear()
        settings.trading_mode, settings.paper_data_source = "PAPER", "KITE"
        settings.validated_only, settings.validated_pead_enabled, settings.validated_notional_scale = True, flag, 1.0
        bot_state._agent_enabled["pead_h60"] = True
        a = V.PeadCandidateAgent()
        a.state.running = True
        with mock.patch.dict(os.environ, {"PAPER_BOOK_FILE": os.path.join(td, "pb.json"), "VALIDATED_PEAD_DIR": td}), \
             mock.patch.object(V, "now_ist", side_effect=lambda: clk[0]), \
             mock.patch.object(perf_mod, "now_ist", side_effect=lambda: clk[0]), \
             mock.patch.object(perf_mod, "performance", p), \
             mock.patch.object(V.VD, "quotes", side_effect=lambda ins: dict(quotes or {})), \
             mock.patch.object(V.asyncio, "sleep", _sleep), \
             mock.patch.object(sebi_compliance, "pre_order_check", return_value=(True, "ALGO-PEAD-010", "ok")), \
             mock.patch.object(sebi_compliance, "record_order_id", return_value=None), \
             mock.patch.object(PD, "_http_get", side_effect=AssertionError("no network in tests")), \
             mock.patch.object(PD, "_Client", side_effect=AssertionError("no network in tests")):
            yield a, clk, p
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)
        kite_client._paper_orders[:] = saved_k[0]
        kite_client._paper_positions[:] = saved_k[1]
        bot_state._agent_enabled["pead_h60"] = saved_en


def _decision_g(day: int = D, events=None, bars=None, skip=None) -> dict:
    cal = _calendar(day)
    dec = {"skip": skip} if skip else {
        "members": {"AAA", "BBB"}, "calendar": cal, "exclusions": [],
        "events": events if events is not None else [
            {"symbol": "AAA", "first_ts": f"{PD.to_date(day).isoformat()} 14:00:00", "period_end": "2026-09-30"},
            {"symbol": "BBB", "first_ts": f"{PD.to_date(day).isoformat()} 14:00:00", "period_end": "2026-09-30"}],
        "bars": bars or {"AAA": _bars(day), "BBB": _bars(day, last_close=103.0)},
        "nifty": {k: v for k, v in _nifty(day).items()}}
    return {"status": {}, "rows": {}, "factors": {}, "traded": {}, "nifty": {}, "decision": dec}


def _orders(tag):
    from kite_client import kite_client
    return [o for o in kite_client._paper_orders if o.get("tag") == tag]


def test_flag_default_off_nothing_scheduled_no_entries():
    import config
    import platform_scheduler as ps
    assert config.Settings.model_fields["validated_pead_enabled"].default is False
    assert settings.validated_pead_enabled is False
    from agents.strategy_agents import ALL_AGENTS
    agent = ALL_AGENTS["pead_h60"]
    assert agent.wanted() is False and agent.entry_block_reason_readonly() in ("kill_switch", "no_kite_data",
                                                                              "validated_only_off")

    class _S:
        def __init__(self):
            self.ids = []

        def add_job(self, *a, **k):
            self.ids.append(k.get("id"))

        def start(self):
            pass

    with mock.patch.object(settings, "trading_mode", "PAPER"):
        s = ps.PlatformScheduler()
        s._sched = _S()
        s.start()
        assert not any(str(i).startswith("validated_pead") for i in s._sched.ids), s._sched.ids
        with mock.patch.object(settings, "validated_pead_enabled", True):
            s2 = ps.PlatformScheduler()
            s2._sched = _S()
            s2.start()
        assert {"validated_pead_evening", "validated_pead_morning", "validated_pead_entry"} <= set(s2._sched.ids)
    with pead_env(_ist(D, 19, 35), flag=False) as (a, clk, p):
        with mock.patch.object(a, "_gather", side_effect=AssertionError("no input fetch while off")):
            assert asyncio.run(a.evening()) == {"skipped": "disabled"}
            assert asyncio.run(a.catch_up(_ist(D, 19, 40))) == {}
        assert a.entry_block_reason() == "kill_switch"
        a.pending = {"decision_day": D, "entry_day": D_NEXT, "entries": [{"symbol": "AAA", "ar": 0.06}]}
        clk[0] = _ist(D_NEXT, 9, 15, 10)
        out = asyncio.run(a.enter_at_open())
        assert out["opened"] == [] and out["skipped"] == {"*": "kill_switch"} and not _orders(V.ENTRY_TAG)


def test_paper_flow_decide_enter_stop_and_clock_exit():
    with pead_env(_ist(D, 19, 35), quotes={"AAA": {"open": 107.0, "ltp": 107.5}}) as (a, clk, p):
        with mock.patch.object(a, "_gather", return_value=_decision_g()):
            out = asyncio.run(a.evening())
        assert out["planned"] == ["AAA"] and a.pending["entry_day"] == D_NEXT, (out, a.pending)
        assert a.state.skip_counts.get("below_threshold") == 1
        # 09:15:10 next session: pre-open order filled at the official open
        clk[0] = _ist(D_NEXT, 9, 15, 10)
        out = asyncio.run(a.enter_at_open())
        assert out["opened"] == ["AAA"], out
        pos = a.positions()[0]
        assert pos.qty == V.entry_qty(107.0) == 934 and pos.entry_px == 107.0 and pos.product == "CNC"
        assert abs(pos.meta["stop"] - round(107.0 * 1.0002 * 0.8, 4)) < 1e-9 and pos.overnight
        o = _orders(V.ENTRY_TAG)
        assert len(o) == 1 and o[0]["fill_kind"] == "AUCTION" and o[0]["price"] == 107.0
        assert a.pending == {} and asyncio.run(a.enter_at_open())["opened"] == [], "entry once"
        # evening of the entry day: no stop; the evening after: the low goes through the stop
        nifty = {D_NEXT: (20000.0, 20100.0), 20261113: (20100.0, 19800.0)}
        g1 = {"status": {}, "nifty": nifty, "factors": {},
              "rows": {D_NEXT: {"AAA": {"open": 107.0, "high": 110.0, "low": 100.0, "close": 108.0,
                                        "prev_close_stored": 106.0}}},
              "traded": {"AAA": {"days": [D_NEXT], "complete": True}}}
        clk[0] = _ist(D_NEXT, 19, 35)
        with mock.patch.object(a, "_gather", return_value={**g1, "decision": {"skip": "no_event_feed"}}):
            asyncio.run(a.evening())
        assert a.positions() and a.positions()[0].meta["held_sessions"] == 1 and a.processed_through == D_NEXT
        assert a.flags.get("no_event_feed") == str(D_NEXT) and a.pending == {}
        g2 = {"status": {}, "nifty": nifty, "factors": {},
              "rows": {20261113: {"AAA": {"open": 95.0, "high": 96.0, "low": 80.0, "close": 82.0,
                                          "prev_close_stored": 108.0}}},
              "traded": {"AAA": {"days": [D_NEXT, 20261113], "complete": True}}}
        clk[0] = _ist(20261113, 19, 35)
        with mock.patch.object(a, "_gather", return_value={**g2, "decision": {"skip": "no_universe"}}):
            asyncio.run(a.evening())
        assert not a.positions()
        stop = round(107.0 * 1.0002 * 0.8, 4)
        row = [t for t in p.trades if t["strategy"] == "pead_h60"][-1]
        assert row["exit"] == round(stop, 2) and row["exit_reason"] == "stop" and row["candidate"] is True, row
        assert row["version"] == a.VERSION and row["product"] == "CNC" and row["cost"] > 0
        exp_ex = (round(stop, 2) / 107.0 - 1) - (19800.0 / 20000.0 - 1)
        assert abs(row["excess_vs_nifty"] - exp_ex) < 1e-4, row
        assert _orders(V.EXIT_TAG)[-1]["fill_kind"] == "STOP"
        # clock exit: the close of the 60th traded session
        a._positions.clear()
        pos = a._open(V.PlannedTrade("ZZZ", 1, 10, 100.0, "09:15", "x", 0.05), "CNC", "NSE", 100.0, "OID-Z",
                      overnight=True, meta={"entry_day": D_NEXT, "stop": 80.0})
        days = PD.sessions_between(PD.to_date(D_NEXT), PD.to_date(D_NEXT) + timedelta(days=120))[:60]
        g3 = {"rows": {days[-1]: {"ZZZ": {"open": 120.0, "high": 125.0, "low": 118.0, "close": 123.0,
                                          "prev_close_stored": 119.0}}},
              "traded": {"ZZZ": {"days": days, "complete": True}}, "factors": {}, "nifty": {}}
        a._manage_session(days[-2], {**g3, "rows": {days[-2]: g3["rows"][days[-1]]}}, late=False)
        assert a.positions(), "59th session: still held"
        a._manage_session(days[-1], g3, late=False)
        row = [t for t in p.trades if t["strategy"] == "pead_h60"][-1]
        assert row["exit"] == 123.0 and row["exit_reason"] == "max_hold" and row["held_sessions"] == 60, row


def test_fail_safe_inputs():
    """No event feed / empty feed window / no universe / missing bars: no entry, a skip key and a flag."""
    with pead_env(_ist(D, 19, 35)) as (a, clk, p):
        base = dict(ensure_bhav=lambda days, **k: {d: "have" for d in days}, ensure_nifty=lambda st, days: {},
                    corporate_actions=lambda raw=None: [], prune_bhav=lambda k: 0, load_state=lambda: {},
                    save_state=lambda st: None, ensure_symbol_changes=lambda st, today: [])

        def gather(**over):
            with contextlib.ExitStack() as es:
                for k, v in {**base, **over}.items():
                    es.enter_context(mock.patch.object(PD, k, v))
                return a._gather(clk[0], D, [], True, {})["decision"]

        assert gather(fetch_feeds=lambda today, feeds=None: {"ok": False, "feeds": {"ifr": "refused"}})["skip"] \
            == "no_event_feed"
        assert gather(fetch_feeds=lambda today, feeds=None: {"ok": True},
                      window_has_announcements=lambda raw, lo, hi: 0)["skip"] == "no_event_feed"
        assert gather(fetch_feeds=lambda today, feeds=None: {"ok": True},
                      window_has_announcements=lambda raw, lo, hi: 12,
                      equity_symbols=lambda st, today: None)["skip"] == "no_universe"
        assert gather(fetch_feeds=lambda today, feeds=None: {"ok": True},
                      window_has_announcements=lambda raw, lo, hi: 12,
                      equity_symbols=lambda st, today: {"AAA"},
                      universe_for=lambda day, st, eq, top=500: (None, "missing_files:3"))["skip"] == "no_universe"
        assert gather(fetch_feeds=lambda today, feeds=None: {"ok": True},
                      ensure_bhav=lambda days, **k: {d: ("missing" if d == D else "have") for d in days})["skip"] \
            == "no_bars"
        for skip in ("no_event_feed", "no_universe", "no_bars", "no_nifty"):
            a.last_decision = {}
            with mock.patch.object(a, "_gather", return_value=_decision_g(skip=skip)):
                asyncio.run(a.evening())
            assert a.pending == {} and a.flags.get(skip) == str(D) and a.state.skip_counts.get(skip, 0) >= 1, skip
        assert not _orders(V.ENTRY_TAG)
        # a later attempt the same evening (the feed came back) still decides
        with mock.patch.object(a, "_gather", return_value=_decision_g()):
            assert asyncio.run(a.evening())["planned"] == ["AAA"]


def test_paper_only_refusal():
    with pead_env(_ist(D, 19, 35)) as (a, clk, p):
        settings.trading_mode = "LIVE"
        with mock.patch.object(a, "_gather", side_effect=AssertionError("nothing in LIVE")):
            assert asyncio.run(a.evening()) == {"skipped": "live_mode_refused"}
        a.pending = {"decision_day": D, "entry_day": D_NEXT, "entries": [{"symbol": "AAA", "ar": 0.06}]}
        clk[0] = _ist(D_NEXT, 9, 15, 10)
        out = asyncio.run(a.enter_at_open())
        assert out["skipped"] == {"*": "live_mode_refused"} and not _orders(V.ENTRY_TAG)
        from kite_client import kite_client
        try:
            kite_client.paper_fill_at("AAA", "NSE", "BUY", 1, "CNC", 100.0, "x", "AUCTION")
            raise AssertionError("paper_fill_at must refuse in LIVE")
        except RuntimeError:
            pass
        import platform_scheduler as ps
        with mock.patch.object(settings, "validated_pead_enabled", True):
            assert ps.PlatformScheduler()._pead_wanted() is False, "no PEAD job is scheduled in LIVE"


def test_entry_window_caps_and_missing_open():
    q = {"AAA": {"open": 100.0}, "BBB": {"open": 0.0}}
    with pead_env(_ist(D_NEXT, 9, 15, 10), quotes=q) as (a, clk, p):
        settings_prev = settings.validated_pead_max_positions
        try:
            settings.validated_pead_max_positions = 1
            a.pending = {"decision_day": D, "entry_day": D_NEXT,
                         "entries": [{"symbol": "AAA", "ar": 0.05}, {"symbol": "BBB", "ar": 0.07},
                                     {"symbol": "CCC", "ar": 0.06}]}
            out = asyncio.run(a.enter_at_open())
        finally:
            settings.validated_pead_max_positions = settings_prev
        assert out["opened"] == ["AAA"] and out["skipped"]["BBB"] == "no_data:open" \
            and out["skipped"]["CCC"] == "no_data:open", out
        assert clk[0].time() >= V.ENTRY_DEADLINE, "missing opens retried until 09:29:50"
    with pead_env(_ist(D_NEXT, 9, 31), quotes=q) as (a, clk, p):
        a.pending = {"decision_day": D, "entry_day": D_NEXT, "entries": [{"symbol": "AAA", "ar": 0.05}]}
        out = asyncio.run(a.enter_at_open())
        assert out["skipped"] == {"*": "entry_window_missed"} and a.deviations.get("entry_window_missed") == 1
    with pead_env(_ist(D_NEXT, 9, 20)) as (a, clk, p):
        # an evening decision that could not be made before the 09:00 cut-off is skipped, not taken late
        a.last_decision = {}
        assert a._decision_open(D, _ist(D_NEXT, 8, 59)) and not a._decision_open(D, _ist(D_NEXT, 9, 0))


def test_corporate_action_during_hold():
    with pead_env(_ist(20261113, 19, 35)) as (a, clk, p):
        pos = a._open(V.PlannedTrade("BON", 1, 100, 200.0, "09:15", "x", 0.05), "CNC", "NSE", 200.0, "OID-B",
                      overnight=True, meta={"entry_day": D_NEXT, "stop": 160.0})
        g = {"rows": {20261113: {"BON": {"open": 101.0, "high": 104.0, "low": 99.0, "close": 103.0,
                                         "prev_close_stored": 201.0}}},
             "factors": {("BON", 20261113): (0.5, "applied")}, "traded": {"BON": {"days": [D_NEXT, 20261113],
                                                                                  "complete": True}}, "nifty": {}}
        a._manage_session(20261113, g, late=False)
        assert a.positions() and pos.qty == 200 and pos.entry_px == 100.0 and pos.meta["stop"] == 80.0, pos
        ca = _orders(V.CA_TAG)
        assert len(ca) == 1 and ca[0]["quantity"] == 100 and ca[0]["price"] == 0.0, ca
        dem = a._open(V.PlannedTrade("DEM", 1, 50, 100.0, "09:15", "x", 0.05), "CNC", "NSE", 100.0, "OID-D",
                      overnight=True, meta={"entry_day": D_NEXT, "stop": 80.0})
        g = {"rows": {20261113: {"DEM": {"open": 60.0, "high": 62.0, "low": 58.0, "close": 61.0,
                                         "prev_close_stored": 101.0}}},
             "factors": {("DEM", 20261113): (0.6, "applied_gap_scheme")}, "traded": {}, "nifty": {}}
        a._manage_session(20261113, g, late=False)
        assert dem.oid not in a._positions and a.deviations.get("corporate_action_exit") == 1
        row = [t for t in p.trades if t["symbol"] == "DEM"][-1]
        assert abs(row["exit"] - round(61.0 / 0.6, 2)) < 0.01 and row["exit_kind"] == "corporate_action_exit", row


def test_held_stock_renamed():
    """NSE renames a held stock: the position and its paper row follow the new symbol, then exit by the rule."""
    from kite_client import kite_client
    with pead_env(_ist(20261113, 19, 35)) as (a, clk, p):
        oid, _ = V.paper_fill_at("OLDCO", "NSE", "BUY", 10, "CNC", 100.0, V.ENTRY_TAG, "AUCTION")
        pos = a._open(V.PlannedTrade("OLDCO", 1, 10, 100.0, "09:15", "x", 0.05), "CNC", "NSE", 100.0, oid,
                      overnight=True, meta={"entry_day": D_NEXT, "stop": 80.0})
        a.processed_through = D_NEXT
        g = {"status": {}, "nifty": {}, "factors": {}, "renames": {"OLDCO": "NEWCO"},
             "rows": {20261113: {"NEWCO": {"open": 90.0, "high": 91.0, "low": 70.0, "close": 75.0,
                                           "prev_close_stored": 100.0}}},
             "traded": {"NEWCO": {"days": [D_NEXT, 20261113], "complete": True}}, "decision": {"skip": "no_bars"}}
        with mock.patch.object(a, "_gather", return_value=g):
            asyncio.run(a.evening())
        assert a.deviations.get("symbol_renamed") == 1 and not a.positions()
        row = [t for t in p.trades if t["strategy"] == "pead_h60"][-1]
        assert row["symbol"] == "NEWCO" and row["exit"] == 80.0, row
        rows = {r["tradingsymbol"]: r["quantity"] for r in kite_client._paper_positions}
        assert rows.get("NEWCO") == 0 and "OLDCO" not in rows, rows


def test_status_readiness_and_restore():
    with pead_env(_ist(D, 20, 0)) as (a, clk, p):
        st = a.get_status()
        assert st["lab_id"] == "EV-PEAD-LONG-H60" and st["candidate"] is True and st["proven"] is False
        assert st["lab_expectation"]["FINAL"]["trades"] == 296 and st["flag"] == "VALIDATED_PEAD_ENABLED"
        lr = st["live_readiness"]
        assert lr["never_switches_to_live"] is True and lr["ready"] is False and "excess" in lr["criterion"].lower()
        assert st["caps"] == {"max_positions": 100, "max_gross": 10_000_000.0, "notional": 100_000.0}
        assert st["limits"] == {"daily_loss_stop": 240_000.0, "dd_kill": 1_680_000.0}
        a.pending = {"decision_day": D, "entry_day": D_NEXT, "entries": [{"symbol": "AAA", "ar": 0.06}]}
        a.processed_through = D
        b = V.PeadCandidateAgent()
        b.import_book(a.export_book(), closed=set(), stale=False)
        assert b.pending == a.pending and b.processed_through == D


def test_shadow_enrolment():
    """Enrolled as name@VERSION only while VALIDATED_PEAD_ENABLED is on: the rule-v2 Bonferroni family then grows
    by one (expected, permanent); off -> not enrolled, the family is unchanged."""
    import shadow_registry as SR
    a = V.PeadCandidateAgent()
    spec = SR.agent_spec(a)
    assert spec.id == "pead_h60@EV-PEAD-LONG-H60/ev1/v1" and spec.family == "EV-PEAD-LONG-H60"
    c = spec.config
    assert c["max_hold_days"] == 95 and c["backtest_max_dd"] == 1118594.0 and c["per_trade_notional"] == 100000
    assert c["lab_fingerprint"] == V.LAB_FINGERPRINT
    for flag in (False, True):
        reg = SR.ShadowRegistry(tempfile.mkdtemp())
        with mock.patch.object(settings, "shadow_registry_enabled", True), \
             mock.patch.object(settings, "trading_mode", "PAPER"), \
             mock.patch.object(settings, "validated_pead_enabled", flag), \
             mock.patch("forward_test._load", return_value={"days": {}}), \
             mock.patch.object(SR, "_today", return_value=date(2026, 11, 16)):
            out = SR.daily_job(reg, {"pead_h60": a})
        assert (spec.id in out["enrolled"]) is flag, out["enrolled"]
        n_fwd = len(SR.forward_test_specs())
        assert len(reg.family_ids()) == n_fwd + (1 if flag else 0)
        if flag:
            e = reg.entries()[spec.id]
            assert e["forward_from"] == (date(2026, 11, 16) + timedelta(days=96)).isoformat()


TESTS = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


def _main() -> int:
    if "--build-fixture" in sys.argv:
        b = PP.build_fixture()
        print(f"fixture: {len(b['days'])} days -> {PP.FIXTURE}")
        return 0
    failed = 0
    for n, f in TESTS:
        try:
            f()
            print(f"  ok    {n}")
        except Skip as exc:
            print(f"  skip  {n}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {str(exc)[:400]}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
