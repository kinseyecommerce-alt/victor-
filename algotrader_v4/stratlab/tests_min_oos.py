"""
stratlab/tests_min_oos.py — tests for the old-era minute-bar OOS harness (protocol_min_oos.py) on SYNTHETIC data only:
the H1 / H2 split by entry date and the era bounds, the gates (each failing reason, the per-half trade minimum by
horizon, the 35 % symbol cap and its one-instrument exemption), that the aggregate judgement equals protocol_oos's
trade-list arithmetic, that worker aggregates merge exactly, the dirty-link rule, the report-only excess line, the
one-look ledger (a judged strategy is returned from the ledger and never recomputed, a changed fingerprint / changed
gates / an id outside the pre-registration are refused) and the pre-registration sha256 / repository-copy checks.
No catalogue strategy is run on the old era here. Run by test_pipeline.py (section "STRATLAB: old-era minute OOS")
or directly: python -m stratlab.tests_min_oos
"""
from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from . import protocol as P
from . import protocol_min_oos as M

CAL = np.array([20160104, 20170601, 20191231, 20200101, 20200102, 20231109], np.int64)


def _t(sym, date, net, side=1, exit_date=None, entry_t="09:15", exit_t="09:20", px=100.0, qty=1000, gross=None):
    g = net + 50.0 if gross is None else gross
    return ("X", sym, date, side, entry_t, px, exit_date or date, exit_t, px, qty, g, g - net, net, "time")


def _many(sym, date, n, net):
    return [_t(sym, date, net) for _ in range(n)]


def t_split_by_entry_date():
    tr = [_t("A", 20191231, 10.0, exit_date=20200102), _t("A", 20200101, -5.0), _t("A", 20151231, 99.0),
          _t("A", 20231110, 99.0)]
    a = M.agg_from_trades(tr, CAL)
    assert a["h1"][0] == 1 and a["h1"][1] == 10.0, a["h1"]          # entered in H1, exited in H2 -> H1
    assert a["h2"][0] == 1 and a["h2"][1] == -5.0, a["h2"]
    assert a["outside_era"] == 2 and a["n"] == 2
    assert M.H1 == (20160101, 20191231) and M.H2 == (20200101, 20231109) and M.ERA == (20160101, 20231109)
    # the H1 trade's net is booked on its exit session in the daily series
    assert a["daily"][list(CAL).index(20200102)] == 10.0


def t_gates_each_reason():
    def j(h1, h2, horizon="intraday", single=False):
        tr = h1 + h2
        return M.judge(M.agg_from_trades(tr, CAL), horizon, single_underlying=single)
    ok = j(_many("A", 20170601, 80, 10.0) + _many("B", 20170601, 80, 10.0) + _many("C", 20170601, 80, -2.0),
           _many("A", 20200102, 60, 10.0) + _many("B", 20200102, 60, 10.0) + _many("C", 20200102, 60, 10.0))
    assert not ok["passed"] and any("A is" in r for r in ok["reasons"]), ok    # A = 1400 / 3260 = 43 % > 35 %
    sym = ["S%d" % k for k in range(10)]
    good = [_t(s, 20170601, 10.0) for s in sym for _ in range(16)] + [_t(s, 20200102, 10.0) for s in sym for _ in range(16)]
    r = M.judge(M.agg_from_trades(good, CAL), "intraday")
    assert r["passed"], r                                     # 160 trades per half >= 150, PF inf, no symbol > 35 %
    few = [_t(s, 20170601, 10.0) for s in sym for _ in range(14)] + [_t(s, 20200102, 10.0) for s in sym for _ in range(16)]
    r = M.judge(M.agg_from_trades(few, CAL), "intraday")
    assert not r["passed"] and r["reasons"] == ["H1 140 trades < 150"], r
    assert M.judge(M.agg_from_trades(few, CAL), "positional")["passed"]          # positional minimum is 60
    neg = [_t(s, 20170601, -1.0) for s in sym for _ in range(16)] + [_t(s, 20200102, 10.0) for s in sym for _ in range(16)]
    r = M.judge(M.agg_from_trades(neg, CAL), "intraday")
    assert "H1 net <= 0" in r["reasons"] and r["passed"] is False
    lowpf = ([_t(s, 20170601, 11.0) for s in sym for _ in range(8)] + [_t(s, 20170601, -10.0) for s in sym for _ in range(8)]
             + [_t(s, 20200102, 11.0) for s in sym for _ in range(8)] + [_t(s, 20200102, -10.0) for s in sym for _ in range(8)])
    r = M.judge(M.agg_from_trades(lowpf, CAL), "intraday")
    assert r["reasons"] == ["PF 1.100 < 1.15"], r
    # one instrument: the symbol cap does not apply, every other gate does
    one = _many("_NIFTY50_min", 20170601, 70, 10.0) + _many("_NIFTY50_min", 20200102, 70, 10.0)
    assert M.judge(M.agg_from_trades(one, CAL), "positional", single_underlying=True)["passed"]
    assert not M.judge(M.agg_from_trades(one, CAL), "positional", single_underlying=False)["passed"]
    # the gates are protocol_oos's constants
    assert M.MIN_PF == P.SCREEN_MIN_PF and M.MIN_TRADES_PER_HALF == P.SCREEN_MIN_TRADES
    assert M.MAX_SYMBOL_SHARE == P.SCREEN_MAX_SYMBOL_SHARE


def t_agg_equals_trade_list_arithmetic():
    rng = np.random.default_rng(3)
    tr = []
    for k in range(400):
        d = int(rng.choice([20160104, 20170601, 20191231, 20200101, 20200102, 20231109]))
        tr.append(_t("S%d" % rng.integers(0, 7), d, float(np.round(rng.normal(3, 40), 2))))
    a = M.agg_from_trades(tr, CAL)
    h1 = [t for t in tr if t[2] <= 20191231]
    h2 = [t for t in tr if t[2] >= 20200101]
    for half, ref in (("h1", h1), ("h2", h2)):
        mine, theirs = M._stats(a[half]), P.stats(ref)
        assert mine["trades"] == theirs["trades"] and abs(mine["net"] - theirs["net"]) < 1e-6
        assert mine["pf"] == theirs["pf"] and mine["win_pct"] == theirs["win_pct"], (mine, theirs)
    j = M.judge(a, "positional")
    assert j["era"]["pf"] == P.stats(tr)["pf"]
    # merging worker aggregates is exact
    b1 = M.agg_from_trades(tr[:150], CAL)
    b2 = M.agg_from_trades(tr[150:], CAL)
    m = M.merge_agg(b1, b2)
    assert m["h1"][0] == a["h1"][0] and abs(m["h1"][1] - a["h1"][1]) < 1e-6 and abs(m["s2"] - a["s2"]) < 1e-3
    assert np.allclose(m["daily"], a["daily"]) and dict(m["by_sym"]).keys() == dict(a["by_sym"]).keys()


def t_dirty_rule():
    dirty = {"A": np.array([20170601], np.int64)}
    nxt = {"A": {20160104: 20170601, 20170601: 20191231, 20191231: 20200101}}
    tr = [_t("A", 20170601, 1.0),                              # entry on a dirty date
          _t("A", 20160104, 1.0, exit_date=20191231),          # span contains it
          _t("A", 20160104, 1.0),                              # the next session after the exit is dirty
          _t("A", 20191231, 1.0),                              # clean
          _t("B", 20170601, 1.0)]                              # other symbol: clean
    a = M.agg_from_trades(tr, CAL, dirty, nxt)
    assert a["dirty_excluded"] == 3 and a["n"] == 2, (a["dirty_excluded"], a["n"])
    b = M.agg_from_trades(tr, CAL, dirty, nxt, apply_dirty=False)
    assert b["n"] == 5


def t_excess_vs_index():
    sd = SimpleNamespace(date=np.array([20170601, 20170601, 20170601], np.int32),
                         t=np.array([555, 560, 929], np.int16), o=np.array([100.0, 101.0, 104.0]),
                         c=np.array([100.5, 101.5, 105.0]))
    ref = M.IndexRef(sd)
    assert ref.price_at(20170601, "09:15") == 100.0 and ref.price_at(20170601, "09:20") == 101.0
    assert ref.price_at(20170601, "15:29", close=True) == 105.0
    assert ref.price_at(20170601, "09:17") == 100.5            # last price before the minute
    tr = [_t("A", 20170601, 500.0, side=1, entry_t="09:15", exit_t="09:20", px=100.0, qty=1000)]
    a = M.agg_from_trades(tr, CAL, refs={"NIFTY": ref})
    # notional 1 lakh, NIFTY +1 % -> excess = 500 - 1000
    assert a["ex"]["h1"][0] == 1 and abs(a["ex"]["h1"][1] + 500.0) < 1e-6, a["ex"]
    tr = [_t("A", 20170601, 500.0, side=-1, entry_t="09:15", exit_t="09:20", px=100.0, qty=1000)]
    ex = M.agg_from_trades(tr, CAL, refs={"NIFTY": ref})["ex"]["h1"]
    assert ex[0] == 1 and abs(ex[1] - 1500.0) < 1e-6, ex


class _Spec:
    def __init__(self, id_, fp, horizon="intraday"):
        self.id, self.fingerprint, self.horizon, self.underlying = id_, fp, horizon, None


def t_ledger_one_look_and_refusals():
    with tempfile.TemporaryDirectory() as d:
        led = os.path.join(d, "ledger.json")
        sp = _Spec("S1", "fp1")
        prereg = {"gates_fingerprint": M.GATES_FINGERPRINT, "strategies": {"S1": {"fingerprint": "fp1"}}}
        done, todo = M.partition([sp], lambda s: "cm", prereg, led)
        assert done == [] and [s.id for s in todo] == ["S1"]
        sym = ["S%d" % k for k in range(10)]
        tr = [_t(s, 20170601, 10.0) for s in sym for _ in range(16)] + [_t(s, 20200102, 10.0) for s in sym for _ in range(16)]
        res = {"agg": M.agg_from_trades(tr, CAL), "sens_agg": None}
        rec = M.record([(sp, res)], lambda s: "cm", ledger_path=led)
        assert rec[0]["verdict"] == "OOS-CONFIRMED", rec[0]
        # second run: returned from the ledger, nothing to recompute
        done, todo = M.partition([sp], lambda s: "cm", prereg, led)
        assert todo == [] and done[0]["from_ledger"] and done[0]["verdict"] == "OOS-CONFIRMED"
        # a second record() of a different result does not overwrite the first (first write wins)
        res2 = {"agg": M.agg_from_trades([_t("A", 20170601, -1.0)], CAL), "sens_agg": None}
        M.record([(sp, res2)], lambda s: "cm", ledger_path=led)
        assert json.load(open(led))["S1|oldera_min|cm"]["result"]["verdict"] == "OOS-CONFIRMED"
        # changed code after the judgement -> refused (also not in the pre-registration with that fingerprint)
        sp2 = _Spec("S1", "fp2")
        done, todo = M.partition([sp2], lambda s: "cm", None, led)
        assert todo == [] and done[0]["verdict"] == "NOT JUDGED" and "no second look" in done[0]["why"]
        done, todo = M.partition([sp2], lambda s: "cm", prereg, led)
        assert todo == [] and "changed since the pre-registration" in done[0]["why"]
        # not pre-registered / gates differ
        done, todo = M.partition([_Spec("S9", "x")], lambda s: "cm", prereg, led)
        assert todo == [] and done[0]["why"] == "not in the pre-registration"
        bad = dict(prereg, gates_fingerprint="000")
        done, todo = M.partition([_Spec("S9", "x")], lambda s: "cm", bad, led)
        assert done[0]["why"] == "gates differ from the pre-registration"
        # another cost model is its own key
        done, todo = M.partition([sp], lambda s: "cm2", prereg, led)
        assert [s.id for s in todo] == ["S1"]


def t_prereg_sha_checks():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pre.json")
        with open(p, "w") as fh:
            json.dump({"gates_fingerprint": M.GATES_FINGERPRINT, "strategies": {}}, fh)
        sha = M._sha(p)
        saved = M.REPO_PREREG
        try:
            M.REPO_PREREG = os.path.join(d, "repo.json")              # no repo copy yet: only the sha is checked
            assert M.load_prereg(p, sha)["_sha256"] == sha
            try:
                M.load_prereg(p, "0" * 64)
                raise AssertionError("a wrong sha256 must be refused")
            except SystemExit as e:
                assert "refused" in str(e)
            with open(M.REPO_PREREG, "w") as fh:
                fh.write("{}")
            try:
                M.load_prereg(p, sha)
                raise AssertionError("a pre-registration that differs from the repository copy must be refused")
            except SystemExit as e:
                assert "differs from the repository copy" in str(e)
        finally:
            M.REPO_PREREG = saved


def t_gates_fingerprint_frozen():
    assert M.GATES["min_trades_per_half"] == {"intraday": 150, "positional": 60}
    assert M.GATES["h1"] == (20160101, 20191231) and M.GATES["h2"] == (20200101, 20231109)
    assert M.GATES_FINGERPRINT == "bf4933d2c1f1", M.GATES_FINGERPRINT


def t_groups_cover_every_minute_spec():
    g = M.groups()
    assert g["a"] == list(M.LIVE) and set(g["b"]) == set(M.CONFIRMED_MINUTE)
    allids = set(M.module_specs()) | set(M.catalogue_minute_specs())
    inel = {i for i, _w in g["ineligible"]}
    assert allids == set(g["a"]) | set(g["b"]) | set(g["c"]) | (inel & allids)
    assert not (set(g["c"]) & inel) and len(set(g["c"])) == len(g["c"])
    assert M.groups()["c"] == g["c"]                                  # fixed permutation (seeded)


TESTS = [
    ("min OOS: H1 / H2 by entry date, era bounds, P&L booked on the exit session", t_split_by_entry_date),
    ("min OOS: gates (each reason, 150 / 60 per half by horizon, 35 % cap, one-instrument exemption)",
     t_gates_each_reason),
    ("min OOS: aggregate judgement == protocol trade-list arithmetic; worker aggregates merge exactly",
     t_agg_equals_trade_list_arithmetic),
    ("min OOS: dirty-link rule (entry / span / next session after the exit)", t_dirty_rule),
    ("min OOS: report-only excess vs NIFTY (long and short)", t_excess_vs_index),
    ("min OOS: one-look ledger: no recompute, first write wins, changed / unregistered / other gates refused",
     t_ledger_one_look_and_refusals),
    ("min OOS: pre-registration sha256 and repository-copy checks", t_prereg_sha_checks),
    ("min OOS: gates fingerprint frozen", t_gates_fingerprint_frozen),
    ("min OOS: groups a / b / c / ineligible cover every minute-bar strategy once, fixed order", t_groups_cover_every_minute_spec),
]


if __name__ == "__main__":
    bad = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:                 # noqa: BLE001
            bad += 1
            print(f"[FAIL] {name}: {exc!r}")
    raise SystemExit(1 if bad else 0)
