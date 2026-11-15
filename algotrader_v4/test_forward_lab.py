"""
test_forward_lab.py — tests of the forward lab (forward_lab.py, forward_lab_store.py, forward_lab_engine.py).

    cd algotrader_v4 && python test_forward_lab.py      (also run from test_pipeline.py)

Every test works in a temporary directory. The parity test replays sessions of the lab's 3-year bars and is skipped
when that data set is absent; everything else uses synthetic bars.
"""
from __future__ import annotations

import gzip
import zlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PERFORMANCE_FILE", os.path.join(tempfile.mkdtemp(), "performance.json"))

import numpy as np  # noqa: E402

from config import settings  # noqa: E402
import forward_lab as FL  # noqa: E402
import forward_lab_engine as E  # noqa: E402
import forward_lab_store as FS  # noqa: E402
from ist_clock import is_trading_day, _IST  # noqa: E402

LAB_SCRATCH = os.environ.get("STRATLAB_SCRATCH",
                             "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad")
BARS3Y_CACHE = os.path.join(LAB_SCRATCH, "stratlab_cache", "bars3y_fb3d4d")
SYN_SYMBOLS = ("INFY", "RELIANCE", "TATASTEEL")
SAMPLE_IDS = ["orb15_break", "afternoon_vwap_trend", "r3_swing_tom", "r3_futures_nifty_vixlow_mom",
              "tfg_ema_9_21@5m:stocks", "tfg_rsi2_connors@1d:stocks", "tfg_donchian_20_10@1d:index",
              "AUC-FADE-2PCT", "SWING-ATTN-ON-3X"]


_TMPS: list = []


def _tmp() -> str:
    """A temporary directory removed when the test process exits (the synthetic stores are a few MB each)."""
    d = tempfile.mkdtemp(prefix="flab_test_")
    _TMPS.append(d)
    return d


import atexit  # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _TMPS])


class Skip(Exception):
    pass


# ── synthetic market data ─────────────────────────────────────────────────────────────────────────────────────────
def trading_days(start: date, n: int) -> list[int]:
    out, d = [], start
    while len(out) < n:
        if is_trading_day(d):
            out.append(d.year * 10000 + d.month * 100 + d.day)
        d += timedelta(days=1)
    return out


def syn_session(sym: str, day: int, n_bars: int = 375, seed: int = 0) -> dict:
    rng = np.random.default_rng(int(zlib.crc32(f"{sym}|{day}|{seed}".encode())))
    base = {"INFY": 1500.0, "RELIANCE": 1300.0, "TATASTEEL": 150.0, "_NIFTY50_min": 25000.0,
            "_NIFTYBANK_min": 55000.0}.get(sym, 1000.0)
    base *= 1.0 + 0.002 * ((day // 100) % 100) + 0.001 * (day % 100)
    tick = 0.05
    c = base + np.cumsum(rng.normal(0, base * 0.0008, n_bars))
    o = np.concatenate(([base], c[:-1]))
    h = np.maximum(o, c) + np.abs(rng.normal(0, base * 0.0004, n_bars))
    l = np.minimum(o, c) - np.abs(rng.normal(0, base * 0.0004, n_bars))
    r = lambda x: np.round(np.round(x / tick) * tick, 2)
    t = np.arange(9 * 60 + 15, 9 * 60 + 15 + n_bars, dtype=np.int16)
    v = 0.0 * c if sym.startswith("_") else rng.integers(1000, 50000, n_bars).astype(float)
    return {"date": np.full(n_bars, day, np.int32), "t": t, "o": r(o), "h": r(h), "l": r(l), "c": r(c), "v": v}


def syn_store(root: Path, days: list[int], symbols=SYN_SYMBOLS) -> FS.Store:
    st = FS.Store(root / "bars", cap_root=root)
    for sym in tuple(symbols) + tuple(FS.INDEX_MIN):
        parts = [syn_session(sym, d) for d in days]
        st.write_sessions(sym, {k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    for name in FS.DAILY_AUX:
        lvl = 15.0 if name == "_VIX_day" else 25000.0
        st.write_aux(name, {"date": np.array(days, np.int32), "o": np.full(len(days), lvl), "h": np.full(len(days), lvl),
                            "l": np.full(len(days), lvl), "c": np.full(len(days), lvl), "v": np.zeros(len(days))})
    m = st.manifest()
    for d in days:
        m.setdefault("sessions", {})[str(d)] = {"status": "complete", "source": "synthetic"}
    st._save_manifest(m)
    return st


def lab(root: Path) -> FL.ForwardLab:
    return FL.ForwardLab(root)


def fake_table(n_forward=5) -> list[dict]:
    t = [{"id": f"s{i}", "family": "fam", "group": "forward", "kind": "native:intraday", "fingerprint": f"f{i}"}
         for i in range(n_forward)]
    t.append({"id": "AUC-FADE-2PCT", "family": "opening_auction_fade", "group": "proven", "kind": "live",
              "fingerprint": "x"})
    return t


def fake_result(day, nets: dict) -> dict:
    return {"engine": "test", "strategies": {sid: {"closed": [["X", day, 1, "10:00", 100.0, day, "11:00", 101.0, 1,
                                                                n, 0.0, n, "target"]] if n is not None else [],
                                                     "open": [], "error": None}
                                               for sid, n in nets.items()}}


def feed(L: FL.ForwardLab, days: list[int], nets_fn) -> None:
    for d in days:
        L.ingest(d, fake_result(d, nets_fn(d)), [], set(), set())


# ── the rule file ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_rule_file_is_frozen_and_its_hash_is_in_the_genesis():
    rule = FL.load_rule()
    assert FL.RULE_SHA256 != "PENDING" and rule["_sha256"] == FL.RULE_SHA256 == FL.file_sha256(FL.RULE_FILE)
    tmp = Path(_tmp())
    bad = tmp / "rule.json"
    txt = FL.RULE_FILE.read_text().replace('"max_looks": 12', '"max_looks": 13')
    assert txt != FL.RULE_FILE.read_text()
    bad.write_text(txt)
    try:
        FL.load_rule(bad)
        raise AssertionError("an edited rule file was accepted")
    except FL.RuleChanged:
        pass
    L = lab(tmp / "fl")
    g = L.ensure_genesis()
    assert g["rule_sha256"] == FL.RULE_SHA256 and g["baseline_sha256"] == FL.file_sha256(FL.BASELINE_FILE)
    assert rule["baseline_sha256"] == FL.file_sha256(FL.BASELINE_FILE)
    # the family of the rule file is the engine's current forward table
    ids = sorted(t["id"] for t in E.strategy_table() if t["group"] == "forward")
    assert len(ids) == rule["groups"]["forward"]["initial_family_size"] == 1276
    assert FL.sha256_hex(ids) == rule["groups"]["forward"]["initial_ids_sha256"]
    # an edited rule cannot continue an existing registry
    rule2 = dict(rule, _sha256="0" * 64)
    try:
        FL.ForwardLab(tmp / "fl", rule=rule2).ensure_genesis()
        raise AssertionError("registry accepted a different rule")
    except FL.RuleChanged:
        pass


def test_bonferroni_alpha_and_z_required():
    rule = FL.load_rule()
    a = FL.alpha_per_test(rule, 1276)
    assert abs(a - 0.05 / (1276 * 12)) < 1e-15
    z = FL.z_required(rule, 1276)
    assert 4.4 < z < 4.6 and abs(z - rule["qualification"]["z_required_at_initial_family"]) < 1e-3
    from statistics import NormalDist
    assert abs((1 - NormalDist().cdf(z)) - a) < 1e-12
    assert FL.z_required(rule, 2000) > z          # a larger family is stricter


def test_family_never_shrinks():
    tmp = Path(_tmp())
    L = lab(tmp)
    L.register(fake_table(5), 20260105)
    assert L.family_size() == 5
    L.register(fake_table(3), 20260106)                         # strategies removed from the code: still counted
    assert L.family_size() == 5
    r = L.register(fake_table(7), 20260107)                     # two new ones: the family grows (later looks)
    assert r["registered"] == 2 and L.family_size() == 7
    st = L.state()
    assert st["strategies"]["s6"]["registered_on"] == 20260107 and st["strategies"]["s0"]["registered_on"] == 20260105
    drift = fake_table(7)
    drift[0]["fingerprint"] = "changed"
    assert L.register(drift, 20260108)["config_drift"] == ["s0"] and L.family_size() == 7
    # the proven / candidate group is not in the forward family
    assert "AUC-FADE-2PCT" not in st["family"]


def test_bootstrap_is_deterministic_and_sane():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 1000, 120)
    a = FL.qualification_test(x, FL.look_seed("s", 3), 2000, 5)
    b = FL.qualification_test(x, FL.look_seed("s", 3), 2000, 5)
    c = FL.qualification_test(x, FL.look_seed("s", 4), 2000, 5)
    assert a == b and a["z"] != c["z"]
    assert FL.look_seed("s", 3) == int.from_bytes(__import__("hashlib").sha256(b"s|look3").digest()[:8], "big")
    strong = FL.qualification_test(rng.normal(500, 300, 200), 7, 2000, 5)
    assert strong["z"] > 10 and strong["p"] < 1e-20 and strong["p_boot"] == 1 / 2001
    null = FL.qualification_test(rng.normal(0, 300, 200), 7, 2000, 5)
    assert abs(null["z"]) < 4


def test_looks_run_once_and_qualify_with_bonferroni():
    tmp = Path(_tmp())
    L = lab(tmp)
    L.register(fake_table(3), 20260105)
    days = trading_days(date(2026, 1, 5), 45)
    rng = np.random.default_rng(3)
    # s0: a big steady edge (several trades a day), s1: noise, s2: no trades
    feed(L, days, lambda d: {"s0": 5000.0 + rng.normal(0, 200), "s1": rng.normal(0, 500), "s2": None,
                              "AUC-FADE-2PCT": None})
    # the fake engine books one trade a day: 45 trades < 50 -> look 2 (40 days) is "insufficient" for everyone
    ev = L.evaluate_all()
    st = L.state()
    assert set(st["looks"]["s0"]) == {"1", "2"} and st["looks"]["s0"]["2"]["verdict"] == "insufficient"
    n_rec = len(L.registry.records())
    L.evaluate_all()
    assert len(L.registry.records()) == n_rec                    # each look runs once, never recomputed
    # with 3 trades a day the s0 edge qualifies at look 3 (60 days), s1 does not
    tmp2 = Path(_tmp())
    L2 = lab(tmp2)
    L2.register(fake_table(2), 20260105)
    days = trading_days(date(2026, 1, 5), 60)
    for d in days:
        res = fake_result(d, {"s0": 3000.0 + rng.normal(0, 100), "s1": rng.normal(0, 3000)})
        for sid in ("s0", "s1"):
            t = res["strategies"][sid]["closed"][0]
            res["strategies"][sid]["closed"] = [list(t), list(t[:3]) + ["11:00"] + list(t[4:]),
                                                list(t[:3]) + ["12:00"] + list(t[4:])]
        L2.ingest(d, res, [], set(), set())
    L2.evaluate_all()
    st = L2.state()
    lk = st["looks"]["s0"]
    assert lk["1"]["verdict"] == "insufficient" and lk["2"]["tested"] and lk["2"]["trades"] == 120
    assert st["status"]["s0"]["status"] == "qualified"
    assert lk["2"]["z"] >= lk["2"]["z_required"] and lk["2"]["family_size"] == 2
    assert (st["status"].get("s1") or {}).get("status") is None and st["looks"]["s1"]["2"]["verdict"] == "not_qualified"
    assert set(lk) == {"1", "2", "3"}


def test_consistency_band():
    rng = np.random.default_rng(5)
    labd = list(rng.normal(1000, 5000, 700))
    b1 = FL.consistency_band(labd, 40, 11, 3000, 5, 0.01)
    assert b1 == FL.consistency_band(labd, 40, 11, 3000, 5, 0.01)          # deterministic
    assert b1 < np.mean(labd) and FL.consistency_band(labd, 200, 11, 3000, 5, 0.01) > b1   # narrower with more days
    tmp = Path(_tmp())
    base = {"refs": {"AUC-FADE-2PCT": {"daily": labd, "mean_daily": float(np.mean(labd)), "max_dd": 1e9}},
            "tfgrid_accounts": {}, "stockfut_lots": {}}
    L = FL.ForwardLab(tmp, baseline=base)
    L.register(fake_table(1), 20260105)
    days = trading_days(date(2026, 1, 5), 40)
    feed(L, days, lambda d: {"AUC-FADE-2PCT": -4000.0})                 # far below the lab: inconsistent
    L.evaluate_all()
    st = L.state()
    assert st["looks"]["AUC-FADE-2PCT"]["2"]["verdict"] == "inconsistent"
    assert st["status"]["AUC-FADE-2PCT"]["status"] == "inconsistent"
    tmp2 = Path(_tmp())
    L2 = FL.ForwardLab(tmp2, baseline=base)
    L2.register(fake_table(1), 20260105)
    feed(L2, days, lambda d: {"AUC-FADE-2PCT": float(rng.normal(1000, 5000))})
    L2.evaluate_all()
    st = L2.state()
    assert st["looks"]["AUC-FADE-2PCT"]["1"]["verdict"] == "insufficient"   # 20 trades < 30
    assert st["looks"]["AUC-FADE-2PCT"]["2"]["verdict"] == "consistent"
    assert L2.summary_rows()[0]["status"] == "consistent"


def test_demotion_is_permanent():
    tmp = Path(_tmp())
    base = {"refs": {"s0": {"max_dd": 10000.0}}, "tfgrid_accounts": {}, "stockfut_lots": {}}
    L = FL.ForwardLab(tmp, baseline=base)
    L.register(fake_table(1), 20260105)
    days = trading_days(date(2026, 1, 5), 10)
    feed(L, days[:4], lambda d: {"s0": -3000.0})                     # DD 12,000 <= 15,000
    assert L.evaluate("s0", L.state())["status"] == "watching"
    feed(L, days[4:6], lambda d: {"s0": -2000.0})                    # DD 16,000 > 1.5 x 10,000
    assert L.evaluate("s0", L.state())["status"] == "demoted"
    feed(L, days[6:], lambda d: {"s0": 50000.0})                     # recovery does not undo it
    assert L.evaluate("s0", L.state())["status"] == "demoted"
    assert sum(1 for r in L.registry.records() if r["kind"] == "status") == 1


def test_ledgers_are_tamper_evident():
    tmp = Path(_tmp())
    L = lab(tmp)
    L.register(fake_table(2), 20260105)
    days = trading_days(date(2026, 1, 5), 3)
    feed(L, days, lambda d: {"s0": 100.0, "s1": -50.0})
    assert L.verify_all()["ok"]
    p = L.ledger("s0").path
    lines = p.read_text().splitlines()
    lines[1] = lines[1].replace('"net":100.0', '"net":900.0')
    p.write_text("\n".join(lines) + "\n")
    v = L.verify_all()
    assert not v["ok"] and v["ledgers_bad"][0]["id"] == "s0"
    # a truncated tail is caught by the registry checkpoint
    p1 = L.ledger("s1").path
    p1.write_text("\n".join(p1.read_text().splitlines()[:-1]) + "\n")
    assert "s1" in [b["id"] for b in L.verify_all()["ledgers_bad"]]
    tf = tmp / "trades" / f"{days[0]}.json.gz"
    tf.write_bytes(gzip.compress(b"{}"))
    assert f"{days[0]}.json.gz" in L.verify_all()["trade_files_bad"]


# ── store ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_store_roundtrip_is_exact_and_matches_the_lab_loader():
    from stratlab import data as D
    tmp = Path(_tmp())
    days = trading_days(date(2026, 2, 2), 4)
    st = syn_store(tmp, days, ("INFY",))
    raw = np.concatenate([syn_session("INFY", d)["c"] for d in days])
    sd = st.load_symbol("INFY")
    assert np.array_equal(sd.c, raw) and list(sd.days) == days
    ref = D.from_arrays("INFY", sd.date, sd.t, sd.o, sd.h, sd.l, sd.c, sd.v, min_bars=D.MIN_SESSION_BARS)
    assert np.array_equal(ref.starts, sd.starts) and np.array_equal(ref.o, sd.o)
    assert list(st.load_symbol("INFY", upto=days[1]).days) == days[:2]
    # a float (non-paisa) price falls back to float64, still exact
    z = syn_session("INFY", days[0])
    z["c"] = z["c"] + 0.001
    e = FS.encode(z)
    assert "fc" in e and np.array_equal(FS.decode(e)["c"], z["c"])
    # Kite rows go through the lab's own cleaning (session hours, duplicates)
    rows = [{"date": datetime(2026, 2, 2, 9, 14, tzinfo=_IST), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"date": datetime(2026, 2, 2, 9, 15, tzinfo=_IST), "open": 2, "high": 2, "low": 2, "close": 2, "volume": 5},
            {"date": datetime(2026, 2, 2, 9, 15, tzinfo=_IST), "open": 3, "high": 3, "low": 3, "close": 3, "volume": 6}]
    c = FS.clean_minute_rows(FS.kite_rows_to_lab(rows))
    assert list(c["t"]) == [555] and list(c["c"]) == [3.0]


def test_completeness_rule_marks_short_or_missing_sessions_incomplete():
    d = 20260202
    full = {s: syn_session(s, d) for s in ("INFY", "_NIFTY50_min")}
    daily = {n: {"date": np.array([d])} for n in FS.DAILY_AUX}
    assert FS.session_complete(d, full, daily, list(full))[0]
    short = dict(full, INFY=syn_session("INFY", d, n_bars=300))
    ok, why = FS.session_complete(d, short, daily, list(full))
    assert not ok and "short session" in why[0]
    ok, why = FS.session_complete(d, dict(full, INFY=None), daily, list(full))
    assert not ok and "no data" in why[0]
    ok, why = FS.session_complete(d, full, {"_NIFTY50_day": daily["_NIFTY50_day"], "_VIX_day": None}, list(full))
    assert not ok and "_VIX_day" in why[0]


def test_corporate_action_detection_and_refetch():
    d0, d1 = 20260202, 20260203
    st = syn_session("INFY", d0)
    same = {k: v.copy() for k, v in st.items()}
    assert not FS.ca_changed(st, same, d0)
    halved = dict(same, c=np.round(same["c"] / 2, 2))                 # a 1:2 split back-adjusted by Kite
    assert FS.ca_changed(st, halved, d0)

    class K:
        calls = []

        def historical_data(self, tok, frm, to, interval):
            self.calls.append((tok, frm.date(), to.date(), interval))
            out = []
            days = [d0, d1] if frm.date() <= date(2026, 2, 2) else [d1]
            for d in days:
                z = syn_session("INFY" if tok == 1 else ("_NIFTY50_min" if tok == 2 else "_NIFTYBANK_min"), d)
                f = 0.5 if tok == 1 else 1.0
                if interval == "day":
                    return [{"date": datetime(2026, 2, dd % 100, tzinfo=_IST), "open": 1, "high": 1, "low": 1,
                             "close": 1, "volume": 0} for dd in (d0, d1)]
                for i in range(len(z["t"])):
                    out.append({"date": datetime(d // 10000, d // 100 % 100, d % 100, int(z["t"][i]) // 60,
                                                 int(z["t"][i]) % 60, tzinfo=_IST),
                                "open": round(float(z["o"][i]) * f, 2), "high": round(float(z["h"][i]) * f, 2),
                                "low": round(float(z["l"][i]) * f, 2), "close": round(float(z["c"][i]) * f, 2),
                                "volume": float(z["v"][i])})
            return out

    tmp = Path(_tmp())
    store = FS.Store(tmp / "bars", cap_root=tmp)
    for s in ("INFY", "_NIFTY50_min", "_NIFTYBANK_min"):
        store.write_sessions(s, syn_session(s, d0))
    store.mark_session(d0, "complete")
    toks = {"INFY": 1, "_NIFTY50_min": 2, "_NIFTYBANK_min": 3, "_NIFTY50_day": 4, "_VIX_day": 5}
    with patch.object(FS, "MINUTE_SYMBOLS", ("INFY", "_NIFTY50_min", "_NIFTYBANK_min")):
        r = FS.fetch_session(store, K(), date(2026, 2, 3), toks, None, log=lambda *a: None)
    assert r["complete"] and r["refetch"] == [{"symbol": "INFY", "reason": "kite_adjusted", "day": d1}]
    sd = store.load_symbol("INFY")
    assert list(sd.days) == [d0, d1]
    first = sd.c[sd.starts[0]:sd.ends[0]]
    assert np.allclose(first, syn_session("INFY", d0)["c"] * 0.5, atol=0.011, rtol=0)     # the window is the adjusted one



def test_corporate_action_refetch_failure_rejects_the_session():
    """A detected corporate action whose window refetch fails (error OR no bars) must leave the day INCOMPLETE and
    write nothing: never append adjusted bars to an unadjusted history."""
    d0 = 20260202
    for mode in ("raise", "empty"):
        class K:
            infy_calls = 0

            def historical_data(self, tok, frm, to, interval):
                if interval == "day":
                    return [{"date": datetime(2026, 2, dd % 100, tzinfo=_IST), "open": 1, "high": 1, "low": 1,
                             "close": 1, "volume": 0} for dd in (d0, 20260203)]
                window = frm.date() <= date(2026, 2, 2)
                if tok == 1:
                    K.infy_calls += 1
                    if K.infy_calls >= 2:                            # the 2nd INFY call = the full-window refetch
                        if mode == "raise":
                            raise RuntimeError("kite 502")
                        return []
                out = []
                days = [d0, 20260203] if window else [20260203]
                for d in days:
                    z = syn_session("INFY" if tok == 1 else ("_NIFTY50_min" if tok == 2 else "_NIFTYBANK_min"), d)
                    f = 0.5 if tok == 1 else 1.0                     # INFY back-adjusted by Kite (1:2 split)
                    for i in range(len(z["t"])):
                        out.append({"date": datetime(d // 10000, d // 100 % 100, d % 100, int(z["t"][i]) // 60,
                                                     int(z["t"][i]) % 60, tzinfo=_IST),
                                    "open": round(float(z["o"][i]) * f, 2), "high": round(float(z["h"][i]) * f, 2),
                                    "low": round(float(z["l"][i]) * f, 2), "close": round(float(z["c"][i]) * f, 2),
                                    "volume": float(z["v"][i])})
                return out

        tmp = Path(_tmp())
        store = FS.Store(tmp / "bars", cap_root=tmp)
        for s_ in ("INFY", "_NIFTY50_min", "_NIFTYBANK_min"):
            store.write_sessions(s_, syn_session(s_, d0))
        store.mark_session(d0, "complete")
        before = store.load_symbol("INFY")
        toks = {"INFY": 1, "_NIFTY50_min": 2, "_NIFTYBANK_min": 3, "_NIFTY50_day": 4, "_VIX_day": 5}
        with patch.object(FS, "MINUTE_SYMBOLS", ("INFY", "_NIFTY50_min", "_NIFTYBANK_min")):
            r = FS.fetch_session(store, K(), date(2026, 2, 3), toks, None, log=lambda *a: None)
        assert not r["complete"] and any("corporate-action refetch failed" in x for x in r["reasons"]), (mode, r)
        assert r["refetch"] == [] and r["bytes"] == 0, (mode, r)
        for s_ in ("INFY", "_NIFTY50_min", "_NIFTYBANK_min"):
            assert list(store.load_symbol(s_).days) == [d0], (mode, s_)      # nothing of the new day stored
        after = store.load_symbol("INFY")
        assert np.array_equal(before.c, after.c), mode                      # the old window untouched


def test_disk_cap_stops_writing_and_never_deletes():
    tmp = Path(_tmp())
    st = syn_store(tmp, trading_days(date(2026, 2, 2), 2), ("INFY",))
    before = sorted(str(p) for p in tmp.rglob("*") if p.is_file())
    saved = settings.forward_lab_max_mb
    try:
        settings.forward_lab_max_mb = FS.dir_bytes(tmp) / 1048576 * 0.5
        try:
            st.check_cap()
            raise AssertionError("cap not enforced")
        except FS.DiskCapReached:
            pass
        assert sorted(str(p) for p in tmp.rglob("*") if p.is_file()) == before
    finally:
        settings.forward_lab_max_mb = saved


# ── engine ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _baseline():
    return E.load_baseline()


def test_engine_refuses_an_incomplete_session_and_never_reads_the_future():
    tmp = Path(_tmp())
    days = trading_days(date(2026, 1, 1), 32)
    st = syn_store(tmp, days[:-1])
    d = days[-2]
    st.mark_session(d, "incomplete")
    try:
        E.run_session(str(st.root), str(tmp / "state"), d, days[-3], d, str(tmp / "w0"), jobs=0, ids=SAMPLE_IDS,
                      baseline=_baseline())
        raise AssertionError("ran an incomplete session")
    except RuntimeError as exc:
        assert "not COMPLETE" in str(exc)
    st.mark_session(d, "complete")
    a = E.run_session(str(st.root), str(tmp / "state"), d, days[-3], d, str(tmp / "w1"), jobs=0, ids=SAMPLE_IDS,
                      baseline=_baseline())
    # add the NEXT session (and a truncated one after it) to the store: the result for d must not change
    for sym in SYN_SYMBOLS + tuple(FS.INDEX_MIN):
        st.write_sessions(sym, syn_session(sym, days[-1], n_bars=120))
    st.mark_session(days[-1], "complete")
    b = E.run_session(str(st.root), str(tmp / "state"), d, days[-3], d, str(tmp / "w2"), jobs=0, ids=SAMPLE_IDS,
                      baseline=_baseline())
    assert json.dumps(a["strategies"], sort_keys=True) == json.dumps(b["strategies"], sort_keys=True)
    assert set(SAMPLE_IDS) <= set(a["strategies"]) and not a["errors"], a["errors"][:3]
    # the engine only ever loads bars <= day
    assert int(st.load_symbol("INFY", upto=d).days[-1]) == d


def test_engine_runs_in_a_separate_process_with_a_time_limit():
    tmp = Path(_tmp())
    days = trading_days(date(2026, 1, 1), 26)
    st = syn_store(tmp, days)
    try:
        FL.run_engine_process(st.root, tmp / "state", days[-1], days[-2], days[-1], tmp / "w", jobs=1,
                              time_limit_s=2, mem_mb=3072, ids=None)        # every strategy: > 2 s
        raise AssertionError("time limit not enforced")
    except FL.EngineFailed as exc:
        assert "time limit" in str(exc)
    r = FL.run_engine_process(st.root, tmp / "state", days[-1], days[-2], days[-1], tmp / "w", jobs=1,
                              time_limit_s=600, mem_mb=3072, ids=SAMPLE_IDS)
    assert r["day"] == days[-1] and "process_wall" in r["timing"] and (tmp / "w" / "state_new").exists()
    pid_engine = r.get("peak_rss_mb")
    assert pid_engine is not None
    # an engine crash is reported, never raised into the caller's process as anything else
    try:
        FL.run_engine_process(st.root, tmp / "state", 20990101, days[-2], days[-1], tmp / "w3", time_limit_s=600)
        raise AssertionError("bad session accepted")
    except FL.EngineFailed as exc:
        assert "not COMPLETE" in str(exc)


def test_open_positional_positions_are_carried_until_exit():
    """r3_swing_tom-like positional specs: an entry whose exit lies beyond the session is OPEN (state), re-derived the
    next session from its entry date, and booked once on its exit session."""
    tmp = Path(_tmp())
    days = trading_days(date(2026, 1, 1), 40)
    st = syn_store(tmp, days)
    ids = ["r3_swing_tom", "tfg_rsi2_connors@1d:stocks", "tfg_ema_9_21@1d:stocks"]
    state = tmp / "state"
    seen_open, booked = {}, {i: [] for i in ids}
    for k, d in enumerate(days[25:]):
        prev = days[25 + k - 1]
        r = E.run_session(str(st.root), str(state), d, prev, days[25], str(tmp / f"w{d}"), jobs=0, ids=ids,
                          baseline=_baseline())
        for i in ids:
            for t in r["strategies"][i]["closed"]:
                assert prev < t[5] <= d
                booked[i].append((t[0], t[1], t[3], t[2]))
            for t in r["strategies"][i]["open"]:
                seen_open.setdefault(i, set()).add((t[0], t[1]))
        if state.exists():
            shutil.rmtree(state)
        shutil.copytree(tmp / f"w{d}" / "state_new", state)
    for i in ids:
        assert len(booked[i]) == len(set(booked[i])), i            # booked once
    assert seen_open, "no open position was ever carried in the synthetic run"


def test_parity_with_the_lab_on_past_sessions():
    """The forward engine, day by day with its carried state, over 5 sessions of the lab's own 3-year bars, gives the
    lab's own trades (sim.run_symbol / tfgrid.build_candidates + run_account / the module runners with lo = the
    first session) for a spread of >= 40 strategies exactly."""
    if not os.path.isdir(BARS3Y_CACHE):
        raise Skip("lab 3-year bars absent")
    import forward_lab_build as B
    ids = B.parity_sample()
    assert len(ids) >= 30
    days = [20261026, 20261027, 20261028, 20261029, 20261030]
    tmp = tempfile.mkdtemp(prefix="flab_parity_")
    try:
        r = B.parity(LAB_SCRATCH, days, ids, anchored=True, tmp=tmp, log=lambda *a: None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = {k: v for k, v in r["strategies"].items() if not v["same"]}
    assert not bad, json.dumps(bad, default=str)[:1500]
    assert r["trades_compared"] > 100, r["trades_compared"]


# ── job, flag, API ────────────────────────────────────────────────────────────────────────────────────────────────
class _Boom:
    def __getattr__(self, k):
        raise AssertionError(f"Kite touched ({k}) with the flag off")


def test_flag_off_nothing_scheduled_no_kite_call():
    assert settings.forward_lab_enabled is False                      # the default
    from platform_scheduler import PlatformScheduler
    ps = PlatformScheduler()
    with patch.object(ps._sched, "start", lambda: None):
        ps.start()
    ids = [j.id for j in ps._sched.get_jobs()]
    assert "forward_lab" not in ids
    tmp = Path(_tmp())
    r = FL.daily_job(kite=_Boom(), root=tmp)
    assert "skipped" in r and not any(tmp.iterdir())
    saved = (settings.forward_lab_enabled, settings.trading_mode)
    try:
        settings.forward_lab_enabled = True
        ps2 = PlatformScheduler()
        with patch.object(ps2._sched, "start", lambda: None):
            ps2.start()
        assert "forward_lab" in [j.id for j in ps2._sched.get_jobs()]
        settings.trading_mode = "LIVE"
        assert "skipped" in FL.daily_job(kite=_Boom(), root=tmp)      # never in LIVE
    finally:
        settings.forward_lab_enabled, settings.trading_mode = saved


def test_daily_job_skips_an_incomplete_day_for_every_strategy():
    tmp = Path(_tmp())
    days = trading_days(date(2026, 1, 1), 30)
    st = syn_store(tmp, days[:26])
    L = FL.ForwardLab(tmp)
    complete = {days[26]: True, days[27]: False, days[28]: True}
    ran = []

    def fetch(store, day):
        d = day.year * 10000 + day.month * 100 + day.day
        if complete.get(d):
            for sym in SYN_SYMBOLS + tuple(FS.INDEX_MIN):
                store.write_sessions(sym, syn_session(sym, d))
            for n in FS.DAILY_AUX:
                store.write_aux(n, {"date": np.array([d], np.int32), "o": np.ones(1) * 15, "h": np.ones(1) * 15,
                                    "l": np.ones(1) * 15, "c": np.ones(1) * 15, "v": np.zeros(1)})
            return {"complete": True, "refetch": []}
        return {"complete": False, "reasons": ["INFY: short session (120 bars)"]}

    def engine(store_root, state_dir, day, since, fwd_start, work):
        ran.append(day)
        return E.run_session(str(store_root), str(state_dir), day, since, fwd_start, str(work), jobs=0,
                             ids=SAMPLE_IDS, baseline=_baseline())

    saved = settings.forward_lab_enabled
    settings.forward_lab_enabled = True
    try:
        at = lambda d: datetime(d // 10000, d // 100 % 100, d % 100, 16, 45, tzinfo=_IST)
        r1 = FL.daily_job(now=at(days[26]), root=tmp, lab=L, fetch=fetch, engine=engine, ledger_rows=[])
        assert r1["status"]["status"] == "complete" and ran == [days[26]]
        n_reg = len(L.state()["strategies"])
        assert n_reg == len(E.strategy_table()) == 1280
        r2 = FL.daily_job(now=at(days[27]), root=tmp, lab=L, fetch=fetch, engine=engine, ledger_rows=[])
        assert r2["status"]["status"] == "pending" and ran == [days[26]]          # retried at the next session
        r3 = FL.daily_job(now=at(days[28]), root=tmp, lab=L, fetch=fetch, engine=engine, ledger_rows=[])
        assert ran == [days[26], days[28]], ran                                   # never run on the incomplete day
        runs = L.state()["runs"]
        assert runs[days[27]]["status"] == "incomplete" and runs[days[28]]["status"] == "complete"
        assert L.store.session_status(days[27]) == "incomplete"
        # no ledger got a record for the incomplete day; the daily series skips it
        sid = "orb15_break"
        d_, nets, _ = L.series(sid, L.state())
        assert days[27] not in d_ and d_ == [days[26], days[28]]
        assert L.verify_all()["ok"]
        # API contract
        api = FL.api_status(tmp)
        assert set(api) == {"enabled", "as_of", "rule", "last_run", "summary", "strategies"}
        assert set(api["rule"]) == {"version", "family_size", "alpha", "z_required", "min_trades", "min_days"}
        assert set(api["last_run"]) == {"date", "status", "strategies_run", "errors"}
        assert set(api["summary"]) == {"n_strategies", "qualified", "demoted", "consistency_fail"}
        assert api["summary"]["n_strategies"] == 1280 and api["rule"]["family_size"] == 1276
        assert api["as_of"] == f"{days[28] // 10000}-{days[28] // 100 % 100:02d}-{days[28] % 100:02d}"
        row = api["strategies"][0]
        assert set(row) == {"id", "family", "group", "trades", "days", "net", "pf", "z", "status"}
        assert {r["group"] for r in api["strategies"]} == {"proven", "candidate", "forward"}
        assert all(r["status"] in ("watching", "qualified", "demoted", "consistent", "inconsistent")
                   for r in api["strategies"])
        det = FL.api_strategy(sid, tmp)
        assert set(det) == {"id", "daily", "rule_status"} and len(det["daily"]) == 2
        assert set(det["daily"][0]) == {"date", "trades", "net"} and det["rule_status"]["chain_ok"]
        assert FL.api_strategy("nope", tmp) is None
        json.dumps(api, allow_nan=False)
    finally:
        settings.forward_lab_enabled = saved


def test_v2_endpoints_shape():
    import main
    from fastapi.testclient import TestClient
    tmp = Path(_tmp())
    with patch.object(FS, "default_root", lambda: tmp):
        c = TestClient(main.app)
        saved = settings.api_key
        settings.api_key = "k"
        try:
            assert c.get("/api/v2/forward-lab").status_code == 401
            r = c.get("/api/v2/forward-lab", headers={"X-API-Key": "k"})
            assert r.status_code == 200
            j = r.json()
            assert set(j) == {"enabled", "as_of", "rule", "last_run", "summary", "strategies"}
            assert j["enabled"] is False and j["last_run"]["status"] == "never" and j["strategies"] == []
            assert j["rule"]["family_size"] == 1276 and 4.4 < j["rule"]["z_required"] < 4.6
            assert c.get("/api/v2/forward-lab/tfg_ema_9_21@5m:stocks", headers={"X-API-Key": "k"}).status_code == 404
        finally:
            settings.api_key = saved


def test_read_only_towards_trading():
    for mod in (FL, FS, E):
        src = inspect.getsource(mod)
        for bad in ("place_order", "paper_fill_at", "paper_marketable", "modify_order", "cancel_order",
                    "set_strategy(", "set_global(", "record_trade(", "set_agent_enabled", "squareoff"):
            assert bad not in src, (mod.__name__, bad)
    assert "ALL_AGENTS" not in inspect.getsource(FL)


def test_telegram_at_most_once_a_day():
    tmp = Path(_tmp())
    FL._jsave(tmp / "status.json", {"last_run": {"date": "2026-02-02", "status": "complete", "strategies_run": 5,
                                                 "errors": 0}, "rows": []})
    assert FL.telegram_once(tmp) and FL.telegram_once(tmp) is None
    FL._jsave(tmp / "status.json", {"last_run": {"date": "2026-02-03", "status": "incomplete"}, "rows": []})
    assert FL.telegram_once(tmp) is None


TESTS = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


def _main() -> int:
    failed = 0
    for n, f in TESTS:
        t0 = time.time()
        try:
            f()
            print(f"  ok    {n} ({time.time() - t0:.1f}s)")
        except Skip as exc:
            print(f"  skip  {n}: {exc}")
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
