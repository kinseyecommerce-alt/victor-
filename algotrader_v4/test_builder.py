"""
test_builder.py — tests of the automatic strategy builder (builder.py, builder_run.py, builder_forward.py,
stratlab/builder_*.py).

    cd algotrader_v4 && python test_builder.py      (also run from test_pipeline.py)

Everything runs in temporary directories on a small SYNTHETIC data set written in the builder data layer's layout
(stratlab.builder_server_panels.make_synthetic_tree): no lab scratch, no network, no real return is looked at.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PERFORMANCE_FILE", os.path.join(tempfile.mkdtemp(), "performance.json"))
_TMP = tempfile.mkdtemp(prefix="builder_test_")
import atexit  # noqa: E402
atexit.register(shutil.rmtree, _TMP, True)
os.environ["BUILDER_CACHE_DIR"] = os.path.join(_TMP, "cache")

import numpy as np  # noqa: E402

from config import settings  # noqa: E402
import builder as B  # noqa: E402
import builder_forward as BF  # noqa: E402
import forward_lab as FL  # noqa: E402
from stratlab import builder_engine as BE  # noqa: E402
from stratlab import builder_eval as EV  # noqa: E402
from stratlab import builder_grammar as G  # noqa: E402
from stratlab import builder_options as BO  # noqa: E402
from stratlab import builder_server_panels as SP  # noqa: E402
from stratlab import trials as TR  # noqa: E402

HERE = Path(__file__).resolve().parent
_SYN: dict = {}
SEED_N = 2552 + 16 + 400 + 2          # N after the sandbox batch 0001 the server registry is seeded with


def syn_root() -> str:
    """One synthetic data tree for the whole module (4 stocks, 2021-01 .. 2026-11)."""
    if "root" not in _SYN:
        root = os.path.join(_TMP, "data")
        SP.make_synthetic_tree(root, start=20210104, end=20261109, n_stocks=4, seed=5)
        _SYN["root"] = root
    return _SYN["root"]


def syn_source():
    return SP.ServerSource(syn_root())


def engine():
    if "E" not in _SYN:
        E = BE.Engine(syn_source())
        E.cache_dir = os.path.join(_TMP, "cache")
        _SYN["E"] = E
    return _SYN["E"]


def new_builder(name: str) -> B.Builder:
    root = os.path.join(_TMP, name)
    shutil.rmtree(root, ignore_errors=True)
    return B.Builder(root)


# ══════════════════════════════════════════════════════════════════════════
# grammar
# ══════════════════════════════════════════════════════════════════════════
def test_grammar_size_and_batch_shape():
    sizes = G.grammar_sizes()
    assert set(sizes) == set(G.AGENTS) and all(v >= 300 for v in sizes.values()), sizes
    b = G.draw_batch("0001", set())
    assert len(b) == G.BATCH_SIZE == 400
    cells = {}
    for c in b:
        cells[c.cell] = cells.get(c.cell, 0) + 1
    assert len(cells) == 20 and set(cells.values()) == {20}
    assert len({c.key for c in b}) == 400


def test_grammar_determinism():
    a = G.draw_batch("0007", set())
    b = G.draw_batch("0007", set())
    assert [c.key for c in a] == [c.key for c in b]
    c = G.draw_batch("0008", set())
    assert [x.key for x in a] != [x.key for x in c]
    assert G.grammar_sha256() == json.loads((HERE / "builder_rule.json").read_text())["grammar"]["sha256"]


def test_no_retest_across_batches():
    drawn = set()
    seen = []
    for bid in ("0001", "0002", "0003"):
        b = G.draw_batch(bid, drawn)
        keys = {c.key for c in b}
        assert not (keys & drawn), "a key of an earlier batch was drawn again"
        drawn |= keys
        seen.append(len(b))
    assert seen == [400, 400, 400]


def test_near_duplicate_rule():
    base = G.Candidate("swing", "RISING", "deliv_up", (("pp", 15.0), ("day", "up")), "next_open",
                       ("bracket_d", 0.03, 2.0, 20), "STOCKS", "cnc/fut")
    one = G.Candidate("swing", "RISING", "deliv_up", (("pp", 25.0), ("day", "up")), "next_open",
                      ("bracket_d", 0.03, 2.0, 20), "STOCKS", "cnc/fut")
    two = G.Candidate("swing", "RISING", "deliv_up", (("pp", 25.0), ("day", "up")), "next_open",
                      ("bracket_d", 0.05, 2.0, 20), "STOCKS", "cnc/fut")
    other_mt = G.Candidate("swing", "FALLING", "deliv_up", (("pp", 15.0), ("day", "up")), "next_open",
                           ("bracket_d", 0.03, 2.0, 20), "STOCKS", "cnc/fut")
    other_exit = G.Candidate("swing", "RISING", "deliv_up", (("pp", 15.0), ("day", "up")), "next_open",
                             ("hold", 5), "STOCKS", "cnc/fut")
    assert G.near_duplicate(base, one)                   # one field differs
    assert not G.near_duplicate(base, two)               # two fields differ
    assert not G.near_duplicate(base, other_mt)          # other market type
    assert not G.near_duplicate(base, other_exit)        # other exit kind
    b = G.draw_batch("0001", set())
    for i, x in enumerate(b):
        for y in b[i + 1:]:
            assert not G.near_duplicate(x, y), (x, y)


def test_plain_english_from_grammar():
    for c in G.draw_batch("0001", set())[::37]:
        t = G.describe(c)
        assert G.MT_TEXT[c.mtype] in t and len(t) > 60 and "{" not in t, t


# ══════════════════════════════════════════════════════════════════════════
# rule, splits, market types
# ══════════════════════════════════════════════════════════════════════════
def test_rule_frozen_and_pinned():
    r = B.load_rule()
    assert r["_sha256"] == B.RULE_SHA256 == hashlib.sha256((HERE / "builder_rule.json").read_bytes()).hexdigest()
    assert r["carry"]["K_per_cell"] == 3 and r["dsr"]["min"] == 0.95 and r["dsr"]["lab_trials_count"] == 2552
    assert r["splits"]["validation_months_before_end"] == 24 and r["splits"]["final_months_before_end"] == 12
    assert "FINAL look" in r["final"]["looks"] or "FINAL look ever" in r["final"]["looks"]
    tmp = Path(_TMP) / "rule_edit.json"
    tmp.write_text((HERE / "builder_rule.json").read_text().replace('"K_per_cell": 3', '"K_per_cell": 4'))
    try:
        B.load_rule(tmp)
        raise AssertionError("an edited rule must be refused")
    except B.RuleChanged:
        pass
    fr = BF.load_rule()
    assert fr["_sha256"] == BF.RULE_SHA256


def test_split_arithmetic():
    assert BE.months_back(20261109, 24) == 20241109
    assert BE.months_back(20261109, 12) == 20251109
    assert BE.months_back(20240229, 12) == 20230228
    assert BE.months_back(20250131, 1) == 20241231
    w = BE.windows(20261109, B.load_rule())
    assert w["discovery"] == (0, 20241109) and w["validation"] == (20241110, 20251109)
    assert w["final"] == (20251110, 20261109)
    h1, h2 = BE.halves_of([1, 2, 3, 4, 5, 6, 7], 2, 7)
    assert h1 == (2, 4) and h2 == (5, 7)


def test_market_types_definitions():
    T = 400
    up = 100 * np.exp(np.linspace(0, 0.5, T))
    vix = np.full(T, 15.0)
    m = BE.market_types(up, vix)
    assert (m[-100:] == 0).all()                                   # RISING
    m = BE.market_types(up[::-1].copy(), vix)
    assert (m[-100:] == 1).all()                                   # FALLING
    v2 = vix.copy()
    v2[-1] = 30.0
    m = BE.market_types(up, v2)
    assert m[-1] == 3 and m[-2] == 0                               # HIGH_VOL overrides on that day only
    assert (BE.market_types(up, vix)[:59] == -1).all()             # no SMA / slope yet


def test_market_types_point_in_time():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900)))
    v = 15 * np.exp(np.cumsum(rng.normal(0, 0.05, 900)))
    full = BE.market_types(c, v)
    for t in rng.integers(60, 900, 40).tolist():
        cut = BE.market_types(c[:t + 1], v[:t + 1])
        assert cut[t] == full[t], t                                # nothing after t changes t's type
    E = engine()
    assert (E.regime_m[1:] == np.where(E.prev_d[1:] >= 0, E.regime_d[np.maximum(E.prev_d[1:], 0)], -1)).all()
    assert (E.dcal[E.prev_d[E.prev_d >= 0]] < E.mcal[E.prev_d >= 0]).all()    # minute uses the PREVIOUS session


# ══════════════════════════════════════════════════════════════════════════
# engine pieces
# ══════════════════════════════════════════════════════════════════════════
def test_walk_session_rules():
    tt = np.arange(555, 555 + 60)
    o = np.full(60, 100.0)
    h = np.full(60, 100.2)
    l_ = np.full(60, 99.8)
    c = np.full(60, 100.0)
    v = np.full(60, 1000.0)
    r = BE.walk_session(tt, o, h, l_, c, v, 1, (560, "time_min", 10), 20250102)
    assert r[0] == 100.0 and r[2] == "09:30" and r[3] == "time"
    h2, l2 = h.copy(), l_.copy()
    h2[8], l2[8] = 101.2, 99.4                                     # one bar touches both: the stop wins
    r = BE.walk_session(tt, o, h2, l2, c, v, 1, (560, "bracket", 0.005, 2.0), 20250102)
    assert r[3] == "stop" and abs(r[1] - 99.5) < 1e-9
    h3 = h.copy()
    h3[9] = 101.0                                                  # bare touch of the target: no fill
    r = BE.walk_session(tt, o, h3, l_, c, v, 1, (560, "bracket_min", 0.005, 2.0, 30), 20250102)
    assert r[3] == "time"
    h3[9] = 101.05
    r = BE.walk_session(tt, o, h3, l_, c, v, 1, (560, "bracket_min", 0.005, 2.0, 30), 20250102)
    assert r[3] == "target" and abs(r[1] - 101.0) < 1e-9
    c4 = c.copy()
    c4[10] = 99.0                                                  # close below VWAP: the long exits next open
    r = BE.walk_session(tt, o, h, l_, c4, v, 1, (560, "signal", "vwap"), 20250102)
    assert r[3] == "signal" and r[2] == "09:26"


def test_signals_and_jobs_synthetic():
    E = engine()
    b = G.draw_batch("syn", set(), per_cell=1)
    jobs = [BE.Job(f"j{k}", c, None, 0, 20241109) for k, c in enumerate(b)]
    res = BE.run_jobs(E, jobs)
    n = sum(len(res[j.key]["trades"]) for j in jobs)
    assert n > 50, n
    for j in jobs:
        for t in res[j.key]["trades"]:
            assert t[2] <= 20241109 and len(t) == 14
            sig = E.signals(j.cand)
    # market-type filter: every trigger is on a session of the candidate's type
    for c in b:
        sig = E.signals(c)
        reg = E.regime_m if c.grid in ("minute", "idx_minute", "prev_eod") else E.regime_d
        assert (reg[sig["t"]] == BE.MT_CODE[c.mtype]).all()


def test_options_zero_settle_is_missing():
    root = syn_root()
    p = os.path.join(root, "nse", "fo_parts", "optidx")
    f = sorted(os.listdir(p))[3]
    tabs = BO.build_tables([(os.path.join(root, "nse"), os.path.join(root, "nse", "bars"), 0, 20210131)])
    t = tabs["chains"]["NIFTY"]
    assert len(t["date"]) > 0 and np.isfinite(t["close"]).all()
    # a zero settle on an expiry day becomes NaN (nsedata then settles at the index close)
    tmp = os.path.join(_TMP, "nse_zero")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(os.path.join(root, "nse"), tmp, ignore=shutil.ignore_patterns("delivery.csv.gz"))
    exp_day = None
    for name in sorted(os.listdir(os.path.join(tmp, "fo_parts", "optidx"))):
        with gzip.open(os.path.join(tmp, "fo_parts", "optidx", name), "rt") as fh:
            lines = fh.read().splitlines()
        rows = [ln.split(",") for ln in lines[1:]]
        if any(r[0] == r[2] for r in rows):
            exp_day = int(name[:8])
            out = [lines[0]] + [",".join(r[:9] + ["0"] + r[10:]) if r[0] == r[2] else ",".join(r) for r in rows]
            with gzip.open(os.path.join(tmp, "fo_parts", "optidx", name), "wt") as fh:
                fh.write("\n".join(out) + "\n")
            break
    tabs = BO.build_tables([(tmp, os.path.join(tmp, "bars"), 0, exp_day)])
    t = tabs["chains"]["NIFTY"]
    m = (t["date"] == exp_day) & (t["expiry"] == exp_day)
    assert m.any() and np.isnan(t["settle"][m]).all()
    del f


# ══════════════════════════════════════════════════════════════════════════
# freeze before evaluate, the batch, N, promotion
# ══════════════════════════════════════════════════════════════════════════
def test_freeze_before_evaluate():
    b = new_builder("freeze")
    try:
        b.verify_frozen("0001")
        raise AssertionError("an unfrozen batch must be refused")
    except B.NotFrozen:
        pass
    snap = {"source": "test", "end": 20261109, "inputs": {"x": 1}}
    rec = b.freeze(snap, per_cell=1)
    assert rec["batch"] == "0002" and rec["n_candidates"] == 20
    kinds = [r["kind"] for r in b.registry.records()]
    assert kinds == ["genesis", "seed", "batch_frozen"]
    b.verify_frozen("0002", snap)
    try:
        b.verify_frozen("0002", {**snap, "end": 20261110})
        raise AssertionError("a changed data snapshot must be refused")
    except B.NotFrozen:
        pass
    p = b.candidates_path("0002")
    doc = json.loads(p.read_text())
    doc["candidates"][0]["text"] += " "
    p.write_text(json.dumps(doc))
    try:
        b.verify_frozen("0002", snap)
        raise AssertionError("an edited candidate file must be refused")
    except B.NotFrozen:
        pass
    assert b.freeze(snap, "0002")["frozen_hash"] == rec["frozen_hash"]      # idempotent per id
    rec2 = b.freeze(snap, per_cell=1)
    assert rec2["batch"] == "0003"


def test_evaluation_process_refuses_unfrozen():
    root = os.path.join(_TMP, "refuse")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    r = subprocess.run([sys.executable, "-m", "builder_run", "--root", root, "--batch", "0001", "--source",
                        "synthetic", "--data-root", syn_root(), "--mem-mb", "0"], cwd=str(HERE),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 1
    err = open(os.path.join(root, "batches", "0001", "work", "error.txt")).read()
    assert "NotFrozen" in err, err[:300]


def _run_synthetic_batch(name: str, per_cell: int = 1):
    import builder_run
    b = new_builder(name)
    src = syn_source()
    rec = b.freeze(src.snapshot(), per_cell=per_cell)
    s = builder_run.evaluate(str(b.root), rec["batch"], "synthetic", syn_root(), jobs=1, log=lambda m: None)
    return b, rec, s


def test_batch_end_to_end_and_n_accounting():
    b, rec, s = _run_synthetic_batch("e2e")
    for k in ("candidates", "discovery_passers", "shuffle_passers", "validation_tested", "validation_passers",
              "final_confirmed", "proven", "n_disc", "n_proof", "cells", "rows", "planted", "windows"):
        assert k in s, k
    lab = SEED_N
    assert rec["n_before"] == lab
    assert s["n_disc"] == lab + 20 and s["n_proof"] == s["n_disc"] + s["final_looks"]
    for a in G.AGENTS:
        assert f"gate_{a}" in s["planted"]
    out = b.ingest(rec["batch"], s)
    st = b.state()
    assert st["evaluated"] == 420 and st["n_used"] == s["n_proof"]
    assert b.n_base(st) == 2552 + 16 + 420 + st["final_looks"]
    # resumable: a re-run returns the stored units (no recompute)
    work = b.batch_dir(rec["batch"]) / "work"
    m0 = {p.name: p.stat().st_mtime for p in work.glob("*.json")}
    import builder_run
    builder_run.evaluate(str(b.root), rec["batch"], "synthetic", syn_root(), jobs=1, log=lambda m: None)
    assert {p.name: p.stat().st_mtime for p in work.glob("*.json")} == m0
    del out


def test_n_never_shrinks_and_counts_final_looks():
    b = new_builder("ncount")
    snap = {"source": "test", "end": 20261109}
    r1 = b.freeze(snap, per_cell=1)
    b.open_final(r1["batch"], ["a", "b"])
    b.open_final(r1["batch"], ["a"])                           # the same look is not counted twice
    summ = {"candidates": 20, "discovery_passers": 0, "shuffle_passers": 0, "validation_tested": 2,
            "validation_passers": 2, "final_looks": 2, "final_confirmed": 0, "proven": [], "n_disc": 5000,
            "n_proof": 5002, "v": 0.001, "best_discovery_dsr": {}, "planted": {}, "rows": [], "windows": {}}
    b.ingest(r1["batch"], summ)
    st = b.state()
    assert st["final_looks"] == 4 and st["n_used"] == 5002          # the seed's 2 FINAL looks + this batch's 2
    r2 = b.freeze(snap, per_cell=1)
    assert r2["n_before"] == SEED_N + 20 + 2
    b.ingest(r2["batch"], {**summ, "n_proof": 10})               # a smaller N can never lower N_used
    assert b.state()["n_used"] == 5002
    assert B.api_status(b.root)["n_used"] >= 5002


def test_dsr_wiring():
    st = {"n_days": 500, "sr_daily": 0.12, "skew_daily": -0.3, "kurt_daily": 5.0}
    a = BE.dsr_of(st, 2970, 0.0012)
    b = TR.dsr(0.12, 500, -0.3, 5.0, 2970, 0.0012)
    assert abs(a["dsr"] - b["dsr"]) < 1e-12
    assert BE.dsr_of(st, 30000, 0.0012)["dsr"] < a["dsr"]       # more trials -> harder
    # carry ranking uses N_disc = n_base + the batch's candidates
    E = engine()
    cands = [(f"c{k}", c) for k, c in enumerate(G.draw_batch("syn", set(), per_cell=1))]
    bt = EV.Batch(E, "t", cands, B.load_rule(), os.path.join(_TMP, "dsrwire"), 1000)
    disc = {cid: {"series": {"n_days": 400, "sr_daily": 0.01 * k, "skew_daily": 0.0, "kurt_daily": 3.0},
                  "judge": {"passed": k % 2 == 0}} for k, (cid, _c) in enumerate(cands)}
    sel = bt.select(disc)
    assert sel["n_disc"] == 1000 + len(cands)
    for cid in sel["carried"]:
        assert disc[cid]["judge"]["passed"]
    per = {}
    for cid in sel["carried"]:
        c = dict(cands)[cid]
        per[c.cell] = per.get(c.cell, 0) + 1
    assert max(per.values() or [0]) <= 3


def _fake_proven_summary(b, bid):
    doc = json.loads(b.candidates_path(bid).read_text())
    rows = []
    for k, r in enumerate(doc["candidates"]):
        c = r["cand"]
        rows.append({"id": r["id"], "agent": c["agent"], "mtype": c["mtype"], "block": c["block"], "key": r["key"],
                     "text": r["text"], "stage": "proven" if k == 0 else "discovery",
                     "discovery": {"trades": 10, "pf": 1, "net": 0, "h1": 0, "h2": 0, "why": [], "dsr": 0.97,
                                   "sr_daily": 0.1, "passed": k == 0, "shuffle_passed": False},
                     "validation": None, "final": None,
                     "proof": {"dsr": {"dsr": 0.97}, "gates": {}, "proven": True,
                               "reference_daily": [100.0, -50.0, 0.0, 80.0] * 30} if k == 0 else None})
    return {"candidates": len(rows), "discovery_passers": 1, "shuffle_passers": 0, "validation_tested": 1,
            "validation_passers": 1, "final_looks": 1, "final_confirmed": 1, "proven": [rows[0]["id"]],
            "n_disc": 3000, "n_proof": 3001, "v": 0.001, "best_discovery_dsr": {}, "planted": {}, "rows": rows,
            "windows": {"T": 20261109, "discovery": [0, 20241109], "validation": [20241110, 20251109],
                        "final": [20251110, 20261109]}}


def test_promotion_only_for_proven_and_forward_family():
    with patch.object(settings, "builder_enabled", True):
        b = new_builder("promo")
        rec = b.freeze({"source": "test", "end": 20261109}, per_cell=1)
        s = _fake_proven_summary(b, rec["batch"])
        out = b.ingest(rec["batch"], s)
    pid = s["proven"][0]
    gen = (b.root / "catalog" / "builder_gen.py").read_text()
    assert pid in gen and gen.count('"id":') == 1 and "GENERATED" in gen
    assert out["promoted"][0]["id"] == pid
    try:
        b.promote(rec["batch"], s["rows"][1]["id"], s)
        raise AssertionError("a non-proven candidate must never be promoted")
    except ValueError:
        pass
    fw = BF.BuilderForward(b.root / "forward")
    st = fw.state()
    assert list(st["strategies"]) == [pid] and fw.family_size(st) == 1
    assert st["strategies"][pid]["registered_on"] == 20261109
    ns = {}
    exec(compile(gen, "builder_gen.py", "exec"), ns)
    assert ns["SPECS"][0]["id"] == pid and ns["RULE_SHA256"] == B.RULE_SHA256


def test_forward_family_rule_and_v1_untouched():
    assert FL.RULE_SHA256 == hashlib.sha256((HERE / "forward_lab_rule.json").read_bytes()).hexdigest()
    assert FL.load_rule()["groups"]["forward"]["initial_family_size"] == 1276
    root = Path(_TMP) / "fwd"
    shutil.rmtree(root, ignore_errors=True)
    fw = BF.BuilderForward(root)
    cand = G.draw_batch("syn", set(), per_cell=1)[8].to_json()
    ref = {"daily": [500.0, -200.0, 0.0, 300.0] * 40, "max_dd": 400.0}
    fw.register([{"id": "s1", "batch": "0001", "cand": cand, "reference": ref, "fingerprint": "x"}], 20260101)
    fw.register([{"id": "s1", "batch": "0001", "cand": cand, "reference": ref, "fingerprint": "x"}], 20260101)
    fw.register([{"id": "s2", "batch": "0002", "cand": cand, "reference": ref, "fingerprint": "y"}], 20260101)
    assert fw.family_size() == 2
    assert fw.z_required(2) > fw.z_required(1)
    days = [20260102 + k for k in range(0, 28)] + [20260201 + k for k in range(0, 27)]
    rng = np.random.default_rng(1)
    for d in days:
        n = 3
        trades = {"s1": [("s1", "SYN00", 20260102, 1, "09:15", 100.0, d, "15:29", 101.0, 10, 0.0, 0.0,
                          float(rng.normal(400, 50)), "time") for _ in range(n)],
                  "s2": [("s2", "SYN01", 20260102, 1, "09:15", 100.0, d, "15:29", 101.0, 10, 0.0, 0.0,
                          float(rng.normal(-400, 50)), "window_end")]}
        fw.ingest_day(d, trades)
    ev = fw.evaluate_all()
    assert ev["s1"]["status"] == "qualified", ev["s1"]
    sb = fw.status_by_id()
    assert sb["s1"]["ready"] is True and sb["s2"]["trades"] == 0      # window_end trades are never booked
    try:
        fw.ingest_day(days[0], {})
        raise AssertionError("a session is ingested once")
    except RuntimeError:
        pass
    assert fw.registry.verify()["ok"]


def test_run_pending_with_runner():
    root = Path(_TMP) / "fwd_pending"
    shutil.rmtree(root, ignore_errors=True)
    fw = BF.BuilderForward(root)
    cand = G.draw_batch("syn", set(), per_cell=1)[8].to_json()
    fw.register([{"id": "s1", "batch": "0001", "cand": cand, "reference": {}, "fingerprint": "x"}], 20260105)

    def runner(r, data_root, day):
        return {"day": day, "calendar": [20260102, 20260105, 20260106, 20260107],
                "trades": {"s1": [["s1", "A", 20260106, 1, "09:15", 1.0, 20260107, "15:29", 1.0, 1, 0, 0, 5.0, "time"]]}}
    out = BF.run_pending(root, runner=runner, today=20260107)
    assert out["processed"] == [20260106, 20260107]
    st = fw.state()
    assert sorted(st["runs"]) == [20260106, 20260107]
    out = BF.run_pending(root, runner=runner, today=20260107)
    assert out["processed"] == []


def test_flag_off_nothing_scheduled():
    with patch.object(settings, "builder_enabled", False):
        assert B.enabled() is False
        assert B.weekly_job(root=os.path.join(_TMP, "off"))["skipped"].startswith("BUILDER_ENABLED")
        assert B.weekly_job_locked()["skipped"].startswith("BUILDER_ENABLED")
        assert not os.path.exists(os.path.join(_TMP, "off"))
        assert "skipped" in BF.run_pending(os.path.join(_TMP, "off2"))
        assert "skipped" in FL.builder_family_step()
        from unittest.mock import MagicMock
        import platform_scheduler as PS
        for on in (False, True):
            with patch.object(settings, "builder_enabled", on):
                ps = PS.PlatformScheduler()
                ps._sched = MagicMock()
                ps.start()
                ids = [c.kwargs.get("id") for c in ps._sched.add_job.call_args_list]
                assert ("builder_weekly" in ids) is on, ids
                if on:
                    c = [c for c in ps._sched.add_job.call_args_list if c.kwargs.get("id") == "builder_weekly"][0]
                    assert c.kwargs["day_of_week"] == "sat" and c.kwargs["hour"] == 6 and c.kwargs["minute"] == 0
    with patch.object(settings, "builder_enabled", True), patch.object(settings, "trading_mode", "LIVE"):
        assert B.enabled() is False


def test_weekly_job_end_to_end_with_runner():
    from datetime import datetime
    root = os.path.join(_TMP, "weekly")
    shutil.rmtree(root, ignore_errors=True)
    calls = []

    def runner(r, bid):
        calls.append(bid)
        b = B.Builder(r)
        doc = b.verify_frozen(bid)
        return {"candidates": len(doc["candidates"]), "discovery_passers": 0, "shuffle_passers": 0,
                "validation_tested": 0, "validation_passers": 0, "final_looks": 0, "final_confirmed": 0,
                "proven": [], "n_disc": 1, "n_proof": 1, "v": 0.0, "best_discovery_dsr": {}, "planted": {},
                "rows": [], "windows": {}}
    sat = datetime(2026, 11, 21, 6, 0)
    with patch.object(settings, "builder_enabled", True):
        s = B.weekly_job(now=sat, root=root, runner=runner, snapshot_fn=lambda: {"source": "test", "end": 1})
        assert s["status"] == "complete" and s["batch"] == "0002", s
        s = B.weekly_job(now=sat, root=root, runner=runner, snapshot_fn=lambda: {"source": "test", "end": 1})
        assert s["batch"] == "0003"
        st = B.Builder(root).state()
        assert st["evaluated"] == 1200 and calls == ["0002", "0003"]


def test_disk_cap():
    root = os.path.join(_TMP, "cap")
    shutil.rmtree(root, ignore_errors=True)
    with patch.object(settings, "builder_max_mb", 0.0001):
        b = B.Builder(root)
        b.ensure_genesis()
        try:
            b.freeze({"source": "test", "end": 1}, per_cell=1)
            raise AssertionError("the disk cap must stop the freeze")
        except B.DiskCapReached:
            pass
        assert list(b.state()["batches"]) == ["0001"] and b.state()["batches"]["0001"]["seeded"]   # only the seed
        with patch.object(settings, "builder_enabled", True):
            from datetime import datetime
            s = B.weekly_job(now=datetime(2026, 11, 21, 6, 0), root=root, runner=lambda r, b_: {},
                             snapshot_fn=lambda: {"source": "test", "end": 1})
        assert s["status"] == "disk_cap", s


def test_subprocess_limits():
    # the address-space limit of the evaluation process
    code = ("import builder_run, sys\nbuilder_run._limit(300)\ntry:\n    b = bytearray(600 * 1024 * 1024)\n"
            "    print('allocated')\nexcept MemoryError:\n    print('memory-limited')\nimport os\nprint(os.nice(0))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(HERE), capture_output=True, text=True, timeout=60)
    assert "memory-limited" in r.stdout and r.stdout.strip().endswith("10"), r.stdout + r.stderr
    # the wall-clock limit kills the process group and keeps the batch resumable
    b = new_builder("timelimit")
    rec = b.freeze(syn_source().snapshot(), per_cell=1)
    t0 = time.time()
    try:
        B.run_process(b.root, rec["batch"], "synthetic", jobs=1, time_limit_s=2, mem_mb=0, data_root=syn_root())
        raise AssertionError("the time limit must stop the run")
    except B.RunFailed as exc:
        assert "time limit" in str(exc)
    assert time.time() - t0 < 30
    assert rec["batch"] not in b.state()["results"]


def test_snapshot_in_its_own_process():
    root = os.path.join(_TMP, "snapproc")
    shutil.rmtree(root, ignore_errors=True)
    snap = B.snapshot_process(Path(root), "synthetic", data_root=syn_root(), mem_mb=0)
    assert snap["end"] == syn_source().end and snap["inputs"]["delivery"]["sha256"]
    from stratlab.builder_source import snapshot_sha
    assert snapshot_sha(snap) == snapshot_sha(syn_source().snapshot())


def test_api_shape():
    empty = B.api_status(os.path.join(_TMP, "api_empty"))
    for k in ("enabled", "rule_version", "rule_sha256", "agents", "market_types", "grid", "batches", "n_used",
              "lab_trials_count", "grammar_sizes"):
        assert k in empty, k
    assert len(empty["grid"]) == 20 and empty["batches"] == [] and empty["n_used"] == SEED_N
    b = new_builder("api")
    rec = b.freeze({"source": "test", "end": 20261109}, per_cell=1)
    s = _fake_proven_summary(b, rec["batch"])
    with patch.object(settings, "builder_enabled", True):
        b.ingest(rec["batch"], s)
    st = B.api_status(b.root)
    row = st["batches"][0]
    for k in ("id", "created", "hash", "status", "candidates", "discovery_passers", "shuffle_expected",
              "validation_passers", "final_confirmed", "proven"):
        assert k in row, k
    cell = [g for g in st["grid"] if g["proven"]]
    assert len(cell) == 1 and cell[0]["in_forward_lab"] == 1
    d = B.api_batch(rec["batch"], b.root)
    assert len(d["candidates"]) == 20 and d["candidates"][0]["text"] and d["candidates"][0]["stage"] == "proven"
    assert B.api_batch("9999", b.root) is None
    # the HTTP routes
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api_v2
    app = FastAPI()
    app.include_router(api_v2.router, prefix="/api")
    with patch.dict(os.environ, {"BUILDER_DIR": str(b.root)}):
        c = TestClient(app)
        r = c.get("/api/v2/builder")
        assert r.status_code == 200 and r.json()["batches"][0]["id"] == rec["batch"]
        r = c.get(f"/api/v2/builder/batches/{rec['batch']}")
        assert r.status_code == 200 and len(r.json()["candidates"]) == 20
        assert c.get("/api/v2/builder/batches/9999").status_code == 404


def test_registry_two_writers():
    root = Path(_TMP) / "writers"
    shutil.rmtree(root, ignore_errors=True)
    a = B.Builder(root)
    a.ensure_genesis()
    other = B.Builder(root)
    other.registry.records()
    a.registry.append("note", {"x": 1})            # another process appends ...
    other.registry.append("note", {"x": 2})        # ... a stale writer must not fork the chain
    seqs = [r["seq"] for r in a.registry.records()]
    assert a.registry.verify()["ok"] and seqs == list(range(len(seqs))) and seqs[-2:] == [len(seqs) - 2, len(seqs) - 1]
    assert [r["kind"] for r in a.registry.records()][-2:] == ["note", "note"]


# ══════════════════════════════════════════════════════════════════════════
# the sandbox seed (batch 0001 of the lab sandbox counts on the server)
# ══════════════════════════════════════════════════════════════════════════
def test_seed_file_pinned_and_consistent():
    sd = B.load_seed()
    assert sd["_sha256"] == B.SEED_SHA256 == hashlib.sha256(B.SEED_FILE.read_bytes()).hexdigest()
    assert sd["batch"] == "0001" and sd["n_candidates"] == 400 == len(sd["candidates"])
    assert sd["frozen_hash"].startswith("f506bc18") and sd["rule_sha256"] == B.RULE_SHA256
    assert sd["grammar_sha256"] == G.grammar_sha256()
    assert sd["n_used_after"] == SEED_N == 2970
    assert len(sd["final_looks"]) == 2 and not any(f["confirmed"] for f in sd["final_looks"])
    assert sd["development_evaluations"]["count"] == B.load_rule()["dsr"]["development_evaluations"] == 16
    # the 400 keys are exactly batch 0001's draw (deterministic grammar)
    assert [c["key"] for c in sd["candidates"]] == [c.key for c in G.draw_batch("0001", set())]


def test_seed_import_idempotent():
    b = new_builder("seed_idem")
    b.ensure_genesis()
    b.ensure_genesis()
    b.ensure_seed()
    kinds = [r["kind"] for r in b.registry.records()]
    assert kinds == ["genesis", "seed"], kinds
    assert b.registry.verify()["ok"]
    st = b.state()
    assert st["batches"]["0001"]["seeded"] and st["evaluated"] == 400 and st["final_looks"] == 2
    assert st["n_used"] == SEED_N and b.n_base(st) == SEED_N
    # a fresh object on the same registry imports nothing again
    B.Builder(b.root).ensure_genesis()
    assert [r["kind"] for r in b.registry.records()] == ["genesis", "seed"]
    st_api = B.api_status(b.root)
    assert st_api["n_used"] == SEED_N and st_api["batches"][0]["status"] == "seeded"
    assert st_api["seed"]["imported"] is True
    assert B.api_batch("0001", b.root)["candidates"] == []          # the seed carries keys, not the sandbox's rows


def test_seed_tampered_refused():
    tmp = Path(_TMP) / "seed_tampered.json"
    txt = B.SEED_FILE.read_text().replace('"n_used_after": 2970', '"n_used_after": 2000')
    tmp.write_text(txt)
    try:
        B.load_seed(tmp)
        raise AssertionError("an edited seed must be refused")
    except B.SeedChanged:
        pass
    b = new_builder("seed_tampered")
    with patch.object(B, "SEED_FILE", tmp):
        try:
            b.ensure_genesis()
            raise AssertionError("a registry must not start on an edited seed")
        except B.SeedChanged:
            pass
        assert [r["kind"] for r in b.registry.records()] == ["genesis"]       # no seed record written
        with patch.object(settings, "builder_enabled", True):
            from datetime import datetime
            s = B.weekly_job(now=datetime(2026, 11, 21, 6, 0), root=b.root, runner=lambda r, x: {},
                             snapshot_fn=lambda: {"source": "test", "end": 1})
        assert s["status"] == "seed_changed", s
        assert not B.Builder(b.root).state()["batches"]


def test_seed_counts_in_n_and_keys_never_redrawn():
    b = new_builder("seed_n")
    rec = b.freeze({"source": "test", "end": 20261109})
    assert rec["batch"] == "0002" and rec["n_before"] == SEED_N
    seeded = {c["key"] for c in B.load_seed()["candidates"]}
    assert seeded <= b.drawn_keys()
    doc = json.loads(b.candidates_path("0002").read_text())
    assert not ({c["key"] for c in doc["candidates"]} & seeded)
    assert len(doc["candidates"]) == 400
    assert B.Builder(b.root).next_batch_id() == "0003"


def test_seed_skipped_in_its_source_registry():
    b = new_builder("seed_source")
    st0 = b.state()
    sd = B.load_seed()
    # a registry whose batch 0001 IS the seed's batch (the sandbox) imports nothing
    b.registry.append("genesis", {"rule_version": b.rule["version"], "rule_sha256": b.rule["_sha256"],
                                  "grammar_version": G.GRAMMAR_VERSION, "grammar_sha256": G.grammar_sha256(),
                                  "lab_trials_count": 2552})
    b.registry.append("batch_frozen", {"batch": "0001", "candidates_sha256": sd["candidates_sha256"],
                                       "n_candidates": 400, "cells": {}, "generator": {}, "rule_sha256": "x",
                                       "snapshot_sha256": sd["snapshot_sha256"], "snapshot_end": sd["snapshot_end"],
                                       "source": "lab", "n_before": 2568})
    with patch.object(b, "state") as fake:
        real = B.Builder.state(b)
        real["batches"]["0001"]["frozen_hash"] = sd["frozen_hash"]
        fake.return_value = real
        assert b.ensure_seed() is None
    assert "seed" not in [r["kind"] for r in b.registry.records()]
    del st0


TESTS = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]


def main() -> int:
    bad = 0
    for n, f in TESTS:
        t0 = time.time()
        try:
            f()
            print(f"PASS  {n}  ({time.time() - t0:.1f}s)")
        except Exception as exc:
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
