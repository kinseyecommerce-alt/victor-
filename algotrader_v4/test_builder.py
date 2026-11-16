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
SEED1_N = 2552 + 16 + 400 + 2         # N after the sandbox batch 0001 (builder_seed_0001.json)
SEED_EVAL, SEED_FINALS = 6 * 400, 2 + 2 + 3 + 0 + 2 + 2   # the six sandbox batches 0001-0006 and their FINAL looks
SEED_N = 2552 + 16 + SEED_EVAL + SEED_FINALS   # = 4,979: N after every sandbox seed the server registry imports
FIRST, SECOND, THIRD = "0007", "0008", "0009"  # the server's own batches start after the six seeded ones
SEED_KINDS = ["genesis"] + ["seed"] * 6 + ["promoted"]   # a fresh registry's first records
PROVEN_SEEDED = "bld0003_int_hig_02_0ea14e"


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
    # each frozen rule pins its own grammar version (bg1 unchanged byte for byte next to bg2)
    assert G.grammar_sha256("bg1") == json.loads((HERE / "builder_rule.json").read_text())["grammar"]["sha256"]
    assert G.grammar_sha256("bg2") == json.loads((HERE / "builder_rule_v2.json").read_text())["grammar"]["sha256"]
    assert G.grammar_sha256() == G.grammar_sha256("bg2") and G.GRAMMAR_VERSION == "bg2"


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
    r = B.load_rule(B.RULE_FILE)                                   # builder/v1 stays valid and pinned
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
    assert rec["batch"] == FIRST and rec["n_candidates"] == 20
    kinds = [r["kind"] for r in b.registry.records()]
    assert kinds == SEED_KINDS + ["batch_frozen"]
    b.verify_frozen(FIRST, snap)
    try:
        b.verify_frozen(FIRST, {**snap, "end": 20261110})
        raise AssertionError("a changed data snapshot must be refused")
    except B.NotFrozen:
        pass
    p = b.candidates_path(FIRST)
    doc = json.loads(p.read_text())
    doc["candidates"][0]["text"] += " "
    p.write_text(json.dumps(doc))
    try:
        b.verify_frozen(FIRST, snap)
        raise AssertionError("an edited candidate file must be refused")
    except B.NotFrozen:
        pass
    assert b.freeze(snap, FIRST)["frozen_hash"] == rec["frozen_hash"]      # idempotent per id
    rec2 = b.freeze(snap, per_cell=1)
    assert rec2["batch"] == SECOND


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
    # a builder/v2 batch: every row carries its universe; the batch records what each universe resolved to
    assert s["rule_version"] == "builder/v2" and all(r["universe"] in G.UNIVERSES for r in s["rows"])
    assert s["universes"]["NIFTY100"]["n"] == 4 and s["universes"]["NIFTY200"]["added_complete"] == []
    assert sum(s["universes"]["counts"].values()) == 20
    out = b.ingest(rec["batch"], s)
    st = b.state()
    assert st["evaluated"] == SEED_EVAL + 20 and st["n_used"] == s["n_proof"]
    assert b.n_base(st) == 2552 + 16 + SEED_EVAL + 20 + st["final_looks"]
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
            "n_proof": SEED_N + 23, "v": 0.001, "best_discovery_dsr": {}, "planted": {}, "rows": [], "windows": {}}
    b.ingest(r1["batch"], summ)
    st = b.state()
    assert st["final_looks"] == SEED_FINALS + 2 and st["n_used"] == SEED_N + 23   # the seeds' looks + this batch's 2
    r2 = b.freeze(snap, per_cell=1)
    assert r2["n_before"] == SEED_N + 20 + 2
    b.ingest(r2["batch"], {**summ, "n_proof": 10})               # a smaller N can never lower N_used
    assert b.state()["n_used"] == SEED_N + 23
    assert B.api_status(b.root)["n_used"] >= SEED_N + 23


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
    assert pid in gen and PROVEN_SEEDED in gen and gen.count('"id":') == 2 and "GENERATED" in gen   # + the seeded
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
    assert [x["id"] for x in ns["SPECS"]] == [PROVEN_SEEDED, pid]
    assert ns["RULE_SHA256"] == B.CURRENT_RULE_SHA256 == B.RULE_SHA256_V2


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
        assert s["status"] == "complete" and s["batch"] == FIRST, s
        assert s["forward_reconciled"] == [PROVEN_SEEDED]          # the seeded promotion joins the forward family
        s = B.weekly_job(now=sat, root=root, runner=runner, snapshot_fn=lambda: {"source": "test", "end": 1})
        assert s["batch"] == SECOND and s["forward_reconciled"] == []
        st = B.Builder(root).state()
        assert st["evaluated"] == SEED_EVAL + 800 and calls == [FIRST, SECOND]


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
        bs = b.state()["batches"]
        assert sorted(bs) == ["0001", "0002", "0003", "0004", "0005", "0006"] and all(
            v["seeded"] for v in bs.values())                                                       # only the seeds
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
    snap, inputs = B.snapshot_process(Path(root), "synthetic", data_root=syn_root(), mem_mb=0)
    assert snap["end"] == syn_source().end and snap["inputs"]["delivery"]["sha256"]
    assert inputs and (Path(root) / "inputs" / inputs["dir"] / "manifest.json").exists()
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
    prom = {p["id"]: p for p in b.state()["promoted"]}
    fc = prom[s["proven"][0]]["cand"]
    g = next(g for g in st["grid"] if (g["agent"], g["mtype"]) == (fc["agent"], fc["mtype"]))
    assert sum(x["proven"] for x in cell) == 2 and g["in_forward_lab"] >= 1        # this batch's + the seeded one
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
# the sandbox seeds (batches 0001-0006 of the lab sandbox count on the server)
# ══════════════════════════════════════════════════════════════════════════
def test_seed_file_pinned_and_consistent():
    sd = B.load_seed()
    assert sd["_sha256"] == B.SEED_SHA256 == hashlib.sha256(B.SEED_FILE.read_bytes()).hexdigest()
    assert sd["batch"] == "0001" and sd["n_candidates"] == 400 == len(sd["candidates"])
    assert sd["frozen_hash"].startswith("f506bc18") and sd["rule_sha256"] == B.RULE_SHA256
    assert sd["grammar_sha256"] == G.grammar_sha256("bg1")
    assert sd["n_used_after"] == SEED1_N == 2970
    assert len(sd["final_looks"]) == 2 and not any(f["confirmed"] for f in sd["final_looks"])
    assert sd["development_evaluations"]["count"] == B.load_rule()["dsr"]["development_evaluations"] == 16
    # the 400 keys are exactly batch 0001's bg1 draw (deterministic grammar; bg1 is kept byte for byte)
    assert [c["key"] for c in sd["candidates"]] == [c.key for c in G.draw_batch("0001", set(), version="bg1")]


def test_seed2_file_pinned_and_consistent():
    sd = B.load_seed(B.SEED2_FILE, expect=B.SEED2_SHA256)
    assert sd["_sha256"] == B.SEED2_SHA256 == hashlib.sha256(B.SEED2_FILE.read_bytes()).hexdigest()
    assert [p for p, _ in B.seed_files()] == [B.SEED_FILE, B.SEED2_FILE]
    assert sd["follows"]["sha256"] == B.SEED_SHA256 and sd["follows"]["frozen_hash"] == B.load_seed()["frozen_hash"]
    bs = B.seed_batches(sd)
    assert [x["batch"] for x in bs] == ["0002", "0003", "0004", "0005", "0006"]
    seen = {c["key"] for c in B.load_seed()["candidates"]}
    n = SEED1_N
    for x in bs:
        assert x["rule_sha256"] == B.RULE_SHA256 and x["grammar_sha256"] == G.grammar_sha256("bg1")
        assert x["n_candidates"] == 400 == len(x["candidates"]) == len({c["key"] for c in x["candidates"]})
        keys = [c["key"] for c in x["candidates"]]
        assert not (set(keys) & seen)                                  # never a key drawn in an earlier batch
        # each batch is the bg1 draw of its id given every earlier batch (deterministic, never re-drawn)
        assert keys == [c.key for c in G.draw_batch(x["batch"], seen, version="bg1")], x["batch"]
        seen |= set(keys)
        assert x["n_before"] == n and x["n_used_after"] == n + 400 + len(x["final_looks"])
        n = x["n_used_after"]
        assert x["counts"]["candidates"] == 400 and x["counts"]["final_looks"] == len(x["final_looks"])
    assert n == sd["n_used_after"] == SEED_N == 4979
    assert [p["id"] for p in sd["promoted"]] == [PROVEN_SEEDED]
    p = sd["promoted"][0]
    assert p["batch"] == "0003" and PROVEN_SEEDED in bs[1]["proven"] and p["dsr"] >= 0.95
    assert p["frozen_hash"] == bs[1]["frozen_hash"] and p["reference"]["daily"] and p["universe_names"]
    assert p["cand"]["agent"] == "intraday" and p["cand"]["mtype"] == "HIGH_VOL"


def test_seed_import_idempotent():
    b = new_builder("seed_idem")
    b.ensure_genesis()
    b.ensure_genesis()
    b.ensure_seed()
    kinds = [r["kind"] for r in b.registry.records()]
    assert kinds == SEED_KINDS, kinds
    assert b.registry.verify()["ok"]
    st = b.state()
    assert sorted(st["batches"]) == ["0001", "0002", "0003", "0004", "0005", "0006"]
    assert all(st["batches"][x]["seeded"] for x in st["batches"]) and st["seed"]["batch"] == "0001"
    assert sorted(st["seeds"]) == sorted(st["batches"])
    assert st["evaluated"] == SEED_EVAL and st["final_looks"] == SEED_FINALS
    assert st["n_used"] == SEED_N and b.n_base(st) == SEED_N
    assert st["batches"]["0004"]["n_before"] == st["results"]["0003"]["n_used"]
    assert st["results"]["0003"]["proven"] == [PROVEN_SEEDED]
    # a fresh object on the same registry imports nothing again
    B.Builder(b.root).ensure_genesis()
    B.Builder(b.root).ensure_seed()
    assert [r["kind"] for r in b.registry.records()] == SEED_KINDS
    st_api = B.api_status(b.root)
    assert st_api["n_used"] == SEED_N and {r["status"] for r in st_api["batches"]} == {"seeded"}
    assert len(st_api["batches"]) == 6 and st_api["seed"]["imported"] is True and len(st_api["seeds"]) == 6
    assert st_api["promoted"] == 1 and st_api["promoted_unregistered"] == 1
    assert B.api_batch("0001", b.root)["candidates"] == []          # the seed carries keys, not the sandbox's rows
    a3 = B.api_batch("0003", b.root)
    assert a3["status"] == "seeded" and a3["seed"]["proven"] == [PROVEN_SEEDED]
    assert a3["seed"]["file"] == B.SEED2_FILE.name and a3["cell_universes"]
    # before the registry exists the API already shows N after every seed
    assert B.api_status(os.path.join(_TMP, "seed_idem_none"))["n_used"] == SEED_N


def test_seeded_promotion_registered_with_forward_family():
    b = new_builder("seed_promo")
    b.ensure_genesis()
    st = b.state()
    p = st["promoted"][0]
    assert p["id"] == PROVEN_SEEDED and p["seeded"] and not p["forward_registered"]
    assert p["batch"] == "0003" and p["frozen_hash"] == st["batches"]["0003"]["frozen_hash"]
    cat = (b.root / "catalog" / "builder_gen.py").read_text()
    assert PROVEN_SEEDED in cat                                        # the generated catalogue lists it
    assert b.reconcile_forward() == [PROVEN_SEEDED]
    assert b.reconcile_forward() == []                                # idempotent
    st = b.state()
    assert st["promoted"][0]["forward_registered"]
    import builder_forward
    fw = builder_forward.BuilderForward(b.root / "forward").state()["strategies"]
    assert PROVEN_SEEDED in fw
    assert B.api_status(b.root)["promoted_unregistered"] == 0
    # no second promotion on a later import attempt
    b.ensure_seed()
    assert sum(1 for r in b.registry.records() if r["kind"] == "promoted") == 1


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


def test_seed2_tampered_refused():
    tmp = Path(_TMP) / "seed2_tampered.json"
    txt = B.SEED2_FILE.read_text()
    assert '"n_used_after": 4979' in txt
    tmp.write_text(txt.replace('"n_used_after": 4979', '"n_used_after": 3000'))
    try:
        B.load_seed(tmp, expect=B.SEED2_SHA256)
        raise AssertionError("an edited second seed must be refused")
    except B.SeedChanged:
        pass
    b = new_builder("seed2_tampered")
    with patch.object(B, "SEED2_FILE", tmp):
        try:
            b.ensure_genesis()
            raise AssertionError("a registry must not import an edited seed")
        except B.SeedChanged:
            pass
        # every file is verified before anything is written: not even the valid seed 0001 is imported
        assert [r["kind"] for r in b.registry.records()] == ["genesis"]
        with patch.object(settings, "builder_enabled", True):
            from datetime import datetime
            s = B.weekly_job(now=datetime(2026, 11, 21, 6, 0), root=b.root, runner=lambda r, x: {},
                             snapshot_fn=lambda: {"source": "test", "end": 1})
        assert s["status"] == "seed_changed", s
        assert not B.Builder(b.root).state()["batches"]


def test_seed_counts_in_n_and_keys_never_redrawn():
    b = new_builder("seed_n")
    rec = b.freeze({"source": "test", "end": 20261109})
    assert rec["batch"] == FIRST and rec["n_before"] == SEED_N
    seeded = {c["key"] for c in B.load_seed()["candidates"]}
    for x in B.seed_batches(B.load_seed(B.SEED2_FILE, expect=B.SEED2_SHA256)):
        seeded |= {c["key"] for c in x["candidates"]}
    assert len(seeded) == SEED_EVAL and seeded <= b.drawn_keys()
    doc = json.loads(b.candidates_path(FIRST).read_text())
    assert not ({c["key"] for c in doc["candidates"]} & seeded)
    assert len(doc["candidates"]) == 400
    assert B.Builder(b.root).next_batch_id() == SECOND


def _fake_source_registry(b, upto: int):
    """A registry that IS the sandbox: its batches 0001..upto carry the seeds' frozen hashes (+ the promotion)."""
    b.registry.append("genesis", {"rule_version": b.rule["version"], "rule_sha256": b.rule["_sha256"],
                                  "grammar_version": G.GRAMMAR_VERSION, "grammar_sha256": G.grammar_sha256(),
                                  "lab_trials_count": 2552})
    sds = [B.load_seed()] + B.seed_batches(B.load_seed(B.SEED2_FILE, expect=B.SEED2_SHA256))
    for sd in sds[:upto]:
        b.registry.append("batch_frozen", {"batch": sd["batch"], "candidates_sha256": sd["candidates_sha256"],
                                           "n_candidates": 400, "cells": {}, "generator": {}, "rule_sha256": "x",
                                           "snapshot_sha256": sd["snapshot_sha256"],
                                           "snapshot_end": sd["snapshot_end"], "source": "lab", "n_before": 2568})
    real = B.Builder.state(b)
    for sd in sds[:upto]:
        real["batches"][sd["batch"]]["frozen_hash"] = sd["frozen_hash"]
    if upto >= 3:
        real["promoted"] = [{"id": PROVEN_SEEDED}]
    return real


def test_seed_skipped_in_its_source_registry():
    # a registry whose batches 0001-0006 ARE the seeds' batches (the sandbox) imports nothing
    b = new_builder("seed_source")
    real = _fake_source_registry(b, 6)
    with patch.object(b, "state") as fake:
        fake.return_value = real
        assert b.ensure_seed() is None
    kinds = [r["kind"] for r in b.registry.records()]
    assert "seed" not in kinds and "promoted" not in kinds
    # the real sandbox registry (when the lab scratch is present): a copy imports nothing and continues at 0007
    src = Path("/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad/builder/root"
               "/registry.jsonl")
    if src.exists():
        root = Path(_TMP) / "seed_source_real"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        shutil.copy(src, root / "registry.jsonl")
        sb = B.Builder(root, B.load_rule(B.RULE_FILE, expect=B.RULE_SHA256))
        n0 = len(sb.registry.records())
        assert sb.ensure_seed() is None and len(sb.registry.records()) == n0
        assert sb.next_batch_id() == FIRST and sb.n_base() == SEED_N


def test_seed_only_0001_source_imports_the_rest():
    # a registry that holds only seed 0001's batch (as the sandbox did on 2026-11-15) imports 0002-0006 under
    # their own ids and the promotion
    b = new_builder("seed_source_0001")
    real = _fake_source_registry(b, 1)
    with patch.object(b, "state", side_effect=lambda: _merge(real, B.Builder.state(b))):
        b.ensure_seed()
    kinds = [r["kind"] for r in b.registry.records()]
    assert kinds == ["genesis", "batch_frozen"] + ["seed"] * 5 + ["promoted"], kinds
    st = B.Builder.state(b)
    assert sorted(st["seeds"]) == ["0002", "0003", "0004", "0005", "0006"]
    assert st["promoted"][0]["batch"] == "0003"


def _merge(fake: dict, live: dict) -> dict:
    """The live state with batch 0001's frozen hash set to the seed's (the source registry's batch 0001)."""
    live["batches"]["0001"]["frozen_hash"] = fake["batches"]["0001"]["frozen_hash"]
    return live


# ══════════════════════════════════════════════════════════════════════════
# regressions (PR #26 review): resume on frozen inputs, terminal failure, own forward schedule, registration retry
# ══════════════════════════════════════════════════════════════════════════
SAT = None


def _sat():
    from datetime import datetime
    return datetime(2026, 11, 21, 6, 0)


def _data_copy(name: str) -> str:
    d = os.path.join(_TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(syn_root(), d)
    return d


def _refresh_data(d: str) -> None:
    """What the data layer does between two runs: one more ban row (an end-of-day file changes)."""
    p = os.path.join(d, "nse", "ban.csv.gz")
    with gzip.open(p, "rt") as fh:
        txt = fh.read()
    with gzip.open(p, "wt") as fh:
        fh.write(txt + "2026-11-10,SYN00\n")


class _Crash(RuntimeError):
    pass


def test_regression_resume_uses_frozen_inputs_after_data_refresh():
    """P1: an evaluation that died must resume on the inputs frozen with the batch, even after the data layer
    refreshed its files (before the fix the retry was refused for ever: NotFrozen on the changed snapshot)."""
    d = _data_copy("data_resume")
    root = os.path.join(_TMP, "resume")
    shutil.rmtree(root, ignore_errors=True)

    def crash(r, bid):
        raise B.RunFailed("time limit 2s exceeded (resumable: finished units are kept)")
    with patch.dict(os.environ, {"BUILDER_DATA_DIR": d}), patch.object(settings, "builder_enabled", True), \
            patch.object(settings, "builder_jobs", 1):
        s1 = B.weekly_job(now=_sat(), root=root, source="server", runner=crash)
        assert s1["status"] == "error" and s1["batch"] == FIRST, s1
        _refresh_data(d)
        s2 = B.weekly_job(now=_sat(), root=root, source="server")
    assert s2["status"] == "complete" and s2["batch"] == FIRST, s2
    st = B.Builder(root).state()
    assert FIRST in st["results"] and st["evaluated"] == SEED_EVAL + 400


def test_regression_terminal_failure_unblocks_queue():
    """P1: a batch that keeps failing is recorded `failed` after the rule's attempts; its candidates stay counted in N
    and in the never-redraw set, and the next run freezes a new batch (before the fix: blocked for ever)."""
    root = os.path.join(_TMP, "terminal")
    shutil.rmtree(root, ignore_errors=True)

    def crash(r, bid):
        raise B.RunFailed("engine exit -9: killed")
    snap = lambda: {"source": "test", "end": 20261109}
    with patch.object(settings, "builder_enabled", True):
        seen = [B.weekly_job(now=_sat(), root=root, runner=crash, snapshot_fn=snap) for _ in range(4)]
    b = B.Builder(root)
    st = b.state()
    assert seen[0]["batch"] == FIRST
    assert any(x.get("status") == "failed" and x.get("batch") == FIRST for x in seen), seen
    assert st["results"][FIRST].get("failed") is True
    assert SECOND in st["batches"], list(st["batches"])
    keys2 = {c["key"] for c in json.loads(b.candidates_path(FIRST).read_text())["candidates"]}
    assert keys2 <= b.drawn_keys(st)
    keys3 = {c["key"] for c in json.loads(b.candidates_path(SECOND).read_text())["candidates"]}
    assert not (keys2 & keys3)
    assert st["evaluated"] >= SEED_EVAL + 400 and b.n_base(st) >= SEED_N + 400


def test_regression_lost_inputs_and_changed_data_fail_terminally():
    """P1: frozen inputs lost (ephemeral disk wiped) AND the live data changed -> the batch is failed at once (it can
    never be evaluated on its frozen inputs again), counted, and the queue moves on."""
    d = _data_copy("data_lost")
    root = os.path.join(_TMP, "lost")
    shutil.rmtree(root, ignore_errors=True)

    def crash(r, bid):
        raise B.RunFailed("crash")
    with patch.dict(os.environ, {"BUILDER_DATA_DIR": d}), patch.object(settings, "builder_enabled", True):
        B.weekly_job(now=_sat(), root=root, source="server", runner=crash)
        for p in Path(root).glob("inputs/*"):
            shutil.rmtree(p, ignore_errors=True)
        _refresh_data(d)
        s = B.weekly_job(now=_sat(), root=root, source="server", runner=crash)
    assert s.get("status") == "failed" and s.get("batch") == FIRST, s
    st = B.Builder(root).state()
    assert st["results"][FIRST]["failed"] and st["evaluated"] == SEED_EVAL + 400


def test_regression_builder_forward_scheduled_without_forward_lab():
    """P2: with BUILDER_ENABLED and FORWARD_LAB_ENABLED off, the builder/v1 family still gets its own job."""
    from unittest.mock import MagicMock
    import asyncio
    import platform_scheduler as PS
    with patch.object(settings, "builder_enabled", True), patch.object(settings, "forward_lab_enabled", False):
        ps = PS.PlatformScheduler()
        ps._sched = MagicMock()
        ps.start()
        ids = [c.kwargs.get("id") for c in ps._sched.add_job.call_args_list]
        assert "forward_lab" not in ids and "builder_forward" in ids, ids
        job = [c for c in ps._sched.add_job.call_args_list if c.kwargs.get("id") == "builder_forward"][0]
        fn = job.args[0]
        with patch.object(BF, "run_pending", return_value={"status": "complete"}) as rp:
            asyncio.run(fn())
        assert rp.called
    with patch.object(settings, "builder_enabled", False):
        ps = PS.PlatformScheduler()
        ps._sched = MagicMock()
        ps.start()
        assert "builder_forward" not in [c.kwargs.get("id") for c in ps._sched.add_job.call_args_list]


def test_regression_forward_registration_retried():
    """P2: a promotion whose forward registration failed is persisted as unregistered and registered on the next run
    (weekly or the daily forward step); idempotent."""
    with patch.object(settings, "builder_enabled", True):
        b = new_builder("regretry")
        rec = b.freeze({"source": "test", "end": 20261109}, per_cell=1)
        s = _fake_proven_summary(b, rec["batch"])
        pid = s["proven"][0]
        with patch.object(BF.BuilderForward, "register", side_effect=OSError("volume full")):
            b.ingest(rec["batch"], s)
        st = b.state()
        assert [p["id"] for p in st["promoted"]] == [PROVEN_SEEDED, pid]      # (the seeded one comes first)
        fw = BF.BuilderForward(b.root / "forward")
        assert pid not in fw.state()["strategies"]
        # the next daily forward step reconciles it
        BF.run_pending(b.root / "forward", runner=lambda r, dr, day: {"day": day, "calendar": [], "trades": {}},
                       today=20261120)
        assert {pid, PROVEN_SEEDED} <= set(BF.BuilderForward(b.root / "forward").state()["strategies"])
        assert all(p.get("forward_registered") is True for p in B.Builder(b.root).state()["promoted"])

        def n_regs():
            return len([r for r in BF.BuilderForward(b.root / "forward").registry.records() if r["kind"] == "register"])
        n0 = n_regs()
        assert 1 <= n0 <= 2                                       # the failed one + the seeded one, each once
        BF.run_pending(b.root / "forward", runner=lambda r, dr, day: {"day": day, "calendar": [], "trades": {}},
                       today=20261120)
        assert n_regs() == n0                                     # idempotent


# ══════════════════════════════════════════════════════════════════════════
# builder/v2: the stock UNIVERSE as a grammar dimension (bg2), N continuity, wider universe in promotion / forward
# ══════════════════════════════════════════════════════════════════════════
def uni_root() -> str:
    """A synthetic tree with a NIFTY200 extension: base SYN00-02, added SYN03-07; SYN03 has complete minute data,
    SYN04-07 start in 2025 (incomplete). Four of eight names without bars would make every earlier session 'special'
    if the special-session rule looked at all names."""
    if "uroot" not in _SYN:
        root = os.path.join(_TMP, "data_uni")
        late = {f"SYN0{k}": 20250602 for k in range(4, 8)}
        SP.make_synthetic_tree(root, start=20230102, end=20261109, n_stocks=8, seed=7, n_extra=5, late=late)
        _SYN["uroot"] = root
    return _SYN["uroot"]


def uni_engine():
    if "uE" not in _SYN:
        E = BE.Engine(SP.ServerSource(uni_root()))
        E.cache_dir = os.path.join(_TMP, "cache")
        _SYN["uE"] = E
    return _SYN["uE"]


def _all_seed_keys() -> set:
    """Every key of the six sandbox seed batches (0001-0006)."""
    keys = {c["key"] for c in B.load_seed()["candidates"]}
    for x in B.seed_batches(B.load_seed(B.SEED2_FILE, expect=B.SEED2_SHA256)):
        keys |= {c["key"] for c in x["candidates"]}
    return keys


def _v1_drawn() -> set:
    """Every bg1 key a server registry holds after the seed and one v1 batch (0002)."""
    seed = {c["key"] for c in B.load_seed()["candidates"]}
    return seed | {c.key for c in G.draw_batch("0002", seed, version="bg1")}


def test_bg2_never_redraws_a_v1_key():
    drawn = _v1_drawn()
    v1 = set(drawn)
    for bid in ("0003", "0004", "0005"):
        b = G.draw_batch(bid, drawn, version="bg2")
        assert len(b) == 400 and all(c.version == "bg2" and c.universe for c in b)
        cells = {}
        for c in b:
            cells[c.cell] = cells.get(c.cell, 0) + 1
            assert c.key not in drawn, "a key of an earlier batch was drawn again"
            assert c.v1_key is None or c.v1_key not in v1, "the bg1 twin of a v1-drawn sentence was drawn again"
            assert c.key not in v1 and not c.key.startswith('["bg1"')
        assert set(cells.values()) == {20} and len(cells) == 20
        drawn |= {c.key for c in b}
    # bg1 keys are unchanged byte for byte (seed rows), a bg1 candidate's JSON carries no universe
    sd = B.load_seed()
    c1 = G.draw_batch("0001", set(), version="bg1")
    assert [c.key for c in c1] == [r["key"] for r in sd["candidates"]]
    assert all("universe" not in c.to_json() and c.v1_key == c.key for c in c1)
    # a bg2 twin of a v1 sentence IS that sentence: excluded once the bg1 key was drawn
    twin = G.Candidate.from_json({**c1[0].to_json(), "universe": c1[0].stock_universe})
    assert twin.v1_key == c1[0].key and not G.never_drawn(twin, {c1[0].key})
    wide = G.Candidate.from_json({**c1[0].to_json(), "universe": "NIFTY200"})
    assert wide.v1_key is None and G.never_drawn(wide, {c1[0].key}) and wide.key != twin.key


def test_universe_dimension_deterministic():
    drawn = _v1_drawn()
    a = G.draw_batch("0009", drawn, version="bg2")
    b = G.draw_batch("0009", drawn, version="bg2")
    assert [c.key for c in a] == [c.key for c in b]
    assert [c.key for c in a] != [c.key for c in G.draw_batch("0010", drawn, version="bg2")]
    us = {c.universe for c in a}
    assert {"NIFTY100", "NIFTY200", "INDEX"} <= us, us
    for c in a:
        assert c.universe in G.universes_for(c.agent, c.block, c.instrument)
        if c.grid in ("minute", "prev_eod"):
            assert c.universe in ("NIFTY100", "NIFTY200")
        if c.instrument != "STOCKS":
            assert c.universe == "INDEX"
    for x in a:                                       # the universe is a near-duplicate field
        for y in a:
            assert x is y or not G.near_duplicate(x, y)
    assert G.grammar_sizes("bg2")["intraday"] == 2 * G.grammar_sizes("bg1")["intraday"]
    assert G.grammar_sizes("bg2")["swing"] == 3 * G.grammar_sizes("bg1")["swing"]
    assert G.grammar_sizes("bg2")["options"] == G.grammar_sizes("bg1")["options"]
    # the widened proven rules are ordinary bg2 sentences: in the grammar, NIFTY200, no bg1 twin (no inherited proof)
    pool = {c.key for a_ in G.AGENTS for m in G.MARKET_TYPES for c in G.cell_grammar(a_, m, "bg2")}
    tw = G.proven_rule_twins()
    assert len(tw) == 12 and all(t.key in pool and t.universe == "NIFTY200" and t.v1_key is None for t in tw)
    assert all("NIFTY 200" in G.describe(t) for t in tw)
    import inspect
    assert "proven_rule_twins" not in inspect.getsource(G.draw_batch)       # the draw never special-cases them


def test_nifty200_minute_candidates_use_only_complete_names():
    import dataclasses
    E = uni_engine()
    M = E.ctx.minute
    names = lambda rows: sorted(M.syms[i] for i in rows)
    assert names(E.universe_rows("NIFTY100")) == ["SYN00", "SYN01", "SYN02"]
    assert names(E.universe_rows("NIFTY200")) == ["SYN00", "SYN01", "SYN02", "SYN03"]
    rep = E.universe_report()
    assert rep["NIFTY200"]["added_complete"] == ["SYN03"]
    assert rep["NIFTY200"]["added_incomplete"] == ["SYN04", "SYN05", "SYN06", "SYN07"]
    # special sessions come from the base names only: the four late names never make 2023-24 'special'
    assert not any(d < 20250602 for d in M.special), sorted(M.special)[:5]
    base = G.Candidate("intraday", "RISING", "gap_fade", (("g", 0.01), ("mode", "fade_day")), "o0915",
                       ("time", "o1515"), "STOCKS", "mis", "", "NIFTY200")
    seen = {}
    for u, want in (("NIFTY200", {"SYN00", "SYN01", "SYN02", "SYN03"}), ("NIFTY100", {"SYN00", "SYN01", "SYN02"}),
                    ("", {"SYN00", "SYN01", "SYN02"})):
        got = set()
        for mt in G.MARKET_TYPES:
            c = dataclasses.replace(base, mtype=mt, universe=u)
            sig = E.signals(c)
            got |= {M.syms[s] for s in sig["s"].tolist()}
            tr = BE.run_jobs(E, [BE.Job("x", c, None, 0, 20261109, False)])["x"]["trades"]
            assert {t[1] for t in tr} <= want, ({t[1] for t in tr}, u)
            es, _et = E.eligible(c, 0, 20261109)
            assert {M.syms[s] for s in es.tolist()} <= want
        assert got <= want and got, (u, got)
        seen[u] = got
    assert "SYN03" in seen["NIFTY200"]                 # the complete added name trades on NIFTY200 only
    # daily grid: TOP500 = no filter; NIFTY100 / NIFTY200 = the base / base + added names (no minute completeness)
    sw = G.Candidate("swing", "RISING", "deliv_up", (("pp", 15.0), ("day", "up")), "next_open", ("hold", 5), "STOCKS",
                     "cnc/fut", "", "TOP500")
    assert E.row_mask(sw) is None and E.row_mask(dataclasses.replace(sw, universe="")) is None
    m100 = E.row_mask(dataclasses.replace(sw, universe="NIFTY100"))
    m200 = E.row_mask(dataclasses.replace(sw, universe="NIFTY200"))
    dn = E.ctx.daily.syms
    assert {s for s, k in zip(dn, m100) if k} == {"SYN00", "SYN01", "SYN02"}
    assert {s for s, k in zip(dn, m200) if k} == {f"SYN0{k}" for k in range(8)}
    # a frozen name list (forward runs) overrides the resolution
    c = dataclasses.replace(base, mtype="RISING")
    sig = E.signals(c, names=("SYN01",))
    assert {M.syms[s] for s in sig["s"].tolist()} <= {"SYN01"}


def test_n_continuity_v1_to_v2_and_v1_batch_evaluated_under_v1():
    root = os.path.join(_TMP, "upgrade")
    shutil.rmtree(root, ignore_errors=True)
    v1 = B.Builder(root, B.load_rule(B.RULE_FILE))
    src = syn_source()
    r1 = v1.freeze(src.snapshot(), per_cell=1)                  # a builder/v1 batch (bg1) left pending
    assert r1["batch"] == FIRST and r1["generator"]["grammar_version"] == "bg1" and r1["n_before"] == SEED_N
    keys1 = {c["key"] for c in json.loads(v1.candidates_path(FIRST).read_text())["candidates"]}
    v2 = B.Builder(root)                                        # the deploy: the current rule is builder/v2
    assert v2.rule["version"] == "builder/v2"
    v2.ensure_genesis()                                         # the weekly job's first step: the upgrade record
    st = v2.state()
    assert st["rules"] == [B.RULE_SHA256, B.RULE_SHA256_V2]
    up = st["upgrades"][0]
    assert up["n_used_before"] == SEED_N and up["evaluated_before"] == SEED_EVAL and up["batches_before"] == 7
    assert up["final_looks_before"] == SEED_FINALS and up["from_version"] == "builder/v1" and up["grammar_version"] == "bg2"
    # the pending v1 batch is resumed first and evaluated under builder/v1 (its own rule); it counts in N
    import builder_run
    s = builder_run.evaluate(root, FIRST, "synthetic", syn_root(), jobs=1, log=lambda m: None)
    assert s["rule_version"] == "builder/v1" and s["n_disc"] == SEED_N + 20
    v2.ingest(FIRST, s)
    st = v2.state()
    assert st["evaluated"] == SEED_EVAL + 20 and v2.n_base(st) == 2552 + 16 + SEED_EVAL + 20 + st["final_looks"]
    assert st["batches"][FIRST]["frozen_hash"] == r1["frozen_hash"]                  # v1 records never rewritten
    # N continues: the first v2 batch starts from the seeds' 2,400 + the v1 batch's 20 + every FINAL look so far
    r2 = v2.freeze({"source": "test", "end": 20261109}, per_cell=1)
    assert r2["batch"] == SECOND and r2["generator"]["grammar_version"] == "bg2" and r2["rule_version"] == "builder/v2"
    assert r2["n_before"] == SEED_N + 20 + st["final_looks"] - SEED_FINALS == v2.n_base(st)
    keys2 = [G.Candidate.from_json(c["cand"]) for c in json.loads(v2.candidates_path(SECOND).read_text())["candidates"]]
    seed = _all_seed_keys()
    assert all(c.key not in keys1 | seed and (c.v1_key is None or c.v1_key not in keys1 | seed) for c in keys2)
    # the registry moved on: a builder/v1 object can no longer freeze; the chain verifies; N never shrinks
    try:
        B.Builder(root, B.load_rule(B.RULE_FILE)).freeze({"source": "test", "end": 1}, per_cell=1)
        raise AssertionError("a v1 builder must not freeze on a registry upgraded to v2")
    except B.RuleChanged:
        pass
    assert v2.registry.verify()["ok"]
    assert [r["kind"] for r in v2.registry.records()].count("rule_upgrade") == 1
    B.Builder(root).ensure_genesis()                            # idempotent
    assert [r["kind"] for r in v2.registry.records()].count("rule_upgrade") == 1
    assert B.api_status(root)["n_used"] >= up["n_used_before"]
    # a fresh registry starts directly under v2 (genesis + seeds, no upgrade record)
    fresh = new_builder("fresh_v2")
    fresh.ensure_genesis()
    assert [r["kind"] for r in fresh.registry.records()] == SEED_KINDS
    assert fresh.state()["genesis"]["grammar_version"] == "bg2"


def test_rule_v2_pinned_and_v1_unchanged():
    r1, r2 = B.load_rule(B.RULE_FILE), B.load_rule()
    assert r1["_sha256"] == B.RULE_SHA256 == "a455009021b9e2cfbc55106c8ee1310ac6a7135995a895b6b8cafda06fbc3c06"
    assert r2["_sha256"] == B.RULE_SHA256_V2 == hashlib.sha256((HERE / "builder_rule_v2.json").read_bytes()).hexdigest()
    assert r2["version"] == "builder/v2" and r2["supersedes"]["sha256"] == B.RULE_SHA256
    for k in ("splits", "costs", "discovery_gates", "carry", "validation", "final", "controls", "proven", "freeze"):
        assert r1[k] == r2[k], k                              # only the grammar (universe) changed
    for k in ("min", "lab_trials_count", "development_evaluations", "formula", "V"):
        assert r1["dsr"][k] == r2["dsr"][k], k
    assert r2["batch"]["per_cell"] == 20 and r2["batch"]["size"] == 400
    assert set(r2["grammar"]["universes"]) >= {"NIFTY100", "NIFTY200", "TOP500", "INDEX"}
    tmp = Path(_TMP) / "rule_v2_edit.json"
    tmp.write_text((HERE / "builder_rule_v2.json").read_text().replace('"min": 0.95', '"min": 0.90'))
    try:
        B.load_rule(tmp)
        raise AssertionError("an edited v2 rule must be refused")
    except B.RuleChanged:
        pass
    assert B.rule_by_sha(B.RULE_SHA256)["version"] == "builder/v1"


def test_promotion_and_forward_of_a_nifty200_strategy():
    """A NIFTY200 candidate proven on the batch's data is promoted like any other; its forward entry trades the
    universe frozen at the proof, read from the builder data layer's minute store (which holds the added names)."""
    import dataclasses
    E = uni_engine()
    c = G.Candidate("intraday", "RISING", "gap_fade", (("g", 0.01), ("mode", "fade_day")), "o0915",
                    ("time", "o1515"), "STOCKS", "mis", "", "NIFTY200")
    frozen = E.frozen_names(c)
    assert frozen == ["SYN00", "SYN01", "SYN02", "SYN03"]
    with patch.object(settings, "builder_enabled", True):
        b = new_builder("promo200")
        rec = b.freeze({"source": "test", "end": 20261109}, per_cell=1)
        doc = json.loads(b.candidates_path(rec["batch"]).read_text())
        doc["candidates"][0]["cand"] = c.to_json()                 # (test fixture: the first row is our candidate)
        doc["candidates"][0]["key"] = c.key
        b.candidates_path(rec["batch"]).write_text(B.canon(doc))
        s = _fake_proven_summary(b, rec["batch"])
        s["rows"][0]["proof"]["universe_names"] = frozen
        s["windows"]["T"] = 20250101                               # registered on 2025-01-01: a forward year
        b.ingest(rec["batch"], s)
    pid = s["proven"][0]
    pr = next(p for p in b.state()["promoted"] if p["id"] == pid)
    assert pr["universe"] == "NIFTY200" and pr["universe_names"] == frozen and pr["forward_registered"]
    fw = BF.BuilderForward(b.root / "forward")
    assert fw.state()["strategies"][pid]["universe_names"] == frozen
    gen = (b.root / "catalog" / "builder_gen.py").read_text()
    assert '"universe": "NIFTY200"' in gen and "SYN03" in gen
    from stratlab import builder_catalog as BC
    spec = [x for x in BC.load(str(b.root / "catalog" / "builder_gen.py")) if x.id == pid][0]
    assert spec.names == tuple(frozen) and spec.cand.universe == "NIFTY200"
    # the forward step (builder data layer's store) trades exactly the frozen names, the added SYN03 included
    res = BF.forward_trades(str(b.root / "forward"), uni_root(), 20261109, source="server")
    syms = {t[1] for t in res["trades"][pid]}
    assert syms and syms <= set(frozen), syms
    E2 = BE.Engine(SP.ServerSource(uni_root()))
    every = set()
    for mt in G.MARKET_TYPES:
        every |= {t[1] for t in BE.run_jobs(E2, [BE.Job("y", dataclasses.replace(c, mtype=mt), None, 20250101,
                                                        20261109, False)])["y"]["trades"]}
    assert "SYN03" in every


def test_api_shows_universes():
    b = new_builder("api_uni")
    rec = b.freeze({"source": "test", "end": 20261109}, per_cell=1)
    st = B.api_status(b.root)
    assert st["rule_version"] == "builder/v2" and st["grammar_version"] == "bg2"
    assert [r["version"] for r in st["rules"]] == ["builder/v1", "builder/v2"]
    assert set(st["universes"]["choices"]) == {"NIFTY100", "NIFTY200", "TOP500", "INDEX"}
    row = [r for r in st["batches"] if r["id"] == rec["batch"]][0]
    assert row["rule_version"] == "builder/v2" and sum(row["universes"].values()) == 20
    seed_row = [r for r in st["batches"] if r["id"] == "0001"][0]
    assert seed_row["rule_version"] == "builder/v1" and sum(seed_row["universes"].values()) == 400
    assert all("universes" in g for g in st["grid"])
    assert sum(u["candidates"] for g in st["grid"] for u in g["universes"].values()) == SEED_EVAL + 20
    d = B.api_batch(rec["batch"], b.root)
    assert all(c["universe"] in G.UNIVERSES and c["universe_explicit"] for c in d["candidates"])
    assert d["rule_version"] == "builder/v2" and sum(sum(v.values()) for v in d["cell_universes"].values()) == 20


def test_new_batches_wait_for_the_minute_universe():
    d = _data_copy("data_wait")
    with patch.dict(os.environ, {"BUILDER_DATA_DIR": d}):
        b = new_builder("wait")
        assert B.data_wait_reason(b, "server") is None             # synthetic tree: universe file, nothing to add
        p = Path(d) / "minute" / "universe.json"
        u = json.loads(p.read_text())
        p.write_text(json.dumps({**u, "extra": ["NEWCO"]}))         # an added name not backfilled yet
        assert "0/1 names" in (B.data_wait_reason(b, "server") or "")
        assert B.data_wait_reason(b, "lab") is None                 # (only the server's data layer is waited for)
        assert B.data_wait_reason(B.Builder(b.root, B.load_rule(B.RULE_FILE)), "server") is None   # bg1: no wait
        with patch.object(settings, "builder_minute_universe", "nifty100"):
            assert "nifty100" in (B.data_wait_reason(b, "server") or "")
        p.unlink()
        assert B.data_wait_reason(b, "server")                      # no minute run under the new data layer yet
        with patch.object(settings, "builder_enabled", True):
            s = B.weekly_job(now=_sat(), root=b.root, source="server")
        assert s["status"] == "waiting_data" and "batch" not in s, s
        assert sorted(B.Builder(b.root).state()["batches"]) == ["0001", "0002", "0003", "0004", "0005", "0006"]
        # (only the seeds: nothing frozen, nothing counted)


def test_flag_defaults_unchanged():
    from config import Settings
    s = Settings(_env_file=None)
    assert s.builder_enabled is False and s.builder_data_enabled is False and s.forward_lab_enabled is False
    assert s.builder_minute_universe == "nifty200" and s.builder_data_max_mb == 2600.0 and s.builder_max_mb == 400.0


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
