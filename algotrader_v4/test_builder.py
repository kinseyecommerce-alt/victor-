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
RETEST = 455                          # builder/v3 dsr.lab_retest_trials: the lab's catalogue re-test at the Rs 10 lakh size
V3_N = SEED_N + RETEST                # = 5,434: N of a fresh registry under the current rule (builder/v3)
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
    assert G.grammar_sha256("bg3") == json.loads((HERE / "builder_rule_v3.json").read_text())["grammar"]["sha256"]
    assert G.grammar_sha256() == G.grammar_sha256("bg4") and G.GRAMMAR_VERSION == "bg4"
    assert G.grammar_sha256("bg4") == json.loads((HERE / "builder_rule_v4.json").read_text())["grammar"]["sha256"]


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
    lab = V3_N
    assert rec["n_before"] == lab
    assert s["n_disc"] == lab + 20 and s["n_proof"] == s["n_disc"] + s["final_looks"]
    for a in G.AGENTS:
        assert f"gate_{a}" in s["planted"]
    # a builder/v3 batch: every row carries its universe; the batch records what each universe resolved to
    assert s["rule_version"] == "builder/v4" and all(r["universe"] in G.UNIVERSES for r in s["rows"])
    assert s["universes"]["NIFTY100"]["n"] == 4 and s["universes"]["NIFTY200"]["added_complete"] == []
    assert sum(s["universes"]["counts"].values()) == 20
    out = b.ingest(rec["batch"], s)
    st = b.state()
    assert st["evaluated"] == SEED_EVAL + 20 and st["n_used"] == s["n_proof"]
    assert b.n_base(st) == 2552 + RETEST + 16 + SEED_EVAL + 20 + st["final_looks"]
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
    assert r2["n_before"] == V3_N + 20 + 2
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
    assert ns["RULE_SHA256"] == B.CURRENT_RULE_SHA256 == B.RULE_SHA256_V4


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
                assert ("builder_nightly" in ids) is on and "builder_weekly" not in ids, ids
                if on:                                  # builder/v3: one batch every night at 02:30 IST
                    c = [c for c in ps._sched.add_job.call_args_list if c.kwargs.get("id") == "builder_nightly"][0]
                    assert c.kwargs["hour"] == 2 and c.kwargs["minute"] == 30 and "day_of_week" not in c.kwargs
                    assert c.kwargs["misfire_grace_time"] <= 2 * 3600        # never started close to the open
                    assert "02:30" in B.load_rule()["schedule"]["when"]
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
    assert len(empty["grid"]) == 20 and empty["batches"] == [] and empty["n_used"] == V3_N
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
    assert st["n_used"] == SEED_N and b.n_base(st) == V3_N
    assert st["batches"]["0004"]["n_before"] == st["results"]["0003"]["n_used"]
    assert st["results"]["0003"]["proven"] == [PROVEN_SEEDED]
    # a fresh object on the same registry imports nothing again
    B.Builder(b.root).ensure_genesis()
    B.Builder(b.root).ensure_seed()
    assert [r["kind"] for r in b.registry.records()] == SEED_KINDS
    st_api = B.api_status(b.root)
    assert st_api["n_used"] == V3_N and {r["status"] for r in st_api["batches"]} == {"seeded"}
    assert len(st_api["batches"]) == 6 and st_api["seed"]["imported"] is True and len(st_api["seeds"]) == 6
    assert st_api["promoted"] == 1 and st_api["promoted_unregistered"] == 1
    assert B.api_batch("0001", b.root)["candidates"] == []          # the seed carries keys, not the sandbox's rows
    a3 = B.api_batch("0003", b.root)
    assert a3["status"] == "seeded" and a3["seed"]["proven"] == [PROVEN_SEEDED]
    assert a3["seed"]["file"] == B.SEED2_FILE.name and a3["cell_universes"]
    # before the registry exists the API already shows N after every seed
    assert B.api_status(os.path.join(_TMP, "seed_idem_none"))["n_used"] == V3_N


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
    assert rec["batch"] == FIRST and rec["n_before"] == V3_N
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


def _nse_window_keys(bd, upto) -> list:
    import builder_data as BD
    from datetime import date, timedelta
    keys = [f"deals:{kind}:{tag}" for _s, _e, tag in bd.quarters(BD.NSE_START, upto) for kind in ("bulk", "block")]
    keys += [f"ca:{tag}" for _s, _e, tag in bd.quarters(BD.NSE_START, upto + timedelta(days=92))]
    keys += [f"mcwb:{tag}" for _s, _e, tag in bd.months(BD.NSE_START, upto)]
    keys += [f"ev:{feed}:{tag}" for feed in BD.FEEDS for _s, e, tag in bd.months(BD.EVENTS_FROM, upto)
             if feed != "ifr" or e >= date(2025, 4, 1)]
    return keys


def _mark_nse_complete(d: str) -> None:
    """Fixture: the data layer's state file as it is once its NSE archive backfill (2012 ->) is complete (every
    calendar day final, every feed window fetched). The synthetic tree itself has no state file."""
    import builder_data as BD
    from datetime import date, timedelta
    bd = BD.Builder(d)
    st = bd.state()
    upto = date.today() + timedelta(days=2)
    k = BD.NSE_START
    while k <= upto:
        st["days"][str(BD.ymd(k))] = {"cash": "no_session"}
        k += timedelta(days=1)
    for key in _nse_window_keys(bd, upto):
        st["windows"][key] = {"status": "ok", "at": 99991231}
    bd.save(st)


def _data_copy(name: str, nse_complete: bool = True) -> str:
    d = os.path.join(_TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(syn_root(), d)
    if nse_complete:
        _mark_nse_complete(d)
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
    v2 = B.Builder(root, B.load_rule(B.RULE_FILE_V2))           # the builder/v2 deploy
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
    B.Builder(root, B.load_rule(B.RULE_FILE_V2)).ensure_genesis()        # idempotent
    assert [r["kind"] for r in v2.registry.records()].count("rule_upgrade") == 1
    assert B.api_status(root)["n_used"] >= up["n_used_before"]
    # a fresh registry started under v2 (genesis + seeds, no upgrade record)
    fresh_root = os.path.join(_TMP, "fresh_v2")
    shutil.rmtree(fresh_root, ignore_errors=True)
    fresh = B.Builder(fresh_root, B.load_rule(B.RULE_FILE_V2))
    fresh.ensure_genesis()
    assert [r["kind"] for r in fresh.registry.records()] == SEED_KINDS
    assert fresh.state()["genesis"]["grammar_version"] == "bg2"


def test_rule_v2_pinned_and_v1_unchanged():
    r1, r2 = B.load_rule(B.RULE_FILE), B.load_rule(B.RULE_FILE_V2)
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
    assert st["rule_version"] == "builder/v4" and st["grammar_version"] == "bg4"
    assert [r["version"] for r in st["rules"]] == ["builder/v1", "builder/v2", "builder/v3", "builder/v4"]
    assert set(st["universes"]["choices"]) == {"NIFTY100", "NIFTY200", "TOP500", "INDEX"}
    row = [r for r in st["batches"] if r["id"] == rec["batch"]][0]
    assert row["rule_version"] == "builder/v4" and sum(row["universes"].values()) == 20
    seed_row = [r for r in st["batches"] if r["id"] == "0001"][0]
    assert seed_row["rule_version"] == "builder/v1" and sum(seed_row["universes"].values()) == 400
    assert all("universes" in g for g in st["grid"])
    assert sum(u["candidates"] for g in st["grid"] for u in g["universes"].values()) == SEED_EVAL + 20
    d = B.api_batch(rec["batch"], b.root)
    assert all(c["universe"] in G.UNIVERSES and c["universe_explicit"] for c in d["candidates"])
    assert d["rule_version"] == "builder/v4" and sum(sum(v.values()) for v in d["cell_universes"].values()) == 20


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
    assert s.builder_night_batches == 12


# ══════════════════════════════════════════════════════════════════════════
# builder/v3: the size regime "c10" (Rs 10 lakh capital per agent), grammar bg3, nightly batches
# ══════════════════════════════════════════════════════════════════════════
BG1_SHA = "2084ade2c07de51ab95700e54c200e3d60c76f97d0ee38eba0ce464f899589ea"
BG2_SHA = "8bbb31f2d0c3b911b5a1581be8b458159e74358792883f9db87cb40f4e41eef3"


def _pin(t) -> float:
    """The raw entry price of a trade tuple (its fill carries 2 bps of slippage)."""
    return t[5] / (1.0 + t[3] * BE.R.SLIP_BPS / 1e4)


def _sized_ok(t, notional: float) -> bool:
    n = notional / _pin(t)
    return n - 1.01 < t[9] < n + 0.01


def _manual_sig(E, c, n: int = 400):
    """Trigger cells of both directions on eligible cells (no market-type filter): a deterministic fixture."""
    es, et = E.eligible(c, 0, 20261109, mtype=None)
    k = np.arange(0, len(es), max(1, len(es) // n))
    sides = np.where(np.arange(len(k)) % 2 == 0, 1, -1)
    return E.sort_sig(c, es[k].tolist(), et[k].tolist(), sides.tolist(), np.linspace(1, 2, len(k)).tolist(),
                      [0] * len(k))


def test_rule_v3_pinned_and_only_size_grammar_schedule_changed():
    r2, r3 = B.load_rule(B.RULE_FILE_V2), B.load_rule(B.RULE_FILE_V3)
    assert r3["_sha256"] == B.RULE_SHA256_V3 == \
        hashlib.sha256((HERE / "builder_rule_v3.json").read_bytes()).hexdigest()
    assert B.CURRENT_RULE_SHA256 == B.RULE_SHA256_V4, "builder/v4 is current; v3 stays pinned and valid"
    assert r3["version"] == "builder/v3" and r3["supersedes"]["sha256"] == B.RULE_SHA256_V2
    assert [v for v, _f, _s in B.RULES] == ["builder/v1", "builder/v2", "builder/v3", "builder/v4"]
    # v1 / v2 files and both grammars are byte for byte what they were
    assert B.RULE_SHA256 == hashlib.sha256((HERE / "builder_rule.json").read_bytes()).hexdigest()
    assert B.RULE_SHA256_V2 == hashlib.sha256((HERE / "builder_rule_v2.json").read_bytes()).hexdigest()
    assert G.grammar_sha256("bg1") == BG1_SHA and G.grammar_sha256("bg2") == BG2_SHA
    # unchanged: gates, splits, controls, carry, validation, final, proven, freeze, fill model, fee schedule, DSR
    for k in ("splits", "discovery_gates", "carry", "validation", "final", "controls", "proven", "freeze"):
        assert r2[k] == r3[k], k
    for k in ("fill_model", "fees"):
        assert r2["costs"][k] == r3["costs"][k], k
    for k in ("min", "lab_trials_count", "development_evaluations", "formula", "V", "series"):
        assert r2["dsr"][k] == r3["dsr"][k], k
    assert r3["batch"]["per_cell"] == 20 and r3["batch"]["size"] == 400
    assert r3["grammar"]["universes"] == r2["grammar"]["universes"]
    assert r3["grammar"]["market_types"] == r2["grammar"]["market_types"]
    # changed: the size, the grammar version and the schedule
    assert r3["costs"]["size"] != r2["costs"]["size"] and "size_rs" not in r2["costs"]
    rs = r3["costs"]["size_rs"]
    assert rs == {"regime": "c10", "equity_mis": 500000, "future": 500000, "equity_cnc": 100000,
                  "options": {"premium_rs": 100000, "max_loss_rs": 100000, "naked_margin_rs": 500000,
                              "naked_margin_pct": 0.12, "max_units_per_leg": {"NIFTY": 1800, "BANKNIFTY": 600}}}
    assert {k: float(v) for k, v in rs.items() if k not in ("regime", "options")} == BE.SIZE_TABLES["c10"]
    assert rs["options"] == BO.OPTION_SIZES["c10"]
    from stratlab import options_vrp
    assert rs["options"]["naked_margin_pct"] == options_vrp.NAKED_MARGIN_PCT           # the lab's own estimate
    assert "IS their Rs 10 lakh result" in r3["costs"]["unchanged"] and "never re-drawn" in r3["costs"]["unchanged"]
    assert "IS its Rs 10 lakh result" in r3["grammar"]["size_regime"]["unchanged"]
    assert BE.SIZE_TABLES[""] == {"equity_mis": 100000.0, "equity_cnc": 100000.0, "future": 100000.0}
    assert r3["costs"]["capital"]["per_agent_rs"] == 1000000 and "NOT capped" in r3["costs"]["capital"]["limitation"]
    assert r3["grammar"]["version"] == "bg3" and r3["grammar"]["sha256"] == G.grammar_sha256("bg3")
    assert r3["grammar"]["v2"]["sha256"] == BG2_SHA and r3["grammar"]["v1"]["sha256"] == BG1_SHA
    assert r3["grammar"]["sizes"] == G.grammar_sizes("bg3") == G.grammar_sizes("bg2")
    for a, d in r3["grammar"]["size_regime"]["counts"].items():
        rows = G.grammar(a, "bg3")
        assert d == {"re_sized": sum(1 for x in rows if x[8]), "unchanged": sum(1 for x in rows if not x[8])}, a
    assert "02:30" in r3["schedule"]["when"] and "Saturday" in r2["schedule"]["when"]
    assert r3["schedule"]["data_gates"] == ["minute_universe", "nse_archives"] and "data_gates" not in r2["schedule"]
    assert r3["schedule"]["night"]["max_batches_default"] == 12 == settings.builder_night_batches
    assert r3["schedule"]["night"]["no_new_batch_from"] == "05:15 IST" and B.NIGHT_STOP_MIN == 5 * 60 + 15
    assert "BUILDER_NIGHT_BATCHES" in r3["schedule"]["when"] and "grammar_exhausted" in r3["schedule"]["when"]
    assert "NSE archive backfill" in r3["schedule"]["data"]
    assert "minute universe is complete" in r3["schedule"]["data"]
    # an edited v3 rule is refused; a rule whose size table is not the engine's is refused by the engine
    tmp = Path(_TMP) / "rule_v3_edit.json"
    tmp.write_text((HERE / "builder_rule_v3.json").read_text().replace('"equity_mis": 500000', '"equity_mis": 900000'))
    tmp2 = Path(_TMP) / "rule_v3_edit_opt.json"
    tmp2.write_text((HERE / "builder_rule_v3.json").read_text().replace('"premium_rs": 100000', '"premium_rs": 900000'))
    try:
        B.load_rule(tmp)
        raise AssertionError("an edited v3 rule must be refused")
    except B.RuleChanged:
        pass
    E = engine()
    E.apply_rule(r3)
    E.apply_rule(r2)
    E.apply_rule(B.load_rule(B.RULE_FILE))
    assert E.sizes == BE.SIZE_TABLES
    bad = json.loads(tmp.read_text())
    try:
        E.apply_rule(bad)
        raise AssertionError("a size table that is not the engine's must be refused")
    except BE.SizeMismatch:
        pass
    assert E.sizes == BE.SIZE_TABLES
    try:
        E.apply_rule(json.loads(tmp2.read_text()))
        raise AssertionError("an option size table that is not the engine's must be refused")
    except BE.SizeMismatch:
        pass
    assert E.option_sizes == BO.OPTION_SIZES
    assert B.rule_by_sha(B.RULE_SHA256_V3)["version"] == "builder/v3"
    assert B.rule_sizes(r3)["regime"] == "c10" and B.rule_sizes(r2)["regime"] == ""
    assert B.rule_sizes(r2)["per_trade_rs"]["equity_mis"] == 100000
    # before the first run (no registry yet) the API already shows the rule, its size and what is left to test
    api = B.api_status(os.path.join(_TMP, "never_ran"))
    assert api["rule_version"] == "builder/v4" and api["size"]["regime"] == "c10" and api["n_used"] == V3_N
    # bg4 = bg3's 26,244 sentences + 2,944 trailing-exit sentences (intraday and scalping x 368 x 4 market types)
    assert api["undrawn"]["total"] == 29188 == 26244 + 2944 and api["undrawn"]["undrawn"] == 25709 + 2944
    assert api["night_batches"] == 12
    assert api["undrawn"]["per_agent"]["intraday"] == {"total": 10880, "undrawn": 10880}     # all new
    assert api["undrawn"]["per_agent"]["options"] == {"total": 2304, "undrawn": 2304}        # option lots re-sized
    assert api["undrawn"]["per_agent"]["futures"]["undrawn"] < api["undrawn"]["per_agent"]["futures"]["total"]
    assert api["size"]["options"] == BO.OPTION_SIZES["c10"] and "IS their Rs 10 lakh result" in api["size"]["unchanged"]


def test_bg3_size_regime_classification():
    """Which sentences are re-sized: every MIS stock sentence, every stock sentence at futures costs (futures agent;
    swing sentences that can go short) and every options sentence (lots from the capital); swing long-only and
    index-futures sentences keep their bg2 identity."""
    for a in G.AGENTS:
        r2, r3 = G.grammar(a, "bg2"), G.grammar(a, "bg3")
        assert [x[:8] for x in r3] == r2, a                        # bg3 = bg2's rows + the size regime
        assert all(x[8] in ("", "c10") for x in r3)
    assert all(x[8] == "c10" for a in ("intraday", "scalping") for x in G.grammar(a, "bg3"))
    assert all(x[8] == "c10" for x in G.grammar("options", "bg3"))
    for x in G.grammar("futures", "bg3"):
        assert x[8] == ("c10" if x[0] == "STOCKS" else ""), x
    E = engine()
    seen = set()
    n_long_only = n_short = 0
    for c in G.cell_grammar("swing", "RISING", "bg3"):
        if (c.block, c.params) in seen:
            continue
        seen.add((c.block, c.params))
        sides = set(E.raw_signal(c)[2].tolist())
        if c.size == "":                                           # a twin: the engine must book only CNC longs
            assert sides <= {1}, (c.block, c.params, sides)
            assert all(BE.cost_kind(c, sd) == "equity_cnc" for sd in sides)
            n_long_only += 1
        else:
            assert G.BLOCK[c.block].side != "long"
            n_short += bool(sides & {-1})
    assert n_long_only >= 10 and n_short >= 5, (n_long_only, n_short)
    # planted carriers take the size regime of their own sentence under bg3; Rs 1 lakh under bg1 / bg2
    c3 = EV.planted_candidates(B.load_rule())
    assert c3["intraday"].size == c3["scalping"].size == "c10"
    assert c3["options"].size == "c10" and c3["swing"].size == c3["futures"].size == ""
    assert all(c.size == "" for c in EV.planted_candidates(B.load_rule(B.RULE_FILE_V2)).values())
    assert all(c.size == "" for c in EV.planted_candidates(B.load_rule(B.RULE_FILE)).values())


def test_bg3_twin_rule_and_new_keys():
    import dataclasses
    sized = G.Candidate("intraday", "RISING", "gap_fade", (("g", 0.01), ("mode", "fade_day")), "o0915",
                        ("time", "o1515"), "STOCKS", "mis", "", "NIFTY100", "c10")
    old = dataclasses.replace(sized, size="")                      # the same sentence at Rs 1 lakh (bg2)
    assert sized in G.cell_grammar("intraday", "RISING", "bg3") and old in G.cell_grammar("intraday", "RISING", "bg2")
    assert old not in G.cell_grammar("intraday", "RISING", "bg3")
    assert sized.version == "bg3" and old.version == "bg2" and sized.key != old.key
    assert json.loads(sized.key)[0] == "bg3" and json.loads(sized.key)[-2:] == ["NIFTY100", "c10"]
    assert sized.v1_key is None and old.v1_key is not None
    # a re-sized sentence is drawable although its Rs 1 lakh versions (bg2 key, bg1 twin) were drawn; once only
    assert G.never_drawn(sized, {old.key, old.v1_key})
    assert not G.never_drawn(sized, {sized.key})
    # JSON: a bg1 / bg2 candidate's JSON is unchanged (no "size"); a re-sized one round-trips with its size
    assert "size" not in old.to_json() and sized.to_json()["size"] == "c10"
    assert G.Candidate.from_json(sized.to_json()) == sized and G.Candidate.from_json(old.to_json()) == old
    assert "Rs 5 lakh per trade" in G.describe(sized) and "Rs 5 lakh" not in G.describe(old)
    sw = [c for c in G.cell_grammar("swing", "RISING", "bg3") if c.size][0]
    assert "CNC, Rs 1 lakh per trade" in G.describe(sw) and "stock futures (Rs 5 lakh per trade)" in G.describe(sw)
    # an unchanged-size sentence IS its bg2 sentence: same key; never drawn after its bg2 key or its bg1 twin
    twin = [c for c in G.cell_grammar("futures", "RISING", "bg3") if c.single][0]        # an index future: 1 lot
    assert twin.size == "" and twin.version == "bg2" and twin in G.cell_grammar("futures", "RISING", "bg2")
    # an options sentence is re-sized (its lots follow the capital): a new bg3 key, drawable again
    opt = G.cell_grammar("options", "RISING", "bg3")[0]
    opt_old = dataclasses.replace(opt, size="")
    assert opt.size == "c10" and opt.version == "bg3" and json.loads(opt.key)[0] == "bg3" and opt.v1_key is None
    assert opt_old in G.cell_grammar("options", "RISING", "bg2") and opt_old not in G.cell_grammar("options", "RISING", "bg3")
    assert G.never_drawn(opt, {opt_old.key, opt_old.v1_key}) and not G.never_drawn(opt, {opt.key})
    assert "lots from Rs 10 lakh capital" in G.describe(opt) and "lots from" not in G.describe(opt_old)
    assert not G.never_drawn(twin, {twin.key}) and not G.never_drawn(twin, {twin.v1_key})
    lo = [c for c in G.cell_grammar("swing", "RISING", "bg3") if c.size == "" and c.universe == "TOP500"][0]
    assert lo.v1_key and not G.never_drawn(lo, {lo.v1_key})
    # the draw: with EVERY bg1 and bg2 key drawn, a bg3 batch holds only re-sized sentences (new trials, new keys)
    drawn = {c.key for a in G.AGENTS for m in G.MARKET_TYPES for v in ("bg1", "bg2") for c in G.cell_grammar(a, m, v)}
    b = G.draw_batch("0007", drawn, version="bg3")
    assert b and all(c.size == "c10" and c.version == "bg3" and c.key not in drawn for c in b)
    cells = {}
    for c in b:
        cells[c.cell] = cells.get(c.cell, 0) + 1
    assert all(cells.get(f"{a}|{m}") == 20 for a in G.AGENTS for m in G.MARKET_TYPES), cells   # options too
    b2 = G.draw_batch("0008", drawn | {c.key for c in b}, version="bg3")
    assert not ({c.key for c in b2} & {c.key for c in b})
    # the seeds (bg1 keys) block only their unchanged twins: a re-sized seed sentence comes back as a new candidate
    seed = _all_seed_keys()
    b3 = G.draw_batch("0007", seed, version="bg3")
    assert len(b3) == 400 and all(G.never_drawn(c, seed) for c in b3)
    assert all(c.key not in seed and (c.v1_key is None or c.v1_key not in seed) for c in b3)
    assert G.draw_batch("0007", seed, version="bg3") == b3          # deterministic
    assert any(c.size == "c10" for c in b3) and any(c.size == "" for c in b3)


def test_v3_engine_trade_sizes_and_fee_per_rupee():
    import dataclasses
    from costs import round_trip_cost
    E = engine()
    hi = 20261109
    # MIS (intraday): Rs 5 lakh per trade under bg3, Rs 1 lakh for the same sentence under bg2
    mis = G.Candidate("intraday", "RISING", "gap_go", (("g", 0.01), ("mode", "go")), "o0920", ("time", "o1430"),
                      "STOCKS", "mis", "", "NIFTY100", "c10")
    mis_old = dataclasses.replace(mis, size="")
    sig = E.signals(mis, mtype=None)
    res = BE.run_jobs(E, [BE.Job("new", mis, sig, 0, hi, False), BE.Job("old", mis_old, sig, 0, hi, False)])
    new, old = res["new"]["trades"], res["old"]["trades"]
    assert len(new) == len(old) > 100
    assert all(_sized_ok(t, 500000.0) for t in new) and all(_sized_ok(t, 100000.0) for t in old)
    for a, b_ in zip(new, old):
        assert a[1:9] == b_[1:9]                                   # the same trade (symbol, days, side, fills)
        if a[9] >= 50:                                             # (whole shares: only the rounding differs)
            assert a[11] / (a[9] * a[5]) < b_[11] / (b_[9] * b_[5])        # the fee per rupee traded falls
    assert round_trip_cost(500000.0, 500000.0, "equity_mis") < 5 * round_trip_cost(100000.0, 100000.0, "equity_mis")
    assert round_trip_cost(500000.0, 500000.0, "future") < 5 * round_trip_cost(100000.0, 100000.0, "future")
    # scalping: MIS too
    sc = G.Candidate("scalping", "RISING", "gap_go", (("g", 0.01), ("mode", "go")), "o0920", ("time_min", 5),
                     "STOCKS", "mis", "", "NIFTY100", "c10")
    tr = BE.run_jobs(E, [BE.Job("x", sc, E.signals(sc, mtype=None), 0, hi, False)])["x"]["trades"]
    assert len(tr) > 100 and all(_sized_ok(t, 500000.0) for t in tr)
    # swing: delivery longs stay Rs 1 lakh (CNC), shorts are booked at futures costs: Rs 5 lakh under bg3
    sw = G.Candidate("swing", "RISING", "low_deliv", (), "next_open", ("hold", 3), "STOCKS", "cnc/fut", "", "TOP500",
                     "c10")
    assert sw in G.cell_grammar("swing", "RISING", "bg3")
    sw_old = dataclasses.replace(sw, size="")
    sg = _manual_sig(E, sw)
    res = BE.run_jobs(E, [BE.Job("new", sw, sg, 0, hi, False), BE.Job("old", sw_old, sg, 0, hi, False)])
    longs, shorts = [t for t in res["new"]["trades"] if t[3] > 0], [t for t in res["new"]["trades"] if t[3] < 0]
    assert len(longs) > 20 and len(shorts) > 20
    assert all(_sized_ok(t, 100000.0) for t in longs) and all(_sized_ok(t, 500000.0) for t in shorts)
    assert all(_sized_ok(t, 100000.0) for t in res["old"]["trades"])
    # the futures agent's stock sentences (futures costs): Rs 5 lakh both ways under bg3; index futures: 1 real lot
    fs = G.Candidate("futures", "RISING", "low_deliv", (), "next_open", ("hold", 2), "STOCKS", "fut", "", "NIFTY100",
                     "c10")
    assert fs in G.cell_grammar("futures", "RISING", "bg3")
    fs_old = dataclasses.replace(fs, size="")
    sg = _manual_sig(E, fs)
    res = BE.run_jobs(E, [BE.Job("new", fs, sg, 0, hi, False), BE.Job("old", fs_old, sg, 0, hi, False)])
    assert len(res["new"]["trades"]) > 40 and {t[3] for t in res["new"]["trades"]} == {1, -1}
    assert all(_sized_ok(t, 500000.0) for t in res["new"]["trades"])
    assert all(_sized_ok(t, 100000.0) for t in res["old"]["trades"])
    ix = [c for c in G.cell_grammar("futures", "RISING", "bg3") if c.instrument == "NIFTY" and c.block == "fii_flow"][0]
    assert ix.size == ""
    tr = BE.run_jobs(E, [BE.Job("x", ix, E.signals(ix, mtype=None), 0, hi, False)])["x"]["trades"]
    assert tr and {t[9] for t in tr} == {BE.sim.LOT_SIZES["NIFTY"]}
    # a bg1 candidate (no universe, no size): Rs 1 lakh, as ever
    v1 = G.Candidate("intraday", "RISING", "gap_go", (("g", 0.01), ("mode", "go")), "o0920", ("time", "o1430"),
                     "STOCKS", "mis", "")
    tr = BE.run_jobs(E, [BE.Job("x", v1, E.signals(v1, mtype=None), 0, hi, False)])["x"]["trades"]
    assert len(tr) > 100 and all(_sized_ok(t, 100000.0) for t in tr)


def test_v3_option_lots_three_formulas():
    cfg = BO.OPTION_SIZES["c10"]
    lot = 75
    # 1. long ATM: Rs 1 lakh of premium at risk -> floor(100000 / (premium x lot))
    n, b = BO.option_lots("long_atm", [("CE", 25000.0, 1)], [200.0], lot, "NIFTY", 25010.0, cfg)
    assert (n, b["rule"], b["per_lot_rs"]) == (6, "premium", 15000.0) and b["uncapped"] == 6
    assert BO.option_lots("long_atm", [("PE", 25000.0, 1)], [1400.0], lot, "NIFTY", None, cfg)[0] == 1     # never < 1
    #    debit spread: the NET premium paid per lot
    n, b = BO.option_lots("debit_spread", [("CE", 25000.0, 1), ("CE", 25500.0, -1)], [200.0, 80.0], lot, "NIFTY",
                          25010.0, cfg)
    assert (n, b["per_lot_rs"]) == (11, 9000.0)                              # 100000 // (120 x 75)
    # 2. iron condor: Rs 1 lakh of MAXIMUM LOSS -> (widest wing - net credit) x lot
    legs = [("CE", 26000.0, -1), ("PE", 24000.0, -1), ("CE", 26750.0, 1), ("PE", 23250.0, 1)]
    n, b = BO.option_lots("iron_condor", legs, [60.0, 70.0, 15.0, 20.0], lot, "NIFTY", 25000.0, cfg)
    assert b["rule"] == "max_loss" and b["per_lot_rs"] == (750.0 - 95.0) * 75 and n == 2          # 100000 // 49125
    from options_replay import Leg, _max_risk
    lab = _max_risk([Leg("CE", 26000.0, -1, 60.0), Leg("PE", 24000.0, -1, 70.0), Leg("CE", 26750.0, 1, 15.0),
                     Leg("PE", 23250.0, 1, 20.0)], 95.0, 75)
    assert lab == b["per_lot_rs"]                                            # the lab's own max-risk formula
    # 3. naked short strangle: Rs 5 lakh of margin, margin = 12 % of spot notional per lot (the lab's estimate)
    legs = [("CE", 26000.0, -1), ("PE", 24000.0, -1)]
    n, b = BO.option_lots("short_strangle", legs, [60.0, 70.0], lot, "NIFTY", 25000.0, cfg)
    assert b["rule"] == "naked_margin" and b["per_lot_rs"] == 0.12 * 25000.0 * 75 == 225000.0 and n == 2
    assert BO.option_lots("short_strangle", legs, [60.0, 70.0], lot, "NIFTY", None, cfg)[1]["per_lot_rs"] == 225000.0
    assert BO.option_lots("short_strangle", legs, [60.0, 70.0], 30, "BANKNIFTY", 55000.0, cfg)[0] == 2   # 198,000 a lot
    # the cap: one un-sliced order per leg (1,800 NIFTY / 600 BANKNIFTY units)
    n, b = BO.option_lots("long_atm", [("CE", 25000.0, 1)], [5.0], lot, "NIFTY", 25010.0, cfg)
    assert b["uncapped"] == 266 and b["cap"] == 24 and n == 24
    assert BO.option_lots("long_atm", [("CE", 55000.0, 1)], [5.0], 30, "BANKNIFTY", 55000.0, cfg)[0] == 20
    # only entry values are arguments: nothing later can be read (no look-ahead by construction)
    import inspect
    assert list(inspect.signature(BO.option_lots).parameters) == ["structure", "legs", "fills", "lot", "symbol",
                                                                   "spot", "cfg"]
    # the fee per lot falls as the lots rise (flat brokerage per order), through the existing leg-cost function
    from options_replay import leg_cost
    one, six = leg_cost(200.0, 260.0, 1, 75), leg_cost(200.0, 260.0, 1, 6 * 75)
    assert six / 6 < one and six > one


def test_v3_option_jobs_are_sized_at_entry_and_v1_v2_trade_one_lot():
    import dataclasses
    from stratlab import nsedata as N
    E = engine()
    cfg = BO.OPTION_SIZES["c10"]
    orig_position = N.Position
    seen = {}
    for structure in ("long_atm", "debit_spread", "iron_condor", "short_strangle"):
        pool = [c for c in G.cell_grammar("options", "RISING", "bg3")
                if c.structure == f"{structure}:weekly" and c.instrument == "NIFTY" and c.exit == ("opt_hold", 3)]
        cand = old = res = None
        for c in pool:                                  # a sentence that trades on the synthetic chain
            sig = E.signals(c, mtype=None)
            o = dataclasses.replace(c, size="")
            r = BE.run_jobs(E, [BE.Job("new", c, sig, 0, 20261109, False), BE.Job("old", o, sig, 0, 20261109, False)])
            if len(r["new"]["trades"]) >= 5:
                cand, old, res = c, o, r
                break
        assert cand is not None, structure
        new, was, lots = res["new"]["trades"], res["old"]["trades"], res["new"]["lots"]
        assert len(new) == len(was) == len(lots) and res["old"]["lots"] == []          # v2 sentence: 1 lot, no sizing
        by_day = {d: (n, b) for d, n, b in lots}
        n_multi = 0
        for a, b_ in zip(new, was):
            assert (a[1], a[2], a[6], a[9]) == (b_[1], b_[2], b_[6], b_[9])            # same position, same lot size
            n, basis = by_day[a[2]]
            # entry / exit values and the gross P&L are the 1-lot ones x lots (same fills: sizing changes nothing else)
            assert abs(a[5] - n * b_[5]) < 1e-2 * n and abs(a[10] - n * b_[10]) <= 0.02 * n + 0.01
            # the lots are the formula's, computed from the 1-lot ENTRY value only
            lot = a[9]
            if structure in ("long_atm", "debit_spread"):
                want = max(1, min(basis["cap"], int(cfg["premium_rs"] // (b_[5] * lot)))) if b_[5] > 0 else 1
                assert basis["rule"] == "premium" and abs(basis["per_lot_rs"] - b_[5] * lot) < 0.5
            elif structure == "iron_condor":
                want = max(1, min(basis["cap"], int(cfg["max_loss_rs"] // basis["per_lot_rs"]))) \
                    if basis["per_lot_rs"] > 0 else 1
                assert basis["rule"] == "max_loss"
            else:
                want = max(1, min(basis["cap"], int(cfg["naked_margin_rs"] // basis["per_lot_rs"])))
                assert basis["rule"] == "naked_margin"
            assert n == want and 1 <= n <= basis["cap"] == max(1, cfg["max_units_per_leg"]["NIFTY"] // lot)
            if n > 1:                                   # the fee per lot falls as the lots rise
                n_multi += 1
                assert a[11] / n < b_[11] and a[11] > b_[11]
        seen[structure] = (len(new), n_multi)
        assert all(t[9] == N.lot_size("NIFTY", t[2]) for t in new)
    assert sum(v[1] for v in seen.values()) > 0, seen                # (some positions held more than one lot)
    assert N.Position is orig_position                               # the simulator's class is restored after a job
    # a bg1 options candidate: 1 lot, the simulator untouched
    v1 = G.Candidate("options", "RISING", cand.block, cand.params, cand.entry, cand.exit, cand.instrument, "opt",
                     cand.structure)
    r1 = BE.run_jobs(E, [BE.Job("x", v1, E.signals(v1, mtype=None), 0, 20261109, False)])["x"]
    assert r1["lots"] == [] and [t[5] for t in r1["trades"]] == [t[5] for t in was]


class _SizeRecorder:
    """Records every (size regime, cost kind, Rs) the engine sizes a stock trade with."""
    def __init__(self):
        self.seen = set()
        self._orig = BE.trade_notional

    def __call__(self, c, kind, sizes=None):
        v = self._orig(c, kind, sizes)
        self.seen.add((c.size, kind, v))
        return v


def _evaluate_recording(root, bid):
    import builder_run
    rec = _SizeRecorder()
    with patch.object(BE, "trade_notional", rec):
        s = builder_run.evaluate(str(root), bid, "synthetic", syn_root(), jobs=1, log=lambda m: None)
    return s, rec.seen


def test_fresh_registry_under_v3_first_batch_0007_and_n_5434():
    b = new_builder("fresh_v3")
    b.ensure_genesis()
    assert [r["kind"] for r in b.registry.records()] == SEED_KINDS
    st = b.state()
    assert st["genesis"]["rule_version"] == "builder/v4" and st["genesis"]["grammar_version"] == "bg4"
    assert st["rules"] == [B.RULE_SHA256_V4] and st["upgrades"] == []
    assert st["n_used"] == SEED_N == 4979 and b.next_batch_id(st) == "0007"
    assert all(st["batches"][k].get("rule_sha256") == B.RULE_SHA256 for k in st["batches"])   # the seeds are v1
    rec = b.freeze(syn_source().snapshot(), per_cell=1)
    assert rec["batch"] == "0007" and rec["n_before"] == V3_N == 5434
    assert rec["rule_version"] == "builder/v4" and rec["generator"]["grammar_version"] == "bg4"
    doc = json.loads(b.candidates_path("0007").read_text())
    cands = [G.Candidate.from_json(r["cand"]) for r in doc["candidates"]]
    seed = _all_seed_keys()
    assert all(G.never_drawn(c, seed) for c in cands) and [c.key for c in cands] == [r["key"] for r in doc["candidates"]]
    assert all(c.size == "c10" for c in cands if c.agent in ("intraday", "scalping"))
    assert all(c.size == "c10" for c in cands if c.agent == "options")
    assert all(c.size == "" for c in cands if c.single and c.agent == "futures")
    # evaluated at the v3 sizes (the batch rule's): MIS Rs 5 lakh, nothing re-sized trades Rs 1 lakh MIS
    s, seen = _evaluate_recording(b.root, "0007")
    assert s["rule_version"] == "builder/v4" and s["n_disc"] == V3_N + 20
    assert ("c10", "equity_mis", 500000.0) in seen and not any(k == "equity_mis" and v != 500000.0 for _r, k, v in seen)
    assert all(v == BE.SIZE_TABLES[r][k] for r, k, v in seen)
    b.ingest("0007", s)
    st = b.state()
    assert st["evaluated"] == SEED_EVAL + 20 and b.n_base(st) == V3_N + 20 + st["final_looks"] - SEED_FINALS
    # the next batch counts every v3 candidate and FINAL look (one N; it never shrinks)
    r2 = b.freeze({"source": "test", "end": 20261109}, per_cell=1)
    assert r2["batch"] == "0008" and r2["n_before"] == b.n_base(st) >= V3_N + 20
    k1 = {r["key"] for r in doc["candidates"]}
    c2 = [G.Candidate.from_json(r["cand"]) for r in json.loads(b.candidates_path("0008").read_text())["candidates"]]
    assert all(G.never_drawn(c, seed | k1) for c in c2)
    # api: the size regime of the rule and of each batch
    api = B.api_status(b.root)
    assert api["rule_version"] == "builder/v4" and api["size"]["regime"] == "c10"
    assert api["size"]["per_trade_rs"]["equity_mis"] == 500000 and "02:30" in api["schedule"]
    rows = {r["id"]: r for r in api["batches"]}
    assert rows["0007"]["size"]["regime"] == "c10" and rows["0001"]["size"]["regime"] == ""
    assert rows["0001"]["size"]["per_trade_rs"]["equity_mis"] == 100000
    d = B.api_batch("0007", b.root)
    assert d["size"]["regime"] == "c10" and B.api_batch("0003", b.root)["size"]["regime"] == ""
    assert {c["size"] for c in d["candidates"]} == {"", "c10"}
    assert all(c["size"] == "c10" for c in d["candidates"] if c["id"].split("_")[1] in ("int", "sca"))


def test_upgrade_v1_to_v3_two_steps_and_pending_v1_batch_at_1_lakh():
    root = os.path.join(_TMP, "upgrade13")
    shutil.rmtree(root, ignore_errors=True)
    v1 = B.Builder(root, B.load_rule(B.RULE_FILE))
    r1 = v1.freeze(syn_source().snapshot(), per_cell=1)           # a builder/v1 batch left pending
    assert r1["batch"] == FIRST and r1["rule_version"] == "builder/v1"
    v3 = B.Builder(root)                                         # the deploy: the current rule is builder/v4
    assert v3.rule["version"] == "builder/v4"
    v3.ensure_genesis()
    st = v3.state()
    assert st["rules"] == [B.RULE_SHA256, B.RULE_SHA256_V2, B.RULE_SHA256_V3, B.RULE_SHA256_V4]
    ups = st["upgrades"]
    assert [(u["from_version"], u["to_version"], u["grammar_version"]) for u in ups] == \
        [("builder/v1", "builder/v2", "bg2"), ("builder/v2", "builder/v3", "bg3"), ("builder/v3", "builder/v4", "bg4")]
    assert [(u["from_sha256"], u["to_sha256"]) for u in ups] == \
        [(B.RULE_SHA256, B.RULE_SHA256_V2), (B.RULE_SHA256_V2, B.RULE_SHA256_V3), (B.RULE_SHA256_V3, B.RULE_SHA256_V4)]
    assert all(u["n_used_before"] == V3_N and u["evaluated_before"] == SEED_EVAL and u["batches_before"] == 7
               and u["final_looks_before"] == SEED_FINALS for u in ups)
    assert st["batches"][FIRST]["frozen_hash"] == r1["frozen_hash"]            # earlier records never rewritten
    B.Builder(root).ensure_genesis()                             # idempotent
    assert [r["kind"] for r in v3.registry.records()].count("rule_upgrade") == 3
    # the pending v1 batch is evaluated under builder/v1 at Rs 1 lakh per stock trade
    s, seen = _evaluate_recording(root, FIRST)
    assert s["rule_version"] == "builder/v1" and s["n_disc"] == SEED_N + 20
    assert seen and all(r == "" and v == 100000.0 for r, _k, v in seen), seen
    v3.ingest(FIRST, s)
    st = v3.state()
    # new batches are builder/v3 (bg3); N continues
    r2 = v3.freeze({"source": "test", "end": 20261109}, per_cell=1)
    assert r2["batch"] == SECOND and r2["rule_version"] == "builder/v4" and r2["generator"]["grammar_version"] == "bg4"
    assert r2["n_before"] == v3.n_base(st) == V3_N + 20 + st["final_looks"] - SEED_FINALS
    keys1 = {c["key"] for c in json.loads(v3.candidates_path(FIRST).read_text())["candidates"]}
    c2 = [G.Candidate.from_json(c["cand"]) for c in json.loads(v3.candidates_path(SECOND).read_text())["candidates"]]
    assert all(G.never_drawn(c, keys1 | _all_seed_keys()) for c in c2)
    for old in (B.RULE_FILE, B.RULE_FILE_V2):                    # the registry moved on
        try:
            B.Builder(root, B.load_rule(old)).freeze({"source": "test", "end": 1}, per_cell=1)
            raise AssertionError("an earlier rule must not freeze on a registry upgraded to v3")
        except B.RuleChanged:
            pass
    assert v3.registry.verify()["ok"]
    assert B.api_status(root)["n_used"] >= SEED_N + 20
    assert [x["version"] for x in B.api_status(root)["rule_chain"]] == ["builder/v1", "builder/v2", "builder/v3",
                                                                        "builder/v4"]


def test_upgrade_v2_to_v3_and_pending_v2_batch_at_1_lakh():
    root = os.path.join(_TMP, "upgrade23")
    shutil.rmtree(root, ignore_errors=True)
    v2 = B.Builder(root, B.load_rule(B.RULE_FILE_V2))
    r1 = v2.freeze(syn_source().snapshot(), per_cell=1)           # a builder/v2 batch (bg2) left pending
    assert r1["batch"] == FIRST and r1["rule_version"] == "builder/v2"
    assert v2.state()["rules"] == [B.RULE_SHA256_V2]
    v3 = B.Builder(root)
    v3.ensure_genesis()
    st = v3.state()
    assert st["rules"] == [B.RULE_SHA256_V2, B.RULE_SHA256_V3, B.RULE_SHA256_V4] and len(st["upgrades"]) == 2
    assert st["upgrades"][0]["from_version"] == "builder/v2" and st["upgrades"][0]["grammar_version"] == "bg3"
    assert st["upgrades"][1]["from_version"] == "builder/v3" and st["upgrades"][1]["grammar_version"] == "bg4"
    s, seen = _evaluate_recording(root, FIRST)                   # under its own rule, at its own size
    assert s["rule_version"] == "builder/v2" and seen and all(r == "" and v == 100000.0 for r, _k, v in seen)
    v3.ingest(FIRST, s)
    r2 = v3.freeze({"source": "test", "end": 20261109}, per_cell=1)
    assert r2["batch"] == SECOND and r2["rule_version"] == "builder/v4"
    keys1 = {c["key"] for c in json.loads(v3.candidates_path(FIRST).read_text())["candidates"]}
    c1 = [G.Candidate.from_json(c["cand"]) for c in json.loads(v3.candidates_path(FIRST).read_text())["candidates"]]
    c2 = [G.Candidate.from_json(c["cand"]) for c in json.loads(v3.candidates_path(SECOND).read_text())["candidates"]]
    assert all(G.never_drawn(c, keys1 | _all_seed_keys()) for c in c2)
    # no unchanged-size sentence of the v2 batch comes back; a re-sized one may (as a new candidate, a new key)
    twins = {(c.agent, c.mtype, c.block, c.params, c.entry, c.exit, c.instrument, c.structure, c.universe)
             for c in c1}
    for c in c2:
        if c.size == "":
            assert (c.agent, c.mtype, c.block, c.params, c.entry, c.exit, c.instrument, c.structure,
                    c.universe) not in twins
    assert v3.registry.verify()["ok"]


def test_forward_family_trades_the_candidates_own_size():
    """builder_forward: a promoted strategy trades the size of its own sentence (a v1 / v2 strategy Rs 1 lakh, a v3
    re-sized one Rs 5 lakh); the seeded proven strategy is a v1 sentence."""
    import dataclasses
    root = Path(_TMP) / "fwd_size"
    shutil.rmtree(root, ignore_errors=True)
    E = engine()
    sized = None
    for c in G.cell_grammar("intraday", "RISING", "bg3"):        # a sentence that trades in the forward window
        if c.universe == "NIFTY100" and c.exit[0] == "time" and len(E.signals(c)["s"]) > 200:
            tr = BE.run_jobs(E, [BE.Job("x", c, None, 20240102, 20261109, False)])["x"]["trades"]
            if len(tr) > 30:
                sized = c
                break
    assert sized is not None and sized.size == "c10"
    old = dataclasses.replace(sized, size="")
    fw = BF.BuilderForward(root)
    fw.register([{"id": "new", "batch": "0007", "cand": sized.to_json(), "reference": {}, "fingerprint": "a"},
                 {"id": "old", "batch": "0002", "cand": old.to_json(), "reference": {}, "fingerprint": "b"}], 20240101)
    out = BF.forward_trades(str(root), syn_root(), 20261109, "synthetic")
    new, was = out["trades"]["new"], out["trades"]["old"]
    assert len(new) == len(was) > 30
    assert all(_sized_ok(t, 500000.0) for t in new) and all(_sized_ok(t, 100000.0) for t in was)
    # the seeded proven strategy (builder/v1): no size regime in its sentence -> Rs 1 lakh, reference unchanged
    b = new_builder("fwd_size_seed")
    b.ensure_genesis()
    p = [x for x in b.state()["promoted"] if x["id"] == PROVEN_SEEDED][0]
    assert "size" not in p["cand"] and G.Candidate.from_json(p["cand"]).size == ""
    assert p["rule_sha256"] == B.RULE_SHA256 and B.sizes_by_sha(p["rule_sha256"])["regime"] == ""


# ══════════════════════════════════════════════════════════════════════════
# builder/v3: nightly batches until the grammar is exhausted; the NSE archive gate
# ══════════════════════════════════════════════════════════════════════════
def _fake_runner(calls):
    def runner(r, bid):
        calls.append(bid)
        doc = B.Builder(r).verify_frozen(bid)
        return {"candidates": len(doc["candidates"]), "discovery_passers": 0, "shuffle_passers": 0,
                "validation_tested": 0, "validation_passers": 0, "final_looks": 0, "final_confirmed": 0,
                "proven": [], "n_disc": 1, "n_proof": 1, "v": 0.0, "best_discovery_dsr": {}, "planted": {},
                "rows": [], "windows": {}}
    return runner


def test_nightly_job_runs_batches_up_to_the_budget():
    from datetime import datetime
    root = os.path.join(_TMP, "night_budget")
    shutil.rmtree(root, ignore_errors=True)
    calls = []
    snap = lambda: {"source": "test", "end": 1}                     # noqa: E731
    t = datetime(2026, 11, 17, 2, 30)
    with patch.object(settings, "builder_enabled", False):
        assert B.nightly_job(now=t, root=root)["skipped"].startswith("BUILDER_ENABLED")
        assert B.nightly_job_locked(now=t, root=root)["skipped"].startswith("BUILDER_ENABLED")
        assert not os.path.exists(root)
    with patch.object(settings, "builder_enabled", True), patch.object(settings, "builder_night_batches", 3):
        fresh = B.Builder(root)
        fresh.ensure_genesis()
        before = B.grammar_progress(fresh)
        assert before["grammar"] == "bg4" and before["total"] == 4 * sum(G.grammar_sizes("bg4").values()) == 29188
        assert set(before["per_agent"]) == set(G.AGENTS) and 0 < before["undrawn"] < before["total"]
        out = B.nightly_job(now=t, root=root, runner=_fake_runner(calls), snapshot_fn=snap)
        night = out["night"]
        assert out["status"] == "complete" and night["batches"] == [FIRST, SECOND, THIRD] == calls, out
        assert night["batches_run"] == 3 and night["stopped"].startswith("night budget (3")
        assert sorted(night["results"]) == [FIRST, SECOND, THIRD] and isinstance(night["proven"], list)
        assert night["n_used"] >= V3_N and all("discovery_passers" in r for r in night["results"].values())
        st = B.Builder(root).state()
        # each batch was frozen after the one before was ingested: N grows batch by batch, no key twice
        assert [st["batches"][b_]["n_before"] for b_ in (FIRST, SECOND, THIRD)] == [V3_N, V3_N + 400, V3_N + 800]
        assert st["evaluated"] == SEED_EVAL + 1200 and sorted(st["results"])[-3:] == [FIRST, SECOND, THIRD]
        keys = [c["key"] for b_ in (FIRST, SECOND, THIRD)
                for c in json.loads(B.Builder(root).candidates_path(b_).read_text())["candidates"]]
        assert len(keys) == len(set(keys)) == 1200
        # progress to "all tested" is visible: undrawn sentences, total and per agent, in the night record and the API
        assert night["undrawn"]["undrawn"] == before["undrawn"] - 1200
        api = B.api_status(root)
        assert api["undrawn"] == night["undrawn"] and api["night_batches"] == 3
        assert sum(v["undrawn"] for v in api["undrawn"]["per_agent"].values()) == api["undrawn"]["undrawn"]
        assert api["last_run"]["night"]["batches_run"] == 3 and api["last_run"]["night"]["batches"] == night["batches"]
        # the next night continues where this one stopped
        out = B.nightly_job(now=t, root=root, runner=_fake_runner(calls), snapshot_fn=snap, max_batches=1)
        assert out["night"]["batches"] == ["0010"] and out["night"]["undrawn"]["undrawn"] == before["undrawn"] - 1600


def test_nightly_job_stops_at_0515_market_hours_and_on_a_batch_that_did_not_complete():
    from datetime import datetime
    assert B.night_window_open(datetime(2026, 11, 17, 2, 30)) and B.night_window_open(datetime(2026, 11, 17, 5, 14))
    assert not B.night_window_open(datetime(2026, 11, 17, 5, 15)) and not B.night_window_open(datetime(2026, 11, 17, 10, 0))
    assert not B.night_window_open(datetime(2026, 11, 17, 15, 45)) and B.night_window_open(datetime(2026, 11, 17, 15, 46))
    seen = []

    def one_batch(**kw):
        seen.append(kw)
        return {"status": "complete", "batch": f"b{len(seen)}"}
    times = iter([datetime(2026, 11, 17, 2, 30), datetime(2026, 11, 17, 4, 0), datetime(2026, 11, 17, 5, 14),
                  datetime(2026, 11, 17, 5, 15), datetime(2026, 11, 17, 6, 0)])
    root = os.path.join(_TMP, "night_clock")
    with patch.object(settings, "builder_enabled", True), patch.object(B, "weekly_job", one_batch):
        out = B.nightly_job(root=root, clock=lambda: next(times), max_batches=12)
        assert out["night"]["batches"] == ["b1", "b2", "b3"] and "05:15" in out["night"]["stopped"], out
        # an evaluation's wall-clock limit ends at 08:30 IST (never into the 09:00 market-hours guard)
        assert [k["time_limit_s"] for k in seen] == [6 * 3600, 4 * 3600 + 1800, 3 * 3600 + 16 * 60]
        assert [k["now"].hour for k in seen] == [2, 4, 5]
        seen.clear()
        out = B.nightly_job(now=datetime(2026, 11, 17, 10, 0), root=root, max_batches=12)      # a trading morning
        assert out["night"]["batches"] == [] and seen == [] and "window closed" in out["night"]["stopped"]
        out = B.nightly_job(now=datetime(2026, 11, 17, 20, 0), root=root, max_batches=2)       # an evening run
        assert out["night"]["batches_run"] == 2 and [k["time_limit_s"] for k in seen] == [None, None]
        assert B.nightly_job(root=root, now=datetime(2026, 11, 17, 2, 30), max_batches=0)["night"]["batches"] == []
    # (c) a batch that does not complete ends the night; the pending batch is resumed first the next night
    root = os.path.join(_TMP, "night_fail")
    shutil.rmtree(root, ignore_errors=True)
    calls = []

    def crash(r, bid):
        calls.append(bid)
        raise B.RunFailed("time limit exceeded")
    snap = lambda: {"source": "test", "end": 1}                     # noqa: E731
    t = datetime(2026, 11, 17, 2, 30)
    with patch.object(settings, "builder_enabled", True):
        out = B.nightly_job(now=t, root=root, runner=crash, snapshot_fn=snap, max_batches=5)
        assert out["status"] == "error" and out["night"]["stopped"] == "error" and out["night"]["batches"] == []
        assert calls == [FIRST] and FIRST not in B.Builder(root).state()["results"]
        out = B.nightly_job(now=t, root=root, runner=_fake_runner(calls), snapshot_fn=snap, max_batches=2)
        assert out["night"]["batches"] == [FIRST, SECOND] and calls == [FIRST, FIRST, SECOND]
    # the scheduler's job is the nightly run under the data layer's lock
    import asyncio
    import platform_scheduler as PS
    for res in ({"night": {}}, {"skipped": "BUILDER_ENABLED is off or not PAPER"},
                {"status": "waiting_data", "reason": "x", "night": {"batches_run": 0, "stopped": "waiting_data"}}):
        with patch.object(B, "nightly_job_locked", return_value=res) as m:
            asyncio.run(PS.PlatformScheduler()._builder_nightly())
            assert m.call_count == 1


def test_nightly_job_stops_when_the_grammar_is_exhausted():
    from datetime import datetime
    every = [c for a in G.AGENTS for m in G.MARKET_TYPES for c in G.cell_grammar(a, m, "bg4")]
    cell = [c for c in every if c.cell == "intraday|RISING"]
    seed = _all_seed_keys()                                      # (an option sentence the seeds did not draw)
    remain = cell[::len(cell) // 25][:25] + [c for c in every if c.cell == "options|FALLING"
                                             and G.never_drawn(c, seed)][:3]
    left = {c.key for c in remain} | {c.v1_key for c in remain if c.v1_key}
    base = ({c.key for c in every} | {c.v1_key for c in every if c.v1_key}) - left      # everything else is drawn
    orig = B.Builder.drawn_keys
    root = os.path.join(_TMP, "night_exhaust")
    shutil.rmtree(root, ignore_errors=True)
    calls = []
    snap = lambda: {"source": "test", "end": 1}                     # noqa: E731
    t = datetime(2026, 11, 17, 2, 30)
    with patch.object(settings, "builder_enabled", True), \
            patch.object(B.Builder, "drawn_keys", lambda self, st=None: orig(self, st) | base):
        b = B.Builder(root)
        b.ensure_genesis()
        assert B.grammar_progress(b)["undrawn"] == 28
        assert B.grammar_progress(b)["per_agent"]["intraday"]["undrawn"] == 25
        out = B.nightly_job(now=t, root=root, runner=_fake_runner(calls), snapshot_fn=snap, max_batches=12)
        night = out["night"]
        assert out["status"] == "grammar_exhausted" == night["stopped"], out
        assert 2 <= night["batches_run"] < 12 and night["undrawn"]["undrawn"] == 0
        assert all(v["undrawn"] == 0 for v in night["undrawn"]["per_agent"].values())
        drawn = [c["key"] for bid in night["batches"]
                 for c in json.loads(B.Builder(root).candidates_path(bid).read_text())["candidates"]]
        assert sorted(drawn) == sorted(c.key for c in remain)          # every remaining sentence once, nothing else
        n_batches = len(B.Builder(root).state()["batches"])
        # nothing is frozen any more: one more night (or one more single batch) changes nothing
        again = B.nightly_job(now=t, root=root, runner=_fake_runner(calls), snapshot_fn=snap, max_batches=12)
        assert again["night"]["batches"] == [] and again["status"] == "grammar_exhausted"
        one = B.weekly_job(now=t, root=root, runner=_fake_runner(calls), snapshot_fn=snap)
        assert one["status"] == "grammar_exhausted" and "batch" not in one and one["undrawn"]["undrawn"] == 0
        assert len(B.Builder(root).state()["batches"]) == n_batches
        assert B.api_status(root)["undrawn"]["undrawn"] == 0


def test_new_v3_batches_wait_for_the_nse_archive_backfill():
    import builder_data as BD
    from datetime import date, timedelta
    d = _data_copy("data_nse", nse_complete=False)
    with patch.dict(os.environ, {"BUILDER_DATA_DIR": d}):
        b = new_builder("wait_nse")
        ns = B.nse_archive_status()
        assert ns["complete"] is False and "not started" in ns["reason"]
        assert "not started" in (B.data_wait_reason(b, "server") or "")
        # the gate belongs to builder/v3 (schedule.data_gates): a v2 / v1 rule does not wait for it
        assert B.data_wait_reason(B.Builder(b.root, B.load_rule(B.RULE_FILE_V2)), "server") is None
        assert B.data_wait_reason(B.Builder(b.root, B.load_rule(B.RULE_FILE)), "server") is None
        assert B.data_wait_reason(b, "lab") is None
        with patch.object(settings, "builder_enabled", True):
            s = B.weekly_job(now=_sat(), root=b.root, source="server")
            assert s["status"] == "waiting_data" and "batch" not in s and "NSE archive" in s["reason"], s
            n = B.nightly_job(now=_sat().replace(hour=2, minute=30), root=b.root, source="server", max_batches=3)
            assert n["night"]["stopped"] == "waiting_data" and n["night"]["batches"] == []
        assert sorted(B.Builder(b.root).state()["batches"]) == ["0001", "0002", "0003", "0004", "0005", "0006"]
        _mark_nse_complete(d)
        bd = BD.Builder(d)
        raw = bd.state_path.read_bytes()
        ns = B.nse_archive_status()
        assert ns == {**ns, "complete": True, "days_pending": 0, "windows_pending": 0, "windows_error": 0,
                      "from": 20120101, "reason": None}
        assert B.data_wait_reason(b, "server") is None

        def edit(fn):
            st = json.loads(raw)
            fn(st)
            bd.state_path.write_text(json.dumps(st))
            return B.nse_archive_status()
        # a calendar day the backfill has not reached
        ns = edit(lambda st: st["days"].pop("20150302"))
        assert not ns["complete"] and ns["days_pending"] == 1 and "20150302" in ns["reason"]
        assert "20150302" in B.data_wait_reason(b, "server")
        # a session whose F&O archive ended in an error (retried by the data layer): not final -> wait
        full = {"cash": "ok", "idx": "ok", "fo": "ok", "poi": "ok", "pvol": "ok", "ban": "ok", "fii": "n/a"}
        ns = edit(lambda st: st["days"].update({"20180105": {**full, "fo": "error:error"}}))
        assert not ns["complete"] and ns["days_pending"] == 1 and "error:error" in ns["reason"]
        # ... a file NSE does not have (an old 404) is final; a session with every set final is complete
        assert edit(lambda st: st["days"].update({"20180105": {**full, "fo": "missing", "ban": "stale"}}))["complete"]
        assert not edit(lambda st: st["days"].update({"20180105": {"cash": "ok", "idx": "ok"}}))["complete"]
        assert not edit(lambda st: st["days"].update({"20180105": {"cash": "error:refused"}}))["complete"]
        # the newest days are re-read by the data layer on every run: they never block
        recent = str(BD.ymd(date.today() - timedelta(days=2)))
        assert edit(lambda st: st["days"].pop(recent, None))["complete"]
        old_edge = str(BD.ymd(date.today() - timedelta(days=B.NSE_GATE_LAG_DAYS + 2)))
        assert not edit(lambda st: st["days"].pop(old_edge))["complete"]
        # feed windows: one never fetched blocks; one whose last answer was an error is reported, not waited for
        ns = edit(lambda st: st["windows"].pop("deals:bulk:2015Q1"))
        assert not ns["complete"] and ns["windows_pending"] == 1 and "deals:bulk:2015Q1" in ns["reason"]
        ns = edit(lambda st: st["windows"].update({"mcwb:201203": {"status": "error:not zip", "at": 20261101}}))
        assert ns["complete"] and ns["windows_error"] == 1 and ns["windows_pending"] == 0
        ns = edit(lambda st: st["windows"].update({"ev:fr:201405": {"status": "missing", "at": 20261101}}))
        assert ns["complete"] and ns["windows_error"] == 0
        # read-only: the data layer's state file is never written by the gate
        bd.state_path.write_bytes(raw)
        m0 = bd.state_path.stat().st_mtime_ns
        assert B.nse_archive_status()["complete"] and bd.state_path.stat().st_mtime_ns == m0
        assert bd.state_path.read_bytes() == raw


def test_server_cash_and_futures_readers_match_the_one_shot_reference_at_any_chunk_size():
    """The memory-light readers (chunked cash table, per-file futures rows) return exactly what the one-shot readers
    returned - on dashed dates, padded lower-case names, duplicate rows and a junk number - and the daily panel does
    not depend on where the chunks are cut (the snapshot process died at its address-space limit on the one-shot
    read at full scale)."""
    import glob
    import pandas as pd
    root = os.path.join(_TMP, "readers")
    SP.make_synthetic_tree(root, start=20250102, end=20261109, n_stocks=5, seed=9)
    p = os.path.join(root, "nse", "delivery.csv.gz")
    df = pd.read_csv(p, dtype=str)
    df.loc[df.index[::7], "date"] = df.loc[df.index[::7], "date"].map(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")
    df.loc[df.index[::5], "symbol"] = df.loc[df.index[::5], "symbol"].map(lambda x: " " + x.lower() + " ")
    df = pd.concat([df, df.iloc[[3, 3, 10, len(df) - 1]]], ignore_index=True)   # duplicates, one far from its twin
    df.loc[20, "volume"] = "n/a"
    df.to_csv(p, index=False, compression="gzip")

    def reference():                                   # the reader before the rewrite, verbatim
        d = pd.read_csv(p, dtype={"symbol": str, "bars_symbol": str},
                        usecols=["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "deliv_pct"])
        d["date"] = d["date"].astype(str).str.replace("-", "").astype(np.int64)
        for k in ("open", "high", "low", "close", "prev_close", "volume", "deliv_pct"):
            d[k] = pd.to_numeric(d[k], errors="coerce")
        d["symbol"] = d["symbol"].str.strip().str.upper()
        return d.drop_duplicates(["symbol", "date"], keep="last")

    def norm(d):
        return d.sort_values(["symbol", "date"]).reset_index(drop=True)

    def panel(chunk, tag):
        old_chunk, old_cache = SP.DELIVERY_CHUNK, os.environ["BUILDER_CACHE_DIR"]
        SP.DELIVERY_CHUNK = chunk
        os.environ["BUILDER_CACHE_DIR"] = os.path.join(_TMP, "cache_readers_" + tag)
        try:
            src = SP.ServerSource(root)
            return norm(src._delivery_frame()), src.daily_panel(), src._fut_parts()
        finally:
            SP.DELIVERY_CHUNK, os.environ["BUILDER_CACHE_DIR"] = old_chunk, old_cache
    ref = norm(reference())
    assert len(ref) == len(df) - 4 and ref["volume"].isna().sum() == 1
    frames = {}
    for chunk, tag in ((SP.DELIVERY_CHUNK, "big"), (7, "seven"), (1, "one")):
        a, (P, extra), fut = panel(chunk, tag)
        pd.testing.assert_frame_equal(ref, a[list(ref.columns)], check_dtype=False)
        frames[tag] = (P, extra, fut)
    P0, e0, f0 = frames["big"]
    assert len(P0.syms) == 5 and P0.c.shape[1] == len(set(ref["date"]))
    for tag in ("seven", "one"):
        P1, e1, _f = frames[tag]
        assert list(P0.syms) == list(P1.syms) and np.array_equal(P0.cal, P1.cal)
        for k in ("o", "h", "l", "c", "v", "member", "fo"):
            x, y = np.asarray(getattr(P0, k)), np.asarray(getattr(P1, k))
            assert x.shape == y.shape and np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")), (tag, k)
        for k in ("dp", "doi"):
            assert np.array_equal(e0[k], e1[k], equal_nan=True), (tag, k)
    # official closes land in the right cell (unadjusted data: no corporate action in the synthetic tree)
    row = ref[(ref["symbol"] == P0.syms[2]) & (ref["date"] == int(P0.cal[40]))].iloc[0]
    assert abs(float(P0.c[2, 40]) - float(row["close"])) < 1e-9 and abs(float(P0.v[2, 40]) - float(row["volume"])) < 1e-6
    # futures rows: the per-file numeric reader == the one-shot string reader
    parts = [pd.read_csv(q, usecols=["date", "symbol", "expiry", "instrument", "oi"], dtype=str)
             for q in sorted(glob.glob(os.path.join(root, "nse", "fo_parts", "fut", "*.csv.gz")))]
    r = pd.concat(parts, ignore_index=True)
    assert len(r) == len(f0) and len(f0) > 1000
    assert np.array_equal(r["date"].str.replace("-", "").astype(np.int64).to_numpy(), f0["day"].to_numpy())
    assert np.array_equal(r["expiry"].str.replace("-", "").astype(np.int64).to_numpy(), f0["exp"].to_numpy())
    assert np.array_equal(pd.to_numeric(r["oi"], errors="coerce").fillna(0.0).to_numpy(), f0["oi"].to_numpy())
    assert list(r["symbol"].str.strip().str.upper()) == list(f0["symbol"]) and list(r["instrument"]) == list(f0["instrument"])


def test_snapshot_process_gets_its_own_memory_limit_and_single_threaded_libraries():
    """The snapshot process takes BUILDER_SNAPSHOT_MEM_MB (not the 4,096 MB default it died at) and both child
    processes run one thread per numeric library with two malloc arenas (RLIMIT_AS counts their reservations)."""
    import subprocess
    assert settings.builder_snapshot_mem_mb == 6144 and B.snapshot_mem_mb() == 6144
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert B.PROCESS_ENV[k] == "1"
    assert B.PROCESS_ENV["MALLOC_ARENA_MAX"] == "2"
    seen = {}

    class P:
        returncode, stderr = 1, "x" * 3000 + "MemoryError: the end"

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return P()
    real, old = subprocess.run, settings.builder_snapshot_mem_mb
    subprocess.run = fake_run
    try:
        settings.builder_snapshot_mem_mb = 5000
        root = Path(_TMP) / "snaproot"
        root.mkdir(exist_ok=True)
        try:
            B.snapshot_process(root, "server")
            raise AssertionError("a failed snapshot process must raise")
        except B.RunFailed as exc:
            assert "MemoryError: the end" in str(exc) and len(str(exc)) > 1000     # enough of the traceback is kept
        assert seen["cmd"][seen["cmd"].index("--mem-mb") + 1] == "5000"
        assert all(seen["env"][k] == v for k, v in B.PROCESS_ENV.items())
        try:
            B.snapshot_process(root, "server", mem_mb=3000)
        except B.RunFailed:
            pass
        assert seen["cmd"][seen["cmd"].index("--mem-mb") + 1] == "3000"
    finally:
        subprocess.run, settings.builder_snapshot_mem_mb = real, old


def _bars(rows, t0=555):
    """rows = [(o, h, l, c)] one per minute from t0 (09:15)."""
    tt = [t0 + i for i in range(len(rows))]
    o, h, l, c = ([r[i] for r in rows] for i in range(4))
    return tt, o, h, l, c, [1000] * len(rows)


def test_v4_trailing_stop_walk_long_and_short():
    day = 20250102
    # long, trail 1%: entry 100 at the 09:15 open; best rises to 110 (stop 108.9); bar 4 touches 108.5 -> filled 108.9
    tt, o, h, l, c, v = _bars([(100, 101, 99.5, 100.5), (100.5, 105, 100, 104.5), (104.5, 110, 104, 109.5),
                               (109.5, 109.8, 108.5, 109)])
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail", 0.01), day)
    assert r[0] == 100 and abs(r[1] - 108.9) < 1e-9 and r[3] == "stop", r
    # the stop is checked before it moves inside a bar: bar 1's low 99.5 is above the first stop (99), no exit
    assert r[2] != BE._hm(555)
    # a bar that opens through the stop fills at its open (gap), not at the stop
    tt, o, h, l, c, v = _bars([(100, 110, 100, 109), (105, 106, 104, 105)])
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail", 0.01), day)
    assert r[1] == 105 and r[3] == "stop", r
    # short, trail 1%: entry 100, best falls to 90 (stop 90.9); a high of 91 touches it -> 90.9
    tt, o, h, l, c, v = _bars([(100, 100.5, 95, 95.5), (95.5, 95.9, 90, 90.5), (90.5, 91, 90.2, 90.8)])
    r = BE.walk_session(tt, o, h, l, c, v, -1, (555, "trail", 0.01), day)
    assert abs(r[1] - 90.9) < 1e-9 and r[3] == "stop", r
    # no touch all day: out at the 15:15 square-off
    tt, o, h, l, c, v = _bars([(100, 100.2, 99.8, 100)] * (BE.SQUARE_MIN - 555 + 5))
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail", 0.01), day)
    assert r[3] == "time" and r[2] == BE._hm(BE.SQUARE_MIN), r


def test_v4_breakeven_trail_arms_only_after_the_move_and_never_below_entry():
    day = 20250102
    # trail_be 1%: before +1% the stop stays at 99 (initial); price goes to 100.8 then 99.2 -> no exit (99 not touched)
    rows = [(100, 100.8, 99.6, 100.5), (100.5, 100.6, 99.2, 99.5), (99.5, 99.7, 98.9, 99.0)]
    tt, o, h, l, c, v = _bars(rows)
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail_be", 0.01), day)
    assert abs(r[1] - 99.0) < 1e-9 and r[3] == "stop", r              # the initial stop, not a trailed one
    # plain trail on the same bars trails from bar 1: best 100.8 -> stop 99.792, touched by 99.2 in bar 2
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail", 0.01), day)
    assert abs(r[1] - 100.8 * 0.99) < 1e-9, r
    # once +1% is reached (best 101), the stop is max(best*0.99, entry) = 100: a dip to 99.9 exits at entry
    tt, o, h, l, c, v = _bars([(100, 101, 99.8, 100.8), (100.8, 100.9, 99.9, 100.1)])
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail_be", 0.01), day)
    assert r[1] == 100 and r[3] == "stop", r


def test_v4_scalping_trail_has_a_time_limit():
    day = 20250102
    tt, o, h, l, c, v = _bars([(100, 100.1, 99.9, 100)] * 60)
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail_min", 0.005, 30), day)
    assert r[3] == "time" and r[2] == BE._hm(555 + 30), r
    r = BE.walk_session(tt, o, h, l, c, v, +1, (555, "trail_be_min", 0.005, 30), day)
    assert r[3] == "time" and r[2] == BE._hm(555 + 30), r


def test_bg4_is_bg3_plus_trailing_sentences_with_new_keys():
    assert G.GRAMMAR_VERSION == "bg4" and G.grammar_sha256("bg3") == \
        json.loads((HERE / "builder_rule_v3.json").read_text())["grammar"]["sha256"]
    for a in G.AGENTS:
        g3, g4 = G.grammar(a, "bg3"), G.grammar(a, "bg4")
        assert g4[:len(g3)] == g3, a                                    # bg3 byte for byte, in order
        extra = g4[len(g3):]
        if a not in G.TRAIL_EXITS:
            assert extra == [], a
            continue
        assert extra and all(r[5] in G.TRAIL_EXITS[a] for r in extra)
        assert all(r[0] == "STOCKS" and G.BLOCK[r[2]].grid == "minute" for r in extra)
        assert len(extra) == len(set(map(repr, extra))) and len(extra) % len(G.TRAIL_EXITS[a]) == 0
    assert G.grammar_sizes("bg4") == {"intraday": 2720, "scalping": 1196, "swing": 2055, "options": 576,
                                      "futures": 750}
    tr = [c for c in G.cell_grammar("intraday", "RISING", "bg4") if c.exit[0] in G.TRAIL_KINDS]
    assert tr and all(c.version == "bg4" and c.v1_key is None and json.loads(c.key)[0] == "bg4" for c in tr)
    c = tr[0]
    assert G.Candidate.from_json(c.to_json()) == c and G.never_drawn(c, set()) and not G.never_drawn(c, {c.key})
    assert "trail" in G.describe(c).lower()
    # a bg3 sentence keeps its key under bg4 (no re-test of anything already drawn)
    base = [c for c in G.cell_grammar("intraday", "RISING", "bg4") if c.exit[0] not in G.TRAIL_KINDS]
    assert base == G.cell_grammar("intraday", "RISING", "bg3")


def test_live_builder_agent_accepts_trailing_rules():
    from agents import builder_promoted as BP
    assert set(G.TRAIL_KINDS) <= set(BP.SUPPORTED_EXITS)
    for agent in ("intraday", "scalping"):
        ok = [x for m in G.MARKET_TYPES for x in G.cell_grammar(agent, m, "bg4")
              if x.exit[0] in G.TRAIL_KINDS and BP.supported(x)[0]]
        assert ok, agent
        for c in ok:
            m0 = G.ENTRY_MIN[c.entry]
            want = min(m0 + int(c.exit[2]), BE.SQUARE_MIN) if c.exit[0].endswith("_min") else BE.SQUARE_MIN
            assert BP.exit_deadline_min(c) == want and BP.walk_of(c) == (m0,) + tuple(c.exit), c


def test_rule_v4_pinned_and_v3_unchanged():
    r4 = B.load_rule()
    assert r4["_sha256"] == B.RULE_SHA256_V4 == hashlib.sha256((HERE / "builder_rule_v4.json").read_bytes()).hexdigest()
    assert r4["version"] == "builder/v4" and r4["supersedes"]["sha256"] == B.RULE_SHA256_V3
    assert r4["grammar"]["version"] == "bg4" and r4["grammar"]["sha256"] == G.grammar_sha256("bg4")
    r3 = B.load_rule(B.RULE_FILE_V3)
    for k in r3:
        if k not in ("version", "written", "supersedes", "grammar", "statement", "_sha256") and \
                not k.startswith("candidate_note"):
            assert r4.get(k) == r3[k], k                                # statistics, schedule, sizing: unchanged


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
