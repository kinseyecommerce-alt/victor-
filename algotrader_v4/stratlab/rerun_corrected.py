"""
stratlab/rerun_corrected.py — the corrected-data re-run (Nov 2026), pre-registered in prereg/rerun_corrected_v1.json
(byte-identical copy: <scratchpad>/audit/PREREGISTERED_RERUN.json) BEFORE any strategy was simulated on corrected
data.  Research only: reads bars, writes to <scratchpad>/audit.

    python -m stratlab.rerun_corrected --preregister          # writes both prereg copies (refuses to overwrite)
    python -m stratlab.rerun_corrected --stage screen         # every catalogue id, corrected 2023-26 DISCOVERY
    python -m stratlab.rerun_corrected --stage tfgrid         # all tfgrid combos, corrected 2023-25 DISCOVERY
    python -m stratlab.rerun_corrected --stage final          # FINAL: one look for new passers, recompute opened ones
    python -m stratlab.rerun_corrected --stage oldera         # 2012-23 daily (PIT NIFTY 100) + 2016-23 minute
    python -m stratlab.rerun_corrected --stage report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SCRATCH = os.environ.get("STRATLAB_SCRATCH", _SCRATCH)
AUDIT = os.path.join(SCRATCH, "audit")
PREREG = os.path.join(AUDIT, "PREREGISTERED_RERUN.json")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "rerun_corrected_v1.json")
LEDGER_DIR = os.path.join(AUDIT, "ledgers")
FINAL_LEDGER = os.path.join(LEDGER_DIR, "final_corrected.json")
OLDERA_DAILY_LEDGER = os.path.join(LEDGER_DIR, "oldera_daily_corrected.json")
OLDERA_MIN_LEDGER = os.path.join(LEDGER_DIR, "oldera_min_corrected.json")
STAGE = "corrected_rerun"
DATA_TAG = "corr1"
CODE_FILES = ("corrected.py", "rerun_corrected.py", "audit_run.py", "refsim.py", "sim.py", "protocol.py",
              "protocol_oos.py", "protocol_min_oos.py", "tfgrid.py", "capital.py", "data.py", "data_daily.py",
              "auction.py", "attention.py", "auction_r2.py", "events_run.py", "trials.py")
SPECIAL = {"proven": ["SWING-ATTN-ON-3X", "AUC-FADE-2PCT"],
           "live_paper": ["AUC-FADE-3PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED"],
           "event": ["ev_pead_long_h60"]}


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def preregister() -> dict:
    from . import catalog  # noqa: F401
    from . import corrected as C
    from . import tfgrid as TG
    from .api import REGISTRY
    if os.path.exists(PREREG) or os.path.exists(REPO_PREREG):
        raise SystemExit("pre-registration already written: refusing to overwrite (one plan, one look)")
    here = os.path.dirname(os.path.abspath(__file__))
    ids = sorted(REGISTRY)
    fp = {i: REGISTRY[i].fingerprint for i in ids}
    doc = {
        "title": "Corrected-data re-run of every lab strategy (same hypotheses, not new trials)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "why": "User: 'Test all strategies with corrected data - this much of strategies won't fail, simulation trade "
               "may be mistaken.' Part 1 of the work audits the simulator independently; this file freezes how the "
               "re-run on CORRECTED data is done and judged, before any strategy is simulated on corrected data.",
        "status_at_writing": {
            "simulator_audit": "refsim (independent engine) vs sim.py on 22 strategies x 97 symbols, DISCOVERY "
                               "2023-11-10..2025-12-16: every trade identical (fills, exits, qty, gross, cost) "
                               "wherever both engines read a strategy's rule the same way; all differences traced to "
                               "ambiguous one-line descriptions or undocumented hygiene filters, none to the engine. "
                               "30/30 hand-checked trades match the raw JSON rows and a manual Zerodha computation. "
                               "Gross/mirror/look-ahead/benchmark checks are still running when this was written: "
                               "if any of them finds a simulator BUG, the clause 'bug_rule' below applies.",
            "corrected_results_seen": "none (no strategy has been simulated on the corrected view)"},
        "data": {
            "definition": "stratlab/corrected.py (see its docstring): minute bars of both eras mapped per session "
                          "onto official NSE prints x F(d) (splits / bonuses / consolidations / rights / demergers; "
                          "dividends NOT back-adjusted = as-traded prices); affine repair of rupee-subtracted "
                          "dividends; Kite's stale 09:15 bar on special pre-open days dropped; sessions whose shape "
                          "disagrees with the official prints (> 1.5 % off, o/h/l ratios > 0.5 % apart) EXCLUDED; "
                          "unadjusted-volume windows rescaled and flagged; the 10 daily-file (nse_old) issues "
                          "repaired by official_factor / applied_factor. Index bars (NIFTY 50, NIFTY BANK, VIX) "
                          "unchanged.",
            "version": C.VERSION,
            "validation": "all 119 minute-era rows of bar_adjustment_issues.csv pass corrected.validate() "
                          "(overnight ratio = official gap / official factor; additive offsets < 0.2 % of price; "
                          "seam 2023-11-09/10 for IOC / ONGC / PETRONET within 0.5 % of the official gap)",
            "maps_sha256": {f: _sha(os.path.join(C.OUT, f)) for f in ("maps_3y.pkl", "maps_old.pkl")},
            "universe_minute": "today's 97 NIFTY-100 names (minute data exists only for them): the minute re-run "
                               "remains SURVIVORSHIP-BIASED, as before",
            "universe_daily_2012_23": "point-in-time NIFTY 100 membership from flowdata index_members "
                                      "(index_membership/index_members.csv.gz, reliable from 2012-01-02): a stock "
                                      "strategy may open a position on decision day D only if the stock is a NIFTY "
                                      "100 constituent in force on D (replaces the turnover top-100 of the original "
                                      "2012-23 round); event strategies keep their point-in-time top-500 universe"},
        "hypotheses": {
            "rule": "the SAME strategies with the SAME code, parameters and gates; nothing is added, dropped or "
                    "tuned. They are NOT new trials: ledger rows get stage 'corrected_rerun' / 'corrected_final' / "
                    "'corrected_oos' and the same params fingerprint, and are NOT added to N for the DSR.",
            "catalogue_ids": len(ids),
            "catalogue_fingerprints_sha256": hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest(),
            "tfgrid": "all 668 grid combos (38 rules x 8 timeframes x universes) + 608 natives with the 1:3 bracket "
                      "and Rs 10 lakh accounts exactly as prereg/tfgrid_v1.json",
            "tfgrid_prereg_sha256": _sha(TG.REPO_PREREG),
            "modules": ["AUC-FADE-2PCT", "AUC-FADE-3PCT", "SCALP-AUC-FADE-1TO3-0925", "SWING-ATTN-ON-3X"],
            "not_rerun": {"option / VRP strategies": "they read only index bars and NSE option files, which the "
                                                      "correction does not change: their results are unchanged by "
                                                      "construction (VRP-STRANGLE-016-WINGED reported as before)"},
            "special": SPECIAL},
        "judgement": {
            "screen_2023_26": "protocol.screen unchanged: DISCOVERY 2023-11-10..2025-12-16 (first 70 % of the "
                              "726-session calendar; H1 to 2024-11-29, H2 from 2024-12-02); net > 0 in H1 and H2, "
                              "PF >= 1.15, >= 150 (intraday) / 60 (positional) trades, top symbol <= 35 % of net "
                              "(not for one-instrument strategies); 2 bps (index 1 bps) slippage, costs.py 2026; "
                              "tfgrid: its own account screen unchanged",
            "result_of_record": "the corrected-data result is the result of record for every strategy",
            "final_one_look": "a strategy that passes the corrected DISCOVERY screen opens its FINAL "
                              "(2025-12-17..2026-11-09) ONLY if its FINAL was never opened before on the 3-year data "
                              "under any cost model, in any ledger (stratlab_cache/*/final_ledger.json, "
                              "tfgrid/final_ledger.json, events_daily ledger); the look is written once to "
                              "audit/ledgers/final_corrected.json. CONFIRMED = net > 0 and PF >= 1.05.",
            "opened_finals": "a FINAL that was already opened is recomputed on corrected data and reported side by "
                             "side; the ORIGINAL look stands (it is not a fresh look) unless bug_rule applies",
            "bug_rule": "if Part 1 finds a simulator bug, the corrected ENGINE's result stands for every affected "
                        "strategy and the report says so explicitly",
            "old_era": "for every strategy that passed the 2023-26 screen before (original verdict PASSED SCREEN / "
                       "CONFIRMED / REJECTED ON FINAL on bars3y, any tfgrid stage-1 passer) or passes now, plus "
                       "SPECIAL: 2012-23 daily (protocol_oos.judge gates unchanged, corrected daily file, PIT NIFTY "
                       "100) where eligible, and 2016-23 minute (protocol_min_oos gates unchanged, corrected old "
                       "minute view; its dirty-link exclusion = the original dirty set plus the session after "
                       "every excluded session). One judgement each, written to audit/ledgers/oldera_*_corrected.json; "
                       "the original one-look ledgers are not touched and their verdicts are reported next to it.",
            "proven": "FINAL-CONFIRMED on 2023-26 AND OOS-CONFIRMED on 2016-23 (tfgrid: also DSR >= 0.95), as before",
            "dsr": "deflated Sharpe with the SAME N as before the re-run (variants 2023-26 in trials_v2.csv, which "
                   "the re-run does not increase)"},
        "code_sha256": {f: _sha(os.path.join(here, f)) for f in CODE_FILES if os.path.exists(os.path.join(here, f))},
        "costs_sha256": _sha(os.path.join(os.path.dirname(here), "costs.py")),
    }
    blob = json.dumps(doc, indent=1, sort_keys=True).encode()
    for p in (PREREG, REPO_PREREG):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(blob)
    with open(PREREG + ".sha256", "w") as fh:
        fh.write(hashlib.sha256(blob).hexdigest() + "\n")
    return doc


def load_prereg() -> dict:
    a, b = open(PREREG, "rb").read(), open(REPO_PREREG, "rb").read()
    if a != b:
        raise SystemExit("the two pre-registration copies differ")
    return json.loads(a)


# ══════════════════════════════════════════════════════════════════════════
# stages (run only after the pre-registration exists; see main())
# ══════════════════════════════════════════════════════════════════════════
def _jdump(path, obj):
    import gzip
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    op = gzip.open if path.endswith(".gz") else open
    with op(tmp, "wt") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def _jload(path, default=None):
    import gzip
    try:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def log(msg):
    from .audit_run import log as _l
    _l(msg)


def stage_screen(jobs: int):
    """every catalogue id on the corrected 2023-26 DISCOVERY (FINAL never simulated here)"""
    import gzip
    import pickle
    from . import catalog  # noqa: F401
    from . import audit_run as AR
    from .api import REGISTRY
    load_prereg()
    res, daily = AR.run_catalog(sorted(REGISTRY), "net", "corrected", AR.DISC[0], AR.DISC[1], jobs, tag="rerun")
    out = os.path.join(AUDIT, "rerun")
    _jdump(os.path.join(out, "screen_corrected.json"), res)
    with gzip.open(os.path.join(out, "screen_corrected_daily.pkl.gz"), "wb") as fh:
        pickle.dump(daily, fh)
    log(f"rerun screen: {sum(r['passed'] for r in res.values())}/{len(res)} pass on corrected data")


def _module_specs():
    from . import attention as AT
    from . import auction as AU
    from . import auction_r2 as A2
    return AU, A2, AT


def module_runs(corrected: bool, periods=("disc", "final")) -> dict:
    """AUC-FADE-2PCT / -3PCT, SCALP-AUC-FADE-1TO3-0925, SWING-ATTN-ON-3X through their own modules."""
    from . import audit_run as AR
    from . import corrected as C
    from . import protocol as P
    from . import tfgrid as G
    AU, A2, AT = _module_specs()
    if corrected:
        C.install()
    try:
        out = {}
        rows, cal, n, cdir = AU.load_universe(AR.BARS3Y, AR.CACHE3Y, verbose=False)
        r2rows, _c2, _n2, _d2 = A2.load_universe(AR.BARS3Y, AR.CACHE3Y, verbose=False)
        trow, _c3, cnts, _n3, _d3 = AT.load_universe(AR.BARS3Y, AR.CACHE3Y, verbose=False)
        keep, _sp = AT.filter_rows(trow, cnts)
        sims = [(AU.SPECS, AU.make_simulator(rows)), (A2.SPECS, A2.make_simulator(r2rows)),
                (AT.SPECS, AT.make_simulator(keep))]
        split = G.split3()
        for specs, simf in sims:
            d = simf(specs, *split.discovery)
            f = simf(specs, *split.final) if "final" in periods else {}
            for sp in specs:
                sc = P.screen(d[sp.id]["trades"], split, sp.horizon)
                fv = P.final_verdict(f[sp.id]["trades"]) if sp.id in f else None
                out[sp.id] = {"screen": {k: sc[k] for k in ("passed", "reasons", "h1", "h2", "disc", "top_symbol",
                                                             "top_share")}, "final": fv,
                              "_trades_disc": d[sp.id]["trades"], "_trades_final": f.get(sp.id, {}).get("trades", [])}
        return out
    finally:
        if corrected:
            C.uninstall()


def stage_modules(jobs: int):
    load_prereg()
    res = {}
    for tag, corr in (("original", False), ("corrected", True)):
        r = module_runs(corr)
        res[tag] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in r.items()}
        trades = {k: {"disc": v["_trades_disc"], "final": v["_trades_final"]} for k, v in r.items()}
        _jdump(os.path.join(AUDIT, "rerun", f"modules_{tag}_trades.json.gz"), trades)
    _jdump(os.path.join(AUDIT, "rerun", "modules.json"), res)
    for k in res["corrected"]:
        a, b = res["original"][k], res["corrected"][k]
        log(f"rerun module {k}: screen {a['screen']['passed']} -> {b['screen']['passed']}, disc net "
            f"{a['screen']['disc']['net']:.0f} -> {b['screen']['disc']['net']:.0f}, FINAL net "
            f"{a['final']['net']:.0f} -> {b['final']['net']:.0f}")


def _tfgrid_stage1(out_dir: str, jobs: int, corrected: bool, zero_cost: bool):
    from . import capital as K
    from . import corrected as C
    from . import sim
    from . import tfgrid as G
    from . import tfgrid_run as TR
    TR.S1 = os.path.join(out_dir, "stage1")
    G.ROOT = out_dir
    G.PROGRESS = os.path.join(out_dir, "progress.txt")
    if corrected:
        C.install()
    if zero_cost:
        K.round_trip_cost = lambda *a, **k: 0.0
        sim.round_trip_cost = lambda *a, **k: 0.0
        G.SLIP_STOCK = G.SLIP_INDEX = 0.0
        sim.SLIPPAGE = sim.INDEX_SLIPPAGE = 0.0
    TR.stage1(jobs)


def stage_tfgrid(jobs: int, which: str):
    """which = corrected (net, the re-run) | gross (original data, zero cost / slippage: Part 1)"""
    if which == "corrected":
        load_prereg()
        _tfgrid_stage1(os.path.join(AUDIT, "tfgrid_corrected"), jobs, True, False)
    elif which == "gross":
        _tfgrid_stage1(os.path.join(AUDIT, "tfgrid_gross"), jobs, False, True)
    else:
        raise SystemExit(which)


# ── one-look FINAL ──────────────────────────────────────────────────────────
def opened_finals() -> dict:
    """{strategy id: [ledger keys]} of every FINAL ever opened on the 3-year data (catalogue, modules, tfgrid,
    events), any cost model."""
    led = {}
    for rel in ("stratlab_cache/bars3y_fb3d4d/final_ledger.json", "stratlab_cache/final_ledger.json",
                "stratlab_cache/events_daily/final_ledger.json", "tfgrid/final_ledger.json"):
        for k, v in (_jload(os.path.join(SCRATCH, rel), {}) or {}).items():
            sid = k.split("|")[0]
            if "726s" in k or rel.startswith("tfgrid"):
                led.setdefault(sid, []).append(rel)
                if sid.startswith("nat:"):          # a tfgrid native FINAL saw the same signal's FINAL period
                    led.setdefault(sid.split(":")[1], []).append(rel + " (as " + sid + ")")
    return led


def original_passers() -> set:
    """ids whose ORIGINAL 2023-26 screen verdict (bars3y) was a pass, from the trials ledger and the FINAL ledgers"""
    import csv
    ok = {"PASSED SCREEN", "CONFIRMED", "REJECTED ON FINAL", "PASS"}
    out = set()
    path = os.path.join(SCRATCH, "stratlab_cache", "trials_v2.csv")
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["era"] == "2023-26" and r["stage"] in ("screen", "final", "tfgrid-screen", "tfgrid-final") \
                    and "12m" not in r["dataset"] and r["verdict"] in ok:
                out.add(r["strategy_id"])
    return out | set(opened_finals())


def _final_ledger():
    return _jload(FINAL_LEDGER, {}) or {}


def _final_write(entries: dict):
    led = _final_ledger()
    for k, v in entries.items():
        led.setdefault(k, v)                   # first write wins: one look
    _jdump(FINAL_LEDGER, led)


def stage_final(jobs: int):
    from . import catalog  # noqa: F401
    from . import audit_run as AR
    from .api import REGISTRY
    load_prereg()
    scr = _jload(os.path.join(AUDIT, "rerun", "screen_corrected.json"), {})
    opened = opened_finals()
    orig = original_passers()
    now_pass = sorted(i for i, r in scr.items() if r["passed"])
    new_look = [i for i in now_pass if i not in opened]
    recompute = sorted(i for i in opened if i in REGISTRY)
    log(f"final: corrected passers {len(now_pass)}; never-opened -> one look: {new_look}; recompute {len(recompute)}")
    ids = sorted(set(new_look) | set(recompute))
    res, daily = AR.run_catalog(ids, "net", "corrected", AR.FINAL[0], AR.FINAL[1], jobs, tag="final") if ids else ({}, {})
    led = _final_ledger()
    entries, report = {}, {}
    for i in ids:
        f = res[i]["final"]
        conf = bool(f["net"] > 0 and f["pf"] >= 1.05)
        key = f"{i}|{AR.FINAL[0]}-{AR.FINAL[1]}|corr1"
        rec = {"fingerprint": REGISTRY[i].fingerprint, "when": time.strftime("%Y-%m-%d %H:%M"),
               "final": {"trades": f["trades"], "net": f["net"], "pf": f["pf"], "confirmed": conf}}
        if i in new_look:
            if key in led:
                rec = led[key]
                rec["note"] = "from ledger (first look)"
            else:
                entries[key] = rec
            report[i] = dict(rec, kind="ONE LOOK (never opened before)")
        else:
            report[i] = dict(rec, kind="RECOMPUTED (FINAL already opened; the original look stands)",
                             opened_in=opened[i])
    _final_write(entries)
    _jdump(os.path.join(AUDIT, "rerun", "final_corrected.json"),
           {"corrected_passers": now_pass, "new_looks": new_look, "original_passers": sorted(orig),
            "results": report})


# ── old eras ────────────────────────────────────────────────────────────────
def nifty100_members_by_symbol(symbols: list) -> dict:
    """{daily-cache symbol: sorted list of (from, to)} NIFTY 100 membership intervals (flowdata, read-only)."""
    from . import corrected as C
    from . import flowdata as F
    F._load_index()
    iv = F._S["idx_members"]["NIFTY100"]
    _cal, symmap = C._db()
    by = {}
    for s, f, t in iv:
        by.setdefault(s, []).append((f, t))
    out = {}
    for sym in symbols:
        key = symmap.lab_symbol(sym, 20231109)
        for k in {key, sym}:
            if k in by:
                out.setdefault(sym, []).extend(by[k])
    return out


def _pit_gate(intervals: dict):
    from . import data_daily as DD
    orig_gate = DD.gate

    def gate(spec, sd):
        if getattr(spec, "underlying", None):
            return spec
        iv = intervals.get(sd.symbol, [])
        mem = np.zeros(len(sd.days), bool)
        for f, t in iv:
            mem |= (sd.days >= f) & (sd.days <= t)
        sd.member = mem
        return orig_gate(spec, sd) if mem.any() else _never(spec)
    return gate


def _never(spec):
    import dataclasses
    return dataclasses.replace(spec, fn=lambda d, p: None)


import numpy as np  # noqa: E402


def stage_oldera_turnover(jobs: int):
    """decomposition (report only, not a verdict): the same 2012-23 daily judgement on the corrected daily file but
    with the ORIGINAL turnover top-100 universe, to separate the data correction from the universe change"""
    import multiprocessing as mp
    from . import catalog  # noqa: F401
    from . import corrected as C
    from . import data_daily as DD
    from . import protocol_oos as PO
    from .api import REGISTRY
    rep0 = _jload(os.path.join(AUDIT, "rerun", "oldera_daily.json"), {}) or {}
    ids = sorted(rep0.get("results", {}))
    C.install(minute=False, daily=True)
    cache = PO.setup()
    syms = DD.symbols(cache)
    run_syms = syms + ["_NIFTY50_min", "_NIFTYBANK_min"]
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        res = PO.evaluate([REGISTRY[i] for i in ids], PO.make_simulator(pool, run_syms, 2.0, 1.0), None,
                          cost_model="2026-11/f2|corr1|turnover100", ledger_path=os.path.join(LEDGER_DIR,
                          "oldera_daily_corrected_turnover_reportonly.json"), prereg=None, dataset="nse_old corrected")
    out = {r["id"]: {k: r.get(k) for k in ("verdict", "why", "era")} for r in res}
    _jdump(os.path.join(AUDIT, "rerun", "oldera_daily_turnover_universe.json"), out)
    log(f"oldera (turnover universe, report only): {[(k, v['verdict']) for k, v in out.items()]}")


def stage_oldera(jobs: int, ids: list | None = None):
    import multiprocessing as mp
    from . import catalog  # noqa: F401
    from . import corrected as C
    from . import data_daily as DD
    from . import protocol_oos as PO
    from .api import REGISTRY
    load_prereg()
    scr = _jload(os.path.join(AUDIT, "rerun", "screen_corrected.json"), {})
    want = sorted((original_passers() | {i for i, r in scr.items() if r["passed"]} |
                   set(SPECIAL["proven"]) | set(SPECIAL["live_paper"])) & set(REGISTRY)) if ids is None else ids
    el_ids = {x[0] for x in PO.eligibility()[0]}
    daily_ids = [i for i in want if i in el_ids and i in REGISTRY]
    log(f"oldera: {len(want)} ids wanted; daily-eligible {len(daily_ids)}")
    # ---- 2012-23 daily, corrected daily file, PIT NIFTY 100 ----
    C.install(minute=False, daily=True)
    cache = PO.setup()
    syms = DD.symbols(cache, members_only=False)
    iv = nifty100_members_by_symbol(syms)
    run_syms = sorted(iv) + [s for s in ("_NIFTY50_min", "_NIFTYBANK_min") if s in DD.manifest(cache)["symbols"]
                             or os.path.exists(os.path.join(cache, s + ".npz"))]
    cover = {}
    for d in (20120601, 20150601, 20180601, 20210601, 20230601):
        cover[d] = sum(1 for s, ivs in iv.items() if any(f <= d <= t for f, t in ivs))
    log(f"oldera daily: NIFTY 100 PIT coverage in the daily cache {cover} (of 100)")
    DD.gate = _pit_gate(iv)
    specs = [REGISTRY[i] for i in daily_ids]
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        simf = PO.make_simulator(pool, run_syms, 2.0, 1.0)
        sensf = PO.make_simulator(pool, run_syms, 5.0, 2.5)
        res = PO.evaluate(specs, simf, sensf, cost_model="2026-11/f2|corr1|nifty100pit",
                          ledger_path=OLDERA_DAILY_LEDGER, prereg=None, dataset="nse_old corrected")
    old_led = _jload(PO.LEDGER, {}) or {}
    rep = {}
    for r in res:
        prev = [v["result"] for k, v in old_led.items() if k.split("|")[0] == r["id"]]
        rep[r["id"]] = {"corrected": {k: r.get(k) for k in ("verdict", "why", "h1", "h2", "era", "top_symbol",
                                                             "top_share", "sens", "note")},
                        "original": {k: prev[0].get(k) for k in ("verdict", "why", "era")} if prev else None}
    _jdump(os.path.join(AUDIT, "rerun", "oldera_daily.json"), {"coverage": cover, "results": rep})
    log(f"oldera daily: {[(k, v['corrected']['verdict']) for k, v in rep.items()]}")


def stage_oldmin(jobs: int, ids: list | None = None):
    """2016-23 minute bars, corrected view, protocol_min_oos gates and runners."""
    from . import catalog  # noqa: F401
    from . import corrected as C
    from . import protocol_min_oos as PM
    from .api import REGISTRY
    load_prereg()
    scr = _jload(os.path.join(AUDIT, "rerun", "screen_corrected.json"), {})
    want = sorted(original_passers() | {i for i, r in scr.items() if r["passed"]} |
                  set(SPECIAL["proven"]) | set(SPECIAL["live_paper"])) if ids is None else ids
    C.install(minute=True, daily=False)
    ctx = PM.setup()
    # dirty set = original dirty links + the session after every excluded (dropped) session
    maps = C._maps("old")
    added = 0
    for sym, m in maps.items():
        exd = [int(d) for d in m["days"][m["exclude"]]]
        if not exd:
            continue
        days = [int(x) for x in m["days"]]
        nxt = {a: b for a, b in zip(days, days[1:])}
        extra = [nxt[d] for d in exd if d in nxt]
        cur = ctx["dirty"].get(sym, np.zeros(0, np.int64))
        ctx["dirty"][sym] = np.unique(np.concatenate([cur, np.array(exd + extra, np.int64)]))
        added += len(exd) + len(extra)
    log(f"oldmin: corrected view; {added} symbol-dates added to the dirty set (excluded sessions and the next)")
    cat = [i for i in want if i in REGISTRY and REGISTRY[i].horizon in ("intraday", "positional")]
    ms = PM.module_specs()
    mods = {}
    for i in want:
        if i in ms and ms[i][0] in ("auction", "auction_r2", "attention"):
            mods.setdefault(ms[i][0], []).append(ms[i][1])
    rep = {}
    led_old = _jload(PM.LEDGER, {}) or {}
    res = PM.run_catalogue([REGISTRY[i] for i in cat], jobs, log=log) if cat else {}
    for i, r in res.items():
        sp = REGISTRY[i]
        v = PM.verdict_of(sp, r)
        prev = [x["result"] for k, x in led_old.items() if k.split("|")[0] == i]
        rep[i] = {"corrected": v, "original": prev[0] if prev else None}
    for runner, specs in mods.items():
        if not specs:
            continue
        rr = PM.run_runner(runner, specs, jobs, log=log)
        for sp in specs:
            v = PM.verdict_of(sp, rr[sp.id])
            prev = [x["result"] for k, x in led_old.items() if k.split("|")[0] == sp.id]
            rep[sp.id] = {"corrected": v, "original": prev[0] if prev else None}
    led = _jload(OLDERA_MIN_LEDGER, {}) or {}
    for i, v in rep.items():
        led.setdefault(f"{i}|2016-23|corr1", {"when": time.strftime("%Y-%m-%d %H:%M"), "result": v["corrected"]})
    _jdump(OLDERA_MIN_LEDGER, led)
    _jdump(os.path.join(AUDIT, "rerun", "oldera_min.json"), rep)
    log(f"oldmin: {[(k, (v['corrected'] or {}).get('verdict')) for k, v in rep.items()]}")


def stage_events(jobs: int):
    """ev_pead_long_h60 on the corrected daily file: 2023-26 DISCOVERY screen + FINAL recompute (already opened)
    and 2012-23 (protocol_oos.judge), original vs corrected side by side."""
    import multiprocessing as mp
    from . import corrected as C
    from . import events_run as ER
    from . import events_strategies as ES
    from . import protocol as P
    from . import protocol_oos as PO
    load_prereg()
    sp = [s for s in ES.DAILY if s.id == "ev_pead_long_h60"][0]
    out = {}
    for tag, corr in (("original", False), ("corrected", True)):
        if corr:
            C.install(minute=False, daily=True)
        ER.setup()
        split = P.split_sessions(ER.lab_calendar())
        with mp.get_context("fork").Pool(jobs) as pool:
            simf = ER.make_daily_simulator(pool, ER.SLIP_BPS)
            d = simf([sp], *split.discovery)[sp.id]["trades"]
            f = simf([sp], *split.final)[sp.id]["trades"]
            o = simf([sp], *PO.ERA)[sp.id]["trades"]
        sc = P.screen(d, split, sp.horizon)
        out[tag] = {"screen": {k: sc[k] for k in ("passed", "reasons", "h1", "h2", "disc", "top_symbol")},
                    "final": P.final_verdict(f), "oldera_2012_23": {k: v for k, v in PO.judge(o, sp.horizon).items()
                                                                     if k != "top_share"},
                    "_trades": {"disc": d, "final": f, "old": o}}
        if corr:
            C.uninstall()
    _jdump(os.path.join(AUDIT, "rerun", "events_trades.json.gz"), {k: v.pop("_trades") for k, v in out.items()})
    _jdump(os.path.join(AUDIT, "rerun", "events.json"), out)
    log(f"rerun events ev_pead_long_h60: FINAL net {out['original']['final']['net']:.0f} -> "
        f"{out['corrected']['final']['net']:.0f}; 2012-23 {out['original']['oldera_2012_23']['passed']} -> "
        f"{out['corrected']['oldera_2012_23']['passed']}")


def _dsr_of(trades_tuples, lo, hi, n_trials, var_sr):
    """daily-P&L DSR (trials.py formulas) of lab trade tuples entered in [lo, hi] on the 3-year calendar"""
    from . import trials as T
    cal = T.calendar_3y(SCRATCH)
    tr = [{"entry": int(t[2]), "exit": int(t[6]), "net": float(t[12]), "gross": float(t[10]), "cost": float(t[11])}
          for t in trades_tuples]
    st = T.series_stats(tr, cal, lo, hi)
    if not st["n_days"] or st["sr_daily"] != st["sr_daily"]:
        return None
    d = T.dsr(st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"], n_trials, var_sr)
    return {"sr_daily": round(st["sr_daily"], 4), "sr_annual": round(st["sr_daily"] * 252 ** 0.5, 2),
            "n_days": st["n_days"], "dsr": round(d["dsr"], 4) if isinstance(d, dict) else round(float(d), 4),
            "n_trials": n_trials}


def stage_ledger():
    """append the re-run rows to trials_v2.csv (backed up first) with stages corrected_*; N must not change"""
    import csv
    import shutil
    from . import trials as T
    path = os.path.join(SCRATCH, "stratlab_cache", "trials_v2.csv")
    rows = list(csv.DictReader(open(path)))
    n_before = T.count_trials([r for r in rows if not r["stage"].startswith("corrected")], "2023-26")
    first = {}
    for r in rows:
        if r["era"] == "2023-26" and not r["stage"].startswith("corrected"):
            first.setdefault(r["strategy_id"], r)
    bak = path + ".bak_corrected_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, bak)
    new = []
    scr = _jload(os.path.join(AUDIT, "rerun", "screen_corrected.json"), {})
    for i, r in scr.items():
        f0 = first.get(i, {})
        d = r["disc"]
        new.append({"strategy_id": i, "family": f0.get("family") or T.family_of(i), "group": f0.get("group", ""),
                    "params_fp": f0.get("params_fp", ""), "era": "2023-26", "protocol": "2023-26 corrected (corr1)",
                    "stage": "corrected_rerun", "dataset": "bars3y+corr1", "period": f"{DISC_[0]}-{DISC_[1]}",
                    "cost_model": "2026-11/f2", "trades": d["trades"], "net": d["net"], "pf": d["pf"],
                    "mean_gross": d["gross"] / d["trades"] if d["trades"] else None,
                    "mean_cost": d["cost"] / d["trades"] if d["trades"] else None,
                    "verdict": "PASSED SCREEN" if r["passed"] else "FAILED SCREEN",
                    "source": os.path.join(AUDIT, "rerun", "screen_corrected.json"),
                    "note": "corrected-data re-run of the same hypothesis; not a new trial"})
    from .audit_run import _tf_rows
    for cid, r in _tf_rows(os.path.join(AUDIT, "tfgrid_corrected", "stage1")).items():
        if "screen" not in r:
            continue
        d = r["screen"]["disc"]
        f0 = first.get(cid, {})
        new.append({"strategy_id": cid, "family": f0.get("family") or "tfgrid", "params_fp": f0.get("params_fp", ""),
                    "era": "2023-26", "protocol": "tfgrid capital 1:3 v1 corrected (corr1)", "stage": "corrected_rerun",
                    "dataset": "bars3y+corr1", "period": f"{DISC_[0]}-{DISC_[1]}", "cost_model": "2026-11/f2",
                    "trades": d["trades"], "net": d["net"], "pf": d["pf"], "mean_gross": r.get("mean_gross"),
                    "mean_cost": r.get("mean_cost"),
                    "verdict": "PASSED SCREEN" if r["screen"]["passed"] else "FAILED SCREEN",
                    "source": os.path.join(AUDIT, "tfgrid_corrected", "stage1"), "note": "corrected data; not a new trial"})
    mods = (_jload(os.path.join(AUDIT, "rerun", "modules.json"), {}) or {}).get("corrected", {})
    for i, r in mods.items():
        d = r["screen"]["disc"]
        f0 = first.get(i, {})
        new.append({"strategy_id": i, "family": f0.get("family") or T.family_of(i), "params_fp": f0.get("params_fp", ""),
                    "era": "2023-26", "protocol": "2023-26 corrected (corr1)", "stage": "corrected_rerun",
                    "dataset": "bars3y+corr1", "period": f"{DISC_[0]}-{DISC_[1]}", "cost_model": "2026-11/f2",
                    "trades": d["trades"], "net": d["net"], "pf": d["pf"],
                    "verdict": "PASSED SCREEN" if r["screen"]["passed"] else "FAILED SCREEN",
                    "source": os.path.join(AUDIT, "rerun", "modules.json"), "note": "corrected data; FINAL recomputed "
                    f"{r['final']['net']} (PF {r['final']['pf']}), original look stands"})
    fin = (_jload(os.path.join(AUDIT, "rerun", "final_corrected.json"), {}) or {}).get("results", {})
    for i, r in fin.items():
        f0 = first.get(i, {})
        new.append({"strategy_id": i, "family": f0.get("family") or T.family_of(i), "params_fp": f0.get("params_fp", ""),
                    "era": "2023-26", "protocol": "2023-26 corrected (corr1)", "stage": "corrected_final",
                    "dataset": "bars3y+corr1", "period": "20251217-20261109", "cost_model": "2026-11/f2",
                    "trades": r["final"]["trades"], "net": r["final"]["net"], "pf": r["final"]["pf"],
                    "verdict": ("CONFIRMED" if r["final"]["confirmed"] else "REJECTED ON FINAL") + " [" + r["kind"] + "]",
                    "source": FINAL_LEDGER, "note": "corrected data"})
    for fn, era in (("oldera_min.json", "2016-23"), ("oldera_daily.json", "2012-23")):
        js = _jload(os.path.join(AUDIT, "rerun", fn), {}) or {}
        res = js.get("results", js)
        for i, v in res.items():
            c = v.get("corrected") or {}
            e = c.get("era") or {}
            new.append({"strategy_id": i, "family": T.family_of(i), "params_fp": first.get(i, {}).get("params_fp", ""),
                        "era": era, "protocol": f"{era} OOS corrected (corr1)", "stage": "corrected_oos",
                        "dataset": "corr1", "period": era, "cost_model": "2026-11/f2", "trades": e.get("trades"),
                        "net": e.get("net"), "pf": e.get("pf"), "verdict": c.get("verdict"),
                        "source": os.path.join(AUDIT, "rerun", fn), "note": "corrected data; original verdict stands"})
    n_app = T.TrialsLedger(path).append(new)
    rows2 = list(csv.DictReader(open(path)))
    n_after = T.count_trials([r for r in rows2 if not r["stage"].startswith("corrected")], "2023-26")
    log(f"ledger: {n_app} rows appended (backup {bak}); N 2023-26 (non-corrected stages) {n_before} -> {n_after}")
    return {"appended": n_app, "backup": bak, "n_before": n_before, "n_after": n_after}


DISC_ = (20231110, 20251216)


def stage_tfgrid_oos(jobs: int):
    """tfgrid stage 2 (2016-23 minute, corrected old view) for every combo that passed tfgrid stage 1 on the
    original OR the corrected data; own ledger under audit/ (the original stage2_ledger.json is not touched)."""
    from . import corrected as C
    from . import tfgrid as G
    from . import tfgrid_run as TR
    from .audit_run import _tf_rows
    load_prereg()
    o = _tf_rows(os.path.join(SCRATCH, "tfgrid", "stage1"))
    c = _tf_rows(os.path.join(AUDIT, "tfgrid_corrected", "stage1"))
    union = {k for k, r in o.items() if r.get("screen", {}).get("passed")} | \
            {k for k, r in c.items() if r.get("screen", {}).get("passed")}
    rows = {k: {"screen": {"passed": k in union}} for k in set(o) | set(c)}
    out = os.path.join(AUDIT, "tfgrid_corrected")
    TR.S2 = os.path.join(out, "stage2")
    TR.STAGE2_LEDGER = os.path.join(out, "stage2_ledger.json")
    G.ROOT, G.PROGRESS = out, os.path.join(out, "progress.txt")
    TR.load_stage1 = lambda: rows
    C.install(minute=True, daily=False)
    led = TR.stage2(jobs)
    orig = _jload(os.path.join(SCRATCH, "tfgrid", "stage2_ledger.json"), {}) or {}
    rep = {k: {"corrected": led.get(k, {}).get("verdict"),
               "corrected_reasons": (led.get(k, {}).get("judge") or {}).get("reasons"),
               "corrected_era": (led.get(k, {}).get("judge") or {}).get("era"),
               "original": orig.get(k, {}).get("verdict"),
               "original_era": (orig.get(k, {}).get("judge") or {}).get("era")} for k in sorted(union)}
    _jdump(os.path.join(AUDIT, "rerun", "tfgrid_oos.json"), rep)
    log(f"tfgrid oos: {[(k, v['original'], v['corrected']) for k, v in rep.items()]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.rerun_corrected")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--stage", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    if a.preregister:
        preregister()
        print(f"written {PREREG} and {REPO_PREREG} (sha256 {_sha(PREREG)})")
        return 0
    ids = [x for x in a.ids.split(",") if x] or None
    st = {"screen": lambda: stage_screen(a.jobs), "modules": lambda: stage_modules(a.jobs),
          "tfgrid": lambda: stage_tfgrid(a.jobs, "corrected"), "tfgrid_gross": lambda: stage_tfgrid(a.jobs, "gross"),
          "final": lambda: stage_final(a.jobs), "oldera": lambda: stage_oldera(a.jobs, ids),
          "oldmin": lambda: stage_oldmin(a.jobs, ids), "events": lambda: stage_events(a.jobs), "ledger": stage_ledger, "tfgrid_oos": lambda: stage_tfgrid_oos(a.jobs), "oldera_turnover": lambda: stage_oldera_turnover(a.jobs)}
    if a.stage not in st:
        raise SystemExit(f"unknown stage {a.stage!r}")
    st[a.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
