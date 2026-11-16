"""
stratlab/size10l_run.py — the OLD LAB CATALOGUE re-tested at the "Rs 10 lakh capital per agent" trade size (Nov 2026).
Research only: reads bars, writes to <scratchpad>/size10l. Pre-registered in PREREGISTERED_SIZE10L.json (byte-identical
copies in <scratchpad> and next to this file) BEFORE any Rs-5-lakh result was computed.

Why: every catalogue strategy was tested at Rs 1 lakh per stock trade (sim.NOTIONAL). At that size Zerodha's flat
Rs 20 per order is about half of an intraday round trip's fees (8.3 bps); at Rs 5 lakh the same schedule costs
4.5 bps. The user's accounts are Rs 10 lakh per agent, so the realistic size of an intraday trade is Rs 5 lakh
(<= 10 trades x 20 % intraday margin).

Size by cost kind (the definition builder/v3 uses):
    equity_mis (intraday, scalping)            Rs 5,00,000 per trade
    future on a STOCK (futures kind, no index) Rs 5,00,000 per trade (about one real lot)
    equity_cnc (swing, delivery)               Rs 1,00,000 (unchanged: proportional fees, PF independent of size)
    index futures                              the spec's lots x sim.LOT_SIZES (unchanged)
Only the quantity changes: qty = max(1, int(notional // fill)) exactly as sim._settle computes it; fills, exits,
slippage (2 bps a side, index 1 bps), costs.round_trip_cost on the real buy / sell values, the gates, the windows and
the engines are the lab's own, unchanged. No frozen lab module is edited: `install()` swaps sim._settle for a wrapper
in this process (and its forked workers) and `uninstall()` puts the original back.

    cd algotrader_v4
    python -m stratlab.size10l_run --stage verify       # Rs 1 lakh through the wrapper == the stored results
    python -m stratlab.size10l_run --preregister        # refuses to overwrite
    python -m stratlab.size10l_run --stage screen       # K ids, corrected 2023-26 DISCOVERY, new size
    python -m stratlab.size10l_run --stage controls     # random-entry placebos + planted look-ahead, new size
    python -m stratlab.size10l_run --stage final        # one look per DISCOVERY passer
    python -m stratlab.size10l_run --stage oldmin       # 2016-23 minute for FINAL-confirmed
    python -m stratlab.size10l_run --stage olddaily     # 2012-23 daily (report) for FINAL-confirmed, where eligible
    python -m stratlab.size10l_run --stage live         # the live rules at the new size (report only)
    python -m stratlab.size10l_run --stage dsr
    python -m stratlab.size10l_run --stage ledger
    python -m stratlab.size10l_run --stage report
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SCRATCH = os.environ.get("STRATLAB_SCRATCH", _SCRATCH)
OUT = os.path.join(SCRATCH, "size10l")
PREREG = os.path.join(SCRATCH, "PREREGISTERED_SIZE10L.json")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREGISTERED_SIZE10L.json")
FINAL_LEDGER = os.path.join(OUT, "final_ledger.json")
OLDMIN_LEDGER = os.path.join(OUT, "oldera_min_ledger.json")
OLDDAILY_LEDGER = os.path.join(OUT, "oldera_daily_ledger.json")
PROGRESS = os.path.join(OUT, "progress.txt")

LAKH = 100_000.0
SIZE_MIS = 5 * LAKH
SIZE_STOCK_FUTURE = 5 * LAKH
SIZE_CNC = 1 * LAKH
N_BEFORE = 4979          # builder registry n_used after sandbox batch 0006 (lab 2,552 + 16 + 2,400 + 11 FINAL looks)
DSR_MIN = 0.95
ID_PREFIX = "sz10l:"     # trials-ledger id of a re-test (a new trial; the Rs 1 lakh row keeps the bare id)
VERIFY_IDS_N = 3
CODE_FILES = ("size10l_run.py", "sim.py", "audit_run.py", "rerun_corrected.py", "corrected.py", "protocol.py",
              "protocol_oos.py", "protocol_min_oos.py", "trials.py", "api.py", "data.py", "data_daily.py")


# ══════════════════════════════════════════════════════════════════════════
# the size rule and the sim._settle wrapper
# ══════════════════════════════════════════════════════════════════════════
def notional_for(cost_kind: str, underlying=None) -> float | None:
    """Rs per trade at the Rs-10-lakh-per-agent size; None = an index future (lots, not a notional)."""
    if underlying:
        return None
    if cost_kind == "equity_mis":
        return SIZE_MIS
    if cost_kind == "future":
        return SIZE_STOCK_FUTURE
    if cost_kind == "equity_cnc":
        return SIZE_CNC
    raise ValueError(f"unknown cost kind {cost_kind!r}")


def size_changed(cost_kind: str, underlying=None) -> bool:
    n = notional_for(cost_kind, underlying)
    return n is not None and n != LAKH


class Trade(tuple):
    """A sim trade tuple (sim.TRADE_FIELDS) that also carries its Rs-1-lakh twin in `.l1` = (qty, gross, cost, net)."""


_ORIG = {}
_FORCE: list = [None]        # [notional] forces every stock trade to that size (the Rs 1 lakh verification)


def settle(spec, symbol, date, side, entry_t, fill, exit_date, exit_t, exit_px, reason, lots=None):
    """sim._settle with the quantity taken from notional_for(); everything else character for character."""
    from . import sim
    if getattr(spec, "underlying", None):
        return _ORIG["settle"](spec, symbol, date, side, entry_t, fill, exit_date, exit_t, exit_px, reason, lots)
    kind = spec.cost_kind

    def at(notional):
        qty = max(1, int(notional // fill))
        gross = side * (exit_px - fill) * qty
        buy_v, sell_v = (fill * qty, exit_px * qty) if side > 0 else (exit_px * qty, fill * qty)
        cost = sim.round_trip_cost(buy_v, sell_v, kind)
        return qty, gross, cost
    n = _FORCE[0] if _FORCE[0] else notional_for(kind, None)
    qty, gross, cost = at(n)
    t = Trade((spec.id, symbol, int(date), int(side), entry_t, round(fill, 4), int(exit_date), exit_t,
               round(exit_px, 4), qty, round(gross, 2), cost, round(gross - cost, 2), reason))
    q1, g1, c1 = at(LAKH)
    t.l1 = (q1, round(g1, 2), c1, round(g1 - c1, 2))
    return t


def install(force: float | None = None) -> None:
    from . import sim
    if "settle" not in _ORIG:
        _ORIG["settle"] = sim._settle
    _FORCE[0] = force
    sim._settle = settle


def uninstall() -> None:
    from . import sim
    if "settle" in _ORIG:
        sim._settle = _ORIG.pop("settle")
    _FORCE[0] = None


# ══════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════
def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    op = gzip.open if path.endswith(".gz") else open
    with op(tmp, "wt") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def _jload(path, default=None):
    try:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(PROGRESS, "a") as fh:
        fh.write(line + "\n")


def split_ids() -> tuple:
    """(re-tested ids, unchanged ids) of the 608-strategy catalogue."""
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    ch = sorted(i for i, sp in REGISTRY.items() if size_changed(sp.cost_kind, sp.underlying))
    return ch, sorted(set(REGISTRY) - set(ch))


def load_prereg() -> dict:
    a, b = open(PREREG, "rb").read(), open(REPO_PREREG, "rb").read()
    if a != b:
        raise SystemExit("the two pre-registration copies differ")
    return json.loads(a)


def daily_series(xnet: dict, cal: list, lo: int, hi: int) -> list:
    """trials.daily_pnl on per-exit-date sums of trades ENTERED in [lo, hi]: the net is booked on the exit session
    (the next session if the exit day is missing, the last session if the exit is after `hi`)."""
    import bisect
    days = [d for d in cal if lo <= d <= hi]
    if not days:
        return []
    pos = {d: i for i, d in enumerate(days)}
    pnl = [0.0] * len(days)
    for x, v in xnet.items():
        i = pos.get(int(x))
        if i is None:
            i = min(bisect.bisect_left(days, int(x)), len(days) - 1)
        pnl[i] += float(v)
    return pnl


def sr_stats(pnl: list) -> dict:
    from . import trials as T
    m = T.moments(pnl)
    return {"n_days": m["n"], "sr_daily": m["sr"], "skew_daily": m["skew"], "kurt_daily": m["kurt"]}


# ══════════════════════════════════════════════════════════════════════════
# the 3-year runner (same per-symbol engine call as audit_run._cat_work, plus the Rs-1-lakh twin)
# ══════════════════════════════════════════════════════════════════════════
def _specs_for(ids, controls: bool):
    from . import audit_run as AR
    if controls:
        cs = [sp for sp in AR._control_specs() if sp.id.startswith(("RAND_TIME30", "RAND_BRACKET05"))]
        _i, _p, cheat = AR._la_specs()
        return cs + [cheat]
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    return [REGISTRY[i] for i in ids]


def _work(args):
    sym, ids, lo, hi, controls, cache = args
    from . import corrected
    from . import sim
    corrected.install()
    specs = _specs_for(ids, controls)
    t0 = time.time()
    res = sim.run_symbol(cache, sym, specs, lo, hi, audit=False)
    out = {}
    for sid, r in res.items():
        tr = r["trades"]
        if not tr and not r["error"]:
            continue
        if tr:
            dts = np.array([t[2] for t in tr], np.int64)
            xds = np.array([t[6] for t in tr], np.int64)

            def agg6(net, gro, cst):
                u, inv = np.unique(dts, return_inverse=True)
                a = np.zeros((len(u), 6))
                np.add.at(a[:, 0], inv, 1)
                np.add.at(a[:, 1], inv, net)
                np.add.at(a[:, 2], inv, np.where(net > 0, net, 0))
                np.add.at(a[:, 3], inv, np.where(net < 0, net, 0))
                np.add.at(a[:, 4], inv, gro)
                np.add.at(a[:, 5], inv, cst)
                return u, a
            net = np.array([t[12] for t in tr], float)
            u, a = agg6(net, np.array([t[10] for t in tr], float), np.array([t[11] for t in tr], float))
            l1 = [getattr(t, "l1", None) for t in tr]
            if all(x is not None for x in l1):
                _u, a1 = agg6(np.array([x[3] for x in l1], float), np.array([x[1] for x in l1], float),
                              np.array([x[2] for x in l1], float))
            else:                                   # index-future trades have no twin: the same numbers
                a1 = a
            xu, xinv = np.unique(xds, return_inverse=True)
            xn = np.zeros(len(xu))
            np.add.at(xn, xinv, net)
            notional = float(np.mean([t[5] * t[9] for t in tr]))
            out[sid] = (u, a, a1, xu, xn, r["error"], r["skipped"], notional, len(tr))
        else:
            z = np.zeros((0, 6))
            out[sid] = (np.zeros(0, np.int64), z, z, np.zeros(0, np.int64), np.zeros(0), r["error"], r["skipped"], 0.0, 0)
    return sym, out, time.time() - t0


def run3y(ids, lo: int, hi: int, jobs: int = 2, controls: bool = False, tag: str = "") -> dict:
    """Every id on the corrected 2023-26 minute view for entries in [lo, hi]. Returns {id: {...screen_from fields...,
    "l1": the Rs-1-lakh twin's partitions, "xnet": {exit date: net}, "mean_notional"}}."""
    from . import audit_run as AR
    specs = {sp.id: sp for sp in _specs_for(ids, controls)}
    syms = AR.stock_symbols(AR.CACHE3Y) + ["_NIFTY50_min", "_NIFTYBANK_min"]
    acc, acc1, xacc = {}, {}, {}
    persym, persym1 = defaultdict(dict), defaultdict(dict)
    errs, skipped = {}, defaultdict(int)
    nsum, ncnt = defaultdict(float), defaultdict(int)
    t0 = time.time()
    args = [(s, list(ids), lo, hi, controls, AR.CACHE3Y) for s in syms]
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
        for n, (sym, out, _secs) in enumerate(pool.imap_unordered(_work, args), 1):
            for sid, (u, a, a1, xu, xn, err, sk, notional, k) in out.items():
                for store, arr in ((acc, a), (acc1, a1)):
                    d = store.setdefault(sid, {})
                    for dt, row in zip(u.tolist(), arr):
                        d[dt] = d[dt] + row if dt in d else row.copy()
                x = xacc.setdefault(sid, {})
                for dt, v in zip(xu.tolist(), xn.tolist()):
                    x[dt] = x.get(dt, 0.0) + v
                m = (u >= AR.DISC[0]) & (u <= AR.DISC[1])
                persym[sid][sym] = float(a[m, 1].sum())
                persym1[sid][sym] = float(a1[m, 1].sum())
                skipped[sid] += sk
                nsum[sid] += notional * k
                ncnt[sid] += k
                if err and sid not in errs:
                    errs[sid] = err
            if n % 10 == 0:
                log(f"  run3y {tag}: {n}/{len(syms)} symbols ({time.time() - t0:.0f}s)")
    res = {}
    for sid, sp in specs.items():
        def part(store, ps):
            d = store.get(sid, {})
            dates = np.array(sorted(d), np.int64)
            A = np.array([d[x] for x in dates]) if len(dates) else np.zeros((0, 6))
            return AR.screen_from(dates, A, ps.get(sid, {}), sp.horizon, bool(sp.underlying))
        sc = part(acc, persym)
        res[sid] = {"kind": sp.kind, "horizon": sp.horizon, "cost_kind": sp.cost_kind, "family": sp.family,
                    "error": errs.get(sid), "skipped": skipped[sid], **sc, "l1": part(acc1, persym1),
                    "xnet": {str(k): round(v, 2) for k, v in sorted(xacc.get(sid, {}).items())},
                    "mean_notional": round(nsum[sid] / ncnt[sid], 0) if ncnt[sid] else None}
    log(f"run3y {tag} {lo}-{hi}: {len(specs)} ids in {time.time() - t0:.0f}s")
    return res


# ══════════════════════════════════════════════════════════════════════════
# stages
# ══════════════════════════════════════════════════════════════════════════
def verify_ids() -> list:
    """Three re-tested ids fixed without looking at any result: the alphabetically first intraday (equity_mis),
    scalping (equity_mis) and stock-futures id whose stored corrected DISCOVERY row has >= 150 trades."""
    from .api import REGISTRY
    ch, _ = split_ids()
    stored = _jload(os.path.join(SCRATCH, "audit", "rerun", "screen_corrected.json"), {}) or {}
    out = []
    for kind in ("intraday", "scalping", "futures"):
        for i in ch:
            if REGISTRY[i].kind == kind and (stored.get(i, {}).get("disc", {}).get("trades") or 0) >= 150:
                out.append(i)
                break
    return out


def _same(a: dict, b: dict) -> bool:
    return all(a[p]["trades"] == b[p]["trades"] and abs(a[p]["net"] - b[p]["net"]) < 0.005 and
               abs(a[p]["gross"] - b[p]["gross"]) < 0.005 and abs(a[p]["cost"] - b[p]["cost"]) < 0.005
               for p in ("h1", "h2", "disc"))


def stage_verify(jobs: int):
    """BEFORE the pre-registration: the wrapper at a forced Rs 1 lakh must reproduce the stored corrected-data
    DISCOVERY rows of three strategies to the paisa. No Rs-5-lakh number is computed here."""
    from . import audit_run as AR
    ids = verify_ids()
    stored = _jload(os.path.join(SCRATCH, "audit", "rerun", "screen_corrected.json"), {})
    install(force=LAKH)
    try:
        res = run3y(ids, AR.DISC[0], AR.DISC[1], jobs, tag="verify@1L")
    finally:
        uninstall()
    rep = {}
    for i in ids:
        s, r = stored[i], res[i]
        rep[i] = {"match": _same(s, r), "passed_match": s["passed"] == r["passed"],
                  "stored": {p: {k: s[p][k] for k in ("trades", "net", "gross", "cost", "pf")} for p in ("h1", "h2", "disc")},
                  "wrapper_1L": {p: {k: r[p][k] for k in ("trades", "net", "gross", "cost", "pf")} for p in ("h1", "h2", "disc")}}
    ok = all(v["match"] and v["passed_match"] for v in rep.values())
    _jdump(os.path.join(OUT, "verify_1lakh.json"), {"ok": ok, "ids": ids, "results": rep})
    log(f"verify @Rs 1 lakh: {'ALL MATCH' if ok else 'MISMATCH'} {[(i, v['match']) for i, v in rep.items()]}")
    if not ok:
        raise SystemExit("the wrapper does not reproduce the stored results: stop")


def preregister() -> dict:
    from . import catalog  # noqa: F401
    from . import protocol as P
    from . import protocol_min_oos as PM
    from . import trials as T
    from .api import REGISTRY
    if os.path.exists(PREREG) or os.path.exists(REPO_PREREG):
        raise SystemExit("pre-registration already written: refusing to overwrite (one plan, one look)")
    ver = _jload(os.path.join(OUT, "verify_1lakh.json"))
    if not ver or not ver.get("ok"):
        raise SystemExit("run --stage verify first (the Rs 1 lakh reproduction must pass before the plan is frozen)")
    for f in ("screen.json", "final.json", "controls.json"):
        if os.path.exists(os.path.join(OUT, f)):
            raise SystemExit(f"{f} exists: a result was computed before the pre-registration")
    here = os.path.dirname(os.path.abspath(__file__))
    ch, un = split_ids()
    kinds = defaultdict(int)
    for i in ch:
        kinds[f"{REGISTRY[i].kind}/{REGISTRY[i].cost_kind}"] += 1
    unk = defaultdict(int)
    for i in un:
        unk[f"{REGISTRY[i].kind}/{REGISTRY[i].cost_kind}" + ("/index" if REGISTRY[i].underlying else "")] += 1
    rows = T.TrialsLedger(os.path.join(SCRATCH, "stratlab_cache", "trials_v2.csv")).read()
    doc = {
        "title": "Old lab catalogue re-tested at the Rs 10 lakh-per-agent trade size (Rs 5 lakh per intraday / "
                 "stock-futures trade)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "why": "User: 'Each agent 10 lac capital testing the All strategies' and 'Add old lab catalogue strategies "
               "also'. The catalogue was tested at Rs 1 lakh per stock trade, where the flat Rs 20-per-order brokerage "
               "is about half of an intraday round trip's fees. This file freezes how the re-test at the larger size "
               "is done and judged, before any Rs-5-lakh result is computed.",
        "results_seen_at_writing": {
            "rs5lakh_catalogue_results": "none",
            "known": "an approximate fee-only rescaling of the BUILDER's 2,400 candidates' DISCOVERY totals (about 19 "
                     "more intraday and 19 more scalping candidates net-positive in both halves); no catalogue "
                     "strategy was re-costed or re-run at the new size before this file",
            "verification": {"what": "the size wrapper forced to Rs 1 lakh reproduces the stored corrected-data "
                                     "DISCOVERY rows (trades, gross, cost, net per half) of three strategies exactly",
                             "ids": ver["ids"], "ok": ver["ok"]}},
        "scope": {
            "family": "the forward lab's family: 608 catalogue strategies (stratlab/catalog) + 668 tfgrid combos",
            "retested_K": len(ch), "retested_by_kind": dict(kinds), "retested_ids": ch,
            "unchanged_catalogue": {"count": len(un), "by_kind": dict(unk),
                                    "rule": "delivery (equity_cnc) stays Rs 1 lakh (proportional fees: PF does not "
                                            "depend on size) and index futures stay at the spec's real lots; their "
                                            "earlier verdicts stand; NOT re-run, NOT counted again"},
            "tfgrid_668": "NOT re-run and NOT counted again: every tfgrid combo was already traded from a Rs 10 lakh "
                          "account per agent (capital.py: 1 % risk per trade from the stop distance, 5x intraday "
                          "buying power, futures margin, slots, daily loss stop; prereg/tfgrid_v1.json). Its sizes "
                          "already are the Rs-10-lakh sizes (not Rs 1 lakh), so nothing changes for it; its verdicts "
                          "(0 proven of 1,276 incl. the 608 natives with the 1:3 bracket) stand. "
                          "Evidence from the stored stage-1 rows: median fee per intraday stock trade Rs 465 and per "
                          "scalping trade Rs 896 (a Rs 5 lakh MIS round trip costs Rs 225), i.e. typical trades of "
                          "Rs 10-25 lakh.",
            "live_rules": "AUC-FADE-2PCT / -3PCT, SWING-ATTN-ON-3X at Rs 5 lakh through their own modules: REPORT "
                          "ONLY (2023-26 DISCOVERY + FINAL recomputed; their verdicts are not re-opened, they are not "
                          "trials here). VRP-STRANGLE-016-WINGED (option lots) and ev_pead_long_h60 (delivery) do not "
                          "change size."},
        "size": {"equity_mis": SIZE_MIS, "stock_future": SIZE_STOCK_FUTURE, "equity_cnc": SIZE_CNC,
                 "index_future": "spec lots x sim.LOT_SIZES (unchanged)",
                 "capital_statement": "Rs 10 lakh per agent: <= 10 intraday trades x Rs 5 lakh x 20 % margin; "
                                      "stock futures about one real lot (~20 % margin). As in the Rs 1 lakh runs the "
                                      "number of concurrent positions is not capped by the account (the account-level "
                                      "test is tfgrid's).",
                 "qty": "max(1, int(notional // fill)) as sim._settle; gross and costs.round_trip_cost recomputed on "
                        "the real buy / sell values; fills, exits, slippage unchanged",
                 "implementation": "size10l_run.install(): a wrapper around sim._settle in this process and its forked "
                                   "workers; no lab module is edited"},
        "data": "the corrected view (stratlab/corrected.py, corr1) exactly as prereg/rerun_corrected_v1.json: 2023-26 "
                "minute bars of the 97 NIFTY-100 names (survivorship-biased), 2016-23 corrected old minute view with "
                "its dirty-link rule, 2012-23 corrected daily file with point-in-time NIFTY 100 membership",
        "judgement": {
            "discovery": {"window": [20231110, 20251216], "h1": [20231110, 20241129], "h2": [20241202, 20251216],
                          "gates": {"h1_net_gt_0": True, "h2_net_gt_0": True, "min_pf": P.SCREEN_MIN_PF,
                                    "min_trades": P.SCREEN_MIN_TRADES, "max_symbol_share": P.SCREEN_MAX_SYMBOL_SHARE},
                          "slippage_bps": {"stock": 2.0, "index": 1.0}},
            "final": {"window": [20251217, 20261109], "rule": "ONE look per DISCOVERY passer at the new size, written "
                      "once to size10l/final_ledger.json (first write wins); CONFIRMED = net > 0 and PF >= "
                      f"{P.FINAL_MIN_PF}"},
            "old_era": {"minute_2016_23": "for every FINAL-confirmed strategy: protocol_min_oos gates unchanged "
                                          f"(fingerprint {PM.GATES_FINGERPRINT}) on the corrected old minute view, "
                                          "new size, base 2 bps + its 5 bps sensitivity pass; one judgement, "
                                          "size10l/oldera_min_ledger.json. OOS-CONFIRMED = all gates met.",
                        "daily_2012_23": "where protocol_oos.eligibility() allows: protocol_oos.judge gates unchanged, "
                                         "corrected daily file, PIT NIFTY 100, new size; REPORTED next to the minute "
                                         "judgement (the OOS gate is the 2016-23 minute judgement, as in the "
                                         "corrected re-run's definition of proven)"},
            "dsr": {"min": DSR_MIN, "formula": "trials.dsr (Bailey & Lopez de Prado): daily Sharpe, skew, non-excess "
                                               "kurtosis",
                    "series": "daily net P&L (trials.daily_pnl convention: booked on the exit session) of DISCOVERY + "
                              "FINAL, 2023-11-10 .. 2026-11-09, at the new size",
                    "N": f"{N_BEFORE} (every trial so far: lab 2,552 + 16 builder development evaluations + 2,400 "
                         "builder candidates + 11 builder FINAL looks = the builder registry's N after batch 0006) + K "
                         f"({len(ch)}) + the number of FINAL looks opened by this re-test",
                    "V": "max(robust (1.4826 x MAD)^2 variance of the K re-tested strategies' DISCOVERY daily Sharpes "
                         "at the new size (ids with >= 1 trade), 1 / (T_days - 1)) - the builder's convention",
                    "report_only": "the same DSR with the ledger's robust V of the Rs 1 lakh screens "
                                   f"({T.var_sr_robust(rows)[0]:.6f})"},
            "proven": "DISCOVERY gates passed AND FINAL CONFIRMED AND 2016-23 minute OOS-CONFIRMED AND DSR >= 0.95 "
                      "(N, V above) AND the controls behave (below). The lab's earlier bar for catalogue strategies "
                      "(FINAL-CONFIRMED + OOS-CONFIRMED, without the DSR) is reported as its own line, labelled.",
            "no_softening": "0 proven is reported as 0; near misses are listed as failures"},
        "controls": {
            "placebo": "audit_run's 10 random-entry intraday controls (RAND_TIME30_s1..5, RAND_BRACKET05_s1..5) at "
                       "Rs 5 lakh on DISCOVERY: expected to FAIL the screen (no information); the count that passes is "
                       "reported",
            "planted": "audit_run's CHEAT_LOOKAHEAD_30BARS (a planted look-ahead edge) at Rs 5 lakh on DISCOVERY: must "
                       "PASS the screen (the pipeline can see an edge at this size)",
            "gate": "if any random control passes or the planted control fails, nothing is proven in this re-test",
            "invariance": "for every re-tested id the Rs-1-lakh twin computed in the same pass must equal its stored "
                          "corrected-data DISCOVERY row (trades, gross, cost, net per half); mismatches are listed "
                          "and such ids cannot be proven"},
        "trials": {"counted": "each of the K re-tests is a NEW trial (ledger id 'sz10l:<id>', stages size10l-screen / "
                              "size10l-final / size10l-oos); nothing is hidden",
                   "second_look": "the FINAL (2025-12-17 .. 2026-11-09) and old-era windows of these strategies were "
                                  "already looked at once at Rs 1 lakh (and again by the corrected-data recomputation). "
                                  "The re-test is a second, strongly correlated look at the same windows with one "
                                  "pre-fixed change (the size). It is counted in N; it is not an independent "
                                  "confirmation.",
                   "ledger_n_2023_26_before": T.count_trials(rows, "2023-26"), "n_before": N_BEFORE},
        "catalogue_fingerprints_sha256": hashlib.sha256(json.dumps(
            {i: REGISTRY[i].fingerprint for i in sorted(REGISTRY)}, sort_keys=True).encode()).hexdigest(),
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
    log(f"pre-registration written: K={len(ch)} sha256 {hashlib.sha256(blob).hexdigest()}")
    return doc


def stage_screen(jobs: int):
    from . import audit_run as AR
    pre = load_prereg()
    ids = pre["scope"]["retested_ids"]
    install()
    try:
        res = run3y(ids, AR.DISC[0], AR.DISC[1], jobs, tag="screen@5L")
    finally:
        uninstall()
    stored = _jload(os.path.join(SCRATCH, "audit", "rerun", "screen_corrected.json"), {})
    bad = [i for i in ids if not _same(stored[i], res[i]["l1"])]
    for i in ids:
        res[i]["l1_matches_stored"] = i not in bad
        res[i]["stored_1L"] = {k: stored[i][k] for k in ("passed", "reasons", "h1", "h2", "disc")}
    _jdump(os.path.join(OUT, "screen.json"), res)
    n1 = sum(1 for i in ids if stored[i]["passed"])
    log(f"screen @Rs 5 lakh: {sum(r['passed'] for r in res.values())}/{len(ids)} pass (Rs 1 lakh: {n1}); "
        f"Rs-1-lakh twin differs from the stored row for {len(bad)} ids {bad[:10]}")


def stage_controls(jobs: int):
    from . import audit_run as AR
    load_prereg()
    install()
    try:
        res = run3y([], AR.DISC[0], AR.DISC[1], jobs, controls=True, tag="controls@5L")
    finally:
        uninstall()
    rand = {i: r for i, r in res.items() if i.startswith("RAND_")}
    cheat = res.get("CHEAT_LOOKAHEAD_30BARS", {})
    out = {"random_passed": sorted(i for i, r in rand.items() if r["passed"]), "random_n": len(rand),
           "planted_passed": bool(cheat.get("passed")), "ok": not any(r["passed"] for r in rand.values())
           and bool(cheat.get("passed")),
           "rows": {i: {"passed": r["passed"], "reasons": r["reasons"], "disc": r["disc"], "l1_disc": r["l1"]["disc"],
                        "mean_notional": r["mean_notional"]} for i, r in res.items()}}
    _jdump(os.path.join(OUT, "controls.json"), out)
    log(f"controls @Rs 5 lakh: random passers {out['random_passed']} of {len(rand)}; planted passes: "
        f"{out['planted_passed']}; ok={out['ok']}")


def stage_final(jobs: int):
    from . import audit_run as AR
    from .api import REGISTRY
    load_prereg()
    scr = _jload(os.path.join(OUT, "screen.json"))
    passers = sorted(i for i, r in scr.items() if r["passed"])
    led = _jload(FINAL_LEDGER, {}) or {}
    todo = [i for i in passers if f"{i}|{AR.FINAL[0]}-{AR.FINAL[1]}|corr1|size10l" not in led]
    log(f"final: {len(passers)} DISCOVERY passers at the new size; one look for {len(todo)}")
    res = {}
    if todo:
        # the look is announced (ledger key with result None) BEFORE the FINAL return is computed
        for i in todo:
            led[f"{i}|{AR.FINAL[0]}-{AR.FINAL[1]}|corr1|size10l"] = {
                "fingerprint": REGISTRY[i].fingerprint, "opened": time.strftime("%Y-%m-%d %H:%M"), "final": None}
        _jdump(FINAL_LEDGER, led)
        install()
        try:
            res = run3y(todo, AR.FINAL[0], AR.FINAL[1], jobs, tag="final@5L")
        finally:
            uninstall()
        for i in todo:
            f = res[i]["final"]
            k = f"{i}|{AR.FINAL[0]}-{AR.FINAL[1]}|corr1|size10l"
            if led[k]["final"] is None:             # first write wins
                led[k]["final"] = {"trades": f["trades"], "net": f["net"], "pf": f["pf"], "gross": f["gross"],
                                   "cost": f["cost"], "confirmed": bool(f["net"] > 0 and f["pf"] >= 1.05)}
                led[k]["l1_final"] = {k2: res[i]["l1"]["final"][k2] for k2 in ("trades", "net", "pf", "gross", "cost")}
                led[k]["xnet"] = res[i]["xnet"]
                led[k]["mean_notional"] = res[i]["mean_notional"]
        _jdump(FINAL_LEDGER, led)
    out = {i: led[f"{i}|{AR.FINAL[0]}-{AR.FINAL[1]}|corr1|size10l"] for i in passers}
    _jdump(os.path.join(OUT, "final.json"), {"passers": passers, "results": out})
    log(f"final @Rs 5 lakh: {[(i, v['final']['confirmed'], v['final']['net'], v['final']['pf']) for i, v in out.items()]}")


def _confirmed() -> list:
    fin = _jload(os.path.join(OUT, "final.json"), {}) or {}
    return sorted(i for i, v in (fin.get("results") or {}).items() if v["final"] and v["final"]["confirmed"])


def stage_oldmin(jobs: int):
    """2016-23 minute, corrected old view, protocol_min_oos gates and runner (as rerun_corrected.stage_oldmin), new
    size, for every FINAL-confirmed strategy. One judgement each."""
    from . import catalog  # noqa: F401
    from . import corrected as C
    from . import protocol_min_oos as PM
    from .api import REGISTRY
    load_prereg()
    led = _jload(OLDMIN_LEDGER, {}) or {}
    want = [i for i in _confirmed() if f"{i}|2016-23|corr1|size10l" not in led]
    log(f"oldmin: FINAL-confirmed {_confirmed()}; to judge {want}")
    if want:
        C.install(minute=True, daily=False)
        ctx = PM.setup()
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
        log(f"oldmin: corrected view; {added} symbol-dates added to the dirty set")
        install()
        try:
            res = PM.run_catalogue([REGISTRY[i] for i in want], jobs, log=log)
        finally:
            uninstall()
        for i, r in res.items():
            led.setdefault(f"{i}|2016-23|corr1|size10l", {"when": time.strftime("%Y-%m-%d %H:%M"),
                                                           "result": PM.verdict_of(REGISTRY[i], r)})
        _jdump(OLDMIN_LEDGER, led)
    orig = _jload(os.path.join(SCRATCH, "audit", "ledgers", "oldera_min_corrected.json"), {}) or {}
    orig0 = _jload(PM.LEDGER, {}) or {}
    rep = {}
    for i in _confirmed():
        prev = [v["result"] for k, v in orig.items() if k.split("|")[0] == i] or \
               [v["result"] for k, v in orig0.items() if k.split("|")[0] == i]
        rep[i] = {"size10l": led[f"{i}|2016-23|corr1|size10l"]["result"], "at_1lakh": prev[0] if prev else None}
    _jdump(os.path.join(OUT, "oldera_min.json"), rep)
    log(f"oldmin @Rs 5 lakh: {[(i, v['size10l']['verdict'], v['size10l']['why']) for i, v in rep.items()]}")


def stage_olddaily(jobs: int):
    """2012-23 daily (report next to the minute judgement), corrected daily file, PIT NIFTY 100, new size."""
    from . import catalog  # noqa: F401
    from . import corrected as C
    from . import data_daily as DD
    from . import protocol_oos as PO
    from . import rerun_corrected as RC
    from .api import REGISTRY
    load_prereg()
    el_ids = {x[0] for x in PO.eligibility()[0]}
    ids = [i for i in _confirmed() if i in el_ids]
    log(f"olddaily: FINAL-confirmed and daily-eligible: {ids}")
    rep = {}
    if ids:
        C.install(minute=False, daily=True)
        cache = PO.setup()
        syms = DD.symbols(cache, members_only=False)
        iv = RC.nifty100_members_by_symbol(syms)
        run_syms = sorted(iv) + [s for s in ("_NIFTY50_min", "_NIFTYBANK_min") if s in DD.manifest(cache)["symbols"]
                                 or os.path.exists(os.path.join(cache, s + ".npz"))]
        DD.gate = RC._pit_gate(iv)
        install()
        try:
            with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
                simf = PO.make_simulator(pool, run_syms, 2.0, 1.0)
                sensf = PO.make_simulator(pool, run_syms, 5.0, 2.5)
                res = PO.evaluate([REGISTRY[i] for i in ids], simf, sensf,
                                  cost_model="2026-11/f2|corr1|nifty100pit|size10l", ledger_path=OLDDAILY_LEDGER,
                                  prereg=None, dataset="nse_old corrected size10l")
        finally:
            uninstall()
        for r in res:
            rep[r["id"]] = {k: r.get(k) for k in ("verdict", "why", "h1", "h2", "era", "top_symbol", "top_share",
                                                  "sens", "note")}
    _jdump(os.path.join(OUT, "oldera_daily.json"), rep)
    log(f"olddaily @Rs 5 lakh: {[(k, v['verdict']) for k, v in rep.items()]}")


def stage_live(jobs: int):
    """REPORT ONLY: AUC-FADE-2PCT / -3PCT and SWING-ATTN-ON-3X through their own modules on corrected 2023-26 data at
    Rs 1 lakh (must equal the stored modules.json) and at Rs 5 lakh. No verdict is re-opened."""
    from . import attention as AT
    from . import auction as AU
    from . import rerun_corrected as RC
    from . import trials as T
    load_prereg()
    ids = ("AUC-FADE-2PCT", "AUC-FADE-3PCT", "SWING-ATTN-ON-3X")
    cal = T.calendar_3y(SCRATCH)
    out = {}
    keep = (AU.NOTIONAL, AT.NOTIONAL)
    try:
        for tag, n in (("at_1lakh", LAKH), ("at_5lakh", 5 * LAKH)):
            AU.NOTIONAL, AT.NOTIONAL = n, n
            r = RC.module_runs(True)
            for i in ids:
                v = r[i]
                tr = v["_trades_disc"] + v["_trades_final"]
                td = [{"entry": int(t[2]), "exit": int(t[6]), "net": float(t[12]), "gross": float(t[10]),
                       "cost": float(t[11])} for t in tr]
                st = T.series_stats(td, cal, 20231110, 20261109)
                f = [x for x in td if x["entry"] >= 20251217]
                out.setdefault(i, {})[tag] = {
                    "screen_passed": v["screen"]["passed"], "disc": v["screen"]["disc"],
                    "final": v["final"], "full": {k: st[k] for k in ("trades", "net", "pf", "mean_gross", "mean_cost",
                                                                     "n_days", "sr_daily", "skew_daily", "kurt_daily")},
                    "final_mean_gross": sum(x["gross"] for x in f) / len(f) if f else None,
                    "final_mean_cost": sum(x["cost"] for x in f) / len(f) if f else None}
    finally:
        AU.NOTIONAL, AT.NOTIONAL = keep
    stored = (_jload(os.path.join(SCRATCH, "audit", "rerun", "modules.json"), {}) or {}).get("corrected", {})
    for i in ids:
        s = stored.get(i, {})
        a = out[i]["at_1lakh"]
        out[i]["reproduces_stored_1lakh"] = bool(
            s and abs(s["screen"]["disc"]["net"] - a["disc"]["net"]) < 0.005 and
            abs(s["final"]["net"] - a["final"]["net"]) < 0.005)
    out["unchanged"] = {"VRP-STRANGLE-016-WINGED": "option lots: size unchanged", "ev_pead_long_h60": "delivery "
                        "(equity_cnc): Rs 1 lakh, proportional fees; on a Rs 10 lakh account each position is scaled "
                        "down, PF unchanged"}
    _jdump(os.path.join(OUT, "live.json"), out)
    log("live @Rs 5 lakh (report only): " + str({i: (out[i]["at_1lakh"]["full"]["net"], out[i]["at_5lakh"]["full"]["net"],
                                                      out[i]["reproduces_stored_1lakh"]) for i in ids}))


def stage_dsr():
    from . import audit_run as AR
    from . import datagen_run as DR
    from . import trials as T
    pre = load_prereg()
    scr = _jload(os.path.join(OUT, "screen.json"))
    fin = (_jload(os.path.join(OUT, "final.json"), {}) or {}).get("results", {})
    cal = T.calendar_3y(SCRATCH)
    K = pre["scope"]["retested_K"]
    srs = {}
    for i, r in scr.items():
        if r["disc"]["trades"] > 0:
            srs[i] = sr_stats(daily_series(r["xnet"], cal, AR.DISC[0], AR.DISC[1]))
    t_days = len([d for d in cal if AR.DISC[0] <= d <= AR.DISC[1]])
    v_mad = DR.robust_var([s["sr_daily"] for s in srs.values()])
    v = max(v_mad, 1.0 / (t_days - 1))
    rows = T.TrialsLedger(os.path.join(SCRATCH, "stratlab_cache", "trials_v2.csv")).read()
    v_ledger = T.var_sr_robust([r for r in rows if not r["strategy_id"].startswith(ID_PREFIX)])[0]
    n = N_BEFORE + K + len(fin)
    out = {"N_before": N_BEFORE, "K": K, "final_looks": len(fin), "N": n, "V": v, "V_mad": v_mad,
           "V_floor": 1.0 / (t_days - 1), "V_ledger_1lakh_report_only": v_ledger,
           "sr0_daily": T.expected_max_sr(n, v), "sr0_annual": T.expected_max_sr(n, v) * 252 ** 0.5, "rows": {}}
    for i, f in fin.items():
        x = {int(k): v_ for k, v_ in scr[i]["xnet"].items()}
        for k, v_ in (f.get("xnet") or {}).items():
            x[int(k)] = x.get(int(k), 0.0) + v_
        st = sr_stats(daily_series(x, cal, AR.DISC[0], AR.FINAL[1]))
        d = T.dsr(st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"], n, v)
        d2 = T.dsr(st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"], n, v_ledger)
        out["rows"][i] = {**st, "sr_annual": st["sr_daily"] * 252 ** 0.5, "dsr": d["dsr"], "psr0": d["psr0"],
                          "dsr_ledger_v_report_only": d2["dsr"], "disc_sr_daily": srs.get(i, {}).get("sr_daily")}
    _jdump(os.path.join(OUT, "dsr.json"), out)
    log(f"dsr: N={n} V={v:.6f} sr0 annual {out['sr0_annual']:.2f}; "
        f"{[(i, round(r['dsr'], 4)) for i, r in out['rows'].items()]}")


def stage_ledger():
    """Append the K re-tests to trials_v2.csv as NEW trials (ids 'sz10l:<id>'); the ledger is backed up first."""
    import shutil
    from . import audit_run as AR
    from . import trials as T
    load_prereg()
    path = os.path.join(SCRATCH, "stratlab_cache", "trials_v2.csv")
    rows = T.TrialsLedger(path).read()
    n0 = T.count_trials(rows, "2023-26")
    first = {}
    for r in rows:
        if r["era"] == "2023-26" and not r["stage"].startswith("corrected"):
            first.setdefault(r["strategy_id"], r)
    bak = path + ".bak_size10l_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, bak)
    cal = T.calendar_3y(SCRATCH)
    scr = _jload(os.path.join(OUT, "screen.json"))
    new = []
    for i, r in scr.items():
        d, f0 = r["disc"], first.get(i, {})
        st = sr_stats(daily_series(r["xnet"], cal, AR.DISC[0], AR.DISC[1])) if d["trades"] else {}
        new.append({"strategy_id": ID_PREFIX + i, "family": f0.get("family") or T.family_of(i),
                    "group": f0.get("group", ""), "params_fp": f0.get("params_fp", ""), "era": "2023-26",
                    "protocol": "2023-26 corrected (corr1), Rs 10 lakh-per-agent size", "stage": "size10l-screen",
                    "dataset": "bars3y+corr1", "period": f"{AR.DISC[0]}-{AR.DISC[1]}", "cost_model": "2026-11/f2",
                    "trades": d["trades"], "net": d["net"], "pf": d["pf"],
                    "mean_gross": d["gross"] / d["trades"] if d["trades"] else None,
                    "mean_cost": d["cost"] / d["trades"] if d["trades"] else None,
                    "n_days": st.get("n_days"), "sr_daily": st.get("sr_daily"), "skew_daily": st.get("skew_daily"),
                    "kurt_daily": st.get("kurt_daily"),
                    "verdict": "PASSED SCREEN" if r["passed"] else "FAILED SCREEN",
                    "source": os.path.join(OUT, "screen.json"),
                    "note": "re-test at Rs 5 lakh per trade (size10l prereg): a NEW trial; " + ", ".join(r["reasons"])})
    fin = (_jload(os.path.join(OUT, "final.json"), {}) or {}).get("results", {})
    for i, v in fin.items():
        f = v["final"]
        new.append({"strategy_id": ID_PREFIX + i, "family": T.family_of(i), "params_fp": first.get(i, {}).get("params_fp", ""),
                    "era": "2023-26", "protocol": "2023-26 corrected (corr1), Rs 10 lakh-per-agent size",
                    "stage": "size10l-final", "dataset": "bars3y+corr1", "period": f"{AR.FINAL[0]}-{AR.FINAL[1]}",
                    "cost_model": "2026-11/f2", "trades": f["trades"], "net": f["net"], "pf": f["pf"],
                    "verdict": "CONFIRMED" if f["confirmed"] else "REJECTED ON FINAL", "source": FINAL_LEDGER,
                    "note": "second look at this FINAL window (first at Rs 1 lakh); counted in N"})
    om = _jload(os.path.join(OUT, "oldera_min.json"), {}) or {}
    for i, v in om.items():
        c = v["size10l"]
        e = c.get("era") or {}
        new.append({"strategy_id": ID_PREFIX + i, "family": T.family_of(i), "params_fp": first.get(i, {}).get("params_fp", ""),
                    "era": "2016-23", "protocol": "2016-23 minute OOS corrected (corr1), Rs 10 lakh-per-agent size",
                    "stage": "size10l-oos", "dataset": "corr1", "period": "2016-23", "cost_model": "2026-11/f2",
                    "trades": e.get("trades"), "net": e.get("net"), "pf": e.get("pf"), "verdict": c.get("verdict"),
                    "source": OLDMIN_LEDGER, "note": str(c.get("why"))[:200]})
    n_app = T.TrialsLedger(path).append(new)
    n1 = T.count_trials(T.TrialsLedger(path).read(), "2023-26")
    out = {"appended": n_app, "backup": bak, "ledger_n_2023_26_before": n0, "ledger_n_2023_26_after": n1}
    _jdump(os.path.join(OUT, "ledger.json"), out)
    log(f"ledger: {n_app} rows appended (backup {bak}); ledger N 2023-26 {n0} -> {n1}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.size10l_run")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--stage", default="")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)
    try:
        os.nice(10)
    except Exception:
        pass
    if a.preregister:
        preregister()
        print(f"written {PREREG} and {REPO_PREREG} (sha256 {_sha(PREREG)})")
        return 0
    st = {"verify": lambda: stage_verify(a.jobs), "screen": lambda: stage_screen(a.jobs),
          "controls": lambda: stage_controls(a.jobs), "final": lambda: stage_final(a.jobs),
          "oldmin": lambda: stage_oldmin(a.jobs), "olddaily": lambda: stage_olddaily(a.jobs),
          "live": lambda: stage_live(a.jobs), "dsr": stage_dsr, "ledger": stage_ledger}
    if a.stage == "report":
        from . import size10l_report
        size10l_report.write()
        return 0
    if a.stage not in st:
        raise SystemExit(f"unknown stage {a.stage!r}")
    st[a.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
