"""
forward_lab_build.py — offline tools of the forward lab (run on the lab machine, never on the server):

    python forward_lab_build.py baseline --scratch <lab scratch> [--jobs 4]   # writes forward_lab_baseline.json
    python forward_lab_build.py bench    --scratch <lab scratch> --day 20261106 [--jobs 1]
    python forward_lab_build.py parity   --scratch <lab scratch> [--days 3] [--sample 40]

baseline: the lab references the pre-registered rule needs, computed by the lab's own code on its 3-year data
(bars3y, 2023-11-10 .. 2026-11-09 = DISCOVERY + FINAL) at lab size:
  natives   stratlab.sim.run_symbol over the whole period (Rs 1 lakh per trade, default slippage), per id: trades, net,
            mean daily net and the max drawdown of DAILY net totals (net booked on the exit session, zero days
            included; trials.daily_pnl's convention)
  tfgrid    tfgrid.build_candidates + tfgrid.run_account over the whole period (Rs 10 lakh account), same numbers
  live4     the module runners (auction / attention / options_vrp) as tfgrid_run._module_trades does; PEAD from the lab's
            events trade file (events/run1/trades/ev_pead_long_h60_2023-26.csv); their full daily series are kept
            (the consistency band is block-bootstrapped from them)
  frozen    the tfgrid account numbers (capital.agents() with the config values of the pre-registration) and the stock
            futures lots (nsedata bhavcopy on 2026-11-09), so the server never reads the live config or lab files
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forward_lab_engine as E  # noqa: E402
import forward_lab_store as FS  # noqa: E402

LAB_LO, LAB_HI = 20231110, 20261109


def lab_cache(scratch: str) -> str:
    return os.path.join(scratch, "stratlab_cache", "bars3y_fb3d4d")


# ── a forward store built from the lab's cache (tests, parity, benchmark) ────────────────────────────────────────────
def store_from_lab_cache(cache: str, store_root: str, upto: int, n_sessions: int = FS.WARMUP_SESSIONS + 5,
                         symbols=None) -> FS.Store:
    """Copy the lab's bars (sessions <= upto, the last n_sessions of the NIFTY 50 calendar) into a forward store and
    mark every session COMPLETE (the lab's own bars)."""
    from stratlab import data as D
    st = FS.Store(store_root, cap_root=os.path.dirname(store_root))
    idx = D.load_symbol(cache, "_NIFTY50_min")
    days = [int(d) for d in idx.days if int(d) <= int(upto)][-n_sessions:]
    lo = days[0]
    for sym in (symbols or FS.MINUTE_SYMBOLS):
        p = os.path.join(cache, sym.replace("/", "_").replace(" ", "_") + ".npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            zz = {k: z[k] for k in z.files}
        keep = (zz["date"] >= lo) & (zz["date"] <= int(upto))
        st.write_sessions(sym, {k: v[keep] for k, v in zz.items()})
    for name in FS.DAILY_AUX:
        a = D.load_daily_aux(cache, name)
        if a is not None:
            st.write_aux(name, {k: v[a["date"] <= int(upto)] for k, v in a.items()})
    m = st.manifest()
    for d in days:
        m.setdefault("sessions", {})[str(d)] = {"status": "complete", "source": "lab bars3y"}
    st._save_manifest(m)
    return st


# ── daily series ──────────────────────────────────────────────────────────────────────────────────────────────────
def daily_series(trades, calendar: list[int], lo: int, hi: int) -> list[float]:
    """Net per session of `calendar` in [lo, hi]: trades whose ENTRY is in [lo, hi], net booked on the exit
    session (the next session if the exit day is missing; exits after hi on the last session); zero days kept."""
    import bisect
    days = [d for d in calendar if lo <= d <= hi]
    pos = {d: i for i, d in enumerate(days)}
    out = [0.0] * len(days)
    for t in trades:
        e, x, net = int(t[2]), int(t[6] or t[2]), float(t[12])
        if not lo <= e <= hi:
            continue
        i = pos.get(x)
        if i is None:
            i = min(bisect.bisect_left(days, x), len(days) - 1)
        out[i] += net
    return out


def max_dd(daily) -> float:
    c = np.cumsum(np.asarray(daily, float))
    if c.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate(([0.0], c)))[1:]
    return float(max(0.0, (peak - c).max()))


def ref_row(trades, cal, lo=LAB_LO, hi=LAB_HI, keep_daily=False) -> dict:
    d = daily_series(trades, cal, lo, hi)
    nets = [float(t[12]) for t in trades if lo <= int(t[2]) <= hi]
    gp, gl = sum(x for x in nets if x > 0), -sum(x for x in nets if x < 0)
    row = {"trades": len(nets), "net": round(sum(nets), 2), "days": len(d),
           "mean_daily": round(float(np.mean(d)) if d else 0.0, 4),
           "sd_daily": round(float(np.std(d, ddof=1)) if len(d) > 1 else 0.0, 4),
           "max_dd": round(max_dd(d), 2), "pf": round(gp / gl, 4) if gl > 0 else None}
    if keep_daily:
        row["daily"] = [round(x, 2) for x in d]
    return row


# ── baseline ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _nat_work(args):
    cache, sym, ids = args
    from stratlab import sim
    specs = E.native_specs()
    sps = [specs[i] for i in ids if sim.applies(specs[i], sym)]
    if not sps:
        return sym, {}
    r = sim.run_symbol(cache, sym, sps, LAB_LO, LAB_HI, audit=False)
    return sym, {i: (v["trades"], v["error"]) for i, v in r.items() if v["trades"] or v["error"]}


def frozen_accounts() -> dict:
    """The tfgrid account numbers exactly as pre-registered (stratlab/prereg/tfgrid_v1.json "account.agents"; the
    live config has changed since - the core rebuild removed those settings - so it is never read)."""
    pre = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stratlab", "prereg",
                                      "tfgrid_v1.json")))
    return {k: dict(v) for k, v in pre["account"]["agents"].items()}


def build_baseline(scratch: str, jobs: int = 4, out_path: str = E.BASELINE_FILE, log=print) -> dict:
    from stratlab import tfgrid as G, capital as K
    from stratlab import data as D
    cache = lab_cache(scratch)
    cal = [int(d) for d in D.load_symbol(cache, "_NIFTY50_min").days if LAB_LO <= int(d) <= LAB_HI]
    syms = E.symbol_table()
    base = {"written": time.strftime("%Y-%m-%d"), "lab_period": [LAB_LO, LAB_HI], "lab_sessions": len(cal),
            "data": "bars3y (stratlab_cache/bars3y_fb3d4d), original bars as in the lab's screens",
            "size": {"natives": "Rs 1 lakh per stock trade (sim.NOTIONAL), spec lots for index futures",
                     "tfgrid": "Rs 10 lakh account per combo (capital.simulate)",
                     "live4": "module runners: Rs 1 lakh per stock trade; strangle 2 lots; PEAD Rs 1 lakh"},
            "daily_convention": "net booked on the exit session, zero days included (trials.daily_pnl)",
            "tfgrid_accounts": frozen_accounts(),
            "stockfut_lots": {k: v for k, v in G.lot_sizes(sorted(FS.UNIVERSE_STOCKS)).items() if v},
            "refs": {}}
    # natives
    t0 = time.time()
    specs = E.native_specs()
    ids = list(specs)
    trades = {i: [] for i in ids}
    errs = {}
    import pickle
    part_dir = os.path.join(scratch, "fwdlab", "baseline_natives")
    os.makedirs(part_dir, exist_ok=True)
    todo = [s for s in syms if not os.path.exists(os.path.join(part_dir, s + ".pkl"))]
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
        for k, (sym, r) in enumerate(pool.imap_unordered(_nat_work, [(cache, s, ids) for s in todo]), 1):
            with open(os.path.join(part_dir, sym + ".pkl"), "wb") as fh:
                pickle.dump(r, fh)
            log(f"  natives {k}/{len(todo)} symbols ({time.time() - t0:.0f}s)")
    for s in syms:
        with open(os.path.join(part_dir, s + ".pkl"), "rb") as fh:
            r = pickle.load(fh)
        for i, (tr, er) in r.items():
            trades[i].extend(tr)
            if er:
                errs.setdefault(i, er)
    for i in ids:
        row = ref_row(trades[i], cal)
        if i in errs:
            row["error"] = errs[i]
        base["refs"][i] = row
    log(f"natives done in {time.time() - t0:.0f}s")
    # tfgrid
    t1 = time.time()
    lots = base["stockfut_lots"]
    ags = {k: K.AgentConfig(**v) for k, v in base["tfgrid_accounts"].items()}
    combos = E.tfgrid_combos()
    for tf in G.TFS:
        sel = [c for c in combos if c[2] == tf]
        rids = sorted({c[1] for c in sel})
        built, tsyms, _cpu = G.build_candidates(cache, tf, LAB_LO, LAB_HI, rids, jobs=jobs, symbols=sorted(FS.UNIVERSE_STOCKS))
        assert tsyms == syms, "symbol table differs"
        for cid, rid, _tf, uni, agent in sel:
            b = built[rid]
            if b["errors"]:
                base["refs"][cid] = {"trades": 0, "error": b["errors"][0]}
                continue
            res = G.run_account(b["index" if uni == "index" else "stocks"], tsyms, uni, agent, lots, cid,
                                agcfg=ags[agent])
            base["refs"][cid] = ref_row(res["trades"], cal)
        log(f"  tfgrid {tf}: {len(sel)} combos ({time.time() - t1:.0f}s)")
    # live 4
    from stratlab import tfgrid_run as TR
    G.BARS3Y = os.path.join(scratch, "bars3y")
    mod, _margins = TR._module_trades("2023-26")
    for sid in ("AUC-FADE-2PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED"):
        base["refs"][sid] = ref_row(mod[sid], cal, keep_daily=True)
    pead = []
    with open(os.path.join(scratch, "events", "run1", "trades", "ev_pead_long_h60_2023-26.csv")) as fh:
        for r in csv.DictReader(fh):
            pead.append((r["id"], r["symbol"], int(r["date"]), int(r["side"]), r["entry_t"], float(r["entry"]),
                         int(r["exit_date"]), r["exit_t"], float(r["exit"]), int(r["qty"]), float(r["gross"]),
                         float(r["cost"]), float(r["net"]), r["reason"]))
    base["refs"]["EV-PEAD-LONG-H60"] = ref_row(pead, cal, keep_daily=True)
    base["calendar_first_last"] = [cal[0], cal[-1]]
    base["n_refs"] = len(base["refs"])
    with open(out_path, "w") as fh:
        json.dump(base, fh, sort_keys=True, separators=(",", ":"))
    log(f"baseline written: {out_path} ({len(base['refs'])} refs, {os.path.getsize(out_path) / 1024:.0f} KB)")
    return base


# ── benchmark: one full session of every strategy ────────────────────────────────────────────────────────────────
def mini_baseline() -> dict:
    """The frozen parts of the baseline only (accounts, lots): enough to run the engine (benchmark, tests)."""
    from stratlab import tfgrid as G
    return {"tfgrid_accounts": frozen_accounts(),
            "stockfut_lots": {k: v for k, v in G.lot_sizes(sorted(FS.UNIVERSE_STOCKS)).items() if v}}


def bench(scratch: str, day: int, jobs: int = 1, n_sessions: int = FS.WARMUP_SESSIONS + 1, ids=None) -> dict:
    tmp = tempfile.mkdtemp(prefix="flab_bench_", dir=os.path.join(scratch, "fwdlab"))
    store = store_from_lab_cache(lab_cache(scratch), os.path.join(tmp, "bars"), day, n_sessions)
    prev = [d for d in store.sessions() if d < day][-1]
    t0 = time.time()
    r = E.run_session(str(store.root), os.path.join(tmp, "state"), day, prev, day, os.path.join(tmp, "work"),
                      jobs=jobs, ids=ids, baseline=E.load_baseline() if os.path.exists(E.BASELINE_FILE)
                      else mini_baseline())
    wall = time.time() - t0
    n = sum(len(v["closed"]) for v in r["strategies"].values())
    print(json.dumps({"day": day, "wall_s": round(wall, 1), "timing": r["timing"], "peak_rss_mb": r.get("peak_rss_mb"),
                      "strategies": len(r["strategies"]), "closed_trades": n, "errors": len(r["errors"]),
                      "store_mb": round(FS.dir_bytes(store.root) / 1048576, 1), "tmp": tmp}, indent=1))
    full = {sid: {"closed": v["closed"], "open": v["open"]} for sid, v in r["strategies"].items()}
    blob = gzip.compress(json.dumps(full, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    r["trades_file_bytes"] = len(blob)
    r["strategies_with_rows"] = sum(1 for v in full.values() if v["closed"] or v["open"])
    print(json.dumps({"trades_file_kb": round(len(blob) / 1024, 1), "strategies_with_rows": r["strategies_with_rows"]}))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return r


def warmup_check(scratch: str, day: int, jobs: int = 2) -> dict:
    """Every strategy on one session twice: store anchored at the lab's first session (the lab's own warm-up) vs the
    live WARMUP_SESSIONS window. Strategies whose output differs need more warm-up than the window (long-memory
    indicators): reported, not hidden."""
    a = bench(scratch, day, jobs, n_sessions=10 ** 6)
    b = bench(scratch, day, jobs, n_sessions=FS.WARMUP_SESSIONS + 1)
    diff = sorted(sid for sid in a["strategies"]
                  if json.dumps(a["strategies"][sid].get("closed")) != json.dumps(b["strategies"].get(sid, {}).get("closed"))
                  or json.dumps(a["strategies"][sid].get("open")) != json.dumps(b["strategies"].get(sid, {}).get("open")))
    out = {"day": day, "strategies": len(a["strategies"]), "differ": diff,
           "trades_anchored": sum(len(v["closed"]) for v in a["strategies"].values()),
           "trades_window": sum(len(v["closed"]) for v in b["strategies"].values())}
    print(json.dumps(out, indent=1))
    return out


# ── the pre-registered rule file ─────────────────────────────────────────────────────────────────────────────────
def write_rule(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_lab_rule.json"),
               written: str = "2026-11-15") -> dict:
    """Write forward_lab_rule.json (once, before any forward result; the commit that adds it is the timestamp)."""
    from statistics import NormalDist
    from shadow_registry import sha256_hex
    table = E.strategy_table()
    fwd = sorted(t["id"] for t in table if t["group"] == "forward")
    n_nat = sum(1 for t in table if t["kind"].startswith("native:"))
    n_tfg = sum(1 for t in table if t["kind"].startswith("tfgrid:"))
    alpha, looks = 0.05, 12
    a_test = alpha / (len(fwd) * looks)
    base_sha = hashlib.sha256(open(E.BASELINE_FILE, "rb").read()).hexdigest()
    rule = {
        "version": "forward-lab/v1",
        "written": written,
        "statement": "Written before any forward-lab result existed (no forward session had been run). Do not tune on "
                     "results: a changed rule is a new version, a changed strategy is a new id (one more test in the "
                     "family). Nothing here switches anything on: QUALIFIED / CONSISTENT are recorded findings, "
                     "trading changes and LIVE remain human decisions, and the kept strategies still refuse LIVE.",
        "groups": {
            "forward": {
                "members": "every catalogue strategy (stratlab.api.REGISTRY, native ids, sim.run_symbol at Rs 1 lakh "
                           "per trade) and every tfgrid portable combo (stratlab.tfgrid.combos(), 1:3 bracket, "
                           "Rs 10 lakh account)",
                "initial_family_size": len(fwd), "composition": {"natives": n_nat, "tfgrid": n_tfg},
                "initial_ids_sha256": sha256_hex(fwd),
                "family_rule": "family_size = every forward strategy ever registered under this rule version "
                               "(demoted, drifted and removed ones included); it never decreases; strategies added "
                               "later raise it for later looks only",
            },
            "proven": ["AUC-FADE-2PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED"],
            "candidate": ["EV-PEAD-LONG-H60"],
            "proven_candidate_sources": {
                "AUC-FADE-2PCT": "stratlab.auction module runner on the forward bars (Rs 1 lakh, 2.1 bps)",
                "SWING-ATTN-ON-3X": "stratlab.attention module runner (Rs 1 lakh, futures costs); entries of a "
                                    "session close at the next session's open",
                "VRP-STRANGLE-016-WINGED": "stratlab.options_vrp (options_replay, Black-Scholes on the NIFTY minute "
                                           "bars, previous VIX close, 2 lots); pending until the expiry day's "
                                           "15:00-15:29 closes exist",
                "EV-PEAD-LONG-H60": "the agent's own performance-ledger rows (pead_h60), normalised to Rs 1 lakh "
                                    "per trade",
            },
        },
        "daily_pnl": "per COMPLETE forward session since the strategy's registration: net (after the lab's costs and "
                     "slippage) of the trades that EXIT that session, zero when none (trials.daily_pnl's convention); "
                     "INCOMPLETE / skipped sessions are not in the series",
        "looks": {"every_sessions": 20, "max_looks": looks,
                  "rule": "look k at the 20k-th forward session of the strategy (k = 1..12, about one year); the "
                          "sample is exactly its first 20k sessions; each look is evaluated once, written to the "
                          "registry chain and never recomputed"},
        "qualification": {
            "group": "forward", "alpha_family": alpha, "min_trades": 50, "min_days": 40,
            "alpha_per_test": "alpha_family / (family_size x max_looks) (Bonferroni over the family and the looks)",
            "alpha_per_test_at_initial_family": a_test,
            "z_required_at_initial_family": round(NormalDist().inv_cdf(1 - a_test), 6),
            "test": "H0: mean daily net <= 0. Stationary block bootstrap (Politis-Romano) of the CENTRED daily "
                    "series; z = mean / sd(bootstrap means); p = 1 - Phi(z). QUALIFIED iff mean > 0 and "
                    "z >= z_required (equivalently p <= alpha_per_test), with >= min_trades closed trades and "
                    ">= min_days sessions in the look sample; looks short of either are recorded 'insufficient'",
            "bootstrap": {"kind": "stationary", "mean_block": 5, "resamples": 10000,
                          "seed": "first 8 bytes of sha256('<id>|look<k>') (numpy default_rng)"},
            "permanence": "QUALIFIED is permanent unless the strategy is later demoted",
        },
        "consistency": {
            "group": "proven / candidate", "min_trades": 30, "quantile": 0.01,
            "test": "H0: live daily mean equals the lab daily mean (2023-26, forward_lab_baseline.json). At each look "
                    "with >= 30 closed trades: band = 1 % quantile of the means of n-session stationary-bootstrap "
                    "samples of the lab's daily series (n = the look's sessions); INCONSISTENT iff the live mean < "
                    "band (one-sided 99 %), CONSISTENT otherwise. INCONSISTENT is latched. Not a promotion.",
            "bootstrap": {"kind": "stationary", "mean_block": 5, "resamples": 10000,
                          "seed": "first 8 bytes of sha256('<id>|consistency<k>')"},
        },
        "demotion": {
            "groups": "all", "dd_multiple": 1.5,
            "rule": "at every evaluation: forward max drawdown of the daily series > 1.5 x the lab's 2023-26 max "
                    "drawdown of daily net totals of that strategy at lab size (forward_lab_baseline.json refs.max_dd) "
                    "-> DEMOTED, permanently; no reference (no lab trades) -> demotion cannot fire",
        },
        "data": {
            "universe": "forward_test.UNIVERSE (the lab's 97 NIFTY-100 symbols of bars3y) + _NIFTY50_min + "
                        "_NIFTYBANK_min (Kite 1-minute) + _NIFTY50_day + _VIX_day (Kite day candles)",
            "warmup_sessions": FS.WARMUP_SESSIONS,
            "store": "anchored: the warm-up is fetched once, one session is appended per day, nothing is deleted; "
                     "positional strategies use daily bars derived from the minute bars, as in the lab",
            "complete": f"every listed instrument has >= {FS.MIN_COMPLETE_BARS} bars, first bar 09:15, last bar >= "
                        "15:14, and both daily series have the session",
            "incomplete": "retried once at the next session's run; still incomplete -> recorded INCOMPLETE and skipped "
                          "for every strategy (never partial); holidays and special sessions are never run",
            "corporate_actions": f"Kite-adjusted history: when Kite's bars of the previous stored session differ from "
                                 f"the stored ones by > {FS.CA_TOLERANCE * 100:.2f} % (median close ratio), or the NSE "
                                 "corporate-action calendar lists a price-adjusting action ex on the session, the "
                                 "symbol's whole stored window is refetched and replaced; dividends not adjusted "
                                 "(as-traded, the lab's convention); booked trades never change",
            "engine_errors": "an engine failure is retried at the next run; a second failure records the session "
                             "as 'error' (skipped for every strategy)",
        },
        "schedule": {"time": "16:45 IST on every trading day", "strategies": "all strategies every session",
                     "catch_up": "missed sessions are processed in order, at most 5 per run"},
        "baseline_file": "forward_lab_baseline.json", "baseline_sha256": base_sha,
    }
    with open(path, "w") as fh:
        json.dump(rule, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return rule


# ── parity: the forward engine, day by day, against the lab's own run of the same sessions ───────────────────────
PARITY_SYMBOLS = ("HDFCBANK", "INFY", "PAYTM", "RELIANCE", "TATASTEEL", "VEDL")


def parity_sample(n_native: int = 24, n_tfg: int = 16) -> list[str]:
    """A deterministic spread of strategies: natives of every kind (intraday, scalping, swing, futures, index
    futures, decision-time positional) and tfgrid combos of every timeframe and universe, plus two live rules."""
    specs = E.native_specs()
    groups: dict = {}
    for sid, sp in specs.items():
        k = (sp.kind, sp.horizon, bool(sp.underlying), sp.decision_time is not None)
        groups.setdefault(k, []).append(sid)
    nat = []
    while len(nat) < n_native and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(nat) < n_native:
                ids = groups[k]
                nat.append(ids.pop(int(hashlib.sha1(k.__repr__().encode()).hexdigest(), 16) % len(ids)))
    combos = E.tfgrid_combos()
    tfg = []
    by = {}
    for c in combos:
        by.setdefault((c[2], c[3]), []).append(c[0])
    keys = sorted(by)
    i = 0
    while len(tfg) < n_tfg and i < 10 * len(keys):
        k = keys[i % len(keys)]
        lst = by[k]
        j = int(hashlib.sha1(f"{k}|{i}".encode()).hexdigest(), 16) % len(lst)
        if lst[j] not in tfg:
            tfg.append(lst[j])
        i += 1
    return nat + tfg + ["AUC-FADE-2PCT", "SWING-ATTN-ON-3X"]


def _norm(t) -> tuple:
    return tuple(t)


def parity(scratch: str, days: list[int], ids: list[str], symbols=PARITY_SYMBOLS, anchored: bool = True,
           tmp: str | None = None, log=print) -> dict:
    """Forward engine over `days` (day by day, state carried) vs the lab's own run with lo = days[0], hi = days[-1]
    on the full lab data. anchored=True: the forward store starts where the lab's data start (identical input);
    False: a WARMUP_SESSIONS window (the live setup). Returns per-strategy comparison."""
    from stratlab import sim, tfgrid as G, capital as K
    from stratlab import auction as A, attention as T
    from stratlab import data as D
    cache = lab_cache(scratch)
    tmp = tmp or tempfile.mkdtemp(prefix="flab_parity_")
    syms = sorted(symbols)
    n_sess = 10 ** 6 if anchored else FS.WARMUP_SESSIONS + len(days)
    store = store_from_lab_cache(cache, os.path.join(tmp, "bars"), days[-1], n_sess, syms + list(FS.INDEX_MIN))
    baseline = E.load_baseline() if os.path.exists(E.BASELINE_FILE) else mini_baseline()
    state = os.path.join(tmp, "state")
    fwd = {i: {"closed": [], "open": []} for i in ids}
    cal = store.sessions()
    for k, d in enumerate(days):
        prev = [x for x in cal if x < d][-1]
        work = os.path.join(tmp, f"work{d}")
        r = E.run_session(str(store.root), state, d, prev, days[0], work, jobs=0, ids=ids, baseline=baseline)
        for i in ids:
            v = r["strategies"].get(i) or {"closed": [], "open": []}
            fwd[i]["closed"] += [t for t in v["closed"] if t[1] >= days[0]]
            fwd[i]["open"] = v["open"]
        import shutil
        if os.path.exists(state):
            shutil.rmtree(state)
        if os.path.exists(os.path.join(work, "state_new")):
            shutil.copytree(os.path.join(work, "state_new"), state)
    lo, hi = days[0], days[-1]
    lab: dict = {i: [] for i in ids}
    specs = E.native_specs()
    nat = [specs[i] for i in ids if i in specs]
    for s in syms + list(G.INDEX_SYMS):
        sps = [sp for sp in nat if sim.applies(sp, s)]
        if sps:
            for i, v in sim.run_symbol(cache, s, sps, lo, hi, audit=False).items():
                lab[i] += [E.compact(t) for t in v["trades"]]
    combos = [c for c in E.tfgrid_combos() if c[0] in ids]
    ags = {k: K.AgentConfig(**v) for k, v in baseline["tfgrid_accounts"].items()}
    for tf in sorted({c[2] for c in combos}):
        sel = [c for c in combos if c[2] == tf]
        built, tsyms, _cpu = G.build_candidates(cache, tf, lo, hi, sorted({c[1] for c in sel}), jobs=1, symbols=syms)
        for cid, rid, _tf, uni, agent in sel:
            res = G.run_account(built[rid]["index" if uni == "index" else "stocks"], tsyms, uni, agent,
                                baseline["stockfut_lots"], cid, agcfg=ags[agent])
            lab[cid] = [E.compact(t) for t in res["trades"]]
    if "AUC-FADE-2PCT" in ids or "SWING-ATTN-ON-3X" in ids:
        arows, trows, counts = [], [], {}
        for s in syms:
            sd = D.load_symbol(cache, s)
            arows += A.candidates_for(s, A.session_features(sd), 0.02)
            trows += T.session_features(sd)
            for dd, nb in zip(sd.days, sd.ends - sd.starts):
                counts.setdefault(int(dd), []).append(int(nb))
            for dd in sd.dropped or []:
                counts.setdefault(int(dd), []).append(0)
        a = A.make_simulator(arows)([x for x in A.SPECS if x.id == "AUC-FADE-2PCT"], lo, hi)
        lab["AUC-FADE-2PCT"] = [E.compact(t) for t in a["AUC-FADE-2PCT"]["trades"]]
        el, _sp = T.filter_rows(trows, counts)
        t_ = T.make_simulator(el)([x for x in T.SPECS if x.id == "SWING-ATTN-ON-3X"], lo, hi)
        lab["SWING-ATTN-ON-3X"] = [E.compact(t) for t in t_["SWING-ATTN-ON-3X"]["trades"]]
    out = {"days": days, "anchored": anchored, "strategies": {}, "tmp": tmp}
    for i in ids:
        lt = lab.get(i, [])
        l_closed = sorted(_norm(t) for t in lt if t[12] not in ("period_end", "data_end"))
        l_open = sorted(tuple(t[:5]) for t in lt if t[12] in ("period_end", "data_end"))
        f_closed = sorted(_norm(t) for t in fwd[i]["closed"])
        f_open = sorted(tuple(t[:5]) for t in fwd[i]["open"] if t[12] != "pending")
        if i == "SWING-ATTN-ON-3X":
            f_open, l_open = [], []          # today's picks exit tomorrow: the lab has no row for them yet
        same = (l_closed == f_closed) and (l_open == f_open)
        out["strategies"][i] = {"same": same, "lab_closed": len(l_closed), "fwd_closed": len(f_closed),
                                "lab_open": len(l_open), "fwd_open": len(f_open)}
        if not same:
            out["strategies"][i]["diff"] = {"lab_only": [x for x in l_closed if x not in f_closed][:3],
                                            "fwd_only": [x for x in f_closed if x not in l_closed][:3],
                                            "open_lab": l_open[:3], "open_fwd": f_open[:3]}
    n_same = sum(1 for v in out["strategies"].values() if v["same"])
    out["matched"] = n_same
    out["n"] = len(ids)
    out["trades_compared"] = sum(v["lab_closed"] + v["lab_open"] for v in out["strategies"].values())
    log(f"parity ({'anchored' if anchored else 'window'}): {n_same}/{len(ids)} strategies identical, "
        f"{out['trades_compared']} lab trades/positions compared")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("baseline", "bench", "parity", "rule", "warmup"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--day", type=int, default=20261106)
    ap.add_argument("--days", default="20261026,20261027,20261028,20261029,20261030")
    a = ap.parse_args(argv)
    if a.cmd == "baseline":
        build_baseline(a.scratch, a.jobs)
    elif a.cmd == "warmup":
        warmup_check(a.scratch, a.day, a.jobs)
    elif a.cmd == "rule":
        print(json.dumps(write_rule(), indent=1)[:400])
    elif a.cmd == "bench":
        bench(a.scratch, a.day, a.jobs)
    elif a.cmd == "parity":
        days = [int(x) for x in a.days.split(",")]
        for anchored in (True, False):
            r = parity(a.scratch, days, parity_sample(), anchored=anchored)
            print(json.dumps({k: v for k, v in r["strategies"].items() if not v["same"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
