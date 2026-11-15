"""
forward_lab_engine.py — runs ONE forward session of every lab strategy through the UNCHANGED lab code. It is started
by forward_lab.py in a separate process (never inside the API process), reads the bar store and the previous engine
state, and writes its results + the proposed next state into a work directory. It places no order, reads no agent,
writes no ledger (the server ingests the results).

    python -m forward_lab_engine --store <bars dir> --state <state dir> --day 20261116 --since 20261113 \
        --fwd-start 20261116 --work <dir> [--jobs 1] [--ids a,b]

What runs (the frozen family, forward_lab_rule.json):
  natives      every catalogue @strategy (stratlab.api.REGISTRY, 608 ids) through stratlab.sim.run_symbol, the lab's
               screen setup: Rs 1 lakh per stock trade, the spec's own stops / targets / exits, sim's default
               slippage (2 bps stocks, 1 bps index) and costs.round_trip_cost. Intraday specs: lo = hi = the session.
               Positional specs: lo = the session, or - while a virtual position of that spec / symbol is open - lo =
               its entry date, so the walk re-derives the open trade and continues exactly as the lab's walk would
               (sim_positional is deterministic and causal; the store is anchored, so earlier sessions never change).
  tfgrid       every portable combo (stratlab.tfgrid.combos(), 668): candles + rule + tfgrid.gen (1:3 bracket, the
               lab's minute fill rules) per instrument, then tfgrid.run_account (Rs 10 lakh account, capital.simulate)
               with the account numbers and stock-futures lots frozen in forward_lab_baseline.json. Intraday
               timeframes are flat every night, so their account runs on the session's candidates alone; the daily /
               weekly candidates are kept (state/tfg_cands) and their accounts re-run from the forward start every
               session (capital.simulate walks in time order and only uses what is known at each fill).
  live4        AUC-FADE-2PCT (stratlab.auction), SWING-ATTN-ON-3X (stratlab.attention), VRP-STRANGLE-016-WINGED
               (stratlab.options_vrp: options_replay Black-Scholes on the NIFTY minute bars with the previous VIX
               close; a structure is "pending" until its expiry day's 15:00-15:29 closes exist). EV-PEAD-LONG-H60 is
               copied from the agent's own ledger by forward_lab.py.
Measured on the lab machine (one session, 2026-11-06, all 1279 engine strategies, 99 instruments, one worker
process): 302 s with the 420-session warm-up store (worker peak RSS ~0.5 GB, 15,205 closed trades), 509 s / ~1.0 GB
with a 726-session store (the anchored store after ~14 months). Warm-up check (forward_lab_build.py warmup): the 420-
session window and a store starting at the lab's first session give identical trades for 1276 of 1279 strategies;
r3_intraday_vwap3sd_fade and vwap_2sd_fade (session VWAP is a running sum over the whole store, last-bit rounding)
and r5_kite_vars_ema100_200 (EMA-200 of daily closes, long memory) differ in a few prices: they depend on where the
data start, in the lab as in the forward store.
A trade belongs to the session it EXITS on for the daily P&L (trials.daily_pnl's convention); open positions are
listed, never booked. Nothing after `day` is loaded: every bar file is cut at `day`, the daily series are copied
truncated, and the session must be marked COMPLETE in the store manifest (asserted).
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ENGINE_VERSION = "fle1"
TF_POSITIONAL = ("1d", "1w")


# ── the strategy table ────────────────────────────────────────────────────────────────────────────────────────────
def native_specs() -> dict:
    from stratlab import catalog  # noqa: F401
    from stratlab.api import REGISTRY
    return dict(sorted(REGISTRY.items()))


def tfgrid_combos() -> list:
    from stratlab import tfgrid as G
    return G.combos()


LIVE4 = {"AUC-FADE-2PCT": ("proven", "opening_auction_fade", "stratlab.auction"),
         "SWING-ATTN-ON-3X": ("proven", "attention_overnight", "stratlab.attention"),
         "VRP-STRANGLE-016-WINGED": ("proven", "vrp_short_strangle", "stratlab.options_vrp"),
         "EV-PEAD-LONG-H60": ("candidate", "post_earnings_drift", "agents.validated_pead (agent ledger)")}


def strategy_table() -> list[dict]:
    """Every forward-lab strategy: id, family, group, kind, fingerprint (sha1 of the rule source + params)."""
    import hashlib
    from stratlab import tfgrid as G
    out = []
    for sid, sp in native_specs().items():
        out.append({"id": sid, "family": sp.family, "group": "forward", "kind": f"native:{sp.kind}",
                    "fingerprint": sp.fingerprint})
    for cid, rid, tf, uni, agent in tfgrid_combos():
        r = G.RULES[rid]
        out.append({"id": cid, "family": r.family, "group": "forward", "kind": f"tfgrid:{tf}:{uni}:{agent}",
                    "fingerprint": G.rule_fingerprint(rid)})
    for sid, (grp, fam, src) in LIVE4.items():
        out.append({"id": sid, "family": fam, "group": grp, "kind": f"live:{src}",
                    "fingerprint": hashlib.sha1(src.encode()).hexdigest()[:12]})
    return out


def symbol_table() -> list[str]:
    """tfgrid.build_candidates' symbol table (sorted stocks, then the two indices): the account's tie order."""
    from stratlab import tfgrid as G
    import forward_lab_store as FS
    return sorted(FS.UNIVERSE_STOCKS) + list(G.INDEX_SYMS)


# ── trades ────────────────────────────────────────────────────────────────────────────────────────────────────────
def trade_key(t) -> str:
    """Identity of a lab trade tuple (sim.TRADE_FIELDS): symbol | entry date | entry time | side."""
    return f"{t[1]}|{int(t[2])}|{t[4]}|{int(t[3])}"


def compact(t) -> list:
    """sim.TRADE_FIELDS without the id: symbol, date, side, entry_t, entry, exit_date, exit_t, exit, qty, gross,
    cost, net, reason."""
    return [str(t[1]), int(t[2]), int(t[3]), str(t[4]), float(t[5]), int(t[6] or 0), str(t[7]), float(t[8]),
            int(t[9]), float(t[10]), float(t[11]), float(t[12]), str(t[13])]


# ── per-symbol work (worker processes) ───────────────────────────────────────────────────────────────────────────
_CTX: dict = {}


def _load(store_root: str, sym: str, day: int):
    import forward_lab_store as FS
    return FS.Store(store_root).load_symbol(sym, upto=day)


def _symbol_task(args) -> dict:
    (store_root, cache_dir, sym, code, day, native_ids, native_open, tfg_rids, tfg_lo, want_live) = args
    from stratlab import sim, tfgrid as G
    from stratlab import auction as A, attention as T
    out = {"sym": sym, "native": {}, "tfg": {}, "live": {}, "errors": [], "secs": {}}
    t0 = time.time()
    sd = _load(store_root, sym, day)
    if sd is None or int(sd.days[-1]) != int(day):
        out["errors"].append(f"{sym}: no session {day} in the store")
        return out
    idx = None
    if sym != "_NIFTY50_min":
        idx = _load(store_root, "_NIFTY50_min", day)
    else:
        idx = sd
    specs = native_specs()
    # natives: intraday specs on the session, positional specs grouped by their lo
    nat = [specs[i] for i in native_ids if i in specs and sim.applies(specs[i], sym)]
    intr = [sp for sp in nat if sp.horizon == "intraday"]
    groups: dict = {}
    for sp in nat:
        if sp.horizon == "positional":
            lo = int((native_open.get(sp.id) or {}).get(sym, day))
            groups.setdefault(lo, []).append(sp)
    res = {}
    try:
        if intr:
            res.update(sim.run_symbol(cache_dir, sym, intr, day, day, audit=False, sd=sd, index=idx))
        for lo, sps in sorted(groups.items()):
            res.update(sim.run_symbol(cache_dir, sym, sps, lo, day, audit=False, sd=sd, index=idx))
    except Exception as exc:
        out["errors"].append(f"{sym} natives: {type(exc).__name__}: {str(exc)[:200]}")
    for i, r in res.items():
        if i in {sp.id for sp in nat}:
            out["native"][i] = {"trades": [compact(t) for t in r["trades"]], "error": r["error"]}
    out["secs"]["native"] = time.time() - t0
    # tfgrid candidates
    t1 = time.time()
    if tfg_rids:
        B = G.Bars(sd, hi=day)
        index = sym in G.INDEX_SYMS
        for tf in G.TFS:
            rids = [r for r in tfg_rids.get(tf, []) if not (index and G.RULES[r].needs_volume)]
            if not rids:
                continue
            C = G.candles(B, tf)
            for rid in rids:
                lo = int(tfg_lo.get(f"{rid}@{tf}", day))
                try:
                    with np.errstate(all="ignore"):
                        S = G.RULES[rid].fn(C)
                    g = G.gen(B, C, S, positional=tf in G.POSITIONAL_TFS, index=index, lo=lo, hi=day)
                    out["tfg"][f"{rid}@{tf}"] = G.to_columns(B, g, code)
                except Exception as exc:
                    out["tfg"][f"{rid}@{tf}"] = {"error": f"{sym}: {type(exc).__name__}: {str(exc)[:160]}"}
    out["secs"]["tfgrid"] = time.time() - t1
    # live rules' per-symbol features (stocks only)
    if want_live and sym not in G.INDEX_SYMS:
        try:
            loosest = min(sp.params["min_gap"] for sp in A.SPECS)
            feats = A.session_features(sd)
            out["live"]["auction"] = [r for r in A.candidates_for(sym, feats, loosest) if r[0] == int(day)]
            rows = T.session_features(sd)
            keep = set(int(x) for x in sd.days[-3:])
            out["live"]["attention"] = [r for r in rows if r["date"] in keep]
            cnt = {int(d): int(n) for d, n in zip(sd.days[-6:], (sd.ends - sd.starts)[-6:])}
            for d in sd.dropped or []:
                cnt.setdefault(int(d), 0)
            out["live"]["counts"] = cnt
        except Exception as exc:
            out["errors"].append(f"{sym} live rules: {type(exc).__name__}: {str(exc)[:200]}")
    out["secs"]["total"] = time.time() - t0
    return out


# ── engine state ──────────────────────────────────────────────────────────────────────────────────────────────────
def _jload(p, default):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return default


def _cand_file(state_dir, key: str) -> Path:
    return Path(state_dir) / "tfg_cands" / (key.replace("@", "__") + ".npz")


_CAND_KEYS = ("sym", "side", "ekey", "fill", "stop", "xkey", "xpx", "reason", "prio", "cday")


def load_cands(state_dir, key: str) -> dict:
    from stratlab import tfgrid as G
    p = _cand_file(state_dir, key)
    if not p.exists():
        z = G.concat([])
        z["cday"] = np.zeros(0, np.int64)
        return z
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def save_cands(state_dir, key: str, cols: dict) -> None:
    p = _cand_file(state_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp.npz"
    np.savez_compressed(tmp, **{k: np.asarray(cols[k]) for k in _CAND_KEYS})
    os.replace(tmp, p)


def open_cand_mask(cols: dict) -> np.ndarray:
    """A daily / weekly candidate whose outcome is still 'period_end' at the session it was computed on (the data
    ended before its stop / target / exit signal / max hold) is an OPEN position: recomputed next session."""
    from stratlab import tfgrid as G
    if not len(cols["ekey"]):
        return np.zeros(0, bool)
    return (np.asarray(cols["reason"]) == G.REASONS.index("period_end")) & \
           ((np.asarray(cols["xkey"]) // 10000) >= np.asarray(cols["cday"]))


# ── one session ──────────────────────────────────────────────────────────────────────────────────────────────────
def frozen_agents(baseline: dict) -> dict:
    from stratlab import capital as K
    return {k: K.AgentConfig(**v) for k, v in baseline["tfgrid_accounts"].items()}


def run_session(store_root: str, state_dir: str, day: int, since: int, fwd_start: int, work_dir: str,
                jobs: int = 1, ids=None, baseline: dict | None = None, live: bool = True, log=print) -> dict:
    """Run the session `day` (must be COMPLETE in the store). Closed trades returned are those with an exit date in
    (since, day]; the proposed next state is written to <work_dir>/state_new. Returns the result dict (also written
    to <work_dir>/result.json.gz)."""
    import forward_lab_store as FS
    from stratlab import tfgrid as G
    t0 = time.time()
    day, since, fwd_start = int(day), int(since), int(fwd_start)
    store = FS.Store(store_root)
    st = store.session_status(day)
    if st != "complete":
        raise RuntimeError(f"session {day} is not COMPLETE in the store (status {st!r}): refusing to run")
    if baseline is None:
        baseline = load_baseline()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    new_state = work / "state_new"
    new_state.mkdir(parents=True, exist_ok=True)
    cache_dir = store.write_lab_cache(work / "labcache", day)
    specs = native_specs()
    combos = tfgrid_combos()
    want = set(ids) if ids else None
    native_ids = [i for i in specs if want is None or i in want]
    sel_combos = [c for c in combos if want is None or c[0] in want]
    tfg_rids: dict = {}
    for cid, rid, tf, uni, agent in sel_combos:
        tfg_rids.setdefault(tf, [])
        if rid not in tfg_rids[tf]:
            tfg_rids[tf].append(rid)
    native_open = _jload(Path(state_dir) / "native_open.json", {})
    syms = symbol_table()
    code = {s: i for i, s in enumerate(syms)}
    # open positional candidates -> per (rid@tf) per symbol the earliest open entry date
    stored = {}
    tfg_lo_by_sym: dict = {s: {} for s in syms}
    for tf in TF_POSITIONAL:
        for rid in tfg_rids.get(tf, []):
            key = f"{rid}@{tf}"
            c = load_cands(state_dir, key)
            stored[key] = c
            m = open_cand_mask(c)
            for i in np.flatnonzero(m):
                s = syms[int(c["sym"][i])]
                d = int(c["ekey"][i]) // 10000
                cur = tfg_lo_by_sym[s].get(key)
                tfg_lo_by_sym[s][key] = d if cur is None else min(cur, d)
    present = set(store.symbols())
    want_live = bool(live and (want is None or want & set(LIVE4)))
    tasks = [(str(store.root), cache_dir, s, code[s], day, native_ids, native_open, tfg_rids, tfg_lo_by_sym[s],
              want_live) for s in syms if s in present]
    results = []
    if jobs <= 0:                       # in-process (tests)
        for a in tasks:
            results.append(_symbol_task(a))
    else:                               # worker processes, recycled every 10 symbols (bounded memory)
        with mp.get_context("fork").Pool(jobs, maxtasksperchild=10) as pool:
            for r in pool.imap(_symbol_task, tasks):
                results.append(r)
    results.sort(key=lambda r: code.get(r["sym"], 0))
    t_sym = time.time() - t0
    out = {"day": day, "since": since, "fwd_start": fwd_start, "engine": ENGINE_VERSION, "strategies": {},
           "errors": [], "timing": {"symbols": round(t_sym, 1)}, "symbols": len(results)}
    errors = out["errors"]
    for r in results:
        errors.extend(r["errors"])
    # natives
    nat_trades: dict = {i: [] for i in native_ids}
    nat_err: dict = {}
    for r in results:
        for i, v in r["native"].items():
            nat_trades[i].extend(v["trades"])
            if v["error"]:
                nat_err.setdefault(i, v["error"])
    new_open: dict = {}
    for i in native_ids:
        closed, opened = [], []
        for t in nat_trades[i]:
            if t[12] == "data_end" and t[5] == day:
                opened.append(t)
            elif since < t[5] <= day:
                closed.append(t)
        for t in opened:
            new_open.setdefault(i, {})[t[0]] = int(t[1])
        out["strategies"][i] = {"closed": closed, "open": opened, "error": nat_err.get(i)}
    with open(new_state / "native_open.json", "w") as fh:
        json.dump(new_open, fh, sort_keys=True)
    # tfgrid candidates -> accounts
    t2 = time.time()
    agents = frozen_agents(baseline)
    lots = {k: v for k, v in baseline["stockfut_lots"].items()}
    merged: dict = {}
    errs_by_key: dict = {}
    for r in results:
        for key, cols in r["tfg"].items():
            if "error" in cols:
                errs_by_key.setdefault(key, cols["error"])
                continue
            merged.setdefault(key, []).append(cols)
    keys = sorted(set(merged) | set(stored))
    final_cols: dict = {}
    for key in keys:
        new = G.concat(merged.get(key, []))
        new["cday"] = np.full(len(new["ekey"]), day, np.int64)
        tf = key.split("@")[1]
        if tf in TF_POSITIONAL:
            old = stored.get(key) or load_cands(state_dir, key)
            # replace every stored candidate of a symbol from that symbol's recompute date on
            drop = np.zeros(len(old["ekey"]), bool)
            for i in range(len(old["ekey"])):
                s = syms[int(old["sym"][i])]
                lo = tfg_lo_by_sym[s].get(key, day)
                if int(old["ekey"][i]) // 10000 >= lo:
                    drop[i] = True
            kept = {k: np.asarray(old[k])[~drop] for k in _CAND_KEYS}
            allc = {k: np.concatenate([kept[k], np.asarray(new[k])]) for k in _CAND_KEYS}
            save_cands(new_state, key, allc)
            final_cols[key] = allc
        else:
            final_cols[key] = new
    for cid, rid, tf, uni, agent in sel_combos:
        key = f"{rid}@{tf}"
        if key in errs_by_key:
            out["strategies"][cid] = {"closed": [], "open": [], "error": errs_by_key[key]}
            continue
        cols = final_cols.get(key)
        if cols is None:
            cols = G.concat([])
        is_idx = np.isin(np.asarray(cols["sym"]), [code[s] for s in G.INDEX_SYMS])
        part = {k: np.asarray(cols[k])[is_idx if uni == "index" else ~is_idx] for k in
                ("sym", "side", "ekey", "fill", "stop", "xkey", "xpx", "reason", "prio")}
        res = G.run_account(part, syms, uni, agent, lots, cid, agcfg=agents[agent])
        closed, opened = [], []
        for t in res["trades"]:
            tt = compact(t)
            if t[13] == "period_end" and int(t[6]) == day and tf in TF_POSITIONAL:
                opened.append(tt)
            elif since < int(t[6]) <= day:
                closed.append(tt)
        out["strategies"][cid] = {"closed": closed, "open": opened, "error": None,
                                  "candidates": int(len(part["ekey"]))}
    out["timing"]["accounts"] = round(time.time() - t2, 1)
    # live rules
    if want_live:
        t3 = time.time()
        try:
            out["strategies"].update(_live_rules(store, results, day, since, fwd_start))
        except Exception as exc:
            errors.append(f"live rules: {type(exc).__name__}: {str(exc)[:300]}")
            for sid in ("AUC-FADE-2PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED"):
                out["strategies"].setdefault(sid, {"closed": [], "open": [], "error": str(exc)[:200]})
        out["timing"]["live"] = round(time.time() - t3, 1)
    out["timing"]["total"] = round(time.time() - t0, 1)
    try:
        import resource
        out["peak_rss_mb"] = {"engine": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                              "workers": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}
    except Exception:
        pass
    with gzip.open(work / "result.json.gz", "wt") as fh:
        json.dump(out, fh)
    return out


def _live_rules(store, results, day: int, since: int, fwd_start: int) -> dict:
    from stratlab import auction as A, attention as T, options_vrp as V
    import options_replay as OR
    out = {}
    # AUC-FADE-2PCT: entry at the session's 09:15 auction, exit at 09:20
    rows = [tuple(r) for x in results for r in x["live"].get("auction", [])]
    sp = [s for s in A.SPECS if s.id == "AUC-FADE-2PCT"]
    tr = A.make_simulator(rows)(sp, day, day)["AUC-FADE-2PCT"]["trades"]
    out["AUC-FADE-2PCT"] = {"closed": [compact(t) for t in tr if since < int(t[6]) <= day], "open": [], "error": None}
    # SWING-ATTN-ON-3X: entries of the previous session exit at this session's open; this session's picks are open
    rows = [r for x in results for r in x["live"].get("attention", [])]
    counts: dict = {}
    for x in results:
        for d, n in x["live"].get("counts", {}).items():
            counts.setdefault(int(d), []).append(int(n))
    elig, special = T.filter_rows(rows, counts)
    sp = [s for s in T.SPECS if s.id == "SWING-ATTN-ON-3X"]
    prev = [d for d in sorted(counts) if d < day]
    lo = prev[-1] if prev else day
    tr = T.make_simulator(elig)(sp, lo, day)["SWING-ATTN-ON-3X"]["trades"]
    today = sorted((r for r in rows if r["date"] == day and day not in special),
                   key=lambda r: (-r["ratio"], r["symbol"]))[: sp[0].params["max_per_day"]]
    out["SWING-ATTN-ON-3X"] = {"closed": [compact(t) for t in tr if since < int(t[6]) <= day],
                               "open": [[r["symbol"], day, 1, "15:21", r["entry_px"], 0, "", 0.0, 0, 0.0, 0.0, 0.0,
                                         "pending"] for r in today], "error": None}
    # VRP-STRANGLE-016-WINGED: the lab's option replay on the stored NIFTY bars, VIX closes <= day
    z = store.raw("_NIFTY50_min", upto=day)
    n_by_day: dict = {}
    for d in z["date"]:
        n_by_day[int(d)] = n_by_day.get(int(d), 0) + 1
    from datetime import datetime as _dt
    bars = []
    for d, t, o, h, l, c in zip(z["date"], z["t"], z["o"], z["h"], z["l"], z["c"]):
        if n_by_day[int(d)] < 200:
            continue
        bars.append((_dt(int(d) // 10000, int(d) // 100 % 100, int(d) % 100, int(t) // 60, int(t) % 60),
                     float(o), float(h), float(l), float(c)))
    vx = store.load_aux("_VIX_day", upto=day)
    from datetime import date as _date
    vix = {_date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100): float(c) for d, c in zip(vx["date"], vx["c"])} \
        if vx is not None else {}
    vs = [s for s in V.SPECS if s.id == "VRP-STRANGLE-016-WINGED"]
    r = V.make_simulator(bars, vix, audit=False)(vs, fwd_start, day)["VRP-STRANGLE-016-WINGED"]
    closed = [compact(t) for t in r["trades"] if t[6] and since < int(t[6]) <= day]
    pending = [compact(t) for t in r["trades"] if not t[6]]
    for p in pending:
        p[12] = "pending"
    out["VRP-STRANGLE-016-WINGED"] = {"closed": closed, "open": pending, "error": r["error"]}
    return out


# ── baseline (lab references; frozen into forward_lab_baseline.json) ─────────────────────────────────────────────
BASELINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_lab_baseline.json")


def load_baseline(path: str = BASELINE_FILE) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--since", type=int, required=True)
    ap.add_argument("--fwd-start", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--ids", default="")
    ap.add_argument("--mem-mb", type=int, default=0)
    a = ap.parse_args(argv)
    if a.mem_mb:
        try:
            import resource
            lim = int(a.mem_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:
            pass
    try:
        os.nice(10)
    except Exception:
        pass
    try:
        run_session(a.store, a.state, a.day, a.since, a.fwd_start, a.work, jobs=a.jobs,
                    ids=[x for x in a.ids.split(",") if x] or None)
    except Exception as exc:
        Path(a.work).mkdir(parents=True, exist_ok=True)
        with open(Path(a.work) / "error.txt", "w") as fh:
            fh.write(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
