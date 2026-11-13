"""
stratlab/protocol_oos.py — the OLD-ERA OUT-OF-SAMPLE judgement (2012-01-01 .. 2023-11-09). protocol.py is NOT
changed and not used for the verdict; its gate constants are imported so the thresholds cannot drift.

The decade before the lab's 3-year data was never used by any strategy. It is ONE out-of-sample block: there is no
DISCOVERY / FINAL split here, and every eligible strategy gets exactly ONE judgement on it.

GATES (frozen; GATES_FINGERPRINT is recorded with every judgement and in the pre-registration file)
  halves                H1 = 2012-01-01 .. 2017-12-31, H2 = 2018-01-01 .. 2023-11-09 (a trade belongs to the half
                        of its ENTRY date; a trade entered in H1 may exit in H2 — the era is one block)
  PASS (OOS-CONFIRMED)  net after costs > 0 in H1 AND in H2
                        profit factor >= protocol.SCREEN_MIN_PF (1.15) over the whole era
                        trades >= protocol.SCREEN_MIN_TRADES[horizon] in EACH half (positional: 60)
                        top symbol <= protocol.SCREEN_MAX_SYMBOL_SHARE (35 %) of the era's net (skipped for
                        one-instrument strategies: index futures / single-underlying options, as in protocol.py)
  REPORTED, NOT A GATE  the same strategy at 5 bps slippage per side (index futures 2.5 bps; options: haircut x2.5)
                        and whether it would pass there
  Costs and fills are the lab's own: sim.sim_positional with costs.round_trip_cost (sim.COST_MODEL) for @strategy
  specs, nsedata's OptionSpec simulator (nsedata.cost_model()) for real-price option specs. Default slippage 2 bps
  (stocks) / 1 bps (index futures).

ONE LOOK: every judgement is written to the OOS ledger (<scratchpad>/stratlab_cache/oldera_oos_ledger.json) keyed by
strategy id + cost model. A second run of the same, unchanged strategy returns the FIRST result (marked "first
judged <when>"); a changed strategy (fingerprint differs) or changed gates is refused ("NOT JUDGED: ...").
With a pre-registration file (--prereg; written by --preregister BEFORE any old-era result exists) a strategy is
judged only if it is listed there with the same fingerprint and the gates are the registered ones.

ELIGIBILITY (eligibility()): daily-only. Strategies that need minute bars — every intraday / scalping /
futures-intraday spec, every positional spec with decision_time=, and the module-level runners auction,
auction_r2, attention, ltr, r3_crosssec, options_vrp, options_vrp_vix, options_r3 — are ineligible, as are specs
whose hard-coded calendars stop before / start after the old era, and option specs that cannot trade in H1 by
construction (NIFTY weeklies start 2019-02).

    cd algotrader_v4
    python -m stratlab.protocol_oos --list                                    # eligible / ineligible with reasons
    python -m stratlab.protocol_oos --preregister <file>                      # freeze ids + fingerprints + gates
    python -m stratlab.protocol_oos --out <scratch dir> --prereg <file> [--ids a,b] [--jobs 4]   # the ONE judgement
    python -m stratlab.protocol_oos --smoke --ids a,b                         # errors only, no P&L, no counts
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from . import data_daily as DD
from . import oldera_fo as OF
from . import protocol as P

ERA = DD.ERA
H1 = (20120101, 20171231)
H2 = (20180101, 20231109)
MIN_PF = P.SCREEN_MIN_PF
MIN_TRADES_PER_HALF = dict(P.SCREEN_MIN_TRADES)
MAX_SYMBOL_SHARE = P.SCREEN_MAX_SYMBOL_SHARE
SENS_BPS = 5.0
GATES = {"era": ERA, "h1": H1, "h2": H2, "min_pf": MIN_PF, "min_trades_per_half": MIN_TRADES_PER_HALF,
         "max_symbol_share": MAX_SYMBOL_SHARE, "net_gt_0": ["h1", "h2"], "sens_bps": SENS_BPS,
         "universe": "universe_monthly.csv: top 100 by 60-session turnover before each month, gate on decision day"}
GATES_FINGERPRINT = hashlib.sha1(json.dumps(GATES, sort_keys=True).encode()).hexdigest()[:12]
LEDGER = os.path.join(DD._SCRATCH, "stratlab_cache", "oldera_oos_ledger.json")
_NET, _SYM, _DATE = 12, 1, 2

# ══════════════════════════════════════════════════════════════════════════
# eligibility
# ══════════════════════════════════════════════════════════════════════════
# positional specs whose hard-coded calendars (r4_mechanics.HOLIDAYS / MUHURAT, 2023-11 .. 2026-11) do not cover
# the old era: the strategy would silently use a wrong holiday calendar / never fire.
CALENDAR_BOUND = {
    "r4_mechanics_nifty_pre_expiry": "counts regular sessions with r4_mechanics.HOLIDAYS, which lists 2023-11..2026-11 "
                                     "holidays only (wrong session counts in 2012-2023)",
    "r4_mechanics_sip_nifty_d1to10": "month's last regular session from r4_mechanics.HOLIDAYS (2023-11..2026-11 only)",
    "r4_mechanics_diwali_window": "r4_mechanics.MUHURAT lists 2023-2026 Muhurat dates only: never fires in the old era",
}
MODULE_RUNNERS = {
    "stratlab.auction": "09:15 call-auction print and 09:20 open (minute bars)",
    "stratlab.auction_r2": "09:15 auction print and 09:25 open (minute bars)",
    "stratlab.attention": "volume 09:15-15:20 and 15:21 open (minute bars)",
    "stratlab.ltr": "executes at the 15:14 one-minute bar",
    "stratlab.r3_crosssec": "decision-time features at 09:30 / 10:15 / 15:20 and fills at 09:30 / 15:21 (minute bars)",
    "stratlab.options_vrp": "Black-Scholes on NIFTY minute bars at 10:20, settlement on 15:00-15:29 minutes",
    "stratlab.options_vrp_vix": "the VRP strangle (minute bars at 10:20) with a VIX filter",
    "stratlab.options_r3": "Black-Scholes on NIFTY / BANKNIFTY minute bars (intraday entries / exits)",
}
OPT_INELIGIBLE = {
    "nf_wk_": "NIFTY WEEKLY cycle: weeklies exist only from 2019-02, so H1 (2012-2017) has no trade by construction",
    "nf_event_": "options_r4.EVENT_DAYS lists 2023-2026 Budget / RBI dates only: never fires in the old era",
}


def eligibility():
    """(eligible, ineligible): eligible = [(id, module, kind)], ineligible = [(id, module, reason)]."""
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    from . import options_r4
    el, inel = [], []
    for sp in sorted(REGISTRY.values(), key=lambda s: (s.module, s.id)):
        if sp.horizon != "positional":
            inel.append((sp.id, sp.module, f"{sp.kind} (intraday horizon): needs 1-minute bars"))
        elif sp.decision_time is not None:
            inel.append((sp.id, sp.module, f"decision_time={sp.decision_time // 60:02d}:{sp.decision_time % 60:02d}: "
                                           "decides on the day's minute bars"))
        elif sp.id in CALENDAR_BOUND:
            inel.append((sp.id, sp.module, CALENDAR_BOUND[sp.id]))
        else:
            el.append((sp.id, sp.module, sp.kind + (f"/{sp.underlying}" if sp.underlying else "")))
    for sp in options_r4.SPECS:
        short = sp.id[len("r4_realopt_"):]
        why = next((w for k, w in OPT_INELIGIBLE.items() if short.startswith(k)), None)
        if why:
            inel.append((sp.id, "stratlab.options_r4", why))
        else:
            el.append((sp.id, "stratlab.options_r4", "options/" + ",".join(sp.underlyings)))
    for mod, why in MODULE_RUNNERS.items():
        inel.append(("(all ids)", mod, why))
    return el, inel


# ══════════════════════════════════════════════════════════════════════════
# judgement
# ══════════════════════════════════════════════════════════════════════════
def _in(trades, rng):
    return [t for t in trades if rng[0] <= t[_DATE] <= rng[1]]


def judge(trades, horizon: str, single_underlying: bool = False) -> dict:
    """The OOS gates on one strategy's era trades."""
    era = _in(trades, ERA)
    h1, h2, a = P.stats(_in(era, H1)), P.stats(_in(era, H2)), P.stats(era)
    by: dict = {}
    for t in era:
        by[t[_SYM]] = by.get(t[_SYM], 0.0) + t[_NET]
    top, top_net = max(by.items(), key=lambda kv: kv[1]) if by else ("", 0.0)
    share = top_net / a["net"] if a["net"] > 0 else float("nan")
    need = MIN_TRADES_PER_HALF[horizon]
    reasons = []
    if h1["net"] <= 0:
        reasons.append("H1 net <= 0")
    if h2["net"] <= 0:
        reasons.append("H2 net <= 0")
    if a["pf"] < MIN_PF:
        reasons.append(f"PF {a['pf']:.3f} < {MIN_PF}")
    if h1["trades"] < need:
        reasons.append(f"H1 {h1['trades']} trades < {need}")
    if h2["trades"] < need:
        reasons.append(f"H2 {h2['trades']} trades < {need}")
    if a["net"] > 0 and share > MAX_SYMBOL_SHARE and not single_underlying:
        reasons.append(f"{top} is {share:.0%} of net (> {MAX_SYMBOL_SHARE:.0%})")
    return {"passed": not reasons, "reasons": reasons, "h1": h1, "h2": h2, "era": a, "top_symbol": top,
            "top_share": share}


def _load_json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}


def evaluate(specs, simulate, sens_simulate=None, cost_model: str = "", ledger_path: str = LEDGER,
             prereg: dict | None = None, dataset: str = "") -> list:
    """simulate(specs, lo, hi) -> {id: {"trades", "skipped", "lookahead", "error"}} (the lab's contract).
    Simulates the WHOLE era once per spec (and once more at the sensitivity slippage), judges, writes the ledger."""
    out, todo = [], []
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        ledger = _load_json(ledger_path)
    for sp in specs:
        key = f"{sp.id}|oldera|{cost_model}"
        old = ledger.get(key)
        why = None
        if prereg is not None:
            reg = prereg.get("strategies", {}).get(sp.id)
            if prereg.get("gates_fingerprint") != GATES_FINGERPRINT:
                why = "gates differ from the pre-registration"
            elif reg is None:
                why = "not in the pre-registration"
            elif reg["fingerprint"] != sp.fingerprint:
                why = "code / parameters changed since the pre-registration"
        if why is None and old is not None:
            if old["fingerprint"] != sp.fingerprint or old.get("gates") != GATES_FINGERPRINT:
                why = f"already judged on {old['when']} with different code or gates: no second look"
            else:
                r = dict(old["result"])
                r.update({"id": sp.id, "spec": sp, "note": f"first judged {old['when']} (ledger)", "from_ledger": True})
                out.append(r)
                continue
        if why:
            out.append({"id": sp.id, "spec": sp, "verdict": "NOT JUDGED", "why": why})
            continue
        todo.append(sp)
    if not todo:
        return out
    res = simulate(todo, ERA[0], ERA[1])
    sens = sens_simulate(todo, ERA[0], ERA[1]) if sens_simulate else {}
    now = time.strftime("%Y-%m-%d %H:%M")
    new = {}
    for sp in todo:
        r = res.get(sp.id, {"trades": [], "error": "no result"})
        single = bool(getattr(sp, "underlying", None))
        j = judge(r["trades"], sp.horizon, single)
        s = sens.get(sp.id)
        js = judge(s["trades"], sp.horizon, single) if s else None
        if r.get("error"):
            verdict, why = "INVALID", "strategy raised: " + str(r["error"])
        elif r.get("lookahead"):
            verdict, why = "INVALID", "LOOK-AHEAD: " + str(r["lookahead"])
        else:
            verdict = "OOS-CONFIRMED" if j["passed"] else "OOS-FAILED"
            why = "all gates met" if j["passed"] else ", ".join(j["reasons"])
        result = {"verdict": verdict, "why": why, "h1": j["h1"], "h2": j["h2"], "era": j["era"],
                  "top_symbol": j["top_symbol"], "top_share": None if j["top_share"] != j["top_share"] else
                  round(j["top_share"], 4), "skipped": r.get("skipped", 0),
                  "sens": None if js is None else {"era": js["era"], "h1": js["h1"], "h2": js["h2"],
                                                   "passed": js["passed"]}}
        new[f"{sp.id}|oldera|{cost_model}"] = {"fingerprint": sp.fingerprint, "gates": GATES_FINGERPRINT, "when": now,
                                               "cost_model": cost_model, "dataset": dataset, "result": result}
        out.append({**result, "id": sp.id, "spec": sp, "note": "", "trades": r["trades"]})
    with open(ledger_path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        ledger = _load_json(ledger_path)
        for k, v in new.items():
            ledger.setdefault(k, v)                     # first write wins, even under a race
        tmp = ledger_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(ledger, fh, indent=1)
        os.replace(tmp, ledger_path)
    return out


# ══════════════════════════════════════════════════════════════════════════
# simulators
# ══════════════════════════════════════════════════════════════════════════
_CTX: dict = {}


def _work(args):
    symbol, ids, lo, hi, slip, islip = args
    from . import sim
    from .api import REGISTRY
    cache = _CTX["cache"]
    out = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None} for i in ids}
    specs = [REGISTRY[i] for i in ids if sim.applies(REGISTRY[i], symbol)]
    if not specs:
        return out
    sd = DD.load_symbol(cache, symbol)
    if sd is None:
        return out
    nifty, vix = _CTX["nifty"], _CTX["vix"]
    for sp in specs:
        s = sim.spec_slippage(sp, slip, islip)
        try:
            tr, sk = sim.sim_positional(DD.gate(sp, sd), sd, lo, hi, nifty, vix, s)
            out[sp.id]["trades"] = tr
            out[sp.id]["skipped"] = sk
        except Exception as exc:
            out[sp.id]["error"] = f"{symbol}: {type(exc).__name__}: {str(exc)[:160]}"
    return out


def make_simulator(pool, symbols, slip_bps: float, islip_bps: float):
    def simulate(specs, lo, hi):
        ids = [sp.id for sp in specs]
        merged = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None} for i in ids}
        for res in pool.imap_unordered(_work, [(s, ids, lo, hi, slip_bps / 1e4, islip_bps / 1e4) for s in symbols]):
            for i, r in res.items():
                m = merged[i]
                m["trades"].extend(r["trades"])
                m["skipped"] += r["skipped"]
                m["error"] = m["error"] or r["error"]
        for m in merged.values():
            m["trades"].sort(key=lambda t: (t[2], t[1]))
        return merged
    return simulate


def setup(cache_dir: str | None = None, view_dir: str | None = OF.VIEW_DIR):
    """Daily cache + nsedata switched to the old-era view (if it exists). Must run before forking workers."""
    cache = DD.ensure_old_cache(cache_dir=cache_dir or DD.OLD_CACHE)
    _CTX["cache"] = cache
    _CTX["nifty"] = DD.load_daily_aux(cache, "_NIFTY50_day")
    _CTX["vix"] = DD.load_daily_aux(cache, "_VIX_day")
    if view_dir and os.path.exists(os.path.join(view_dir, "delivery.csv.gz")):
        N = OF.use(view_dir)
        N._load()                                     # build / load nsedata tables once, before fork
    return cache


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def _f(x):
    return "inf" if x == float("inf") else f"{x:,.0f}"


def _write(out, results, meta):
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    from . import sim
    L = [f"OLD-ERA OUT-OF-SAMPLE  {ERA[0]}..{ERA[1]}  H1 {H1[0]}..{H1[1]}  H2 {H2[0]}..{H2[1]}  "
         f"gates {GATES_FINGERPRINT}", f"{meta}", "",
         f"{'id':<44}{'tr H1':>6}{'tr H2':>6}{'PF':>6}{'net H1':>11}{'net H2':>11}{'net era':>11}"
         f"{'net@5':>11}{'scr5':>5}  verdict           why"]
    rows = []
    for r in results:
        if r.get("verdict") == "NOT JUDGED":
            L.append(f"{r['id']:<44}{'':>67}  NOT JUDGED        {r['why']}")
            rows.append({"id": r["id"], "verdict": "NOT JUDGED", "why": r["why"]})
            continue
        s = r.get("sens") or {}
        L.append(f"{r['id']:<44}{r['h1']['trades']:>6}{r['h2']['trades']:>6}"
                 f"{('inf' if r['era']['pf'] == float('inf') else format(r['era']['pf'], '.2f')):>6}"
                 f"{_f(r['h1']['net']):>11}{_f(r['h2']['net']):>11}{_f(r['era']['net']):>11}"
                 f"{_f(s['era']['net']) if s else '':>11}{('Y' if s.get('passed') else 'N') if s else '':>5}  "
                 f"{r['verdict']:<18}{r['why']}" + (f" | {r['note']}" if r.get("note") else "")
                 + (f" | top {r['top_symbol']} {r['top_share']:.0%}" if r.get("top_share") is not None else ""))
        rows.append({"id": r["id"], "verdict": r["verdict"], "why": r["why"], "h1_trades": r["h1"]["trades"],
                     "h2_trades": r["h2"]["trades"], "h1_net": r["h1"]["net"], "h2_net": r["h2"]["net"],
                     "era_trades": r["era"]["trades"], "era_pf": r["era"]["pf"], "era_net": r["era"]["net"],
                     "era_win_pct": r["era"]["win_pct"], "top_symbol": r["top_symbol"], "top_share": r["top_share"],
                     "sens_era_net": s.get("era", {}).get("net", "") if s else "",
                     "sens_era_pf": s.get("era", {}).get("pf", "") if s else "",
                     "sens_passed": s.get("passed", "") if s else "", "skipped": r.get("skipped", 0),
                     "note": r.get("note", "")})
        if r.get("trades") is not None and not r.get("from_ledger"):
            with open(os.path.join(out, "trades", r["id"] + ".csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(sim.TRADE_FIELDS)
                w.writerows(r["trades"])
    n = len(results)
    judged = [r for r in results if r.get("verdict") in ("OOS-CONFIRMED", "OOS-FAILED")]
    conf = [r for r in judged if r["verdict"] == "OOS-CONFIRMED"]
    L += ["", f"{n} strategies: {len(judged)} judged, {len(conf)} OOS-CONFIRMED. About {P.GATE_LUCK:.0%} of useless "
          f"strategies pass each coin-flip-like gate: with {len(judged)} judged, a few confirmations are expected by "
          "luck alone. A confirmation earns a paper-trading trial, nothing more."]
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return "\n".join(L)


def _select(ids):
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    from . import options_r4
    el, _ = eligibility()
    ok = {i for i, _m, _k in el}
    opts = {s.id: s for s in options_r4.SPECS}
    want = [x.strip() for x in ids.split(",") if x.strip()] if ids else sorted(ok)
    bad = [w for w in want if w not in ok]
    if bad:
        raise SystemExit(f"not eligible / unknown: {bad}")
    return [REGISTRY[w] for w in want if w in REGISTRY], [opts[w] for w in want if w in opts]


def preregister(path: str):
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    from . import options_r4
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: a pre-registration is written once")
    el, inel = eligibility()
    opts = {s.id: s for s in options_r4.SPECS}
    st = {}
    for i, m, k in el:
        sp = REGISTRY.get(i) or opts[i]
        st[i] = {"module": m, "kind": k, "fingerprint": sp.fingerprint, "params": dict(sp.params)}
    from . import sim
    from . import nsedata as N
    doc = {"written": time.strftime("%Y-%m-%d %H:%M"), "gates": GATES, "gates_fingerprint": GATES_FINGERPRINT,
           "cost_models": {"positional": sim.COST_MODEL, "options": N.cost_model()},
           "slippage_bps": {"stocks": 2.0, "index": 1.0, "sens_stocks": SENS_BPS, "sens_index": 2.5,
                            "options_sens_haircut_mult": N.SENS_MULT},
           "strategies": st, "ineligible": [{"id": i, "module": m, "reason": w} for i, m, w in inel],
           "note": "written before any old-era strategy result was computed"}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, default=str)
    print(f"pre-registered {len(st)} strategies (gates {GATES_FINGERPRINT}) -> {path}")


def run_options(specs, prereg, out_unused=None):
    """Real-price option specs (options_r4) through nsedata's simulator on the old-era view."""
    from . import nsedata as N
    cal = DD.calendar(_CTX["cache"])
    sim_ = N.make_simulator(cal)
    sens_ = N.make_simulator(cal, mult=N.SENS_MULT, audit=False)
    return evaluate(specs, sim_, sens_, cost_model=N.cost_model(), prereg=prereg, dataset=OF.VIEW_DIR)


def _opt_cm() -> str:
    from . import nsedata as N
    return N.cost_model()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.protocol_oos", description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preregister", default="")
    ap.add_argument("--prereg", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--smoke", action="store_true", help="run on 2012-02..2012-06 and report only errors")
    ap.add_argument("--smoke-symbols", type=int, default=15)
    a = ap.parse_args(argv)
    if a.list:
        el, inel = eligibility()
        print(f"ELIGIBLE {len(el)}")
        for i, m, k in el:
            print(f"  {i:<46}{m:<36}{k}")
        print(f"INELIGIBLE {len(inel)}")
        for i, m, w in inel:
            print(f"  {i:<46}{m:<36}{w}")
        return 0
    if a.preregister:
        preregister(a.preregister)
        return 0
    from . import sim
    pos, opts = _select(a.ids)
    setup()
    syms = DD.symbols(_CTX["cache"]) + list(DD.INDEX_NAMES.values())
    if a.smoke:
        # errors only: no P&L, no trade counts (the old era stays unseen); a short slice and a few symbols
        lo, hi = 20120201, 20120630
        stock = [s for s in DD.symbols(_CTX["cache"]) if s not in DD.INDEX_NAMES.values()][:a.smoke_symbols]
        with mp.get_context("fork").Pool(a.jobs) as pool:
            res = make_simulator(pool, stock + list(DD.INDEX_NAMES.values()), 2.0, 1.0)(pos, lo, hi) if pos else {}
        if opts:
            from . import nsedata as N
            cal = DD.calendar(_CTX["cache"])
            res.update(N.make_simulator(cal)(opts, lo, hi))
        bad = {i: r["error"] or r.get("lookahead") for i, r in res.items() if r["error"] or r.get("lookahead")}
        print(f"smoke {lo}..{hi}: {len(res)} specs ran, {len(bad)} raised / look-ahead")
        for i, e in bad.items():
            print(f"  {i}: {e}")
        return 1 if bad else 0
    if not a.out:
        ap.error("--out is required")
    from . import data as D
    out = D.assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    prereg = _load_json(a.prereg) if a.prereg else None
    if a.prereg and not prereg:
        raise SystemExit(f"cannot read {a.prereg}")
    results = []
    if pos:
        with mp.get_context("fork").Pool(a.jobs) as pool:
            simu = make_simulator(pool, syms, sim.DEFAULT_SLIPPAGE_BPS, sim.INDEX_SLIPPAGE_BPS)
            sens = make_simulator(pool, syms, SENS_BPS, sim.INDEX_SLIPPAGE_BPS * SENS_BPS / sim.DEFAULT_SLIPPAGE_BPS)
            results += evaluate(pos, simu, sens, cost_model=sim.COST_MODEL, prereg=prereg, dataset=DD.OLD_DIR)
    if opts:
        results += run_options(opts, prereg)
    meta = (f"positional: sim.sim_positional, cost model {sim.COST_MODEL}, slippage 2 bps stocks / 1 bps index "
            f"(sens {SENS_BPS:g} / 2.5); options: nsedata EOD fills, cost model {_opt_cm()} (sens haircut x2.5); "
            f"universe gate on decision day; "
            f"prereg {a.prereg or 'NONE'}")
    print(_write(out, results, meta))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
