"""
stratlab/tfgrid_run.py — the run stages of the timeframe grid (see tfgrid.py for the plan). Research only.

Every stage refuses to run unless the pre-registration's sha256 matches and no rule changed after it. Outputs go to
<scratch>/tfgrid (compact: per-combo summaries; trade files only for stage-1 passers, stage-2 runs and the live
strategies).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np

from . import capital as K
from . import data as D
from . import protocol as P
from . import tfgrid as G
from .tfgrid import log

S1 = os.path.join(G.ROOT, "stage1")
S2 = os.path.join(G.ROOT, "stage2")
FIN = os.path.join(G.ROOT, "final")
CTRL = os.path.join(G.ROOT, "controls")
LIVE = os.path.join(G.ROOT, "live")
NATIVE_CHUNK = 120
STAGE2_LEDGER = os.path.join(G.ROOT, "stage2_ledger.json")


def _jload(p, default=None):
    try:
        if p.endswith(".gz"):
            with gzip.open(p, "rt") as fh:
                return json.load(fh)
        with open(p) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def _jsave(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    if p.endswith(".gz"):
        with gzip.open(tmp, "wt") as fh:
            json.dump(obj, fh, default=G._js)
    else:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, default=G._js, indent=1)
    os.replace(tmp, p)


# ══════════════════════════════════════════════════════════════════════════
# stage 1
# ══════════════════════════════════════════════════════════════════════════
def stage1(jobs: int, only_tfs=None, natives: bool = True):
    pre = G.load_prereg()
    cal = G.calendar3()
    lo, hi = G.DISC
    log(f"STAGE 1 start: {len(pre['grid_combos'])} grid combos + {len(pre['native_combos'])} natives, DISCOVERY {lo}-{hi}")
    grid = [(g["id"], g["rule"], g["tf"], g["universe"], g["agent"]) for g in pre["grid_combos"]]
    if only_tfs:
        grid = [c for c in grid if c[2] in only_tfs]
    for tf in G.TFS:
        sel = [c for c in grid if c[2] == tf]
        if not sel or os.path.exists(os.path.join(S1, f"results_{tf}.json.gz")):
            continue
        G.run_grid("stage1", lo, hi, G.CACHE3Y, sel, jobs, S1, cal)
    if natives:
        stage1_natives(pre, jobs, cal)
    log("STAGE 1 done")


def _native_judge(sid, sp, res, cal, lo, hi):
    met = K.metrics(res, cal, lo, hi)
    ag, uni = G.native_agent(sp)
    scr = P.screen(res["trades"], G.split3(), sp.horizon, single_underlying=(uni == "index"))
    row = G.summary_row(sid, {"rule": sp.id, "tf": "native", "universe": uni, "agent": ag, "kind": sp.kind,
                              "horizon": sp.horizon}, res, met, scr)
    row["daily"] = [round(x, 2) for x in K.daily_series(res, cal, lo, hi)]
    return row


def stage1_natives(pre, jobs, cal):
    lo, hi = G.DISC
    specs = G.native_specs()
    ids = [n["catalogue_id"] for n in pre["native_combos"]]
    stocks = G.stock_symbols(G.CACHE3Y)
    lots = G.lot_sizes(stocks)
    for k in range(0, len(ids), NATIVE_CHUNK):
        part = os.path.join(S1, f"natives_{k // NATIVE_CHUNK:02d}.json.gz")
        if os.path.exists(part):
            continue
        chunk = ids[k:k + NATIVE_CHUNK]
        t0 = time.time()
        built, syms = G.build_native(G.CACHE3Y, chunk, lo, hi, jobs=jobs)
        rows = {}
        for i in chunk:
            sp = specs[i]
            sid = f"nat:{i}:rr3"
            b = built[i]
            if b["errors"]:
                rows[sid] = {"id": sid, "rule": i, "tf": "native", "invalid": b["errors"][:3],
                             "agent": G.native_agent(sp)[0], "universe": G.native_agent(sp)[1]}
                continue
            ag, uni = G.native_agent(sp)
            res = G.run_account(b["cols"], syms, uni, ag, lots, sid, fb_lots=int(getattr(sp, "lots", 1) or 1))
            row = _native_judge(sid, sp, res, cal, lo, hi)
            comp = G.run_account(b["cols"], syms, uni, ag, lots, sid, compound=True,
                                 fb_lots=int(getattr(sp, "lots", 1) or 1))
            row["compounded"] = K.metrics(comp, cal, lo, hi)
            rows[sid] = row
            if row["screen"]["passed"]:
                os.makedirs(os.path.join(S1, "trades"), exist_ok=True)
                G.save_trades(os.path.join(S1, "trades", G._fn(sid) + ".pkl.gz"), res["trades"])
        _jsave(part, rows)
        log(f"  stage1 natives {k}-{k + len(chunk)}: {len(chunk)} ids in {time.time() - t0:.0f}s, "
            f"{sum(1 for r in rows.values() if r.get('screen', {}).get('passed'))} pass")


def load_stage1() -> dict:
    out = {}
    if not os.path.isdir(S1):
        return out
    for f in sorted(os.listdir(S1)):
        if f.endswith(".json.gz") and (f.startswith("results_") or f.startswith("natives_")):
            out.update(_jload(os.path.join(S1, f), {}))
    return out


def passers(rows: dict) -> list:
    return sorted(k for k, r in rows.items() if r.get("screen", {}).get("passed"))


# ══════════════════════════════════════════════════════════════════════════
# FINAL: one look per stage-1 passer
# ══════════════════════════════════════════════════════════════════════════
def final(jobs: int):
    pre = G.load_prereg()
    rows = load_stage1()
    ps = passers(rows)
    led = _jload(G.FINAL_LEDGER, {}) or {}
    todo = [p for p in ps if p not in led]
    log(f"FINAL: {len(ps)} stage-1 passers, {len(todo)} not yet looked at")
    if not todo:
        return led
    cal = G.calendar3()
    lo, hi = G.SPLIT3["final"]
    stocks = G.stock_symbols(G.CACHE3Y)
    lots = G.lot_sizes(stocks)
    gmap = {g["id"]: g for g in pre["grid_combos"]}
    grid = [p for p in todo if p in gmap]
    nat = [p for p in todo if p.startswith("nat:")]
    out = {}
    for tf in G.TFS:
        sel = [gmap[p] for p in grid if gmap[p]["tf"] == tf]
        if not sel:
            continue
        rids = sorted({g["rule"] for g in sel})
        cand, syms, _ = G.build_candidates(G.CACHE3Y, tf, lo, hi, rids, jobs=jobs)
        for g in sel:
            cols = cand[g["rule"]]["index" if g["universe"] == "index" else "stocks"]
            res = G.run_account(cols, syms, g["universe"], g["agent"], lots, g["id"])
            out[g["id"]] = (res, K.metrics(res, cal, lo, hi))
    if nat:
        specs = G.native_specs()
        ids = [p.split(":")[1] for p in nat]
        built, syms = G.build_native(G.CACHE3Y, ids, lo, hi, jobs=jobs)
        for p, i in zip(nat, ids):
            sp = specs[i]
            ag, uni = G.native_agent(sp)
            res = G.run_account(built[i]["cols"], syms, uni, ag, lots, p, fb_lots=int(getattr(sp, "lots", 1) or 1))
            out[p] = (res, K.metrics(res, cal, lo, hi))
    os.makedirs(os.path.join(FIN, "trades"), exist_ok=True)
    for p, (res, met) in out.items():
        fv = P.final_verdict([t for t in res["trades"] if lo <= t[2] <= hi])
        led[p] = {"looked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "period": [lo, hi], "final": fv, "metrics": met,
                  "verdict": "CONFIRMED" if fv["confirmed"] else "REJECTED ON FINAL"}
        G.save_trades(os.path.join(FIN, "trades", G._fn(p) + ".pkl.gz"), res["trades"])
        _jsave(G.FINAL_LEDGER, led)
        log(f"  FINAL {p}: {led[p]['verdict']} ({fv['trades']} tr, PF {fv['pf']}, net {fv['net']:,.0f})")
    return led


# ══════════════════════════════════════════════════════════════════════════
# stage 2: 2016-01-01 .. 2023-11-09, once per stage-1 passer
# ══════════════════════════════════════════════════════════════════════════
def stage2(jobs: int):
    from . import protocol_min_oos as PMO
    pre = G.load_prereg()
    rows = load_stage1()
    ps = passers(rows)
    led = _jload(STAGE2_LEDGER, {}) or {}
    todo = [p for p in ps if p not in led]
    log(f"STAGE 2: {len(ps)} stage-1 passers, {len(todo)} to judge on 2016-23")
    if not todo:
        return led
    ctx = PMO.setup(G.CACHE_OLD)
    cal = [int(x) for x in ctx["cal"]]
    lo, hi = G.OLD_ERA
    stocks = G.stock_symbols(G.CACHE_OLD)
    lots = G.lot_sizes(stocks)
    gmap = {g["id"]: g for g in pre["grid_combos"]}
    res_all = {}
    for tf in G.TFS:
        sel = [gmap[p] for p in todo if p in gmap and gmap[p]["tf"] == tf]
        if not sel:
            continue
        rids = sorted({g["rule"] for g in sel})
        cand, syms, _ = G.build_candidates(G.CACHE_OLD, tf, lo, hi, rids, jobs=jobs)
        for g in sel:
            if cand[g["rule"]]["errors"]:
                res_all[g["id"]] = ({"invalid": cand[g["rule"]]["errors"][:3]}, g["tf"], g["universe"])
                continue
            cols = cand[g["rule"]]["index" if g["universe"] == "index" else "stocks"]
            res = G.run_account(cols, syms, g["universe"], g["agent"], lots, g["id"], dirty=ctx["dirty"],
                                nxt=ctx["nxt"])
            res_all[g["id"]] = (res, G.horizon_of(g["tf"]), g["universe"])
    nat = [p for p in todo if p.startswith("nat:")]
    if nat:
        specs = G.native_specs()
        ids = [p.split(":")[1] for p in nat]
        elig = [i for i in ids if i not in PMO.CALENDAR_BOUND]
        for i in ids:
            if i in PMO.CALENDAR_BOUND:
                res_all[f"nat:{i}:rr3"] = ({"ineligible": PMO.CALENDAR_BOUND[i]}, None, None)
        if elig:
            built, syms = G.build_native(G.CACHE_OLD, elig, lo, hi, jobs=jobs, years=True)
            for i in elig:
                sp = specs[i]
                ag, uni = G.native_agent(sp)
                p = f"nat:{i}:rr3"
                if built[i]["errors"]:
                    res_all[p] = ({"invalid": built[i]["errors"][:3]}, None, None)
                    continue
                res = G.run_account(built[i]["cols"], syms, uni, ag, lots, p, dirty=ctx["dirty"], nxt=ctx["nxt"],
                                    fb_lots=int(getattr(sp, "lots", 1) or 1))
                res_all[p] = (res, sp.horizon, uni)
    os.makedirs(os.path.join(S2, "trades"), exist_ok=True)
    for p, (res, hz, uni) in res_all.items():
        if "trades" not in res:
            led[p] = {"judged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "verdict": "INVALID" if "invalid" in res
                      else "INELIGIBLE", "detail": res}
            continue
        agg = PMO.agg_from_trades(res["trades"], np.asarray(cal), ctx["dirty"], ctx["nxt"], ctx["refs"], PMO.SYM_MAP,
                                  apply_dirty=False)
        j = PMO.judge(agg, hz, single_underlying=(uni == "index"))
        met = K.metrics(res, cal, lo, hi)
        led[p] = {"judged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "era": [lo, hi], "gates": PMO.GATES_FINGERPRINT,
                  "judge": {k: v for k, v in j.items()}, "metrics": met, "excluded_dirty": res.get("excluded", 0),
                  "excess": {"h1": agg["ex"]["h1"], "h2": agg["ex"]["h2"]},
                  "long_share": round(agg["long"] / max(agg["n"], 1), 3),
                  "by_year": {str(k): round(v, 2) for k, v in sorted(agg["by_year"].items())},
                  "daily": [round(x, 2) for x in K.daily_series(res, cal, lo, hi)],
                  "verdict": "OOS-CONFIRMED" if j["passed"] else "OOS-FAILED"}
        G.save_trades(os.path.join(S2, "trades", G._fn(p) + ".pkl.gz"), res["trades"])
        _jsave(STAGE2_LEDGER, led)
        log(f"  STAGE 2 {p}: {led[p]['verdict']} {j['reasons']}")
    return led


# ══════════════════════════════════════════════════════════════════════════
# controls
# ══════════════════════════════════════════════════════════════════════════
def controls(jobs: int):
    cal = G.calendar3()
    lo, hi = G.DISC
    syms = G.stock_symbols(G.CACHE3Y)[:G.CONTROL_SYMBOLS]
    lots = G.lot_sizes(syms)
    out = _jload(os.path.join(CTRL, "controls.json"), {}) or {}
    rids = sorted(G.CONTROL_RULES)
    for tf in G.TFS:
        if tf in out:
            continue
        cand, table, _ = G.build_candidates(G.CACHE3Y, tf, lo, hi, rids, jobs=jobs, symbols=syms, with_index=False)
        out[tf] = {}
        for rid in rids:
            res = G.run_account(cand[rid]["stocks"], table, "stocks", G.STOCK_AGENT[tf], lots, rid)
            met = K.metrics(res, cal, lo, hi)
            scr = G.judge_stage1(res, tf, False)
            tr = res["trades"]
            out[tf][rid] = {"trades": met["trades"], "pf": met["pf"], "net": met["net"], "win_pct": met["win_pct"],
                            "gross_per_trade": round(met["gross"] / max(met["trades"], 1), 2),
                            "cost_per_trade": round(met["costs"] / max(met["trades"], 1), 2),
                            "gross_r_per_trade": round(float(np.mean([t[10] / 10000.0 for t in tr])), 4) if tr else None,
                            "screen_passed": scr["passed"], "reasons": scr["reasons"], "candidates": res["candidates"],
                            "skipped": res["skipped"]}
        _jsave(os.path.join(CTRL, "controls.json"), out)
        log(f"  controls {tf}: " + ", ".join(f"{r.replace('CTRL_', '')} PF {v['pf']}" for r, v in out[tf].items()
                                             if r in ("CTRL_RANDOM_1", "CTRL_PLANTED_5", "CTRL_PLANTED_20",
                                                      "CTRL_PLANTED_100")))
    return out


# ══════════════════════════════════════════════════════════════════════════
# the live PAPER strategies + AUC-FADE-2PCT, Rs 10 lakh accounts, both eras (report only)
# ══════════════════════════════════════════════════════════════════════════
def validated_agent(sid: str) -> K.AgentConfig:
    cv = K.config_values()
    if sid.startswith("AUC-FADE"):
        dl = cv["validated_auction_daily_loss_stop"] * (20000.0 / 16000.0 if sid == "AUC-FADE-2PCT" else 1.0)
        return K.AgentConfig("auction", int(cv["validated_auction_max_gross"] // 100000), None, dl,
                             fixed_notional=100000.0, min_value=0.0)
    if sid.startswith("SWING-ATTN"):
        return K.AgentConfig("attention", int(cv["validated_attention_max_gross"] // 100000), None,
                             cv["validated_attention_daily_loss_stop"], fixed_notional=100000.0, min_value=0.0)
    return K.AgentConfig("strangle", 1, None, None, min_value=0.0)


def _module_trades(era: str) -> dict:
    """{id: trade tuples} from each strategy's own runner on one era (2023-26 = DISCOVERY + FINAL, both already seen;
    2016-23 = the one-look era, dirty rule applied to the candidate rows as protocol_min_oos does)."""
    from . import auction as A, attention as T, options_vrp as V
    out = {}
    if era == "2023-26":
        lo, hi = G.DISC[0], G.SPLIT3["final"][1]
        rows, _, _, _ = A.load_universe(G.BARS3Y, None, verbose=False)
        out.update({k: v["trades"] for k, v in A.make_simulator(rows)(A.SPECS, lo, hi).items()})
        raw, _, counts, _, _ = T.load_universe(G.BARS3Y, None, verbose=False)
        rows, _ = T.filter_rows(raw, counts)
        out.update({k: v["trades"] for k, v in T.make_simulator(rows)(T.SPECS, lo, hi).items()})
        bars, vix = V.load(G.BARS3Y)
        vs = [sp for sp in V.SPECS if sp.id == "VRP-STRANGLE-016-WINGED"]
        r = V.make_simulator(bars, vix, audit=False)(vs, lo, hi)
        out.update({k: v["trades"] for k, v in r.items()})
        margins = V.DETAILS.get(("VRP-STRANGLE-016-WINGED", 1.0, "base"), {})
        return {k: v for k, v in out.items() if k in G.LIVE4}, {d: x["margin"] for d, x in margins.items()}
    from . import protocol_min_oos as PMO
    PMO.setup(G.CACHE_OLD)
    for which, mod in (("auction", A), ("attention", T)):
        specs = [sp for sp in mod.SPECS if sp.id in G.LIVE4]
        r = PMO.run_auction(specs) if which == "auction" else PMO.run_attention(specs)
        out.update({k: v["trades"] for k, v in r.items()})
    vs = [sp for sp in V.SPECS if sp.id == "VRP-STRANGLE-016-WINGED"]
    r = PMO.run_options(vs, "options_vrp")
    out.update({k: v["trades"] for k, v in r.items()})
    margins = V.DETAILS.get(("VRP-STRANGLE-016-WINGED", 1.0, "base"), {})
    return {k: v for k, v in out.items() if k in G.LIVE4}, {d: x["margin"] for d, x in margins.items()}


def _cands_from_trades(sid, trades, inst, margins=None, prio=None) -> list:
    out = []
    for t in trades:
        e = int(t[2]) * 10000 + G._hm2m(str(t[4]), 555)
        xt = str(t[7])
        x = int(t[6] or t[2]) * 10000 + G._hm2m(xt, 929) + (1 if xt in ("", "15:29") else 0)
        pr = (prio or {}).get((t[1], int(t[2])), 0.0)
        if inst == "opt":
            m = float((margins or {}).get(int(t[2]), 0.0))
            out.append(K.cand(t[1], int(t[3]), e, float(t[5]), float("nan"), x, float(t[8]), str(t[13]), pr, "opt",
                              1, 1, sid, opt=(int(t[9]), float(t[10]), float(t[11]), m)))
        else:
            out.append(K.cand(t[1], int(t[3]), e, float(t[5]), float("nan"), x, float(t[8]), str(t[13]), pr, inst,
                              1, 1, sid))
    return out


def bracket_rr(trades, cache, sid, era_cal=None) -> list:
    """Report-only 1:3 version of AUC-FADE / ATTN trades: stop = 1.5 x ATR(14) of the trade's own timeframe
    (AUC-FADE: 5-minute candles before the 09:15 decision; ATTN: daily candles before the decision day), target =
    3 x that distance from the fill; minutes from the fill to the minute before the original exit are checked
    (stop first; a target needs one tick through), else the original exit."""
    from .sim import target_fill
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t[1]].append(t)
    out = []
    for sym, tl in by_sym.items():
        sd = D.load_symbol(cache, sym)
        if sd is None:
            out.extend(tl)
            continue
        B = G.Bars(sd)
        C5 = G.candles(B, "5m")
        a5 = G.KI.atr(C5.h, C5.l, C5.c, 14)
        Cd = G.candles(B, "1d")
        ad = G.KI.atr(Cd.h, Cd.l, Cd.c, 14)
        for t in tl:
            side, fill = int(t[3]), float(t[5])
            ek = int(t[2]) * 10000 + G._hm2m(str(t[4]), 555)
            f = int(np.searchsorted(B.key, ek))
            xk = int(t[6]) * 10000 + G._hm2m(str(t[7]), 929)
            xm = int(np.searchsorted(B.key, xk))
            if f >= len(B) or B.key[f] != ek:
                out.append(t)
                continue
            if sid.startswith("AUC"):
                j = int(np.searchsorted(C5.last, f)) - 1          # last 5-minute candle complete before the fill
                dist = ATR = float(a5[j]) if j >= 0 else float("nan")
            else:
                j = int(B.sess[f]) - 1                            # daily ATR of the sessions before the decision day
                dist = ATR = float(ad[j]) if j >= 0 else float("nan")
            if not np.isfinite(ATR):
                out.append(t)
                continue
            stop = fill - side * G.ATR_STOP * dist
            tgt = fill + side * G.RR * G.ATR_STOP * dist
            slip = 2.1e-4 if sid.startswith("AUC") else 2e-4
            px, why, k_ = None, None, None
            for k in range(f, max(f, xm)):
                lo_, hi_, o_ = float(B.l[k]), float(B.h[k]), float(B.o[k])
                if (lo_ <= stop) if side > 0 else (hi_ >= stop):
                    q = min(stop, o_) if side > 0 else max(stop, o_)
                    px, why, k_ = q * (1 - side * slip), "stop", k
                    break
                tk = G._tick(tgt, int(B.date[k]), False)
                tp = target_fill(side, tgt, o_, hi_ if side > 0 else lo_, tk)
                if tp is not None:
                    px, why, k_ = tp, "target", k
                    break
            if px is None:
                out.append(t)
                continue
            q = int(t[9])
            gross = side * (px - fill) * q
            bv, sv = (fill * q, px * q) if side > 0 else (px * q, fill * q)
            from costs import round_trip_cost
            cost = round_trip_cost(bv, sv, "equity_mis" if sid.startswith("AUC") else "future")
            tm = int(B.t[k_])
            out.append((t[0], t[1], t[2], t[3], t[4], t[5], int(B.date[k_]), f"{tm // 60:02d}:{tm % 60:02d}",
                        round(px, 4), q, round(gross, 2), cost, round(gross - cost, 2), why))
    return out


def live(jobs: int):
    res_all = _jload(os.path.join(LIVE, "live.json"), {}) or {}
    for era in ("2023-26", "2016-23"):
        if era in res_all:
            continue
        trades, margins = _module_trades(era)
        if era == "2023-26":
            cal = G.calendar3()
            lo, hi = G.DISC[0], G.SPLIT3["final"][1]
            cache, dirty, nxt = G.CACHE3Y, None, None
        else:
            from . import protocol_min_oos as PMO
            ctx = PMO.setup(G.CACHE_OLD)
            cal = [int(x) for x in ctx["cal"]]
            lo, hi = G.OLD_ERA
            cache, dirty, nxt = G.CACHE_OLD, ctx["dirty"], ctx["nxt"]
        out = {}
        os.makedirs(os.path.join(LIVE, "trades"), exist_ok=True)
        for sid in G.LIVE4:
            tr = trades.get(sid, [])
            if dirty is not None and not sid.startswith("VRP"):
                tr = [t for t in tr if not _dirty(t, dirty, nxt)]
            inst = "mis" if sid.startswith("AUC") else ("fut" if sid.startswith("SWING") else "opt")
            for variant in ("own_exits", "rr3_report_only"):
                if variant == "rr3_report_only":
                    if sid.startswith("VRP"):
                        out[f"{sid}|{variant}"] = {"note": "not applicable: the defined risk is the wing loss; a 3x "
                                                           "target exceeds the maximum profit (the credit), so the "
                                                           "bracket never triggers: identical to own exits"}
                        continue
                    tr_v = bracket_rr(tr, cache, sid)
                else:
                    tr_v = tr
                ag = validated_agent(sid)
                res = K.simulate(_cands_from_trades(sid, tr_v, inst, margins), ag, sid)
                met = K.metrics(res, cal, lo, hi)
                comp = K.simulate(_cands_from_trades(sid, tr_v, inst, margins), replace(ag, compound=True), sid)
                row = {"agent": ag.as_dict(), "module_trades": len(tr_v), "metrics": met,
                       "compounded": K.metrics(comp, cal, lo, hi)}
                if era == "2023-26":
                    for name, (a, b) in (("discovery", G.DISC), ("final_already_seen", G.SPLIT3["final"])):
                        row[name] = K.metrics(res, cal, a, b)
                else:
                    row["h1"] = K.metrics(res, cal, 20160101, 20191231)
                    row["h2"] = K.metrics(res, cal, 20200101, 20231109)
                row["daily"] = [round(x, 2) for x in K.daily_series(res, cal, lo, hi)]
                out[f"{sid}|{variant}"] = row
                G.save_trades(os.path.join(LIVE, "trades", G._fn(f"{era}_{sid}_{variant}") + ".pkl.gz"),
                              res["trades"])
                log(f"  live {era} {sid} {variant}: {met['trades']} tr, net {met['net']:,.0f}, PF {met['pf']}, "
                    f"maxDD {met['max_dd_rs']:,.0f}")
        res_all[era] = out
        _jsave(os.path.join(LIVE, "live.json"), res_all)
    return res_all


def _dirty(t, dirty, nxt) -> bool:
    from .protocol_min_oos import _dirty_hit
    return _dirty_hit(dirty, t[1], int(t[2]), int(t[6] or t[2]), nxt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="tfgrid run stages")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--no-natives", action="store_true")
    ap.add_argument("--tfs", default="")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a, _ = ap.parse_known_args(argv)
    if a.stage1:
        stage1(a.jobs, [x for x in a.tfs.split(",") if x] or None, natives=not a.no_natives)
    if a.final:
        final(a.jobs)
    if a.stage2:
        stage2(a.jobs)
    if a.controls:
        controls(a.jobs)
    if a.live:
        live(a.jobs)
    if a.report:
        from . import tfgrid_report
        tfgrid_report.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
