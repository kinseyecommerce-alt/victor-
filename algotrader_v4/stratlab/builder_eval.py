"""
stratlab/builder_eval.py — one batch through the frozen pipeline of builder_rule.json (stages DISCOVERY -> VALIDATION
-> FINAL -> proof). Research only. Resumable: every finished unit is a JSON file in the batch's work directory and is
returned as stored on a re-run (never recomputed); FINAL is opened only through the caller's `open_final` callback,
which records the look in the hash-chained registry BEFORE any FINAL return is computed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time

import numpy as np

from . import builder_engine as BE
from . import builder_grammar as G
from . import protocol as P
from . import sim


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")


def _jsave(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(BE.clean(obj), f, separators=(",", ":"))
    os.replace(tmp, path)


def _jload(path: str):
    with open(path) as f:
        return json.load(f)


def write_trades(path: str, trades: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(",".join(sim.TRADE_FIELDS) + "\n")
        for t in trades:
            f.write(",".join(str(x) for x in t) + "\n")


def read_trades(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            out.append((p[0], p[1], int(p[2]), int(p[3]), p[4], float(p[5]), int(p[6]), p[7], float(p[8]), int(p[9]),
                        float(p[10]), float(p[11]), float(p[12]), p[13]))
    return out


# ══════════════════════════════════════════════════════════════════════════
# controls
# ══════════════════════════════════════════════════════════════════════════
def shuffled_sig(E: BE.Engine, c: G.Candidate, lo: int, hi: int, seed: int) -> dict:
    """SHUFFLE-THE-FEATURE: inside the window each symbol's trigger cells (side, rank, hold kept) move to dates drawn
    without replacement from that symbol's eligible dates OF THE SAME MARKET TYPE."""
    sig = E.signals(c)
    cal = E.calendar(c)
    off = 1 if c.grid in ("daily", "idx_daily") else 0
    rng = np.random.default_rng(seed)
    es, et = E.eligible(c, lo, hi)
    pool: dict = {}
    for s, t in zip(es.tolist(), et.tolist()):
        pool.setdefault(s, []).append(t)
    by: dict = {}
    for k, t in enumerate(sig["t"].tolist()):
        if lo <= int(cal[min(t + off, len(cal) - 1)]) <= hi:
            by.setdefault(int(sig["s"][k]), []).append(k)
    S, T_, SD, RK, HD = [], [], [], [], []
    for s in sorted(by):
        ks = by[s]
        p = pool.get(s, [])
        if not p:
            continue
        n = min(len(ks), len(p))
        new_t = rng.choice(np.asarray(p), size=n, replace=False)
        for k, t in zip(ks[:n], new_t.tolist()):
            S.append(s), T_.append(t), SD.append(int(sig["side"][k])), RK.append(float(sig["rank"][k]))
            HD.append(int(sig["hold"][k]))
    return E.sort_sig(c, S, T_, SD, RK, HD)


def placebo_sig(E: BE.Engine, c: G.Candidate, lo: int, hi: int, n: int, long_share: float, hold: int, seed: int):
    rng = np.random.default_rng(seed)
    es, et = E.eligible(c, lo, hi)
    if not len(es) or n <= 0:
        return E.sort_sig(c, [], [], [], [], [])
    k = rng.choice(len(es), size=min(n, len(es)), replace=False)
    sd = np.where(rng.random(len(k)) < long_share, 1, -1)
    return E.sort_sig(c, es[k].tolist(), et[k].tolist(), sd.tolist(), rng.random(len(k)).tolist(), [hold] * len(k))


def planted_candidates(rule: dict) -> dict:
    """{agent: Candidate} — the planted-edge carriers (rule["controls"]["planted"]["carriers"]). Under a bg3 rule a
    carrier takes the size regime of its own sentence (builder_grammar.size_regime); bg1 / bg2: Rs 1 lakh."""
    out = {}
    sized = (rule.get("grammar") or {}).get("version") == G.GRAMMAR_V3
    for agent, d in rule["controls"]["planted"]["carriers"].items():
        params = tuple(tuple(p) for p in d.get("params", []))
        size = G.size_regime(agent, d["block"], params, d["instrument"]) if sized else ""
        out[agent] = G.Candidate(agent, "RISING", d["block"], params, d["entry"], tuple(d["exit"]), d["instrument"],
                                 d["product"], d.get("structure", ""), "", size)
    return out


def planted_sig(E: BE.Engine, c: G.Candidate, lo: int, hi: int, frac: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    es, et = E.eligible(c, lo, hi, mtype=None)
    k = rng.random(len(es)) < frac
    es, et = es[k], et[k]
    return E.sort_sig(c, es.tolist(), et.tolist(), [1] * len(es), rng.random(len(es)).tolist(), [0] * len(es))


# ══════════════════════════════════════════════════════════════════════════
# the batch
# ══════════════════════════════════════════════════════════════════════════
class Batch:
    def __init__(self, E: BE.Engine, batch_id: str, cands: list, rule: dict, work: str, n_base: int,
                 log=print, open_final=None):
        """cands: [(id, Candidate)]; n_base = N before this batch (lab count + earlier builder candidates + earlier
        FINAL looks)."""
        self.E, self.id, self.cands, self.rule, self.work = E, batch_id, cands, rule, work
        self.by = dict(cands)
        self.n_base = int(n_base)
        self.log = log
        self.open_final = open_final
        self.W = BE.windows(E.src.end, rule)
        os.makedirs(work, exist_ok=True)

    # windows per candidate
    def disc(self, c):
        cal = self.E.calendar(c)
        lo = int(cal[0])
        return lo, int(self.W["discovery"][1])

    def p(self, name):
        return os.path.join(self.work, name)

    # ── DISCOVERY (+ the per-candidate shuffle) ──
    def discovery(self) -> dict:
        rows = {}
        for agent in G.AGENTS:
            rows.update(self.discovery_agent(agent))
        return rows

    def discovery_agent(self, agent: str) -> dict:
        path = self.p(f"discovery_{agent}.json")
        if os.path.exists(path):
            return _jload(path)
        t0 = time.time()
        jobs = []
        mine = [(cid, c) for cid, c in self.cands if c.agent == agent]
        for cid, c in mine:
            lo, hi = self.disc(c)
            jobs.append(BE.Job(cid, c, None, lo, hi, True))
            sh = shuffled_sig(self.E, c, lo, hi, _seed(self.rule["version"], self.id, cid, "shuffle"))
            jobs.append(BE.Job(cid + "#shuffle", c, sh, lo, hi, True))
        res = BE.run_jobs(self.E, jobs)
        out = {}
        for cid, c in mine:
            lo, hi = self.disc(c)
            cal = self.E.calendar(c)
            h1, h2 = BE.halves_of(cal, lo, hi)
            r = res[cid]
            j = BE.judge_discovery(c, r["trades"], r["trades5"], h1, h2, self.rule)
            st = BE.series(cal, r["trades"], lo, hi)
            rs = res[cid + "#shuffle"]
            js = BE.judge_discovery(c, rs["trades"], rs["trades5"], h1, h2, self.rule)
            out[cid] = {"judge": j, "series": st, "window": [lo, hi], "halves": [h1, h2],
                        "shuffle": {"passed": js["passed"], "trades": js["disc"]["trades"], "net": js["disc"]["net"],
                                    "pf": js["disc"]["pf"], "why": js["why"]}}
        _jsave(path, out)
        self.log(f"[builder {self.id}] discovery {agent}: {sum(1 for v in out.values() if v['judge']['passed'])} "
                 f"of {len(out)} passed, shuffle {sum(1 for v in out.values() if v['shuffle']['passed'])} "
                 f"({time.time() - t0:.0f}s)")
        return out

    # ── planted / random controls per agent ──
    def planted(self) -> dict:
        path = self.p("planted.json")
        if os.path.exists(path):
            return _jload(path)
        cfg = self.rule["controls"]["planted"]
        carriers = planted_candidates(self.rule)
        jobs, meta = [], {}
        for agent, c in carriers.items():
            lo, hi = self.disc(c)
            frac = float(cfg["carriers"][agent]["frac"])
            for p_ in (float(cfg["p"]), float(cfg["p_random"])):
                key = f"ctl_{agent}_p{int(round(p_ * 100))}"
                sig = planted_sig(self.E, c, lo, hi, frac, _seed(self.rule["version"], self.id, key))
                jobs.append(BE.Job(key, c, sig, lo, hi, True, planted_p=p_,
                                   seed=_seed(self.rule["version"], self.id, key, "side")))
                meta[key] = (agent, c, p_)
        res = BE.run_jobs(self.E, jobs)
        out = {}
        for key, (agent, c, p_) in meta.items():
            lo, hi = self.disc(c)
            h1, h2 = BE.halves_of(self.E.calendar(c), lo, hi)
            j = BE.judge_discovery(c, res[key]["trades"], res[key]["trades5"], h1, h2, self.rule)
            gross = lambda tr: [t[:12] + (t[10],) + t[13:] for t in tr]       # net := gross (before fees)
            jg = BE.judge_discovery(c, gross(res[key]["trades"]), gross(res[key]["trades5"]), h1, h2, self.rule)
            out[key] = {"agent": agent, "p": p_, "passed": jg["passed"], "why": jg["why"],
                        "trades": j["disc"]["trades"], "pf_gross": jg["disc"]["pf"], "gross": jg["disc"]["net"],
                        "pf": j["disc"]["pf"], "net": j["disc"]["net"], "passed_after_costs": j["passed"],
                        "why_after_costs": j["why"]}
        for agent in carriers:
            pl = out[f"ctl_{agent}_p{int(round(float(cfg['p']) * 100))}"]
            rn = out[f"ctl_{agent}_p{int(round(float(cfg['p_random']) * 100))}"]
            out[f"gate_{agent}"] = bool(pl["passed"] and not rn["passed"])
        _jsave(path, out)
        self.log(f"[builder {self.id}] planted controls: " +
                 ", ".join(f"{a} {'ok' if out[f'gate_{a}'] else 'FAIL'}" for a in carriers))
        return out

    # ── selection ──
    def select(self, disc: dict) -> dict:
        path = self.p("selection.json")
        if os.path.exists(path):
            return _jload(path)
        srs = [v["series"]["sr_daily"] for v in disc.values()
               if v["series"]["n_days"] and v["series"]["sr_daily"] is not None]
        v_mad = BE.robust_var(srs)
        floors = [1.0 / (v["series"]["n_days"] - 1) for v in disc.values() if (v["series"]["n_days"] or 0) > 1]
        floor = float(np.median(floors)) if floors else float("nan")
        v = max(x for x in (v_mad, floor) if x == x) if (v_mad == v_mad or floor == floor) else float("nan")
        n_disc = self.n_base + len(self.cands)
        dsr = {cid: BE.dsr_of(d["series"], n_disc, v) for cid, d in disc.items()}
        K = int(self.rule["carry"]["K_per_cell"])
        carried = []
        for agent in G.AGENTS:
            for mt in G.MARKET_TYPES:
                ok = [cid for cid, c in self.cands if c.agent == agent and c.mtype == mt and disc[cid]["judge"]["passed"]
                      and dsr[cid]["dsr"] == dsr[cid]["dsr"]]
                ok.sort(key=lambda cid: (-dsr[cid]["dsr"], cid))
                carried += ok[:K]
        out = {"n_disc": n_disc, "v_mad": v_mad, "v_floor": floor, "v": v, "dsr": dsr, "carried": carried}
        _jsave(path, out)
        return out

    # ── VALIDATION ──
    def validation(self, carried: list) -> dict:
        path = self.p("validation.json")
        if os.path.exists(path):
            return _jload(path)
        lo, hi = self.W["validation"]
        jobs = [BE.Job(cid, self.by[cid], None, lo, hi, False) for cid in carried]
        res = BE.run_jobs(self.E, jobs) if jobs else {}
        out = {}
        for cid in carried:
            c = self.by[cid]
            cal = self.E.calendar(c)
            tr = res[cid]["trades"]
            write_trades(self.p(f"trades/val_{cid}.csv.gz"), tr)
            sc = BE.screen_window(c, tr, cal, lo, hi)
            out[cid] = {"screen": sc, "series": BE.series(cal, tr, lo, hi), "passed": sc["passed"]}
        _jsave(path, out)
        self.log(f"[builder {self.id}] validation: {sum(1 for v in out.values() if v['passed'])} of {len(out)} passed")
        return out

    # ── FINAL: one look ──
    def final(self, passers: list) -> dict:
        path = self.p("final.json")
        if os.path.exists(path):
            return _jload(path)
        if passers and self.open_final is not None:
            self.open_final(passers)                   # registry record BEFORE any FINAL return exists
        lo, hi = self.W["final"]
        jobs = [BE.Job(cid, self.by[cid], None, lo, hi, False) for cid in passers]
        res = BE.run_jobs(self.E, jobs) if jobs else {}
        out = {}
        for cid in passers:
            c = self.by[cid]
            tr = res[cid]["trades"]
            write_trades(self.p(f"trades/fin_{cid}.csv.gz"), tr)
            fv = P.final_verdict(tr)
            out[cid] = {"final": fv, "confirmed": bool(fv["confirmed"]),
                        "series": BE.series(self.E.calendar(c), tr, lo, hi)}
        _jsave(path, out)
        self.log(f"[builder {self.id}] FINAL: {sum(1 for v in out.values() if v['confirmed'])} of {len(out)} confirmed")
        return out

    # ── proof ──
    def proof(self, confirmed: list, disc: dict, sel: dict, planted: dict, n_proof: int) -> dict:
        path = self.p("proof.json")
        if os.path.exists(path):
            return _jload(path)
        out = {}
        v0, T = self.W["validation"][0], self.W["final"][1]
        nplac = int(self.rule["controls"]["placebo"]["n"])
        for cid in confirmed:
            c = self.by[cid]
            cal = self.E.calendar(c)
            dlo, dhi = self.disc(c)
            r = BE.run_jobs(self.E, [BE.Job("d", c, None, dlo, dhi, False), BE.Job("u", c, None, v0, T, False)])
            tr_d, tr_u = r["d"]["trades"], r["u"]["trades"]
            st_full = BE.series(cal, tr_d + tr_u, dlo, T)
            d = BE.dsr_of(st_full, n_proof, sel["v"])
            n = len(tr_u)
            ls = (sum(1 for t in tr_u if t[3] > 0) / n) if n else 1.0
            sig = self.E.signals(c)
            hold = int(np.median([x for x in sig["hold"].tolist() if x > 0] or [0]))
            pj = [BE.Job(f"pl{k}", c, placebo_sig(self.E, c, v0, T, n, ls, hold, _seed(self.id, cid, "placebo", k)),
                         v0, T, False) for k in range(nplac)]
            pr = BE.run_jobs(self.E, pj)
            nets = [sum(t[12] for t in pr[j.key]["trades"]) for j in pj]
            p95 = float(np.percentile(nets, 95)) if nets else float("nan")
            actual = sum(t[12] for t in tr_u)
            gates = {"validation": True, "final": True,
                     "dsr": (d["dsr"] or 0) >= float(self.rule["dsr"]["min"]),
                     "placebo": actual > p95,
                     "shuffle": not disc[cid]["shuffle"]["passed"],
                     "planted": bool(planted.get(f"gate_{c.agent}"))}
            from . import trials as TR
            ref = TR.daily_pnl(BE.R._tdicts(tr_u), [int(x) for x in cal], v0, T)
            out[cid] = {"dsr": d, "sr_daily_full": st_full["sr_daily"], "n_days_full": st_full["n_days"],
                        "reference_daily": [round(x, 2) for x in ref],
                        "placebo": {"n": n, "long_share": round(ls, 3), "p95": p95, "median": float(np.median(nets))
                                    if nets else None, "actual": actual}, "gates": gates,
                        "proven": all(gates.values()),
                        # the universe as resolved on the frozen data: a promoted strategy trades exactly these names
                        # in its forward family (None = no stock universe / TOP500 = no filter)
                        "universe": c.stock_universe, "universe_names": self.E.frozen_names(c)}
            self.log(f"[builder {self.id}] proof {cid}: DSR {d['dsr']:.3f} placebo {actual:,.0f} vs p95 {p95:,.0f} "
                     f"-> {'PROVEN' if out[cid]['proven'] else 'not proven'}")
        _jsave(path, out)
        return out

    # ── everything ──
    def run(self) -> dict:
        t0 = time.time()
        disc = self.discovery()
        planted = self.planted()
        sel = self.select(disc)
        val = self.validation(sel["carried"])
        passers = [cid for cid in sel["carried"] if val[cid]["passed"]]
        fin = self.final(passers)
        confirmed = [cid for cid in passers if fin[cid]["confirmed"]]
        n_proof = sel["n_disc"] + len(passers)
        prf = self.proof(confirmed, disc, sel, planted, n_proof)
        return self.summary(disc, planted, sel, val, fin, prf, n_proof, time.time() - t0)

    def summary(self, disc, planted, sel, val, fin, prf, n_proof, secs) -> dict:
        cells = {}
        rows = []
        for cid, c in self.cands:
            d = disc[cid]
            stage = "discovery"
            if d["judge"]["passed"]:
                stage = "discovery_passed"
            if cid in sel["carried"]:
                stage = "validation_failed"
                if val[cid]["passed"]:
                    stage = "final_rejected"
                    if fin.get(cid, {}).get("confirmed"):
                        stage = "proven" if prf.get(cid, {}).get("proven") else "final_confirmed"
            j = d["judge"]
            rows.append({"id": cid, "agent": c.agent, "mtype": c.mtype, "block": c.block, "key": c.key,
                         "universe": c.stock_universe, "text": G.describe(c), "stage": stage,
                         "discovery": {"trades": j["disc"]["trades"], "pf": j["disc"]["pf"], "net": j["disc"]["net"],
                                       "h1": j["h1"]["net"], "h2": j["h2"]["net"], "why": j["why"],
                                       "dsr": sel["dsr"][cid]["dsr"], "sr_daily": d["series"]["sr_daily"],
                                       "passed": j["passed"], "shuffle_passed": d["shuffle"]["passed"]},
                         "validation": val.get(cid), "final": fin.get(cid), "proof": prf.get(cid)})
            k = f"{c.agent}|{c.mtype}"
            cl = cells.setdefault(k, {"candidates": 0, "discovery": 0, "shuffle": 0, "validation_tested": 0,
                                      "validation": 0, "final": 0, "proven": 0})
            cl["candidates"] += 1
            cl["discovery"] += int(j["passed"])
            cl["shuffle"] += int(d["shuffle"]["passed"])
            cl["validation_tested"] += int(cid in sel["carried"])
            cl["validation"] += int(bool(val.get(cid, {}).get("passed")))
            cl["final"] += int(bool(fin.get(cid, {}).get("confirmed")))
            cl["proven"] += int(stage == "proven")
        best = max(rows, key=lambda r: (r["discovery"]["dsr"] if r["discovery"]["dsr"] is not None
                                        and r["discovery"]["dsr"] == r["discovery"]["dsr"] else -1))
        return {"batch": self.id, "rule_version": self.rule["version"], "windows": self.W,
                "candidates": len(self.cands), "discovery_passers": sum(1 for r in rows if r["discovery"]["passed"]),
                "shuffle_passers": sum(1 for r in rows if r["discovery"]["shuffle_passed"]),
                "validation_tested": len(sel["carried"]), "validation_passers": len(fin),
                "final_looks": len(fin), "final_confirmed": sum(1 for v in fin.values() if v["confirmed"]),
                "proven": [r["id"] for r in rows if r["stage"] == "proven"],
                "n_disc": sel["n_disc"], "n_proof": n_proof, "v": sel["v"], "v_mad": sel["v_mad"],
                "v_floor": sel["v_floor"], "best_discovery_dsr": {"id": best["id"], "dsr": best["discovery"]["dsr"]},
                "planted": planted, "cells": cells, "rows": rows, "seconds": round(secs, 1),
                "engine_stats": self.E.stats, "universes": self.universes()}

    def universes(self) -> dict:
        """What each universe resolved to on the batch's frozen data (+ the frozen name list of every proven
        candidate's universe, which its forward family entry trades)."""
        rep = self.E.universe_report()
        rep["counts"] = {}
        for _cid, c in self.cands:
            rep["counts"][c.stock_universe] = rep["counts"].get(c.stock_universe, 0) + 1
        return rep
