"""
stratlab/size10l_options.py — AMENDMENT 1 of the Rs 10 lakh-per-agent re-test: the lab's OPTIONS strategies are
re-sized and re-tested too (user: "not only intraday, swing - test all"), because the flat Rs 20 per order per leg is a
large share of a one-lot option trade. Pre-registered in PREREGISTERED_SIZE10L_AMENDMENT1.json (it supersedes
PREREGISTERED_SIZE10L.json, which is kept and referenced) BEFORE any option result at the new size was computed.

Lots per trade, decided at ENTRY from entry prices only (builder/v3's definition):
    long options and debit structures     lots = max(1, floor(Rs 1,00,000 / net premium paid per lot))
    defined-risk short structures         lots = max(1, floor(Rs 1,00,000 / max risk per lot))
    naked short structures                lots = max(1, floor(Rs 5,00,000 / margin per lot)); margin = the lab's own
                                          estimate where the trade record carries one (options_replay / options_vrp),
                                          else 12 % of (mean short strike x lot size)
Per-leg fees at the real lot count (options_replay.leg_cost / costs.round_trip_cost "option"); the fill haircut per unit
is unchanged. Two engines, neither edited:
    replay  (options_vrp, options_r3: Black-Scholes on the index minute bars)  the stored per-leg trade files are
            re-sized exactly: gross scales with the lots, fees = flat part (Rs 20 + GST per order with a positive value)
            + the proportional part x the lot ratio;
    nse     (options_r4, kite_options: real NSE EOD prices)  the modules' own simulator runs again with
            nsedata._close_position wrapped (the wrapper re-computes gross and options_replay.leg_cost per leg at the
            new lot count; its one-lot twin is the engine's own trade).

    python -m stratlab.size10l_options --preregister
    python -m stratlab.size10l_options --stage screen | final | dsr | ledger
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

from . import size10l_run as Z

SCRATCH = Z.SCRATCH
OUT = Z.OUT
PREREG2 = os.path.join(SCRATCH, "PREREGISTERED_SIZE10L_AMENDMENT1.json")
REPO_PREREG2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREGISTERED_SIZE10L_AMENDMENT1.json")
FINAL_LEDGER = os.path.join(OUT, "options_final_ledger.json")
BUDGET_PREMIUM = 100_000.0
BUDGET_RISK = 100_000.0
BUDGET_MARGIN = 500_000.0
MARGIN_PCT = 0.12
FLAT = 20.0 * 1.18                       # Rs 20 per order + 18 % GST
MIN_TRADES = 60                          # positional
MIN_PF = 1.15
REPLAY = (("dd/vrp/final_run", ("VRP-STRANGLE-016-NAKED",)),
          ("r3/options/full", None))     # None = every trade file in the directory
REPORT_ONLY = ("VRP-STRANGLE-016-WINGED",)          # a live rule: re-sized for the report, not a trial


# ══════════════════════════════════════════════════════════════════════════
# the lot rule
# ══════════════════════════════════════════════════════════════════════════
def new_lots(legs: list, lot: int, margin_per_lot: float | None = None) -> tuple:
    """legs = [(side +1/-1, leg lots, strike, 'CE'|'PE', entry price, expiry)] of ONE structure (one 'lot' of it).
    Returns (lots, rule). Uses entry prices only."""
    debit = sum(s * n * px for s, n, _k, _t, px, _e in legs)            # > 0: net premium paid (points)
    net_call = sum(s * n for s, n, _k, t, _px, _e in legs if t == "CE")
    net_put = sum(s * n for s, n, _k, t, _px, _e in legs if t == "PE")
    one_expiry = len({e for *_x, e in legs}) == 1
    covered = net_call >= 0 and net_put >= 0
    if all(s > 0 for s, *_x in legs) or (debit > 0 and covered):
        per_lot = debit * lot
        return (max(1, int(BUDGET_PREMIUM // per_lot)) if per_lot > 0 else 1), "premium"
    if covered and one_expiry:
        pts = [0.0] + sorted({k for _s, _n, k, _t, _px, _e in legs})

        def payoff(x):
            return sum(s * n * ((max(0.0, x - k) if t == "CE" else max(0.0, k - x)) - px) for s, n, k, t, px, _e in legs)
        worst = -min(payoff(x) for x in pts)
        per_lot = worst * lot
        return (max(1, int(BUDGET_RISK // per_lot)) if per_lot > 0 else 1), "max_risk"
    if margin_per_lot is None or not margin_per_lot > 0:
        shorts = [k for s, _n, k, _t, _px, _e in legs if s < 0]
        margin_per_lot = MARGIN_PCT * (sum(shorts) / len(shorts)) * lot
    return max(1, int(BUDGET_MARGIN // margin_per_lot)), "margin"


def flat_orders(legs_px: list) -> int:
    """Orders that pay the flat brokerage in costs.round_trip_cost: every leg side with a positive value."""
    return sum((1 if a > 0 else 0) + (1 if b > 0 else 0) for a, b in legs_px)


def resize_fees(fees0: float, n_flat: int, ratio: float) -> float:
    return round(n_flat * FLAT + ratio * (fees0 - n_flat * FLAT), 2)


# ══════════════════════════════════════════════════════════════════════════
# replay engine: stored per-leg trade files
# ══════════════════════════════════════════════════════════════════════════
def _ymd(s: str) -> int:
    return int(str(s).split()[0].replace("-", ""))


def replay_rows(path: str) -> list:
    """[{date, exit, net1, gross1, fees1, lots0, lots, rule, net, gross, fees}] of one stored trade file."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            lot, qty = int(float(r["lot"])), int(float(r["qty"]))
            lots0 = max(1, qty // lot)
            legs, px = [], []
            for part in r["legs"].split(" | "):
                sd, kt, pp = part.split()
                a, b = pp.split("->")
                legs.append((1 if sd == "B" else -1, 1, float(kt[:-2]), kt[-2:], float(a), 0))
                px.append((float(a), float(b)))
            margin = float(r["margin"]) / lots0 if r.get("margin") not in (None, "") else None
            lots, rule = new_lots(legs, lot, margin)
            ratio = lots / lots0
            g1, f1, n1 = float(r["gross"]), float(r["fees"]), float(r["net"])
            fees = resize_fees(f1, flat_orders(px) , ratio)
            gross = round(g1 * ratio, 2)
            out.append({"date": int(r["date"]), "exit": _ymd(r["exit"]) if r.get("exit") else _ymd(r["expiry"]),
                        "net1": n1, "gross1": g1, "fees1": f1, "lots0": lots0, "lots": lots, "rule": rule,
                        "gross": gross, "fees": fees, "net": round(gross - fees, 2)})
    return out


# ══════════════════════════════════════════════════════════════════════════
# nse engine: wrapper around nsedata._close_position
# ══════════════════════════════════════════════════════════════════════════
_O: dict = {}
_LOG: list = []


def _leg_cost(entry_px, exit_px, side, qty, settled=False):
    _LOG.append((float(entry_px), float(exit_px), int(side), int(qty), bool(settled)))
    return _O["leg_cost"](entry_px, exit_px, side, qty, settled=settled)


def _close(spec, pos, day, reason, mult, settle):
    del _LOG[:]
    tup, det = _O["close"](spec, pos, day, reason, mult, settle)
    log = list(_LOG)
    legs = [(l.side, l.lots, float(l.strike), l.typ, float(f), int(l.expiry or pos.plan.expiry))
            for l, f in zip(pos.plan.legs, pos.fills)]
    if len(log) != len(legs):
        raise RuntimeError(f"{spec.id}: {len(log)} leg costs for {len(legs)} legs")
    lots, rule = new_lots(legs, int(pos.qty_per_lot))
    gross = sum(sd * (xo - xi) * q * lots for xi, xo, sd, q, _st in log)
    fees = sum(_O["leg_cost"](xi, xo, sd, q * lots, settled=st) for xi, xo, sd, q, st in log)
    t = Z.Trade(tup[:9] + (tup[9] * lots, round(gross, 2), round(fees, 2), round(gross - fees, 2), tup[13]))
    t.l1 = (tup[9], tup[10], tup[11], tup[12])
    t.rule = rule
    det = dict(det, lots=lots, rule=rule, gross_1lot=det["gross"], fees_1lot=det["fees"], net_1lot=det["net"],
               gross=round(gross, 2), fees=round(fees, 2), net=round(gross - fees, 2))
    return t, det


def install() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import options_replay as OR
    from . import nsedata as N
    if "close" not in _O:
        _O["close"], _O["leg_cost"] = N._close_position, OR.leg_cost
    N._close_position, OR.leg_cost = _close, _leg_cost


def uninstall() -> None:
    import options_replay as OR
    from . import nsedata as N
    if "close" in _O:
        N._close_position, OR.leg_cost = _O.pop("close"), _O.pop("leg_cost")


def nse_specs() -> list:
    from . import kite_options as KO
    from . import options_r4 as R4
    return list(R4.SPECS) + list(KO.SPECS)


def nse_rows(specs, lo: int, hi: int) -> dict:
    """{id: rows as replay_rows} from the modules' own simulator on [lo, hi] at the new lots."""
    from . import audit_run as AR
    from . import nsedata as N
    cal = N.lab_calendar(AR.BARS3Y)
    out = {}
    install()
    try:
        for sp in specs:
            details: dict = {}
            res = N.make_simulator(cal, audit=False, details=details)([sp], lo, hi).get(sp.id, {})
            if res.get("error"):
                raise RuntimeError(f"{sp.id}: {res['error']}")
            rows = []
            for t in res.get("trades", []):
                rows.append({"date": int(t[2]), "exit": int(t[6]), "net1": float(t.l1[3]), "gross1": float(t.l1[1]),
                             "fees1": float(t.l1[2]), "lots0": 1, "lots": int(t[9] // t.l1[0]),
                             "rule": getattr(t, "rule", ""), "gross": float(t[10]), "fees": float(t[11]),
                             "net": float(t[12])})
            dets = details.get(sp.id) or []
            for r, d in zip(rows, dets if len(dets) == len(rows) else []):
                r["rule"] = d.get("rule", "")
            out[sp.id] = rows
            Z.log(f"  options nse {sp.id}: {len(rows)} trades {lo}-{hi}")
    finally:
        uninstall()
    return out


# ══════════════════════════════════════════════════════════════════════════
# judgement (protocol.screen's gates for a one-instrument positional strategy)
# ══════════════════════════════════════════════════════════════════════════
def _part(rows, rng, key="net", g="gross", c="fees"):
    sel = [r for r in rows if rng[0] <= r["date"] <= rng[1]]
    nets = [r[key] for r in sel]
    gp, gl = sum(x for x in nets if x > 0), -sum(x for x in nets if x < 0)
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    return {"trades": len(sel), "net": round(sum(nets), 2), "pf": round(pf, 3) if pf != float("inf") else pf,
            "gross": round(sum(r[g] for r in sel), 2), "cost": round(sum(r[c] for r in sel), 2)}


def judge(rows, one_lot: bool = False) -> dict:
    from . import audit_run as AR
    k = ("net1", "gross1", "fees1") if one_lot else ("net", "gross", "fees")
    h1, h2, d = _part(rows, AR.H1, *k), _part(rows, AR.H2, *k), _part(rows, AR.DISC, *k)
    why = []
    if h1["net"] <= 0:
        why.append("H1 net <= 0")
    if h2["net"] <= 0:
        why.append("H2 net <= 0")
    if d["pf"] < MIN_PF:
        why.append(f"PF {d['pf']:.3f} < {MIN_PF}")
    if d["trades"] < MIN_TRADES:
        why.append(f"{d['trades']} trades < {MIN_TRADES}")
    return {"passed": not why, "reasons": why, "h1": h1, "h2": h2, "disc": d, "final": _part(rows, AR.FINAL, *k)}


def _xnet(rows, rng) -> dict:
    x: dict = {}
    for r in rows:
        if rng[0] <= r["date"] <= rng[1]:
            x[str(r["exit"])] = round(x.get(str(r["exit"]), 0.0) + r["net"], 2)
    return x


def replay_files() -> dict:
    out = {}
    for rel, only in REPLAY:
        d = os.path.join(SCRATCH, rel, "trades")
        for f in sorted(os.listdir(d)):
            i = f[:-4]
            if f.endswith(".csv") and (only is None or i in only):
                out[i] = os.path.join(d, f)
    return out


def option_ids() -> dict:
    """{id: engine} of every re-tested options strategy (trials)."""
    ids = {i: "replay" for i in replay_files()}
    ids.update({sp.id: "nse" for sp in nse_specs()})
    return ids


def load_prereg() -> dict:
    a, b = open(PREREG2, "rb").read(), open(REPO_PREREG2, "rb").read()
    if a != b:
        raise SystemExit("the two amendment copies differ")
    return json.loads(a)


def preregister() -> dict:
    if os.path.exists(PREREG2) or os.path.exists(REPO_PREREG2):
        raise SystemExit("amendment already written: refusing to overwrite")
    for f in ("options_screen.json", "options_final.json", "dsr.json"):
        if os.path.exists(os.path.join(OUT, f)):
            raise SystemExit(f"{f} exists: a result was computed before the amendment")
    v1 = Z.load_prereg()
    ids = option_ids()
    scr = Z._jload(os.path.join(OUT, "screen.json"))
    ctl = Z._jload(os.path.join(OUT, "controls.json"))
    here = os.path.dirname(os.path.abspath(__file__))
    k1 = v1["scope"]["retested_K"]
    doc = {
        "title": "Amendment 1: options strategies are re-sized and re-tested as well",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "supersedes": {"file": "PREREGISTERED_SIZE10L.json", "sha256": Z._sha(Z.PREREG),
                       "what_stays": "everything in it (the K = 396 stock strategies, sizes, gates, windows, controls, "
                                     "one FINAL look, old-era rule, proven definition) except N, K and V below"},
        "why": "User, after the first plan was frozen: 'not only intraday, swing - test all'. The flat Rs 20 per order "
               "per leg is a large share of a one-lot option trade, so option strategies change with the account size.",
        "results_seen_at_writing": {
            "options_at_new_size": "none (no option trade was re-sized or re-run before this file)",
            "stock_part_of_the_first_plan": {
                "controls": None if not ctl else {"random_passed": ctl["random_passed"], "planted_passed": ctl["planted_passed"]},
                "discovery_screen_done": bool(scr),
                "discovery_passers_at_new_size": None if not scr else sum(1 for r in scr.values() if r["passed"]),
                "final_looks_opened": os.path.exists(os.path.join(OUT, "final.json"))}},
        "options": {
            "ids": ids, "count": len(ids),
            "report_only": {i: "a live rule: re-sized for the report, verdict not re-opened, not a trial" for i in REPORT_ONLY},
            "not_included": "option strategies without a stored per-leg trade file or a callable EOD simulator "
                            "(VRP-WINGED-HIGHVIX, old-era option runs): their earlier verdicts stand",
            "lots": {"long_or_debit": f"max(1, floor({BUDGET_PREMIUM:.0f} / net premium paid per lot))",
                     "defined_risk_short": f"max(1, floor({BUDGET_RISK:.0f} / max risk per lot)); max risk = the worst "
                                           "expiry payoff of the structure at its entry prices (all legs one expiry, "
                                           "no uncovered short call or put)",
                     "naked_short": f"max(1, floor({BUDGET_MARGIN:.0f} / margin per lot)); margin = the trade record's "
                                    "own margin estimate (options_replay) where present, else "
                                    f"{MARGIN_PCT:.0%} x mean short strike x lot size; a credit structure with an "
                                    "uncovered short or legs in different expiries is sized by this rule",
                     "decided": "at entry, from entry prices only; one lot count per trade for all its legs",
                     "caveat": "exchange freeze quantities (orders above them are split, each paying Rs 20) are not "
                               "modelled; cheap long structures can reach hundreds of lots"},
            "fees": "per leg at the real lot count: options_replay.leg_cost (costs.round_trip_cost 'option'); fill "
                    "haircut per unit unchanged",
            "engines": {"replay": "stored per-leg trade files of options_vrp / options_r3 re-sized exactly (gross x lot "
                                  "ratio; fees = Rs 23.60 per order with a positive value + the proportional part x the "
                                  "lot ratio); the stored files reproduce the stored results at the original lots",
                        "nse": "options_r4 / kite_options through nsedata.make_simulator with nsedata._close_position "
                               "wrapped; the wrapper's one-lot twin must equal the stored DISCOVERY result"},
            "discovery": "the lab's screen for a one-instrument positional strategy on 2023-11-10 .. 2025-12-16: net > 0 "
                         f"in H1 and H2, PF >= {MIN_PF}, >= {MIN_TRADES} trades",
            "final": "one look per DISCOVERY passer at the new size (options_final_ledger.json, first write wins); "
                     "CONFIRMED = net > 0 and PF >= 1.05. Replay strategies: the stored FINAL trades when the file has "
                     "them (FINAL already opened at one lot), otherwise NOT AVAILABLE here (the model engine is not "
                     "re-run) and the strategy cannot be proven; nse strategies: the simulator on the FINAL window",
            "old_era": "no old-era option trade record carries per-leg prices, so an OOS judgement at the new size is "
                       "NOT AVAILABLE; an option strategy therefore cannot be PROVEN by this re-test. Its one-lot "
                       "old-era verdict is reported next to its FINAL result. (Known erratum: old-era EOD option "
                       "settlement prices are 0 on weekly expiries before 2020-01.)"},
        "N_K_V": {"K": k1 + len(ids), "K_stock": k1, "K_options": len(ids),
                  "N": f"{Z.N_BEFORE} + K ({k1 + len(ids)}) + every FINAL look opened by this re-test (stock + options)",
                  "V": "max(robust (1.4826 x MAD)^2 of the DISCOVERY daily Sharpes of all K re-tested strategies with "
                       ">= 1 trade at the new size, 1 / (T_days - 1))",
                  "dsr_series": "daily net P&L of DISCOVERY + FINAL at the new size, booked on the exit session"},
        "report": "every type is covered: intraday (MIS), stock futures, options re-tested; delivery swing and "
                  "index futures (identical at Rs 10 lakh: proportional fees / one real lot) and tfgrid (already Rs 10 "
                  "lakh accounts) keep their existing results, restated",
        "code_sha256": {f: Z._sha(os.path.join(here, f)) for f in ("size10l_options.py", "size10l_run.py", "nsedata.py",
                                                                 "options_r4.py", "kite_options.py", "options_vrp.py",
                                                                 "options_r3.py")},
    }
    blob = json.dumps(doc, indent=1, sort_keys=True).encode()
    for p in (PREREG2, REPO_PREREG2):
        with open(p, "wb") as fh:
            fh.write(blob)
    with open(PREREG2 + ".sha256", "w") as fh:
        fh.write(hashlib.sha256(blob).hexdigest() + "\n")
    Z.log(f"amendment 1 written: options {len(ids)}, K {k1 + len(ids)}, sha256 {hashlib.sha256(blob).hexdigest()}")
    return doc


def _stored_results() -> dict:
    out = {}
    for rel in ("dd/vrp/final_run", "r3/options/full", "r4/realopt/full", "kite/opts3y"):
        p = os.path.join(SCRATCH, rel, "results.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                out[r["id"]] = r
    return out


def stage_screen():
    from . import audit_run as AR
    pre = load_prereg()
    ids = pre["options"]["ids"]
    files = replay_files()
    rows = {i: replay_rows(files[i]) for i, e in ids.items() if e == "replay"}
    for i in REPORT_ONLY:
        rows[i] = replay_rows(os.path.join(SCRATCH, "dd/vrp/final_run/trades", i + ".csv"))
    rows.update(nse_rows([sp for sp in nse_specs() if sp.id in ids], AR.DISC[0], AR.DISC[1]))
    stored = _stored_results()
    res = {}
    for i, rr in rows.items():
        j, j1 = judge(rr), judge(rr, one_lot=True)
        s = stored.get(i, {})
        ok = bool(s) and abs(float(s["disc_net"]) - j1["disc"]["net"]) < 0.5 and int(s["disc_trades"]) == j1["disc"]["trades"]
        d = [r for r in rr if AR.DISC[0] <= r["date"] <= AR.DISC[1]]
        rules: dict = {}
        for r in d:
            rules[r["rule"]] = rules.get(r["rule"], 0) + 1
        lots = sorted(r["lots"] / r["lots0"] for r in d)
        res[i] = {"kind": "options", "engine": ids.get(i, "replay"), "report_only": i in REPORT_ONLY, **j,
                  "stored_1L": {k: j1[k] for k in ("passed", "reasons", "h1", "h2", "disc")},
                  "stored_verdict": s.get("verdict"), "l1_matches_stored": ok, "rules": rules,
                  "lot_ratio_median": lots[len(lots) // 2] if lots else None, "lot_ratio_max": lots[-1] if lots else None,
                  "lots_median": sorted(r["lots"] for r in d)[len(d) // 2] if d else None,
                  "xnet": _xnet(rr, AR.DISC), "has_final_rows": any(r["date"] >= AR.FINAL[0] for r in rr)}
    Z._jdump(os.path.join(OUT, "options_screen.json"), res)
    Z._jdump(os.path.join(OUT, "options_rows_replay.json.gz"), {i: rows[i] for i in rows if ids.get(i, "replay") == "replay"})
    tr = [i for i in res if not res[i]["report_only"]]
    Z.log(f"options screen at the new lots: {sum(res[i]['passed'] for i in tr)}/{len(tr)} pass (one lot: "
          f"{sum(res[i]['stored_1L']['passed'] for i in tr)}); twin differs from stored for "
          f"{[i for i in tr if not res[i]['l1_matches_stored']]}")


def stage_final():
    from . import audit_run as AR
    pre = load_prereg()
    ids = pre["options"]["ids"]
    scr = Z._jload(os.path.join(OUT, "options_screen.json"))
    passers = sorted(i for i, r in scr.items() if r["passed"] and not r["report_only"])
    led = Z._jload(FINAL_LEDGER, {}) or {}
    todo = [i for i in passers if i not in led]
    for i in todo:
        led[i] = {"opened": time.strftime("%Y-%m-%d %H:%M"), "final": None}
    Z._jdump(FINAL_LEDGER, led)
    rep = Z._jload(os.path.join(OUT, "options_rows_replay.json.gz"), {})
    nse = nse_rows([sp for sp in nse_specs() if sp.id in todo and ids[sp.id] == "nse"], AR.FINAL[0], AR.FINAL[1]) \
        if any(ids[i] == "nse" for i in todo) else {}
    for i in todo:
        rr = nse.get(i) if ids[i] == "nse" else [r for r in rep[i] if r["date"] >= AR.FINAL[0]]
        if ids[i] == "replay" and not rr:
            led[i]["final"] = {"available": False, "confirmed": False,
                               "note": "FINAL trades are not stored (never opened at one lot): not available"}
            continue
        f, f1 = _part(rr, AR.FINAL), _part(rr, AR.FINAL, "net1", "gross1", "fees1")
        led[i]["final"] = {"available": True, **f, "confirmed": bool(f["net"] > 0 and f["pf"] >= 1.05)}
        led[i]["l1_final"] = f1
        led[i]["xnet"] = _xnet(rr, AR.FINAL)
    Z._jdump(FINAL_LEDGER, led)
    # report-only: the live winged strangle's FINAL at the new lots
    ro = {}
    for i in REPORT_ONLY:
        rr = replay_rows(os.path.join(SCRATCH, "dd/vrp/final_run/trades", i + ".csv"))
        ro[i] = {"final": _part(rr, AR.FINAL), "l1_final": _part(rr, AR.FINAL, "net1", "gross1", "fees1"),
                 "xnet": _xnet(rr, AR.FINAL)}
    Z._jdump(os.path.join(OUT, "options_final.json"), {"passers": passers, "results": {i: led[i] for i in passers},
                                                       "report_only": ro})
    Z.log(f"options final: {[(i, led[i]['final'].get('confirmed'), led[i]['final'].get('net'), led[i]['final'].get('pf')) for i in passers]}")


def stage_dsr():
    """ONE DSR for the whole re-test (stock + options): N = N_before + K + all FINAL looks; V over all K."""
    from . import audit_run as AR
    from . import datagen_run as DR
    from . import trials as T
    pre = load_prereg()
    scr = dict(Z._jload(os.path.join(OUT, "screen.json")))
    osc = {i: r for i, r in Z._jload(os.path.join(OUT, "options_screen.json")).items() if not r["report_only"]}
    fin = dict((Z._jload(os.path.join(OUT, "final.json"), {}) or {}).get("results", {}))
    ofin = (Z._jload(os.path.join(OUT, "options_final.json"), {}) or {}).get("results", {})
    allscr = {**scr, **osc}
    allfin = {**fin, **ofin}
    cal = T.calendar_3y(SCRATCH)
    K = pre["N_K_V"]["K"]
    assert K == len(allscr), (K, len(allscr))
    srs = {i: Z.sr_stats(Z.daily_series(r["xnet"], cal, AR.DISC[0], AR.DISC[1])) for i, r in allscr.items()
           if r["disc"]["trades"] > 0}
    t_days = len([d for d in cal if AR.DISC[0] <= d <= AR.DISC[1]])
    v_mad = DR.robust_var([s["sr_daily"] for s in srs.values()])
    v = max(v_mad, 1.0 / (t_days - 1))
    rows = T.TrialsLedger(os.path.join(SCRATCH, "stratlab_cache", "trials_v2.csv")).read()
    v_led = T.var_sr_robust([r for r in rows if not r["strategy_id"].startswith(Z.ID_PREFIX)])[0]
    n = Z.N_BEFORE + K + len(allfin)
    out = {"N_before": Z.N_BEFORE, "K": K, "final_looks": len(allfin), "N": n, "V": v, "V_mad": v_mad,
           "V_floor": 1.0 / (t_days - 1), "V_ledger_1lakh_report_only": v_led, "sr0_daily": T.expected_max_sr(n, v),
           "sr0_annual": T.expected_max_sr(n, v) * 252 ** 0.5, "rows": {}}
    for i, f in allfin.items():
        x = {int(k): v_ for k, v_ in allscr[i]["xnet"].items()}
        for k, v_ in (f.get("xnet") or {}).items():
            x[int(k)] = x.get(int(k), 0.0) + v_
        st = Z.sr_stats(Z.daily_series(x, cal, AR.DISC[0], AR.FINAL[1]))
        d = T.dsr(st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"], n, v)
        d2 = T.dsr(st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"], n, v_led)
        out["rows"][i] = {**st, "sr_annual": st["sr_daily"] * 252 ** 0.5, "dsr": d["dsr"], "psr0": d["psr0"],
                          "dsr_ledger_v_report_only": d2["dsr"], "disc_sr_daily": srs.get(i, {}).get("sr_daily")}
    Z._jdump(os.path.join(OUT, "dsr.json"), out)
    Z.log(f"dsr (stock + options): N={n} V={v:.6f} luck bar {out['sr0_annual']:.2f} a year; "
          f"{[(i, round(r['dsr'], 4)) for i, r in out['rows'].items()]}")


def stage_ledger():
    from . import audit_run as AR
    from . import trials as T
    load_prereg()
    path = os.path.join(SCRATCH, "stratlab_cache", "trials_v2.csv")
    cal = T.calendar_3y(SCRATCH)
    scr = Z._jload(os.path.join(OUT, "options_screen.json"))
    fin = (Z._jload(os.path.join(OUT, "options_final.json"), {}) or {}).get("results", {})
    n0 = T.count_trials(T.TrialsLedger(path).read(), "2023-26")
    new = []
    for i, r in scr.items():
        if r["report_only"]:
            continue
        d = r["disc"]
        st = Z.sr_stats(Z.daily_series(r["xnet"], cal, AR.DISC[0], AR.DISC[1])) if d["trades"] else {}
        new.append({"strategy_id": Z.ID_PREFIX + i, "family": T.family_of(i), "group": "options", "params_fp": "",
                    "era": "2023-26", "protocol": "2023-26, Rs 10 lakh-per-agent option lots (size10l amendment 1)",
                    "stage": "size10l-screen", "dataset": "bars3y", "period": f"{AR.DISC[0]}-{AR.DISC[1]}",
                    "cost_model": "2026-11/options", "trades": d["trades"], "net": d["net"], "pf": d["pf"],
                    "mean_gross": d["gross"] / d["trades"] if d["trades"] else None,
                    "mean_cost": d["cost"] / d["trades"] if d["trades"] else None, "n_days": st.get("n_days"),
                    "sr_daily": st.get("sr_daily"), "skew_daily": st.get("skew_daily"), "kurt_daily": st.get("kurt_daily"),
                    "verdict": "PASSED SCREEN" if r["passed"] else "FAILED SCREEN",
                    "source": os.path.join(OUT, "options_screen.json"),
                    "note": "re-test at the Rs 10 lakh lot rule: a NEW trial; " + ", ".join(r["reasons"])})
    for i, v in fin.items():
        f = v["final"]
        new.append({"strategy_id": Z.ID_PREFIX + i, "family": T.family_of(i), "group": "options", "params_fp": "",
                    "era": "2023-26", "protocol": "2023-26, Rs 10 lakh-per-agent option lots (size10l amendment 1)",
                    "stage": "size10l-final", "dataset": "bars3y", "period": f"{AR.FINAL[0]}-{AR.FINAL[1]}",
                    "cost_model": "2026-11/options", "trades": f.get("trades"), "net": f.get("net"), "pf": f.get("pf"),
                    "verdict": ("CONFIRMED" if f.get("confirmed") else "REJECTED ON FINAL") if f.get("available")
                    else "FINAL NOT AVAILABLE", "source": FINAL_LEDGER,
                    "note": "second look at this FINAL window (first at one lot); counted in N"})
    n_app = T.TrialsLedger(path).append(new)
    n1 = T.count_trials(T.TrialsLedger(path).read(), "2023-26")
    Z._jdump(os.path.join(OUT, "options_ledger.json"), {"appended": n_app, "ledger_n_2023_26_before": n0,
                                                        "ledger_n_2023_26_after": n1})
    Z.log(f"options ledger: {n_app} rows appended; ledger N 2023-26 {n0} -> {n1}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.size10l_options")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--stage", default="")
    a = ap.parse_args(argv)
    try:
        os.nice(10)
    except Exception:
        pass
    if a.preregister:
        preregister()
        print(f"written {PREREG2} (sha256 {Z._sha(PREREG2)})")
        return 0
    st = {"screen": stage_screen, "final": stage_final, "dsr": stage_dsr, "ledger": stage_ledger}
    if a.stage not in st:
        raise SystemExit(f"unknown stage {a.stage!r}")
    st[a.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
