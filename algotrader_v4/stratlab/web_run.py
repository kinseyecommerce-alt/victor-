"""
stratlab/web_run.py — run the pre-registered WEB study (web_strategies.py, prereg/web_v1.json) ONCE and judge it.
Research only; writes outside the repository (<scratchpad>/webstrats).

    cd algotrader_v4
    python -m stratlab.web_run --data-check            # availability counts only (no price, no return)
    python -m stratlab.web_run --preregister           # freeze ids, fingerprints, gates, data hashes (once)
    python -m stratlab.web_run --smoke                 # plumbing: errors only, no P&L, no trade counts
    python -m stratlab.web_run --run --trials <ledger> # the one run: both eras, controls, accounts, trials ledger

Judgement (the lab's functions, unchanged):
  2023-26  protocol.evaluate (DISCOVERY 70 % = H1 + H2, FINAL 30 % once and only for screen passers). Daily specs on
           the events_daily point-in-time cache (one-look ledger <events_daily cache>/final_ledger.json, calendar
           trials.calendar_3y); option specs through nsedata's real-price EOD simulator (ledger <bars3y cache>/
           final_ledger.json, calendar nsedata.lab_calendar, cost model nsedata.cost_model()).
  2012-23  protocol_oos.evaluate (H1 2012-17 / H2 2018-23, >= 60 trades in EACH half, one look, ledger
           <scratch>/webstrats/oos_ledger.json); option specs on the old-era nsedata view (oldera_fo). EVERY variant
           gets the old-era look. No variant reads minute bars: the 2016-23 minute era and the corrected-data overlay
           (corrected.py, minute bars) do not apply.
  proven   = CONFIRMED on the 2023-26 FINAL AND OOS-CONFIRMED on 2012-23 AND DSR >= 0.95 at N = every 2023-26 trial
           in the ledger (fair N = the 4 variants also reported) AND, for the two long-only stock variants, the
           month-clustered 95 % CI of the mean excess return per trade over their benchmark (NIFTY 100 / NIFTY 500)
           above 0 in BOTH eras.
Report-only: 5 bps slippage (options: haircut x2.5), the corporate-action rule both ways, excess vs NIFTY 50 / Next 50
/ 100 / 500, placebo (same count, random dates or random names) and planted-edge controls, the Rs 10 lakh account.
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import numpy as np

from . import data as D
from . import data_daily as DD
from . import events_daily as ED
from . import events_data as E
from . import events_run as ER
from . import flowdata as F
from . import nsedata as N
from . import protocol as P
from . import protocol_oos as PO
from . import sim
from . import web_strategies as W

SCRATCH = DD._SCRATCH
WEB_DIR = os.environ.get("STRATLAB_WEB_OUT", os.path.join(SCRATCH, "webstrats"))
OUT = os.path.join(WEB_DIR, "run1")
PREREG = os.path.join(WEB_DIR, "PREREGISTERED_WEB.json")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "web_v1.json")
OOS_LEDGER = os.path.join(WEB_DIR, "oos_ledger.json")
DATA_CHECK = os.path.join(WEB_DIR, "data_check.json")
SHORTLIST = os.path.join(WEB_DIR, "SHORTLIST.md")
BARS3Y = os.path.join(SCRATCH, "bars3y")
FILL_MODEL = "web1"
COST_MODEL = f"{sim.COST_MODEL}/{FILL_MODEL}"
SLIP_BPS, SENS_BPS = sim.DEFAULT_SLIPPAGE_BPS, sim.SENSITIVITY_SLIPPAGE_BPS
DSR_MIN = 0.95
SEED = 20261115
N_VARIANTS = 4
_NET, _SYM, _DATE = 12, 1, 2
_CTX: dict = {}


# ══════════════════════════════════════════════════════════════════════════
# set-up
# ══════════════════════════════════════════════════════════════════════════
def setup(lv: bool = False, flow_dir: str | None = None, events_dir: str | None = None) -> dict:
    """events_run's set-up (daily cache, calendar, exclusions, F&O list, event tables) + corporate actions and index
    membership; DIV_CREDIT from the REAL corporate-action table; lv=True also builds the low-volatility selection
    (reads prices). Must run before forking workers."""
    ctx = ER.setup(events_dir=events_dir)
    _CTX.update(ctx)
    if "real_flow_dir" not in _CTX:
        _CTX["real_flow_dir"] = F.FLOW_DIR
    if not W.DIV_CREDIT:
        F.use(_CTX["real_flow_dir"])
        W.DIV_CREDIT.update(W.build_div_credit(_CTX["symbols"]))
    if flow_dir is not None:
        F.use(flow_dir)
    F._load_ca()
    F._load_index()
    if lv:
        W.LV_SELECTION.clear()
        W.LV_SELECTION.update(lv_selection())
    return _CTX


def _series(sym: str):
    if "series" not in _CTX:
        _CTX["series"] = {}
    if sym not in _CTX["series"]:
        sd = DD.load_symbol(_CTX["cache"], sym)
        if sd is None:
            _CTX["series"][sym] = None
        else:
            d = sd.daily()
            _CTX["series"][sym] = (np.asarray(d["date"], np.int64), np.asarray(d["c"], float),
                                   np.asarray(d["o"], float))
    return _CTX["series"][sym]


def lv_eligible(cutoff: int) -> set:
    syms = set(_CTX["symbols"])
    mem = F.index_members("NIFTY100", int(cutoff) + 1)
    return {s for s in mem if s in syms and E.fo_listed(s, int(cutoff))}


def lv_selection(cheat=None, calendar=None) -> dict:
    ser = lambda s: (lambda x: None if x is None else (x[0], x[1]))(_series(s))
    return W.build_lv_selection(calendar or _CTX["cal"], lv_eligible, ser, excluded=E.ca_excluded, cheat=cheat)


def option_underlyings(nse_dir: str | None = None) -> list:
    cdir = os.path.join(nse_dir or N.NSE_DIR, "_cache", "optstk")
    return sorted(f[:-4] for f in os.listdir(cdir) if f.endswith(".npz"))


# ══════════════════════════════════════════════════════════════════════════
# simulators
# ══════════════════════════════════════════════════════════════════════════
class _Guard:
    """fn(d, p) inside nsedata.cutoff(decision day + 1): flowdata accessors (corporate actions, index membership)
    see only rows known by the evening of the decision day."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "guarded")

    def __call__(self, d, p):
        with N.cutoff(int(d.date[-1]) + 1):
            return self.fn(d, p)


def guarded(spec, sd):
    g = DD.gate(spec, sd)
    return dataclasses.replace(g, fn=_Guard(g.fn))


def _work_daily(args):
    symbol, ids, lo, hi, slip, keep_ca, credit = args
    out = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None, "ca_dropped": 0, "div": 0.0}
           for i in ids}
    sd = DD.load_symbol(_CTX["cache"], symbol)
    if sd is None:
        return out
    for i in ids:
        try:
            tr, sk = sim.sim_positional(guarded(W.BY_ID[i], sd), sd, lo, hi, _CTX["nifty"], _CTX["vix"], slip)
            if not keep_ca:
                tr, n = ER.drop_excluded(tr)
                out[i]["ca_dropped"] = n
            if credit:
                tr, dv = W.credit_dividends(tr)
                out[i]["div"] = dv
            out[i]["trades"], out[i]["skipped"] = tr, sk
        except Exception as exc:
            out[i]["error"] = f"{symbol}: {type(exc).__name__}: {str(exc)[:160]}"
    return out


def make_daily_simulator(pool, slip_bps: float = SLIP_BPS, keep_ca: bool = False, credit: bool = True,
                         stats: dict | None = None, symbols=None):
    def simulate(specs, lo, hi):
        ids = [sp.id for sp in specs]
        merged = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None} for i in ids}
        jobs = [(s, ids, lo, hi, slip_bps / 1e4, keep_ca, credit) for s in (symbols or _CTX["symbols"])]
        for res in (pool.imap_unordered(_work_daily, jobs, chunksize=8) if pool is not None
                    else map(_work_daily, jobs)):
            for i, r in res.items():
                m = merged[i]
                m["trades"].extend(r["trades"])
                m["skipped"] += r["skipped"]
                m["error"] = m["error"] or r["error"]
                if stats is not None:
                    st = stats.setdefault(i, {"ca_dropped": 0, "div": 0.0})
                    st["ca_dropped"] += r["ca_dropped"]
                    st["div"] += r["div"]
        for m in merged.values():
            m["trades"].sort(key=lambda t: (t[2], t[1]))
        return merged
    return simulate


def _work_opt(args):
    sym, ids, cal, lo, hi, mult, audit = args
    specs = [sp for sp in _CTX["opt_specs"] if sp.id in ids]
    out = {}
    N._load()
    N._S["chains"] = {}                    # one symbol's chains in memory (keeps the masked audit cheap)
    N._S["daily_fut"] = {}
    for sp in specs:
        try:
            t, det, s = N._simulate_one(sp, sym, cal, lo, hi, mult)
            la = N.audit_lookahead(sp, sym, cal, t, n=2) if audit else None
            out[sp.id] = {"trades": t, "details": det, "skipped": s, "lookahead": la, "error": None}
        except Exception as exc:
            out[sp.id] = {"trades": [], "details": [], "skipped": 0, "lookahead": None,
                          "error": f"{sym}: {type(exc).__name__}: {str(exc)[:160]}"}
    return out


def make_opt_simulator(pool, cal: list, mult: float = 1.0, audit: bool = True, details: dict | None = None,
                       symbols=None):
    def simulate(specs, lo, hi):
        ids = [sp.id for sp in specs]
        merged = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None} for i in ids}
        jobs = [(s, ids, cal, lo, hi, mult, audit) for s in (symbols or _CTX["opt_syms"])]
        for res in (pool.imap_unordered(_work_opt, jobs, chunksize=4) if pool is not None else map(_work_opt, jobs)):
            for i, r in res.items():
                m = merged[i]
                m["trades"].extend(r["trades"])
                m["skipped"] += r["skipped"]
                m["lookahead"] = m["lookahead"] or r["lookahead"]
                m["error"] = m["error"] or r["error"]
                if details is not None:
                    for x in r["details"]:
                        details.setdefault(i, []).append(x)
        for m in merged.values():
            m["trades"].sort(key=lambda t: (t[2], t[1]))
        return merged
    return simulate


def _pool(jobs: int):
    E.set_calendar(_CTX["cal"])
    return mp.get_context("fork").Pool(jobs)


# ══════════════════════════════════════════════════════════════════════════
# report-only statistics
# ══════════════════════════════════════════════════════════════════════════
def abnormal_returns(trades: list, bench: dict, with_div: bool) -> list:
    """[(entry yyyymm, excess, raw)] per trade: (exit / fill - 1) [+ dividends received / fill] minus the benchmark
    over the same window (open of the entry session -> close of the exit session)."""
    nd, no, nc = bench["date"], bench["o"], bench["c"]
    out = []
    for t in trades:
        e, x = int(t[_DATE]), int(t[6]) if t[6] else int(t[_DATE])
        i, j = int(np.searchsorted(nd, e)), int(np.searchsorted(nd, x))
        if i >= len(nd) or j >= len(nd) or nd[i] != e or nd[j] != x or not no[i] > 0:
            continue
        r = float(t[8]) / float(t[5]) - 1.0
        if with_div:
            r += W.dividend_credit(t[_SYM], e, x) / float(t[5])
        out.append((e // 100, r - (float(nc[j]) / float(no[i]) - 1.0), r))
    return out


def excess(trades: list, bench: dict, ranges: dict, with_div: bool) -> dict:
    return {k: {b: ER.block_bootstrap(abnormal_returns([t for t in trades if lo <= t[_DATE] <= hi], s, with_div))
                for b, s in bench.items()} for k, (lo, hi) in ranges.items()}


def load_benchmarks() -> dict:
    from . import flows_run as FR
    if not os.path.exists(FR.BENCH_FILE):
        raise SystemExit(f"benchmark file missing: {FR.BENCH_FILE}")
    with gzip.open(FR.BENCH_FILE, "rt") as f:
        raw = json.load(f)
    out = {ix: {"date": np.array(v["date"], np.int64), "o": np.array(v["o"]), "c": np.array(v["c"])}
           for ix, v in raw.items()}
    nf = _CTX["nifty"]
    return {"NIFTY50": {"date": np.asarray(nf["date"], np.int64), "o": np.asarray(nf["o"]),
                        "c": np.asarray(nf["c"])}, **out}


def account_daily(trades: list, sp, cal: list, lo: int, hi: int) -> dict:
    """capital.py swing agent (Rs 10 lakh, config.py slots, 1 % risk); no-stop variants are sized as if the stop were
    20 % away (Rs 50,000 a position), stated in the pre-registration."""
    q = dict(sp.params)
    q["stop_pct"] = 20.0
    return ER.account(trades, dataclasses.replace(sp, params=q), cal, lo, hi)


def account_options(details: list, cal: list, lo: int, hi: int) -> dict:
    """capital.py options agent (Rs 10 lakh, config.py fno slots): each straddle takes its premium as margin."""
    from . import capital as C
    ag = C.agents()["options"]
    cands = []
    for x in details:
        e, xd = int(x["date"]), int(x["exit_date"])
        if not lo <= e <= hi:
            continue
        q = int(x["lot"])
        prem = abs(float(x["entry_val"])) * q
        cands.append(C.cand(x["symbol"], 1, e * 10000 + 930, abs(float(x["entry_val"])), None, xd * 10000 + 931,
                            abs(float(x["exit_val"])), x["reason"], 0.0, "opt", 1, 1, "web_ea",
                            opt=(q, float(x["gross"]), float(x["fees"]), prem)))
    res = C.simulate(cands, ag, "web_ea")
    mt = C.metrics(res, cal, lo, hi)
    return {k: mt[k] for k in ("trades", "candidates", "net", "return_pct_per_year", "max_dd_rs", "max_dd_pct",
                               "worst_month", "worst_month_rs", "win_pct", "pf", "sharpe", "skipped")}


# ══════════════════════════════════════════════════════════════════════════
# data check (counts only)
# ══════════════════════════════════════════════════════════════════════════
def _period(day: int, split=None) -> str:
    if day <= PO.H1[1]:
        return "2012-17"
    if day <= PO.ERA[1]:
        return "2018-23"
    return "2023-26"


def data_check(path: str = DATA_CHECK, write: bool = True) -> dict:
    """Availability counts: predicted ex-dividend month events on top-500 member decision days; NIFTY 100 / F&O /
    1-year-history eligibility per low-volatility cut-off; results meetings of option underlyings and how many have
    an option chain on the entry day (sampled). No price level, return or volatility is computed."""
    setup()
    cal = _CTX["cal"]
    out: dict = {"written": time.strftime("%Y-%m-%d %H:%M"), "calendar": [cal[0], cal[-1], len(cal)],
                 "lab_calendar_3y": (lambda c: [c[0], c[-1], len(c)])(ER.lab_calendar())}
    # B. dividend month
    month_ends = [d for i, d in enumerate(cal[:-1]) if cal[i + 1] // 100 != d // 100]
    me_set = set(month_ends)
    cnt = collections.Counter()
    divrows = collections.Counter()
    for s in _CTX["symbols"]:
        sd = DD.load_symbol(_CTX["cache"], s)
        if sd is None:
            continue
        mem = [int(x) for x in sd.days[sd.member] if int(x) in me_set]
        for d in mem:
            nxt = E.next_session(d)
            if nxt and W.predicted_ex_month(s, d, nxt // 100):
                cnt[str(nxt // 10000)] += 1
        for ex in W.DIV_CREDIT.get(s, (np.array([]), None))[0]:
            divrows[str(int(ex) // 10000)] += 1
    out["divm_predicted_events_by_year"] = dict(sorted(cnt.items()))
    out["cash_dividend_exdates_by_year_universe_symbols"] = dict(sorted(divrows.items()))
    # A. low volatility eligibility
    lv = []
    for cutoff, first in W.lv_cutoffs(cal):
        el = lv_eligible(cutoff)
        hist = 0
        for s in el:
            sd = DD.load_symbol(_CTX["cache"], s)
            if sd is not None and int(np.searchsorted(sd.days, cutoff, side="right")) >= W.LV_WINDOW + 1:
                hist += 1
        lv.append({"cutoff": cutoff, "first": first, "nifty100_members": len(F.index_members("NIFTY100", cutoff + 1)),
                   "in_cache_and_fo": len(el), "with_1y_history": hist})
    out["lowvol_eligibility"] = lv
    # C/D. results meetings of option underlyings
    res: dict = {}
    for era, nse_dir in (("2023-26", ED.NSE_DIR), ("2012-23", os.path.join(SCRATCH, "nse_old_view"))):
        syms = set(option_underlyings(nse_dir))
        c = collections.Counter()
        for s in syms:
            t = E._load("boardmeet").get(s)
            if t is None:
                continue
            for m, isr in zip(t["meeting"], t["results"]):
                if isr and (int(m) > PO.ERA[1]) == (era == "2023-26") and int(m) >= PO.ERA[0]:
                    c[str(int(m) // 10000)] += 1
        res[era] = {"underlyings": len(syms), "results_meeting_notices_by_year": dict(sorted(c.items()))}
    out["ea_meetings"] = res
    out["note"] = "counts only: no price level, return or volatility was computed to produce this file"
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1, default=ER._js)
    return out


# ══════════════════════════════════════════════════════════════════════════
# pre-registration
# ══════════════════════════════════════════════════════════════════════════
CODE_FILES = ("web_strategies.py", "web_run.py", "events_run.py", "events_data.py", "events_daily.py", "flowdata.py",
              "nsedata.py", "oldera_fo.py", "sim.py", "protocol.py", "protocol_oos.py", "capital.py", "data_daily.py",
              "trials.py")


def code_sha() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: ER._sha(os.path.join(here, f)) for f in CODE_FILES}


def data_sha() -> dict:
    out = {}
    for rel in ("corpactions/corpactions.csv.gz", "index_membership/index_members.csv.gz"):
        p = os.path.join(_CTX.get("real_flow_dir", F.FLOW_DIR), rel)
        out[rel] = ER._sha(p) if os.path.exists(p) else None
    for name in ("boardmeet.csv.gz", "results.csv.gz"):
        out["events/" + name] = ER._sha(os.path.join(ED.EVENTS_DIR, name))
    return out


def fingerprints() -> dict:
    fp = {sp.id: sp.fingerprint for sp in W.DAILY}
    fp.update({i: W.option_fingerprint(i) for i in W.OPTION_IDS})
    return fp


def prereg_doc() -> dict:
    m = DD.manifest(ED.CACHE)
    dc = json.load(open(DATA_CHECK)) if os.path.exists(DATA_CHECK) else {}
    fp = fingerprints()
    variants = [{"id": sp.id, "kind": sp.kind, "horizon": sp.horizon, "family": sp.family, "rule": sp.desc,
                 "params": sp.params, "fingerprint": fp[sp.id]} for sp in W.DAILY]
    variants += [{"id": i, "kind": "options", "horizon": "positional", "family": "web_ea", "rule": d, "params": p,
                  "fingerprint": fp[i]} for i, (d, p) in W.EA_PARAMS.items()]
    return {
        "name": "web_v1", "written": time.strftime("%Y-%m-%d %H:%M"),
        "question": "Do the highest-evidence strategies found in the published literature (chosen on EXTERNAL "
                    "evidence only) give a tradable edge on NSE after full costs, in two eras?",
        "selection": {"shortlist_file": "webstrats/SHORTLIST.md",
                      "shortlist_sha256": ER._sha(SHORTLIST) if os.path.exists(SHORTLIST) else None,
                      "rule": "score >= 5/10 on (India evidence, post-publication survival, cross-country "
                              "replication, economic reason, tradability), feasible with the lab's data, never "
                              "tested by the lab, a published rule, able to reach the trade minimum on availability "
                              "counts; at most 10 variants (4 taken)",
                      "no_data_look": "only availability counts (data_check.json) were computed before this file"},
        "variants": variants,
        "citations": {
            "web_lowvol30_nifty100": [
                "NSE Indices, Methodology Document (Nov 2026) pp. 190-192, NIFTY100 Low Volatility 30; factsheet Oct 2026",
                "NSE whitepaper NIFTY100 Low Volatility 30 (June 2018)",
                "Frazzini & Pedersen (2014) JFE 'Betting against beta'; Blitz & van Vliet (2007) JPM",
                "Joshipura (2020) IMFI; Pandey & Samanta (contrary, CNX 100 2001-14)"],
            "web_divm_pred_exmonth": [
                "Hartzmark & Solomon (2013) JFE 109(3) 640-660 'The dividend month premium'",
                "Hameed, Xie & Zhong (2024) 'Preferences for dividends and stock returns around the world' "
                "(India: predicted ex-month minus other months 1.010 %/month, t = 8.35, 1993-2018)"],
            "web_ea_straddle_m3_0": [
                "Gao, Xing & Zhang (2018) JFQA 53(6) 2587-2617 'Anticipating uncertainty: straddles around earnings "
                "announcements' ([-3,0] 3.34 %)",
                "Saurav, Agarwalla & Varma (2024) American Business Review 27(2) 459-487 (IV rises into results in "
                "Indian stock options)", "Chung & Louis (2017) JEF 40 220-235"],
            "web_ea_straddle_m1_p1": ["Gao, Xing & Zhang (2018/2013 WP): [-1,+1] 3.09 %", "as above"]},
        "strategies": {k: {"fingerprint": v} for k, v in fp.items()},
        "n_variants": N_VARIANTS,
        "universe": {
            "web_lowvol30_nifty100": "point-in-time NIFTY 100 members (flowdata.index_members, membership in force at "
                                     "the cut-off's close) that are in the events_daily cache, trade a stock future on "
                                     "the cut-off day (F&O bhavcopy) and have >= 250 daily returns; positions open only "
                                     "on top-500 member days (data_daily.gate)",
            "web_divm_pred_exmonth": "point-in-time monthly top 500 by turnover (events_daily), delisted names kept; "
                                     "member on the decision day",
            "options": "every stock with per-strike EOD option data in the era (nsedata optstk cache: 2023-26 "
                       "<nse>, 2012-23 <nse_old_view>); renamed symbols mapped to current NSE symbols"},
        "availability_rule": {
            "corporate_actions": "flowdata.corp_actions with before = decision day + 1 (nsedata.cutoff(decision day + 1) "
                                 "clips every flowdata read): only dividends whose ex-date is before the next session",
            "index_membership": "flowdata.index_members(NIFTY100, cut-off + 1): changes in force by the cut-off",
            "lowvol_selection": "built once per cut-off from daily closes dated <= the cut-off (W.lv_volatility slices "
                                "by date; tested on truncated data); the holding quarter starts ~1 month later",
            "board_meetings": "the LATEST results-meeting notice broadcast before 15:00 IST of the entry session "
                              "(events_data cutoff); skipped if results were already published since that notice",
            "option_chain": "strike and expiry chosen from the previous F&O session's EOD chain (nsedata rule); filled "
                            "at the entry session's EOD close only if both legs traded that day",
            "calendar": "exchange session calendar treated as public (day 0 = first session on / after the meeting)"},
        "eras": {"2023-26": {"daily_calendar": "trials.calendar_3y", "option_calendar": "nsedata.lab_calendar(bars3y)",
                             "protocol": "protocol.evaluate (DISCOVERY 70 % H1/H2 + FINAL 30 %, FINAL once for "
                                         "screen passers)",
                             "ledgers": ["<events_daily cache>/final_ledger.json", "<bars3y cache>/final_ledger.json"]},
                 "2012-23": {"era": list(PO.ERA), "h1": list(PO.H1), "h2": list(PO.H2),
                             "protocol": "protocol_oos.evaluate (one look, webstrats/oos_ledger.json)",
                             "options_data": "oldera_fo view (<scratch>/nse_old_view)"},
                 "2016-23_minute": "not applicable: no variant reads minute bars",
                 "corrected_overlay": "not applicable (corrected.py corrects minute bars; the daily cache is NSE "
                                      "bhavcopy, adjusted for splits / bonuses / large demergers)",
                 "old_era_look": "EVERY variant (recommended and adopted)"},
        "gates": {"screen": {"min_pf": P.SCREEN_MIN_PF, "min_trades": P.SCREEN_MIN_TRADES,
                             "max_symbol_share": P.SCREEN_MAX_SYMBOL_SHARE, "net_gt_0": ["H1", "H2"]},
                  "final": {"min_pf": P.FINAL_MIN_PF, "net_gt_0": True}, "oos": PO.GATES, "dsr_min": DSR_MIN,
                  "long_only_excess": {"benchmarks": W.LONG_ONLY_BENCH,
                                       "rule": "for 'proven' only: month-clustered bootstrap 95 % CI lower bound of the "
                                               "mean excess return per trade (dividends included) > 0 in 2023-26 "
                                               "(DISCOVERY + FINAL trades) AND in 2012-23"},
                  "no_relaxation": "nothing is pooled, relaxed or re-defined after results",
                  "proven": "CONFIRMED on 2023-26 FINAL and OOS-CONFIRMED on 2012-23 and DSR >= 0.95 at N = all "
                            "2023-26 trials in the ledger (fair N = 4 also reported) [+ long_only_excess]"},
        "gates_fingerprint": PO.GATES_FINGERPRINT,
        "fills_costs": {
            "daily": {"cost_model": COST_MODEL, "simulator": "sim.sim_positional unchanged: next-open fill, close "
                      "exits, CNC costs (costs.round_trip_cost), slippage 2 bps per side (5 bps report-only)",
                      "notional": sim.NOTIONAL, "stops": "none (published rules have none)",
                      "dividends": "web_divm_pred_exmonth AND web_lowvol30_nifty100: cash dividends going ex while "
                                   "held (entry < ex-date <= exit) credited gross to the trade (prices are not "
                                   "dividend-adjusted); report-only: the net without the credit",
                      "corporate_action_rule": "events_run.drop_excluded (unexplained gaps / scheme ex-days), "
                                               "reported both ways"},
            "options": {"cost_model": N.cost_model(), "simulator": "nsedata OptionSpec simulator: EOD-close fills, "
                        "haircut max(Rs 0.05, 1 %) per leg per fill, options_replay.leg_cost fees (Zerodha 2026), "
                        "1 lot (exchange lot in force), exit at the EOD close of the hold-th session (stale close "
                        "flagged), settlement at intrinsic if a leg expires first",
                        "sensitivity": "haircut x2.5 (report-only, DISCOVERY and old era)"}},
        "report_only": {
            "excess": "mean / median excess per trade vs NIFTY 50, NIFTY Next 50, NIFTY 100, NIFTY 500 with "
                      f"{ER.BOOT_N} month-block bootstrap resamples (long-only stock variants; with and without "
                      "dividends)",
            "placebo": {"web_divm_pred_exmonth": "every cash-dividend row moved to a random session of the same stock "
                                                 "in the same era (same count); real dividends still credited",
                        "web_lowvol30_nifty100": "30 names drawn at random from the same eligible set each quarter",
                        "options": "board-meeting / results / insider tables with every event moved to a random "
                                   "session of the same stock and era (events_run.placebo_tables)",
                        "seed": SEED},
            "planted_edge": {"web_divm_pred_exmonth": "dividend rows planted 12 months before each stock's best 2 % "
                                                      "of calendar months (2012-23) must give a large profit; planted "
                                                      "one month later (buying the month after) must not",
                             "web_lowvol30_nifty100": "selection by the NEXT quarter's return (cheat) must give a large "
                                                      "profit",
                             "options": "results meetings planted on each underlying's top 2 % |3-session move| "
                                        "sessions (2012-23, every 3rd underlying) must give web_ea_straddle_m3_0 a "
                                        "large profit; the same delayed 25 sessions must not"},
            "account": "capital.py Rs 10 lakh: swing agent for the stock variants (sized as a 20 % stop = Rs 50,000 a "
                       "position), options agent for the straddles (premium as margin)"},
        "data": {"daily_cache": {"version": m.get("events_daily"), "sessions": len(m["calendar"]),
                                 "first": m["calendar"][0], "last": m["calendar"][-1], "top_n": m.get("top_n")},
                 "tables_sha256": data_sha(),
                 "data_check_sha256": ER._sha(DATA_CHECK) if os.path.exists(DATA_CHECK) else None,
                 "data_check_summary": {k: dc.get(k) for k in ("divm_predicted_events_by_year", "ea_meetings")}},
        "expected": "small or no edge after costs is the base case for every variant (published gross edges: "
                    "low-vol ~1-3 %/yr excess; dividend month ~1 %/month gross vs ~0.3 % costs; straddles 1-3 % per "
                    "trade gross vs ~2-3 % round-trip haircut + fees)",
        "trials": "each of the 4 variants is one trial on 2023-26 and one on 2012-23 (stage 'oos'); controls are not "
                  "trials",
        "code_sha256": code_sha(),
    }


def preregister(path: str = PREREG, repo_copy: str = REPO_PREREG) -> str:
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: the pre-registration is written once")
    setup()
    blob = json.dumps(prereg_doc(), indent=1, sort_keys=True, default=str).encode()
    for p in (path, repo_copy):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(blob)
    sha = hashlib.sha256(blob).hexdigest()
    with open(os.path.splitext(path)[0] + ".sha256", "w") as f:
        f.write(sha + "\n")
    print(f"pre-registered {N_VARIANTS} variants: {path}\nsha256 {sha}")
    return sha


def load_prereg(path: str = PREREG) -> dict:
    with open(path, "rb") as f:
        blob = f.read()
    want = open(os.path.splitext(path)[0] + ".sha256").read().strip()
    if hashlib.sha256(blob).hexdigest() != want:
        raise SystemExit("pre-registration file does not match its recorded sha256")
    if os.path.exists(REPO_PREREG) and open(REPO_PREREG, "rb").read() != blob:
        raise SystemExit("repository copy of the pre-registration differs from the scratch copy")
    pre = json.loads(blob)
    for k, v in fingerprints().items():
        if pre["strategies"].get(k, {}).get("fingerprint") != v:
            raise SystemExit(f"{k}: code / parameters changed since the pre-registration")
    if pre["data"]["tables_sha256"] != data_sha():
        raise SystemExit("data tables changed since the pre-registration")
    if pre["code_sha256"].get("web_strategies.py") != code_sha()["web_strategies.py"]:
        raise SystemExit("web_strategies.py changed since the pre-registration")
    return pre


# ══════════════════════════════════════════════════════════════════════════
# controls: placebo / planted tables (outside the repository)
# ══════════════════════════════════════════════════════════════════════════
def _ca_rows(src: str) -> tuple:
    with gzip.open(os.path.join(src, "corpactions", "corpactions.csv.gz"), "rt", newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def _write_flow(dst: str, cols, rows) -> None:
    os.makedirs(os.path.join(dst, "corpactions"), exist_ok=True)
    with gzip.open(os.path.join(dst, "corpactions", "corpactions.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    src = _CTX["real_flow_dir"]
    for sub in ("index_membership",):
        if not os.path.exists(os.path.join(dst, sub)):
            os.symlink(os.path.join(src, sub), os.path.join(dst, sub))


def divm_placebo_tables(dst: str, seed: int = SEED) -> int:
    cols, rows = _ca_rows(_CTX["real_flow_dir"])
    rng = random.Random(seed)
    out, n = [], 0
    by: dict = {}
    for r in rows:
        if r["kind"] != "dividend":
            continue
        ex = int(r["ex_date"])
        key = (r["lab_symbol"], ex > PO.ERA[1])
        if key not in by:
            ser = _series(r["lab_symbol"])
            by[key] = [] if ser is None else [int(x) for x in ser[0] if (int(x) > PO.ERA[1]) == key[1]]
        dd = by[key]
        if not dd:
            continue
        nd = rng.choice(dd)
        out.append(dict(r, ex_date=nd, known_date=nd, record_date=nd, first_listed=nd))
        n += 1
    _write_flow(dst, cols, out)
    return n


def divm_planted_tables(dst: str, shift_months: int = 0) -> int:
    """A dividend row (Rs 0.0001) in month m-12 (+ shift) for each stock's best 2 % of calendar months (first open
    -> last close of the month), 2012-23."""
    cols, _rows = _ca_rows(_CTX["real_flow_dir"])
    out = []
    for s in _CTX["symbols"]:
        ser = _series(s)
        if ser is None:
            continue
        d, c, o = ser
        mon = d // 100
        rets = {}
        for m in np.unique(mon):
            if m * 100 > PO.ERA[1]:
                continue
            k = np.flatnonzero(mon == m)
            if len(k) >= 15 and o[k[0]] > 0:
                rets[int(m)] = c[k[-1]] / o[k[0]] - 1.0
        if len(rets) < 50:
            continue
        thr = np.quantile(list(rets.values()), 0.98)
        for m, r in rets.items():
            if r >= thr:
                y, mo = divmod(m, 100)
                tm = (y - 1) * 12 + (mo - 1) + shift_months
                pm = (tm // 12) * 100 + tm % 12 + 1
                ex = pm * 100 + 15
                out.append({k: "" for k in cols} | {"ex_date": ex, "symbol": s, "symbol_then": s, "lab_symbol": s,
                                                    "kind": "dividend", "dividend": 0.0001, "known_date": ex,
                                                    "first_listed": ex, "subject": "planted"})
    _write_flow(dst, cols, out)
    return len(out)


def ea_planted_tables(dst: str, symbols: list, delay: int = 0, top: float = 0.02) -> int:
    from .events_parse import BM_COLS, INS_COLS, RES_COLS
    os.makedirs(dst, exist_ok=True)
    cal = _CTX["cal"]
    pos = {d: i for i, d in enumerate(cal)}
    rows = []
    for s in symbols:
        ser = _series(s)
        if ser is None:
            continue
        d, c, _o = ser
        n = len(c)
        if n < 100:
            continue
        mv = np.full(n, np.nan)
        mv[3:] = np.abs(c[3:] / c[:-3] - 1.0)
        ok = np.isfinite(mv) & (d <= PO.ERA[1])
        if ok.sum() < 100:
            continue
        thr = np.nanquantile(mv[ok], 1.0 - top)
        for i in np.flatnonzero(ok & (mv >= thr)):
            day = int(d[i])
            if day not in pos or pos[day] + delay >= len(cal):
                continue
            m = cal[pos[day] + delay]
            nt = cal[max(0, pos[m] - 15)]
            rows.append({"symbol": s, "isin": "", "meeting_date": f"{m // 10000:04d}-{m // 100 % 100:02d}-{m % 100:02d}",
                         "purpose": "planted results", "is_results": 1,
                         "notice_ts": f"{nt // 10000:04d}-{nt // 100 % 100:02d}-{nt % 100:02d} 18:00:00"})
    for name, cols, rr in (("boardmeet.csv.gz", BM_COLS, rows), ("results.csv.gz", RES_COLS, []),
                           ("insider.csv.gz", INS_COLS, [])):
        with gzip.open(os.path.join(dst, name), "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cols), lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            w.writerows(rr)
    return len(rows)


def lv_placebo(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    out = {}
    for cutoff, first in W.lv_cutoffs(_CTX["cal"]):
        el = sorted(s for s in lv_eligible(cutoff) if _series(s) is not None
                    and W.lv_volatility(_series(s)[0], _series(s)[1], cutoff) == W.lv_volatility(
                        _series(s)[0], _series(s)[1], cutoff))
        out[first] = frozenset(rng.sample(el, min(W.LV_N, len(el))))
    return out


def lv_cheat():
    def score(s, first):
        ser = _series(s)
        if ser is None:
            return float("nan")
        d, c, o = ser
        q = W.sessions_in_quarter(first)
        cal = _CTX["cal"]
        i = cal.index(first)
        last = cal[min(i + q - 1, len(cal) - 1)]
        a, b = int(np.searchsorted(d, first)), int(np.searchsorted(d, last, side="right")) - 1
        if a >= len(d) or b <= a or not o[a] > 0:
            return float("nan")
        return -(c[b] / o[a] - 1.0)                     # lowest "volatility" = best next-quarter return
    return score


# ══════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════
def _row(s: dict) -> dict:
    return {"trades": s["trades"], "pf": s["pf"], "net": s["net"], "win_pct": s.get("win_pct")}


def _write_details(path: str, det: list) -> None:
    if not det:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(det[0].keys()))
        w.writeheader()
        for x in sorted(det, key=lambda r: (r["date"], r["symbol"])):
            w.writerow(x)


def backup_ledger(path: str) -> str:
    dst = path + ".bak_web_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, dst)
    return dst


def _progress(msg: str) -> None:
    os.makedirs(WEB_DIR, exist_ok=True)
    with open(os.path.join(WEB_DIR, "progress.txt"), "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def run(out: str = OUT, jobs: int = 4, trials_ledger: str | None = None, log=_progress) -> dict:
    from . import oldera_fo as OF
    from . import trials as TR
    pre = load_prereg()
    D.assert_outside_repo(out)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    setup(lv=True)
    bench = load_benchmarks()
    cal3 = ER.lab_calendar()
    cal_all = list(_CTX["cal"])
    cal_old = [d for d in cal_all if PO.ERA[0] <= d <= PO.ERA[1]]
    split = P.split_sessions(cal3)
    DISC = split.discovery
    daily = list(W.DAILY)
    pre_po = {"gates_fingerprint": pre["gates_fingerprint"], "strategies": {}}
    report: dict = {"prereg_sha256": open(os.path.splitext(PREREG)[0] + ".sha256").read().strip(),
                    "cost_model_daily": COST_MODEL, "cost_model_options": N.cost_model(),
                    "split_daily": {"h1": split.h1, "h2": split.h2, "final": split.final},
                    "lv_quarters": len(W.LV_SELECTION), "variants": {}, "controls": {}}
    V = report["variants"]
    tp = lambda sid, era: os.path.join(out, "trades", f"{sid}_{era}.csv")
    st_new, st_old = {}, {}

    # ── A/B daily variants ───────────────────────────────────────────────────────────────────────────────
    with _pool(jobs) as pool:
        res_d, tr_d = P.evaluate(daily, cal3, make_daily_simulator(pool, SLIP_BPS, stats=st_new),
                                 n_symbols=len(_CTX["symbols"]),
                                 ledger_path=os.path.join(_CTX["cache"], "final_ledger.json"),
                                 dataset="events_daily", cost_model=COST_MODEL)
        log(f"2023-26 daily: {[(r['id'], r['verdict']) for r in res_d]}")
        sens_d = make_daily_simulator(pool, SENS_BPS)(daily, DISC[0], DISC[1])
        keep_d = make_daily_simulator(pool, SLIP_BPS, keep_ca=True)(daily, DISC[0], DISC[1])
        nodiv_d = make_daily_simulator(pool, SLIP_BPS, credit=False)(daily, DISC[0], DISC[1])
        for r in res_d:
            sp, sc = r["spec"], r["screen"]
            tr = tr_d[r["id"]]
            disc = [t for t in tr if DISC[0] <= t[2] <= DISC[1]]
            fin = [t for t in tr if split.final[0] <= t[2] <= split.final[1]]
            rngs = {"disc": DISC, "disc_plus_final": (DISC[0], split.final[1] if fin else DISC[1])}
            if fin:
                rngs["final"] = split.final
            V[r["id"]] = {"rule": sp.desc, "new": {
                "verdict": r["verdict"], "why": r.get("why", ""), "h1": _row(sc["h1"]), "h2": _row(sc["h2"]),
                "disc": _row(sc["disc"]), "top_symbol": sc["top_symbol"],
                "top_share": None if sc["top_share"] != sc["top_share"] else round(sc["top_share"], 3),
                "final": r.get("final"), "final_note": r.get("final_note", ""), "skipped": r.get("skipped", 0),
                "ca_dropped_and_dividends": st_new.get(r["id"]),
                "disc_5bps": _row(P.stats(sens_d[r["id"]]["trades"])),
                "disc_ca_kept": _row(P.stats(keep_d[r["id"]]["trades"])),
                "disc_no_dividend_credit": _row(P.stats(nodiv_d[r["id"]]["trades"])),
                "excess_with_div": excess(disc + fin, bench, rngs, True),
                "excess_price_only": excess(disc + fin, bench, rngs, False),
                "_st": TR.series_stats(ER._tdicts(tr), cal3, DISC[0], DISC[1])}}
            ER._write_trades(tp(r["id"], "2023-26"), tr)
            pre_po["strategies"][r["id"]] = {"fingerprint": sp.fingerprint}
        res_o = PO.evaluate(daily, make_daily_simulator(pool, SLIP_BPS, stats=st_old),
                            make_daily_simulator(pool, SENS_BPS), cost_model=COST_MODEL, ledger_path=OOS_LEDGER,
                            prereg=pre_po, dataset="events_daily")
        log(f"2012-23 daily: {[(r['id'], r['verdict']) for r in res_o]}")
        nodiv_o = make_daily_simulator(pool, SLIP_BPS, credit=False)(daily, PO.ERA[0], PO.ERA[1])
        for r in res_o:
            tr = r.get("trades") or ER._read_trades(tp(r["id"], "2012-23"))
            if r.get("trades"):
                ER._write_trades(tp(r["id"], "2012-23"), tr)
            V[r["id"]]["old"] = {"verdict": r["verdict"], "why": r.get("why", ""), "note": r.get("note", ""),
                                 "h1": r.get("h1"), "h2": r.get("h2"), "all": r.get("era"),
                                 "top_symbol": r.get("top_symbol"), "top_share": r.get("top_share"),
                                 "sens_5bps": r.get("sens"), "ca_dropped_and_dividends": st_old.get(r["id"]),
                                 "no_dividend_credit": _row(P.stats(nodiv_o[r["id"]]["trades"])),
                                 "excess_with_div": excess(tr, bench, {"h1": PO.H1, "h2": PO.H2, "era": PO.ERA}, True),
                                 "excess_price_only": excess(tr, bench, {"era": PO.ERA}, False),
                                 "_st": TR.series_stats(ER._tdicts(tr), cal_old, PO.ERA[0], PO.ERA[1])}
    log("daily main runs done")
    for sp in daily:
        V[sp.id]["account"] = {"disc_2023_26": account_daily(tr_d[sp.id], sp, cal3, DISC[0], DISC[1]),
                               "old_era": account_daily(ER._read_trades(tp(sp.id, "2012-23")), sp, cal_old,
                                                        PO.ERA[0], PO.ERA[1])}

    # ── daily controls ───────────────────────────────────────────────────────────────────────────────────
    C = report["controls"]
    real_sel = dict(W.LV_SELECTION)
    lvs = W.BY_ID["web_lowvol30_nifty100"]
    dvs = W.BY_ID["web_divm_pred_exmonth"]
    W.LV_SELECTION.clear()
    W.LV_SELECTION.update(lv_placebo())
    with _pool(jobs) as pool:
        a = make_daily_simulator(pool)([lvs], DISC[0], DISC[1])[lvs.id]["trades"]
        b = make_daily_simulator(pool)([lvs], PO.ERA[0], PO.ERA[1])[lvs.id]["trades"]
    C["lowvol_placebo_random30"] = {"disc_2023_26": _row(P.stats(a)), "old_era": _row(P.stats(b)),
                                    "excess_old_nifty100": ER.block_bootstrap(abnormal_returns(b, bench["NIFTY100"], True))}
    W.LV_SELECTION.clear()
    W.LV_SELECTION.update(lv_selection(cheat=lv_cheat()))
    with _pool(jobs) as pool:
        b = make_daily_simulator(pool)([lvs], PO.ERA[0], PO.ERA[1])[lvs.id]["trades"]
    C["lowvol_planted_next_quarter_return"] = {"old_era": _row(P.stats(b))}
    W.LV_SELECTION.clear()
    W.LV_SELECTION.update(real_sel)
    log("low-vol controls done")
    pdir = os.path.join(out, "divm_placebo_tables")
    n = divm_placebo_tables(pdir)
    setup(flow_dir=pdir)
    with _pool(jobs) as pool:
        a = make_daily_simulator(pool)([dvs], DISC[0], DISC[1])[dvs.id]["trades"]
        b = make_daily_simulator(pool)([dvs], PO.ERA[0], PO.ERA[1])[dvs.id]["trades"]
    C["divm_placebo"] = {"rows": n, "disc_2023_26": _row(P.stats(a)), "old_era": _row(P.stats(b)),
                         "excess_old_nifty500": ER.block_bootstrap(abnormal_returns(b, bench["NIFTY500"], True))}
    for name, shift in (("divm_planted", 0), ("divm_planted_month_after", 1)):
        ddir = os.path.join(out, name + "_tables")
        n = divm_planted_tables(ddir, shift)
        setup(flow_dir=ddir)
        with _pool(jobs) as pool:
            b = make_daily_simulator(pool, credit=False)([dvs], PO.ERA[0], PO.ERA[1])[dvs.id]["trades"]
        C[name] = {"rows": n, "old_era": _row(P.stats(b))}
    setup(flow_dir=_CTX["real_flow_dir"])
    log("dividend controls done")

    # ── C/D option variants: 2023-26 ─────────────────────────────────────────────────────────────────────
    cal_opt = N.lab_calendar(BARS3Y)
    split_o = P.split_sessions(cal_opt)
    DISC_O = split_o.discovery
    _CTX["opt_syms"] = option_underlyings()
    ospecs = W.option_specs(_CTX["opt_syms"])
    _CTX["opt_specs"] = ospecs
    det_new: dict = {}
    oledger = os.path.join(D.default_cache_dir(BARS3Y), "final_ledger.json")
    with _pool(jobs) as pool:
        res_on, tr_on = P.evaluate(ospecs, cal_opt, make_opt_simulator(pool, cal_opt, details=det_new),
                                   n_symbols=len(_CTX["opt_syms"]), ledger_path=oledger,
                                   dataset=os.path.realpath(BARS3Y), cost_model=N.cost_model())
        log(f"2023-26 options: {[(r['id'], r['verdict']) for r in res_on]}")
        sens_on = make_opt_simulator(pool, cal_opt, mult=N.SENS_MULT, audit=False)(ospecs, DISC_O[0], DISC_O[1])
    for r in res_on:
        sp, sc = r["spec"], r["screen"]
        tr = tr_on[r["id"]]
        dets = det_new.get(r["id"], [])
        keep = {(t[2], t[1]) for t in tr}
        dets = [x for x in dets if (x["date"], x["symbol"]) in keep]
        V[r["id"]] = {"rule": sp.desc, "new": {
            "verdict": r["verdict"], "why": r.get("why", ""), "h1": _row(sc["h1"]), "h2": _row(sc["h2"]),
            "disc": _row(sc["disc"]), "top_symbol": sc["top_symbol"],
            "top_share": None if sc["top_share"] != sc["top_share"] else round(sc["top_share"], 3),
            "final": r.get("final"), "final_note": r.get("final_note", ""), "skipped_untraded": r.get("skipped", 0),
            "disc_haircut_x2_5": _row(P.stats(sens_on[r["id"]]["trades"])),
            "stale_exits": sum(1 for x in dets if "stale" in str(x["reason"])),
            "mean_premium_rs": round(float(np.mean([abs(x["entry_val"]) * x["lot"] for x in dets])), 0) if dets else None,
            "_st": TR.series_stats(ER._tdicts(tr), cal_opt, DISC_O[0], DISC_O[1])},
            "account": {"disc_2023_26": account_options([x for x in dets if x["date"] <= DISC_O[1]], cal_opt,
                                                        DISC_O[0], DISC_O[1])}}
        ER._write_trades(tp(r["id"], "2023-26"), tr)
        _write_details(os.path.join(out, "trades", f"{r['id']}_2023-26_legs.csv"), dets)
    # placebo (2023-26 DISCOVERY) before switching nsedata to the old view
    ev_real = ED.EVENTS_DIR
    ev_pl = os.path.join(out, "ea_placebo_tables")
    ER.placebo_tables(ev_pl, seed=SEED)
    E.use(ev_pl)
    for nm in ("insider", "results", "boardmeet"):
        E._load(nm)
    with _pool(jobs) as pool:
        pl_new = make_opt_simulator(pool, cal_opt, audit=False)(ospecs, DISC_O[0], DISC_O[1])
    E.use(ev_real)
    for nm in ("insider", "results", "boardmeet"):
        E._load(nm)
    log("options 2023-26 done")

    # ── C/D option variants: 2012-23 (old-era nsedata view) ──────────────────────────────────────────────
    OF.use(OF.VIEW_DIR)
    N._load()
    _CTX["opt_syms"] = option_underlyings(OF.VIEW_DIR)
    ospecs_o = W.option_specs(_CTX["opt_syms"])
    _CTX["opt_specs"] = ospecs_o
    det_old: dict = {}
    pre_po_o = {"gates_fingerprint": pre["gates_fingerprint"],
                "strategies": {sp.id: {"fingerprint": sp.fingerprint} for sp in ospecs_o}}
    with _pool(jobs) as pool:
        res_oo = PO.evaluate(ospecs_o, make_opt_simulator(pool, cal_old, details=det_old),
                             make_opt_simulator(pool, cal_old, mult=N.SENS_MULT, audit=False),
                             cost_model=N.cost_model(), ledger_path=OOS_LEDGER, prereg=pre_po_o, dataset=OF.VIEW_DIR)
    log(f"2012-23 options: {[(r['id'], r['verdict']) for r in res_oo]}")
    for r in res_oo:
        tr = r.get("trades") or ER._read_trades(tp(r["id"], "2012-23"))
        if r.get("trades"):
            ER._write_trades(tp(r["id"], "2012-23"), tr)
        dets = det_old.get(r["id"], [])
        _write_details(os.path.join(out, "trades", f"{r['id']}_2012-23_legs.csv"), dets)
        V[r["id"]]["old"] = {"verdict": r["verdict"], "why": r.get("why", ""), "note": r.get("note", ""),
                             "h1": r.get("h1"), "h2": r.get("h2"), "all": r.get("era"),
                             "top_symbol": r.get("top_symbol"), "top_share": r.get("top_share"),
                             "sens_haircut_x2_5": r.get("sens"),
                             "stale_exits": sum(1 for x in dets if "stale" in str(x["reason"])),
                             "_st": TR.series_stats(ER._tdicts(tr), cal_old, PO.ERA[0], PO.ERA[1])}
        V[r["id"]]["account"]["old_era"] = account_options(dets, cal_old, PO.ERA[0], PO.ERA[1])
    E.use(ev_pl)
    for nm in ("insider", "results", "boardmeet"):
        E._load(nm)
    with _pool(jobs) as pool:
        pl_old = make_opt_simulator(pool, cal_old, audit=False)(ospecs_o, PO.ERA[0], PO.ERA[1])
    E.use(ev_real)
    for nm in ("insider", "results", "boardmeet"):
        E._load(nm)
    C["ea_placebo_random_dates"] = {i: {"disc_2023_26": _row(P.stats(pl_new[i]["trades"])),
                                        "old_era": _row(P.stats(pl_old[i]["trades"]))} for i in W.OPTION_IDS}
    sub = _CTX["opt_syms"][::3]
    m30 = [sp for sp in ospecs_o if sp.id == "web_ea_straddle_m3_0"]
    for name, delay in (("ea_planted", 0), ("ea_planted_delayed_25", 25)):
        ddir = os.path.join(out, name + "_tables")
        n = ea_planted_tables(ddir, sub, delay=delay)
        E.use(ddir)
        for nm in ("insider", "results", "boardmeet"):
            E._load(nm)
        with _pool(jobs) as pool:
            b = make_opt_simulator(pool, cal_old, audit=False, symbols=sub)(m30, PO.ERA[0], PO.ERA[1])
        C[name] = {"rows": n, "underlyings": len(sub), "web_ea_straddle_m3_0_old_era": _row(P.stats(
            b["web_ea_straddle_m3_0"]["trades"]))}
    E.use(ev_real)
    for nm in ("insider", "results", "boardmeet"):
        E._load(nm)
    log("options 2012-23 + controls done")

    # ── trials ledger ────────────────────────────────────────────────────────────────────────────────────
    if trials_ledger:
        n1 = TR.record_evaluation(res_d, tr_d, dataset="events_daily", cost_model=COST_MODEL,
                                  ledger_path=trials_ledger, calendar=cal3, source=out)
        n2 = TR.record_evaluation(res_on, tr_on, dataset="bars3y", cost_model=N.cost_model(),
                                  ledger_path=trials_ledger, calendar=cal_opt, source=out)
        rows_old = ER.ledger_rows_old([dict(r, trades=r.get("trades") or ER._read_trades(tp(r["id"], "2012-23")))
                                       for r in res_o], cal_old, source="webstrats/oos_ledger.json",
                                      cost_model=COST_MODEL)
        rows_old += ER.ledger_rows_old([dict(r, trades=r.get("trades") or ER._read_trades(tp(r["id"], "2012-23")))
                                        for r in res_oo], cal_old, source="webstrats/oos_ledger.json",
                                       dataset="nse_old_view", cost_model=N.cost_model())
        n3 = TR.TrialsLedger(trials_ledger).append(rows_old)
        log(f"trials ledger: {n1} + {n2} rows (2023-26) + {n3} rows (2012-23)")

    # ── DSR / PSR and the 'proven' call ──────────────────────────────────────────────────────────────────
    rows = TR.TrialsLedger(trials_ledger).read() if trials_ledger else []
    n_full = TR.count_trials(rows, "2023-26") if rows else None
    n_old = TR.count_trials(rows, "2012-23") if rows else None
    v_full = TR.var_sr_robust(rows, "2023-26", "screen", "bars3y")[0] if rows else float("nan")
    srs = [v["new"]["_st"]["sr_daily"] for v in V.values()]
    srs = [x for x in srs if x == x]
    v_fair = float(np.var(srs, ddof=1)) if len(srs) > 1 else float("nan")
    vv = v_full if v_full == v_full else v_fair
    report["trials"] = {"n_2023_26": n_full, "n_2012_23": n_old, "n_fair": N_VARIANTS, "v_sr_robust_2023_26": v_full,
                        "v_sr_study": v_fair}
    for sid, v in V.items():
        st = v["new"].pop("_st")
        v["new"]["dsr_disc"] = ER.dsr_block(st, n_full or N_VARIANTS, vv, N_VARIANTS, v_fair)
        v["new"]["mean_gross"], v["new"]["mean_cost"] = st["mean_gross"], st["mean_cost"]
        v["new"]["haircut_per_trade"] = None if st["mean_gross"] != st["mean_gross"] else round(
            0.5 * st["mean_gross"] - st["mean_cost"], 2)
        so = v.get("old", {}).pop("_st", None)
        if so and so.get("sr_daily") == so.get("sr_daily"):
            v["old"]["dsr"] = ER.dsr_block(so, n_old or N_VARIANTS, vv, N_VARIANTS, v_fair)
        d = v["new"]["dsr_disc"].get("dsr_full")
        ok = bool(v["new"]["verdict"] == "CONFIRMED" and v.get("old", {}).get("verdict") == "OOS-CONFIRMED"
                  and d is not None and d >= DSR_MIN)
        if sid in W.LONG_ONLY_BENCH and ok:
            b = W.LONG_ONLY_BENCH[sid]
            lo_new = v["new"]["excess_with_div"]["disc_plus_final"][b].get("ci95_mean_ar_pct", [None])[0]
            lo_old = v["old"]["excess_with_div"]["era"][b].get("ci95_mean_ar_pct", [None])[0]
            ok = lo_new is not None and lo_old is not None and lo_new > 0 and lo_old > 0
        v["proven"] = ok
    report["n_proven"] = sum(1 for v in V.values() if v["proven"])
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=ER._js)
    log(f"report: {os.path.join(out, 'report.json')}; proven: {report['n_proven']} of {len(V)}")
    return report


def smoke(jobs: int = 2) -> int:
    """Plumbing only: every variant runs without error on a few symbols and a short window. Prints NO P&L and no
    trade counts."""
    setup(lv=True)
    syms = _CTX["symbols"][::60]
    r = make_daily_simulator(None, symbols=syms)(list(W.DAILY), 20190101, 20190630)
    bad = {i: v["error"] for i, v in r.items() if v["error"]}
    _CTX["opt_syms"] = option_underlyings()[::40]
    _CTX["opt_specs"] = W.option_specs(_CTX["opt_syms"])
    cal_opt = N.lab_calendar(BARS3Y)
    r2 = make_opt_simulator(None, cal_opt)(_CTX["opt_specs"], 20240101, 20240331)
    bad.update({i: v["error"] or v["lookahead"] for i, v in r2.items() if v["error"] or v["lookahead"]})
    fires = {i: bool(v["trades"]) for i, v in list(r.items()) + list(r2.items())}
    print("smoke:", "OK (no errors)" if not bad else bad, "| fires at least once (yes/no only):", fires)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.web_run")
    ap.add_argument("--data-check", action="store_true")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--trials", default=os.environ.get("STRATLAB_TRIALS"))
    a = ap.parse_args(argv)
    if a.data_check:
        print(json.dumps(data_check(), indent=1, default=ER._js)[:6000])
        return 0
    if a.smoke:
        return smoke()
    if a.preregister:
        preregister()
        return 0
    if a.run:
        if a.trials and os.path.exists(a.trials):
            print("ledger backed up:", backup_ledger(a.trials))
        run(a.out, a.jobs, a.trials)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
