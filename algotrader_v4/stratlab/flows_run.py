"""
stratlab/flows_run.py — run the pre-registered FLOW study (flows_strategies.py: bulk / block deals and NIFTY index
inclusions / deletions) ONCE and judge it. Research only; writes outside the repository.

    cd algotrader_v4
    python -m stratlab.flows_run --data-check            # counts only (no prices, no returns)
    python -m stratlab.flows_run --preregister           # freeze ids, fingerprints, keyword list, gates, data hashes
    python -m stratlab.flows_run --smoke                 # plumbing: errors only, no P&L, no trade counts
    python -m stratlab.flows_run --run --trials <ledger> # the one run: both eras, controls, account, trials ledger

Judgement (the lab's functions, unchanged — the framework of events_run.py):
  2023-26   protocol.evaluate on the lab's 3-year session calendar (DISCOVERY 70 % = H1 + H2, FINAL 30 %, FINAL once
            and only for screen passers, one-look ledger <events_daily cache>/final_ledger.json).
  2012-23   protocol_oos.evaluate (H1 2012-17 / H2 2018-23, >= 60 trades in EACH half, one look,
            ledger <scratch>/flows/oos_ledger.json). EVERY variant gets the old-era look (pre-registered).
  "proven" = CONFIRMED on 2023-26 FINAL AND OOS-CONFIRMED on 2012-23 AND DSR >= 0.95 (N = every 2023-26 trial in
            the ledger; the fair N = the 8 variants of this study is also reported).
Data: events_daily's cache (point-in-time monthly top 500 by turnover 2012-2026, delisted names kept, official NSE
bars); flow tables from flowdata.py (deals, index changes). Fills / costs: sim.sim_positional unchanged (next-open
fill, 2 bps per side, CNC costs for longs, futures costs on the cash price for shorts, Rs 1 lakh notional).
Look-ahead: every strategy call runs inside nsedata.cutoff(decision day + 1), which clips every flowdata accessor to
rows known by the decision evening; positions open only on universe-member days (data_daily.gate); the
corporate-action rule of events_run drops trades across unexplained gaps / scheme ex-days.

Report-only (never part of a verdict): 1:3 bracket twins, the Rs 10 lakh account (capital.py), excess return vs
NIFTY 50 / NIFTY Next 50 / NIFTY 100 / NIFTY 500 with a month-clustered bootstrap, a placebo (same stocks, random
dates, same count) and planted-edge controls (deals and index additions).
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import glob
import gzip
import hashlib
import json
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
from . import flows_strategies as FS
from . import nsedata as N
from . import protocol as P
from . import protocol_oos as PO
from . import sim

SCRATCH = DD._SCRATCH
FLOWS_DIR = os.environ.get("STRATLAB_FLOWS_OUT", os.path.join(SCRATCH, "flows"))
OUT = os.path.join(FLOWS_DIR, "run1")
PREREG = os.path.join(FLOWS_DIR, "PREREGISTERED_FLOWS.json")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "flows_v1.json")
OOS_LEDGER = os.path.join(FLOWS_DIR, "oos_ledger.json")
DATA_CHECK = os.path.join(FLOWS_DIR, "data_check.json")
BENCH_FILE = os.path.join(FLOWS_DIR, "benchmarks.json.gz")
FILL_MODEL = "fl1"
COST_MODEL = f"{sim.COST_MODEL}/{FILL_MODEL}"
SLIP_BPS, SENS_BPS = sim.DEFAULT_SLIPPAGE_BPS, sim.SENSITIVITY_SLIPPAGE_BPS
DSR_MIN = 0.95
PLACEBO_SEED = 20261115
BENCH_NAMES = {"NIFTYNEXT50": ("CNX Nifty Junior", "Nifty Next 50"), "NIFTY100": ("CNX 100", "Nifty 100"),
               "NIFTY500": ("S&P CNX 500", "CNX 500", "Nifty 500")}
_DATE = 2
_CTX: dict = {}


# ══════════════════════════════════════════════════════════════════════════
# set-up
# ══════════════════════════════════════════════════════════════════════════
def setup(flow_dir: str | None = None) -> dict:
    """events_run's set-up (daily cache, calendar, exclusions, F&O list, NIFTY) + the flow tables. Must run before
    forking workers (they inherit every table)."""
    ctx = ER.setup()
    if flow_dir is not None:
        F.use(flow_dir)
    F._load_deals()
    F._load_index()
    _CTX.update(ctx)
    return _CTX


class _Guard:
    """fn(d, p) inside nsedata.cutoff(decision day + 1): every flowdata accessor is clipped to rows known by the
    evening of the decision day (deals dated <= that day, index changes announced <= that day)."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "guarded")

    def __call__(self, d, p):
        with N.cutoff(int(d.date[-1]) + 1):
            return self.fn(d, p)


def guarded(spec, sd):
    g = DD.gate(spec, sd)
    return dataclasses.replace(g, fn=_Guard(g.fn))


def _specs_by_id() -> dict:
    return {sp.id: sp for sp in list(FS.SPECS) + list(FS.RR3) + list(_CTX.get("extra_specs", []))}


def _work_daily(args):
    symbol, ids, lo, hi, slip, keep_ca = args
    by = _specs_by_id()
    out = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None, "ca_dropped": 0} for i in ids}
    sd = DD.load_symbol(_CTX["cache"], symbol)
    if sd is None:
        return out
    for i in ids:
        try:
            tr, sk = sim.sim_positional(guarded(by[i], sd), sd, lo, hi, _CTX["nifty"], _CTX["vix"], slip)
            if not keep_ca:
                tr, n = ER.drop_excluded(tr)
                out[i]["ca_dropped"] = n
            out[i]["trades"], out[i]["skipped"] = tr, sk
        except Exception as exc:                                    # reported as INVALID by the protocol
            out[i]["error"] = f"{symbol}: {type(exc).__name__}: {str(exc)[:160]}"
    return out


def make_daily_simulator(pool, slip_bps: float = SLIP_BPS, keep_ca: bool = False, stats: dict | None = None):
    def simulate(specs, lo, hi):
        ids = [sp.id for sp in specs]
        merged = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None} for i in ids}
        jobs = [(s, ids, lo, hi, slip_bps / 1e4, keep_ca) for s in _CTX["symbols"]]
        for res in (pool.imap_unordered(_work_daily, jobs, chunksize=8) if pool is not None
                    else map(_work_daily, jobs)):
            for i, r in res.items():
                m = merged[i]
                m["trades"].extend(r["trades"])
                m["skipped"] += r["skipped"]
                m["error"] = m["error"] or r["error"]
                if stats is not None:
                    stats[i] = stats.get(i, 0) + r["ca_dropped"]
        for m in merged.values():
            m["trades"].sort(key=lambda t: (t[2], t[1]))
        return merged
    return simulate


# ══════════════════════════════════════════════════════════════════════════
# benchmarks (report-only)
# ══════════════════════════════════════════════════════════════════════════
def build_benchmarks(path: str = BENCH_FILE) -> dict:
    """NIFTY Next 50 / NIFTY 100 / NIFTY 500 official daily open + close: NSE index files 2012-01-02 .. 2023-11-09
    (<nse_old>/index_all_daily_pr.csv.gz), then NSE ind_close_all day files (<scratch>/github2_r7/ind_close_all)."""
    want = {n: ix for ix, names in BENCH_NAMES.items() for n in names}
    rows: dict = {ix: {} for ix in BENCH_NAMES}
    with gzip.open(os.path.join(DD.OLD_DIR, "index_all_daily_pr.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            ix = want.get(r["index"])
            if ix and float(r["open"] or 0) > 0 and float(r["close"] or 0) > 0:
                rows[ix][int(r["date"])] = (float(r["open"]), float(r["close"]))
    for fn in sorted(glob.glob(os.path.join(SCRATCH, "github2_r7", "ind_close_all", "ind_close_all_*.csv"))):
        with open(fn, newline="") as f:
            for r in csv.DictReader(f):
                ix = want.get((r.get("Index Name") or "").strip())
                if not ix:
                    continue
                dd = r["Index Date"].strip()
                day = int(dd[6:10] + dd[3:5] + dd[0:2])
                if day <= DD.ERA[1]:
                    continue
                try:
                    o, c = float(r["Open Index Value"]), float(r["Closing Index Value"])
                except (TypeError, ValueError):
                    continue
                if o > 0 and c > 0:
                    rows[ix][day] = (o, c)
    out = {ix: {"date": sorted(v), "o": [v[k][0] for k in sorted(v)], "c": [v[k][1] for k in sorted(v)]}
           for ix, v in rows.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(out, f)
    return out


def load_benchmarks(path: str = BENCH_FILE) -> dict:
    if not os.path.exists(path):
        build_benchmarks(path)
    with gzip.open(path, "rt") as f:
        raw = json.load(f)
    out = {ix: {"date": np.array(v["date"], np.int64), "o": np.array(v["o"]), "c": np.array(v["c"])}
           for ix, v in raw.items()}
    nf = _CTX.get("nifty")
    if nf is not None:
        out = {"NIFTY50": {"date": np.asarray(nf["date"], np.int64), "o": np.asarray(nf["o"]),
                           "c": np.asarray(nf["c"])}, **out}
    return out


def excess(trades: list, bench: dict, ranges: dict) -> dict:
    """Mean / median excess return per trade vs each benchmark, month-clustered bootstrap CI, per range."""
    return {k: {b: ER.block_bootstrap(ER.abnormal_returns([t for t in trades if lo <= t[_DATE] <= hi], s))
                for b, s in bench.items()} for k, (lo, hi) in ranges.items()}


# ══════════════════════════════════════════════════════════════════════════
# data check (counts only)
# ══════════════════════════════════════════════════════════════════════════
def _era_of(day: int) -> str:
    return "2012-23" if day <= PO.ERA[1] else "2023-26"


def data_check(path: str = DATA_CHECK, write: bool = True) -> dict:
    """Counts of deals by client class and year, qualifying net-flow events (all, and those landing on a top-500
    member session of the cache), index-change rows by index / announcement source / precision and the pooled
    events per era and half. No price or return is read."""
    setup()
    m = DD.manifest(_CTX["cache"])
    t = F._S["deals_all"]
    cls_rows = collections.Counter()
    by_year = collections.defaultdict(collections.Counter)
    cls_names = collections.defaultdict(collections.Counter)
    for dt, cl in zip(t["date"], t["client"]):
        c = FS.classify(cl)
        cls_rows[c] += 1
        by_year[str(int(dt) // 10000)][c] += 1
        cls_names[c][" ".join(str(cl).upper().split())] += 1
    # net flows per (symbol, date, client), intra-group transfers removed (as the strategies see them)
    acc: dict = {}
    ix = F._S["deals_ix"]
    tt = F._S["deals"]
    n_transfer = collections.Counter()
    for lab, sl in ix.items():
        rows = {c: tt[c][sl] for c in ("date", "client", "side", "qty", "price", "value_cr")}
        mk = FS.group_transfer_mask(rows)
        for dt in rows["date"][mk]:
            n_transfer[str(int(dt) // 10000)] += 1
        for d0, cl, n in FS.net_flows(rows):
            acc[(lab, d0, cl)] = n
    member = {}
    for s, info in m["symbols"].items():
        if s.startswith("_") or info.get("member_days", 0) <= 0:
            continue
        sd = DD.load_symbol(_CTX["cache"], s)
        if sd is not None:
            member[s] = set(int(x) for x in sd.days[sd.member])
    ev = collections.defaultdict(collections.Counter)
    for (lab, dt, cl), net in acc.items():
        c = FS.classify(cl)
        if c not in FS.INSTITUTION or abs(net) < 10.0:
            continue
        side = "buy" if net > 0 else "sell"
        y = str(dt // 10000)
        ev[f"all_{side}"][y] += 1
        if dt in member.get(lab, ()):
            ev[f"top500_{side}"][y] += 1
            ev[f"top500_{side}_by_class"][c] += 1
    ch = list(csv.DictReader(open(os.path.join(F.FLOW_DIR, "index_membership", "index_changes.csv"))))
    idx = collections.defaultdict(collections.Counter)
    for r in ch:
        idx[r["index"]][f"{r['action']} announce={'press' if r['announce_source'] == 'press' else 'none'} "
                        f"effective={r['effective_precision']}"] += 1
    # pooled events (as the strategies see them; the full table, counted once per (symbol, effective, kind))
    pooled = collections.defaultdict(set)
    for r in ch:
        for kind, pairs in FS.POOL.items():
            if (r["index"], r["action"]) in pairs:
                pooled[kind].add((r["lab_symbol"], int(r["announce_date"]), int(r["effective_date"]),
                                  r["announce_source"] == "press", r["effective_precision"]))
    pev: dict = {}
    for kind, s in pooled.items():
        uniq: dict = {}
        for lab, a, e, press, prec in s:
            k = (lab, e)
            o = uniq.get(k)
            uniq[k] = (lab, a, e, press or (o[3] if o else False), prec if not o or prec != "month" else o[4])
        cnt = collections.Counter()
        for lab, a, e, press, prec in uniq.values():
            per = "2012-17" if e <= PO.H1[1] else ("2018-23" if e <= PO.ERA[1] else "2023-26")
            in_cache = lab in member
            cnt[f"{per} all"] += 1
            if prec == "month":
                cnt[f"{per} dropped_month_precision"] += 1
                continue
            if not in_cache:
                cnt[f"{per} not_in_cache"] += 1
                continue
            cnt[f"{per} usable_effective_based"] += 1
            if press:
                cnt[f"{per} usable_runup_press"] += 1
            else:
                cnt[f"{per} excluded_from_runup_no_real_announcement"] += 1
        pev[kind] = dict(sorted(cnt.items()))
    out = {"written": time.strftime("%Y-%m-%d %H:%M"), "flow_dir": F.FLOW_DIR,
           "deals_rows": int(len(t["date"])), "deal_rows_by_class": dict(cls_rows),
           "deal_rows_by_year_and_class": {y: dict(c) for y, c in sorted(by_year.items())},
           "distinct_client_names_by_class": {c: len(v) for c, v in cls_names.items()},
           "top_client_names_by_class": {c: [n for n, _ in v.most_common(25)] for c, v in cls_names.items()},
           "institution_net_events_ge_10cr": {k: dict(sorted(v.items())) for k, v in ev.items()},
           "intra_group_transfer_rows_dropped_by_year": dict(sorted(n_transfer.items())),
           "index_change_rows": {k: dict(sorted(v.items())) for k, v in idx.items()},
           "pooled_index_events": pev,
           "note": "counts only: no price or return was read to produce this file"}
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1, default=ER._js)
    return out


# ══════════════════════════════════════════════════════════════════════════
# pre-registration
# ══════════════════════════════════════════════════════════════════════════
CODE_FILES = ("flows_run.py", "flows_strategies.py", "flowdata.py", "events_run.py", "events_data.py",
              "events_daily.py", "sim.py", "protocol.py", "protocol_oos.py", "capital.py", "data_daily.py", "trials.py")


def code_sha() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: ER._sha(os.path.join(here, f)) for f in CODE_FILES}


def data_sha() -> dict:
    out = {}
    for rel in ("deals/deals.csv.gz", "index_membership/index_changes.csv", "index_membership/index_members.csv.gz"):
        p = os.path.join(F.FLOW_DIR, rel)
        out[rel] = ER._sha(p) if os.path.exists(p) else None
    return out


def prereg_doc() -> dict:
    m = DD.manifest(ED.CACHE)
    dc = json.load(open(DATA_CHECK)) if os.path.exists(DATA_CHECK) else {}
    return {
        "name": "flows_v1", "written": time.strftime("%Y-%m-%d %H:%M"),
        "question": "Do big-investor bulk / block deals and NIFTY index inclusions / deletions give a tradable edge "
                    "on NSE stocks after costs, in two eras?",
        "variants": [{"id": sp.id, "kind": sp.kind, "horizon": sp.horizon, "family": sp.family, "rule": sp.desc,
                      "params": sp.params, "fingerprint": sp.fingerprint} for sp in FS.SPECS],
        "report_only_twins": [{"id": sp.id, "params": sp.params, "fingerprint": sp.fingerprint} for sp in FS.RR3],
        "strategies": {sp.id: {"fingerprint": sp.fingerprint} for sp in list(FS.SPECS) + list(FS.RR3)},
        "n_variants": len(FS.SPECS),
        "client_classes": {
            "rule": "client name upper-cased, dots -> spaces, whitespace collapsed; the FIRST class whose regex "
                    "list matches wins, in this order; no match = OTHER",
            "ordered_keywords": [{"class": c, "regex": list(p)} for c, p in FS.CLIENT_CLASSES],
            "institution_classes": list(FS.INSTITUTION),
            "excluded": ["PROP_BROKER (prop / HFT / brokers / ODI issuers / arbitrage desks)",
                         "OTHER (individuals, companies, promoters, family trusts, banks, unclassified foreign "
                         "entities)"],
            "netting": "per (session, stock, client name): net = sum(BUY value) - sum(SELL value) over its bulk and "
                       "block deals that session (value_cr = qty x price / 1e7, unadjusted); a same-day round trip "
                       "nets to ~0 and does not qualify. Before netting, intra-group transfers are removed: a BUY and "
                       "a SELL in the same stock and session with equal quantity and price whose client names start "
                       "with the same word (ignoring THE, M/S) are both dropped",
            "counts": {k: dc.get(k) for k in ("deal_rows_by_class", "distinct_client_names_by_class",
                                              "institution_net_events_ge_10cr")}},
        "index_pooling": {
            "pooled": True,
            "additions": "IN to NIFTY 50 (from anywhere, incl. NIFTY Next 50) or IN to NIFTY 100 (a new NIFTY Next "
                         "50 member from outside the top 100); one event per (stock, effective date)",
            "deletions": "OUT of NIFTY 50 (incl. demotion to NIFTY Next 50) or OUT of NIFTY 100",
            "not_events": "NIFTY Next 50 IN / OUT rows that are only moves to / from NIFTY 50 (already counted as "
                          "NIFTY 50 deletions / additions)",
            "month_precision": "rows whose effective date is known only to the month are never used (any variant)",
            "runup_variants": "fl_idx_add_runup and fl_idx_del_rundown use ONLY rows with announce_source == "
                              "'press' (a real press-release date); rows dated 'the session before the effective "
                              "date' are excluded from them (counts in data_check.pooled_index_events)",
            "counts": dc.get("pooled_index_events")},
        "availability_rule": {
            "deals": "a deal dated D is published after D's close: decision at D's close, fill at the next "
                     "session's open (flowdata.deals with before = D + 1; nsedata.cutoff(D + 1) clips everything)",
            "index_announcement": "only a date exists: assumed published after that date's close; the decision is "
                                  "the first symbol session ON or AFTER the announcement date (announce_date <= "
                                  "decision day), the fill the next session's open (a weekend release fills on the "
                                  "second session after it)",
            "index_effective": "effective_date = first session of the new composition; index funds trade at the "
                               "close of the session before it. Effective-date variants decide at the close of the "
                               "first symbol session on / after the effective date and fill at the next open",
            "runup_exit": "max_hold = sessions from the entry session to the last calendar session before the "
                          "effective date (exchange calendar, public in advance); exit at that session's close",
            "fo_listed": "short variants need a stock future traded on the decision day (F&O bhavcopy, EOD)"},
        "data": {"sources": ["NSE bulk / block deals historical report (flowdata deals)",
                             "NSE Indices month-end constituent files + press releases (flowdata index_changes)",
                             "events_daily point-in-time cache (NSE bhavcopy prices)",
                             "NSE F&O bhavcopy (stock futures list)",
                             "NSE index files (NIFTY Next 50 / 100 / 500 benchmark, report-only)"],
                 "flow_tables_sha256": data_sha(),
                 "daily_cache": {"version": m.get("events_daily"), "sessions": len(m["calendar"]),
                                 "first": m["calendar"][0], "last": m["calendar"][-1], "top_n": m.get("top_n"),
                                 "symbols_ever_members": m.get("universe_symbols")}},
        "universe": "point-in-time monthly top 500 NSE equity shares by 60-session turnover (events_daily), delisted "
                    "names kept; a position may be opened only on a member day; index variants trade the changed "
                    "stocks themselves under the same member-day rule; shorts need a listed stock future",
        "eras": {"2023-26": {"calendar": "the lab's 3-year session calendar (trials.calendar_3y)",
                             "protocol": "protocol.evaluate, DISCOVERY 70 % (H1 / H2) + FINAL 30 %, FINAL once and "
                                         "only for screen passers",
                             "ledger": "<events_daily cache>/final_ledger.json"},
                 "2012-23": {"era": list(PO.ERA), "h1": list(PO.H1), "h2": list(PO.H2),
                             "protocol": "protocol_oos.evaluate (one look, ledger flows/oos_ledger.json)"},
                 "old_era_look": "EVERY variant gets its old-era judgement, whatever its 2023-26 result"},
        "gates": {"screen": {"min_pf": P.SCREEN_MIN_PF, "min_trades": P.SCREEN_MIN_TRADES,
                             "max_symbol_share": P.SCREEN_MAX_SYMBOL_SHARE, "net_gt_0": ["H1", "H2"]},
                  "final": {"min_pf": P.FINAL_MIN_PF, "net_gt_0": True},
                  "oos": PO.GATES, "dsr_min": DSR_MIN,
                  "no_relaxation": "a variant short of a trade minimum fails on trade count; nothing is pooled, "
                                   "relaxed or re-defined after results",
                  "proven": "CONFIRMED on 2023-26 FINAL and OOS-CONFIRMED on 2012-23 and DSR >= 0.95 with N = all "
                            "2023-26 trials in the ledger (the fair N = this study's 8 variants is also reported)"},
        "gates_fingerprint": PO.GATES_FINGERPRINT,
        "expected_from_counts": "index events are few: about 6-7 pooled additions and as many deletions per "
                                "half-yearly review, i.e. well under 60 in the 2023-26 DISCOVERY window; every index "
                                "variant is therefore expected to FAIL the 2023-26 screen on trade count and cannot "
                                "be 'proven' (stated before any return was computed). The old era has 58-71 usable "
                                "pooled events per half (data_check.pooled_index_events), before the member-day, "
                                "stock-future and one-position-at-a-time rules: at or below the 60-trade minimum, so "
                                "the trade-count gate may decide the index variants there too",
        "benchmarks_file_sha256": ER._sha(BENCH_FILE) if os.path.exists(BENCH_FILE) else None,
        "fills_costs": {"cost_model": COST_MODEL, "slippage_bps": SLIP_BPS, "sensitivity_bps": SENS_BPS,
                        "notional": sim.NOTIONAL, "longs": "delivery (CNC) costs",
                        "shorts": "stock future on the cash price, futures costs, no basis / lot / roll",
                        "stops": "20 % catastrophe stop on every variant; exit by the clock",
                        "dividends": "prices are NOT dividend-adjusted"},
        "corporate_action_rule": "events_run.drop_excluded: a trade whose (decision day, exit day] holds an "
                                 "unexplained overnight gap or a demerger / scheme / large-rights ex-day is dropped "
                                 "and counted; reported both ways",
        "report_only": {"rr3": FS.RR3_DAILY,
                        "account": "capital.py: Rs 10 lakh, 1 % risk per trade from the stop distance, config.py "
                                   "limits, signals in time order; stock-future lot ~ Rs 7.5 lakh contract",
                        "excess": "mean / median excess return per trade vs NIFTY 50 and (mid-cap-fair) NIFTY Next 50, "
                                  "NIFTY 100, NIFTY 500 over each holding window (index open of the entry session -> "
                                  f"close of the exit session), {ER.BOOT_N} month-block bootstrap resamples",
                        "placebo": "deals: every (stock, session) group of deal rows moved to a random session of the "
                                   "same stock in the same era; index changes: every (stock, announcement, effective) "
                                   f"group moved to a random session of the same stock and era with the same session "
                                   f"gap; same count, same pipeline, seed {PLACEBO_SEED}",
                        "planted_edge": "synthetic mutual-fund net buys of Rs 50 cr dated the session before each "
                                        "stock's best 2 % of 20-session forward returns (old era, member days) must give "
                                        "fl_inst_buy_h20 a large profit and the same rows delayed 25 sessions must not; "
                                        "synthetic NIFTY 50 additions announced on those sessions (effective 21 "
                                        "sessions later) do the same for fl_idx_add_runup"},
        "adaptations_before_results": [
            "Size filter: the sketch said 'deal value >= Rs 10 cr OR >= 0.5 % of shares'. A bulk deal is by "
            "definition >= 0.5 % of the listed shares, and shares outstanding are not in the data, so the OR clause "
            "would admit every bulk deal: the filter is the client's NET value of the session >= Rs 10 crore.",
            "Netting per client and session (BUY - SELL): bulk deals 2021-26 are dominated by HFT / prop desks that "
            "buy and sell the same stock the same day; netting removes round trips mechanically.",
            "Intra-group transfers removed (flows_strategies.group_transfer_mask): the data check found ~1,000 "
            "2022 block 'institution' buys AND sells that are iShares Mauritius subsidiaries handing shares to "
            "iShares ETFs at equal quantity and price (a restructuring, not a flow); found from counts by month / "
            "client, no return looked at.",
            "Client classes: fixed keyword list (client_classes), written from the list of the largest client NAMES "
            "and counts only. Broker-dealer / ODI / arbitrage names (Societe Generale, BNP Paribas Arbitrage, "
            "'- ODI' accounts, Citigroup Global Markets) are not institutions here; family trusts are OTHER (the "
            "word TRUST alone is not a keyword).",
            "Index pooling (index_pooling): NIFTY 50 and NIFTY 100 changes pooled into additions / deletions, "
            "deduplicated per (stock, effective date); decided before any return because each index alone has far "
            "too few events.",
            "Index rows with month-precision effective dates are dropped everywhere; rows without a real "
            "announcement date are dropped from the two run-up variants only (counts in data_check).",
            "Effective-date variants enter at the open AFTER the effective date (the sketch allowed 'effective-date "
            "close or next open'; the lab fills at next opens).",
            "flowdata.index_changes now also returns announce_source and effective_precision (needed for the "
            "availability rule; a file without the columns counts as 'no real announcement').",
            "Stock futures do not fit a Rs 10 lakh account at 1 % risk: the account rows of the short variants "
            "will show skipped signals (stated before the run)."],
        "data_check_sha256": ER._sha(DATA_CHECK) if os.path.exists(DATA_CHECK) else None,
        "trials": "each of the 8 variants is one trial on 2023-26 and one on 2012-23; the 8 report-only twins are "
                  "recorded as stage 'explore' (they count in N, they cannot be proven)",
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
    print(f"pre-registered {len(FS.SPECS)} variants: {path}\nsha256 {sha}")
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
    for sp in list(FS.SPECS) + list(FS.RR3):
        if pre["strategies"].get(sp.id, {}).get("fingerprint") != sp.fingerprint:
            raise SystemExit(f"{sp.id}: code / parameters changed since the pre-registration")
    if pre["data"]["flow_tables_sha256"] != data_sha():
        raise SystemExit("flow tables changed since the pre-registration")
    if pre["code_sha256"].get("flows_strategies.py") != code_sha()["flows_strategies.py"]:
        raise SystemExit("flows_strategies.py changed since the pre-registration")
    return pre


# ══════════════════════════════════════════════════════════════════════════
# controls: placebo and planted tables (written outside the repository)
# ══════════════════════════════════════════════════════════════════════════
DEAL_COLS = ("date", "kind", "symbol", "lab_symbol", "in_lab", "security", "client", "side", "qty", "price",
             "value_cr", "remarks")
CHANGE_COLS = ("index", "announce_date", "effective_date", "action", "symbol", "lab_symbol", "in_lab", "company",
               "source", "announce_source", "effective_precision", "note")


def _sessions_of(sym: str, cache: dict) -> list:
    if sym not in cache:
        sd = DD.load_symbol(_CTX["cache"], sym)
        cache[sym] = [] if sd is None else [int(x) for x in sd.days]
    return cache[sym]


def _write_csv(path: str, cols, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    op = gzip.open(path, "wt", newline="") if path.endswith(".gz") else open(path, "w", newline="")
    with op as f:
        w = csv.DictWriter(f, fieldnames=list(cols), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def placebo_tables(dst: str, src: str | None = None, seed: int = PLACEBO_SEED) -> dict:
    """Same stocks, random dates, same count (see prereg report_only.placebo)."""
    src = src or F.FLOW_DIR
    rng = random.Random(seed)
    cache: dict = {}
    cal = list(_CTX["cal"])
    pos = {d: i for i, d in enumerate(cal)}
    with gzip.open(os.path.join(src, "deals", "deals.csv.gz"), "rt", newline="") as f:
        deals = list(csv.DictReader(f))
    groups: dict = {}
    for r in deals:
        groups.setdefault((r["lab_symbol"], int(r["date"])), []).append(r)
    out = []
    for (sym, day), rr in sorted(groups.items()):
        dd = [x for x in _sessions_of(sym, cache) if _era_of(x) == _era_of(day)]
        if not dd:
            continue
        nd = rng.choice(dd)
        for r in rr:
            out.append(dict(r, date=nd))
    out.sort(key=lambda r: (int(r["date"]), r["lab_symbol"]))
    _write_csv(os.path.join(dst, "deals", "deals.csv.gz"), DEAL_COLS, out)
    ch = list(csv.DictReader(open(os.path.join(src, "index_membership", "index_changes.csv"))))
    cg: dict = {}
    for r in ch:
        cg.setdefault((r["lab_symbol"], int(r["announce_date"]), int(r["effective_date"])), []).append(r)
    och = []
    for (sym, a, e), rr in sorted(cg.items()):
        ia, ie = E.session_index(a), E.session_index(e)
        gap = max(1, ie - ia)
        dd = [x for x in _sessions_of(sym, cache) if _era_of(x) == _era_of(e) and x in pos and pos[x] + gap < len(cal)]
        if not dd:
            continue
        na = rng.choice(dd)
        ne = cal[pos[na] + gap]
        for r in rr:
            och.append(dict(r, announce_date=na, effective_date=ne))
    _write_csv(os.path.join(dst, "index_membership", "index_changes.csv"), CHANGE_COLS, och)
    return {"deal_rows": len(out), "deal_groups": len(groups), "index_rows": len(och)}


def _best_days(top: float = 0.02, hold: int = 20):
    """Yield (symbol, i, dates) for each stock's best `top` share of `hold`-session forward returns (next open ->
    close `hold` sessions on), old era, member days."""
    for sym in _CTX["symbols"]:
        sd = DD.load_symbol(_CTX["cache"], sym)
        if sd is None:
            continue
        d = sd.daily()
        n = len(d["c"])
        if n < hold + 60:
            continue
        fwd = np.full(n, np.nan)
        k = np.arange(0, n - hold - 1)
        fwd[k] = d["c"][k + hold] / d["o"][k + 1] - 1.0
        ok = np.isfinite(fwd) & sd.member[:n] & (d["date"] <= PO.ERA[1])
        if ok.sum() < 100:
            continue
        thr = np.nanquantile(fwd[ok], 1.0 - top)
        for i in np.flatnonzero(ok & (fwd >= thr)):
            yield sym, int(i), d["date"]


def planted_tables(dst: str, delay: int = 0) -> dict:
    deals, ch = [], []
    cal = list(_CTX["cal"])
    pos = {d: i for i, d in enumerate(cal)}
    for sym, i, dates in _best_days():
        j = min(i + delay, len(dates) - 1)
        day = int(dates[j])
        deals.append({"date": day, "kind": "block", "symbol": sym, "lab_symbol": sym, "in_lab": 0, "security": "",
                      "client": "PLANTED MUTUAL FUND", "side": "BUY", "qty": 1, "price": 1, "value_cr": 50.0,
                      "remarks": "planted"})
        if day in pos and pos[day] + 21 < len(cal):
            ch.append({"index": "NIFTY50", "announce_date": day, "effective_date": cal[pos[day] + 21], "action": "IN",
                       "symbol": sym, "lab_symbol": sym, "in_lab": 0, "company": "", "source": "planted",
                       "announce_source": "press", "effective_precision": "day", "note": "planted"})
    deals.sort(key=lambda r: (r["date"], r["lab_symbol"]))
    ch.sort(key=lambda r: (r["announce_date"], r["lab_symbol"]))
    _write_csv(os.path.join(dst, "deals", "deals.csv.gz"), DEAL_COLS, deals)
    _write_csv(os.path.join(dst, "index_membership", "index_changes.csv"), CHANGE_COLS, ch)
    return {"deals": len(deals), "index_rows": len(ch)}


# ══════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════
def _stats_row(s: dict) -> dict:
    return {"trades": s["trades"], "pf": s["pf"], "net": s["net"], "win_pct": s.get("win_pct")}


def _pool(jobs: int):
    import multiprocessing as mp
    E.set_calendar(_CTX["cal"])
    return mp.get_context("fork").Pool(jobs)


def run(out: str = OUT, jobs: int = 4, trials_ledger: str | None = None, log=print) -> dict:
    from . import trials as TR
    pre = load_prereg()
    D.assert_outside_repo(out)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    setup()
    bench = load_benchmarks()
    cal3 = ER.lab_calendar()
    cal_all = list(_CTX["cal"])
    cal_old = [d for d in cal_all if PO.ERA[0] <= d <= PO.ERA[1]]
    split = P.split_sessions(cal3)
    DISC = split.discovery
    specs = list(FS.DAILY)
    report: dict = {"prereg_sha256": open(os.path.splitext(PREREG)[0] + ".sha256").read().strip(),
                    "cost_model": COST_MODEL, "split": {"h1": split.h1, "h2": split.h2, "final": split.final},
                    "benchmarks": {b: [int(s["date"][0]), int(s["date"][-1]), int(len(s["date"]))]
                                   for b, s in bench.items()},
                    "variants": {}, "controls": {}}
    V = report["variants"]
    tp = lambda sid, era: os.path.join(out, "trades", f"{sid}_{era}.csv")
    ca_new: dict = {}
    ca_old: dict = {}
    with _pool(jobs) as pool:
        res_d, tr_d = P.evaluate(specs, cal3, make_daily_simulator(pool, SLIP_BPS, stats=ca_new),
                                 n_symbols=len(_CTX["symbols"]),
                                 ledger_path=os.path.join(_CTX["cache"], "final_ledger.json"),
                                 dataset="events_daily", cost_model=COST_MODEL)
        log(f"2023-26: {[(r['id'], r['verdict']) for r in res_d]}")
        sens_d = make_daily_simulator(pool, SENS_BPS)(specs, DISC[0], DISC[1])
        keep_d = make_daily_simulator(pool, SLIP_BPS, keep_ca=True)(specs, DISC[0], DISC[1])
        for r in res_d:
            sp, sc = r["spec"], r["screen"]
            tr = tr_d[r["id"]]
            disc = [t for t in tr if DISC[0] <= t[2] <= DISC[1]]
            fin = [t for t in tr if split.final[0] <= t[2] <= split.final[1]]
            rngs = {"disc": DISC}
            if fin:
                rngs["final"] = split.final
            V[r["id"]] = {"rule": sp.desc, "kind": sp.kind, "new": {
                "verdict": r["verdict"], "why": r.get("why", ""), "h1": _stats_row(sc["h1"]),
                "h2": _stats_row(sc["h2"]), "disc": _stats_row(sc["disc"]), "top_symbol": sc["top_symbol"],
                "top_share": None if sc["top_share"] != sc["top_share"] else round(sc["top_share"], 3),
                "final": r.get("final"), "final_note": r.get("final_note", ""), "skipped": r.get("skipped", 0),
                "ca_dropped": ca_new.get(r["id"], 0), "disc_5bps": _stats_row(P.stats(sens_d[r["id"]]["trades"])),
                "disc_ca_kept": _stats_row(P.stats(keep_d[r["id"]]["trades"])),
                "excess": excess(disc + fin, bench, rngs),
                "_st": TR.series_stats(ER._tdicts(tr), cal3, DISC[0], DISC[1])}}
            ER._write_trades(tp(r["id"], "2023-26"), tr)
        res_o = PO.evaluate(specs, make_daily_simulator(pool, SLIP_BPS, stats=ca_old),
                            make_daily_simulator(pool, SENS_BPS), cost_model=COST_MODEL, ledger_path=OOS_LEDGER,
                            prereg=pre, dataset="events_daily")
        log(f"2012-23: {[(r['id'], r['verdict']) for r in res_o]}")
        for r in res_o:
            tr = r.get("trades") or ER._read_trades(tp(r["id"], "2012-23"))
            if r.get("trades"):
                ER._write_trades(tp(r["id"], "2012-23"), tr)
            V[r["id"]]["old"] = {"era": "2012-23", "verdict": r["verdict"], "why": r.get("why", ""),
                                 "note": r.get("note", ""), "h1": r.get("h1"), "h2": r.get("h2"), "all": r.get("era"),
                                 "top_symbol": r.get("top_symbol"), "top_share": r.get("top_share"),
                                 "sens_5bps": r.get("sens"), "ca_dropped": ca_old.get(r["id"]),
                                 "excess": excess(tr, bench, {"h1": PO.H1, "h2": PO.H2, "era": PO.ERA}),
                                 "_st": TR.series_stats(ER._tdicts(tr), cal_old, PO.ERA[0], PO.ERA[1])}
        rr_new = make_daily_simulator(pool, SLIP_BPS)(FS.RR3, DISC[0], DISC[1])
        rr_old = make_daily_simulator(pool, SLIP_BPS)(FS.RR3, PO.ERA[0], PO.ERA[1])
    log("main runs done")

    # ── trials ledger ────────────────────────────────────────────────────────────────────────────────────
    expl = []
    for tw in FS.RR3:
        st = TR.series_stats(ER._tdicts(rr_new[tw.id]["trades"]), cal3, DISC[0], DISC[1])
        expl.append({"strategy_id": tw.id, "family": TR.family_of(tw.id), "group": tw.family,
                     "params_fp": tw.fingerprint, "era": "2023-26", "protocol": "report-only", "stage": "explore",
                     "dataset": "events_daily", "period": "%d-%d" % DISC, "cost_model": COST_MODEL,
                     **TR._row_stats(st), "verdict": "REPORT-ONLY", "source": out + "#rr3", "note": "1:3 bracket twin"})
    if trials_ledger:
        n1 = TR.record_evaluation(res_d, tr_d, dataset="events_daily", cost_model=COST_MODEL,
                                  ledger_path=trials_ledger, calendar=cal3, source=out)
        n3 = TR.TrialsLedger(trials_ledger).append(
            ER.ledger_rows_old([dict(r, trades=r.get("trades") or ER._read_trades(tp(r["id"], "2012-23")))
                                for r in res_o], cal_old, source="flows/oos_ledger.json", cost_model=COST_MODEL)
            + expl)
        log(f"trials ledger: {n1} rows (2023-26) + {n3} rows (old era, report-only twins)")

    # ── report-only: 1:3 twins and the Rs 10 lakh account ─────────────────────────────────────────────────
    for sp in FS.SPECS:
        v = V[sp.id]
        tw = FS.rr3_twin(sp)
        a, b = rr_new[tw.id]["trades"], rr_old[tw.id]["trades"]
        old_tr = ER._read_trades(tp(sp.id, "2012-23"))
        v["rr3"] = {"disc_2023_26": _stats_row(P.stats(a)), "old_era": _stats_row(P.stats(b)),
                    "old_h1": _stats_row(P.stats([t for t in b if t[2] <= PO.H1[1]])),
                    "old_h2": _stats_row(P.stats([t for t in b if t[2] > PO.H1[1]])),
                    "account_disc_2023_26": ER.account(a, tw, cal3, DISC[0], DISC[1]),
                    "account_old_era": ER.account(b, tw, cal_old, PO.ERA[0], PO.ERA[1])}
        v["account"] = {"disc_2023_26": ER.account(tr_d[sp.id], sp, cal3, DISC[0], DISC[1]),
                        "old_era": ER.account(old_tr, sp, cal_old, PO.ERA[0], PO.ERA[1])}

    # ── controls ──────────────────────────────────────────────────────────────────────────────────────────
    real_dir = F.FLOW_DIR
    pdir = os.path.join(out, "placebo_tables")
    report["controls"]["placebo_tables"] = placebo_tables(pdir, real_dir)
    setup(flow_dir=pdir)
    with _pool(jobs) as pool:
        pn = make_daily_simulator(pool, SLIP_BPS)(specs, DISC[0], DISC[1])
        po = make_daily_simulator(pool, SLIP_BPS)(specs, PO.ERA[0], PO.ERA[1])
    report["controls"]["placebo"] = {
        sp.id: {"disc_2023_26": _stats_row(P.stats(pn[sp.id]["trades"])),
                "old_era": _stats_row(P.stats(po[sp.id]["trades"])),
                "excess_old_era": {b: ER.block_bootstrap(ER.abnormal_returns(po[sp.id]["trades"], s))
                                   for b, s in bench.items() if b in ("NIFTY50", "NIFTY500")}}
        for sp in FS.SPECS}
    log("placebo done")
    plant = {}
    for name, delay in (("planted", 0), ("planted_delayed_25", 25)):
        dd = os.path.join(out, name + "_tables")
        setup(flow_dir=real_dir)
        n = planted_tables(dd, delay=delay)
        setup(flow_dir=dd)
        with _pool(jobs) as pool:
            r = make_daily_simulator(pool, SLIP_BPS)([FS.BY_ID["fl_inst_buy_h20"], FS.BY_ID["fl_idx_add_runup"]],
                                                     PO.ERA[0], PO.ERA[1])
        plant[name] = {"rows": n, "fl_inst_buy_h20": _stats_row(P.stats(r["fl_inst_buy_h20"]["trades"])),
                       "fl_idx_add_runup": _stats_row(P.stats(r["fl_idx_add_runup"]["trades"]))}
    report["controls"]["planted_edge"] = plant
    setup(flow_dir=real_dir)
    log("planted done")

    # ── DSR / PSR ─────────────────────────────────────────────────────────────────────────────────────────
    rows = TR.TrialsLedger(trials_ledger).read() if trials_ledger else []
    n_full = TR.count_trials(rows, "2023-26") if rows else None
    n_old = TR.count_trials(rows, "2012-23") if rows else None
    v_full = TR.var_sr_robust(rows, "2023-26", "screen", "bars3y")[0] if rows else float("nan")
    srs = [v["new"]["_st"]["sr_daily"] for v in V.values()]
    srs = [x for x in srs if x == x]
    v_fair = float(np.var(srs, ddof=1)) if len(srs) > 1 else float("nan")
    vv = v_full if v_full == v_full else v_fair
    nf = len(FS.SPECS)
    report["trials"] = {"n_2023_26": n_full, "n_2012_23": n_old, "n_fair": nf, "v_sr_robust_2023_26": v_full,
                        "v_sr_study": v_fair}
    for sid, v in V.items():
        st = v["new"].pop("_st")
        v["new"]["dsr_disc"] = ER.dsr_block(st, n_full or nf, vv, nf, v_fair)
        v["new"]["mean_gross"], v["new"]["mean_cost"] = st["mean_gross"], st["mean_cost"]
        v["new"]["haircut_per_trade"] = None if st["mean_gross"] != st["mean_gross"] else round(
            0.5 * st["mean_gross"] - st["mean_cost"], 2)
        so = v.get("old", {}).pop("_st", None)
        if so and so.get("sr_daily") == so.get("sr_daily"):
            v["old"]["dsr"] = ER.dsr_block(so, n_old or nf, vv, nf, v_fair)
        d = v["new"]["dsr_disc"].get("dsr_full")
        v["proven"] = bool(v["new"]["verdict"] == "CONFIRMED" and v.get("old", {}).get("verdict") == "OOS-CONFIRMED"
                           and d is not None and d >= DSR_MIN)
    report["n_proven"] = sum(1 for v in V.values() if v["proven"])
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=ER._js)
    log(f"report: {os.path.join(out, 'report.json')}; proven: {report['n_proven']} of {len(V)}")
    return report


def smoke() -> int:
    """Plumbing only: every variant and twin runs without error on a few symbols and a short window. Prints NO
    P&L and no trade counts."""
    setup()
    syms = _CTX["symbols"]
    _CTX["symbols"] = syms[::40]
    E.set_calendar(_CTX["cal"])
    r = make_daily_simulator(None)(list(FS.SPECS) + list(FS.RR3), 20190101, 20190630)
    bad = {i: v["error"] for i, v in r.items() if v["error"]}
    _CTX["symbols"] = syms
    print("smoke:", "OK (no errors)" if not bad else bad)
    return 1 if bad else 0


def backup_ledger(path: str) -> str:
    dst = path + ".bak_flows_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, dst)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.flows_run")
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
