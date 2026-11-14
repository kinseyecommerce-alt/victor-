"""
stratlab/events_run.py — run the pre-registered EVENT study (events_strategies.py) ONCE and judge it.

    cd algotrader_v4
    python -m stratlab.events_run --preregister          # freeze ids, fingerprints, gates, data hashes (no results)
    python -m stratlab.events_run --run [--jobs 4]       # the one run: both eras, controls, account, trials ledger
    python -m stratlab.events_run --smoke                # synthetic-free plumbing check: errors only, no P&L

Judgement (nothing here is new; the lab's own functions are called unchanged):
  2023-26   protocol.evaluate: DISCOVERY / FINAL split on the lab's 3-year session calendar, screen PF >= 1.15,
            net > 0 in both DISCOVERY halves, >= 60 trades (positional) / 150 (intraday), top symbol <= 35 %;
            FINAL (PF >= 1.05, net > 0) is simulated ONCE and only for screen passers (one-look ledger).
  2012-23   protocol_oos.evaluate (daily variants; H1 2012-17 / H2 2018-23, same gates per half, one-look ledger).
            Insider disclosures exist from 2015-11-27 only, so the insider variants' H1 is 2015-12 .. 2017-12.
  2016-23   protocol_min_oos.judge for the intraday variant (H1 2016-19 / H2 2020-23; today's NIFTY-100 names).
  Every variant gets the old-era look whatever its 2023-26 result (pre-registered).
  "proven" = CONFIRMED on 2023-26 FINAL AND OOS-CONFIRMED on the old era AND DSR >= 0.95.

Data: daily variants run on events_daily's cache (point-in-time top-500 universe, both eras, official NSE bars);
the intraday variant on the lab's minute bars (2023-26: 97 of today's NIFTY-100; 2016-23: today's NIFTY-100) —
both survivorship-biased. Fills / costs: sim.sim_positional and sim.sim_session unchanged (next-open fill, 2 bps
slippage per side, costs.py: CNC for longs, futures costs for shorts on the cash price, MIS intraday).
Corporate-action rule: a trade whose (decision day, exit day] contains an unexplained overnight gap or a demerger /
scheme / large-rights ex-day of its symbol is dropped (counted in `skipped`); `--run` also reports them kept.

Report-only extras (never part of a verdict): 1:3 bracket twins, Rs 10 lakh account (capital.py), event-study
abnormal returns vs NIFTY with a month-block bootstrap, a placebo with shuffled event dates and a planted-edge check.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

from . import data as D
from . import data_daily as DD
from . import events_daily as ED
from . import events_data as E
from . import events_strategies as ES
from . import protocol as P
from . import protocol_oos as PO
from . import sim

SCRATCH = DD._SCRATCH
EVENTS_DIR = ED.EVENTS_DIR
OUT = os.path.join(EVENTS_DIR, "run1")
PREREG = os.path.join(EVENTS_DIR, "PREREGISTERED_EVENTS.json")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "events_v1.json")
OOS_LEDGER = os.path.join(EVENTS_DIR, "oos_ledger.json")
MIN_LEDGER = os.path.join(EVENTS_DIR, "min_oos_ledger.json")
FILL_MODEL = "ev1"
COST_MODEL = f"{sim.COST_MODEL}/{FILL_MODEL}"
SLIP_BPS, SENS_BPS = sim.DEFAULT_SLIPPAGE_BPS, sim.SENSITIVITY_SLIPPAGE_BPS
DSR_MIN = 0.95
BOOT_N, BOOT_SEED = 2000, 20261114
PLACEBO_SEED = 20261114
FUT_CONTRACT_VALUE = 750_000.0          # report-only: approximate stock-future contract value for the account sim
_NET, _SYM, _DATE = 12, 1, 2
_CTX: dict = {}


# ══════════════════════════════════════════════════════════════════════════
# set-up
# ══════════════════════════════════════════════════════════════════════════
def load_fo(events_dir: str = EVENTS_DIR) -> dict:
    """{yyyymmdd: frozenset(current symbols with a stock future traded that day)} from both F&O bhavcopy files."""
    cache = os.path.join(events_dir, "fo_by_day.json.gz")
    if os.path.exists(cache):
        with gzip.open(cache, "rt") as f:
            return {int(k): frozenset(v) for k, v in json.load(f).items()}
    import pandas as pd
    from .events_parse import load_changes, canon
    ch = load_changes(os.path.join(events_dir, "raw", "ref", "symbolchange.csv"))
    out: dict = {}
    for p in (os.path.join(DD.OLD_DIR, "fut.csv.gz"), os.path.join(ED.NSE_DIR, "fut.csv.gz")):
        df = pd.read_csv(p, usecols=["date", "symbol", "instrument"], dtype=str)
        df = df[df["instrument"].isin(["FUTSTK", "STF"])].drop_duplicates(["date", "symbol"])
        df["day"] = df["date"].str.replace("-", "").astype(int)
        for day, g in df.groupby("day"):
            out.setdefault(int(day), set()).update(canon(s, int(day), ch) for s in g["symbol"])
    with gzip.open(cache, "wt") as f:
        json.dump({str(k): sorted(v) for k, v in sorted(out.items())}, f)
    return {k: frozenset(v) for k, v in out.items()}


def setup(cache: str = ED.CACHE, events_dir: str | None = None, fo: dict | None = None) -> dict:
    """Daily cache, calendar, exclusions, F&O list, index series. Must run before forking workers."""
    m = DD.manifest(cache)
    if events_dir is not None:
        E.use(events_dir)
    E.set_calendar(m["calendar"])
    E.set_exclusions([tuple(x) for x in m.get("bad_gaps", [])] + [tuple(x) for x in m.get("ca_ambiguous", [])])
    E.set_fo(load_fo() if fo is None else fo)
    for name in ("insider", "results", "boardmeet"):
        E._load(name)
    _CTX.update({"cache": cache, "cal": m["calendar"], "nifty": DD.load_daily_aux(cache, "_NIFTY50_day"),
                 "vix": DD.load_daily_aux(cache, "_VIX_day"),
                 "symbols": sorted(s for s, v in m["symbols"].items() if v["member_days"] > 0
                                   and not s.startswith("_"))})
    return _CTX


def lab_calendar() -> list:
    from . import trials
    return trials.calendar_3y(SCRATCH)


class _Guard:
    """fn(d, p) under the look-ahead cutoff of its decision: no event row broadcast at / after the next session's
    order cut-off can be read, whatever the strategy asks for."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "guarded")

    def __call__(self, d, p):
        day = int(d.date[-1])
        x = E.next_session(day)
        st = E.stamp(x, E.ORDER_CUTOFF) if x is not None else E.stamp(day, E.CLOSE)
        with E.cutoff(st):
            return self.fn(d, p)


class _GuardSession:
    """fn(s, p) of an intraday strategy under the cutoff of today's order cut-off (09:00)."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "guarded")

    def __call__(self, s, p):
        with E.cutoff(E.stamp(int(s.date), E.ORDER_CUTOFF)):
            return self.fn(s, p)


def guarded(spec, sd):
    g = DD.gate(spec, sd)
    return dataclasses.replace(g, fn=_Guard(g.fn))


def drop_excluded(trades: list) -> tuple[list, int]:
    """The corporate-action rule: (kept trades, number dropped)."""
    keep, n = [], 0
    for t in trades:
        e = int(t[_DATE])
        x = int(t[6]) if t[6] else e
        d0 = E.prev_session(e) or e
        if E.ca_excluded(t[_SYM], d0, max(x, e)):
            n += 1
        else:
            keep.append(t)
    return keep, n


def _specs_by_id() -> dict:
    return {sp.id: sp for sp in list(ES.SPECS) + list(ES.RR3)}


def _work_daily(args):
    symbol, ids, lo, hi, slip, keep_ca = args
    by = _specs_by_id()
    out = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None, "ca_dropped": 0} for i in ids}
    sd = DD.load_symbol(_CTX["cache"], symbol)
    if sd is None:
        return out
    for i in ids:
        sp = by[i]
        try:
            tr, sk = sim.sim_positional(guarded(sp, sd), sd, lo, hi, _CTX["nifty"], _CTX["vix"], slip)
            if not keep_ca:
                tr, n = drop_excluded(tr)
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


# ── intraday (minute bars) ──────────────────────────────────────────────────
def _work_intraday(args):
    cache, symbol, ids, lo, hi, slip, audit = args
    by = _specs_by_id()
    specs = [dataclasses.replace(by[i], fn=_GuardSession(by[i].fn)) for i in ids]
    sd = D.load_symbol(cache, symbol)
    out = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None} for i in ids}
    if sd is None or not E._load("results").get(symbol):
        return out                                    # no results rows for this symbol: no signal possible
    for y in range(lo // 10000, hi // 10000 + 1):     # one year at a time (memory), identical trades
        a, b = max(lo, y * 10000 + 101), min(hi, y * 10000 + 1231)
        if a > b:
            continue
        res = sim.run_symbol(cache, symbol, specs, a, b, audit=audit, sd=sd, slippage=slip)
        for i, r in res.items():
            o = out[i]
            o["trades"].extend(r["trades"])
            o["skipped"] += r["skipped"]
            o["lookahead"] = o["lookahead"] or r["lookahead"]
            o["error"] = o["error"] or r["error"]
    return out


def make_intraday_simulator(pool, cache: str, symbols: list, slip_bps: float = SLIP_BPS, audit: bool = True):
    def simulate(specs, lo, hi):
        ids = [sp.id for sp in specs]
        merged = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None} for i in ids}
        jobs = [(cache, s, ids, lo, hi, slip_bps / 1e4, audit) for s in symbols]
        for res in (pool.imap_unordered(_work_intraday, jobs) if pool is not None else map(_work_intraday, jobs)):
            for i, r in res.items():
                m = merged[i]
                m["trades"].extend(r["trades"])
                m["skipped"] += r["skipped"]
                m["lookahead"] = m["lookahead"] or r["lookahead"]
                m["error"] = m["error"] or r["error"]
        for m in merged.values():
            m["trades"].sort(key=lambda t: (t[2], t[1]))
        return merged
    return simulate


def lab_minute_cache() -> tuple[str, list]:
    cache = D.default_cache_dir(os.path.join(SCRATCH, "bars3y"))
    syms = sorted(f[:-4] for f in os.listdir(cache) if f.endswith(".npz") and not f.startswith("_"))
    return cache, syms


# ══════════════════════════════════════════════════════════════════════════
# pre-registration
# ══════════════════════════════════════════════════════════════════════════
CODE_FILES = ("events_run.py", "events_strategies.py", "events_data.py", "events_daily.py", "events_parse.py",
              "events_fetch.py", "sim.py", "protocol.py", "protocol_oos.py", "protocol_min_oos.py", "capital.py",
              "data_daily.py")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: _sha(os.path.join(here, f)) for f in CODE_FILES}


def data_sha(events_dir: str = EVENTS_DIR) -> dict:
    out = {}
    for f in ("insider.csv.gz", "results.csv.gz", "boardmeet.csv.gz"):
        p = os.path.join(events_dir, f)
        out[f] = _sha(p) if os.path.exists(p) else None
    return out


def prereg_doc() -> dict:
    from . import protocol_min_oos as PM
    m = DD.manifest(ED.CACHE)
    return {
        "name": "events_v1", "written": time.strftime("%Y-%m-%d %H:%M"),
        "question": "Do promoter / insider open-market purchases and earnings-announcement moves give a tradable "
                    "edge on NSE stocks after costs, in two eras?",
        "variants": [{"id": sp.id, "kind": sp.kind, "horizon": sp.horizon, "family": sp.family, "rule": sp.desc,
                      "params": sp.params, "fingerprint": sp.fingerprint} for sp in ES.SPECS],
        "report_only_twins": [{"id": sp.id, "params": sp.params, "fingerprint": sp.fingerprint} for sp in ES.RR3],
        "strategies": {sp.id: {"fingerprint": sp.fingerprint} for sp in list(ES.SPECS) + list(ES.RR3)},
        "n_variants": len(ES.SPECS),
        "availability_rule": {
            "insider": "a disclosure is usable for an order only if its NSE broadcast time is before 09:00 IST of "
                       "the entry session (transaction / intimation dates are never used for timing)",
            "results": "first_ts = min(earliest structured results filing broadcast, earliest 'Financial Result "
                       "Updates' announcement within 48 h before it, earliest 'Outcome of Board Meeting' announcement "
                       "within 24 h before it); reaction session = the first session whose 15:30 close is after "
                       "first_ts; entry at the following session's open",
            "board_meetings": "a notice is usable if broadcast before 09:00 IST of the entry session; the session "
                              "calendar (exchange holidays) is treated as public",
            "intraday": "results first public at / after the previous session's 15:30 and before 09:00 today",
            "order_cutoff": E.ORDER_CUTOFF, "close": E.CLOSE},
        "data": {"sources": ["NSE corporates-pit (insider trading, Reg 7(2))", "NSE corporates-financial-results",
                             "NSE corporate-announcements (Financial Result Updates, Outcome of Board Meeting)",
                             "NSE corporate-board-meetings", "NSE corporates-corporateActions", "NSE symbolchange.csv"],
                 "event_tables_sha256": data_sha(), "coverage": E.coverage(),
                 "daily_cache": {"version": m.get("events_daily"), "sessions": len(m["calendar"]),
                                 "first": m["calendar"][0], "last": m["calendar"][-1], "top_n": m.get("top_n"),
                                 "symbols_ever_members": m.get("universe_symbols"),
                                 "unexplained_gaps": len(m.get("bad_gaps", [])),
                                 "scheme_ex_days": len(m.get("ca_ambiguous", []))}},
        "universe": {
            "daily_variants": "point-in-time: each month's top 500 NSE equity shares by 60-session average turnover "
                              "(computed from data before the month); a position may be opened only on a member day; "
                              "short variants additionally need a stock future traded on the decision day",
            "intraday_variant": "lab minute bars: 2023-26 = 97 of today's NIFTY-100 names, 2016-23 = today's "
                                "NIFTY-100 names (both SURVIVORSHIP-BIASED)"},
        "eras": {"2023-26": {"calendar": "the lab's 3-year session calendar (trials.calendar_3y)",
                             "protocol": "protocol.evaluate, DISCOVERY 70 % (H1 / H2) + FINAL 30 %, FINAL once and "
                                         "only for screen passers",
                             "ledgers": ["<events_daily cache>/final_ledger.json", "<bars3y cache>/final_ledger.json"]},
                 "2012-23": {"era": list(PO.ERA), "h1": list(PO.H1), "h2": list(PO.H2),
                             "protocol": "protocol_oos.evaluate (one look, ledger " + os.path.basename(OOS_LEDGER) + ")",
                             "note": "insider disclosures start 2015-11-27: insider variants' H1 = 2015-12..2017-12"},
                 "2016-23": {"era": list(PM.ERA), "h1": list(PM.H1), "h2": list(PM.H2),
                             "protocol": "protocol_min_oos.judge on the intraday variant, dirty-link rule applied"},
                 "old_era_look": "EVERY variant gets its old-era judgement, whatever its 2023-26 result"},
        "gates": {"screen": {"min_pf": P.SCREEN_MIN_PF, "min_trades": P.SCREEN_MIN_TRADES,
                             "max_symbol_share": P.SCREEN_MAX_SYMBOL_SHARE, "net_gt_0": ["H1", "H2"]},
                  "final": {"min_pf": P.FINAL_MIN_PF, "net_gt_0": True},
                  "oos": PO.GATES, "min_oos": PM.GATES, "dsr_min": DSR_MIN,
                  "proven": "CONFIRMED on 2023-26 FINAL and OOS-CONFIRMED on the old era and DSR >= 0.95 with N = "
                            "all 2023-26 trials in the ledger (the fair N = this study's variants is also reported)"},
        "gates_fingerprint": PO.GATES_FINGERPRINT,
        "fills_costs": {"cost_model": COST_MODEL, "slippage_bps": SLIP_BPS, "sensitivity_bps": SENS_BPS,
                        "notional": sim.NOTIONAL,
                        "longs": "delivery (CNC) costs", "shorts": "stock future on the cash price, futures costs, "
                        "no basis / lot / roll", "dividends": "prices are NOT dividend-adjusted (understates longs, "
                        "overstates shorts by the dividends paid during the hold)"},
        "corporate_action_rule": "a trade whose (decision day, exit day] contains an unexplained overnight gap "
                                 "(open / previous close < 0.6 or > 1.67) or a demerger / scheme / large-rights ex-day "
                                 "of its symbol is dropped and counted; reported both ways",
        "report_only": {"rr3": {"daily": ES.RR3_DAILY, "intraday": list(ES.RR3_INTRADAY)},
                        "account": "capital.py: Rs 10 lakh, 1 % risk per trade from the stop distance, config.py "
                                   "position limits, signals in time order; stock-future lot ~ Rs 7.5 lakh contract "
                                   "(approximate)",
                        "event_study": f"mean / median abnormal return vs NIFTY over each holding window, "
                                       f"{BOOT_N} month-block bootstrap resamples, seed {BOOT_SEED}",
                        "placebo": f"every event moved to a random session of the same symbol and era (same count, "
                                   f"time of day kept, seed {PLACEBO_SEED}), same pipeline",
                        "planted_edge": "synthetic promoter purchases broadcast the evening before each symbol's "
                                        "best 2 % of 20-session forward returns must show a large profit; the same "
                                        "events delayed 25 sessions must not"},
        "adaptations_before_results": [
            "Universe of the daily variants: point-in-time top 500 by turnover in BOTH eras (events_daily.py), not the "
            "lab's 97 current NIFTY-100 names / old-era top 100. Decided from counts of disclosure ROWS (promoter "
            "market purchases >= Rs 1 cr in the top 100: a few dozen a year) and because the published tests use "
            "NIFTY 500; no return was looked at.",
            "Results timestamp: first version (announcement within 48 h / 24 h before the structured filing) put the "
            "reaction on the wrong day in 2013-2018 (UNSIGNED check: median |abnormal return| and volume on the "
            "assigned day were at the level of ordinary days; 4-20 % agreement with the board-meeting feed). Replaced "
            "by the rule in availability_rule.results; after it every era peaks on the assigned day (|AR| 2.7-2.9 % "
            "vs 1.4-1.5 %, volume 1.7-2.4 x) and 76-99 % of dates agree with the board-meeting feed. Only unsigned "
            "magnitudes and counts were used.",
            "Results from 2025-04 come from NSE's 'Integrated Filing - Financials' feed (the older results feed "
            "stops there).",
            "Earnings-gap variant: Chan's published rule (gap >= 0.5 x 90-day std of overnight returns, both "
            "directions, exit at the close) on minute bars, instead of the sketched 3 % gap + opening-range rule "
            "(too few events in 97 names for the 150-trade gate).",
            "Insider disclosures exist on NSE from 2015-11-27 (SEBI PIT 2015): no insider test before that.",
            "Stock futures do not fit a Rs 10 lakh account at 1 % risk (one lot risks far more): the account rows "
            "of the short variants will show skipped signals; stated here before the run."],
        "data_check_sha256": _sha(os.path.join(EVENTS_DIR, "data_check.json"))
        if os.path.exists(os.path.join(EVENTS_DIR, "data_check.json")) else None,
        "trials": "each of the 9 variants is one trial on 2023-26 and one on its old era; the 9 report-only twins "
                  "are recorded as stage 'explore' (they count in N, they cannot be proven)",
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
    print(f"pre-registered {len(ES.SPECS)} variants: {path}\nsha256 {sha}")
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
    for sp in list(ES.SPECS) + list(ES.RR3):
        if pre["strategies"].get(sp.id, {}).get("fingerprint") != sp.fingerprint:
            raise SystemExit(f"{sp.id}: code / parameters changed since the pre-registration")
    if pre["data"]["event_tables_sha256"] != data_sha():
        raise SystemExit("event tables changed since the pre-registration")
    return pre


# ══════════════════════════════════════════════════════════════════════════
# report-only statistics
# ══════════════════════════════════════════════════════════════════════════
def abnormal_returns(trades: list, nifty: dict) -> list:
    """[(entry yyyymm, abnormal return)] per trade: side x (exit / entry - 1) minus side x NIFTY over the same
    window (NIFTY open of the entry session -> NIFTY close of the exit session)."""
    nd, no, nc = nifty["date"], nifty["o"], nifty["c"]
    out = []
    for t in trades:
        e, x = int(t[_DATE]), int(t[6]) if t[6] else int(t[_DATE])
        i, j = int(np.searchsorted(nd, e)), int(np.searchsorted(nd, x))
        if i >= len(nd) or j >= len(nd) or nd[i] != e or nd[j] != x or not no[i] > 0:
            continue
        side = int(t[3])
        r = side * (float(t[8]) / float(t[5]) - 1.0)
        out.append((e // 100, r - side * (float(nc[j]) / float(no[i]) - 1.0), r))
    return out


def block_bootstrap(ar: list, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Mean / median abnormal return with a 95 % interval from resampling whole entry MONTHS (events in the same
    month share one market and overlap in time: they are not independent)."""
    if not ar:
        return {"n": 0}
    by: dict = {}
    for mth, a, _r in ar:
        by.setdefault(mth, []).append(a)
    keys = sorted(by)
    sums = np.array([sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    allv = np.array([a for _m, a, _r in ar])
    raw = np.array([r for _m, _a, r in ar])
    return {"n": int(len(allv)), "months": len(keys), "mean_ar_pct": round(100 * float(allv.mean()), 3),
            "median_ar_pct": round(100 * float(np.median(allv)), 3),
            "ci95_mean_ar_pct": [round(100 * float(np.percentile(means, 2.5)), 3),
                                 round(100 * float(np.percentile(means, 97.5)), 3)],
            "mean_raw_pct": round(100 * float(raw.mean()), 3), "share_beating_nifty": round(float((allv > 0).mean()), 3)}


def account(trades: list, spec, cal: list, lo: int, hi: int) -> dict:
    """The Rs 10 lakh account (capital.py) fed with a variant's trades in time order."""
    from . import capital as C
    ag = C.agents()[{"swing": "swing", "futures": "futures", "intraday": "intraday"}[spec.kind]]
    inst = {"swing": "cnc", "futures": "fut", "intraday": "mis"}[spec.kind]
    stop_pct = float(spec.params.get("stop_pct", 20.0))
    if spec.id.endswith("_rr3") and spec.horizon == "intraday":
        stop_pct = float(ES.RR3_INTRADAY[1])
    cands = []
    for t in trades:
        e = int(t[_DATE])
        if not lo <= e <= hi:
            continue
        side, fill, xpx = int(t[3]), float(t[5]), float(t[8])
        x = int(t[6]) if t[6] else e
        hm = lambda s, d: (int(s[:2]) * 60 + int(s[3:5])) if isinstance(s, str) and len(s) == 5 else d
        ekey = e * 10000 + hm(t[4], 555)
        xkey = x * 10000 + (hm(t[7], 720) + (1 if t[7] == "15:29" else 0))
        if xkey <= ekey:
            xkey = ekey + 1
        lot = max(1, int(round(FUT_CONTRACT_VALUE / fill))) if inst == "fut" else 1
        cands.append(C.cand(t[_SYM], side, ekey, fill, fill * (1 - side * stop_pct / 100.0), xkey, xpx, str(t[13]),
                            0.0, inst, lot, 1, spec.id))
    res = C.simulate(cands, ag, spec.id)
    mt = C.metrics(res, cal, lo, hi)
    return {k: mt[k] for k in ("trades", "candidates", "net", "return_pct_per_year", "max_dd_rs", "max_dd_pct",
                               "worst_month", "worst_month_rs", "win_pct", "pf", "sharpe", "skipped")}


def placebo_tables(dst: str, seed: int = PLACEBO_SEED) -> None:
    """Copies of the three event tables with every event moved to a random session of the same symbol in the same
    era (2012-23 / 2023-26), time of day kept; board-meeting notices keep their notice-to-meeting distance."""
    import csv as _csv
    import datetime as _dt
    os.makedirs(dst, exist_ok=True)
    rng = random.Random(seed)
    m = DD.manifest(ED.CACHE)
    days_of = {}

    def sessions(sym, new_era):
        k = (sym, new_era)
        if k not in days_of:
            sd = DD.load_symbol(ED.CACHE, sym)
            dd = [] if sd is None else [int(x) for x in sd.days if (int(x) > PO.ERA[1]) == new_era]
            days_of[k] = dd
        return days_of[k]

    def moved(ts: str, sym: str):
        day = int(ts[:10].replace("-", ""))
        dd = sessions(sym, day > PO.ERA[1])
        if not dd:
            return None
        nd = rng.choice(dd)
        return f"{nd // 10000:04d}-{nd // 100 % 100:02d}-{nd % 100:02d}" + ts[10:]

    for name, col in (("insider.csv.gz", "broadcast"), ("results.csv.gz", "first_ts"), ("boardmeet.csv.gz", "notice_ts")):
        with gzip.open(os.path.join(EVENTS_DIR, name), "rt", newline="") as f:
            rd = _csv.DictReader(f)
            cols, rows = rd.fieldnames, list(rd)
        out = []
        for r in rows:
            if r["symbol"] not in m["symbols"] or not r.get(col):
                continue
            new = moved(r[col], r["symbol"])
            if new is None:
                continue
            if name == "boardmeet.csv.gz" and r.get("meeting_date"):
                d0 = _dt.date.fromisoformat(r[col][:10])
                gap = (_dt.date.fromisoformat(r["meeting_date"]) - d0).days
                r["meeting_date"] = (_dt.date.fromisoformat(new[:10]) + _dt.timedelta(days=gap)).isoformat()
            r[col] = new
            out.append(r)
        with gzip.open(os.path.join(dst, name), "wt", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(out)


def planted_tables(dst: str, delay: int = 0, top: float = 0.02, hold: int = 20) -> int:
    """Synthetic promoter purchases (Rs 2 crore, broadcast 18:00) on the session before each symbol's best `top`
    share of `hold`-session forward returns (open of the next session -> close `hold` sessions on), old era, member
    days only. delay > 0 moves every event that many sessions later (the control)."""
    import csv as _csv
    os.makedirs(dst, exist_ok=True)
    m = DD.manifest(ED.CACHE)
    rows = []
    for sym in _CTX["symbols"]:
        sd = DD.load_symbol(ED.CACHE, sym)
        if sd is None:
            continue
        d = sd.daily()
        n = len(d["c"])
        if n < hold + 30:
            continue
        fwd = np.full(n, np.nan)
        k = np.arange(0, n - hold - 1)
        fwd[k] = d["c"][k + hold] / d["o"][k + 1] - 1.0
        ok = np.isfinite(fwd) & sd.member[:n] & (d["date"] <= PO.ERA[1])
        if ok.sum() < 100:
            continue
        thr = np.nanquantile(fwd[ok], 1.0 - top)
        for i in np.flatnonzero(ok & (fwd >= thr)):
            j = min(i + delay, n - 1)
            day = int(d["date"][j])
            rows.append({"key": f"P{sym}{day}", "symbol": sym, "company": "", "person": "planted", "person_cat":
                         "Promoters", "is_promoter": 1, "mode": "Market Purchase", "txn": "Buy", "sec_type":
                         "Equity Shares", "qty": 1, "value": 2e7, "from_date": "", "to_date": "", "intim_date": "",
                         "broadcast": f"{day // 10000:04d}-{day // 100 % 100:02d}-{day % 100:02d} 18:00:00",
                         "exchange": "NSE", "source": "planted"})
    from .events_parse import INS_COLS, RES_COLS, BM_COLS
    for name, cols, rr in (("insider.csv.gz", INS_COLS, rows), ("results.csv.gz", RES_COLS, []),
                           ("boardmeet.csv.gz", BM_COLS, [])):
        with gzip.open(os.path.join(dst, name), "wt", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(cols), lineterminator="\n")
            w.writeheader()
            w.writerows(rr)
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════
# trials ledger + DSR
# ══════════════════════════════════════════════════════════════════════════
def _tdicts(trades):
    from . import trials as TR
    return [{"entry": int(t[2]), "exit": TR._d8(t[6]) or int(t[2]), "net": float(t[12]), "gross": float(t[10]),
             "cost": float(t[11])} for t in trades]


def ledger_rows_old(res: list, cal_old: list, era=PO.ERA, era_tag="2012-23", protocol="2012-23 OOS",
                    dataset="events_daily", cost_model=COST_MODEL, source="events/oos_ledger.json") -> list:
    from . import trials as TR
    rows = []
    for r in res:
        if r.get("verdict") == "NOT JUDGED":
            continue
        sp = r["spec"]
        st = TR.series_stats(_tdicts(r.get("trades", [])), cal_old, era[0], era[1])
        rows.append({"strategy_id": r["id"], "family": TR.family_of(r["id"]), "group": sp.family,
                     "params_fp": sp.fingerprint, "era": era_tag, "protocol": protocol, "stage": "oos",
                     "dataset": dataset, "period": "%d-%d" % tuple(era), "cost_model": cost_model,
                     **TR._row_stats(st), "verdict": r["verdict"], "source": source + "#" + r["id"],
                     "note": r.get("why", "")})
    return rows


def dsr_block(st: dict, n_full: int, v_full: float, n_fair: int, v_fair: float) -> dict:
    from . import trials as TR
    sr, t, sk, ku = st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"]
    a = TR.dsr(sr, t, sk, ku, n_full, v_full)
    b = TR.dsr(sr, t, sk, ku, n_fair, v_fair)
    f = lambda v: None if v != v else round(float(v), 4)
    return {"sr_daily": f(sr), "sr_annual": f(sr * (250 ** 0.5)) if sr == sr else None, "t_days": t,
            "psr0": f(a["psr0"]), "dsr_full": f(a["dsr"]), "n_full": n_full, "sr0_full": f(a["sr0"]),
            "dsr_fair": f(b["dsr"]), "n_fair": n_fair, "sr0_fair": f(b["sr0"])}


def _stats_row(s: dict) -> dict:
    return {"trades": s["trades"], "pf": s["pf"], "net": s["net"], "win_pct": s.get("win_pct")}


# ══════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════
def _pool(jobs: int, cal):
    """A worker pool forked AFTER the session calendar is set (workers inherit the tables and the calendar)."""
    E.set_calendar(cal)
    return mp.get_context("fork").Pool(jobs)


def _write_trades(path: str, trades: list) -> None:
    with open(path, "w") as f:
        f.write(",".join(sim.TRADE_FIELDS) + "\n")
        for t in trades:
            f.write(",".join(str(x) for x in t) + "\n")


def _clean_min(PM, trades: list) -> list:
    return [t for t in trades if PM.ERA[0] <= t[2] <= PM.ERA[1] and PM._clean_row(t[1], t[2], t[2])]


def run(out: str = OUT, jobs: int = 4, trials_ledger: str | None = None, log=print) -> dict:
    from . import protocol_min_oos as PM
    from . import trials as TR
    pre = load_prereg()
    D.assert_outside_repo(out)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    setup()
    cal3 = lab_calendar()
    cal_all = list(_CTX["cal"])
    cal_old = [d for d in cal_all if PO.ERA[0] <= d <= PO.ERA[1]]
    split = P.split_sessions(cal3)
    DISC = split.discovery
    nifty = _CTX["nifty"]
    daily, intr = list(ES.DAILY), list(ES.INTRADAY)
    rr_daily = [s for s in ES.RR3 if s.horizon == "positional"]
    rr_intr = [s for s in ES.RR3 if s.horizon == "intraday"]
    report: dict = {"prereg_sha256": open(os.path.splitext(PREREG)[0] + ".sha256").read().strip(),
                    "cost_model": COST_MODEL, "split": {"h1": split.h1, "h2": split.h2, "final": split.final},
                    "variants": {}, "controls": {}}
    V = report["variants"]
    lab_cache, lab_syms = lab_minute_cache()
    ctx = PM.setup(nsedata_view=False)
    cal_min = [int(x) for x in ctx["cal"]]
    old_syms = PM.symbols(PM.CACHE)
    tp = lambda sid, era: os.path.join(out, "trades", f"{sid}_{era}.csv")

    def new_block(r, tr, sens, kept, ca):
        sp, sc = r["spec"], r["screen"]
        disc = [t for t in tr if DISC[0] <= t[2] <= DISC[1]]
        st = TR.series_stats(_tdicts(tr), cal3, DISC[0], DISC[1])
        return {"verdict": r["verdict"], "why": r.get("why", ""), "h1": _stats_row(sc["h1"]),
                "h2": _stats_row(sc["h2"]), "disc": _stats_row(sc["disc"]), "top_symbol": sc["top_symbol"],
                "top_share": None if sc["top_share"] != sc["top_share"] else round(sc["top_share"], 3),
                "final": r.get("final"), "final_note": r.get("final_note", ""), "skipped": r.get("skipped", 0),
                "ca_dropped": ca, "disc_5bps": _stats_row(P.stats(sens)),
                "disc_ca_kept": None if kept is None else _stats_row(P.stats(kept)),
                "event_study_disc": block_bootstrap(abnormal_returns(disc, nifty)) if sp.horizon == "positional"
                else None, "_st": st}

    # ── daily variants: 2023-26 (protocol.evaluate), then the old era (protocol_oos.evaluate) ─────────────
    ca_new: dict = {}
    ca_old: dict = {}
    with _pool(jobs, cal_all) as pool:
        res_d, tr_d = P.evaluate(daily, cal3, make_daily_simulator(pool, SLIP_BPS, stats=ca_new),
                                 n_symbols=len(_CTX["symbols"]),
                                 ledger_path=os.path.join(_CTX["cache"], "final_ledger.json"),
                                 dataset="events_daily", cost_model=COST_MODEL)
        log(f"2023-26 daily: {[(r['id'], r['verdict']) for r in res_d]}")
        sens_d = make_daily_simulator(pool, SENS_BPS)(daily, DISC[0], DISC[1])
        keep_d = make_daily_simulator(pool, SLIP_BPS, keep_ca=True)(daily, DISC[0], DISC[1])
        for r in res_d:
            V[r["id"]] = {"rule": r["spec"].desc, "kind": r["spec"].kind,
                          "new": new_block(r, tr_d[r["id"]], sens_d[r["id"]]["trades"], keep_d[r["id"]]["trades"],
                                           ca_new.get(r["id"], 0))}
            _write_trades(tp(r["id"], "2023-26"), tr_d[r["id"]])
        res_o = PO.evaluate(daily, make_daily_simulator(pool, SLIP_BPS, stats=ca_old),
                            make_daily_simulator(pool, SENS_BPS), cost_model=COST_MODEL, ledger_path=OOS_LEDGER,
                            prereg=pre, dataset="events_daily")
        log(f"2012-23 daily: {[(r['id'], r['verdict']) for r in res_o]}")
        for r in res_o:
            tr = r.get("trades") or _read_trades(tp(r["id"], "2012-23"))
            if r.get("trades"):
                _write_trades(tp(r["id"], "2012-23"), tr)
            ex = {k: block_bootstrap(abnormal_returns([t for t in tr if rng[0] <= t[2] <= rng[1]], nifty))
                  for k, rng in (("h1", PO.H1), ("h2", PO.H2), ("era", PO.ERA))}
            V[r["id"]]["old"] = {"era": "2012-23", "verdict": r["verdict"], "why": r.get("why", ""),
                                 "note": r.get("note", ""), "h1": r.get("h1"), "h2": r.get("h2"), "all": r.get("era"),
                                 "top_symbol": r.get("top_symbol"), "top_share": r.get("top_share"),
                                 "sens_5bps": r.get("sens"), "ca_dropped": ca_old.get(r["id"]),
                                 "event_study": ex,
                                 "_st": TR.series_stats(_tdicts(tr), cal_old, PO.ERA[0], PO.ERA[1])}
        rr_new = make_daily_simulator(pool, SLIP_BPS)(rr_daily, DISC[0], DISC[1])
        rr_old = make_daily_simulator(pool, SLIP_BPS)(rr_daily, PO.ERA[0], PO.ERA[1])

    # ── intraday variant: 2023-26 on the lab's minute bars (the lab's own FINAL ledger) ───────────────────
    with _pool(jobs, cal3) as pool:
        res_i, tr_i = P.evaluate(intr, cal3, make_intraday_simulator(pool, lab_cache, lab_syms, SLIP_BPS),
                                 n_symbols=len(lab_syms), ledger_path=os.path.join(lab_cache, "final_ledger.json"),
                                 dataset="bars3y", cost_model=COST_MODEL)
        log(f"2023-26 intraday: {[(r['id'], r['verdict']) for r in res_i]}")
        sens_i = make_intraday_simulator(pool, lab_cache, lab_syms, SENS_BPS, audit=False)(intr, DISC[0], DISC[1])
        for r in res_i:
            V[r["id"]] = {"rule": r["spec"].desc, "kind": r["spec"].kind,
                          "new": new_block(r, tr_i[r["id"]], sens_i[r["id"]]["trades"], None, 0)}
            _write_trades(tp(r["id"], "2023-26"), tr_i[r["id"]])
        rr_new.update(make_intraday_simulator(pool, lab_cache, lab_syms, SLIP_BPS, audit=False)(
            rr_intr, DISC[0], DISC[1]))

    # ── intraday variant: 2016-23 minute bars, one look ───────────────────────────────────────────────────
    min_led = {}
    if os.path.exists(MIN_LEDGER):
        with open(MIN_LEDGER) as f:
            min_led = json.load(f)
    min_rows = []
    with _pool(jobs, cal_min) as pool:
        for sp in intr:
            key = f"{sp.id}|{COST_MODEL}"
            if key in min_led:
                same = min_led[key]["fingerprint"] == sp.fingerprint
                V[sp.id]["old"] = dict(min_led[key]["result"], from_ledger=True) if same else \
                    {"era": "2016-23", "verdict": "NOT JUDGED", "why": "already judged with other code"}
                continue
            r0 = make_intraday_simulator(pool, PM.CACHE, old_syms, SLIP_BPS)([sp], PM.ERA[0], PM.ERA[1])[sp.id]
            r5 = make_intraday_simulator(pool, PM.CACHE, old_syms, SENS_BPS, audit=False)(
                [sp], PM.ERA[0], PM.ERA[1])[sp.id]
            agg, agg5 = PM._agg(r0["trades"]), PM._agg(r5["trades"])
            j, j5 = PM.judge(agg, "intraday"), PM.judge(agg5, "intraday")
            bad = r0["error"] or r0["lookahead"]
            verdict = "INVALID" if bad else ("OOS-CONFIRMED" if j["passed"] else "OOS-FAILED")
            clean = _clean_min(PM, r0["trades"])
            st = TR.series_stats(_tdicts(clean), cal_min, PM.ERA[0], PM.ERA[1])
            result = {"era": "2016-23", "verdict": verdict, "why": str(bad) if bad else
                      ("all gates met" if j["passed"] else ", ".join(j["reasons"])), "h1": j["h1"], "h2": j["h2"],
                      "all": j["era"], "top_symbol": j["top_symbol"],
                      "top_share": None if j["top_share"] != j["top_share"] else round(j["top_share"], 4),
                      "sens_5bps": {"era": j5["era"], "passed": j5["passed"]},
                      "dirty_excluded": int(agg["dirty_excluded"]), "skipped": r0["skipped"],
                      "st": {k: st[k] for k in ("n_days", "sr_daily", "skew_daily", "kurt_daily", "mean_gross",
                                                "mean_cost")}}
            min_led[key] = {"fingerprint": sp.fingerprint, "when": time.strftime("%Y-%m-%d %H:%M"), "result": result}
            with open(MIN_LEDGER, "w") as f:
                json.dump(min_led, f, indent=1, default=_js)
            V[sp.id]["old"] = dict(result)
            _write_trades(tp(sp.id, "2016-23"), clean)
            min_rows.append({"strategy_id": sp.id, "family": TR.family_of(sp.id), "group": sp.family,
                             "params_fp": sp.fingerprint, "era": PM.ERA_TAG, "protocol": PM.PROTOCOL_TAG,
                             "stage": "oos", "dataset": "bars_min_old", "period": "%d-%d" % PM.ERA,
                             "cost_model": COST_MODEL, **TR._row_stats(st), "verdict": verdict,
                             "source": "events/min_oos_ledger.json#" + sp.id, "note": result["why"]})
        log(f"2016-23 intraday: {[(sp.id, V[sp.id]['old']['verdict']) for sp in intr]}")
        rri = make_intraday_simulator(pool, PM.CACHE, old_syms, SLIP_BPS, audit=False)(rr_intr, PM.ERA[0], PM.ERA[1])
        for v in rri.values():
            v["trades"] = _clean_min(PM, v["trades"])
        rr_old.update(rri)

    # ── trials ledger: every variant, both eras; twins as 'explore' ───────────────────────────────────────
    expl = []
    for tw in ES.RR3:
        st = TR.series_stats(_tdicts(rr_new[tw.id]["trades"]), cal3, DISC[0], DISC[1])
        expl.append({"strategy_id": tw.id, "family": TR.family_of(tw.id), "group": tw.family,
                     "params_fp": tw.fingerprint, "era": "2023-26", "protocol": "report-only", "stage": "explore",
                     "dataset": "events_daily" if tw.horizon == "positional" else "bars3y",
                     "period": "%d-%d" % DISC, "cost_model": COST_MODEL, **TR._row_stats(st),
                     "verdict": "REPORT-ONLY", "source": out + "#rr3", "note": "1:3 bracket twin"})
    if trials_ledger:
        n1 = TR.record_evaluation(res_d, tr_d, dataset="events_daily", cost_model=COST_MODEL,
                                  ledger_path=trials_ledger, calendar=cal3, source=out)
        n2 = TR.record_evaluation(res_i, tr_i, dataset="bars3y", cost_model=COST_MODEL,
                                  ledger_path=trials_ledger, calendar=cal3, source=out)
        n3 = TR.TrialsLedger(trials_ledger).append(
            ledger_rows_old([dict(r, trades=r.get("trades") or _read_trades(tp(r["id"], "2012-23"))) for r in res_o],
                            cal_old) + min_rows + expl)
        log(f"trials ledger: {n1 + n2} rows (2023-26) + {n3} rows (old eras, report-only twins)")

    # ── report-only: 1:3 twins and the Rs 10 lakh account ─────────────────────────────────────────────────
    for sp in ES.SPECS:
        v = V[sp.id]
        pos = sp.horizon == "positional"
        tw = ES.rr3_twin(sp)
        a, b = rr_new[tw.id]["trades"], rr_old[tw.id]["trades"]
        h1_end = PO.H1[1] if pos else PM.H1[1]
        era_old, co = (PO.ERA, cal_old) if pos else (PM.ERA, cal_min)
        new_tr = (tr_d if pos else tr_i)[sp.id]
        old_tr = _read_trades(tp(sp.id, "2012-23" if pos else "2016-23"))
        v["rr3"] = {"disc_2023_26": _stats_row(P.stats(a)), "old_era": _stats_row(P.stats(b)),
                    "old_h1": _stats_row(P.stats([t for t in b if t[2] <= h1_end])),
                    "old_h2": _stats_row(P.stats([t for t in b if t[2] > h1_end])),
                    "account_disc_2023_26": account(a, tw, cal3, DISC[0], DISC[1]),
                    "account_old_era": account(b, tw, co, era_old[0], era_old[1])}
        v["account"] = {"disc_2023_26": account(new_tr, sp, cal3, DISC[0], DISC[1]),
                        "old_era": account(old_tr, sp, co, era_old[0], era_old[1])}

    # ── controls: placebo (shuffled event dates) and planted edge ─────────────────────────────────────────
    pdir = os.path.join(out, "placebo_tables")
    placebo_tables(pdir)
    setup(events_dir=pdir)
    with _pool(jobs, cal_all) as pool:
        pn = make_daily_simulator(pool, SLIP_BPS)(daily, DISC[0], DISC[1])
        po = make_daily_simulator(pool, SLIP_BPS)(daily, PO.ERA[0], PO.ERA[1])
    with _pool(jobs, cal3) as pool:
        pn.update(make_intraday_simulator(pool, lab_cache, lab_syms, SLIP_BPS, audit=False)(intr, DISC[0], DISC[1]))
    with _pool(jobs, cal_min) as pool:
        pi = make_intraday_simulator(pool, PM.CACHE, old_syms, SLIP_BPS, audit=False)(intr, PM.ERA[0], PM.ERA[1])
    for v in pi.values():
        v["trades"] = _clean_min(PM, v["trades"])
    po.update(pi)
    report["controls"]["placebo"] = {
        sp.id: {"disc_2023_26": _stats_row(P.stats(pn[sp.id]["trades"])),
                "old_era": _stats_row(P.stats(po[sp.id]["trades"])),
                "event_study_old": block_bootstrap(abnormal_returns(po[sp.id]["trades"], nifty))
                if sp.horizon == "positional" else None} for sp in ES.SPECS}
    plant = {}
    for name, delay in (("planted", 0), ("planted_delayed_25", 25)):
        d = os.path.join(out, name + "_tables")
        n = planted_tables(d, delay=delay)
        setup(events_dir=d)
        with _pool(jobs, cal_all) as pool:
            r = make_daily_simulator(pool, SLIP_BPS)([ES.BY_ID["ev_promoter_buy_h20"]], PO.ERA[0], PO.ERA[1])
        plant[name] = {"events": n, **_stats_row(P.stats(r["ev_promoter_buy_h20"]["trades"]))}
    report["controls"]["planted_edge"] = plant
    setup(events_dir=EVENTS_DIR)

    # ── DSR / PSR ─────────────────────────────────────────────────────────────────────────────────────────
    rows = TR.TrialsLedger(trials_ledger).read() if trials_ledger else []
    n_full = TR.count_trials(rows, "2023-26") if rows else None
    n_old = TR.count_trials(rows, "2012-23") if rows else None
    n_min = TR.count_trials(rows, "2016-23") if rows else None
    v_full = TR.var_sr_robust(rows, "2023-26", "screen", "bars3y")[0] if rows else float("nan")
    srs = [v["new"]["_st"]["sr_daily"] for v in V.values()]
    srs = [x for x in srs if x == x]
    v_fair = float(np.var(srs, ddof=1)) if len(srs) > 1 else float("nan")
    vv = v_full if v_full == v_full else v_fair
    report["trials"] = {"n_2023_26": n_full, "n_2012_23": n_old, "n_2016_23": n_min, "n_fair": len(ES.SPECS),
                        "v_sr_robust_2023_26": v_full, "v_sr_study": v_fair}
    nf = len(ES.SPECS)
    for sid, v in V.items():
        st = v["new"].pop("_st")
        v["new"]["dsr_disc"] = dsr_block(st, n_full or nf, vv, nf, v_fair)
        v["new"]["mean_gross"], v["new"]["mean_cost"] = st["mean_gross"], st["mean_cost"]
        v["new"]["haircut_per_trade"] = None if st["mean_gross"] != st["mean_gross"] else round(
            0.5 * st["mean_gross"] - st["mean_cost"], 2)
        so = v.get("old", {}).pop("_st", None) or v.get("old", {}).get("st")
        if so and so.get("sr_daily") == so.get("sr_daily"):
            v["old"]["dsr"] = dsr_block(so, (n_min if v["kind"] == "intraday" else n_old) or nf, vv, nf, v_fair)
        d = v["new"]["dsr_disc"].get("dsr_full")
        v["proven"] = bool(v["new"]["verdict"] == "CONFIRMED" and v.get("old", {}).get("verdict") == "OOS-CONFIRMED"
                           and d is not None and d >= DSR_MIN)
    report["n_proven"] = sum(1 for v in V.values() if v["proven"])
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=_js)
    log(f"report: {os.path.join(out, 'report.json')}; proven: {report['n_proven']} of {len(V)}")
    return report


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    if dataclasses.is_dataclass(o):
        return getattr(o, "id", str(o))
    return str(o)


def _read_trades(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            x = line.rstrip("\n").split(",")
            out.append((x[0], x[1], int(x[2]), int(x[3]), x[4], float(x[5]), int(x[6]) if x[6] else int(x[2]), x[7],
                        float(x[8]), float(x[9]), float(x[10]), float(x[11]), float(x[12]), x[13]))
    return out


def smoke(jobs: int = 4) -> int:
    """Plumbing only: every variant must run without error on a few symbols and a short window. Prints NO P&L and
    no trade counts (nothing that could inform a design choice)."""
    setup()
    syms = _CTX["symbols"]
    _CTX["symbols"] = syms[::40]
    r = make_daily_simulator(None)(list(ES.DAILY) + [s for s in ES.RR3 if s.horizon == "positional"], 20190101, 20190630)
    bad = {i: v["error"] for i, v in r.items() if v["error"]}
    cache, lsyms = lab_minute_cache()
    E.set_calendar(lab_calendar())
    ri = make_intraday_simulator(None, cache, lsyms[:3])(list(ES.INTRADAY), 20240101, 20240331)
    bad.update({i: v["error"] or v["lookahead"] for i, v in ri.items() if v["error"] or v["lookahead"]})
    _CTX["symbols"] = syms
    print("smoke:", "OK (no errors)" if not bad else bad)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.events_run")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--trials", default=os.environ.get("STRATLAB_TRIALS"))
    a = ap.parse_args(argv)
    if a.smoke:
        return smoke(a.jobs)
    if a.preregister:
        preregister()
        return 0
    if a.run:
        run(a.out, a.jobs, a.trials)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
