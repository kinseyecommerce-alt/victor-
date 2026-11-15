"""
pead_parity.py — test harness for EV-PEAD-LONG-H60 (agents/validated_pead.py): replays lab days from the lab's own
stored data through the live decision function plan_pead() and compares with the lab's trade list.

Used only by test_validated_pead.py (and through it test_pipeline.py). Two sources:
  * the committed fixture (fixtures/pead_parity_sample.json.gz, ~40 sampled lab days, a few KB) — always available;
  * the lab's full data set in the research scratch (events_daily cache, results.csv.gz, run1 trade CSVs) — every
    session of both eras, skipped when absent. `python test_validated_pead.py --build-fixture` rebuilds the fixture
    from it.
Inputs per day exactly as the lab simulator saw them: the cache's adjusted bars (close = LAST), NIFTY closes,
the per-session universe flag, the results table (first_ts), the exclusion list (bad gaps + ambiguous corporate
actions); the stocks the lab already held come from its trade list (entry <= day < exit).
"""
from __future__ import annotations

import bisect
import csv
import gzip
import json
import os
import random
from typing import Optional

import numpy as np

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
LAB_CACHE = os.getenv("PEAD_PARITY_CACHE", os.path.join(_SCRATCH, "stratlab_cache", "events_daily"))
LAB_EVENTS = os.getenv("PEAD_PARITY_EVENTS", os.path.join(_SCRATCH, "events"))
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pead_parity_sample.json.gz")
SPEC_ID = "ev_pead_long_h60"


def lab_available() -> bool:
    return (os.path.exists(os.path.join(LAB_CACHE, "_manifest.json"))
            and os.path.exists(os.path.join(LAB_EVENTS, "results.csv.gz"))
            and os.path.exists(os.path.join(LAB_EVENTS, "run1", "trades", f"{SPEC_ID}_2023-26.csv")))


class Lab:
    """Lazy view of the lab's data set."""

    def __init__(self):
        from stratlab import data_daily as DD
        from stratlab import events_data as E
        self.DD, self.E = DD, E
        m = DD.manifest(LAB_CACHE)
        self.cal = [int(x) for x in m["calendar"]]
        self.idx = {d: i for i, d in enumerate(self.cal)}
        self.excl = [(str(s), int(d)) for s, d in m.get("bad_gaps", [])] + \
                    [(str(s), int(d)) for s, d in m.get("ca_ambiguous", [])]
        self.excl_by = {}
        for s, d in self.excl:
            self.excl_by.setdefault(s, []).append(d)
        self.symbols = set(m["symbols"])
        nf = DD.load_daily_aux(LAB_CACHE, "_NIFTY50_day")
        self.nifty = {int(d): float(c) for d, c in zip(nf["date"], nf["c"])}
        rows = []
        with gzip.open(os.path.join(LAB_EVENTS, "results.csv.gz"), "rt", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("first_ts"):
                    rows.append((E.ts_stamp(r["first_ts"]), r["symbol"], r["first_ts"], r["period_end"]))
        rows.sort()
        self.res = rows
        self.res_ts = [r[0] for r in rows]
        self._sd: dict = {}
        self.trades = []
        for era in ("2012-23", "2023-26"):
            with open(os.path.join(LAB_EVENTS, "run1", "trades", f"{SPEC_ID}_{era}.csv"), newline="") as f:
                for r in csv.DictReader(f):
                    self.trades.append((r["symbol"], int(r["date"]), int(r["exit_date"]), r["reason"]))
        self.held = {}
        for s, e, x, _ in self.trades:
            self.held.setdefault(s, []).append((e, x))

    def sym(self, s: str) -> Optional[dict]:
        if s not in self._sd:
            z = self.DD._npz(LAB_CACHE, s) if s in self.symbols else None
            self._sd[s] = None if z is None else {k: z[k] for k in ("date", "o", "h", "l", "c", "v", "member")}
        return self._sd[s]

    def events_between(self, lo_stamp: int, hi_stamp: int) -> list:
        i, j = bisect.bisect_left(self.res_ts, lo_stamp), bisect.bisect_left(self.res_ts, hi_stamp)
        return [(r[1], r[2], r[3]) for r in self.res[i:j]]

    def decision_day_of(self, s: str, entry: int) -> Optional[int]:
        z = self.sym(s)
        if z is None:
            return None
        i = int(np.searchsorted(z["date"], entry))
        return int(z["date"][i - 1]) if 0 < i < len(z["date"]) and int(z["date"][i]) == entry else None

    def holding(self, day: int) -> set:
        return {s for s, v in self.held.items() if any(e <= day < x for e, x in v)}

    def day_inputs(self, day: int, n_bars: int = 30, event_margin_days: int = 3) -> Optional[dict]:
        """Everything plan_pead() needs for lab session `day`, restricted to the stocks with results events around
        it (the fixture keeps rows a few days either side of the reaction window, so the window rule is tested)."""
        from pead_data import Bars
        E = self.E
        i = self.idx.get(day)
        if i is None or i == 0 or i + 1 >= len(self.cal):
            return None
        prv, nxt = self.cal[i - 1], self.cal[i + 1]
        lo = E.stamp(self.cal[max(0, i - 1 - event_margin_days)], E.CLOSE)
        hi = E.stamp(self.cal[min(len(self.cal) - 1, i + event_margin_days)], E.CLOSE)
        events = self.events_between(lo, hi)
        syms = sorted({e[0] for e in events})
        bars, members, excl = {}, [], []
        for s in syms:
            z = self.sym(s)
            if z is None:
                continue
            k = int(np.searchsorted(z["date"], day, side="right"))
            if k == 0:
                continue
            a = max(0, k - n_bars)
            bars[s] = Bars(tuple(int(x) for x in z["date"][a:k]), tuple(z["o"][a:k].tolist()),
                           tuple(z["h"][a:k].tolist()), tuple(z["l"][a:k].tolist()), tuple(z["c"][a:k].tolist()),
                           tuple(z["v"][a:k].tolist()))
            if int(z["date"][k - 1]) == day and bool(z["member"][k - 1]):
                members.append(s)
            excl += [(s, d) for d in self.excl_by.get(s, []) if self.cal[max(0, i - 40)] <= d <= day]
        nifty = {d: self.nifty[d] for d in self.cal[max(0, i - n_bars - 5):i + 1] if d in self.nifty}
        return {"day": day, "calendar": self.cal[max(0, i - n_bars - 5):i + 2], "events": events, "bars": bars,
                "nifty": nifty, "members": members, "holding": sorted(self.holding(day) & set(syms)),
                "exclusions": excl, "prev": prv, "next": nxt}

    def expected(self) -> dict:
        """{decision day: set(symbols)} of the lab's ev_pead_long_h60 entries (decision = the stock's bar before
        the entry bar)."""
        out: dict = {}
        for s, e, _x, _r in self.trades:
            d = self.decision_day_of(s, e)
            if d is not None:
                out.setdefault(d, set()).add(s)
        return out


def run_day(inp: dict):
    from agents.validated_pead import plan_pead
    return plan_pead(inp["day"], inp["calendar"], inp["events"], inp["bars"], inp["nifty"], inp["members"],
                     inp["holding"], inp["exclusions"])


def replay(lab: Optional[Lab] = None, days=None) -> dict:
    """Every lab session (or `days`) through plan_pead(); compare with the lab's entries."""
    lab = lab or Lab()
    exp = lab.expected()
    first_entry = min(e for _s, e, _x, _r in lab.trades)
    last_entry = max(e for _s, e, _x, _r in lab.trades)
    days = days or [d for d in lab.cal if lab.cal.index(d) + 1 < len(lab.cal)
                    and first_entry <= lab.cal[lab.cal.index(d) + 1] <= last_entry]
    matched, missing, extra, n_days = 0, [], [], 0
    for d in days:
        inp = lab.day_inputs(d)
        if inp is None:
            continue
        n_days += 1
        got = {e.symbol for e in run_day(inp)[0]}
        want = exp.get(d, set())
        matched += len(got & want)
        missing += [(d, s) for s in sorted(want - got)]
        extra += [(d, s) for s in sorted(got - want)]
    return {"days": n_days, "lab_entries": sum(len(exp.get(d, ())) for d in days), "matched": matched,
            "missing": missing, "extra": extra}


def classify_extra(lab: Lab, extra: list) -> dict:
    """An extra live entry the lab's trade list lacks: the lab dropped trades whose holding window contains an
    excluded corporate-action / gap day (events_run.drop_excluded) — those were still entered by the rule."""
    out = {"lab_dropped_ca": [], "unexplained": []}
    for d, s in extra:
        i = lab.idx[d]
        window = lab.cal[i:i + 90]
        if any(window[0] < x <= window[-1] for x in lab.excl_by.get(s, [])):
            out["lab_dropped_ca"].append((d, s))
        else:
            out["unexplained"].append((d, s))
    return out


# ── fixture ──────────────────────────────────────────────────────────────────
def _bars_json(b) -> list:
    # the rule reads date, close and volume of the last vol_n + 2 bars only; o/h/l are stored as the close
    return [list(b.dates[-22:]), [round(x, 4) for x in b.c[-22:]], [round(x) for x in b.v[-22:]]]


def build_fixture(path: str = FIXTURE, n_new: int = 16, n_old: int = 10, max_cands: int = 12,
                  seed: int = 20261115) -> dict:
    """~30 lab days with at most `max_cands` universe members reacting (small fixture): the reaction-window events
    with their bars, a few events just outside the window (no bars needed: the window rule drops them) and a few
    non-members (the universe gate drops them)."""
    lab = Lab()
    exp = lab.expected()
    E = lab.E
    rng = random.Random(seed)

    def cands(d):
        inp = lab.day_inputs(d, n_bars=22, event_margin_days=1)
        if inp is None:
            return None, []
        lo, hi = E.stamp(inp["prev"], E.CLOSE), E.stamp(d, E.CLOSE)
        inwin = sorted({s for s, ts, _ in inp["events"] if lo <= E.ts_stamp(ts) < hi})
        return inp, inwin

    pool_new = sorted(d for d in exp if d >= 20231109)
    pool_old = sorted(d for d in exp if d < 20231109)
    rng.shuffle(pool_new)
    rng.shuffle(pool_old)
    quiet = [d for d in lab.cal[3000:3600:23] if d not in exp]
    picked = []
    for pool, n in ((pool_new, n_new), (pool_old, n_old), (quiet, 4)):
        k = 0
        for d in pool:
            inp, inwin = cands(d)
            if inp is None or not inwin:
                continue
            mem = [s for s in inwin if s in inp["members"]]
            if len(mem) > max_cands:
                continue
            picked.append((d, inp, inwin, mem))
            k += 1
            if k >= n:
                break
    out = []
    for d, inp, inwin, mem in sorted(picked, key=lambda x: x[0]):
        nonmem = [s for s in inwin if s not in inp["members"]][:3]
        outside = sorted({s for s, _ts, _ in inp["events"]} - set(inwin))[:3]
        keep = set(mem) | set(nonmem) | set(outside)
        out.append({"day": d, "calendar": inp["calendar"][-25:],
                    "events": [e for e in inp["events"] if e[0] in keep],
                    "bars": {s: _bars_json(inp["bars"][s]) for s in mem if s in inp["bars"]},
                    "nifty": {str(k): v for k, v in inp["nifty"].items() if k >= inp["calendar"][-25]},
                    "members": mem, "holding": [s for s in inp["holding"] if s in keep],
                    "exclusions": [x for x in inp["exclusions"] if x[0] in keep],
                    "expected": sorted(exp.get(d, set()))})
    blob = {"source": "stratlab events run1 (events_daily cache ed1, results.csv.gz, ev_pead_long_h60 trade CSVs)",
            "seed": seed, "days": out}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", compresslevel=9) as f:
        json.dump(blob, f, separators=(",", ":"))
    return blob


def load_fixture(path: str = FIXTURE) -> list[dict]:
    from pead_data import Bars
    with gzip.open(path, "rt") as f:
        blob = json.load(f)
    out = []
    for x in blob["days"]:
        bars = {s: Bars(tuple(v[0]), tuple(v[1]), tuple(v[1]), tuple(v[1]), tuple(v[1]), tuple(v[2]))
                for s, v in x["bars"].items()}
        out.append({"day": x["day"], "calendar": x["calendar"], "events": [tuple(e) for e in x["events"]],
                    "bars": bars, "nifty": {int(k): v for k, v in x["nifty"].items()}, "members": x["members"],
                    "holding": x["holding"], "exclusions": [tuple(e) for e in x["exclusions"]],
                    "expected": set(x["expected"])})
    return out


# ── live data path (scratch only): NSE bhavcopy files -> pead_data -> plan_pead ──────────────────────────────
LAB_NSE = os.getenv("PEAD_PARITY_NSE", os.path.join(_SCRATCH, "nse"))


def live_path_available() -> bool:
    return lab_available() and os.path.isdir(os.path.join(LAB_NSE, "raw_cm")) and \
        os.path.exists(os.path.join(LAB_EVENTS, "raw", "ref", "EQUITY_L.csv"))


def live_path_replay(lab: Optional[Lab] = None, lo: int = 20240201, hi: int = 20261106, step: int = 1) -> dict:
    """The live data path on the lab's own raw files: the universe from events_daily.monthly_universe() on the raw
    sec_bhavdata_full files with pead_data.parse_equity_list(), the bars from pead_data.build_bars() (read_new rows,
    new_factors() on the lab's corporate-action feed, close = LAST), the lab's results table and NIFTY. Symbols are
    followed to the current name with the lab's own rename step (live sees each day's own symbols in both the
    bhavcopy and the filing feeds, so needs none). Compared with the lab's entries."""
    import pandas as pd
    import pead_data as PD
    from stratlab import events_daily as ED
    from stratlab.events_parse import load_changes
    lab = lab or Lab()
    eq = ED.read_new(os.path.join(LAB_NSE, "raw_cm"), 20231031, hi)
    ch = [(d, o, n) for d, o, n in load_changes(os.path.join(LAB_EVENTS, "raw", "ref", "symbolchange.csv"))
          if d >= 20120101]
    eq, _ = ED.apply_renames_fast(eq, ch)
    ref = os.path.join(LAB_EVENTS, "raw", "ref")
    with open(os.path.join(ref, "EQUITY_L.csv"), "rb") as f1, open(os.path.join(ref, "eq_etfseclist.csv"), "rb") as f2:
        equity = PD.parse_equity_list(f1.read(), f2.read())
    uni = ED.monthly_universe(eq, equity, 500)
    lab_uni = pd.read_csv(os.path.join(LAB_CACHE, "universe_monthly.csv"))
    umatch = {}
    for m, g in uni.groupby("month"):
        if int(m) * 100 + 1 < lo:
            continue
        a, b = set(g["symbol"]), set(lab_uni.loc[lab_uni["month"] == m, "symbol"])
        umatch[int(m)] = (len(a & b), len(b))
    members = {int(m): set(g["symbol"]) for m, g in uni.groupby("month")}
    actions = ED.load_actions(os.path.join(LAB_EVENTS, "raw"))
    exp = lab.expected()
    days = [d for d in lab.cal if lo <= d <= hi][::step]
    eq_by_date = {int(d): g for d, g in eq.groupby("date")}
    matched, missing, extra, n = 0, [], [], 0
    for d in days:
        i = lab.idx[d]
        cal = lab.cal[max(0, i - 30):i + 2]
        inp = lab.day_inputs(d, event_margin_days=0)
        if inp is None:
            continue
        syms = {e[0] for e in inp["events"]}
        parts = [eq_by_date[x] for x in cal[:-1] if x in eq_by_date]
        win = pd.concat(parts, ignore_index=True) if parts else None
        bars, excl = {}, []
        if win is not None:
            bars, excl, _ = PD.build_bars(win[win["symbol"].isin(syms)], actions, cal[:-1])
        got = {e.symbol for e in run_day({**inp, "bars": bars, "members": members.get(d // 100, set()),
                                          "exclusions": excl + inp["exclusions"]})[0]}
        want = exp.get(d, set())
        n += 1
        matched += len(got & want)
        missing += [(d, s) for s in sorted(want - got)]
        extra += [(d, s) for s in sorted(got - want)]
    return {"days": n, "lab_entries": sum(len(exp.get(d, ())) for d in days), "matched": matched,
            "missing": missing, "extra": extra, "universe_match": umatch}
