"""
stratlab/builder_options.py — the builder's OPTIONS agent on REAL end-of-day option prices, through nsedata's
real-option simulator UNCHANGED (nsedata._simulate_one: decide() under cutoff(D) sees only data dated < D, strikes are
chosen on the previous F&O session's real chain, fills at the entry session's official close with the haircut
max(Rs 0.05, 1 %) per leg per fill, expiry settlement at intrinsic vs the final settlement price, costs via
options_replay.leg_cost, lots from options_vrp.LOT_SCHEDULE / nsedata.BANKNIFTY_LOT_SCHEDULE). No VIX-modelled prices.

What differs from nsedata's own loader is only WHERE the tables come from: load_compact() fills nsedata's in-memory
tables (_S) from the F&O bhavcopy parts (fo_parts/optidx + fo_parts/fut) of one or more nsedata-layout directories
(the sandbox: nse_old_view 2012-2023 + nse 2023 ->; the server: the builder data layer's nse/), keeping NIFTY /
BANKNIFTY options with strikes within +-15 % of the underlying and expiries <= 75 days away (every structure here
uses strikes within 7 % and expiries within ~40 days), plus the index futures (final settlement). A zero option
`settle` on an expiry day (old-format files, weekly expiries) is treated as missing (fix of 2026-11-15, after batch
0001: see the registry's erratum record). nsedata's on-disk
cache (~2 GB for 2012 ->) is not built; the compact tables are cached as one compressed npz.

Entry: the signal of the evening of t (index daily grid) -> decide() on the next session D = t+1 (chain of t, filled at
D's close). Structures (the side of a directional block picks calls / puts):
  long_atm        buy the ATM call (+) / put (-)
  debit_spread    buy ATM, sell the strike nearest spot x (1 +- 2 %), same type
  short_strangle  sell the CE nearest spot x 1.04 and the PE nearest spot x 0.96
  iron_condor     the strangle + long wings 3 % further out
Expiry: weekly = the nearest expiry alive >= 2 calendar days after D (else the next); monthly = options_r4's
_monthly_chain (the first monthly expiry >= 10 days away).
Exits: opt_hold h (close at the h-th session after entry; settlement first), opt_expiry, opt_bracket (previous
session's mark: debit stop at value <= 50 % of the debit / target >= 250 %; credit stop at cost-to-close >= 2 x credit /
target <= 50 %), opt_signal (NIFTY's previous close through its 10-session mean against the side, or VIX percentile
< 0.5 for neutral structures). manage() reads only data dated before the session (marks, closes <= D-1).

LOTS (grammar bg3, size regime "c10" = Rs 10 lakh capital per agent; OPTION_SIZES = builder_rule_v3.json
costs.size_rs.options). A bg1 / bg2 candidate trades 1 lot, as ever. A re-sized candidate's lots are decided AT ENTRY
from the entry fills only (option_lots; no later price is read), the same for every leg:
  long_atm, debit_spread   max(1, floor(Rs 1,00,000 / net premium paid per lot))        - Rs 1 lakh of premium at risk
  iron_condor              max(1, floor(Rs 1,00,000 / max loss per lot)), max loss = (widest wing - net credit) x lot
  short_strangle (naked)   max(1, floor(Rs 5,00,000 / margin per lot)), margin = 12 % of spot notional per lot (the
                           lab's own estimate, options_vrp.NAKED_MARGIN_PCT; spot = the index close of the entry
                           session, else the mean of the two short strikes)
  cap                      at most `max_units_per_leg` units per leg (NIFTY 1,800 / BANKNIFTY 600: one un-sliced order
                           per leg, about NSE's freeze quantities), so the flat brokerage per order stays honest
Fees are options_replay.leg_cost per leg at the real quantity, the haircut per unit is unchanged. The simulator is not
edited: while a re-sized job runs, nsedata.Position is replaced by a subclass that sets the plan's lots when the
position is opened (the fills are known there) and is restored afterwards.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os

import numpy as np

from . import builder_grammar as G
from . import nsedata as N
from . import options_r4 as O4

INDICES = ("NIFTY", "BANKNIFTY")
# option lots by size regime (builder_grammar.Candidate.size); "c10" is builder_rule_v3.json costs.size_rs.options
OPTION_SIZES = {
    G.SIZE_C10: {"premium_rs": 100000, "max_loss_rs": 100000, "naked_margin_rs": 500000, "naked_margin_pct": 0.12,
                 "max_units_per_leg": {"NIFTY": 1800, "BANKNIFTY": 600}},
}
STRIKE_BAND = 0.15
MAX_DTE = 75
CHAIN_FIELDS = ("open", "high", "low", "close", "settle", "contracts", "oi", "chg_oi", "underlying", "lot")
_LOADED: dict = {}


def _ok_days(d: str) -> list:
    p = os.path.join(d, "fo_parts", "status.csv")
    out = []
    if os.path.exists(p):
        import csv
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok":
                    out.append(int(r["date"].replace("-", "")))
    return sorted(out)


def _spot_series(bars_dir: str) -> dict:
    out = {}
    try:
        b = json.load(open(os.path.join(bars_dir, "_NIFTY50_day.json")))["bars"]
        out["NIFTY"] = {int(x[0][:10].replace("-", "")): float(x[4]) for x in b}
    except Exception:
        out["NIFTY"] = {}
    try:
        b = json.load(open(os.path.join(bars_dir, "_NIFTYBANK_min.json")))["bars"]
        last = {}
        for x in b:
            last[int(x[0][:10].replace("-", ""))] = float(x[4])
        out["BANKNIFTY"] = last
    except Exception:
        out["BANKNIFTY"] = {}
    return out


def signature(dirs: list) -> str:
    h = hashlib.sha256(b"bo2")
    for d, b, lo, hi in dirs:
        h.update(f"{d}|{b}|{lo}|{hi}".encode())
        p = os.path.join(d, "fo_parts", "status.csv")
        h.update(str(os.path.getsize(p) if os.path.exists(p) else -1).encode())
        for sub in ("optidx", "fut"):
            fs = sorted(glob.glob(os.path.join(d, "fo_parts", sub, "*.csv.gz")))
            h.update(f"{sub}:{len(fs)}:{os.path.basename(fs[-1]) if fs else ''}".encode())
    return h.hexdigest()[:16]


def _read_day(path: str, kind: str):
    import pandas as pd
    df = pd.read_csv(path, dtype={"symbol": str, "type": str})
    df = df[df["symbol"].isin(INDICES)]
    if kind == "fut":
        df = df[df["instrument"] == "FUTIDX"]
    return df


def build_tables(dirs: list, log=None) -> dict:
    """{"chains": {sym: table}, "fut": table, "fo_days": array, "spot": {sym: (dates, closes)}} from fo_parts."""
    import pandas as pd
    chains = {s: [] for s in INDICES}
    futs, fo_days = [], []
    spot = {s: {} for s in INDICES}
    for d, bars, lo, hi in dirs:
        sp = _spot_series(bars)
        for s in INDICES:
            spot[s].update({k: v for k, v in sp[s].items() if lo <= k <= hi})
        days = [x for x in _ok_days(d) if lo <= x <= hi]
        fo_days += days
        for i, day in enumerate(days):
            iso = f"{day // 10000:04d}{day // 100 % 100:02d}{day % 100:02d}"
            po = os.path.join(d, "fo_parts", "optidx", iso + ".csv.gz")
            pf = os.path.join(d, "fo_parts", "fut", iso + ".csv.gz")
            if os.path.exists(pf):
                futs.append(_read_day(pf, "fut"))
            if not os.path.exists(po):
                continue
            df = _read_day(po, "opt")
            if not len(df):
                continue
            df = df.assign(dint=day, eint=df["expiry"].str.replace("-", "").astype(np.int64))
            dte = (pd.to_datetime(df["eint"].astype(str)) - pd.Timestamp(iso)).dt.days
            df = df[(dte >= 0) & (dte <= MAX_DTE)]
            for s in INDICES:
                g = df[df["symbol"] == s]
                if not len(g):
                    continue
                und = pd.to_numeric(g["underlying"], errors="coerce")
                ref = spot[s].get(day, np.nan)
                u = und.where(np.isfinite(und), ref)
                g = g[(np.abs(g["strike"] / u - 1.0) <= STRIKE_BAND) | ~np.isfinite(u)]
                part = {"date": g["dint"].to_numpy(np.int64), "expiry": g["eint"].to_numpy(np.int64),
                        "strike": g["strike"].to_numpy(float), "ce": (g["type"] == "CE").to_numpy(bool)}
                for k in CHAIN_FIELDS:
                    part[k] = pd.to_numeric(g[k], errors="coerce").to_numpy(float, copy=True)
                # the old-format bhavcopy (nse_old_view) carries settle = 0 on WEEKLY expiry days (monthly expiries
                # carry the underlying's settlement price): a zero settle is a missing value, never a price, so
                # nsedata.final_settlement falls back to the official index close (NSE's weekly settlement price)
                bad = (part["date"] == part["expiry"]) & ~(part["settle"] > 0)
                part["settle"][bad] = np.nan
                chains[s].append(part)
            del df
            if log and i % 500 == 0:
                log(f"options: {d} {day} ({i + 1}/{len(days)})")
    out = {"chains": {}, "fo_days": np.array(sorted(set(fo_days)), np.int64), "spot": {}}
    for s in INDICES:
        if chains[s]:
            t = {k: np.concatenate([p[k] for p in chains[s]]) for k in chains[s][0]}
            chains[s] = None
            o = np.lexsort((~t["ce"], t["strike"], t["expiry"], t["date"]))
            out["chains"][s] = {k: v[o] for k, v in t.items()}
            del t
        ks = sorted(spot[s])
        out["spot"][s] = (np.array(ks, np.int64), np.array([spot[s][k] for k in ks], float))
    if futs:
        f = pd.concat(futs, ignore_index=True)
        f["dint"] = f["date"].str.replace("-", "").astype(np.int64)
        f["eint"] = f["expiry"].str.replace("-", "").astype(np.int64)
        f = f.sort_values(["symbol", "dint", "eint"])
        ft = {"sym": f["symbol"].to_numpy(str), "date": f["dint"].to_numpy(np.int64),
              "expiry": f["eint"].to_numpy(np.int64), "idx": np.ones(len(f), bool)}
        for k in CHAIN_FIELDS:
            ft[k] = pd.to_numeric(f[k], errors="coerce").to_numpy(float)
        out["fut"] = ft
    else:
        out["fut"] = None
    return out


def _save(path: str, tabs: dict) -> None:
    flat = {"fo_days": tabs["fo_days"]}
    for s, t in tabs["chains"].items():
        for k, v in t.items():
            flat[f"ch_{s}_{k}"] = v
    for s, (d, c) in tabs["spot"].items():
        flat[f"sp_{s}_d"], flat[f"sp_{s}_c"] = d, c
    if tabs["fut"] is not None:
        for k, v in tabs["fut"].items():
            flat[f"fu_{k}"] = v
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def _load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {"fo_days": z["fo_days"], "chains": {}, "spot": {}, "fut": None}
    fut = {}
    for k in z.files:
        if k.startswith("ch_"):
            _, s, f = k.split("_", 2)
            out["chains"].setdefault(s, {})[f] = z[k]
        elif k.startswith("sp_"):
            _, s, f = k.split("_", 2)
            out["spot"].setdefault(s, [None, None])["dc".index(f)] = z[k]
        elif k.startswith("fu_"):
            fut[k[3:]] = z[k]
    out["spot"] = {s: (v[0], v[1]) for s, v in out["spot"].items()}
    out["fut"] = fut or None
    return out


def load_compact(dirs: list, cache_dir: str, log=None) -> dict:
    """Fill nsedata's in-memory tables from the compact tables (built once per data signature)."""
    sig = signature(dirs)
    if _LOADED.get("sig") == sig:
        return _LOADED["tabs"]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"options_{sig}.npz")
    if os.path.exists(path):
        tabs = _load(path)
    else:
        tabs = build_tables(dirs, log)
        _save(path, tabs)
    install(tabs)
    _LOADED.update({"sig": sig, "tabs": tabs})
    return tabs


def install(tabs: dict) -> None:
    """nsedata._S from the compact tables (keys nsedata's accessors read; stock tables empty)."""
    S = N._S
    S.clear()
    S["alias"] = {}
    S["dv"] = {"sym": np.array([], str), "date": np.array([], np.int64),
               **{k: np.array([], float) for k in ("open", "high", "low", "close", "prev_close", "avg_price", "volume",
                                                    "turnover_lacs", "trades", "deliv_qty", "deliv_pct")}}
    S["dv_ix"] = {}
    S["fut"] = tabs["fut"]
    S["fut_ix"] = N._by_symbol(tabs["fut"]) if tabs["fut"] is not None else {}
    S["agg"], S["agg_ix"] = None, {}
    S["chains"] = {s: tabs["chains"].get(s) for s in INDICES}
    S["fo_days"] = np.asarray(tabs["fo_days"], np.int64)
    S["daily_fut"], S["daily_opt"] = {}, {}
    S["part_oi"], S["part_vol"] = None, None
    S["ban"], S["ban_days"] = {}, np.asarray(tabs["fo_days"], np.int64)
    S["fii"] = {}
    S["spot_idx"] = dict(tabs["spot"])


# ══════════════════════════════════════════════════════════════════════════
# decide / manage
# ══════════════════════════════════════════════════════════════════════════
def _weekly(ctx):
    exs = [e for e in O4._expiries(ctx) if e > ctx.date and O4._days(ctx.date, e) >= 2]
    if not exs:
        return None
    return O4._chain(ctx, exs[0])


def _pick_chain(ctx, expiry: str):
    return _weekly(ctx) if expiry == "weekly" else O4._monthly_chain(ctx, 10)


def make_plan(c, structure: str, side: int, exit_):
    if structure == "long_atm":
        k = O4._atm(c)
        typ = "CE" if side > 0 else "PE"
        return O4._plan(c, [(typ, k, +1)], exit_, f"spot {c.spot:.1f}")
    if structure == "debit_spread":
        k = O4._atm(c)
        if side > 0:
            k2 = O4._nearest(c, "CE", c.spot * 1.02)
            legs = [("CE", k, +1), ("CE", k2, -1)]
        else:
            k2 = O4._nearest(c, "PE", c.spot * 0.98)
            legs = [("PE", k, +1), ("PE", k2, -1)]
        if k is None or k2 is None or k2 == k:
            return None
        return O4._plan(c, legs, exit_, f"spot {c.spot:.1f}")
    if structure in ("short_strangle", "iron_condor"):
        return O4._strangle_pct(c, 0.04, exit_, wing=0.03 if structure == "iron_condor" else None)
    raise KeyError(structure)


def _exit_of(cand) -> object:
    k = cand.exit[0]
    if k == "opt_hold":
        return int(cand.exit[1])
    return "expiry"


class _Signals:
    """Index daily series for manage() (values of sessions strictly before the session asked)."""

    def __init__(self, E, inst: str):
        self.cal = E.dcal
        self.close = E.idx[inst]["c"]
        vp = E.ctx.d("vix_pct")
        self.vix_pct = vp

    def before(self, day: int) -> int:
        return int(np.searchsorted(self.cal, int(day), side="left")) - 1      # last daily session < day

    def sma_cross_against(self, day: int, side: int) -> bool:
        k = self.before(day)
        if k < 9:
            return False
        w = self.close[k - 9:k + 1]
        if not np.isfinite(w).all():
            return False
        return side * (self.close[k] - w.mean()) < 0

    def vix_below_median(self, day: int) -> bool:
        k = self.before(day)
        return k >= 0 and np.isfinite(self.vix_pct[k]) and self.vix_pct[k] < 0.5


def option_spec(E, cand, sid: str, entries: dict) -> N.OptionSpec:
    """entries: {entry session D: side (+1 / -1; neutral structures ignore it)}."""
    structure, expiry = cand.structure.split(":")
    exit_ = _exit_of(cand)
    sig = _Signals(E, cand.instrument)
    kind = cand.exit[0]
    neutral = structure in ("short_strangle", "iron_condor")

    def decide(ctx, p):
        sd = entries.get(int(ctx.date))
        if sd is None:
            return None
        c = _pick_chain(ctx, expiry)
        if c is None:
            return None
        plan = make_plan(c, structure, sd, exit_)
        if plan is not None:
            plan.note = f"{plan.note} side {sd:+d}"
        return plan

    def manage(ctx, pos, p):
        if kind == "opt_bracket":
            marks = O4._marks(ctx, pos)
            if not marks or not marks[-1][1] == marks[-1][1]:
                return False
            v = marks[-1][1]
            ent = sum(l.side * l.lots * f for l, f in zip(pos.plan.legs, pos.fills))
            if ent > 0:                                 # debit
                return v <= 0.5 * ent or v >= 2.5 * ent
            credit, cost = -ent, -v
            return credit > 0 and (cost >= 2.0 * credit or cost <= 0.5 * credit)
        if kind == "opt_signal":
            if neutral:
                return sig.vix_below_median(int(ctx.date))
            sd = 1 if "side +1" in (pos.plan.note or "") else -1
            return sig.sma_cross_against(int(ctx.date), sd)
        return False

    return N.OptionSpec(sid, G.describe(cand), {}, decide, underlyings=(cand.instrument,),
                        manage=manage if kind in ("opt_bracket", "opt_signal") else None, family="builder_options")


def option_lots(structure: str, legs: list, fills: list, lot: int, symbol: str, spot: float | None,
                cfg: dict) -> tuple:
    """(lots, basis) of one position from its ENTRY fills. legs: [(typ, strike, side)]; fills: per unit, per leg;
    lot: units per lot; spot: the index level at entry (naked margin). basis = {"rule", "per_lot_rs", "budget_rs",
    "uncapped", "cap"}. Never below 1 lot."""
    lot = int(lot)
    net = sum(sd * f for (_t, _k, sd), f in zip(legs, fills))           # per unit: > 0 debit, < 0 credit
    if structure in ("long_atm", "debit_spread"):
        rule, budget, per_lot = "premium", float(cfg["premium_rs"]), net * lot
    elif structure == "iron_condor":
        widths = [abs(a[1] - b[1]) for a in legs for b in legs if a[0] == b[0] and a[2] < 0 < b[2]]
        rule, budget = "max_loss", float(cfg["max_loss_rs"])
        per_lot = max(0.0, (max(widths) if widths else 0.0) + net) * lot    # widest wing - net credit
    elif structure == "short_strangle":
        shorts = [k for _t, k, sd in legs if sd < 0]
        s = float(spot) if spot is not None and spot == spot and spot > 0 else sum(shorts) / max(1, len(shorts))
        rule, budget, per_lot = "naked_margin", float(cfg["naked_margin_rs"]), float(cfg["naked_margin_pct"]) * s * lot
    else:
        raise KeyError(structure)
    n = int(budget // per_lot) if per_lot > 0 else 1
    cap = max(1, int(cfg["max_units_per_leg"][symbol]) // lot) if lot > 0 else 1
    return max(1, min(n, cap)), {"rule": rule, "per_lot_rs": round(per_lot, 2), "budget_rs": budget,
                                 "uncapped": max(1, n), "cap": cap}


class _sized_positions:
    """While active, every position nsedata's simulator opens gets its lots from option_lots (the fills are known
    when the Position is created). cfg None (a bg1 / bg2 candidate): nothing is replaced, 1 lot."""

    def __init__(self, cfg, structure: str, spot_at):
        self.cfg, self.structure, self.spot_at = cfg, structure, spot_at
        self.orig = None
        self.log: list = []                              # (entry day, lots, basis) per opened position

    def __enter__(self):
        if self.cfg is None:
            return self
        import dataclasses
        outer = self

        @dataclasses.dataclass
        class SizedPosition(N.Position):
            def __post_init__(self):
                legs = [(l.typ, l.strike, l.side) for l in self.plan.legs]
                n, basis = option_lots(outer.structure, legs, self.fills, self.qty_per_lot, self.symbol,
                                       outer.spot_at(self.entry_date), outer.cfg)
                for l in self.plan.legs:
                    l.lots = n
                outer.log.append((int(self.entry_date), n, basis))
        self.orig, N.Position = N.Position, SizedPosition
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            N.Position = self.orig
        return False


def ensure_loaded(E) -> None:
    fixed = getattr(E.src, "option_tables_path", None)
    if fixed:                                         # a frozen batch's own tables (stratlab/builder_bundle.py)
        if _LOADED.get("sig") != "file:" + fixed:
            tabs = _load(fixed)
            install(tabs)
            _LOADED.update({"sig": "file:" + fixed, "tabs": tabs})
        return
    cache = getattr(E, "cache_dir", None) or os.path.join(os.environ.get("BUILDER_CACHE_DIR", "/tmp"), "builder_cache")
    load_compact(E.src.option_dirs(), cache, log=getattr(E, "log", None))


def run_option_job(E, job) -> dict:
    """The options counterpart of builder_engine.run_jobs for one job."""
    ensure_loaded(E)
    cand = job.cand
    sig = job.sig if job.sig is not None else E.signals(cand)
    cal = E.dcal
    entries = {}
    planted = job.planted_p is not None
    rng = np.random.default_rng(job.seed) if planted else None
    close = E.idx[cand.instrument]["c"]
    minute_grid = cand.grid in ("idx_minute", "minute")
    for s, t, side in zip(sig["s"].tolist(), sig["t"].tolist(), sig["side"].tolist()):
        if minute_grid:                             # minute-grid block (expiry calendar): the same date's daily index
            d = int(E.mcal[t])
            t = int(np.searchsorted(cal, d))
            if t >= len(cal) or int(cal[t]) != d:
                continue
        e = t + 1
        if e >= len(cal) or not (job.lo <= int(cal[e]) <= job.hi):
            continue
        if planted:
            x = min(e + 3, len(cal) - 1)
            if not (np.isfinite(close[x]) and np.isfinite(close[e])):
                continue
            mv = 1 if close[x] > close[e] else -1
            side = mv if rng.random() < job.planted_p else -mv
        entries.setdefault(int(cal[e]), int(side) if side else 1)
    spec = option_spec(E, cand, job.key, entries)
    calendar = [int(x) for x in cal if int(x) <= job.hi]
    cfg = (getattr(E, "option_sizes", None) or OPTION_SIZES).get(cand.size) if cand.size else None

    def spot_at(day: int):
        k = int(np.searchsorted(cal, int(day)))
        return float(close[k]) if k < len(cal) and int(cal[k]) == int(day) and np.isfinite(close[k]) else None
    with _sized_positions(cfg, cand.structure.split(":")[0], spot_at) as sz:
        tr, _det, _sk = N._simulate_one(spec, cand.instrument, calendar, job.lo, job.hi, 1.0)
        lots = list(sz.log)
        tr5 = []
        if job.stress:
            tr5, _d5, _s5 = N._simulate_one(spec, cand.instrument, calendar, job.lo, job.hi, N.SENS_MULT)
    key = lambda t: (t[2], t[1])
    return {"trades": sorted(tr, key=key), "trades5": sorted(tr5, key=key), "lots": lots}
