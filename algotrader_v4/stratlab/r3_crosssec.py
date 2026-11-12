"""
stratlab/r3_crosssec.py — round 3, AI-trader cross-sectional rank strategies (ids r3_ai_crosssec_*).

Research code only: reads bar files, writes reports to a scratch directory; never touches a broker or live agent.

The per-symbol @strategy API cannot rank symbols against each other on the same day, so this module (like
attention.py / ltr.py) builds a (symbol x session) panel of DECISION-TIME features, ranks the cross-section each day,
settles trades with the lab's costs (costs.round_trip_cost, Rs 1,00,000 notional per trade) and judges them with the
UNCHANGED stratlab.protocol.evaluate (same 70/30 split, H1/H2 screen, PF, trade-count and symbol-share gates, FINAL
once through the cache's final_ledger.json; there is deliberately no --peek-final).

Pre-registration (rules, parameters, rationale) is in the scratchpad file r3/ai_crosssec/PREREGISTERED.txt; the
parameters below are copied from it and must not be tuned.

Look-ahead: two audits run on every DISCOVERY simulation.
  * feature audit: on sampled sessions the decision-time features are recomputed from the session's bars truncated at
    the decision minute and must be identical;
  * panel audit: on sampled decision days each strategy's scores are recomputed on a panel where every later column
    and every field of the decision day that is not known at the decision minute is NaN; they must be identical.

    cd algotrader_v4
    python -m stratlab.r3_crosssec --data <bars3y> --cache <cache> --out <scratch dir> [--ids a,b] [--screen-only]
    python -m stratlab.r3_crosssec ... --symbols 25 --counts-only     # correctness smoke: trade counts + audits only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

from . import data as D  # noqa: E402
from . import protocol as P  # noqa: E402
from . import sim  # noqa: E402
from .auction import block, quarters, tstat_by_date  # noqa: E402
from .ltr import CA_GAPS  # noqa: E402

NOTIONAL = 100_000.0
DEFAULT_SLIPPAGE_BPS = 2.0
STRESS_SLIPPAGE_BPS = 5.0
MIN_UNIVERSE = 20
SPECIAL_MEDIAN_BARS = 300
FILL_MODEL = "xsec1"            # bump if a fill rule here changes (part of the FINAL-ledger key)
COST_MODEL = f"{sim.COST_MODEL}+{FILL_MODEL}"

M = lambda s: int(s[:2]) * 60 + int(s[3:])   # noqa: E731  "HH:MM" -> minute of day
T0915, T0916, T0930, T1014, T1015, T1515, T1520 = (M(x) for x in
                                                   ("09:15", "09:16", "09:30", "10:14", "10:15", "15:15", "15:20"))
T_LATE = 24 * 60                # "known after the close"

# Field -> the minute from which it is known (a decision at the CLOSE of bar m may use fields with avail <= m).
AVAIL = {"O": T0915, "C1014": T1014, "V1014": T1014, "VWAP1014": T1014, "N1014": T1014,
         "P1520": T1520, "V1520": T1520,
         "C": T_LATE, "Vfull": T_LATE, "NC": T_LATE,
         "P0916o": T_LATE, "P0930o": T_LATE, "P1015o": T_LATE, "P1521o": T_LATE, "PexitMIS": T_LATE}
TIME_FIELDS = ("P0916o", "P0930o", "P1015o", "P1521o", "PexitMIS")   # each has a "<f>_t" bar-minute array


# ── per-session features ────────────────────────────────────────────────────
def _first_at(tt, m, within=5):
    i = int(np.searchsorted(tt, m))
    return i if i < len(tt) and int(tt[i]) < m + within else -1


def _last_le(tt, m, earliest):
    i = int(np.searchsorted(tt, m, side="right")) - 1
    return i if i >= 0 and int(tt[i]) >= earliest else -1


def one_session(t, o, h, l, c, v, upto: int = T_LATE) -> dict:
    """Features of one session from its bars (arrays of that session only). With `upto` < T_LATE the bars after
    minute `upto` are removed first (used by the feature audit)."""
    if upto < T_LATE:
        keep = t <= upto
        t, o, h, l, c, v = t[keep], o[keep], h[keep], l[keep], c[keep], v[keep]
    nan = float("nan")
    f = {k: nan for k in AVAIL}
    for k in TIME_FIELDS:
        f[k + "_t"] = -1
    if len(t) == 0:
        return f
    if int(t[0]) < T0930:
        f["O"] = float(o[0])
    f["C"] = float(c[-1])
    f["Vfull"] = float(v.sum())
    i = _last_le(t, T1014, T1014 - 4)
    if i >= 0:
        vv = v[:i + 1]
        tp = (h[:i + 1] + l[:i + 1] + c[:i + 1]) / 3.0
        f["C1014"] = float(c[i])
        f["V1014"] = float(vv.sum())
        f["VWAP1014"] = float((tp * vv).sum() / vv.sum()) if vv.sum() > 0 else float(tp[-1])
    i = _last_le(t, T1520, T1515)
    if i >= 0:
        f["P1520"] = float(c[i])
        f["V1520"] = float(v[:i + 1].sum())
        if i + 1 < len(t) and int(t[i + 1]) <= T1520 + 4:
            f["P1521o"], f["P1521o_t"] = float(o[i + 1]), int(t[i + 1])
    for name, m in (("P0916o", T0916), ("P0930o", T0930), ("P1015o", T1015)):
        j = _first_at(t, m)
        if j >= 0:
            f[name], f[name + "_t"] = float(o[j]), int(t[j])
    j = _first_at(t, T1515, 15)
    if j >= 0:
        f["PexitMIS"], f["PexitMIS_t"] = float(o[j]), int(t[j])
    else:
        j = int(np.searchsorted(t, T1515)) - 1
        if j >= 0:
            f["PexitMIS"], f["PexitMIS_t"] = float(c[j]), int(t[j])
    return f


def symbol_features(sd) -> list:
    out = []
    for k in range(sd.n_sessions):
        a, b = int(sd.starts[k]), int(sd.ends[k])
        f = one_session(sd.t[a:b], sd.o[a:b], sd.h[a:b], sd.l[a:b], sd.c[a:b], sd.v[a:b])
        f["date"] = int(sd.days[k])
        f["nbars"] = b - a
        out.append(f)
    return out


def audit_features(sd, every: int = 37) -> str | None:
    """Recompute the decision-time features of sampled sessions from bars truncated at each decision minute."""
    for k in range(0, sd.n_sessions, every):
        a, b = int(sd.starts[k]), int(sd.ends[k])
        arr = (sd.t[a:b], sd.o[a:b], sd.h[a:b], sd.l[a:b], sd.c[a:b], sd.v[a:b])
        full = one_session(*arr)
        for m in (T0915, T1014, T1520):
            cut = one_session(*arr, upto=m)
            for fld, av in AVAIL.items():
                if av <= m and not _same(full[fld], cut[fld]):
                    return f"{sd.symbol} {int(sd.days[k])}: feature {fld} differs when bars after {sim._hm(m)} are removed"
    return None


def _same(x, y) -> bool:
    return (x != x and y != y) or x == y


# ── panel ───────────────────────────────────────────────────────────────────
@dataclass
class Panel:
    symbols: list
    dates: np.ndarray                 # trading-panel sessions (special sessions removed)
    f: dict                           # field -> (n_sym, n_days) float array (NaN = no data)
    ti: dict                          # time field -> (n_sym, n_days) int bar minute (-1 = none)
    calendar: list                    # every session date (for the protocol split)
    special: list
    n_dates_all: int = 0

    def cut(self, hi: int) -> "Panel":
        k = int(np.searchsorted(self.dates, hi, side="right"))
        return Panel(self.symbols, self.dates[:k].copy(), {a: b[:, :k].copy() for a, b in self.f.items()},
                     {a: b[:, :k].copy() for a, b in self.ti.items()}, self.calendar, self.special)

    def masked(self, j: int, minute: int) -> "Panel":
        """Copy with every column after j removed and every field of column j not known at `minute` set to NaN."""
        p = self.cut(int(self.dates[j]))
        for fld, av in AVAIL.items():
            if av > minute:
                p.f[fld][:, j] = np.nan
        for fld in p.ti:
            if AVAIL[fld] > minute:
                p.ti[fld][:, j] = -1
        return p


def build_panel(feats: dict, nifty: list | None) -> Panel:
    syms = sorted(feats)
    counts = defaultdict(list)
    for s in syms:
        for r in feats[s]:
            counts[r["date"]].append(r["nbars"])
    calendar = sorted(counts)
    special = sorted(d for d in calendar if float(np.median(counts[d])) < SPECIAL_MEDIAN_BARS)
    sp = set(special)
    dates = np.array([d for d in calendar if d not in sp], np.int64)
    col = {int(d): j for j, d in enumerate(dates)}
    shp = (len(syms), len(dates))
    f = {k: np.full(shp, np.nan) for k in AVAIL}
    ti = {k: np.full(shp, -1, np.int32) for k in TIME_FIELDS}
    for i, s in enumerate(syms):
        for r in feats[s]:
            j = col.get(r["date"])
            if j is None:
                continue
            for k in AVAIL:
                if k not in ("N1014", "NC"):
                    f[k][i, j] = r[k]
            for k in TIME_FIELDS:
                ti[k][i, j] = r[k + "_t"]
    if nifty:
        for r in nifty:
            j = col.get(r["date"])
            if j is not None:
                f["N1014"][:, j] = r["C1014"]
                f["NC"][:, j] = r["C"]
    return Panel(syms, dates, f, ti, calendar, special, len(calendar))


# ── score helpers (column j = decision day; only columns <= j may be read) ──
def lag(x, j, n):
    return x[:, j - n] if j - n >= 0 else np.full(x.shape[0], np.nan)


def ret_since(p, j, price, n):
    """price on day j / last close n sessions before j - 1."""
    return price / lag(p.f["C"], j, n) - 1.0


def window(x, j, a, b):
    """columns j-a .. j-b inclusive (a >= b >= 0) as (n_sym, a-b+1); NaN-padded before the panel start."""
    lo, hi = j - a, j - b + 1
    if lo < 0:
        pad = np.full((x.shape[0], -lo), np.nan)
        return np.concatenate([pad, x[:, 0:hi]], axis=1) if hi > 0 else np.full((x.shape[0], a - b + 1), np.nan)
    return x[:, lo:hi]


def zscore(x):
    """Cross-sectional z-score over finite entries, winsorised at +/-3 (NaN stays NaN)."""
    m = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if m.sum() < 3:
        return out
    mu, sd = x[m].mean(), x[m].std()
    if not sd > 0:
        out[m] = 0.0
        return out
    out[m] = np.clip((x[m] - mu) / sd, -3.0, 3.0)
    return out


def _all(x):
    return np.ones(x.shape, bool)


# score functions: (panel, j, params) -> dict(score=..., long_ok=..., short_ok=..., side=...)
def sc_rev1d_1520(p, j, q):
    r = ret_since(p, j, p.f["P1520"][:, j], 1)
    return {"score": r}


def sc_rev1d_yday(p, j, q):
    r = lag(p.f["C"], j, 1) / lag(p.f["C"], j, 2) - 1.0
    ok = np.isfinite(p.f["O"][:, j])                      # the stock opened today (known at 09:15)
    return {"score": np.where(ok, r, np.nan)}


def sc_rev5d(p, j, q):
    return {"score": ret_since(p, j, p.f["P1520"][:, j], 5)}


def _mom6_1(p, j):
    return lag(p.f["C"], j, 21) / lag(p.f["C"], j, 126) - 1.0


def sc_mom6_1(p, j, q):
    ok = np.isfinite(p.f["P1520"][:, j])
    return {"score": np.where(ok, _mom6_1(p, j), np.nan)}


def sc_rs1015(p, j, q):
    c = p.f["C1014"][:, j]
    rs = (c / lag(p.f["C"], j, 1) - 1.0) - (p.f["N1014"][:, j] / lag(p.f["NC"], j, 1) - 1.0)
    vw = p.f["VWAP1014"][:, j]
    return {"score": rs, "long_ok": c > vw, "short_ok": c < vw}


def sc_gaprank(p, j, q):
    g = p.f["O"][:, j] / lag(p.f["C"], j, 1) - 1.0
    g = np.where(np.abs(g) <= q["max_abs_gap"], g, np.nan)
    return {"score": g, "long_ok": g < 0, "short_ok": g > 0}


def sc_onmom20(p, j, q):
    n = q["n"]
    on = np.log(window(p.f["O"], j, n - 1, 0) / window(p.f["C"], j, n, 1))
    s = on.sum(axis=1)                                     # NaN if any day is missing
    ok = np.isfinite(p.f["P1520"][:, j])
    return {"score": np.where(ok, s, np.nan)}


def sc_idmom20(p, j, q):
    n = q["n"]
    idr = np.log(window(p.f["C"], j, n, 1) / window(p.f["O"], j, n, 1))
    ok = np.isfinite(p.f["O"][:, j])
    return {"score": np.where(ok, idr.sum(axis=1), np.nan)}


def sc_volsurp1520(p, j, q):
    base = window(p.f["Vfull"], j, q["n"], 1).mean(axis=1)
    vs = p.f["V1520"][:, j] / np.where(base > 0, base, np.nan)
    up = p.f["P1520"][:, j] > lag(p.f["C"], j, 1)
    return {"score": vs, "long_ok": up}


def sc_volsurp1014(p, j, q):
    base = window(p.f["V1014"], j, q["n"], 1).mean(axis=1)
    vs = p.f["V1014"][:, j] / np.where(base > 0, base, np.nan)
    side = np.sign(p.f["C1014"][:, j] - lag(p.f["C"], j, 1))
    return {"score": np.where(side != 0, vs, np.nan), "side": side}


def _sigma(p, j, n):
    c = window(p.f["C"], j, n + 1, 1)
    r = np.log(c[:, 1:] / c[:, :-1])
    return r.std(axis=1, ddof=1)                          # NaN if any day is missing


def sc_lowvol(p, j, q):
    ok = np.isfinite(p.f["P1520"][:, j])
    return {"score": np.where(ok, _sigma(p, j, q["n"]), np.nan)}


def sc_composite3(p, j, q):
    a = -ret_since(p, j, p.f["P1520"][:, j], 5)
    b = _mom6_1(p, j)
    c = -_sigma(p, j, 60)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = (np.where(ok, x, np.nan) for x in (a, b, c))
    return {"score": (zscore(a) + zscore(b) + zscore(c)) / 3.0}


# ── walk-forward ridge ──────────────────────────────────────────────────────
RIDGE_FEATURES = ("ret1", "r5", "ret20", "sigma20", "vs1520", "gap", "onmom20", "idmom20")


def ridge_features(p, j) -> np.ndarray:
    """(n_sym, 8) per-day z-scored features known at 15:20 of day j (NaN rows = ineligible)."""
    px = p.f["P1520"][:, j]
    base = window(p.f["Vfull"], j, 20, 1).mean(axis=1)
    on = np.log(window(p.f["O"], j, 19, 0) / window(p.f["C"], j, 20, 1)).sum(axis=1)
    idr = np.log(window(p.f["C"], j, 20, 1) / window(p.f["O"], j, 20, 1)).sum(axis=1)
    raw = [ret_since(p, j, px, 1), ret_since(p, j, px, 5), ret_since(p, j, px, 20), _sigma(p, j, 20),
           p.f["V1520"][:, j] / np.where(base > 0, base, np.nan), p.f["O"][:, j] / lag(p.f["C"], j, 1) - 1.0, on, idr]
    X = np.column_stack(raw)
    ok = np.isfinite(X).all(axis=1)
    X[~ok] = np.nan
    return np.column_stack([zscore(X[:, k]) for k in range(X.shape[1])])


def ridge_label(p, d, h):
    """Cross-sectionally demeaned forward return P1521o[d] -> P1521o[d+h]; reads column d+h (caller guarantees < t)."""
    y = p.f["P1521o"][:, d + h] / p.f["P1521o"][:, d] - 1.0
    m = np.isfinite(y)
    if m.sum() < MIN_UNIVERSE:
        return np.full(y.shape, np.nan)
    y = np.where(m, y - y[m].mean(), np.nan)
    return y


def sc_wfridge(p, j, q, state=None):
    """Score = ridge prediction from a model trained ONLY on decision days d with d + h <= j - 1."""
    state = state if state is not None else {}
    h, alpha, refit, min_days = q["h"], q["alpha"], q["refit"], q["min_train_days"]
    force = state.pop("audit_fit_at", None)          # audit: refit as of the same day the live state was fitted
    if force is not None or "fit_j" not in state or j - state["fit_j"] >= refit:
        fj = j if force is None else force
        last_d = fj - 1 - h                                # label exit column d+h <= j-1  (strictly before day j)
        XS, YS, ndays = [], [], 0
        for d in range(0, last_d + 1):
            X = state.setdefault("X", {}).get(d)
            if X is None:
                X = ridge_features(p, d)
                state["X"][d] = X
            y = ridge_label(p, d, h)
            m = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if m.sum() >= MIN_UNIVERSE:
                XS.append(X[m]); YS.append(y[m]); ndays += 1
        state["fit_j"] = fj
        if ndays >= min_days:
            Xa, ya = np.vstack(XS), np.concatenate(YS)
            state["beta"] = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(Xa.shape[1]), Xa.T @ ya)
        else:
            state["beta"] = None
        state["ndays"] = ndays
    if state.get("beta") is None:
        return {"score": np.full(len(p.symbols), np.nan)}
    X = ridge_features(p, j)
    return {"score": X @ state["beta"]}


# ── strategy specs ──────────────────────────────────────────────────────────
@dataclass
class XSpec:
    id: str
    kind: str                 # intraday | swing | futures
    family: str
    desc: str
    score: object
    mode: str                 # mis | overnight | positional
    select: str               # long_bot | long_top | ls_bot | ls_top | dir
    decision: int             # minute whose CLOSE the decision is made at
    entry: str                # price field of the entry fill
    params: dict = field(default_factory=dict)
    underlying: str | None = None

    @property
    def horizon(self) -> str:
        return "intraday" if self.kind == "intraday" else "positional"

    @property
    def cost_kind(self) -> str:
        return {"intraday": "equity_mis", "swing": "equity_cnc", "futures": "future"}[self.kind]

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(fn) for fn in (self.score, one_session, settle, simulate_spec, pick))
        blob = repr((self.id, self.kind, self.mode, self.select, self.decision, self.entry,
                     sorted(self.params.items()), MIN_UNIVERSE, SPECIAL_MEDIAN_BARS, sorted(CA_GAPS), src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _S(id_, kind, family, desc, score, mode, select, decision, entry, **params):
    params.setdefault("k", 5)
    params.setdefault("H", 0 if mode == "mis" else 1)
    params.setdefault("lookback", 1)
    return XSpec("r3_ai_crosssec_" + id_, kind, family, desc, score, mode, select, decision, entry, params)


SPECS = [
    _S("rev1d_bot5_cnc", "swing", "st_reversal_1d",
       "Every session at 15:20 buy the 5 stocks with the lowest return since yesterday's close; CNC, sell next 15:21",
       sc_rev1d_1520, "positional", "long_bot", T1520, "P1521o", H=1, lookback=1),
    _S("rev1d_ls5_fut", "futures", "st_reversal_1d",
       "Every session at 15:20 long the 5 worst and short the 5 best returns since yesterday's close; futures, 1 session",
       sc_rev1d_1520, "positional", "ls_bot", T1520, "P1521o", H=1, lookback=1),
    _S("rev1d_bot5_mis", "intraday", "st_reversal_1d",
       "Buy the 5 worst yesterday close-to-close returns at the 09:30 open, MIS to 15:15",
       sc_rev1d_yday, "mis", "long_bot", T0930 - 1, "P0930o", lookback=2),
    _S("rev5d_bot5_cnc", "swing", "st_reversal_5d",
       "Every 5 sessions at 15:20 buy the 5 lowest 5-session returns; CNC, hold 5 sessions",
       sc_rev5d, "positional", "long_bot", T1520, "P1521o", H=5, lookback=5),
    _S("rev5d_ls5_fut", "futures", "st_reversal_5d",
       "Every 5 sessions at 15:20 long the 5 lowest and short the 5 highest 5-session returns; futures, hold 5",
       sc_rev5d, "positional", "ls_bot", T1520, "P1521o", H=5, lookback=5),
    _S("mom6_1_top5_cnc", "swing", "momentum_6_1",
       "Every 20 sessions buy the 5 highest 6-1 month returns C[t-21]/C[t-126]-1; CNC, hold 20 sessions",
       sc_mom6_1, "positional", "long_top", T1520, "P1521o", H=20, lookback=126),
    _S("mom6_1_ls5_fut", "futures", "momentum_6_1",
       "Every 20 sessions long the top 5 and short the bottom 5 6-1 month returns; futures, hold 20 sessions",
       sc_mom6_1, "positional", "ls_top", T1520, "P1521o", H=20, lookback=126),
    _S("rs1015_ls5_vwap", "intraday", "rel_strength_1015",
       "At 10:14 long the 5 strongest stocks vs NIFTY since yesterday's close above VWAP, short the 5 weakest below VWAP; MIS to 15:15",
       sc_rs1015, "mis", "ls_top", T1014, "P1015o", lookback=1),
    _S("rs1015_ls3_vwap", "intraday", "rel_strength_1015",
       "At 10:14 long the 3 strongest stocks vs NIFTY since yesterday's close above VWAP, short the 3 weakest below VWAP; MIS to 15:15",
       sc_rs1015, "mis", "ls_top", T1014, "P1015o", k=3, lookback=1),
    _S("gaprank_fade_ls5", "intraday", "gap_rank",
       "At 09:29 short the 5 largest gaps up and buy the 5 largest gaps down (|gap| <= 20 %) at the 09:30 open; MIS to 15:15",
       sc_gaprank, "mis", "ls_bot", T0930 - 1, "P0930o", lookback=1, max_abs_gap=0.20),
    _S("onmom20_top5_fut", "futures", "overnight_persist",
       "At 15:20 buy the 5 highest 20-session sums of overnight returns; stock futures 15:21 -> next 09:15 open",
       sc_onmom20, "overnight", "long_top", T1520, "P1521o", lookback=21, n=20),
    _S("idmom20_ls5_mis", "intraday", "intraday_persist",
       "At 09:15 long the 5 highest and short the 5 lowest 20-session sums of open-to-close returns; 09:16 open, MIS to 15:15",
       sc_idmom20, "mis", "ls_top", T0915, "P0916o", lookback=20, n=20),
    _S("volsurp_up5_cnc20", "swing", "volume_surprise",
       "Every 20 sessions at 15:20 buy the 5 up-stocks with the highest volume vs their 20-session mean; CNC, hold 20",
       sc_volsurp1520, "positional", "long_top", T1520, "P1521o", H=20, lookback=20, n=20),
    _S("volsurp1015_dir5", "intraday", "volume_surprise",
       "At 10:14 take the 5 highest 09:15-10:14 volume surprises (vs 20 sessions) in the sign of their move since yesterday; MIS to 15:15",
       sc_volsurp1014, "mis", "dir", T1014, "P1015o", lookback=20, n=20),
    _S("lowvol_bot5_cnc20", "swing", "low_volatility",
       "Every 20 sessions buy the 5 stocks with the lowest 60-session return volatility; CNC, hold 20 sessions",
       sc_lowvol, "positional", "long_bot", T1520, "P1521o", H=20, lookback=61, n=60),
    _S("lowvol_ls5_fut20", "futures", "low_volatility",
       "Every 20 sessions long the 5 lowest and short the 5 highest 60-session volatility stocks; futures, hold 20",
       sc_lowvol, "positional", "ls_bot", T1520, "P1521o", H=20, lookback=61, n=60),
    _S("composite3_top5_cnc10", "swing", "composite",
       "Every 10 sessions buy the top 5 of an equal-weight z-composite of 5-day reversal, 6-1 momentum and low volatility; CNC, hold 10",
       sc_composite3, "positional", "long_top", T1520, "P1521o", H=10, lookback=126),
    _S("wfridge_ls5_fut5", "futures", "walkforward_ridge",
       "Every 5 sessions long/short the top/bottom 5 of a walk-forward ridge forecast of 5-session returns (8 features); futures, hold 5",
       sc_wfridge, "positional", "ls_top", T1520, "P1521o", H=5, lookback=21, h=5, alpha=10.0, refit=21,
       min_train_days=100),
]
BY_ID = {s.id: s for s in SPECS}


# ── selection and settlement ────────────────────────────────────────────────
def pick(sc: dict, k: int, how: str, elig: np.ndarray) -> list:
    """[(sym_index, side)]; ties by symbol index (symbols are sorted by name)."""
    s = sc["score"]
    ok = elig & np.isfinite(s)
    if ok.sum() < MIN_UNIVERSE:
        return []
    lo_ok = ok & sc.get("long_ok", _all(s))
    sh_ok = ok & sc.get("short_ok", _all(s))
    idx = np.arange(len(s))

    def order(mask, ascending):
        c = idx[mask]
        key = s[c] if ascending else -s[c]
        return [int(i) for i in c[np.lexsort((c, key))]]

    if how == "dir":
        side = sc["side"]
        return [(i, int(side[i])) for i in order(ok, False)[:k]]
    if how in ("long_bot", "long_top"):
        return [(i, 1) for i in order(lo_ok, how == "long_bot")[:k]]
    longs = order(lo_ok, how == "ls_bot")[:k]
    shorts = [i for i in order(sh_ok, how == "ls_top") if i not in longs][:k]
    return [(i, 1) for i in longs] + [(i, -1) for i in shorts]


def settle(sp: XSpec, sym, date, side, entry_t, entry_px, exit_date, exit_t, exit_px, slip, reason="time") -> tuple:
    fill = entry_px * (1.0 + side * slip)
    xp = exit_px * (1.0 - side * slip)
    qty = max(1, int(NOTIONAL // fill))
    gross = side * (xp - fill) * qty
    buy_v, sell_v = (fill * qty, xp * qty) if side > 0 else (xp * qty, fill * qty)
    cost = round_trip_cost(buy_v, sell_v, sp.cost_kind)
    return (sp.id, sym, int(date), int(side), sim._hm(entry_t), round(fill, 4), int(exit_date), sim._hm(exit_t),
            round(xp, 4), qty, round(gross, 2), cost, round(gross - cost, 2), reason)


def ca_blocked(p: Panel, sp: XSpec, j: int, e: int) -> np.ndarray:
    """Symbols with a known corporate-action gap in (date[j - lookback], date[e]]."""
    out = np.zeros(len(p.symbols), bool)
    a = int(p.dates[max(0, j - sp.params["lookback"])])
    b = int(p.dates[min(e, len(p.dates) - 1)])
    for (s, g) in CA_GAPS:
        if s in p.symbols and a < g <= b:
            out[p.symbols.index(s)] = True
    return out


def decision_days(sp: XSpec, cols: list) -> list:
    if sp.mode == "mis":
        return list(cols)
    if sp.mode == "overnight":
        return list(cols[:-1])
    return list(cols[:-1:sp.params["H"]]) if len(cols) > 1 else []


def simulate_spec(sp: XSpec, panel: Panel, lo: int, hi: int, slip: float, audit_every: int = 0) -> dict:
    """Trades with entry dates in [lo, hi]; nothing after hi is read (the panel is cut at hi first)."""
    p = panel.cut(hi)
    cols = [j for j, d in enumerate(p.dates) if lo <= d <= hi]
    out = {"trades": [], "skipped": 0, "lookahead": None, "error": None, "days": 0}
    if not cols:
        return out
    last = cols[-1]
    state: dict = {}
    days = decision_days(sp, cols)
    for n_day, j in enumerate(days):
        if sp.mode == "mis":
            e = j
        elif sp.mode == "overnight":
            e = j + 1
        else:
            e = min(j + sp.params["H"], last)
        elig = ~ca_blocked(p, sp, j, e)
        sc = sp.score(p, j, sp.params, state) if sp.score is sc_wfridge else sp.score(p, j, sp.params)
        if audit_every and n_day % audit_every == 0:
            q = p.masked(j, sp.decision)
            sc2 = sp.score(q, j, sp.params, {"audit_fit_at": state.get("fit_j")}) if sp.score is sc_wfridge else sp.score(q, j, sp.params)
            a, b = sc["score"], sc2["score"]
            if not np.array_equal(np.isfinite(a), np.isfinite(b)) or \
                    not np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=0, atol=1e-12):
                out["lookahead"] = f"scores on {int(p.dates[j])} change when later data are hidden"
        chosen = pick(sc, sp.params["k"], sp.select, elig)
        if chosen:
            out["days"] += 1
        d = int(p.dates[j])
        for i, side in chosen:
            sym = p.symbols[i]
            ep = p.f[sp.entry][i, j]
            et = int(p.ti[sp.entry][i, j])
            if not np.isfinite(ep):
                out["skipped"] += 1
                continue
            if sp.mode == "mis":
                xp, xt, xd = p.f["PexitMIS"][i, j], int(p.ti["PexitMIS"][i, j]), d
                if not (np.isfinite(xp) and xt > et):
                    out["skipped"] += 1
                    continue
            elif sp.mode == "overnight":
                xp, xt, xd = p.f["O"][i, e], T0915, int(p.dates[e])
                if not np.isfinite(xp):
                    out["skipped"] += 1
                    continue
            else:
                xe = next((c for c in range(e, last + 1) if np.isfinite(p.f["P1521o"][i, c])), None)
                if xe is None:
                    xe = next((c for c in range(e, j, -1) if np.isfinite(p.f["P1521o"][i, c])), None)
                if xe is None:
                    out["skipped"] += 1
                    continue
                xp, xt, xd = p.f["P1521o"][i, xe], int(p.ti["P1521o"][i, xe]), int(p.dates[xe])
            out["trades"].append(settle(sp, sym, d, side, et, float(ep), xd, xt, float(xp), slip))
    out["trades"].sort(key=lambda t: (t[2], t[1], t[3]))
    return out


def make_simulator(panel: Panel, slippage_bps: float = DEFAULT_SLIPPAGE_BPS, audit_every: int = 0):
    def simulate(specs, lo, hi):
        res = {}
        for sp in specs:
            try:
                res[sp.id] = simulate_spec(sp, panel, lo, hi, slippage_bps / 1e4, audit_every)
            except Exception as ex:                      # reported as INVALID by the protocol
                res[sp.id] = {"trades": [], "skipped": 0, "lookahead": None, "error": f"{type(ex).__name__}: {ex}"}
        return res
    return simulate


# ── loading ─────────────────────────────────────────────────────────────────
def load(data_dir: str, cache: str | None, n_symbols: int = 0, verbose: bool = True):
    man = D.prepare(data_dir, cache or None, verbose=False)
    cdir = man["cache_dir"]
    syms = sorted(man["symbols"])
    if n_symbols:
        syms = syms[:n_symbols]
    feats, audit = {}, None
    for s in syms:
        sd = D.load_symbol(cdir, s)
        if sd is None:
            continue
        feats[s] = symbol_features(sd)
        audit = audit or audit_features(sd)
        if verbose:
            print(f"  {s}: {sd.n_sessions} sessions", flush=True)
    nsd = D.load_symbol(cdir, "_NIFTY50_min")
    nifty = symbol_features(nsd) if nsd is not None else None
    return build_panel(feats, nifty), audit, cdir


# ── reporting ───────────────────────────────────────────────────────────────
def _fmt(k, b):
    return (f"  {k:12s} n={b['trades']:5d} net={b['net']:>12,.0f} PF={b['pf']:.2f} win={b['win_pct']}% "
            f"gross={b['gross_bps']}bps net={b['net_bps']}bps t_gross={b['t_gross']} "
            f"net_ex_best5d={b['net_ex_best5_dates']:,.0f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.r3_crosssec",
                                 description="Round-3 AI-trader cross-sectional rank strategies through the lab protocol")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--ids", default="", help="comma-separated ids (default: all)")
    ap.add_argument("--symbols", type=int, default=0, help="first N symbols only (smoke tests)")
    ap.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    ap.add_argument("--stress-bps", type=float, default=STRESS_SLIPPAGE_BPS)
    ap.add_argument("--jobs", type=int, default=1, help="accepted for CLI compatibility; the panel run is serial")
    ap.add_argument("--screen-only", action="store_true", help="never simulate FINAL (no ledger entry)")
    ap.add_argument("--counts-only", action="store_true",
                    help="correctness smoke: DISCOVERY trade counts and the look-ahead audits only; no P&L shown or written")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)                               # deliberately no --peek-final
    specs = SPECS if not a.ids else [BY_ID[x.strip()] for x in a.ids.split(",") if x.strip()]
    if a.list:
        for s in specs:
            print(f"{s.id:42s} {s.kind:9s} {s.family:20s} {s.desc}")
        return 0
    out = D.assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    panel, faudit, cdir = load(a.data, a.cache, a.symbols, verbose=False)
    split = P.split_sessions(panel.calendar)
    print(f"panel: {len(panel.symbols)} symbols, {len(panel.dates)} trading sessions "
          f"({len(panel.special)} special removed), calendar {len(panel.calendar)}; feature audit: {faudit or 'OK'}")
    if a.counts_only:
        sim_d = make_simulator(panel, a.slippage_bps, audit_every=7)
        r = sim_d(specs, split.discovery[0], split.discovery[1])
        lines = []
        for s in specs:
            x = r[s.id]
            n = len(x["trades"])
            sides = (sum(1 for t in x["trades"] if t[3] > 0), sum(1 for t in x["trades"] if t[3] < 0))
            lines.append(f"{s.id:42s} DISC trades={n:5d} (long {sides[0]}, short {sides[1]}) trade-days={x['days']} "
                         f"skipped={x['skipped']} lookahead={x['lookahead'] or 'OK'} error={x['error'] or '-'}")
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(out, "counts.txt"), "w") as fh:
            fh.write(f"feature audit: {faudit or 'OK'}\n{txt}\n")
        return 0
    simulate = make_simulator(panel, a.slippage_bps, audit_every=7)
    if faudit:                                            # a feature look-ahead invalidates everything
        base = simulate

        def simulate(sp_, lo, hi):                        # noqa: F811
            r = base(sp_, lo, hi)
            for v in r.values():
                v["lookahead"] = v["lookahead"] or faudit
            return r
    results, trades = P.evaluate(specs, panel.calendar, simulate, n_symbols=len(panel.symbols),
                                 ledger_path=os.path.join(cdir, "final_ledger.json"),
                                 dataset=os.path.realpath(a.data), screen_only=a.screen_only, cost_model=COST_MODEL)
    stress = make_simulator(panel, a.stress_bps)
    lines = [f"Round-3 AI cross-sectional | {len(panel.symbols)} symbols, {split.n} sessions | H1 {split.h1} "
             f"H2 {split.h2} FINAL {split.final} | slippage {a.slippage_bps} bps/side (stress {a.stress_bps}) | "
             f"cost model {COST_MODEL} | special sessions removed {len(panel.special)} | feature audit "
             f"{faudit or 'OK'}" + (" | SCREEN ONLY" if a.screen_only else "")]
    rows = []
    for r in results:
        sid, sp = r["id"], r["spec"]
        tr = trades[sid]
        disc = [t for t in tr if split.discovery[0] <= t[2] <= split.discovery[1]]
        h1 = [t for t in disc if t[2] <= split.h1[1]]
        h2 = [t for t in disc if t[2] >= split.h2[0]]
        fin = [t for t in tr if t[2] >= split.final[0]]
        st = stress([sp], split.discovery[0], split.discovery[1])[sid]["trades"]
        ssc = P.screen(st, split, sp.horizon)
        lines.append(f"\n{sid} [{sp.kind}/{sp.family}]: {r['verdict']} ({r['why']}) {r.get('final_note', '')}")
        lines.append(f"  {sp.desc}")
        for k, b in (("H1", h1), ("H2", h2), ("DISC", disc)):
            lines.append(_fmt(k, block(b)))
        lines.append(_fmt(f"DISC@{a.stress_bps:g}bps", block(st)) + f" screen={'PASS' if ssc['passed'] else 'FAIL'}")
        if disc:
            ls = {sd_: round(sum(t[12] for t in disc if t[3] == sd_), 0) for sd_ in (1, -1)}
            lines.append(f"  DISC long net {ls[1]:,.0f} short net {ls[-1]:,.0f}; top symbol "
                         f"{r['screen']['top_symbol']} {r['screen']['top_share']:.1%}; quarters {quarters(disc)}")
        if r["final"] is not None:
            lines.append(_fmt("FINAL", block(fin)) + f"  t_net_by_date={round(tstat_by_date(fin, 12), 2)}")
        rows.append({"id": sid, "kind": sp.kind, "family": sp.family, "verdict": r["verdict"], "why": r["why"],
                     "disc_trades": len(disc), "h1_net": P.stats(h1)["net"], "h2_net": P.stats(h2)["net"],
                     "disc_net": P.stats(disc)["net"], "disc_pf": P.stats(disc)["pf"],
                     "disc_net_5bps": P.stats(st)["net"], "screen_5bps": ssc["passed"],
                     "final_trades": r["final"]["trades"] if r["final"] else "",
                     "final_net": r["final"]["net"] if r["final"] else "",
                     "final_pf": r["final"]["pf"] if r["final"] else "", "fingerprint": sp.fingerprint})
        os.makedirs(os.path.join(out, "trades"), exist_ok=True)
        with open(os.path.join(out, "trades", f"{sid}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(sim.TRADE_FIELDS)
            w.writerows(tr)
    lines.append("\n" + P.luck_reminder(results))
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
