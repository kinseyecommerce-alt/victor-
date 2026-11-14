"""
stratlab/factor_momentum.py — pre-registered slow factor-momentum portfolio test (research only).

Pre-registration: stratlab/prereg/momentum_v1.json (a copy of scratchpad momentum_oos/PREREGISTERED_MOMENTUM.json,
sha256 PREREG_SHA256 below). Read it before changing anything here: every rule in this file is fixed by it and
nothing may be tuned after results. A changed rule is a new strategy and needs a new pre-registration and id.

What it does
  * Monthly decisions at the close of the last session of each calendar month (decision session t). The score uses
    closes up to and including t only: average of the cross-sectional z-scores of volatility-adjusted 12-1 and 6-1
    month momentum (R12 = C[t-21]/C[t-252]-1, R6 = C[t-21]/C[t-126]-1, divided by the annualised std of the last
    252 daily log returns; z clipped to +-3).
  * Execution at the OPEN of the next session (the first session of the next month). Top 20, equal weight
    (1/min(20, n eligible), cap 10 %), no-trade band 0.5 % of equity for names that stay held.
  * Optional trend filter: NIFTY close < SMA200 at t -> all cash (checked every month-end).
  * Costs: costs.round_trip_cost(..., "equity_cnc") per buy and per sell (STT, stamp, exchange, SEBI, GST, DP) plus
    10 bps slippage per side on traded value.
  * Two eras: 2013-2023 point-in-time top-100 by turnover from nse_old (delisted names kept) and 2023-11..2026-11 on
    the ~97 current NIFTY-100 names in bars3y (SURVIVORSHIP-BIASED; history spliced from nse_old).

CLI (writes only outside the repository; refuses to re-run a variant already in the ledger):
  python -m stratlab.factor_momentum --prereg <PREREGISTERED_MOMENTUM.json> --old <nse_old dir> \
      --bars <bars3y dir> --nse <nse dir> --ledger <ledger.json> --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:                                    # run as python -m stratlab.factor_momentum from algotrader_v4/
    from costs import round_trip_cost
except ImportError:                     # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from costs import round_trip_cost

PREREG_SHA256 = "2a7cf8bf595cd9f7289aa55d92be0d904b03e1897bf4063f3c62bf2fc372a202"

# ── pre-registered constants (do not tune) ──────────────────────────────────────────────────────────────────────
SKIP, M6, M12 = 21, 126, 252
VOL_N = 252
Z_CLIP = 3.0
MIN_REAL_FRAC = 0.90
TOP_N = 20
MAX_W = 0.10
BAND = 0.005
SLIP_BPS = 10.0
SMA_N = 200
CAPITAL = 1_000_000.0
SEMI_EVERY = 6

VARIANTS = {
    "fm_monthly_nofilter":    {"rebalance": "monthly",    "trend_filter": False},
    "fm_monthly_filter":      {"rebalance": "monthly",    "trend_filter": True},
    "fm_semiannual_nofilter": {"rebalance": "semiannual", "trend_filter": False},
    "fm_semiannual_filter":   {"rebalance": "semiannual", "trend_filter": True},
}

ERA1 = {"first_decision": 20130131, "end": 20231109, "h1_end": 20171231}
ERA2 = {"first_decision": 20231130, "splice_last_old": 20231109, "junction": 20231110}
SPLICE_MAP = {"ETERNAL": "ZOMATO", "UNITDSPR": "MCDOWELL-N"}


# ── data container ──────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Panel:
    dates: np.ndarray                    # (N,) int yyyymmdd market sessions
    symbols: list                        # (M,)
    close: np.ndarray                    # (N, M) real closes, NaN where the stock did not trade
    open: np.ndarray                     # (N, M) real opens, NaN where the stock did not trade
    nifty: np.ndarray                    # (N,) NIFTY close (forward-filled)
    universe: dict | None = None         # yyyymm -> set of symbols (None = every symbol, every month)
    ca_gaps: dict = field(default_factory=dict)   # symbol -> list of gap dates (eligibility exclusion)
    label: str = ""

    def __post_init__(self):
        self.dates = np.asarray(self.dates, np.int64)
        self.close = np.asarray(self.close, float)
        self.open = np.asarray(self.open, float)
        self.nifty = ffill_1d(np.asarray(self.nifty, float))
        self.cf = ffill_2d(self.close)
        self.real = np.isfinite(self.close)


def ffill_1d(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for i in range(1, len(x)):
        if not np.isfinite(x[i]):
            x[i] = x[i - 1]
    return x


def ffill_2d(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    for i in range(1, a.shape[0]):
        m = ~np.isfinite(a[i])
        a[i, m] = a[i - 1, m]
    return a


def decision_sessions(dates) -> list[int]:
    """Indices of the last session of each calendar month that has a following session (the exchange calendar is
    known in advance, so knowing that t is the month's last session is not look-ahead)."""
    d = np.asarray(dates, np.int64) // 100
    return [i for i in range(len(d) - 1) if d[i] != d[i + 1]]


# ── signal ──────────────────────────────────────────────────────────────────────────────────────────────────────

def eligible(p: Panel, t: int, exec_month: int) -> np.ndarray:
    """Boolean (M,) eligibility at decision session t for a portfolio executed in month exec_month (yyyymm)."""
    M = len(p.symbols)
    ok = np.zeros(M, bool)
    if t < M12:
        return ok
    win = slice(t - M12, t + 1)                                   # 253 sessions t-252..t
    frac = p.real[win].sum(axis=0) / (M12 + 1)
    ok = p.real[t] & (frac >= MIN_REAL_FRAC) & np.isfinite(p.cf[t - M12])
    if p.universe is not None:
        mem = p.universe.get(int(exec_month), set())
        ok &= np.array([s in mem for s in p.symbols])
    if p.ca_gaps:
        lo, hi = p.dates[t - M12], p.dates[t]
        for j, s in enumerate(p.symbols):
            if ok[j] and any(lo < g <= hi for g in p.ca_gaps.get(s, ())):
                ok[j] = False
    return ok


def scores(p: Panel, t: int, exec_month: int) -> np.ndarray:
    """(M,) momentum score at decision session t (NaN = ineligible). Reads rows <= t only."""
    ok = eligible(p, t, exec_month)
    out = np.full(len(p.symbols), np.nan)
    if not ok.any():
        return out
    cf = p.cf[: t + 1]
    c_skip, c6, c12 = cf[t - SKIP], cf[t - M6], cf[t - M12]
    lr = np.diff(np.log(cf[t - VOL_N: t + 1]), axis=0)            # 252 log returns ending at t
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = lr.std(axis=0) * math.sqrt(252)
        ok &= np.isfinite(sig) & (sig > 0) & np.isfinite(c_skip) & np.isfinite(c6) & np.isfinite(c12)
        v6 = (c_skip / c6 - 1) / sig
        v12 = (c_skip / c12 - 1) / sig
    if ok.sum() < 2:
        return out

    def z(v):
        m, s = v[ok].mean(), v[ok].std()
        return np.clip((v - m) / s, -Z_CLIP, Z_CLIP) if s > 0 else np.zeros_like(v)

    sc = (z(v6) + z(v12)) / 2
    out[ok] = sc[ok]
    return out


def select(p: Panel, sc: np.ndarray) -> dict:
    """Top-N symbols -> target weight."""
    idx = [j for j in range(len(sc)) if np.isfinite(sc[j])]
    idx.sort(key=lambda j: (-sc[j], p.symbols[j]))
    pick = idx[:TOP_N]
    if not pick:
        return {}
    w = min(1.0 / len(pick), MAX_W)
    return {j: w for j in pick}


def sma(x: np.ndarray, i: int, n: int = SMA_N) -> float:
    if i + 1 < n:
        return float("nan")
    return float(np.mean(x[i - n + 1: i + 1]))


# ── costs ───────────────────────────────────────────────────────────────────────────────────────────────────────

def trade_costs(value: float, side: str) -> tuple[float, float]:
    """(statutory, slippage) rupees for one delivery buy or sell of `value` rupees of one scrip on one day."""
    if value <= 0:
        return 0.0, 0.0
    stat = round_trip_cost(value, 0.0, "equity_cnc") if side == "BUY" else round_trip_cost(0.0, value, "equity_cnc")
    return float(stat), value * SLIP_BPS / 1e4


# ── backtest ────────────────────────────────────────────────────────────────────────────────────────────────────

def backtest(p: Panel, rebalance: str, trend_filter: bool, first_decision: int, end: int | None = None,
             capital: float = CAPITAL, cost_fn=trade_costs) -> dict:
    """Run one variant. Returns the daily equity curve (from the first decision close) and trade statistics."""
    dates = p.dates
    last = len(dates) - 1 if end is None else int(np.searchsorted(dates, end, side="right")) - 1
    decs = [t for t in decision_sessions(dates[: last + 1]) if dates[t] >= first_decision]
    if not decs:
        raise ValueError("no decision session in range")
    t0 = decs[0]
    decset = {t: k for k, t in enumerate(decs)}
    M = len(p.symbols)
    sh = np.zeros(M)
    cash = capital
    eq_dates, eq = [int(dates[t0])], [capital]
    stat_tot = slip_tot = traded = 0.0
    stale_exits, n_rebal, n_cash_switch = 0, 0, 0
    pending = None                                                  # target {j: w} to execute at next open
    in_filter_cash = False
    log = []

    def decide(t):
        nonlocal in_filter_cash, n_cash_switch
        k = decset[t]
        scheduled = rebalance == "monthly" or k % SEMI_EVERY == 0
        exec_month = int(dates[t + 1]) // 100
        if trend_filter and p.nifty[t] < sma(p.nifty, t):
            if not in_filter_cash:
                n_cash_switch += 1
            in_filter_cash = True
            return {}
        if scheduled or in_filter_cash:
            in_filter_cash = False
            return select(p, scores(p, t, exec_month))
        return None

    pending = decide(t0)
    for i in range(t0 + 1, last + 1):
        if pending is not None:                                    # execute at today's open
            px = p.open[i].copy()
            stale = ~np.isfinite(px)
            px[stale] = p.cf[i - 1][stale]
            val = sh * np.nan_to_num(px)
            equity = cash + val.sum()
            buys, sells = [], []
            for j in range(M):
                tgt = pending.get(j, 0.0) * equity
                cur = val[j]
                d = tgt - cur
                if abs(d) < 1e-9:
                    continue
                if tgt > 0 and cur > 0 and abs(d) < BAND * equity:
                    continue                                       # no-trade band for names that stay held
                if stale[j] and tgt > 0:
                    continue                                       # no buy / re-weight of a stock not trading today
                (buys if d > 0 else sells).append((j, d))
            for j, d in sells + buys:
                v = abs(d)
                stat, slip = cost_fn(v, "BUY" if d > 0 else "SELL")
                sh[j] += d / px[j]
                if tgt_zero := (pending.get(j, 0.0) == 0.0 and d < 0):
                    sh[j] = 0.0                                    # sold completely (no float residue)
                cash -= d + stat + slip
                stat_tot += stat
                slip_tot += slip
                traded += v
                if tgt_zero and stale[j]:
                    stale_exits += 1                               # sold at its last close (suspended/delisted)
                log.append((int(dates[i]), p.symbols[j], "BUY" if d > 0 else "SELL", round(v, 2),
                            round(stat + slip, 2), bool(stale[j]) and tgt_zero))
            n_rebal += 1
            pending = None
        e = cash + float((sh * np.nan_to_num(p.cf[i])).sum())
        eq_dates.append(int(dates[i]))
        eq.append(e)
        if i in decset and i < last:
            pending = decide(i)
    return {"dates": np.array(eq_dates), "equity": np.array(eq), "statutory": stat_tot, "slippage": slip_tot,
            "traded": traded, "stale_exits": stale_exits, "rebalances": n_rebal, "cash_switches": n_cash_switch,
            "trades": log, "first_decision": int(dates[t0]), "last": int(dates[last])}


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _days_between(a: int, b: int) -> int:
    f = lambda x: _dt.date(x // 10000, x // 100 % 100, x % 100)
    return (f(b) - f(a)).days


def segment_metrics(res: dict, nifty_by_date: dict, start: int, end: int, trades=None) -> dict:
    d, e = res["dates"], res["equity"]
    i0 = int(np.searchsorted(d, start, side="left"))
    i1 = int(np.searchsorted(d, end, side="right")) - 1
    seg_d, seg_e = d[i0: i1 + 1], e[i0: i1 + 1]
    days = _days_between(int(seg_d[0]), int(seg_d[-1]))
    yrs = days / 365.25
    cagr = (seg_e[-1] / seg_e[0]) ** (1 / yrs) - 1
    n0, n1 = nifty_by_date[int(seg_d[0])], nifty_by_date[int(seg_d[-1])]
    ncagr = (n1 / n0) ** (1 / yrs) - 1
    r = seg_e[1:] / seg_e[:-1] - 1
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    peak = np.maximum.accumulate(seg_e)
    mdd = float(((seg_e - peak) / peak).min())
    mean_eq = float(seg_e.mean())
    lo, hi = int(seg_d[0]), int(seg_d[-1])
    tl = res["trades"] if trades is None else trades
    tv = sum(t[3] for t in tl if lo < t[0] <= hi)
    tc = sum(t[4] for t in tl if lo < t[0] <= hi)
    return {"from": lo, "to": hi, "years": round(yrs, 3), "cagr": cagr, "nifty_cagr": ncagr, "excess": cagr - ncagr,
            "sharpe": sharpe, "max_dd": mdd, "turnover": tv / 2 / mean_eq / yrs, "cost_drag": tc / mean_eq / yrs,
            "end_equity": float(seg_e[-1])}


def judge(era1: dict, era2: dict) -> dict:
    checks = {
        "H1 net CAGR > 0": era1["h1"]["cagr"] > 0,
        "H1 excess > 0": era1["h1"]["excess"] > 0,
        "H2 net CAGR > 0": era1["h2"]["cagr"] > 0,
        "H2 excess > 0": era1["h2"]["excess"] > 0,
        "era1 Sharpe >= 0.5": era1["whole"]["sharpe"] >= 0.5,
        "era2 net CAGR > 0": era2["whole"]["cagr"] > 0,
        "era2 excess > 0": era2["whole"]["excess"] > 0,
    }
    failed = [k for k, v in checks.items() if not v]
    return {"checks": {k: bool(v) for k, v in checks.items()}, "verdict": "PASS" if not failed else "FAIL", "failed": failed}


# ── data loading ────────────────────────────────────────────────────────────────────────────────────────────────

def _read_csv(path, usecols):
    import pandas as pd
    return pd.read_csv(path, usecols=usecols)


def load_oldera(old_dir: str, cache: str | None = None):
    """Era-1 panel from nse_old (split/bonus adjusted, delisted names kept, point-in-time universe)."""
    import pandas as pd
    uni = _read_csv(os.path.join(old_dir, "universe_monthly.csv"), ["month", "symbol"])
    universe = {int(m): set(g.symbol) for m, g in uni.groupby("month")}
    names = set(uni.symbol)
    idx = _read_csv(os.path.join(old_dir, "index_daily.csv.gz"), ["date", "index", "close"])
    nf = idx[idx["index"] == "NIFTY"].sort_values("date")
    dates = nf.date.to_numpy(np.int64)
    if cache and os.path.exists(cache):
        eq = pd.read_pickle(cache)
    else:
        eq = _read_csv(os.path.join(old_dir, "eq_daily_adj.csv.gz"), ["date", "symbol", "series", "open", "close"])
        if cache:
            eq.to_pickle(cache)
    eq = eq.assign(pri=(eq.series != "EQ").astype(int)).sort_values(["symbol", "date", "pri"])
    eq = eq.drop_duplicates(["symbol", "date"], keep="first")      # all symbols (the era-2 splice needs non-members)
    eq_all, eq = eq, eq[eq.symbol.isin(names)]
    syms = sorted(names)
    close = eq.pivot(index="date", columns="symbol", values="close").reindex(index=dates, columns=syms)
    opn = eq.pivot(index="date", columns="symbol", values="open").reindex(index=dates, columns=syms)
    p = Panel(dates, syms, close.to_numpy(float), opn.to_numpy(float), nf.close.to_numpy(float),
              universe=universe, label="era1 nse_old 2012-2023 point-in-time top-100")
    return p, dict(zip(dates.tolist(), p.nifty.tolist())), eq_all


def load_era2(bars_dir: str, old_eq, old_dir: str, nse_dir: str):
    """Era-2 panel: bars3y daily (current ~97 names, SURVIVORSHIP-BIASED) with lookback spliced from nse_old."""
    from . import data as D
    from .ltr import CA_GAPS
    man = D.prepare(bars_dir)
    cdir = man["cache_dir"]
    syms = sorted(man["symbols"])
    daily = {}
    for s in syms:
        sd = D.load_symbol(cdir, s)
        if sd is not None:
            daily[s] = sd.daily()
    syms = sorted(daily)
    new_dates = sorted({int(x) for d in daily.values() for x in d["date"] if x >= ERA2["junction"]})
    idx = _read_csv(os.path.join(old_dir, "index_daily.csv.gz"), ["date", "index", "close"])
    nf_old = idx[idx["index"] == "NIFTY"].sort_values("date")
    old_dates = [int(x) for x in nf_old.date if x <= ERA2["splice_last_old"]]
    dates = np.array(old_dates + new_dates, np.int64)
    pos = {int(d): i for i, d in enumerate(dates)}
    N, M = len(dates), len(syms)
    close = np.full((N, M), np.nan)
    opn = np.full((N, M), np.nan)
    bh = _read_csv(os.path.join(nse_dir, "delivery.csv.gz"), ["date", "symbol", "close", "prev_close"])
    bh = bh[bh.date == ERA2["junction"]].set_index("symbol")
    oe = old_eq[old_eq.date <= ERA2["splice_last_old"]]
    oe_by = {s: g for s, g in oe.groupby("symbol")}
    notes = {"no_old_history": [], "no_junction": []}
    for j, s in enumerate(syms):
        d = daily[s]
        m = d["date"] >= ERA2["junction"]
        for dt, o, c in zip(d["date"][m], d["o"][m], d["c"][m]):
            close[pos[int(dt)], j], opn[pos[int(dt)], j] = c, o
        old_sym = SPLICE_MAP.get(s, s)
        g = oe_by.get(old_sym)
        if g is None or g.empty:
            notes["no_old_history"].append(s)
            continue
        jd = d["date"][m]
        if len(jd) == 0 or int(jd[0]) != ERA2["junction"] or old_sym not in bh.index \
                or int(g.date.iloc[-1]) != ERA2["splice_last_old"]:
            notes["no_junction"].append(s)
            continue
        r_j = float(bh.loc[old_sym, "close"]) / float(bh.loc[old_sym, "prev_close"])
        k = float(d["c"][m][0]) / (r_j * float(g.close.iloc[-1]))
        for dt, o, c in zip(g.date, g.open, g.close):
            if int(dt) in pos:
                close[pos[int(dt)], j], opn[pos[int(dt)], j] = c * k, o * k
    gaps = {}
    for (s, gd), _why in CA_GAPS.items():
        if s in syms and gd in pos:
            j, g = syms.index(s), pos[gd]
            prev = [i for i in range(g) if np.isfinite(close[i, j])][-1]
            f = opn[g, j] / close[prev, j]
            close[:g, j] *= f
            opn[:g, j] *= f
            gaps.setdefault(s, []).append(int(gd))
    aux = D.load_daily_aux(cdir, "_NIFTY50_day")
    nmap = dict(zip(nf_old.date.astype(int), nf_old.close.astype(float)))
    first_aux = int(aux["date"][0])
    nmap = {k: v for k, v in nmap.items() if k < first_aux}
    nmap.update({int(a): float(b) for a, b in zip(aux["date"], aux["c"])})
    nifty = np.array([nmap.get(int(x), np.nan) for x in dates])
    p = Panel(dates, syms, close, opn, nifty, universe=None, ca_gaps=gaps,
              label="era2 bars3y 2023-11..2026-11, current NIFTY-100 names (SURVIVORSHIP-BIASED)")
    return p, dict(zip(dates.tolist(), p.nifty.tolist())), notes


# ── ledger / one-look ───────────────────────────────────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prereg(path: str) -> dict:
    """Refuse to run unless the pre-registration file is byte-identical to the registered one and agrees with the
    constants in this module."""
    h = sha256_file(path)
    if h != PREREG_SHA256:
        raise SystemExit(f"pre-registration sha256 {h} != registered {PREREG_SHA256}: refusing to run")
    pr = json.load(open(path))
    assert pr["variants"] == VARIANTS, "variants differ from the pre-registration"
    assert pr["signal"]["lookbacks_sessions"] == {"skip": SKIP, "m6": M6, "m12": M12}
    assert pr["portfolio"]["holdings"] == TOP_N and pr["costs"]["slippage_bps_per_side"] == SLIP_BPS
    return pr


def load_ledger(path: str) -> dict:
    return json.load(open(path)) if os.path.exists(path) else {}


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def save_ledger(path: str, led: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f, indent=1, sort_keys=True, default=_jsonable)
    os.replace(tmp, path)


def _fmt(m: dict) -> str:
    return (f"CAGR {m['cagr']*100:6.2f}%  NIFTY {m['nifty_cagr']*100:6.2f}%  excess {m['excess']*100:+6.2f}%  "
            f"Sharpe {m['sharpe']:5.2f}  maxDD {m['max_dd']*100:6.1f}%  turnover {m['turnover']*100:5.0f}%/yr  "
            f"cost drag {m['cost_drag']*100:4.2f}%/yr")


def main(argv=None):
    from .data import assert_outside_repo
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--old", required=True, help="nse_old directory")
    ap.add_argument("--bars", required=True, help="bars3y directory")
    ap.add_argument("--nse", required=True, help="nse directory (bhavcopy delivery.csv.gz, for the splice junction)")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args(argv)
    check_prereg(a.prereg)
    ledger_path = assert_outside_repo(a.ledger)
    out = assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    led = load_ledger(ledger_path)
    todo = [v for v in a.variants.split(",") if v]
    for v in todo:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    done = [v for v in todo if v in led]
    for v in done:
        print(f"REFUSED: {v} is already in the ledger ({led[v]['when']}); one look only. Stored verdict: "
              f"{led[v]['result']['verdict']}")
    todo = [v for v in todo if v not in led]
    if not todo:
        return 0
    p1, n1, eq = load_oldera(a.old, cache=os.path.join(out, "_eqadj_cache.pkl"))
    p2, n2, notes = load_era2(a.bars, eq, a.old, a.nse)
    print(f"era1: {len(p1.dates)} sessions, {len(p1.symbols)} symbols ever in the universe")
    print(f"era2: {len(p2.dates)} sessions, {len(p2.symbols)} symbols (SURVIVORSHIP-BIASED); splice notes {notes}")
    for v in todo:
        cfg = VARIANTS[v]
        r1 = backtest(p1, cfg["rebalance"], cfg["trend_filter"], ERA1["first_decision"], ERA1["end"])
        h1_end = int(r1["dates"][r1["dates"] <= ERA1["h1_end"]][-1])
        era1 = {"h1": segment_metrics(r1, n1, r1["dates"][0], h1_end),
                "h2": segment_metrics(r1, n1, h1_end, r1["dates"][-1]),
                "whole": segment_metrics(r1, n1, r1["dates"][0], r1["dates"][-1]),
                "stale_exits": r1["stale_exits"], "rebalances": r1["rebalances"],
                "cash_switches": r1["cash_switches"]}
        r2 = backtest(p2, cfg["rebalance"], cfg["trend_filter"], ERA2["first_decision"])
        era2 = {"whole": segment_metrics(r2, n2, r2["dates"][0], r2["dates"][-1]),
                "stale_exits": r2["stale_exits"], "rebalances": r2["rebalances"],
                "cash_switches": r2["cash_switches"], "label": p2.label}
        res = {"era1": era1, "era2": era2, **judge(era1, era2)}
        led[v] = {"when": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "prereg_sha256": PREREG_SHA256,
                  "params": cfg, "result": res}
        save_ledger(ledger_path, led)                              # written immediately: one look
        for name, r in (("era1", r1), ("era2", r2)):
            with open(os.path.join(out, f"{v}_{name}_equity.csv"), "w") as f:
                f.write("date,equity\n")
                for d_, e_ in zip(r["dates"], r["equity"]):
                    f.write(f"{d_},{e_:.2f}\n")
            with open(os.path.join(out, f"{v}_{name}_trades.csv"), "w") as f:
                f.write("date,symbol,side,value,cost,stale_exit\n")
                for t in r["trades"]:
                    f.write(",".join(str(x) for x in t) + "\n")
        print(f"\n{v}  ->  {res['verdict']}  {('failed: ' + ', '.join(res['failed'])) if res['failed'] else ''}")
        for k in ("h1", "h2", "whole"):
            print(f"  era1 {k:5s} {era1[k]['from']}..{era1[k]['to']}  {_fmt(era1[k])}")
        print(f"  era2 whole {era2['whole']['from']}..{era2['whole']['to']}  {_fmt(era2['whole'])}  [survivorship-biased]")
        print(f"  stale exits era1 {era1['stale_exits']}, rebalances {era1['rebalances']}/{era2['rebalances']}, "
              f"cash switches {era1['cash_switches']}/{era2['cash_switches']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
