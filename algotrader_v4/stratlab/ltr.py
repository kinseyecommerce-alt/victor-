"""
stratlab/ltr.py — cross-sectional long-term reversal on NIFTY-100 cash equities (CNC, long-only, monthly basket).

Research code only: reads bar files, writes reports to a scratch directory. It never touches a broker, a live agent
or a production API.

Pre-registered strategy LTR-12-1-LOSERS-CNC (parameters fixed before any run; DO NOT TUNE):

  universe   the stocks in the bar directory with >= 252 valid daily closes before the rebalance day (daily closes
             = last 1-minute bar of each valid session; read-only history-bars day data extends the lookback only)
  signal     R = C[t-21] / C[t-252] - 1 on each stock's own series of valid closes (the latest month is skipped)
  basket     the 10 lowest R (ties by symbol name), long only, equal Rs 1,00,000 per name, capital Rs 10 lakh
  schedule   every 21 trading sessions from the first session of the simulated period; everything is sold on the
             last session of the period (so DISCOVERY never holds a position into FINAL)
  execution  close of the 15:14 one-minute bar (the last bar at or before 15:14) on the rebalance day; leavers sold,
             entrants bought with Rs 1,00,000 each or whatever cash is left; names kept are NOT topped up
  costs      CNC, pre-registered: STT 0.1 % buy and sell, NSE 0.00297 % per side, stamp 0.015 % on buys, SEBI Rs 10 /
             crore, GST 18 % on exchange + SEBI, DP Rs 15.34 + 18 % GST per scrip per sell day, zero brokerage;
             slippage 2.1 bps per side (5 bps reported as a sensitivity)
  corporate  overnight gaps that are known unadjusted corporate actions (CA_GAPS) are excluded: the overnight return
  actions    of that day is set to zero (prices before it are scaled by open/previous close), and a name is not traded
             on such a day (it is held if already held, not bought if new)
  metric     the PRIMARY metric is EXCESS over the equal-weight universe (all eligible stocks, buy-and-hold over the
             same period, no costs): a pass on absolute return alone is market beta and does not count

How it goes through the lab's protocol (stratlab/protocol.py, unchanged): the 3-year calendar is split 70/30 by
`protocol.split_sessions` (DISCOVERY 2023-11-10..2025-12-16, H1 to 2024-11-29, FINAL 2025-12-17..2026-11-09), and
`protocol.evaluate` judges a list of "trades". Here one trade = one name held over one rebalance period (bought or
kept at its start, sold or marked at its end), entry date = the period start, and its `net` is its EXCESS P&L:
net P&L after costs minus (value at the start x the universe return of that period). The screen gates (net > 0 in
H1 and H2, PF >= 1.15, >= 60 trades, no symbol > 35 % of net) apply to those; FINAL is simulated only if they pass,
once, recorded in the same final_ledger.json. Position-periods are NOT independent (10 per period, ~24 periods in
DISCOVERY, ~11 in FINAL): the per-period table and its t statistic are reported next to the verdict.

    cd algotrader_v4
    python -m stratlab.ltr --data <bars3y dir> --hist <daily history dir> --out <scratch dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from . import data as D
from . import protocol as P

STRATEGY_ID = "LTR-12-1-LOSERS-CNC"
PARAMS = {"lookback": 252, "skip": 21, "n_names": 10, "hold": 21, "capital": 1_000_000.0, "per_name": 100_000.0,
          "exec_minute": 15 * 60 + 14, "min_closes": 252}
SLIPPAGE_BPS = 2.1
SENS_SLIPPAGE_BPS = 5.0
COST_MODEL = "ltr-cnc-prereg-1"          # the FINAL ledger key: pre-registered CNC rates below, fill model of this file

# Pre-registered CNC rates (per side, fractions of value). costs.py's equity_cnc differs slightly (NSE 0.00307 %, DP
# Rs 15.34 with GST already inside): `cost_side(..., model="costs_py")` gives that version for comparison.
CNC = {"stt": 0.001, "exch": 0.0000297, "stamp_buy": 0.00015, "sebi": 10 / 1e7, "gst": 0.18, "dp": 15.34 * 1.18}

# Known unadjusted corporate-action gaps in the minute data (> 15 % overnight; the other > 15 % gaps in the data are
# real news moves: ADANIGREEN/ADANIPOWER 2024-11-21, INDUSINDBK 2025-03-12, PAYTM 2024-02-01/02, OFSS 2025-09-11,
# VBL 2023-12-20 - their open-to-open path also shows in the adjusted history-bars day data).
CA_GAPS = {
    ("SIEMENS", 20250407): "Siemens Energy India demerger (record date 2025-04-07), unadjusted in minute and day data",
    ("VEDL", 20260430): "Vedanta demerger, unadjusted in the minute data",
}


def cost_side(value: float, side: str, model: str = "prereg") -> float:
    """Rs cost of one CNC buy or sell of `value` (no slippage; that is in the fill price)."""
    if value <= 0:
        return 0.0
    if model == "costs_py":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from costs import round_trip_cost
        return round_trip_cost(value, 0.0, "equity_cnc") if side == "buy" else round_trip_cost(0.0, value, "equity_cnc")
    c = CNC
    exch, sebi = value * c["exch"], value * c["sebi"]
    out = value * c["stt"] + exch + sebi + c["gst"] * (exch + sebi)
    if side == "buy":
        out += value * c["stamp_buy"]
    else:
        out += c["dp"]
    return out


# ── data ────────────────────────────────────────────────────────────────────
@dataclass
class Panel:
    """Daily panel. dates = sorted union of every symbol's days; close / px / opn are (n_sym, n_days) with NaN where the
    symbol has no valid day. px = execution price (15:14 close) only on minute-data days; opn = the day's open.
    sessions = the lab calendar (days with minute data), the only days that can be traded."""
    symbols: list
    dates: np.ndarray
    close: np.ndarray
    px: np.ndarray
    opn: np.ndarray
    sessions: np.ndarray
    notes: list = field(default_factory=list)

    def upto(self, hi: int) -> "Panel":
        """A copy holding only days <= hi: nothing later can be read."""
        k = int(np.searchsorted(self.dates, hi, side="right"))
        return Panel(self.symbols, self.dates[:k].copy(), self.close[:, :k].copy(), self.px[:, :k].copy(),
                     self.opn[:, :k].copy(), self.sessions[self.sessions <= hi].copy(), list(self.notes))


def _hist_rows(hist_dir: str, sym: str) -> dict:
    """{yyyymmdd: (open, close)} from every <sym>.json / <sym>__*.json day file in hist_dir."""
    out = {}
    if not hist_dir:
        return out
    files = [os.path.join(hist_dir, sym + ".json")] + sorted(glob.glob(os.path.join(hist_dir, glob.escape(sym) + "__*.json")))
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for r in json.load(fh).get("bars", []):
                d = int(str(r[0])[:10].replace("-", ""))
                if r[4] and r[4] > 0:
                    out[d] = (float(r[1]), float(r[4]))
    return out


def minute_daily(sd, exec_minute: int = PARAMS["exec_minute"]) -> dict:
    """Per valid session: date, open (first bar), close (last bar), px (close of the last bar at or before exec_minute)."""
    n = len(sd.days)
    px = np.full(n, np.nan)
    for k in range(n):
        a, b = int(sd.starts[k]), int(sd.ends[k])
        j = a + int(np.searchsorted(sd.t[a:b], exec_minute, side="right")) - 1
        if j >= a:
            px[k] = float(sd.c[j])
    return {"date": sd.days.astype(np.int64), "o": sd.o[sd.starts].astype(float),
            "c": sd.c[sd.ends - 1].astype(float), "px": px}


def build_panel(per_symbol: dict, hist: dict | None = None, hist_max_date: int | None = None) -> Panel:
    """per_symbol: {sym: minute_daily(...) dict}; hist: {sym: {date: (open, close)}} used ONLY for days before the
    symbol's first minute day (and <= hist_max_date)."""
    hist = hist or {}
    syms = sorted(per_symbol)
    sessions = np.array(sorted({int(x) for s in syms for x in per_symbol[s]["date"]}), np.int64)
    rows = {}
    notes = []
    for s in syms:
        m = per_symbol[s]
        first = int(m["date"][0]) if len(m["date"]) else 10 ** 9
        h = {d: v for d, v in hist.get(s, {}).items() if d < first and (hist_max_date is None or d <= hist_max_date)}
        r = {d: (o, c, np.nan) for d, (o, c) in h.items()}
        for d, o, c, x in zip(m["date"], m["o"], m["c"], m["px"]):
            r[int(d)] = (float(o), float(c), float(x))
        rows[s] = r
        if h:
            notes.append(f"{s}: {len(h)} history days before {first}")
    dates = np.array(sorted({d for r in rows.values() for d in r} | set(sessions.tolist())), np.int64)
    pos = {int(d): i for i, d in enumerate(dates)}
    shp = (len(syms), len(dates))
    close, px, opn = np.full(shp, np.nan), np.full(shp, np.nan), np.full(shp, np.nan)
    for i, s in enumerate(syms):
        for d, (o, c, x) in rows[s].items():
            j = pos[d]
            opn[i, j], close[i, j], px[i, j] = o, c, x
    return Panel(syms, dates, close, px, opn, sessions, notes)


def load_panel(data_dir: str, hist_dir: str = "", cache_dir: str | None = None, hist_max_date: int | None = None,
               verbose: bool = False) -> Panel:
    man = D.prepare(data_dir, cache_dir, verbose=False)
    per, hist = {}, {}
    for s in sorted(man["symbols"]):
        sd = D.load_symbol(man["cache_dir"], s)
        if sd is None:
            continue
        per[s] = minute_daily(sd)
        hist[s] = _hist_rows(hist_dir, s)
        if verbose:
            print(f"  {s}: {len(per[s]['date'])} sessions, {len(hist[s])} history days", flush=True)
    p = build_panel(per, hist, hist_max_date)
    p.notes.append(f"cache {man['cache_dir']}")
    return p


def adjust_gaps(p: Panel, gaps: dict = CA_GAPS) -> tuple[Panel, list]:
    """Remove each known corporate-action gap (on or before the panel's last day): every price before the gap day is
    multiplied by open(gap day) / close(previous valid day), so that day's overnight return is zero."""
    close, px, opn = p.close.copy(), p.px.copy(), p.opn.copy()
    done = []
    for (sym, g), why in sorted(gaps.items(), key=lambda kv: kv[0][1]):
        if sym not in p.symbols or g > p.dates[-1]:
            continue
        i = p.symbols.index(sym)
        j = int(np.searchsorted(p.dates, g))
        if j >= len(p.dates) or p.dates[j] != g or not np.isfinite(opn[i, j]):
            continue
        prev = np.flatnonzero(np.isfinite(close[i, :j]))
        if not len(prev):
            continue
        f = opn[i, j] / close[i, prev[-1]]
        close[i, :j] *= f
        px[i, :j] *= f
        opn[i, :j] *= f
        done.append((sym, g, round(float(f), 4), why))
    return Panel(p.symbols, p.dates, close, px, opn, p.sessions, list(p.notes)), done


# ── signal ──────────────────────────────────────────────────────────────────
def signal(p: Panel, j: int, prm: dict = PARAMS) -> dict:
    """{symbol: R} for every stock eligible on panel day index j: tradable that day (a 15:14 price) and with at least
    `min_closes` valid closes BEFORE day j. R = C[t-skip] / C[t-lookback] - 1 on the stock's own valid closes, where
    C[t] is day j's close (never read: the trade happens at 15:14, before the close)."""
    out = {}
    lb, sk = int(prm["lookback"]), int(prm["skip"])
    for i, s in enumerate(p.symbols):
        if not np.isfinite(p.px[i, j]):
            continue
        prev = p.close[i, :j]
        prev = prev[np.isfinite(prev)]
        if len(prev) < prm["min_closes"]:
            continue
        p_now = len(prev)                       # index day j would have in the stock's valid-close series
        a, b = prev[p_now - sk], prev[p_now - lb]
        if b > 0:
            out[s] = a / b - 1.0
    return out


def select(sig: dict, n: int) -> list:
    """The n lowest R, ties broken by symbol."""
    return [s for _r, s in sorted((r, s) for s, r in sig.items())[:n]]


# ── portfolio simulation ────────────────────────────────────────────────────
TRADE_FIELDS = ("id", "symbol", "date", "side", "entry_t", "entry", "exit_date", "exit_t", "exit", "qty",
                "gross", "cost", "net", "reason")       # same layout as sim.TRADE_FIELDS (net at index 12)


def simulate_period(panel: Panel, lo: int, hi: int, slip_bps: float = SLIPPAGE_BPS, prm: dict = PARAMS,
                    cost_model: str = "prereg", gaps: dict = CA_GAPS) -> dict:
    """Run the basket over the lab sessions in [lo, hi]. Reads no data after hi (the panel is cut first).
    Returns {"excess": [trade tuples], "absolute": [...], "periods": [...], "rebalances": [...], "gaps": [...]}."""
    p, gaps_done = adjust_gaps(panel.upto(hi), gaps)
    gap_days = {(s, g) for (s, g) in gaps}
    sess = [int(d) for d in p.sessions if lo <= d <= hi]
    if not sess:
        return {"excess": [], "absolute": [], "periods": [], "rebalances": [], "gaps": gaps_done}
    col = {int(d): k for k, d in enumerate(p.dates)}
    reb = list(range(0, len(sess), int(prm["hold"])))
    last = len(sess) - 1
    marks = reb + ([last] if reb[-1] != last else [])
    slip = slip_bps / 1e4
    sidx = {s: i for i, s in enumerate(p.symbols)}

    cash = float(prm["capital"])
    hold: dict = {}            # sym -> {"qty", "basis_px", "basis_val", "bought": bool, "buy_cost"}
    periods, rebs, exc_tr, abs_tr = [], [], [], []
    last_px: dict = {}
    prev_mark = None           # (date, nav_pre, universe {sym: px})

    def price(sym, j):
        x = p.px[sidx[sym], j]
        return float(x) if np.isfinite(x) else None

    def mark_px(sym, j):
        x = price(sym, j)
        if x is not None:
            last_px[sym] = x
        return last_px.get(sym)

    for m_i, k in enumerate(marks):
        d = sess[k]
        j = col[d]
        is_last = k == last
        # 1) value the book before trading
        for s in list(hold):
            mark_px(s, j)
        nav_pre = cash + sum(h["qty"] * last_px[s] for s, h in hold.items())
        uni_ret = None
        if prev_mark is not None:
            pd_, nav0, uni0 = prev_mark
            rs = []
            for s, x0 in uni0.items():
                x1 = price(s, j)
                if x1 is None:                       # no price today: last known price up to today
                    row = p.px[sidx[s], col[pd_]:j + 1]
                    row = row[np.isfinite(row)]
                    x1 = float(row[-1]) if len(row) else x0
                rs.append(x1 / x0 - 1.0)
            uni_ret = float(np.mean(rs)) if rs else 0.0
        # 2) decide
        if is_last:
            target = []
        else:
            sig = signal(p, j, prm)
            target = select(sig, int(prm["n_names"]))
        tradable = lambda s: price(s, j) is not None and (s, d) not in gap_days  # noqa: E731
        sells = [s for s in sorted(hold) if s not in target and tradable(s)]
        buys = [s for s in target if s not in hold and tradable(s)]
        # 3) close the previous period's position-trades (kept names are marked, sold names are filled)
        sold_px, sell_cost = {}, {}
        for s in sells:
            fill = price(s, j) * (1.0 - slip)
            val = hold[s]["qty"] * fill
            c = cost_side(val, "sell", cost_model)
            sold_px[s], sell_cost[s] = fill, c
            cash += val - c
        if prev_mark is not None:
            pd_ = prev_mark[0]
            for s, h in sorted(hold.items()):
                exit_px = sold_px.get(s, last_px[s])
                gross = h["qty"] * (exit_px - h["basis_px"])
                cst = h["buy_cost"] + sell_cost.get(s, 0.0)
                net = gross - cst
                reason = ("bought" if h["bought"] else "kept") + "->" + ("sold" if s in sold_px else "kept")
                row = (STRATEGY_ID, s, int(pd_), 1, "15:14", round(h["basis_px"], 4), int(d), "15:14",
                       round(exit_px, 4), int(h["qty"]), round(gross, 2), round(cst, 2), round(net, 2), reason)
                abs_tr.append(row)
                ex = net - h["basis_val"] * uni_ret
                exc_tr.append(row[:12] + (round(ex, 2), reason))
        for s in sells:
            del hold[s]
        # 4) buy entrants; reset the basis of kept names to today's mark
        for s, h in hold.items():
            h.update(basis_px=last_px[s], basis_val=h["qty"] * last_px[s], bought=False, buy_cost=0.0)
        bought = []
        for s in buys:
            fill = price(s, j) * (1.0 + slip)
            budget = min(float(prm["per_name"]), cash)
            qty = int(budget // (fill * (1.0 + 0.0013)))         # keep room for the buy costs (~0.12 %)
            while qty > 0 and qty * fill + cost_side(qty * fill, "buy", cost_model) > cash:
                qty -= 1
            if qty <= 0:
                continue
            val = qty * fill
            c = cost_side(val, "buy", cost_model)
            cash -= val + c
            hold[s] = {"qty": qty, "basis_px": fill, "basis_val": val, "bought": True, "buy_cost": c}
            last_px[s] = price(s, j)
            bought.append(s)
        # 5) period record
        if prev_mark is not None:
            pd_, nav0, _u = prev_mark
            if is_last:                                         # the final liquidation's costs are in the last period
                nav_pre = cash + sum(h["qty"] * last_px[s] for s, h in hold.items())
            r = nav_pre / nav0 - 1.0
            periods.append({"start": int(pd_), "end": int(d), "nav_start": round(nav0, 2), "nav_end": round(nav_pre, 2),
                            "ret": r, "universe": uni_ret, "excess": r - uni_ret, "n_universe": len(prev_mark[2])})
        if not is_last:
            uni = {}
            for s in signal(p, j, prm):                         # the eligible universe on this day
                uni[s] = price(s, j)
            rebs.append({"date": int(d), "held": sorted(hold), "sold": sells, "bought": bought,
                         "target": target, "cash": round(cash, 2), "n_universe": len(uni),
                         "skipped_gap": sorted(s for s in set(target) | set(hold) if (s, d) in gap_days)})
            # the period's return runs from the PRE-trade value, so today's trading costs fall in the new period
            prev_mark = (d, nav_pre, uni)
    # whatever could not be sold on the last day (no price) is closed at its last known price
    if hold:
        for s, h in hold.items():
            cash += h["qty"] * last_px[s] - cost_side(h["qty"] * last_px[s], "sell", cost_model)
    return {"excess": exc_tr, "absolute": abs_tr, "periods": periods, "rebalances": rebs, "gaps": gaps_done}


# ── the protocol wrapper ────────────────────────────────────────────────────
@dataclass
class PortfolioSpec:
    """What protocol.evaluate needs from a strategy (id, horizon, fingerprint, kind)."""
    id: str = STRATEGY_ID
    kind: str = "swing"
    family: str = "long-term reversal (CNC monthly basket)"
    desc: str = "Bottom decile of 12-1 month return among NIFTY-100 stocks, long-only CNC, rebalanced every 21 sessions"
    horizon: str = "positional"
    underlying: str | None = None
    params: dict = field(default_factory=lambda: dict(PARAMS))

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(f) for f in (signal, select, simulate_period, adjust_gaps, cost_side))
        blob = repr((self.id, sorted(self.params.items()), SLIPPAGE_BPS, sorted(CNC.items()), sorted(CA_GAPS), src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def period_stats(periods: list, lo: int, hi: int) -> dict:
    ps = [x for x in periods if lo <= x["start"] <= hi]
    if not ps:
        return {"n": 0}
    ex = np.array([x["excess"] for x in ps])
    ab = np.array([x["ret"] for x in ps])
    un = np.array([x["universe"] for x in ps])
    sd = float(ex.std(ddof=1)) if len(ex) > 1 else float("nan")
    return {"n": len(ps), "mean_excess_bps": round(1e4 * float(ex.mean()), 1),
            "t_excess": round(float(ex.mean()) / (sd / math.sqrt(len(ex))), 2) if sd and sd == sd and sd > 0 else None,
            "hit_excess": round(float((ex > 0).mean()), 2),
            "mean_abs_bps": round(1e4 * float(ab.mean()), 1), "mean_universe_bps": round(1e4 * float(un.mean()), 1),
            "cum_abs": round(float(np.prod(1 + ab) - 1), 4), "cum_universe": round(float(np.prod(1 + un) - 1), 4)}


def run_protocol(panel: Panel, ledger_path: str | None, slip_bps: float = SLIPPAGE_BPS,
                 sens_bps: float = SENS_SLIPPAGE_BPS, screen_only: bool = False) -> dict:
    """protocol.evaluate on the excess position-trades. FINAL is simulated only if the DISCOVERY screen passes."""
    spec = PortfolioSpec()
    store = {}
    calls = []

    def simulate(specs, lo, hi):
        calls.append((lo, hi))
        r = simulate_period(panel, lo, hi, slip_bps)
        store[(lo, hi)] = r
        return {spec.id: {"trades": r["excess"], "skipped": 0, "lookahead": None, "error": None}}

    results, _tr = P.evaluate([spec], [int(x) for x in panel.sessions], simulate, n_symbols=len(panel.symbols),
                              ledger_path=ledger_path, dataset="ltr", screen_only=screen_only, cost_model=COST_MODEL)
    res = results[0]
    sp = res["split"]
    disc = store[sp.discovery]
    out = {"result": res, "split": sp, "disc": disc, "calls": calls}
    out["abs_screen"] = P.screen(disc["absolute"], sp, "positional")
    out["disc_periods"] = {"H1": period_stats(disc["periods"], *sp.h1), "H2": period_stats(disc["periods"], *sp.h2),
                           "DISCOVERY": period_stats(disc["periods"], *sp.discovery)}
    sens = simulate_period(panel, sp.discovery[0], sp.discovery[1], sens_bps)
    out["sens"] = {"screen": P.screen(sens["excess"], sp, "positional"),
                   "abs": P.stats(sens["absolute"]),
                   "periods": period_stats(sens["periods"], *sp.discovery)}
    cpy = simulate_period(panel, sp.discovery[0], sp.discovery[1], slip_bps, cost_model="costs_py")
    out["costs_py"] = {"screen": P.screen(cpy["excess"], sp, "positional"), "abs": P.stats(cpy["absolute"])}
    if sp.final in store:                                    # FINAL was legitimately simulated (a screen pass)
        fin = store[sp.final]
        out["final"] = fin
        out["final_periods"] = period_stats(fin["periods"], *sp.final)
        out["final_abs"] = P.stats(fin["absolute"])
        f5 = simulate_period(panel, sp.final[0], sp.final[1], sens_bps)
        out["final_sens"] = {"excess": P.stats(f5["excess"]), "abs": P.stats(f5["absolute"]),
                             "periods": period_stats(f5["periods"], *sp.final)}
    return out


def _write(out_dir: str, o: dict, panel: Panel, args) -> str:
    os.makedirs(out_dir, exist_ok=True)
    res, sp, disc = o["result"], o["split"], o["disc"]
    for name, rows in (("disc_excess_trades.csv", disc["excess"]), ("disc_absolute_trades.csv", disc["absolute"])):
        with open(os.path.join(out_dir, name), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(TRADE_FIELDS); w.writerows(rows)
    with open(os.path.join(out_dir, "disc_periods.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(disc["periods"][0])); w.writeheader(); w.writerows(disc["periods"])
    with open(os.path.join(out_dir, "disc_rebalances.json"), "w") as fh:
        json.dump(disc["rebalances"], fh, indent=1)
    if "final" in o:
        fin = o["final"]
        for name, rows in (("final_excess_trades.csv", fin["excess"]), ("final_absolute_trades.csv", fin["absolute"])):
            with open(os.path.join(out_dir, name), "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(TRADE_FIELDS); w.writerows(rows)
        with open(os.path.join(out_dir, "final_periods.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fin["periods"][0])); w.writeheader(); w.writerows(fin["periods"])
        with open(os.path.join(out_dir, "final_rebalances.json"), "w") as fh:
            json.dump(fin["rebalances"], fh, indent=1)
    sc = res["screen"]
    L = [f"{STRATEGY_ID}  (stratlab/ltr.py, cost model {COST_MODEL}, fingerprint {PortfolioSpec().fingerprint})",
         f"data={args.data}  hist={args.hist}  symbols={len(panel.symbols)}  sessions={len(panel.sessions)}",
         f"DISCOVERY {sp.discovery[0]}..{sp.discovery[1]} (H1 to {sp.h1[1]})  FINAL {sp.final[0]}..{sp.final[1]}",
         f"slippage {SLIPPAGE_BPS} bps per side (sensitivity {SENS_SLIPPAGE_BPS} bps); capital Rs 10 lakh, 10 x Rs 1 lakh",
         f"corporate-action gaps excluded: {disc['gaps']}",
         "",
         "SCREEN (protocol.screen on EXCESS position-trades = net after costs - start value x EW-universe return)",
         f"  H1   trades {sc['h1']['trades']:>4}  PF {sc['h1']['pf']}  net {sc['h1']['net']:>12,.0f}",
         f"  H2   trades {sc['h2']['trades']:>4}  PF {sc['h2']['pf']}  net {sc['h2']['net']:>12,.0f}",
         f"  DISC trades {sc['disc']['trades']:>4}  PF {sc['disc']['pf']}  net {sc['disc']['net']:>12,.0f}  "
         f"top symbol {sc['top_symbol']} {sc['top_share']:.0%}" if sc['top_share'] == sc['top_share'] else
         f"  DISC trades {sc['disc']['trades']:>4}  PF {sc['disc']['pf']}  net {sc['disc']['net']:>12,.0f}",
         f"  VERDICT: {res['verdict']} - {res['why']}" + (f" | {res.get('final_note')}" if res.get("final_note") else ""),
         "",
         "Absolute (not the metric; beta): " + json.dumps({k: o['abs_screen'][k] for k in ('h1', 'h2', 'disc')})
         + f"  screen passed on absolute: {o['abs_screen']['passed']}",
         "Per-period (portfolio level, includes idle cash):"]
    for k, v in o["disc_periods"].items():
        L.append(f"  {k:<10}{json.dumps(v)}")
    L.append(f"At {SENS_SLIPPAGE_BPS} bps: excess screen passed={o['sens']['screen']['passed']} "
             f"disc excess net {o['sens']['screen']['disc']['net']:,.0f} (H1 {o['sens']['screen']['h1']['net']:,.0f}, "
             f"H2 {o['sens']['screen']['h2']['net']:,.0f}, PF {o['sens']['screen']['disc']['pf']}); absolute net "
             f"{o['sens']['abs']['net']:,.0f}; periods {json.dumps(o['sens']['periods'])}")
    L.append(f"With costs.py equity_cnc instead of the pre-registered rates: excess net "
             f"{o['costs_py']['screen']['disc']['net']:,.0f}, absolute net {o['costs_py']['abs']['net']:,.0f}")
    if "final" in o:
        fv = res["final"]
        L += ["", f"FINAL (once): excess trades {fv['trades']} PF {fv['pf']} net {fv['net']:,.0f} -> {res['verdict']}",
              f"  absolute: {json.dumps(o['final_abs'])}",
              f"  periods: {json.dumps(o['final_periods'])}",
              f"  at {SENS_SLIPPAGE_BPS} bps: {json.dumps(o['final_sens'])}"]
    else:
        L += ["", "FINAL: not simulated (screen failed or --screen-only)."]
    L += ["", "Position-periods are not independent: 10 per period. The per-period t statistic is the honest one; "
              "the fact this strategy came from wanted t ~3.3 after multiple-testing (Bonferroni). Survivorship: "
              "the universe is today's NIFTY-100 members, so past losers that are still members recovered by "
              "construction - a pass would overstate the edge."]
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.ltr", description="LTR-12-1-LOSERS-CNC through the lab protocol")
    ap.add_argument("--data", required=True, help="directory of 1-minute bar JSON files (bars3y)")
    ap.add_argument("--hist", default="", help="directory of read-only history-bars day files <SYM>[__x].json (lookback only)")
    ap.add_argument("--out", required=True, help="output directory (outside the repository)")
    ap.add_argument("--cache", default="", help="stratlab cache directory (default: the lab's)")
    ap.add_argument("--screen-only", action="store_true", help="never simulate FINAL")
    a = ap.parse_args(argv)
    out = D.assert_outside_repo(a.out)
    man = D.prepare(a.data, a.cache or None)
    panel = load_panel(a.data, a.hist, a.cache or None)
    first_min = int(panel.sessions[0])
    # history day data is used for lookback before the minute data only
    hist_days = panel.dates[panel.dates < first_min]
    print(f"panel: {len(panel.symbols)} symbols, {len(panel.sessions)} sessions, {len(hist_days)} earlier history days")
    o = run_protocol(panel, os.path.join(man["cache_dir"], "final_ledger.json"), screen_only=a.screen_only)
    print(_write(out, o, panel, a))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
