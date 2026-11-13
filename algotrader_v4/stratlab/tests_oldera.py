"""
stratlab/tests_oldera.py — tests for the old-era daily lab mode (data_daily.py, oldera_fo.py, protocol_oos.py).

No catalogue strategy is run on the old era here (the old era stays unseen until its one judgement); look-ahead
and gating tests use synthetic test strategies. The parity test runs catalogue strategies on the 3-year era only.

    cd algotrader_v4
    python -m stratlab.tests_oldera [--skip-parity] [--skip-fo]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

from . import data_daily as DD
from . import oldera_fo as OF
from . import protocol_oos as PO

RESULTS: list = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


def _old():
    return DD._read_old(DD.OLD_DIR)


# ── 1. look-ahead on the daily loader ─────────────────────────────────────
def t_loader_lookahead(eq, idx, uni, ch, cache):
    """A cache built from data truncated at X equals the full cache up to X up to one constant per symbol
    (back-adjustment scale), with identical dates, membership flags and price ratios."""
    from . import sim
    from .api import Spec, Swing
    cuts = [20131231, 20170630, 20200315]
    probe = ["INFY", "TCS", "HDFCBANK", "RELIANCE", "SBIN", "TATAMOTORS", "WIPRO", "HINDUNILVR", "ITC", "MARUTI",
             "YESBANK", "BAJFINANCE", "ADANIPORTS", "VEDL", "UPL", "AXISBANK", "SUNPHARMA", "COALINDIA", "DHFL", "JSWSTEEL"]
    keep = set(probe)
    # keep every name that is renamed into a probe symbol
    while True:                                       # every name that is renamed (through a chain) into a probe
        more = {r.old_symbol for r in ch.itertuples(index=False) if r.new_symbol in keep} - keep
        if not more:
            break
        keep |= more
    sub = eq[eq.symbol.isin(keep)]

    def rule(d, p):                                   # synthetic: 20-day close breakout, ratio-only
        if len(d) < 21:
            return None
        if d.c[-1] > d.c[-21:-1].max() and d.v[-1] > 0:
            return Swing(1, stop_pct=5.0, target_pct=8.0, max_hold=10)
        return None
    sp = Spec("t_brk", "swing", "t", "t", rule, horizon="positional")
    worst_ratio, bad_dates, bad_member, bad_trades, n_cmp, n_tr = 0.0, 0, 0, 0, 0, 0
    bad_special: list = []
    nifty = DD.load_daily_aux(cache, "_NIFTY50_day")
    vix = DD.load_daily_aux(cache, "_VIX_day")
    for X in cuts:
        with tempfile.TemporaryDirectory(dir=os.path.dirname(cache)) as td:
            spec_x = DD.special_sessions(eq[eq.date <= X])          # market-wide, data up to X only
            full_x = [d for d in DD.manifest(cache)["special_dropped"] if d <= X]
            if spec_x != full_x:
                bad_special.append((X, sorted(set(spec_x) ^ set(full_x))))
            DD.build_cache(sub, idx, uni, ch, td, until=X, special=spec_x)
            for s in probe:
                a, b = DD.load_symbol(cache, s), DD.load_symbol(td, s)
                if a is None or b is None:
                    continue
                k = int(np.searchsorted(a.days, X, side="right"))
                if not np.array_equal(a.days[:k], b.days):
                    bad_dates += 1
                    continue
                if not np.array_equal(a.member[:k], b.member):
                    bad_member += 1
                r = a.c[:k] / b.c
                worst_ratio = max(worst_ratio, float(np.nanmax(r) / np.nanmin(r) - 1.0))
                for x, y in ((a.o[:k], b.o), (a.h[:k], b.h), (a.l[:k], b.l)):
                    rr = x / y
                    worst_ratio = max(worst_ratio, float(np.nanmax(rr) / np.nanmin(rr) - 1.0))
                n_cmp += 1
                # end to end: the same decisions / exits up to X from both caches (entries <= X, exits <= X)
                ta, _ = sim.sim_positional(DD.gate(sp, a), a, 20120101, X, nifty, vix)
                tb, _ = sim.sim_positional(DD.gate(sp, b), b, 20120101, X, nifty, vix)
                # exit reason "data_end" on the truncated cache == "period_end" on the full one (same close of X)
                norm = lambda r: "period_end" if r == "data_end" else r   # noqa: E731
                ka = [(t[1], t[2], t[3], t[6], norm(t[13]), round(t[8] / t[5], 6)) for t in ta]
                kb = [(t[1], t[2], t[3], t[6], norm(t[13]), round(t[8] / t[5], 6)) for t in tb]
                n_tr += len(ka)
                bad_trades += ka != kb
    check("loader look-ahead: truncated cache = full cache up to X (dates)", bad_dates == 0 and n_cmp > 30,
          f"{n_cmp} symbol x cut comparisons, {bad_dates} date mismatches")
    check("loader look-ahead: special-session calendar up to X is the same when later data is removed",
          not bad_special, str(bad_special))
    check("loader look-ahead: membership flags identical when later data is removed", bad_member == 0)
    check("loader look-ahead: prices differ only by a per-symbol constant (back-adjustment scale)",
          worst_ratio < 1e-6, f"max relative spread of full/truncated ratio {worst_ratio:.2e}")
    check("loader look-ahead: synthetic strategy trades identical on truncated vs full cache", bad_trades == 0,
          f"{n_tr} synthetic trades compared (entry date, side, exit date, reason, exit/entry ratio)")
    # the simulator never shows the strategy a day after the decision day
    seen = []

    def spy(d, p):
        seen.append((int(d.date[-1]), len(d.c), len(d.nifty) if d.nifty is not None else -1))
        return None
    sd = DD.load_symbol(cache, "INFY")
    sim.sim_positional(Spec("t_spy", "swing", "t", "t", spy, horizon="positional"), sd, 20150101, 20150331, nifty, vix)
    ok = all(n == int(np.searchsorted(sd.days, day, side="right")) == m for day, n, m in seen) and len(seen) > 50
    check("DailyView ends at the decision day (o/h/l/c/v, nifty aligned)", ok, f"{len(seen)} decisions checked")


# ── 2. universe point-in-time ─────────────────────────────────────────────
def t_universe(eq_raw, uni, ch, cache):
    import pandas as pd
    w = (uni.window_end.astype(int) < uni.effective_from.astype(int)).all()
    check("universe: every month's window ends strictly before its effective date", w)
    sizes = uni.groupby("month").size()
    check("universe: 100 names per month", (sizes == 100).all(), f"{len(sizes)} months")
    # recompute the rule from the UNADJUSTED bhavcopy for sampled months
    e = eq_raw[(eq_raw.series == "EQ") & eq_raw["isin"].astype(str).str.startswith("INE")][["date", "symbol", "turnover"]]
    days = np.array(sorted(eq_raw.date.unique()))
    piv = e.pivot_table(index="date", columns="symbol", values="turnover", aggfunc="sum").reindex(days)
    months = sorted(uni.month.unique())
    sample = months[3::17]
    agree = []
    for m in sample:
        g = uni[uni.month == m]
        eff = int(g.effective_from.iloc[0])
        k = int(np.searchsorted(days, eff))
        win = piv.iloc[max(0, k - 60):k]
        avg = win.fillna(0).mean()
        traded = win.notna().mean()
        okk = (traded >= 0.75) & win.iloc[-1].notna()
        top = set(avg[okk].sort_values(ascending=False).index[:100])
        agree.append(len(top & set(g.symbol)))
    check("universe: recomputed from data dated before each month reproduces the list", min(agree) >= 98,
          f"overlap per sampled month {agree}")
    # cache flags = list membership (canonical names), and some members later disappear (survivorship-free)
    man = DD.manifest(cache)
    u = uni.copy()
    u["date"] = u["effective_from"].astype(int)
    u, _ = DD.apply_renames(u, ch)
    pairs = set(zip(u.symbol, u.month.astype(int)))
    bad = 0
    for s in list(man["symbols"])[:4000]:
        if man["symbols"][s]["member_days"] == 0:
            continue
        sd = DD.load_symbol(cache, s)
        exp = np.array([(s, int(d) // 100) in pairs for d in sd.days])
        bad += int((exp != sd.member).sum())
    check("universe: per-session member flags match the monthly lists", bad == 0, f"{bad} mismatching sessions")
    gone = [s for s, x in man["symbols"].items() if x["member_days"] > 0 and x["last"] < 20231001]
    check("universe: delisted / suspended members are included (no survivorship filter)", len(gone) >= 10,
          f"{len(gone)} ever-members end before 2023-10 (e.g. {', '.join(sorted(gone)[:6])})")
    # the gate: a strategy that always signals opens positions only on member decision days
    from . import sim
    from .api import Spec, Swing
    sp = Spec("t_always", "swing", "t", "t", lambda d, p: Swing(1, max_hold=1), horizon="positional")
    flips = [s for s in man["symbols"] if 200 < man["symbols"][s]["member_days"] < man["symbols"][s]["n"] - 200]
    viol, n = 0, 0
    for s in flips[:15]:
        sd = DD.load_symbol(cache, s)
        tr, _ = sim.sim_positional(DD.gate(sp, sd), sd, DD.ERA[0], DD.ERA[1])
        mem = dict(zip(sd.days.tolist(), sd.member.tolist()))
        dec = {int(sd.days[i + 1]): int(sd.days[i]) for i in range(len(sd.days) - 1)}
        for t in tr:
            n += 1
            viol += not mem[dec[t[2]]]
    check("universe gate: no position is opened on a non-member decision day", viol == 0 and n > 1000,
          f"{n} synthetic entries on {min(15, len(flips))} symbols that enter / leave the universe")


# ── 3. adjustment sanity ──────────────────────────────────────────────────
def t_adjustment(cache, eq_raw):
    import pandas as pd
    af = pd.read_csv(os.path.join(DD.OLD_DIR, "adj_factors.csv"))
    ca = af[af.kind.isin(["split", "bonus", "consolidation"]) & af.status.str.startswith("applied")]
    ca_keys = set(zip(ca.symbol, ca.ex_date.astype(int)))
    man = DD.manifest(cache)
    members = [s for s, x in man["symbols"].items() if x["member_days"] > 0]
    big, at_ca = [], []
    for s in members:
        sd = DD.load_symbol(cache, s)
        d = sd.daily()
        if len(d["c"]) < 2:
            continue
        r = np.maximum(np.abs(d["c"][1:] / d["c"][:-1] - 1), np.abs(d["o"][1:] / d["c"][:-1] - 1))
        for k in np.flatnonzero(r > 0.40):
            big.append((s, int(d["date"][k + 1]), round(float(r[k]), 3)))
            if (s, int(d["date"][k + 1])) in ca_keys:
                at_ca.append(big[-1])
    check("adjustment: no > 40 % one-day move left at a split / bonus ex-date (universe members)", not at_ca,
          f"{len(big)} > 40 % moves remain in ever-members, none at a split/bonus ex-date: "
          f"{big[:8]}" if not at_ca else f"{at_ca[:10]}")
    # the raw bhavcopy does have them (the test has teeth)
    raw = eq_raw[eq_raw.symbol.isin(set(ca.symbol))][["symbol", "date", "close", "prev_close"]]
    raw = raw.sort_values(["symbol", "date"])
    raw["pc"] = raw.groupby("symbol").close.shift()
    raw["r"] = (raw.close / raw.pc - 1).abs()
    k = raw[raw.r > 0.40]
    n_raw = sum((s, int(d)) in ca_keys for s, d in zip(k.symbol, k.date))
    check("adjustment: the unadjusted data has > 40 % moves at those ex-dates (sanity of the check)", n_raw > 50,
          f"{n_raw} raw split/bonus jumps removed by the adjustment")
    infy = DD.load_symbol(cache, "INFY").daily()
    j = int(np.searchsorted(infy["date"], 20141202))
    check("adjustment: INFY 1:1 bonus 2014-12-02 is smooth", abs(infy["o"][j] / infy["c"][j - 1] - 1) < 0.05,
          f"open/prev close {infy['o'][j] / infy['c'][j - 1]:.3f}")


# ── 4. F&O view / nsedata switch ──────────────────────────────────────────
def t_fo(cache):
    N = OF.use(OF.VIEW_DIR)
    N._load()
    cal = DD.calendar(cache)
    fo = N.fo_sessions()
    check("nsedata view: F&O sessions cover the era", fo[0] <= 20120103 and fo[-1] == 20231109 and len(fo) > 2900,
          f"{len(fo)} sessions {fo[0]}..{fo[-1]}")
    D = 20150610
    r = N.delivery("INFY", D, 5)
    f = N.fut_oi("INFY", D, 5)
    p = N.participant(D, 5)
    ok = len(r) == 5 and int(r.date.max()) < D and len(f) == 5 and int(f.date.max()) < D
    check("nsedata view: accessors return only rows dated before `before` (delivery, fut OI)", ok,
          f"delivery {list(r.date)} fut {list(f.date)}")
    check("nsedata view: participant OI present and dated before `before`", len(p) == 5 and int(p.date.max()) < D,
          f"{list(p.date)}")
    raised = False
    try:
        with N.cutoff(D):
            N.option_eod("NIFTY", 20150625, 8000.0, "CE", D)
    except N.LookAheadError:
        raised = True
    check("nsedata view: cutoff(D) refuses same-day option data (EOD of D usable from D+1)", raised)

    class _D:
        pass
    dd = _D()
    dd.symbol, dd.date = "INFY", np.array([20150609, 20150610])
    a = N.asof_swing(dd)
    check("nsedata view: asof_swing on D sees D's EOD (before = D+1)", a.before == 20150611 and
          int(a.delivery(1).date[-1]) == 20150610)
    pd_days = N._S["part_oi"]["date"] if N._S.get("part_oi") is not None else np.array([])
    cov = len(set(int(x) for x in pd_days) & set(int(x) for x in fo)) / len(fo)
    check("participant OI covers >= 99 % of F&O sessions", cov >= 0.99, f"{cov:.2%}")
    po = N._S["part_oi"]
    cols = ["fut_idx_long", "fut_idx_short", "fut_stk_long", "opt_idx_call_long", "total_long", "total_short"]
    off = 0
    for c in cols:
        tot = po[f"TOTAL_{c}"]
        sm = sum(np.nan_to_num(po[f"{ct}_{c}"]) for ct in ("Client", "DII", "FII", "Pro"))
        off += int((np.abs(sm - tot) > 2).sum())
    dii_w = int((np.nan_to_num(po["DII_opt_idx_call_short"]) > 0).sum())
    fii_big = float(np.nanmedian(po["FII_fut_idx_long"] / np.maximum(po["DII_fut_idx_long"], 1)))
    check("participant OI: participants sum to TOTAL; FII/DII labels plausible", off == 0 and fii_big > 2,
          f"{off} day-columns off by > 2 contracts; median FII/DII index-futures long ratio {fii_big:.1f}; "
          f"DII index-call writing on {dii_w} days")
    nb = len(N._S["ban_days"])
    check("ban files cover the era", nb > 2900, f"{nb} ban files, {sum(len(v) for v in N._S['ban'].values())} "
          f"symbol-days")
    al = N.aliases()
    check("nsedata alias maps old F&O names to canonical symbols", al.get("MUNDRAPORT") == "ADANIPORTS"
          and al.get("CROMPGREAV") == "CGPOWER", f"{len(al)} aliases")
    # expiry calendar: rule vs bhavcopy
    for sym in ("NIFTY", "BANKNIFTY"):
        listed = set(OF.listed_expiries(sym, OF.VIEW_DIR))
        rule = set(OF.rule_expiries(sym, cal))
        lst = {x for x in listed if DD.ERA[0] <= x <= DD.ERA[1]}
        miss, extra = sorted(rule - lst), sorted(lst - rule)
        check(f"expiry calendar {sym}: rule (monthly last Thu + weeklies, holiday-shifted) == bhavcopy",
              len(miss) + len(extra) <= 2, f"{len(lst)} listed; rule-only {miss[:6]}; bhavcopy-only {extra[:6]}")
    lots = [(OF.lot_size("NIFTY", 20130101), 50), (OF.lot_size("NIFTY", 20150301), 25),
            (OF.lot_size("NIFTY", 20180101), 75), (OF.lot_size("NIFTY", 20220101), 50),
            (OF.lot_size("BANKNIFTY", 20170101), 40), (OF.lot_size("BANKNIFTY", 20190101), 20),
            (OF.lot_size("BANKNIFTY", 20231101), 15),
            (OF.lot_size("NIFTY", 20141020, expiry=20141127), 25)]
    check("lot sizes: old-era NIFTY / BANKNIFTY lots", all(a == b for a, b in lots), f"{lots}")
    lh = N.lot_size("NIFTY", 20170105, historical=True)
    check("nsedata historical lot (sensitivity) reads the filled lot column", lh == 75, f"NIFTY 2017-01-05 -> {lh}")


# ── 5. parity: minute-derived vs NSE daily bars, 2023-11..2026-11 ───────────
PARITY_IDS = ["swing_20d_breakout", "r3_swing_rsi2_sma200", "r3_swing_3down_sma100", "r3_tfdaily_ema20_50_cross",
              "r3_tfdaily_turtle55", "r3_futures_stk_rsi2_ob_short", "cs_bull_engulfing_d", "fut_donchian20",
              "r3_futures_nifty_ema_state", "r3_tfdaily_nifty_supertrend_regime"]


def t_parity():
    from . import catalog  # noqa: F401
    from . import data as D
    from . import sim
    from .api import REGISTRY
    bars = os.path.join(DD._SCRATCH, "bars3y")
    man = D.prepare(bars)
    lab = man["cache_dir"]
    syms = sorted(man["symbols"])
    pc = os.path.join(DD._SCRATCH, "stratlab_cache", "parity_nse3y_daily")
    DD.build_from_delivery(os.path.join(DD._SCRATCH, "nse", "delivery.csv.gz"), bars, pc, symbols=syms, ref_cache=lab)
    nl, vl = D.load_daily_aux(lab, "_NIFTY50_day"), D.load_daily_aux(lab, "_VIX_day")
    nn, vn = DD.load_daily_aux(pc, "_NIFTY50_day"), DD.load_daily_aux(pc, "_VIX_day")
    lo, hi = 20231110, 20261112
    diffs = {"o": [], "c": [], "h": [], "l": []}
    tot, sig = {}, {}
    sig_syms = set(syms[::3]) | {"_NIFTY50_min", "_NIFTYBANK_min"}
    for sym in syms + ["_NIFTY50_min", "_NIFTYBANK_min"]:
        a = D.load_symbol(lab, sym)
        b = DD.load_symbol(pc, sym)
        if a is None or b is None:
            continue
        da, db = a.daily(), b.daily()
        common, ia, ib = np.intersect1d(da["date"], db["date"], return_indices=True)
        for k in diffs:
            x = np.abs(da[k][ia] / db[k][ib] - 1)
            diffs[k].append(np.median(x))
        for sid in PARITY_IDS:
            sp = REGISTRY[sid]
            if not sim.applies(sp, sym):
                continue
            ta, _ = sim.sim_positional(sp, a, lo, hi, nl, vl)
            tb, _ = sim.sim_positional(sp, b, lo, hi, nn, vn)
            t = tot.setdefault(sid, [0, 0, 0, 0.0, 0.0])
            ka = {(x[1], x[2], x[3]) for x in ta}
            kb = {(x[1], x[2], x[3]) for x in tb}
            t[0] += len(ka); t[1] += len(kb); t[2] += len(ka & kb)
            t[3] += sum(x[12] for x in ta); t[4] += sum(x[12] for x in tb)
            if sym in sig_syms:                           # decision-level: same signal on the same day?
                g = sig.setdefault(sid, [0, 0])
                sa, sb = _signals(sp, a, lo, hi, nl, vl), _signals(sp, b, lo, hi, nn, vn)
                g[0] += len(sa & sb); g[1] += len(sa | sb)
    med = {k: float(np.median(v)) * 1e4 for k, v in diffs.items()}
    check("parity: daily bars minute-derived vs NSE daily (median over symbols of median abs diff, bps)",
          max(med.values()) < 1.0, ", ".join(f"{k} {v:.2f}" for k, v in med.items()) +
          " (close = NSE LAST; differs from the 15:29 minute close on ~30 % of days: post-close trades)")
    lines, worst = [], 1.0
    for sid, (na, nb, both, neta, netb) in tot.items():
        j = both / max(1, na + nb - both)
        worst = min(worst, j)
        lines.append(f"{sid}: minute {na} / NSE {nb} trades, {both} identical (symbol, entry date, side), "
                     f"Jaccard {j:.2f}, net {neta:,.0f} vs {netb:,.0f}")
    for l in lines:
        print("      " + l)
    for sid, (both, either) in sig.items():
        print(f"      {sid}: signal days identical {both}/{either} = {both / max(1, either):.2f}")
    stock_sig = [b / max(1, e) for k, (b, e) in sig.items() if not REGISTRY[k].underlying]
    idx_sig = [b / max(1, e) for k, (b, e) in sig.items() if REGISTRY[k].underlying]
    allt = [t for k, t in tot.items() if not REGISTRY[k].underlying]
    agg = sum(t[2] for t in allt) / max(1, sum(t[0] + t[1] - t[2] for t in allt))
    check("parity (stocks): same signal on the same day from minute-derived vs NSE daily bars, >= 0.80 per strategy",
          min(stock_sig) >= 0.80, f"min {min(stock_sig):.2f}, mean {np.mean(stock_sig):.2f} "
          f"(signal-day Jaccard, 1/3 of the symbols)")
    check("parity (stocks): trades (symbol, entry date, side) identical, aggregate Jaccard >= 0.70", agg >= 0.70,
          f"aggregate {agg:.2f}, worst strategy {worst:.2f} (one changed entry shifts the later ones: "
          f"one position per symbol)")
    check("parity (index futures, official index close vs last minute close): signal-day Jaccard reported",
          True, f"{', '.join(f'{x:.2f}' for x in idx_sig)}")
    return tot


def _signals(sp, sd, lo, hi, nifty, vix) -> set:
    """{(date, side)} of every day the strategy would signal (position state ignored)."""
    from . import sim
    d = sd.daily()
    n_, v_ = sim._aligned(d["date"], nifty), sim._aligned(d["date"], vix)
    out = set()
    for day in range(len(d["c"])):
        if not lo <= d["date"][day] <= hi:
            continue
        v = sim.DailyView(d["date"][:day + 1], *[d[k][:day + 1] for k in "ohlcv"])
        v.symbol = sd.symbol
        v.nifty = None if n_ is None else n_[:day + 1]
        v.vix = None if v_ is None else v_[:day + 1]
        r = sp.fn(v, sp.p)
        if r is not None:
            out.add((int(d["date"][day]), int(r.side)))
    return out


def t_eligibility():
    el, inel = PO.eligibility()
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    ids = {i for i, _m, _k in el}
    bad = [i for i in ids if i in REGISTRY and (REGISTRY[i].horizon != "positional" or REGISTRY[i].decision_time)]
    check("eligibility: no intraday / decision_time spec is eligible", not bad, f"{len(el)} eligible, "
          f"{len(inel)} ineligible rows")
    j = PO.judge([("x", "A", 20120105, 1, "", 0, 0, "", 0, 0, 0, 0, 10.0, "")] * 60 +
                 [("x", "B", 20180105, 1, "", 0, 0, "", 0, 0, 0, 0, 10.0, "")] * 60, "positional")
    j2 = PO.judge([("x", "A", 20120105, 1, "", 0, 0, "", 0, 0, 0, 0, 10.0, "")] * 59 +
                  [("x", "B", 20180105, 1, "", 0, 0, "", 0, 0, 0, 0, 10.0, "")] * 60, "positional")
    check("OOS gates: 60 trades per half needed; symbol cap applies", (not j["passed"]) and
          any("of net" in r for r in j["reasons"]) and any("H1 59" in r for r in j2["reasons"]),
          f"{j['reasons']} / {j2['reasons']}")
    jj = PO.judge([("x", f"S{i}", 20120105, 1, "", 0, 0, "", 0, 0, 0, 0, 10.0, "") for i in range(60)] +
                  [("x", f"S{i}", 20180105, 1, "", 0, 0, "", 0, 0, 0, 0, 10.0, "") for i in range(60)], "positional")
    check("OOS gates: a clean synthetic result passes", jj["passed"], str(jj["reasons"]))


def t_ledger():
    """One look: a second evaluate() returns the first stored result; changed code or a missing pre-registration
    is refused."""
    import dataclasses
    from .api import Spec, Swing
    calls = []
    sp = Spec("t_once", "swing", "t", "t", lambda d, p: None, horizon="positional")
    tr = [("t_once", f"S{i}", 20130105 + 10000 * (i % 2) * 5, 1, "", 100, 0, "", 101, 1, 1, 0, 10.0, "")
          for i in range(130)]

    def simulate(specs, lo, hi):
        calls.append(len(specs))
        return {s.id: {"trades": list(tr), "skipped": 0, "lookahead": None, "error": None} for s in specs}
    with tempfile.TemporaryDirectory() as td:
        led = os.path.join(td, "l.json")
        a = PO.evaluate([sp], simulate, simulate, cost_model="cm", ledger_path=led)
        b = PO.evaluate([sp], simulate, simulate, cost_model="cm", ledger_path=led)
        sp2 = dataclasses.replace(sp, params={"x": 1})
        c = PO.evaluate([sp2], simulate, simulate, cost_model="cm", ledger_path=led)
        d = PO.evaluate([sp], simulate, simulate, cost_model="cm", ledger_path=os.path.join(td, "m.json"),
                        prereg={"gates_fingerprint": PO.GATES_FINGERPRINT, "strategies": {}})
    ok = (a[0]["verdict"] == "OOS-CONFIRMED" and b[0].get("from_ledger") and b[0]["verdict"] == a[0]["verdict"]
          and c[0]["verdict"] == "NOT JUDGED" and d[0]["verdict"] == "NOT JUDGED" and calls == [1, 1])
    check("OOS ledger: one look per strategy; changed code / unregistered strategy not judged", ok,
          f"{a[0]['verdict']} / {b[0].get('note')} / {c[0].get('why')} / {d[0].get('why')}; simulate calls {calls}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-fo", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    cache = DD.ensure_old_cache()
    import pandas as pd
    eq, idx, uni, ch = _old()
    t_eligibility()
    t_ledger()
    t_loader_lookahead(eq, idx, uni, ch, cache)
    del eq
    eq_raw = pd.read_csv(os.path.join(DD.OLD_DIR, "eq_daily.csv.gz"),
                         usecols=["date", "symbol", "series", "close", "prev_close", "turnover", "isin"],
                         dtype={"symbol": str, "series": str, "isin": str}, keep_default_na=False, na_values=[""])
    t_universe(eq_raw, uni, ch, cache)
    t_adjustment(cache, eq_raw)
    del eq_raw
    if not a.skip_fo:
        t_fo(cache)
    if not a.skip_parity:
        t_parity()
    nf = sum(1 for _n, ok, _d in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nf}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
