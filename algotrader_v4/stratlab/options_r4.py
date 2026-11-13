"""
stratlab/options_r4.py — round-4 REAL-PRICE OPTIONS candidates (NSE EOD option bhavcopy), judged by the unchanged
protocol through nsedata's OptionSpec simulator (fills at NSE's official EOD close of the entry session, haircut
max(₹0.05, 1 %) per leg per fill, expiry settlement at intrinsic vs the final settlement price, 2026 option fees).

    cd algotrader_v4
    python -m stratlab.options_r4 --data <bars3y> --cache <lab cache> --out <scratch dir> [--ids a,b] [--jobs 3]
                                  [--screen-only]
    python -m stratlab.options_r4 --smoke [--ids a,b]       # correctness only: counts / errors / look-ahead, no P&L

Pre-registration: <scratchpad>/r4/realopt/PREREGISTERED.txt (19 ids, prefix r4_realopt_). Do not tune on results.

Look-ahead: decide() / manage() run under nsedata.cutoff(D) (D = the entry / management session) and may read only EOD
data dated < D. On top of that every chain / row set read here goes through _chain / _rows / _marks, which RAISE
nsedata.LookAheadError if anything is dated >= D, and return None (no trade) if the chain is not the previous F&O
session's (no fill-forward of a missing file). Strikes are therefore chosen from D-1's real chain and filled at D's
real close (only if the strike traded on D).
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
from datetime import date as _date

import numpy as np

from . import nsedata as N

SYM_NF, SYM_BNF = "NIFTY", "BANKNIFTY"

# ── hard-coded scheduled macro events (decision day E), public months in advance — see PREREGISTERED.txt ──────────
BUDGET_DAYS = (20240201, 20240723, 20250201, 20260201)
RBI_MPC_DAYS = (20231208, 20240208, 20240405, 20240607, 20240808, 20241009, 20241206, 20250207,
                20250409, 20250606, 20250806, 20251001, 20251205, 20260206,
                20260408, 20260605, 20260806, 20261001)          # FY2026-27 dates UNVERIFIED (see prereg)
EVENT_DAYS = frozenset(BUDGET_DAYS + RBI_MPC_DAYS)


# ══════════════════════════════════════════════════════════════════════════
# look-ahead guards (explicit, on top of nsedata's cutoff)
# ══════════════════════════════════════════════════════════════════════════

def _prev_fo_session(day: int) -> int | None:
    d = N.fo_sessions(before=day)
    return int(d[-1]) if len(d) else None


def _chain(ctx, expiry="near", sym=None):
    """The EOD chain of the PREVIOUS F&O session before ctx.date. Raises LookAheadError if the chain is dated on or
    after the decision day; None if it is missing or stale (not exactly the previous F&O session)."""
    return _check_chain(ctx, N.options_chain(sym or ctx.symbol, int(ctx.date), expiry))


def _check_chain(ctx, c):
    """Explicit guard: a chain dated on/after the decision day raises; a stale one (not the previous F&O session)
    gives None."""
    if c is None:
        return None
    if int(c.date) >= int(ctx.date):
        raise N.LookAheadError(f"options_r4: chain dated {c.date} used for a decision on {ctx.date}")
    if c.date != _prev_fo_session(int(ctx.date)):
        return None
    return c


def _rows(ctx, rows):
    """Guard for accessor results: nothing dated >= the decision day."""
    if rows is not None and len(rows) and int(np.max(rows["date"])) >= int(ctx.date):
        raise N.LookAheadError(f"options_r4: rows dated {int(np.max(rows['date']))} used for a decision on {ctx.date}")
    return rows


def _marks(ctx, pos):
    """The position's EOD marks of sessions BEFORE today (guarded)."""
    for d, _v in pos.marks:
        if int(d) >= int(ctx.date):
            raise N.LookAheadError(f"options_r4: mark dated {d} read on {ctx.date}")
    return pos.marks


def _expiries(ctx, sym=None) -> list:
    return [int(e) for e in N.expiries(sym or ctx.symbol, int(ctx.date))]


def _d(x: int) -> _date:
    return _date(x // 10000, x // 100 % 100, x % 100)


def _days(a: int, b: int) -> int:
    return (_d(int(b)) - _d(int(a))).days


# ══════════════════════════════════════════════════════════════════════════
# strike selection on the previous session's real chain
# ══════════════════════════════════════════════════════════════════════════

def _traded(c, typ):
    close = c.ce_close if typ == "CE" else c.pe_close
    vol = c.ce_vol if typ == "CE" else c.pe_vol
    return np.isfinite(close) & (close > 0) & np.isfinite(vol) & (vol > 0)


def _nearest(c, typ, x, otm_only=True):
    """Strike nearest x among traded strikes of type typ; OTM only (CE > spot, PE < spot)."""
    ok = _traded(c, typ)
    if otm_only:
        ok &= (c.strike > c.spot) if typ == "CE" else (c.strike < c.spot)
    if not ok.any():
        return None
    k = c.strike[ok]
    return float(k[np.argmin(np.abs(k - x))])


def _atm(c):
    ok = _traded(c, "CE") & _traded(c, "PE")
    if not ok.any():
        return None
    k = c.strike[ok]
    return float(k[np.argmin(np.abs(k - c.spot))])


def _by_premium(c, typ, target):
    ok = _traded(c, typ) & ((c.strike > c.spot) if typ == "CE" else (c.strike < c.spot))
    if not ok.any():
        return None
    close = (c.ce_close if typ == "CE" else c.pe_close)[ok]
    return float(c.strike[ok][np.argmin(np.abs(close - target))])


def _wing(c, typ, short, w):
    """Long wing of the same type, strike nearest short*(1±w), strictly further OTM than the short."""
    x = short * (1 + w) if typ == "CE" else short * (1 - w)
    ok = _traded(c, typ) & ((c.strike > short) if typ == "CE" else (c.strike < short))
    if not ok.any():
        return None
    k = c.strike[ok]
    return float(k[np.argmin(np.abs(k - x))])


def _walls(c):
    """(call wall, put wall): max CE OI among strikes > spot, max PE OI among strikes < spot."""
    up = (c.strike > c.spot) & np.isfinite(c.ce_oi)
    dn = (c.strike < c.spot) & np.isfinite(c.pe_oi)
    if not up.any() or not dn.any():
        return None, None
    wc = float(c.strike[up][np.argmax(c.ce_oi[up])])
    wp = float(c.strike[dn][np.argmax(c.pe_oi[dn])])
    return wc, wp


def _nearest_any(c, typ, x, beyond):
    """Traded strike nearest x with strike >= beyond (CE) / <= beyond (PE)."""
    ok = _traded(c, typ) & ((c.strike >= beyond) if typ == "CE" else (c.strike <= beyond))
    ok &= (c.strike > c.spot) if typ == "CE" else (c.strike < c.spot)
    if not ok.any():
        return None
    k = c.strike[ok]
    return float(k[np.argmin(np.abs(k - x))])


# ══════════════════════════════════════════════════════════════════════════
# schedules
# ══════════════════════════════════════════════════════════════════════════

def _weekly_chain(ctx):
    """NIFTY weekly cycle: D is the first F&O session after an expiry day (D-1 is a listed expiry), near expiry
    alive on D, <= 10 calendar days away. Returns the near chain or None."""
    c = _chain(ctx, "near")
    if c is None:
        return None
    if c.date not in _expiries(ctx):
        return None
    if not (c.expiry > ctx.date and _days(ctx.date, c.expiry) <= 10):
        return None
    return c


def _monthly_chain(ctx, min_days=10):
    """First MONTHLY expiry (last listed expiry of its calendar month) >= min_days calendar days after D."""
    exs = _expiries(ctx)
    if not exs:
        return None
    last_of_month = {}
    for e in exs:
        last_of_month[e // 100] = max(last_of_month.get(e // 100, 0), e)
    monthly = sorted(e for e in last_of_month.values() if e > ctx.date and _days(ctx.date, e) >= min_days)
    if not monthly:
        return None
    # the last expiry of the latest listed month may be a quarterly with later weeklies unlisted: fine, it is the
    # last listed expiry of that month either way
    return _chain(ctx, monthly[0])


def _sessions_ahead(day: int, k: int) -> int | None:
    d = int(day)
    for _ in range(k):
        d = N.next_fo_session(d)
        if d is None:
            return None
    return d


# ══════════════════════════════════════════════════════════════════════════
# plan builders
# ══════════════════════════════════════════════════════════════════════════

def _plan(c, legs, exit="expiry", note=""):
    if any(l[1] is None for l in legs):
        return None
    if len({(l[0], l[1]) for l in legs}) != len(legs):
        return None
    return N.Plan(int(c.expiry), [N.Leg(t, float(k), s, 1) for t, k, s in legs], exit=exit, note=note)


def _strangle_pct(c, otm, exit="expiry", wing=None):
    kc = _nearest(c, "CE", c.spot * (1 + otm))
    kp = _nearest(c, "PE", c.spot * (1 - otm))
    legs = [("CE", kc, -1), ("PE", kp, -1)]
    if wing is not None and kc is not None and kp is not None:
        legs += [("CE", _wing(c, "CE", kc, wing), +1), ("PE", _wing(c, "PE", kp, wing), +1)]
    return _plan(c, legs, exit, f"spot {c.spot:.1f}")


def _strangle_prem(c, prem, exit="expiry", wing=None):
    tgt = c.spot * prem
    kc, kp = _by_premium(c, "CE", tgt), _by_premium(c, "PE", tgt)
    legs = [("CE", kc, -1), ("PE", kp, -1)]
    if wing is not None and kc is not None and kp is not None:
        legs += [("CE", _wing(c, "CE", kc, wing), +1), ("PE", _wing(c, "PE", kp, wing), +1)]
    return _plan(c, legs, exit, f"spot {c.spot:.1f} target {tgt:.1f}")


def _oiwall(c, buffer, exit="expiry"):
    wc, wp = _walls(c)
    if wc is None:
        return None
    if buffer > 0:
        kc = _nearest_any(c, "CE", wc + buffer * c.spot, wc + buffer * c.spot)
        kp = _nearest_any(c, "PE", wp - buffer * c.spot, wp - buffer * c.spot)
    else:
        kc = wc if _traded(c, "CE")[c.strike == wc].any() else None
        kp = wp if _traded(c, "PE")[c.strike == wp].any() else None
    return _plan(c, [("CE", kc, -1), ("PE", kp, -1)], exit, f"spot {c.spot:.1f} walls {wc:g}/{wp:g}")


# ══════════════════════════════════════════════════════════════════════════
# decide / manage functions (one per idea; parameters from p)
# ══════════════════════════════════════════════════════════════════════════

def d_wk_strangle_pct(ctx, p):
    c = _weekly_chain(ctx)
    return None if c is None else _strangle_pct(c, p.otm, wing=getattr(p, "wing", None))


def d_wk_strangle_prem(ctx, p):
    c = _weekly_chain(ctx)
    return None if c is None else _strangle_prem(c, p.prem, wing=getattr(p, "wing", None))


def d_wk_straddle(ctx, p):
    c = _weekly_chain(ctx)
    if c is None:
        return None
    k = _atm(c)
    return _plan(c, [("CE", k, -1), ("PE", k, -1)], "expiry", f"spot {c.spot:.1f}")


def m_straddle_stop(ctx, pos, p):
    """Previous session's EOD cost-to-close vs the entry credit (one-session lag, see prereg)."""
    marks = _marks(ctx, pos)
    if not marks:
        return False
    v = marks[-1][1]
    if not v == v:
        return False
    credit = -sum(l.side * l.lots * f for l, f in zip(pos.plan.legs, pos.fills))
    if credit <= 0:
        return False
    cost = -v
    if cost >= p.stop_mult * credit:
        return True
    tp = getattr(p, "tp_mult", None)
    return tp is not None and cost <= tp * credit


def d_wk_trendspread(ctx, p):
    c = _weekly_chain(ctx)
    if c is None:
        return None
    r = _rows(ctx, ctx.asof.fut_oi(n=p.sma, sym=SYM_NF))
    if r is None or len(r) < p.sma or r.date[-1] != c.date:
        return None
    sp = np.asarray(r.spot, float)
    if not np.isfinite(sp).all():
        return None
    m = float(sp.mean())
    if sp[-1] > m:
        ks = _nearest(c, "PE", c.spot * (1 - p.otm))
        kl = None if ks is None else _wing(c, "PE", ks, p.width)
        return _plan(c, [("PE", ks, -1), ("PE", kl, +1)], "expiry", f"UP {sp[-1]:.1f}>{m:.1f}")
    if sp[-1] < m:
        ks = _nearest(c, "CE", c.spot * (1 + p.otm))
        kl = None if ks is None else _wing(c, "CE", ks, p.width)
        return _plan(c, [("CE", ks, -1), ("CE", kl, +1)], "expiry", f"DOWN {sp[-1]:.1f}<{m:.1f}")
    return None


def d_bnf_mo_strangle(ctx, p):
    c = _monthly_chain(ctx)
    return None if c is None else _strangle_pct(c, p.otm, exit=p.hold, wing=getattr(p, "wing", None))


def d_expwk(ctx, p):
    """Enter at the close of the session `p.ahead` F&O sessions before the near expiry E."""
    c = _chain(ctx, "near")
    if c is None or not c.expiry > ctx.date:
        return None
    if _sessions_ahead(ctx.date, p.ahead) != c.expiry:
        return None
    return _strangle_pct(c, p.otm, wing=getattr(p, "wing", None))


def d_event_straddle(ctx, p):
    e = _sessions_ahead(ctx.date, p.ahead)
    if e is None or e not in EVENT_DAYS:
        return None
    exs = [x for x in _expiries(ctx) if x > e]
    if not exs:
        return None
    c = _chain(ctx, exs[0])
    if c is None:
        return None
    k = _atm(c)
    return _plan(c, [("CE", k, +1), ("PE", k, +1)], p.ahead, f"event {e} spot {c.spot:.1f}")


def d_wk_oiwall(ctx, p):
    c = _weekly_chain(ctx)
    return None if c is None else _oiwall(c, p.buffer)


def d_bnf_mo_oiwall(ctx, p):
    c = _monthly_chain(ctx)
    return None if c is None else _oiwall(c, p.buffer, exit=p.hold)


# ══════════════════════════════════════════════════════════════════════════
# the pre-registered specs
# ══════════════════════════════════════════════════════════════════════════
_P = "r4_realopt_"
NF, BNF = (SYM_NF,), (SYM_BNF,)


def _S(i, desc, params, decide, und=NF, manage=None):
    return N.OptionSpec(_P + i, desc, params, decide, underlyings=und, manage=manage, family="r4_realopt")


SPECS = [
    _S("nf_wk_strangle_2pct", "NIFTY weekly short strangle 2% OTM, hold to expiry", {"otm": 0.02}, d_wk_strangle_pct),
    _S("nf_wk_strangle_3pct", "NIFTY weekly short strangle 3% OTM, hold to expiry", {"otm": 0.03}, d_wk_strangle_pct),
    _S("nf_wk_strangle_prem10bp", "NIFTY weekly short strangle, legs priced ~0.10% of spot", {"prem": 0.001},
       d_wk_strangle_prem),
    _S("nf_wk_condor_2pct_w1", "NIFTY weekly iron condor 2% OTM shorts, wings 1% further", {"otm": 0.02, "wing": 0.01},
       d_wk_strangle_pct),
    _S("nf_wk_condor_prem10bp_w1", "NIFTY weekly iron condor, shorts ~0.10% of spot, wings 1% further",
       {"prem": 0.001, "wing": 0.01}, d_wk_strangle_prem),
    _S("nf_wk_straddle_stop15", "NIFTY weekly short ATM straddle, EOD stop 1.5x credit (1-session lag)",
       {"stop_mult": 1.5}, d_wk_straddle, manage=m_straddle_stop),
    _S("nf_wk_straddle_stop15_tp50", "NIFTY weekly short ATM straddle, stop 1.5x / take profit 0.5x credit",
       {"stop_mult": 1.5, "tp_mult": 0.5}, d_wk_straddle, manage=m_straddle_stop),
    _S("nf_wk_trendspread_sma20", "NIFTY weekly credit spread in SMA20 trend direction, 1% OTM, 1% wide",
       {"sma": 20, "otm": 0.01, "width": 0.01}, d_wk_trendspread),
    _S("nf_wk_trendspread_sma50", "NIFTY weekly credit spread in SMA50 trend direction, 1% OTM, 1% wide",
       {"sma": 50, "otm": 0.01, "width": 0.01}, d_wk_trendspread),
    _S("bnf_mo_strangle_3pct_h5", "BANKNIFTY monthly short strangle 3% OTM, exit after 5 sessions",
       {"otm": 0.03, "hold": 5}, d_bnf_mo_strangle, und=BNF),
    _S("bnf_mo_condor_3pct_w2_h5", "BANKNIFTY monthly iron condor 3% OTM, wings 2%, exit after 5 sessions",
       {"otm": 0.03, "wing": 0.02, "hold": 5}, d_bnf_mo_strangle, und=BNF),
    _S("nf_expwk_last2_strangle_1pct", "NIFTY short strangle 1% OTM entered 2 sessions before expiry, to settlement",
       {"otm": 0.01, "ahead": 2}, d_expwk),
    _S("nf_expwk_last2_condor_1pct_w1", "NIFTY iron condor 1% OTM, wings 1%, entered 2 sessions before expiry",
       {"otm": 0.01, "wing": 0.01, "ahead": 2}, d_expwk),
    _S("nf_exp_eve_strangle_1pct", "NIFTY short strangle 1% OTM entered the session before expiry, to settlement",
       {"otm": 0.01, "ahead": 1}, d_expwk),
    _S("nf_event_straddle_eve", "NIFTY long ATM straddle at the close before RBI/Budget day, exit event-day close",
       {"ahead": 1}, d_event_straddle),
    _S("nf_event_straddle_2d", "NIFTY long ATM straddle 2 sessions before RBI/Budget day, exit event-day close",
       {"ahead": 2}, d_event_straddle),
    _S("nf_wk_oiwall_beyond", "NIFTY weekly short strangle 0.5% beyond the max-OI call/put walls", {"buffer": 0.005},
       d_wk_oiwall),
    _S("nf_wk_oiwall_at", "NIFTY weekly short strangle at the max-OI call/put walls", {"buffer": 0.0}, d_wk_oiwall),
    _S("bnf_mo_oiwall_beyond_h5", "BANKNIFTY monthly short strangle 0.5% beyond the OI walls, exit after 5 sessions",
       {"buffer": 0.005, "hold": 5}, d_bnf_mo_oiwall, und=BNF),
]
assert len({s.id for s in SPECS}) == len(SPECS) == 19


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _select(ids: str) -> list:
    if not ids:
        return list(SPECS)
    want = [x.strip() for x in ids.split(",") if x.strip()]
    known = {s.id: s for s in SPECS}
    bad = [w for w in want if w not in known]
    if bad:
        raise SystemExit(f"unknown ids: {bad}")
    return [known[w] for w in want]


def _worker(args):
    sid, data, out, cache, screen_only = args
    spec = _select(sid)[0]
    part = os.path.join(out, "_parts", sid)
    res = N.run_option_specs([spec], data, part, cache, screen_only=screen_only)
    return sid, res


def run(specs, data, out, cache, jobs, screen_only):
    from . import data as D
    from . import protocol as P
    out = D.assert_outside_repo(out)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    tasks = [(s.id, data, out, cache, screen_only) for s in specs]
    if jobs > 1:
        with mp.get_context("fork").Pool(jobs) as pool:
            got = dict(pool.map(_worker, tasks, chunksize=1))
    else:
        got = dict(_worker(t) for t in tasks)
    texts, rows, results = [], [], []
    for s in specs:
        part = os.path.join(out, "_parts", s.id)
        with open(os.path.join(part, "summary.txt")) as fh:
            texts.append(fh.read())
        rp = os.path.join(part, "results.csv")
        if os.path.exists(rp):
            with open(rp) as fh:
                rows += list(csv.DictReader(fh))
        tp = os.path.join(part, "trades", s.id + ".csv")
        if os.path.exists(tp):
            shutil.copy(tp, os.path.join(out, "trades", s.id + ".csv"))
        results += got[s.id]
    text = "\n".join(texts) + "\nALL r4_realopt SPECS:\n" + P.luck_reminder(results) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text)
    if rows:
        with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out}/summary.txt, results.csv, trades/ ({len(specs)} specs)")


def smoke(specs, data, lo, hi):
    """Correctness only: run the simulator (with the look-ahead audit) on a short window and print trade counts,
    skipped entries, exit reasons, errors and look-ahead flags - never P&L. Also checks the explicit guard."""
    cal = N.lab_calendar(data)
    # the guard must fire if a chain dated the decision day were ever read
    d0 = int(cal[len(cal) // 4])
    fake = N.Ctx(SYM_NF, d0, N.Asof(SYM_NF, d0, d0), 0)
    c_today = N.options_chain(SYM_NF, d0 + 1, "near")       # a chain dated d0 itself (what must never be used)
    assert c_today is not None and c_today.date == d0
    try:
        _check_chain(fake, c_today)
        print("GUARD FAILED: same-day chain accepted")
    except N.LookAheadError as e:
        print("guard OK (chain):", e)
    try:
        _rows(fake, N.fut_oi(SYM_NF, d0 + 1, 3))
        print("GUARD FAILED: same-day rows accepted")
    except N.LookAheadError as e:
        print("guard OK (rows):", e)
    try:
        with N.cutoff(d0):
            _chain(fake)
            print("guard OK (normal read under cutoff returns the previous session's chain)")
    except N.LookAheadError as e:
        print("GUARD FAILED: normal read raised", e)
    det = {}
    res = N.make_simulator(cal, audit=True, details=det)(specs, lo, hi)
    for s in specs:
        r = res[s.id]
        reasons = {}
        for t in r["trades"]:
            reasons[t[13]] = reasons.get(t[13], 0) + 1
        print(f"{s.id}: trades={len(r['trades'])} skipped={r['skipped']} reasons={reasons} "
              f"error={r['error']} lookahead={r['lookahead']}")
        for k in sorted(det.get(s.id, {}))[:2]:
            x = det[s.id][k]
            print(f"    e.g. entry {x['date']} exit {x['exit_date']} expiry {x['expiry']} lot {x['lot']} legs {x['legs']}"
                  f" ({x['note']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.options_r4")
    ap.add_argument("--data", default=N.BARS_DIR)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="short window, counts / errors / look-ahead only")
    ap.add_argument("--smoke-from", type=int, default=20240101)
    ap.add_argument("--smoke-to", type=int, default=20240415)
    a = ap.parse_args(argv)
    specs = _select(a.ids)
    if a.list:
        for s in specs:
            print(f"{s.id:40s} {s.underlyings[0]:10s} {s.desc}")
        return 0
    if a.smoke:
        smoke(specs, a.data, a.smoke_from, a.smoke_to)
        return 0
    if not a.out:
        ap.error("--out is required")
    run(specs, a.data, a.out, a.cache, a.jobs, a.screen_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
