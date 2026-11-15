"""
stratlab/kite_options.py — round-5 Kite-ecosystem option strategies that real NSE END-OF-DAY option prices can
approximate (Tradetron blog builds), run through nsedata's OptionSpec simulator and the unchanged protocol
(DISCOVERY / FINAL split, FINAL once via the data directory's final_ledger.json). Research only.

Pre-registration: stratlab/prereg/kite_ecosystem_v1.json. What is approximated (stated there, before any run):
  * fills at NSE's official EOD close of the entry / exit session (the sources trade at 09:20 / intraday), with
    the simulator's haircut max(Rs 0.05, 1 %) per leg per fill and 2026 option fees;
  * stops are checked on EOD marks only, with a one-session lag (manage() sees marks dated before today and closes
    at today's close): intraday stop touches are invisible, so an EOD-only stop is LOOSER than the published one.

    r5_kite_tt_ic_wed       Tradetron "automate an options strategy": weekly NIFTY iron condor, sell ATM+-300, buy
                            ATM+-500, 1 lot, entered on the first session after each weekly expiry (the source's
                            Wednesday, under today's Tuesday expiry), held to expiry, exit all if P&L <= -Rs 4,000.
    r5_kite_tt_dst_spreads  Tradetron double-Supertrend "low-risk" version: both daily Supertrends (10,3) and (20,2)
                            green -> bull call spread (buy ~0.60-delta CE, sell ~0.30-delta CE); both red -> bear put
                            spread (buy ~-0.60, sell ~-0.30 PE); near MONTHLY expiry >= 10 calendar days away.
                            Exit (unspecified in the source; default fixed in the prereg): when the two Supertrends
                            disagree, or at expiry.

    cd algotrader_v4
    python -m stratlab.kite_options --data <bars3y> --out <scratch dir> [--ids ...] [--screen-only]
    python -m stratlab.kite_options --oos --prereg <PREREGISTERED_KITE.json> --out <scratch dir> --ids r5_kite_tt_dst_spreads
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from . import nsedata as N
from . import options_r4 as R4
from . import kite_ind as K

_SCRATCH = os.environ.get("STRATLAB_SCRATCH",
                          "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad")
OLD_INDEX = os.path.join(_SCRATCH, "nse_old", "index_daily.csv.gz")
NEW_INDEX = os.path.join(_SCRATCH, "stratlab_cache", "bars3y_fb3d4d", "_NIFTY50_day.npz")


# ══════════════════════════════════════════════════════════════════════════
# Tradetron weekly iron condor
# ══════════════════════════════════════════════════════════════════════════
def d_ic_wed(ctx, p):
    c = R4._weekly_chain(ctx)               # first F&O session after a weekly expiry; previous session's chain
    if c is None:
        return None
    atm = R4._atm(c)
    if atm is None:
        return None
    ks = {k: atm + k for k in (p.short_otm, p.long_otm, -p.short_otm, -p.long_otm)}
    legs = [("CE", ks[p.short_otm], -1), ("CE", ks[p.long_otm], +1),
            ("PE", ks[-p.short_otm], -1), ("PE", ks[-p.long_otm], +1)]
    for typ, k, _s in legs:
        ok = R4._traded(c, typ) & (c.strike == k)
        if not ok.any():
            return None
    return R4._plan(c, legs, "expiry", f"spot {c.spot:.1f} atm {atm:g}")


def m_ic_stop(ctx, pos, p):
    """Previous session's EOD mark-to-market P&L (rupees, 1 lot) <= -stop_rs -> close at today's close."""
    marks = R4._marks(ctx, pos)
    if not marks:
        return False
    v = marks[-1][1]
    if not v == v:
        return False
    entry_val = sum(l.side * l.lots * f for l, f in zip(pos.plan.legs, pos.fills))
    pnl = (v - entry_val) * pos.qty_per_lot
    return pnl <= -p.stop_rs


# ══════════════════════════════════════════════════════════════════════════
# Tradetron double Supertrend, debit spreads on NIFTY monthly options
# ══════════════════════════════════════════════════════════════════════════
_NIFTY_D: dict = {}


def nifty_daily() -> dict:
    """NIFTY 50 daily OHLC: official NSE values 2012-01-02 .. 2023-11-09 (nse_old/index_daily.csv.gz), then the lab's
    minute-derived daily bars (bars3y _NIFTY50_day) from 2023-11-10. Loaded once; callers slice to dates < today."""
    if _NIFTY_D:
        return _NIFTY_D
    import pandas as pd
    parts = []
    if os.path.exists(OLD_INDEX):
        df = pd.read_csv(OLD_INDEX, usecols=["date", "index", "open", "high", "low", "close"])
        df = df[df["index"] == "NIFTY"].sort_values("date")
        parts.append(df[["date", "open", "high", "low", "close"]].to_numpy(float))
    if os.path.exists(NEW_INDEX):
        with np.load(NEW_INDEX) as z:
            a = np.column_stack([z["date"], z["o"], z["h"], z["l"], z["c"]]).astype(float)
        last_old = parts[0][-1, 0] if parts else 0
        parts.append(a[a[:, 0] > last_old])
    m = np.vstack(parts)
    _NIFTY_D.update({"date": m[:, 0].astype(np.int64), "o": m[:, 1], "h": m[:, 2], "l": m[:, 3], "c": m[:, 4]})
    return _NIFTY_D


def dst_state_before(day: int) -> int:
    """Double-Supertrend state from NIFTY daily candles dated strictly before `day` (EOD of the previous session)."""
    d = nifty_daily()
    k = int(np.searchsorted(d["date"], int(day), "left"))
    if k < 25:
        return 0
    if int(d["date"][k - 1]) >= int(day):
        raise N.LookAheadError("dst_state_before read a candle dated on/after the decision day")
    lo = max(0, k - 400)                   # 400 sessions of history: the Wilder ATR has long converged
    return int(K.double_supertrend(d["h"][lo:k], d["l"][lo:k], d["c"][lo:k])[-1])


def _iv(price, S, K_, T, typ):
    from options_replay import bs
    if not (price > 0 and S > 0 and T > 0):
        return float("nan")
    intr = max(0.0, S - K_) if typ == "CE" else max(0.0, K_ - S)
    if price <= intr + 1e-9:
        return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs(S, K_, T, mid, typ) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _delta_strike(c, typ, target, T):
    """Traded strike of type typ whose Black-Scholes delta (implied vol from its own EOD close) is nearest target."""
    from options_replay import bs_delta
    close = c.ce_close if typ == "CE" else c.pe_close
    ok = R4._traded(c, typ)
    best, bk = None, None
    for i in np.flatnonzero(ok):
        sig = _iv(float(close[i]), float(c.spot), float(c.strike[i]), T, typ)
        if not sig == sig:
            continue
        dlt = bs_delta(float(c.spot), float(c.strike[i]), T, sig, typ)
        err = abs(dlt - target)
        if best is None or err < best:
            best, bk = err, float(c.strike[i])
    return bk


def d_dst_spread(ctx, p):
    st = dst_state_before(ctx.date)
    if st == 0:
        return None
    c = R4._monthly_chain(ctx, min_days=p.min_days)
    if c is None or not c.spot == c.spot:
        return None
    T = max(R4._days(c.date, c.expiry), 1) / 365.0
    typ = "CE" if st == 1 else "PE"
    sgn = 1.0 if st == 1 else -1.0
    kb, ks = _delta_strike(c, typ, sgn * p.buy_delta, T), _delta_strike(c, typ, sgn * p.sell_delta, T)
    if kb is None or ks is None or kb == ks:
        return None
    return R4._plan(c, [(typ, kb, +1), (typ, ks, -1)], "expiry", f"dst {st:+d} spot {c.spot:.1f}")


def m_dst_exit(ctx, pos, p):
    """Close at today's close when the Supertrends (as of yesterday's EOD) no longer agree in the trade's direction."""
    want = 1 if pos.plan.legs[0].typ == "CE" else -1
    return dst_state_before(ctx.date) != want


_P = "r5_kite_"
SPECS = [
    N.OptionSpec(_P + "tt_ic_wed", "Tradetron weekly NIFTY iron condor ATM+-300 / +-500, held to expiry, EOD stop -Rs4,000 (1 lot)",
                 {"short_otm": 300, "long_otm": 500, "stop_rs": 4000}, d_ic_wed, underlyings=("NIFTY",),
                 manage=m_ic_stop, family="kite_tradetron"),
    N.OptionSpec(_P + "tt_dst_spreads", "Tradetron double Supertrend debit spreads (0.60/0.30 delta), NIFTY monthly, exit on disagreement",
                 {"st1": (10, 3.0), "st2": (20, 2.0), "buy_delta": 0.60, "sell_delta": 0.30, "min_days": 10},
                 d_dst_spread, underlyings=("NIFTY",), manage=m_dst_exit, family="kite_tradetron"),
]
BY_ID = {s.id: s for s in SPECS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.kite_options")
    ap.add_argument("--data", default=N.BARS_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--oos", action="store_true", help="2012-23 OOS judgement (protocol_oos, one look)")
    ap.add_argument("--prereg", default="")
    a = ap.parse_args(argv)
    want = [x for x in a.ids.split(",") if x] or list(BY_ID)
    specs = [BY_ID[w] for w in want]
    from .data import assert_outside_repo
    out = assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    if a.oos:
        import json
        from . import protocol_oos as PO
        prereg = json.load(open(a.prereg)) if a.prereg else None
        PO.setup()
        res = PO.run_options(specs, prereg)
        print(PO._write(out, res, f"kite_options OOS, prereg {a.prereg or 'NONE'}"))
        return 0
    N.run_option_specs(specs, a.data, out, a.cache, screen_only=a.screen_only)
    print("record it in the trials ledger with: python -m stratlab.trials backfill (reads this results.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
