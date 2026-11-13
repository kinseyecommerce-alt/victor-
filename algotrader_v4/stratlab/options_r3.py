"""
stratlab/options_r3.py — round-3 F&O OPTIONS candidates (replacement for the losing stock-option buying agent),
judged by the lab's protocol exactly like stratlab/options_vrp.py.

    cd algotrader_v4
    python -m stratlab.options_r3 --data <bars3y dir> --cache <lab cache> --out <scratch dir> [--ids a,b] [--jobs 3]
                                  [--screen-only]

Pre-registration: scratchpad r3/options/PREREGISTERED.txt (17 ids, prefix r3_options_). Do not tune on results.

Machinery reused from options_vrp / options_replay: Black-Scholes on the real NIFTY 50 / NIFTY BANK minute bars, India
VIX previous close as one flat vol, the same per-leg haircut (max(1 pt / 2 pts, 2 % of premium)), the same
costs.round_trip_cost(..., "option") per leg, settlement at intrinsic vs the 15:00-15:29 mean, and the unchanged
protocol.evaluate() (DISCOVERY / FINAL split, H1 / H2, PF >= 1.15, >= 60 positional trades, FINAL once through the
data directory's final_ledger.json, no --peek-final). Additions here (own fill-model tag "o1r3" in the ledger key):
  * 0-DTE entries (ReplayConfig.allow_0dte for those specs only),
  * a session-end exit on a later session (BANKNIFTY weekly harvest: 15:15 of the last session of the week),
  * VIX-percentile gates (past VIX closes only), weekly-cycle / first- / last-session-of-week schedules,
  * delta-chosen wings, trend-signed credit spreads, intraday long straddles.
Sensitivities (DISCOVERY only): haircut x2.5 ("net@5", with its own screen), approximate historical lots, and a
trading-time clock (decay only in market hours) that exposes weekend-theta artefacts of the calendar clock.

Model limits: flat vol (no skew: OTM puts richer, calls cheaper, far wings richer in reality); VIX is a 30-day vol,
not the weekly or 0-DTE implied vol; vol is constant within a session; credits are model values, not quotes. A pass
earns a paper-trading trial, nothing more. Nothing here places orders.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import costs as _costs  # noqa: E402
import options_replay as OR  # noqa: E402

from . import data as D  # noqa: E402
from . import options_vrp as V  # noqa: E402
from . import protocol as P  # noqa: E402

OPT_FILL_MODEL = "o1r3"
COST_MODEL = f"{_costs.COST_MODEL_VERSION}/{OPT_FILL_MODEL}"
SENS_MULT = V.SENS_MULT
NAKED_MARGIN_PCT = V.NAKED_MARGIN_PCT
ELM_PCT = V.ELM_PCT
VIX_LOOKBACK = 250

LOTS = {"NIFTY": V.LOT_SCHEDULE, "BANKNIFTY": ((date(2000, 1, 1), 30),)}          # pre-registered
LOTS_HIST = {"NIFTY": V.LOT_SCHEDULE_HISTORICAL,                                   # sensitivity, approximate
             "BANKNIFTY": ((date(2000, 1, 1), 15), (date(2024, 11, 20), 30), (date(2025, 4, 25), 35),
                           (date(2025, 12, 31), 30))}


# ══════════════════════════════════════════════════════════════════════════
# schedule / filter helpers (past data and the published trading calendar only)
# ══════════════════════════════════════════════════════════════════════════

def vix_pct(vix: dict, entry_day: date, lookback: int = VIX_LOOKBACK) -> Optional[float]:
    """Share of the `lookback` VIX closes before p that are strictly below VIX[p], p = last VIX day < entry_day."""
    days = sorted(d for d in vix if d < entry_day)
    if len(days) < lookback + 1:
        return None
    p = days[-1]
    hist = [vix[d] for d in days[-1 - lookback:-1]]
    return sum(1 for x in hist if x < vix[p]) / len(hist)


def first_session_of_week(d: date, is_td) -> bool:
    prev = OR.previous_trading_day(d, is_td)
    return prev.isocalendar()[:2] != d.isocalendar()[:2]


def last_session_of_week(d: date, is_td) -> bool:
    x = d + timedelta(days=1)
    while x.isocalendar()[:2] == d.isocalendar()[:2]:
        if is_td(x):
            return False
        x += timedelta(days=1)
    return True


def after_weekly_expiry(v: OR.MarketView) -> bool:
    is_td = v.is_td or OR._calendar_trading_day
    prev = OR.previous_trading_day(v.now.date(), is_td)
    return OR.next_expiry(v.underlying, prev, is_td, skip_today=False) == prev


# ══════════════════════════════════════════════════════════════════════════
# one generic strategy class, configured by the spec's params
# ══════════════════════════════════════════════════════════════════════════

class R3Strategy(OR.OptionStrategy):
    """params keys:
      schedule   weekly_cycle | expiry_day | week_first | week_last | daily
      entry      "HH:MM"; hold "expiry" | "intraday" (15:15 exit); week_end_exit: close at 15:15 of the last session
                 of the ISO week (unless that day is the expiry, which settles)
      structure  strangle (short_delta, wing_delta | wing_pts) | straddle_short (wing_pts) | straddle_long |
                 credit_trend (sma, short_delta, width_pts)
      stop_mult / target_frac (None = none); min_days_left; vix_pct_min / vix_pct_max."""
    name = "R3"

    def __init__(self, params: dict, vix: dict):
        self.p, self.vix = dict(params), vix
        hh, mm = map(int, self.p["entry"].split(":"))
        self.decision_times = [time(hh, mm)]
        self.hold = self.p.get("hold", "expiry")
        self.time_exit = time(15, 15)
        self.stop_mult = self.p.get("stop_mult")
        self.target_frac = self.p.get("target_frac")
        self.week_end_exit = bool(self.p.get("week_end_exit"))
        self.exits_early = not (self.hold == "expiry" and self.stop_mult is None and self.target_frac is None)
        self.skipped: list = []
        self.last_kind = ""

    def exit_reason(self, entry_net, pnl_close, close_cost_worst, pnl_worst):
        if entry_net > 0:
            if self.stop_mult is not None and close_cost_worst >= self.stop_mult * entry_net:
                return "stop"
            if self.target_frac is not None and pnl_close >= self.target_frac * entry_net:
                return "target"
        else:
            debit = -entry_net
            if self.p.get("stop_frac") is not None and pnl_worst <= -self.p["stop_frac"] * debit:
                return "stop"
            if self.target_frac is not None and pnl_close >= self.target_frac * debit:
                return "target"
        return None

    def week_end(self, tr, d: date, is_td) -> bool:
        return self.week_end_exit and d != tr.expiry and last_session_of_week(d, is_td)

    # ── schedule ─────────────────────────────────────────────────────────
    def _scheduled(self, v: OR.MarketView) -> bool:
        is_td = v.is_td or OR._calendar_trading_day
        d, sch = v.now.date(), self.p["schedule"]
        if sch == "weekly_cycle":
            if not after_weekly_expiry(v):
                return False
        elif sch == "expiry_day":
            return v.expiry == d                                        # 0-DTE: needs allow_0dte
        elif sch == "week_first":
            if not first_session_of_week(d, is_td):
                return False
        elif sch == "week_last":
            if not last_session_of_week(d, is_td):
                return False
        elif sch != "daily":
            raise ValueError(f"unknown schedule {sch}")
        if v.expiry <= d:                                               # never 0-DTE outside the expiry_day schedule
            return False
        if OR.trading_days_left(d, v.expiry, is_td) < self.p.get("min_days_left", 0):
            self.skipped.append(d)
            return False
        return True

    def _vix_ok(self, d: date) -> bool:
        lo, hi = self.p.get("vix_pct_min"), self.p.get("vix_pct_max")
        if lo is None and hi is None:
            return True
        q = vix_pct(self.vix, d)
        if q is None:
            return False
        return (lo is None or q >= lo - 1e-9) and (hi is None or q <= hi + 1e-9)

    # ── legs ─────────────────────────────────────────────────────────────
    def decide(self, v: OR.MarketView):
        if not self._scheduled(v) or not self._vix_ok(v.now.date()):
            return None
        s = self.p["structure"]
        if s == "strangle":
            kc = v.strike_for_delta("CE", self.p["short_delta"])
            kp = v.strike_for_delta("PE", self.p["short_delta"])
            legs = [OR.LegSpec("CE", kc, -1), OR.LegSpec("PE", kp, -1)]
            if self.p.get("wing_delta"):
                wc = v.strike_for_delta("CE", self.p["wing_delta"])
                wp = v.strike_for_delta("PE", self.p["wing_delta"])
                wc, wp = max(wc, kc + v.step), min(wp, kp - v.step)     # a wing is always beyond its short
                legs += [OR.LegSpec("CE", wc, +1), OR.LegSpec("PE", wp, +1)]
            elif self.p.get("wing_pts"):
                legs += [OR.LegSpec("CE", kc + self.p["wing_pts"], +1), OR.LegSpec("PE", kp - self.p["wing_pts"], +1)]
            return legs
        if s == "straddle_short":
            k = v.atm()
            legs = [OR.LegSpec("CE", k, -1), OR.LegSpec("PE", k, -1)]
            if self.p.get("wing_pts"):
                legs += [OR.LegSpec("CE", k + self.p["wing_pts"], +1), OR.LegSpec("PE", k - self.p["wing_pts"], +1)]
            return legs
        if s == "straddle_long":
            k = v.atm()
            return [OR.LegSpec("CE", k, +1), OR.LegSpec("PE", k, +1)]
        if s == "credit_trend":
            closes = [c for _, c in v.prior_closes][-self.p["sma"]:]
            if len(closes) < self.p["sma"]:
                return None
            w = self.p["width_pts"]
            if closes[-1] > sum(closes) / len(closes):
                self.last_kind = "BULL_PUT"
                k = v.strike_for_delta("PE", self.p["short_delta"])
                return [OR.LegSpec("PE", k, -1), OR.LegSpec("PE", k - w, +1)]
            self.last_kind = "BEAR_CALL"
            k = v.strike_for_delta("CE", self.p["short_delta"])
            return [OR.LegSpec("CE", k, -1), OR.LegSpec("CE", k + w, +1)]
        raise ValueError(f"unknown structure {s}")


class R3Replay(OR.Replay):
    """options_replay.Replay plus one exit: R3Strategy.week_end() closes at the first bar at/after 15:15 (its open,
    executable model values) of the last session of the ISO week, or at the last bar's close if none."""

    def _manage(self, tr, b, sigma, d, day, i):
        st = self.strat
        if getattr(st, "week_end_exit", False) and st.week_end(tr, d, self.is_td):
            ts, o, h, lo, c = b
            sig = sigma if sigma is not None else tr.sigma
            if ts.time() >= st.time_exit:
                _, fills = self._close_value(tr, o, ts, sig)
                self._close(tr, ts, o, fills, "week_end")
                return
            if i == len(day) - 1:
                _, fills = self._close_value(tr, c, ts, sig)
                self._close(tr, ts, c, fills, "week_end")
                return
        super()._manage(tr, b, sigma, d, day, i)


# ══════════════════════════════════════════════════════════════════════════
# specs
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class R3Spec(V.OptSpec):
    allow_0dte: bool = False
    vix: Optional[dict] = field(default=None, repr=False, compare=False)

    def strategy(self) -> R3Strategy:
        return R3Strategy(self.params, self.vix or {})

    def margin(self, tr: OR.Trade) -> float:
        notional = tr.spot_in * tr.qty
        if tr.entry_net <= 0:                                   # long premium: the debit
            return max(tr.max_risk, 1.0)
        if tr.max_risk > 0:
            return tr.max_risk + ELM_PCT * notional
        return NAKED_MARGIN_PCT * notional

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(x) for x in (R3Strategy, R3Replay, R3Spec, vix_pct, first_session_of_week,
                                                     last_session_of_week, after_weekly_expiry, OR.Replay,
                                                     OR.leg_cost, OR.fill_price, OR.next_expiry, _run_replay,
                                                     V._trade_tuple))
        blob = repr((self.id, self.horizon, self.underlying, self.lots, self.allow_0dte,
                     sorted(self.params.items()), src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _S(id_, family, desc, params, und="NIFTY", allow_0dte=False):
    return R3Spec(id_, desc, params, family=family, underlying=und, lots=1, allow_0dte=allow_0dte)


WC = {"schedule": "weekly_cycle", "entry": "10:20", "min_days_left": 3}
SPECS = [
    _S("r3_options_ic_10_03", "iron_condor",
       "Weekly NIFTY iron condor: short 0.10-delta CE/PE, long 0.03-delta wings, 10:20 after expiry, hold to expiry",
       {**WC, "structure": "strangle", "short_delta": 0.10, "wing_delta": 0.03}),
    _S("r3_options_ic_25_10", "iron_condor",
       "Weekly NIFTY iron condor: short 0.25-delta CE/PE, long 0.10-delta wings, hold to expiry",
       {**WC, "structure": "strangle", "short_delta": 0.25, "wing_delta": 0.10}),
    _S("r3_options_ibf_400", "iron_butterfly",
       "Weekly NIFTY iron butterfly: short ATM CE+PE, long +/-400 wings, hold to expiry",
       {**WC, "structure": "straddle_short", "wing_pts": 400.0}),
    _S("r3_options_ibf_400_tp25", "iron_butterfly",
       "As r3_options_ibf_400 with a 25 %-of-credit profit target",
       {**WC, "structure": "straddle_short", "wing_pts": 400.0, "target_frac": 0.25}),
    _S("r3_options_ss16_stop2x", "strangle_stop",
       "Weekly NIFTY naked 0.16-delta strangle, stop at 2x credit, else expiry",
       {**WC, "structure": "strangle", "short_delta": 0.16, "stop_mult": 2.0}),
    _S("r3_options_ss16_stop2x_tp50", "strangle_stop",
       "Weekly NIFTY naked 0.16-delta strangle, stop 2x credit, target 50 % of credit",
       {**WC, "structure": "strangle", "short_delta": 0.16, "stop_mult": 2.0, "target_frac": 0.5}),
    _S("r3_options_ss16_stop2x_vix33", "strangle_stop",
       "As r3_options_ss16_stop2x only in weeks with VIX 250-session percentile >= 1/3",
       {**WC, "structure": "strangle", "short_delta": 0.16, "stop_mult": 2.0, "vix_pct_min": 1 / 3}),
    _S("r3_options_0dte_ic_1030", "zero_dte",
       "NIFTY expiry day 10:30: short 0.16-delta same-day CE/PE, +/-200 wings, hold to settlement",
       {"schedule": "expiry_day", "entry": "10:30", "structure": "strangle", "short_delta": 0.16, "wing_pts": 200.0},
       allow_0dte=True),
    _S("r3_options_0dte_ss_1130", "zero_dte",
       "NIFTY expiry day 11:30: short ATM same-day straddle, stop at 1.5x credit, else settlement",
       {"schedule": "expiry_day", "entry": "11:30", "structure": "straddle_short", "stop_mult": 1.5},
       allow_0dte=True),
    _S("r3_options_cs_trend20", "credit_trend",
       "Weekly NIFTY credit spread in the 20-session trend: 0.25-delta short, 200-pt width, hold to expiry",
       {**WC, "structure": "credit_trend", "sma": 20, "short_delta": 0.25, "width_pts": 200.0}),
    _S("r3_options_cs_trend50", "credit_trend",
       "As r3_options_cs_trend20 with the 50-session mean",
       {**WC, "structure": "credit_trend", "sma": 50, "short_delta": 0.25, "width_pts": 200.0}),
    _S("r3_options_wc_fri", "calendar_timing",
       "Last session of the week 15:00: 0.16-delta NIFTY strangle + 400 wings, nearest expiry, hold to expiry",
       {"schedule": "week_last", "entry": "15:00", "min_days_left": 1, "structure": "strangle",
        "short_delta": 0.16, "wing_pts": 400.0}),
    _S("r3_options_wc_mon", "calendar_timing",
       "First session of the week 10:20: 0.16-delta NIFTY strangle + 400 wings, nearest expiry, hold to expiry",
       {"schedule": "week_first", "entry": "10:20", "min_days_left": 1, "structure": "strangle",
        "short_delta": 0.16, "wing_pts": 400.0}),
    _S("r3_options_bnf_wk_ic", "banknifty_monthly",
       "BANKNIFTY monthly: Monday 10:20 short 0.16-delta CE/PE + 1000 wings, exit 15:15 last session of the week",
       {"schedule": "week_first", "entry": "10:20", "min_days_left": 3, "structure": "strangle",
        "short_delta": 0.16, "wing_pts": 1000.0, "week_end_exit": True}, und="BANKNIFTY"),
    _S("r3_options_bnf_wk_ss_stop2x", "banknifty_monthly",
       "BANKNIFTY monthly: Monday 10:20 naked 0.16-delta strangle, stop 2x credit, exit 15:15 last session of week",
       {"schedule": "week_first", "entry": "10:20", "min_days_left": 3, "structure": "strangle",
        "short_delta": 0.16, "stop_mult": 2.0, "week_end_exit": True}, und="BANKNIFTY"),
    _S("r3_options_ls_lowvix", "long_straddle",
       "NIFTY ATM long straddle 09:20-15:15 on sessions with VIX percentile <= 1/3",
       {"schedule": "daily", "entry": "09:20", "hold": "intraday", "structure": "straddle_long",
        "vix_pct_max": 1 / 3}),
    _S("r3_options_ls_highvix", "long_straddle",
       "NIFTY ATM long straddle 09:20-15:15 on sessions with VIX percentile >= 2/3",
       {"schedule": "daily", "entry": "09:20", "hold": "intraday", "structure": "straddle_long",
        "vix_pct_min": 2 / 3}),
]
BY_ID = {s.id: s for s in SPECS}


# ══════════════════════════════════════════════════════════════════════════
# data / simulation
# ══════════════════════════════════════════════════════════════════════════

def load(data_dir: str) -> tuple[dict, dict]:
    """{underlying: minute bars without short sessions}, {date: VIX close}."""
    out = {}
    for und, fn in (("NIFTY", "_NIFTY50_min.json"), ("BANKNIFTY", "_NIFTYBANK_min.json")):
        pth = os.path.join(data_dir, fn)
        if not os.path.exists(pth):
            continue
        with open(pth) as fh:
            bars = OR._parse_bars(json.load(fh)["bars"])
        n: dict = {}
        for b in bars:
            n[b[0].date()] = n.get(b[0].date(), 0) + 1
        out[und] = [b for b in bars if n[b[0].date()] >= D.MIN_SESSION_BARS]
    with open(os.path.join(data_dir, "_VIX_day.json")) as fh:
        vix = OR._daily_closes(OR._parse_bars(json.load(fh)["bars"]))
    return out, vix


def _cfg(spec: R3Spec, haircut_mult=1.0, lot_tag="base", time_basis="calendar") -> OR.ReplayConfig:
    sched = (LOTS_HIST if lot_tag == "histlot" else LOTS)[spec.underlying]
    return OR.ReplayConfig(underlying=spec.underlying, lots=spec.lots, lot_schedule=sched,
                           slip_min=OR.SLIP_MIN_PTS[spec.underlying] * haircut_mult,
                           slip_pct=OR.SLIP_PCT * haircut_mult, allow_0dte=spec.allow_0dte, time_basis=time_basis)


def _run_replay(spec: R3Spec, bars, vix, hi: int, haircut_mult=1.0, lot_tag="base", time_basis="calendar"):
    hi_d = V._d(hi)
    use = [b for b in bars if b[0].date() <= hi_d]
    strat = R3Strategy(spec.params, vix)
    res = R3Replay(use, strat, _cfg(spec, haircut_mult, lot_tag, time_basis), vix).run()
    return res, strat


def audit_lookahead(spec: R3Spec, bars, vix, trades: list, n: int = 4) -> Optional[str]:
    """Re-run sampled entries on bars AND VIX closes that end at the entry minute: legs and fills must not change."""
    if not trades:
        return None
    step = max(1, len(trades) // n)
    for tr in trades[::step][:n]:
        cut = [b for b in bars if b[0] <= tr.entry_ts]
        vcut = {k: v for k, v in vix.items() if k <= tr.entry_ts.date()}
        r = R3Replay(cut, R3Strategy(spec.params, vcut), _cfg(spec), vcut).run()
        same = [t for t in r.trades if t.entry_ts == tr.entry_ts]
        if not same:
            return f"entry {tr.entry_ts} disappears when later bars are removed"
        a = [(l.opt, l.strike, l.side, round(l.entry_px, 6)) for l in same[0].legs]
        b = [(l.opt, l.strike, l.side, round(l.entry_px, 6)) for l in tr.legs]
        if a != b:
            return f"entry {tr.entry_ts} changes when later bars are removed: {b} vs {a}"
    return None


_G: dict = {}                     # bars / vix for forked workers


def _work(job):
    sid, lo, hi, haircut_mult, lot_tag, time_basis, audit = job
    sp = BY_ID[sid]
    bars, vix = _G["bars"][sp.underlying], _G["vix"]
    try:
        res, strat = _run_replay(sp, bars, vix, hi, haircut_mult, lot_tag, time_basis)
        trs = [t for t in res.trades if lo <= int(t.entry_ts.strftime("%Y%m%d")) <= hi]
        la = audit_lookahead(sp, [b for b in bars if b[0].date() <= V._d(hi)], vix, trs) if audit else None
        det = {int(t.entry_ts.strftime("%Y%m%d")): _detail(sp, t) for t in trs}
        return sid, {"trades": [V._trade_tuple(sp, t) for t in trs], "lookahead": la, "error": None,
                     "skipped": sum(1 for d in strat.skipped if lo <= int(d.strftime("%Y%m%d")) <= hi)}, det
    except Exception as exc:
        import traceback
        return sid, {"trades": [], "skipped": 0, "lookahead": None,
                     "error": (repr(exc) + " " + traceback.format_exc()[-400:])[:600]}, {}


def _detail(spec: R3Spec, tr: OR.Trade) -> dict:
    m = spec.margin(tr)
    return {"date": int(tr.entry_ts.strftime("%Y%m%d")), "entry": tr.entry_ts.strftime("%H:%M"),
            "expiry": tr.expiry.isoformat(), "kind": tr.kind, "spot_in": round(tr.spot_in, 2),
            "spot_out": round(tr.spot_out, 2), "exit": tr.exit_ts.strftime("%Y%m%d %H:%M") if tr.exit_ts else "",
            "vix": round(tr.sigma * 100, 2), "qty": tr.qty, "lot": tr.qty // spec.lots,
            "credit_pts": round(tr.entry_net, 2), "max_risk": tr.max_risk, "margin": round(m, 2), "gross": tr.gross,
            "haircut": tr.slippage, "fees": tr.costs, "net": round(tr.net, 2),
            "net_per_lot": round(tr.net / spec.lots, 2), "bps": round(1e4 * tr.net / m, 2),
            "gross_bps": round(1e4 * (tr.gross + tr.slippage) / m, 2), "reason": tr.reason,
            "legs": " | ".join(f"{'B' if l.side > 0 else 'S'} {int(l.strike)}{l.opt} {l.entry_px:.2f}->{l.exit_px:.2f}"
                               for l in tr.legs)}


DETAILS: dict = {}               # (id, tag) -> {entry date: detail}


def make_simulator(jobs: int, haircut_mult=1.0, lot_tag="base", time_basis="calendar", tag="base", audit=True):
    def simulate(specs, lo, hi):
        js = [(sp.id, lo, hi, haircut_mult, lot_tag, time_basis, audit) for sp in specs]
        if jobs > 1 and len(js) > 1:
            with mp.get_context("fork").Pool(min(jobs, len(js))) as pool:
                rows = pool.map(_work, js, chunksize=1)
        else:
            rows = [_work(j) for j in js]
        out = {}
        for sid, r, det in rows:
            out[sid] = r
            DETAILS.setdefault((sid, tag), {}).update(det)
        return out
    return simulate


# ══════════════════════════════════════════════════════════════════════════
# CLI / report
# ══════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.options_r3", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="", help="directory with _NIFTY50_min.json, _NIFTYBANK_min.json, _VIX_day.json")
    ap.add_argument("--out", default="", help="output directory (outside the repository)")
    ap.add_argument("--cache", default="", help="lab cache directory of --data (holds final_ledger.json)")
    ap.add_argument("--ids", default="", help="comma-separated ids (default: all r3_options_*)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--screen-only", action="store_true", help="never simulate FINAL")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for s in SPECS:
            print(f"{s.id:<32} {s.family:<18} {s.underlying:<9} {s.desc}")
        return 0
    if not a.data or not a.out:
        ap.error("--data and --out are required")

    out = D.assert_outside_repo(a.out)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    cache = D.assert_outside_repo(a.cache or D.default_cache_dir(a.data))
    os.makedirs(cache, exist_ok=True)
    specs = [BY_ID[x.strip()] for x in a.ids.split(",") if x.strip()] if a.ids else list(SPECS)
    bars, vix = load(a.data)
    for sp in specs:
        if sp.underlying not in bars:
            raise SystemExit(f"{sp.id}: no minute bars for {sp.underlying} in {a.data}")
        sp.vix = vix
    _G["bars"], _G["vix"] = bars, vix
    calendar = V.calendar_of(bars["NIFTY"])
    split = P.split_sessions(calendar)
    DETAILS.clear()

    ledger = os.path.join(cache, "final_ledger.json")
    with open(ledger + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        results, _ = P.evaluate(specs, calendar, make_simulator(a.jobs), n_symbols=1, ledger_path=ledger,
                                dataset=os.path.realpath(a.data), screen_only=a.screen_only, cost_model=COST_MODEL)
    lo, hi = split.discovery
    sens = make_simulator(a.jobs, haircut_mult=SENS_MULT, tag="sens", audit=False)(specs, lo, hi)
    make_simulator(a.jobs, lot_tag="histlot", tag="histlot", audit=False)(specs, lo, hi)
    make_simulator(a.jobs, time_basis="trading", tag="tclock", audit=False)(specs, lo, hi)

    L = [f"OPTIONS R3 (F&O options replacement candidates)  data={os.path.realpath(a.data)}",
         f"sessions={len(calendar)} ({calendar[0]}..{calendar[-1]})  DISCOVERY {lo}..{hi} (H1 to {split.h1[1]}, "
         f"H2 from {split.h2[0]})  FINAL {split.final[0]}..{split.final[1]}",
         f"cost model {COST_MODEL}; haircut max(1 pt NIFTY / 2 pts BANKNIFTY, 2 % of premium) per leg per fill; "
         f"net@5 = haircut x{SENS_MULT:g}; histlot = approximate historical lots; tclock = trading-time clock. "
         f"Sensitivities on DISCOVERY only. Lots: NIFTY 75 / 65 from 2025-12-31, BANKNIFTY 30 (all dates).", ""]
    rows = []
    for r in results:
        sp, fv = r["spec"], r["final"]
        det = DETAILS.get((sp.id, "base"), {})
        blocks = {"H1": V.summarize(V._rng(det, split.h1), sp.lots), "H2": V.summarize(V._rng(det, split.h2), sp.lots),
                  "DISC": V.summarize(V._rng(det, split.discovery), sp.lots)}
        if fv is not None:
            blocks["FINAL"] = V.summarize(V._rng(det, split.final), sp.lots)
        sdet = DETAILS.get((sp.id, "sens"), {})
        s5 = V.summarize(V._rng(sdet, split.discovery), sp.lots)
        s5h1, s5h2 = V.summarize(V._rng(sdet, split.h1), sp.lots), V.summarize(V._rng(sdet, split.h2), sp.lots)
        sc5 = P.screen(sens.get(sp.id, {}).get("trades", []), split, sp.horizon, single_underlying=True)
        sh = V.summarize(V._rng(DETAILS.get((sp.id, "histlot"), {}), split.discovery), sp.lots)
        stc = V.summarize(V._rng(DETAILS.get((sp.id, "tclock"), {}), split.discovery), sp.lots)
        reasons: dict = {}
        for x in V._rng(det, split.discovery):
            reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
        L.append(f"{sp.id}  [{sp.family}, {sp.underlying}]  ({sp.desc}; {sp.lots} lot)")
        L.append(f"  VERDICT {r['verdict']}: {r['why']}" + (f" | {r['final_note']}" if r.get("final_note") else "")
                 + f"   [skipped short weeks in DISCOVERY: {r['skipped']}; DISCOVERY exits {reasons}]")
        for k, s in blocks.items():
            L.append(V._fmt_block(k, s))
        L.append(f"    net@5 DISCOVERY ₹{s5.get('net', 0):,.0f} ({s5.get('bps', float('nan')):.1f} bps/trade, PF "
                 f"{s5.get('pf', 0):.2f}; H1 ₹{s5h1.get('net', 0):,.0f}, H2 ₹{s5h2.get('net', 0):,.0f}) -> screen "
                 f"{'PASS' if sc5['passed'] else 'FAIL: ' + ', '.join(sc5['reasons'])}")
        L.append(f"    histlot DISCOVERY net ₹{sh.get('net', 0):,.0f} ({sh.get('bps', float('nan')):.1f} bps, PF "
                 f"{sh.get('pf', 0):.2f})   tclock DISCOVERY net ₹{stc.get('net', 0):,.0f} "
                 f"({stc.get('bps', float('nan')):.1f} bps, PF {stc.get('pf', 0):.2f})")
        L.append("")
        d, f = blocks["DISC"], blocks.get("FINAL", {})
        rows.append({"id": sp.id, "family": sp.family, "underlying": sp.underlying, "verdict": r["verdict"],
                     "why": r["why"], "final_note": r.get("final_note", ""),
                     "disc_trades": d.get("trades", 0), "disc_net": d.get("net"), "disc_pf": d.get("pf"),
                     "disc_bps": d.get("bps"), "disc_gross_bps": d.get("gross_bps"), "disc_t": d.get("t_bps"),
                     "h1_net": blocks["H1"].get("net"), "h2_net": blocks["H2"].get("net"),
                     "disc_worst": d.get("worst_net"), "disc_max_dd": d.get("max_dd"),
                     "sens5_disc_net": s5.get("net"), "sens5_disc_pf": s5.get("pf"),
                     "sens5_screen_passed": sc5["passed"], "histlot_disc_net": sh.get("net"),
                     "tclock_disc_net": stc.get("net"), "tclock_disc_pf": stc.get("pf"),
                     "final_trades": f.get("trades", ""), "final_net": f.get("net", ""), "final_pf": f.get("pf", ""),
                     "final_bps": f.get("bps", ""), "cost_model": COST_MODEL, "fingerprint": sp.fingerprint})
        with open(os.path.join(out, "trades", sp.id + ".csv"), "w", newline="") as fh:
            keys = list(next(iter(det.values())).keys()) if det else ["date"]
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for k in sorted(det):
                if k <= hi or fv is not None:
                    w.writerow(det[k])
    L.append(P.luck_reminder(results))
    L.append("Model limits: flat VIX vol (no skew; OTM puts and far wings richer, calls cheaper in reality); VIX is a "
             "30-day vol, not the weekly / 0-DTE implied vol; vol constant intraday; credits are model values. "
             "Short-vol families (iron condor/fly, strangles, credit spreads, calendar timing, BANKNIFTY) are correlated.")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text)
    if rows:
        with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
