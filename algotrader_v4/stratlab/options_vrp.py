"""
stratlab/options_vrp.py — pre-registered weekly NIFTY short strangles (volatility risk premium), judged by the
lab's protocol.

The equity simulator (sim.py) cannot hold option legs, so this module runs the strategies through
options_replay.py (ShortStrangle: Black-Scholes on the real NIFTY 50 minute bars, India VIX as a flat vol) and
hands the trades to the SAME judgement as every other lab strategy: protocol.evaluate() with the lab's
DISCOVERY / FINAL split of the sessions, the H1 / H2 screen (net > 0 in both, PF >= 1.15, >= 60 positional
trades; the 35 % one-symbol gate does not apply to a one-instrument strategy, as for index futures) and FINAL
evaluated once, for screen passers only, through the same per-data-directory ledger (final_ledger.json in the
lab cache of the data directory). There is no --peek-final here.

    cd algotrader_v4
    python -m stratlab.options_vrp --data <bars3y dir> --out <scratch dir> [--screen-only]

Pre-registered rules (do not tune them on results):
  entry     10:20 on the first trading day after each weekly NIFTY expiry, in the next weekly expiry; the week is
            skipped if fewer than 3 trading days remain after the entry day (up to and including the expiry).
            Expiry calendar: options_replay.next_expiry (Thursday, Tuesday from 2025-09-01, a holiday moves it to
            the previous trading day; trading days = sessions present in the data).
  strikes   Black-Scholes delta with India VIX as one flat vol (the previous session's close: the data has no
            intraday VIX), spot = NIFTY at 10:20 (the open of the 10:20 bar, i.e. the 10:19 close; the decision
            sees only bars stamped before 10:20), r = 6.5 %; sell the CE closest to +0.16 and the PE closest to
            -0.16 on the 50-point grid. WINGED also buys CE short+400 and PE short-400.
  exit      hold to expiry and settle at intrinsic against NIFTY's expiry close (options_replay: the mean of the
            15:00-15:29 closes, NSE's last-30-minute settlement rule); no stop, no target, no adjustment.
  size      NAKED 1 lot, WINGED 2 lots; lot size in force on the entry date as pre-registered: 75, then 65 from
            2025-12-31 (the January-2026 expiries, NSE circular FAOP70616).
  costs     costs.round_trip_cost(..., "option") per leg (COST_MODEL_VERSION 2026-11: ₹20 per order, STT 0.15 % of
            premium on sells, exchange 0.03553 %, SEBI, stamp 0.003 % on buys, GST 18 %) and a fill haircut of
            max(1 point, 2 % of model premium) per leg per fill. A settlement is not an order: no haircut; it is
            still charged as a round trip leg by options_replay.leg_cost (₹20 + charges on ITM settlement value,
            a small over-charge). The pre-registration said exchange 0.03503 %; the lab's fixed model charges
            0.03553 % (Zerodha 2026), 0.0005 % of premium more.
  margin    for bps: NAKED 12 % of spot notional (SPAN + ELM); WINGED max loss + 2 % of spot notional.
Slippage sensitivity: "net@5" for options is the haircut scaled by 2.5 (max(2.5 points, 5 % of premium)), the
same 5 / 2 ratio as the stock lab's 5 bps column; DISCOVERY dates only.

Known model limits: one flat vol (no skew: real OTM puts are richer and calls cheaper, far wings richer still),
VIX is a 30-day vol, not the weekly implied vol; the credit is therefore unverified. Nothing here places orders.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import costs as _costs  # noqa: E402
import options_replay as OR  # noqa: E402

from . import data as D  # noqa: E402
from . import protocol as P  # noqa: E402

OPT_FILL_MODEL = "o1"           # bump when the option fill / settlement rules change (ledger key)
COST_MODEL = f"{_costs.COST_MODEL_VERSION}/{OPT_FILL_MODEL}"
CAPITAL = 1_000_000.0
LOT_65_FROM = date(2025, 12, 31)
LOT_SCHEDULE = ((date(2000, 1, 1), 75), (LOT_65_FROM, 65))                    # pre-registered
# Sensitivity only (approximate exchange history): 50, 25 from 2024-04-26, 75 from 2024-11-20, 65 from 2025-12-31.
LOT_SCHEDULE_HISTORICAL = ((date(2000, 1, 1), 50), (date(2024, 4, 26), 25), (date(2024, 11, 20), 75),
                           (LOT_65_FROM, 65))
SENS_MULT = 2.5                 # haircut multiple for the slippage sensitivity column
NAKED_MARGIN_PCT = 0.12
ELM_PCT = 0.02


@dataclass
class OptSpec:
    id: str
    desc: str
    params: dict
    kind: str = "options"
    family: str = "vrp_short_strangle"
    horizon: str = "positional"
    underlying: str = "NIFTY"
    lots: int = 1
    needs_index: bool = False
    extra: dict = field(default_factory=dict)

    def strategy(self) -> OR.ShortStrangle:
        p = self.params
        return OR.ShortStrangle(entry=time(10, 20), hold="expiry", short_delta=p["short_delta"],
                                target_frac=None, stop_mult=None, wing_pts=p.get("wing_pts"),
                                weekly_cycle=True, min_days_left=p["min_days_left"])

    def margin(self, tr: OR.Trade) -> float:
        notional = tr.spot_in * tr.qty
        if self.params.get("wing_pts"):
            return tr.max_risk + ELM_PCT * notional
        return NAKED_MARGIN_PCT * notional

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(x) for x in (OR.ShortStrangle, OR.Replay, OR.leg_cost, OR.fill_price,
                                                     OR.next_expiry, OptSpec, _run_replay, _trade_tuple))
        blob = repr((self.id, self.horizon, self.underlying, self.lots, sorted(self.params.items()), src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


SPECS = [
    OptSpec("VRP-STRANGLE-016-NAKED",
            "Weekly NIFTY short strangle: sell the 0.16-delta CE and PE at 10:20 the day after expiry, hold to expiry",
            {"short_delta": 0.16, "min_days_left": 3, "wing_pts": None}, lots=1,
            extra={"expected_bps_per_trade": 50, "expected_trades": 47}),
    OptSpec("VRP-STRANGLE-016-WINGED",
            "As NAKED plus long CE +400 / PE -400 wings (defined risk), 2 lots",
            {"short_delta": 0.16, "min_days_left": 3, "wing_pts": 400.0}, lots=2,
            extra={"expected_bps_per_trade": 30, "expected_trades": 47}),
]
BY_ID = {s.id: s for s in SPECS}


# ══════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════

def load(data_dir: str) -> tuple[list, dict]:
    """NIFTY 50 minute bars without short sessions (< MIN_SESSION_BARS, e.g. Muhurat) and India VIX closes."""
    with open(os.path.join(data_dir, "_NIFTY50_min.json")) as fh:
        bars = OR._parse_bars(json.load(fh)["bars"])
    n: dict = {}
    for b in bars:
        n[b[0].date()] = n.get(b[0].date(), 0) + 1
    bars = [b for b in bars if n[b[0].date()] >= D.MIN_SESSION_BARS]
    with open(os.path.join(data_dir, "_VIX_day.json")) as fh:
        vix = OR._daily_closes(OR._parse_bars(json.load(fh)["bars"]))
    return bars, vix


def calendar_of(bars) -> list[int]:
    return sorted({int(b[0].strftime("%Y%m%d")) for b in bars})


def _d(x: int) -> date:
    x = int(x)
    return date(x // 10000, x // 100 % 100, x % 100)


# ══════════════════════════════════════════════════════════════════════════
# simulation
# ══════════════════════════════════════════════════════════════════════════

def _run_replay(spec: OptSpec, bars, vix, hi: int, haircut_mult: float = 1.0, lot_schedule=LOT_SCHEDULE):
    """Replay with bars up to session `hi` only (a DISCOVERY run never reads a FINAL bar)."""
    hi_d = _d(hi)
    use = [b for b in bars if b[0].date() <= hi_d]
    cfg = OR.ReplayConfig(underlying=spec.underlying, lots=spec.lots, lot_schedule=lot_schedule,
                          slip_min=OR.SLIP_MIN_PTS[spec.underlying] * haircut_mult,
                          slip_pct=OR.SLIP_PCT * haircut_mult)
    strat = spec.strategy()
    res = OR.Replay(use, strat, cfg, vix).run()
    return res, strat


def _trade_tuple(spec: OptSpec, tr: OR.Trade) -> tuple:
    exit_val = sum(l.exit_px * (-l.side) for l in tr.legs)          # per unit paid to close (+ = we pay)
    return (spec.id, spec.underlying, int(tr.entry_ts.strftime("%Y%m%d")), -1, tr.entry_ts.strftime("%H:%M"),
            round(tr.entry_net, 4), int(tr.exit_ts.strftime("%Y%m%d")) if tr.exit_ts else 0,
            tr.exit_ts.strftime("%H:%M") if tr.exit_ts else "", round(exit_val, 4), tr.qty,
            round(tr.gross, 2), round(tr.costs, 2), round(tr.net, 2), tr.reason)


def _detail(spec: OptSpec, tr: OR.Trade) -> dict:
    m = spec.margin(tr)
    return {"date": int(tr.entry_ts.strftime("%Y%m%d")), "expiry": tr.expiry.isoformat(), "spot_in": round(tr.spot_in, 2),
            "spot_out": round(tr.spot_out, 2), "vix": round(tr.sigma * 100, 2), "qty": tr.qty,
            "lot": tr.qty // spec.lots, "credit_pts": round(tr.entry_net, 2), "max_risk": tr.max_risk,
            "margin": round(m, 2), "gross": tr.gross, "haircut": tr.slippage, "fees": tr.costs, "net": round(tr.net, 2),
            "net_per_lot": round(tr.net / spec.lots, 2), "bps": round(1e4 * tr.net / m, 2),
            "gross_bps": round(1e4 * (tr.gross + tr.slippage) / m, 2), "reason": tr.reason,
            "legs": " | ".join(f"{'B' if l.side > 0 else 'S'} {int(l.strike)}{l.opt} {l.entry_px:.2f}->{l.exit_px:.2f}"
                               for l in tr.legs)}


def audit_lookahead(spec: OptSpec, bars, vix, trades: list, n: int = 4) -> str | None:
    """Re-run sampled entries on bars that END at the entry bar: the legs and fills must not change."""
    if not trades:
        return None
    step = max(1, len(trades) // n)
    for tr in trades[::step][:n]:
        cut = [b for b in bars if b[0] <= tr.entry_ts]
        r, _ = OR.Replay(cut, spec.strategy(),
                         OR.ReplayConfig(underlying=spec.underlying, lots=spec.lots, lot_schedule=LOT_SCHEDULE),
                         vix).run(), None
        same = [t for t in r.trades if t.entry_ts == tr.entry_ts]
        if not same:
            return f"entry {tr.entry_ts} disappears when later bars are removed"
        a = [(l.opt, l.strike, l.side, round(l.entry_px, 6)) for l in same[0].legs]
        b = [(l.opt, l.strike, l.side, round(l.entry_px, 6)) for l in tr.legs]
        if a != b:
            return f"entry {tr.entry_ts} changes when later bars are removed: {b} vs {a}"
    return None


DETAILS: dict = {}               # (spec id, haircut_mult, lot schedule tag) -> {entry date: detail}


def make_simulator(bars, vix, haircut_mult: float = 1.0, lot_schedule=LOT_SCHEDULE, tag: str = "base",
                   audit: bool = True):
    def simulate(specs, lo, hi):
        out = {}
        for sp in specs:
            try:
                res, strat = _run_replay(sp, bars, vix, hi, haircut_mult, lot_schedule)
                trs = [t for t in res.trades if lo <= int(t.entry_ts.strftime("%Y%m%d")) <= hi]
                la = audit_lookahead(sp, [b for b in bars if b[0].date() <= _d(hi)], vix, trs) if audit else None
                det = DETAILS.setdefault((sp.id, haircut_mult, tag), {})
                for t in trs:
                    det[int(t.entry_ts.strftime("%Y%m%d"))] = _detail(sp, t)
                out[sp.id] = {"trades": [_trade_tuple(sp, t) for t in trs], "lookahead": la, "error": None,
                              "skipped": sum(1 for d in strat.skipped_short_weeks if lo <= int(d.strftime("%Y%m%d")) <= hi)}
            except Exception as exc:                      # reported as INVALID by the protocol
                out[sp.id] = {"trades": [], "skipped": 0, "lookahead": None, "error": repr(exc)[:300]}
        return out
    return simulate


# ══════════════════════════════════════════════════════════════════════════
# reporting
# ══════════════════════════════════════════════════════════════════════════

def summarize(det: list, lots: int) -> dict:
    """Per-trade statistics in ₹ (for the spec's lots), ₹ per lot, and bps of margin."""
    if not det:
        return {"trades": 0}
    det = sorted(det, key=lambda x: x["date"])
    net = [x["net"] for x in det]
    bps = [x["bps"] for x in det]
    gw, gl = sum(v for v in net if v > 0), -sum(v for v in net if v <= 0)
    eq = pk = dd = 0.0
    for v in net:
        eq += v
        pk = max(pk, eq)
        dd = max(dd, pk - eq)
    mb = sum(bps) / len(bps)
    sd = math.sqrt(sum((b - mb) ** 2 for b in bps) / (len(bps) - 1)) if len(bps) > 2 else float("nan")
    worst = min(det, key=lambda x: x["net"])
    return {"trades": len(det), "net": round(sum(net), 2), "net_per_trade": round(sum(net) / len(net), 2),
            "net_per_lot_per_trade": round(sum(net) / len(net) / lots, 2), "pf": round(gw / gl, 3) if gl else float("inf"),
            "win_pct": round(100.0 * sum(1 for v in net if v > 0) / len(net), 1),
            "bps": round(mb, 2), "t_bps": round(mb / (sd / math.sqrt(len(bps))), 2) if sd == sd and sd > 0 else float("nan"),
            "gross_bps": round(sum(x["gross_bps"] for x in det) / len(det), 2),
            "fees_bps": round(sum(1e4 * x["fees"] / x["margin"] for x in det) / len(det), 2),
            "haircut_bps": round(sum(1e4 * x["haircut"] / x["margin"] for x in det) / len(det), 2),
            "credit_pts": round(sum(x["credit_pts"] for x in det) / len(det), 2),
            "margin": round(sum(x["margin"] for x in det) / len(det), 0),
            "worst_net": worst["net"], "worst_bps": worst["bps"], "worst_date": worst["date"],
            "worst_pct_capital": round(100 * worst["net"] / CAPITAL, 2),
            "max_dd": round(dd, 2), "max_dd_pct_capital": round(100 * dd / CAPITAL, 2)}


def _rng(det: dict, rng) -> list:
    return [v for k, v in det.items() if rng[0] <= k <= rng[1]]


def _fmt_block(name: str, s: dict) -> str:
    if not s.get("trades"):
        return f"    {name:<5} no trades"
    return (f"    {name:<5} n={s['trades']:>3}  net ₹{s['net']:>10,.0f}  per trade ₹{s['net_per_trade']:>7,.0f} "
            f"(per lot ₹{s['net_per_lot_per_trade']:>6,.0f})  {s['bps']:>6.1f} bps of margin (t {s['t_bps']:>5.2f})  "
            f"gross {s['gross_bps']:>6.1f}  fees {s['fees_bps']:>5.1f}  haircut {s['haircut_bps']:>5.1f} bps  "
            f"credit {s['credit_pts']:>6.1f} pts  margin ₹{s['margin']:>8,.0f}  win {s['win_pct']:>5.1f}%  PF {s['pf']:>5.2f}  "
            f"worst ₹{s['worst_net']:>9,.0f} ({s['worst_bps']:>6.0f} bps, {s['worst_pct_capital']:>5.1f}% of capital, "
            f"{s['worst_date']})  maxDD ₹{s['max_dd']:>9,.0f} ({s['max_dd_pct_capital']:.1f}% of capital)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.options_vrp", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="directory with _NIFTY50_min.json and _VIX_day.json")
    ap.add_argument("--out", required=True, help="output directory (outside the repository)")
    ap.add_argument("--cache", default="", help="lab cache directory of --data (holds final_ledger.json)")
    ap.add_argument("--ids", default="", help="comma-separated ids (default: both)")
    ap.add_argument("--screen-only", action="store_true", help="never simulate FINAL")
    a = ap.parse_args(argv)

    out = D.assert_outside_repo(a.out)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    cache = D.assert_outside_repo(a.cache or D.default_cache_dir(a.data))
    os.makedirs(cache, exist_ok=True)
    specs = [BY_ID[x.strip()] for x in a.ids.split(",") if x.strip()] if a.ids else list(SPECS)
    bars, vix = load(a.data)
    calendar = calendar_of(bars)
    split = P.split_sessions(calendar)
    DETAILS.clear()

    ledger = os.path.join(cache, "final_ledger.json")
    with open(ledger + ".lock", "w") as lk:            # the ledger is read-modify-written: one writer at a time
        fcntl.flock(lk, fcntl.LOCK_EX)
        results, trades_by_id = P.evaluate(specs, calendar, make_simulator(bars, vix), n_symbols=1,
                                           ledger_path=ledger, dataset=os.path.realpath(a.data),
                                           screen_only=a.screen_only, cost_model=COST_MODEL)
    # sensitivities: DISCOVERY dates only (never FINAL)
    lo, hi = split.discovery
    sens = make_simulator(bars, vix, haircut_mult=SENS_MULT, tag="sens", audit=False)(specs, lo, hi)
    hist = make_simulator(bars, vix, lot_schedule=LOT_SCHEDULE_HISTORICAL, tag="histlot", audit=False)(specs, lo, hi)

    L = [f"OPTIONS VRP (weekly NIFTY short strangle)  data={os.path.realpath(a.data)}",
         f"sessions={len(calendar)} ({calendar[0]}..{calendar[-1]})  DISCOVERY {split.discovery[0]}..{split.discovery[1]} "
         f"(H1 to {split.h1[1]}, H2 from {split.h2[0]})  FINAL {split.final[0]}..{split.final[1]}",
         f"cost model {COST_MODEL}; haircut max(1 pt, 2 % of premium) per leg per fill; 'net@5' = haircut x{SENS_MULT:g} "
         f"(max(2.5 pts, 5 %)), DISCOVERY only; capital ₹{CAPITAL:,.0f}; lots: 75, 65 from {LOT_65_FROM}.", ""]
    rows = []
    for r in results:
        sp, sc, fv = r["spec"], r["screen"], r["final"]
        det = DETAILS.get((sp.id, 1.0, "base"), {})
        sdet = DETAILS.get((sp.id, SENS_MULT, "sens"), {})
        hdet = DETAILS.get((sp.id, 1.0, "histlot"), {})
        blocks = {"H1": summarize(_rng(det, split.h1), sp.lots), "H2": summarize(_rng(det, split.h2), sp.lots),
                  "DISC": summarize(_rng(det, split.discovery), sp.lots)}
        if fv is not None:
            blocks["FINAL"] = summarize(_rng(det, split.final), sp.lots)
        s5 = summarize(_rng(sdet, split.discovery), sp.lots)
        sc5 = P.screen(sens.get(sp.id, {}).get("trades", []), split, sp.horizon, single_underlying=True)
        s5h1, s5h2 = summarize(_rng(sdet, split.h1), sp.lots), summarize(_rng(sdet, split.h2), sp.lots)
        sh = summarize(_rng(hdet, split.discovery), sp.lots)
        L.append(f"{sp.id}  ({sp.desc}; {sp.lots} lot{'s' if sp.lots > 1 else ''})")
        L.append(f"  VERDICT {r['verdict']}: {r['why']}" + (f" | {r['final_note']}" if r.get("final_note") else "")
                 + f"   [skipped short weeks in DISCOVERY: {r['skipped']}]")
        for k, s in blocks.items():
            L.append(_fmt_block(k, s))
        L.append(f"    net@5 DISCOVERY ₹{s5.get('net', 0):,.0f} ({s5.get('bps', float('nan')):.1f} bps/trade, PF {s5.get('pf', 0):.2f}; "
                 f"H1 ₹{s5h1.get('net', 0):,.0f}, H2 ₹{s5h2.get('net', 0):,.0f}) -> screen {'PASS' if sc5['passed'] else 'FAIL: ' + ', '.join(sc5['reasons'])}")
        L.append(f"    approximate historical lots (50 / 25 / 75 / 65) DISCOVERY: net ₹{sh.get('net', 0):,.0f}, "
                 f"{sh.get('bps', float('nan')):.1f} bps/trade, PF {sh.get('pf', 0):.2f}")
        L.append("")
        d, f = blocks["DISC"], blocks.get("FINAL", {})
        rows.append({"id": sp.id, "verdict": r["verdict"], "why": r["why"], "final_note": r.get("final_note", ""),
                     "disc_trades": d.get("trades", 0), "disc_net": d.get("net"), "disc_pf": d.get("pf"),
                     "disc_bps": d.get("bps"), "disc_gross_bps": d.get("gross_bps"), "disc_t": d.get("t_bps"),
                     "h1_net": blocks["H1"].get("net"), "h2_net": blocks["H2"].get("net"),
                     "h1_bps": blocks["H1"].get("bps"), "h2_bps": blocks["H2"].get("bps"),
                     "disc_net_per_lot_per_trade": d.get("net_per_lot_per_trade"),
                     "disc_worst": d.get("worst_net"), "disc_worst_bps": d.get("worst_bps"), "disc_max_dd": d.get("max_dd"),
                     "sens_disc_net": s5.get("net"), "sens_disc_pf": s5.get("pf"), "sens_screen_passed": sc5["passed"],
                     "histlot_disc_net": sh.get("net"),
                     "final_trades": f.get("trades", ""), "final_net": f.get("net", ""), "final_pf": f.get("pf", ""),
                     "final_bps": f.get("bps", ""), "final_worst": f.get("worst_net", ""), "final_max_dd": f.get("max_dd", ""),
                     "cost_model": COST_MODEL})
        with open(os.path.join(out, "trades", sp.id + ".csv"), "w", newline="") as fh:
            keys = list(next(iter(det.values())).keys()) if det else ["date"]
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for k in sorted(det):
                if k <= split.discovery[1] or fv is not None:
                    w.writerow(det[k])
    L.append(P.luck_reminder(results))
    L.append("Model limits: flat VIX vol (no skew; far-OTM wings and puts are richer in reality), VIX is a 30-day vol, "
             "the credit is not a market quote. Rupee figures use the pre-registered lots (75 / 65) for every date.")
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
