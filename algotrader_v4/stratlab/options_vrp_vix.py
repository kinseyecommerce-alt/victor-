"""
stratlab/options_vrp_vix.py — the pre-registered VIX-regime filter on the weekly winged NIFTY short strangle,
judged by the lab's protocol.

    cd algotrader_v4
    python -m stratlab.options_vrp_vix --data <bars3y dir> --vix-extra <older VIX day json> --out <scratch dir>

VRP-WINGED-HIGHVIX (pre-registered; do not tune):
  structure  exactly VRP-STRANGLE-016-WINGED (stratlab/options_vrp.py): 10:20 on the first trading day after a
             weekly NIFTY expiry, sell the 0.16-delta CE and PE, buy CE +400 / PE -400 wings, hold to expiry
             settlement, no stop, 2 lots (75, 65 from 2025-12-31), the same cost model, haircut and margin
             (max loss + 2 % of spot notional). Short weeks (< 3 trading days left) are skipped as there.
  filter     past data only. p = the last India VIX session strictly before the entry day.
             TAKE the week only if  VIX[p] > median(VIX closes of the 250 VIX sessions before p)   (strict)
             AND NOT  VIX[p] / VIX[p - 5 sessions] - 1 >= 0.15   (no entry after a VIX spike; a fixed exclusion).
             Fewer than 250 earlier VIX closes -> no trade that week (counted as "no history").
             A week that fails is not entered later: stay flat until the next expiry.
  judgement  protocol.evaluate(): the lab's DISCOVERY / FINAL split of the 3-year sessions, net > 0 in H1 and H2,
             PF >= 1.15, >= 60 trades (positional), the one-symbol gate skipped for a one-instrument strategy,
             FINAL once, for screen passers only, through the per-data-directory final_ledger.json.
  comparison the unconditional VRP-STRANGLE-016-WINGED on the SAME DISCOVERY weeks (never on FINAL here: that is the
             unconditional strategy's own once-only hold-out). The filter adds value only if the conditional
             per-week mean is higher than the unconditional one.
VIX history: _VIX_day.json starts 2022-11-10, 249 sessions before the first 3-year session, so the 250-session
median of the first weeks needs older closes: --vix-extra merges read-only history-bars closes (earlier dates only;
a date present in both keeps _VIX_day.json's value). The option model's vol is the previous VIX close as in
options_vrp, unchanged.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import os
import statistics
import sys
from datetime import date, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import options_replay as OR  # noqa: E402

from . import data as D  # noqa: E402
from . import options_vrp as V  # noqa: E402
from . import protocol as P  # noqa: E402

VIX_LOOKBACK = 250            # sessions in the median, pre-registered
SPIKE_SESSIONS = 5            # pre-registered fixed exclusion: VIX up >= 15 % over the last 5 sessions
SPIKE_PCT = 0.15


def vix_filter(vix: dict, entry_day: date, lookback: int = VIX_LOOKBACK, spike_n: int = SPIKE_SESSIONS,
               spike_pct: float = SPIKE_PCT) -> dict:
    """Decision for a week entered on `entry_day`, from VIX closes dated strictly before it only.
    Returns {"take", "why", "prev", "vix", "median", "chg5"}; why in take | below_median | spike | no_history."""
    days = sorted(d for d in vix if d < entry_day)             # nothing from the entry day or later is read
    out = {"take": False, "why": "no_history", "prev": None, "vix": None, "median": None, "chg5": None}
    if len(days) < lookback + 1:
        return out
    p = days[-1]
    hist = [vix[d] for d in days[-1 - lookback:-1]]           # the `lookback` sessions before p
    med = statistics.median(hist)
    chg = vix[p] / vix[days[-1 - spike_n]] - 1.0
    out.update(prev=p, vix=vix[p], median=med, chg5=chg)
    if not vix[p] > med:
        out["why"] = "below_median"
    elif chg >= spike_pct - 1e-9:                             # 11.50 / 10.00 - 1 is 0.1499999... in floats
        out["why"] = "spike"
    else:
        out["take"], out["why"] = True, "take"
    return out


class VixRegimeStrangle(OR.ShortStrangle):
    """ShortStrangle on the weekly cycle that trades a week only when vix_filter() takes it."""
    name = "SHORT_STRANGLE_HIGHVIX"

    def __init__(self, vix: dict, lookback: int = VIX_LOOKBACK, spike_n: int = SPIKE_SESSIONS,
                 spike_pct: float = SPIKE_PCT, **kw):
        kw.setdefault("weekly_cycle", True)
        super().__init__(**kw)
        self.vix, self.lookback, self.spike_n, self.spike_pct = vix, lookback, spike_n, spike_pct
        self.filter_log: list = []                                # (entry day, decision) of every cycle week

    def decide(self, v: OR.MarketView):
        if not self.in_cycle(v):
            return None
        f = vix_filter(self.vix, v.now.date(), self.lookback, self.spike_n, self.spike_pct)
        self.filter_log.append((v.now.date(), f))
        if not f["take"]:
            return None
        kc = v.strike_for_delta("CE", self.short_delta)
        kp = v.strike_for_delta("PE", self.short_delta)
        legs = [OR.LegSpec("CE", kc, -1), OR.LegSpec("PE", kp, -1)]
        if self.wing_pts:
            legs += [OR.LegSpec("CE", kc + self.wing_pts, +1), OR.LegSpec("PE", kp - self.wing_pts, +1)]
        return legs


class VixOptSpec(V.OptSpec):
    """An OptSpec whose strategy reads the VIX closes given at construction (set by load())."""
    vix: dict | None = None

    def strategy(self) -> VixRegimeStrangle:
        if self.vix is None:
            raise RuntimeError("VixOptSpec.vix not set")
        p = self.params
        return VixRegimeStrangle(self.vix, lookback=p["vix_lookback"], spike_n=p["spike_sessions"],
                                 spike_pct=p["spike_pct"], entry=time(10, 20), hold="expiry",
                                 short_delta=p["short_delta"], target_frac=None, stop_mult=None,
                                 wing_pts=p.get("wing_pts"), min_days_left=p["min_days_left"])

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(x) for x in (vix_filter, VixRegimeStrangle, VixOptSpec, OR.ShortStrangle,
                                                     OR.Replay, OR.leg_cost, OR.fill_price, OR.next_expiry,
                                                     V.OptSpec, V._run_replay, V._trade_tuple))
        blob = repr((self.id, self.horizon, self.underlying, self.lots, sorted(self.params.items()), src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


HIGHVIX = VixOptSpec(
    "VRP-WINGED-HIGHVIX",
    "VRP-STRANGLE-016-WINGED only in weeks with VIX > its 250-session median and no >= 15 % 5-session VIX spike",
    {"short_delta": 0.16, "min_days_left": 3, "wing_pts": 400.0, "vix_lookback": VIX_LOOKBACK,
     "spike_sessions": SPIKE_SESSIONS, "spike_pct": SPIKE_PCT},
    lots=2, family="vrp_vix_regime", extra={"expected_bps_per_trade": 60, "expected_trades": 25})
SPECS = [HIGHVIX]
BASELINE = V.BY_ID["VRP-STRANGLE-016-WINGED"]


def merge_vix(base: dict, extra_paths) -> dict:
    """VIX closes of _VIX_day.json plus older closes from extra files (a date in both keeps base's value)."""
    out = dict(base)
    for pth in extra_paths:
        with open(pth) as fh:
            for d, c in OR._daily_closes(OR._parse_bars(json.load(fh)["bars"])).items():
                out.setdefault(d, c)
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _week_compare(base_det: dict, cond_det: dict, decisions: dict, rng) -> dict:
    """Per-week bps of margin: unconditional (all weeks), conditional (taken weeks), the complement, and a
    check that a taken week's trade is identical in both runs."""
    b = {k: v for k, v in base_det.items() if rng[0] <= k <= rng[1]}
    c = {k: v for k, v in cond_det.items() if rng[0] <= k <= rng[1]}
    taken = [k for k in b if decisions.get(k, {}).get("take")]
    mismatch = [k for k in c if k not in b or abs(c[k]["net"] - b[k]["net"]) > 0.01]
    missing = [k for k in taken if k not in c]
    why = {}
    for k in b:
        w = decisions.get(k, {}).get("why", "unknown")
        why[w] = why.get(w, 0) + 1
    return {"weeks": len(b), "taken": len(c), "why": why,
            "uncond_bps": _mean([v["bps"] for v in b.values()]), "uncond_net_week": _mean([v["net"] for v in b.values()]),
            "cond_bps": _mean([v["bps"] for v in c.values()]), "cond_net_week": _mean([v["net"] for v in c.values()]),
            "rest_bps": _mean([v["bps"] for k, v in b.items() if k not in c]),
            "rest_net_week": _mean([v["net"] for k, v in b.items() if k not in c]),
            "spike_bps": _mean([v["bps"] for k, v in b.items() if decisions.get(k, {}).get("why") == "spike"]),
            "below_bps": _mean([v["bps"] for k, v in b.items() if decisions.get(k, {}).get("why") == "below_median"]),
            "mismatch": mismatch, "missing": missing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.options_vrp_vix", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="directory with _NIFTY50_min.json and _VIX_day.json")
    ap.add_argument("--out", required=True, help="output directory (outside the repository)")
    ap.add_argument("--vix-extra", default="", help="comma-separated older VIX day files (history-bars layout)")
    ap.add_argument("--cache", default="", help="lab cache directory of --data (holds final_ledger.json)")
    ap.add_argument("--screen-only", action="store_true", help="never simulate FINAL")
    a = ap.parse_args(argv)

    out = D.assert_outside_repo(a.out)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    cache = D.assert_outside_repo(a.cache or D.default_cache_dir(a.data))
    os.makedirs(cache, exist_ok=True)
    bars, vix0 = V.load(a.data)
    vix = merge_vix(vix0, [x for x in a.vix_extra.split(",") if x])
    HIGHVIX.vix = vix
    calendar = V.calendar_of(bars)
    split = P.split_sessions(calendar)
    V.DETAILS.clear()

    ledger = os.path.join(cache, "final_ledger.json")
    with open(ledger + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        results, _ = P.evaluate(SPECS, calendar, V.make_simulator(bars, vix), n_symbols=1, ledger_path=ledger,
                                dataset=os.path.realpath(a.data), screen_only=a.screen_only, cost_model=V.COST_MODEL)
    lo, hi = split.discovery
    # DISCOVERY only from here on: the 5-bps-equivalent sensitivity and the unconditional baseline
    sens = V.make_simulator(bars, vix, haircut_mult=V.SENS_MULT, tag="sens", audit=False)(SPECS + [BASELINE], lo, hi)
    V.make_simulator(bars, vix, audit=False)([BASELINE], lo, hi)
    disc_days = sorted({int(b[0].strftime("%Y%m%d")) for b in bars if lo <= int(b[0].strftime("%Y%m%d")) <= hi})
    decisions = {int(d.strftime("%Y%m%d")): vix_filter(vix, d) for d in (V._d(x) for x in disc_days)}

    r = results[0]
    sp, fv = r["spec"], r["final"]
    det = V.DETAILS.get((sp.id, 1.0, "base"), {})
    bdet = V.DETAILS.get((BASELINE.id, 1.0, "base"), {})
    sdet = V.DETAILS.get((sp.id, V.SENS_MULT, "sens"), {})
    bsdet = V.DETAILS.get((BASELINE.id, V.SENS_MULT, "sens"), {})
    blocks = {"H1": V.summarize(V._rng(det, split.h1), sp.lots), "H2": V.summarize(V._rng(det, split.h2), sp.lots),
              "DISC": V.summarize(V._rng(det, split.discovery), sp.lots)}
    if fv is not None:
        blocks["FINAL"] = V.summarize(V._rng(det, split.final), sp.lots)
    s5 = {k: V.summarize(V._rng(sdet, rng), sp.lots) for k, rng in (("H1", split.h1), ("H2", split.h2),
                                                                    ("DISC", split.discovery))}
    sc5 = P.screen(sens.get(sp.id, {}).get("trades", []), split, sp.horizon, single_underlying=True)
    cmp = {k: _week_compare(bdet, det, decisions, rng) for k, rng in (("H1", split.h1), ("H2", split.h2),
                                                                      ("DISC", split.discovery))}
    cmp5 = _week_compare(bsdet, sdet, decisions, split.discovery)

    L = [f"OPTIONS VRP x VIX REGIME  data={os.path.realpath(a.data)}",
         f"sessions={len(calendar)} ({calendar[0]}..{calendar[-1]})  DISCOVERY {lo}..{hi} (H1 to {split.h1[1]}, "
         f"H2 from {split.h2[0]})  FINAL {split.final[0]}..{split.final[1]}",
         f"VIX closes {min(vix)}..{max(vix)} ({len(vix)}; {len(vix) - len(vix0)} older from --vix-extra); cost model "
         f"{V.COST_MODEL}; haircut max(1 pt, 2 %) per leg per fill; net@5 = haircut x{V.SENS_MULT:g}, DISCOVERY only.", "",
         f"{sp.id}  ({sp.desc}; {sp.lots} lots)",
         f"  VERDICT {r['verdict']}: {r['why']}" + (f" | {r['final_note']}" if r.get("final_note") else "")]
    for k, s in blocks.items():
        L.append(V._fmt_block(k, s))
    L.append(f"    net@5 DISCOVERY ₹{s5['DISC'].get('net', 0):,.0f} ({s5['DISC'].get('bps', float('nan')):.1f} bps/trade, "
             f"PF {s5['DISC'].get('pf', 0):.2f}; H1 ₹{s5['H1'].get('net', 0):,.0f}, H2 ₹{s5['H2'].get('net', 0):,.0f}) -> "
             f"screen {'PASS' if sc5['passed'] else 'FAIL: ' + ', '.join(sc5['reasons'])}")
    L.append("")
    L.append("  Filter vs unconditional VRP-STRANGLE-016-WINGED (same weeks, bps of margin per week, DISCOVERY only):")
    for k, c in cmp.items():
        L.append(f"    {k:<4} cycle weeks {c['weeks']:>3}  taken {c['taken']:>3} {c['why']}  unconditional "
                 f"{c['uncond_bps']:>7.1f} bps (₹{c['uncond_net_week']:>7,.0f}/wk)  conditional {c['cond_bps']:>7.1f} bps "
                 f"(₹{c['cond_net_week']:>7,.0f}/wk)  skipped weeks {c['rest_bps']:>7.1f} bps  [below median "
                 f"{c['below_bps']:.1f}, spike {c['spike_bps']:.1f}]  -> filter {'ADDS' if c['cond_bps'] > c['uncond_bps'] else 'does NOT add'} value"
                 + (f"  MISMATCH {c['mismatch'][:5]} MISSING {c['missing'][:5]}" if c["mismatch"] or c["missing"] else ""))
    L.append(f"    net@5 DISC: unconditional {cmp5['uncond_bps']:.1f} bps/wk, conditional {cmp5['cond_bps']:.1f} bps/wk, "
             f"skipped {cmp5['rest_bps']:.1f} bps/wk")
    L.append("")
    L.append(P.luck_reminder(results))
    L.append("Model limits: flat VIX vol (no skew), VIX is a 30-day vol; the credit is not a market quote. "
             "The unconditional baseline is shown on DISCOVERY only; its FINAL belongs to its own ledger entry.")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text)
    d, f = blocks["DISC"], blocks.get("FINAL", {})
    row = {"id": sp.id, "verdict": r["verdict"], "why": r["why"], "final_note": r.get("final_note", ""),
           "disc_trades": d.get("trades", 0), "disc_net": d.get("net"), "disc_pf": d.get("pf"), "disc_bps": d.get("bps"),
           "disc_gross_bps": d.get("gross_bps"), "h1_net": blocks["H1"].get("net"), "h2_net": blocks["H2"].get("net"),
           "sens_disc_net": s5["DISC"].get("net"), "sens_disc_pf": s5["DISC"].get("pf"), "sens_screen_passed": sc5["passed"],
           "uncond_disc_bps_per_week": cmp["DISC"]["uncond_bps"], "cond_disc_bps_per_week": cmp["DISC"]["cond_bps"],
           "final_trades": f.get("trades", ""), "final_net": f.get("net", ""), "final_pf": f.get("pf", ""),
           "cost_model": V.COST_MODEL, "fingerprint": sp.fingerprint}
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    with open(os.path.join(out, "trades", sp.id + ".csv"), "w", newline="") as fh:
        keys = list(next(iter(det.values())).keys()) if det else ["date"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for k in sorted(det):
            if k <= hi or fv is not None:
                w.writerow(det[k])
    with open(os.path.join(out, "filter_weeks_discovery.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entry_date", "take", "why", "prev_vix_day", "vix", "median250", "chg5", "uncond_net", "uncond_bps"])
        for k in sorted(bdet):
            if lo <= k <= hi:
                dd = decisions[k]
                w.writerow([k, dd["take"], dd["why"], dd["prev"], dd["vix"], round(dd["median"] or 0, 3),
                            round(dd["chg5"] or 0, 4), bdet[k]["net"], bdet[k]["bps"]])
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
