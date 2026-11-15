"""
stratlab/r7_tps.py — round 7: Connors TPS (Time, Price, Scale-in; "High Probability ETF Trading", TuringTrader port
BooksAndPubs/Connors_HighProbEtfTrading.cs) on NIFTY / BANKNIFTY index futures. The per-symbol simulator holds one
fixed-size position, so the scale-in (10 / 20 / 30 / 40 % tranches) has its own trade simulator here; the trades it
produces go through the UNCHANGED protocol.evaluate (2023-26, DISCOVERY / FINAL, FINAL once) and protocol_oos.evaluate
(2012-23, one look). Pre-registration: stratlab/prereg/github_v2.json.

Rules (r7_ind.tps_signals; Wilder RSI(2) - the port's RSI(2) is an EMA-smoothed RSI, fixed to Connors' Wilder RSI):
  long  flat, close > SMA(200), RSI(2) < 25 today and yesterday -> buy 10 %; every later close below the last entry
        price buys 20 %, 30 %, 40 % (at most 4 tranches); exit everything when RSI(2) > 70
  short mirror: close < SMA(200), RSI(2) > 75 two days -> sell 10 % ...; adds on closes above the last entry; exit
        RSI(2) < 30
  fills: every action decided at a close is filled at the NEXT session's open (the port and the book trade at the
        close; the lab's next-open convention is used in both eras), 1 bps slippage (2.5 bps sensitivity)
  size: 100 % = 1 lot (2026 lot sizes, fractional tranches); futures costs per tranche (costs.round_trip_cost)
  one TRADE = one scale-in sequence (first fill to exit): qty = total units, entry = their average fill price, gross
        / cost = the sums over the tranches. A sequence is counted in the period of its first fill and closed at the
        last close of the period if still open (period_end).

    python -m stratlab.r7_tps --era 2023-26 --prereg <PREREGISTERED_GITHUB2.json> --out <dir> [--trials <ledger>]
    python -m stratlab.r7_tps --era 2012-23 --prereg <PREREGISTERED_GITHUB2.json> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from . import r7_ind as R7

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
BARS3Y = os.path.join(SCRATCH, "bars3y")
FILL_MODEL = "tps1"
SLIP_BPS, SENS_BPS = 1.0, 2.5
INDEX_SYM = {"NIFTY": "_NIFTY50_min", "BANKNIFTY": "_NIFTYBANK_min"}


def simulate_tps(dates, o, c, side_lot: int, lo: int, hi: int, slip: float, sid: str, sym: str,
                 prm: dict = R7.TPS) -> list:
    """Trades (sim.TRADE_FIELDS tuples) of one TPS run over daily arrays; sequences whose first fill is in [lo, hi]."""
    dates = np.asarray(dates, np.int64)
    acts = R7.tps_signals(c, prm)
    hi_day = int(np.searchsorted(dates, hi, side="right")) - 1
    out = []
    seq = None                                   # {"side", "fills": [(date, px, qty)], "start"}
    for i in range(len(dates) - 1):
        a = acts[i]
        j = i + 1                                 # fill session
        if seq is not None and j > hi_day:        # the period ended with the sequence open: close at its last close
            out.append(_close(seq, int(dates[hi_day]), float(c[hi_day]) * (1 - seq["side"] * slip), "period_end",
                              sid, sym))
            seq = None
        if a is None:
            continue
        kind, side, frac = a
        px = float(o[j])
        if kind == "enter":
            if lo <= dates[j] <= hi:
                seq = {"side": side, "fills": [(int(dates[j]), px * (1 + side * slip), frac * side_lot)]}
        elif kind == "add" and seq is not None and j <= hi_day:
            seq["fills"].append((int(dates[j]), px * (1 + side * slip), frac * side_lot))
        elif kind == "exit" and seq is not None and j <= hi_day:
            out.append(_close(seq, int(dates[j]), px * (1 - side * slip), "signal", sid, sym))
            seq = None
    if seq is not None:
        k = min(hi_day, len(dates) - 1)
        out.append(_close(seq, int(dates[k]), float(c[k]) * (1 - seq["side"] * slip),
                          "period_end" if k < len(dates) - 1 else "data_end", sid, sym))
    return out


def _close(seq, exit_date, exit_px, reason, sid, sym) -> tuple:
    side = seq["side"]
    qty = sum(q for _d, _p, q in seq["fills"])
    gross = sum(side * (exit_px - p) * q for _d, p, q in seq["fills"])
    cost = 0.0
    for _d, p, q in seq["fills"]:
        bv, sv = (p * q, exit_px * q) if side > 0 else (exit_px * q, p * q)
        cost += round_trip_cost(bv, sv, "future")
    entry = sum(p * q for _d, p, q in seq["fills"]) / qty
    return (sid, sym, seq["fills"][0][0], int(side), "09:15", round(entry, 4), int(exit_date),
            "09:15" if reason == "signal" else "15:29", round(exit_px, 4), round(qty, 4), round(gross, 2),
            round(cost, 2), round(gross - cost, 2), reason)


@dataclass
class TpsSpec:
    id: str
    desc: str
    underlying: str
    params: dict = field(default_factory=lambda: {k: (list(v) if isinstance(v, tuple) else v)
                                                   for k, v in R7.TPS.items()})
    kind: str = "futures"
    family: str = "gh2_connors_tps"
    horizon: str = "positional"
    needs_index: bool = False

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(f) for f in (simulate_tps, _close, R7.tps_signals))
        blob = repr((self.id, self.underlying, sorted(self.params.items()), FILL_MODEL, src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


SPECS = [
    TpsSpec("r7_gh2_tps_nifty", "Connors TPS on NIFTY futures: >SMA200, RSI2<25 two days -> 10 %, scale in 20/30/40 % on lower closes; exit RSI2>70 (mirror short)", "NIFTY"),
    TpsSpec("r7_gh2_tps_banknifty", "Connors TPS on BANKNIFTY futures: >SMA200, RSI2<25 two days -> 10 %, scale in 20/30/40 % on lower closes; exit RSI2>70 (mirror)", "BANKNIFTY"),
]
BY_ID = {s.id: s for s in SPECS}


def cost_model() -> str:
    from .sim import COST_MODEL
    return f"{COST_MODEL}/{FILL_MODEL}"


def _daily(era: str, sym: str) -> dict:
    if era == "2012-23":
        from . import data_daily as DD
        sd = DD.load_symbol(DD.ensure_old_cache(verbose=False), sym)
    else:
        from . import data as D
        sd = D.load_symbol(D.prepare(BARS3Y)["cache_dir"], sym)
    return sd.daily()


def make_simulator(era: str, slip_bps: float):
    from .sim import LOT_SIZES

    def simulate(specs, lo, hi):
        out = {}
        for sp in specs:
            try:
                d = _daily(era, INDEX_SYM[sp.underlying])
                tr = simulate_tps(d["date"], d["o"], d["c"], LOT_SIZES[sp.underlying], lo, hi, slip_bps / 1e4,
                                  sp.id, INDEX_SYM[sp.underlying])
                out[sp.id] = {"trades": tr, "skipped": 0, "lookahead": None, "error": None}
            except Exception as exc:                      # noqa: BLE001
                out[sp.id] = {"trades": [], "skipped": 0, "lookahead": None, "error": repr(exc)[:300]}
        return out
    return simulate


def calendar_3y() -> list:
    from . import data as D
    man = D.prepare(BARS3Y)
    days: set = set()
    for s in sorted(man["symbols"]):
        sd = D.load_symbol(man["cache_dir"], s)
        if sd is not None:
            days.update(int(x) for x in sd.days)
    return sorted(days)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.r7_tps")
    ap.add_argument("--era", required=True, choices=("2012-23", "2023-26"))
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--trials", default=os.environ.get("STRATLAB_TRIALS", ""))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for s in SPECS:
            print(s.id, s.fingerprint, s.desc)
        return 0
    from .data import assert_outside_repo
    out = assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    prereg = json.load(open(a.prereg))
    specs = [BY_ID[x] for x in (a.ids.split(",") if a.ids else BY_ID)]
    for sp in specs:
        reg = prereg.get("strategies", {}).get(sp.id)
        if reg is None or reg.get("fingerprint") != sp.fingerprint:
            raise SystemExit(f"{sp.id}: not pre-registered with this fingerprint ({sp.fingerprint}): refusing to run")
    if a.era == "2012-23":
        from . import protocol_oos as PO
        from . import data_daily as DD
        res = PO.evaluate(specs, make_simulator(a.era, SLIP_BPS), make_simulator(a.era, SENS_BPS),
                          cost_model=cost_model(), prereg=prereg, dataset=DD.OLD_DIR)
        print(PO._write(out, res, f"r7_tps era 2012-23, cost model {cost_model()}, prereg {a.prereg}"))
        return 0
    from . import protocol as P
    from . import data as D
    from . import run as R
    cal = calendar_3y()
    cache = D.prepare(BARS3Y)["cache_dir"]
    results, trades = P.evaluate(specs, cal, make_simulator(a.era, SLIP_BPS), n_symbols=1,
                                 ledger_path=os.path.join(cache, "final_ledger.json"),
                                 dataset=os.path.realpath(BARS3Y) + "#r7_tps", cost_model=cost_model())
    split = results[0]["split"]
    sens = make_simulator(a.era, SENS_BPS)(specs, *split.discovery)
    for r in results:
        sc = P.screen(sens[r["id"]]["trades"], split, "positional", single_underlying=True)
        r["sens"] = {"disc": sc["disc"], "passed": sc["passed"]}
    meta = {"kind": "futures", "data": os.path.realpath(BARS3Y), "n_symbols": 1, "n_sessions": len(cal),
            "first": cal[0], "last": cal[-1], "jobs": 1, "prep_secs": 0.0, "quality_notes": [],
            "slippage_bps": "", "index_slippage_bps": SLIP_BPS, "sens_bps": "", "sens_index_bps": SENS_BPS,
            "screen_only": False, "cost_model": cost_model()}
    print(R.write_outputs(out, results, trades, {}, meta, [], []))
    if a.trials:
        from . import trials as TR
        n = TR.record_evaluation(results, trades, dataset="bars3y", cost_model=cost_model(), ledger_path=a.trials,
                                 calendar=cal, source=out)
        print(f"trials ledger: {n} row(s) appended to {a.trials}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
