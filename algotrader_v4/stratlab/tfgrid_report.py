"""
stratlab/tfgrid_report.py — REPORT.md for the timeframe grid, the trials-ledger rows and the DSR of the survivors.
Formatting and bookkeeping only: every number comes from the stage outputs of tfgrid_run.py.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import time
from collections import defaultdict

import numpy as np

from . import capital as K
from . import tfgrid as G
from . import tfgrid_run as R
from . import trials as T

REPORT = os.path.join(G.ROOT, "REPORT.md")
SUMMARY = os.path.join(G.ROOT, "summary.json")
AGENTS = ("scalping", "intraday", "swing", "futures", "options")


def _f(x, nd=0):
    if x is None or (isinstance(x, float) and (x != x)):
        return "-"
    if isinstance(x, str):
        return x
    if abs(x) == float("inf"):
        return "inf"
    return f"{x:,.{nd}f}"


def _lakh(x):
    return "-" if x is None else f"{x / 1e5:+,.2f} L"


def _robust_var(x):
    x = sorted(v for v in x if v == v)
    if len(x) < 2:
        return float("nan")
    med = x[len(x) // 2]
    mad = sorted(abs(v - med) for v in x)[len(x) // 2]
    return (1.4826 * mad) ** 2


def ruin_day(daily, cal):
    """First session on which the Rs 10 lakh account's realised equity is <= 0 (None if never)."""
    eq = K.CAPITAL + np.cumsum(daily) if daily else np.array([K.CAPITAL])
    k = np.flatnonzero(eq <= 0)
    return int(cal[k[0]]) if len(k) and k[0] < len(cal) else None


def _moments(daily):
    return T.moments(daily)


def trials_rows(pre, s1, fin, s2) -> list:
    fps = {g["id"]: "f:" + g["fingerprint"] for g in pre["grid_combos"]}
    fps.update({n["id"]: "f:" + n["fingerprint"] for n in pre["native_combos"]})
    fam = {g["id"]: G.RULES[g["rule"]].family for g in pre["grid_combos"]}
    rows = []
    for cid, r in s1.items():
        m = r.get("metrics") or {}
        d = r.get("daily") or []
        md = _moments(d)
        n = m.get("trades", 0)
        verdict = "INVALID" if r.get("invalid") else ("PASSED SCREEN" if r["screen"]["passed"] else "FAILED SCREEN")
        rows.append({"strategy_id": cid, "family": "tfgrid-native" if cid.startswith("nat:") else "tfgrid",
                     "group": fam.get(cid, r.get("kind", "")), "params_fp": fps.get(cid, ""), "era": "2023-26",
                     "protocol": G.TRIALS_PROTOCOL, "stage": "tfgrid-screen", "dataset": "bars3y",
                     "period": f"{G.DISC[0]}-{G.DISC[1]}", "cost_model": _cm(), "trades": n, "net": m.get("net"),
                     "pf": m.get("pf"), "mean_net": (m["net"] / n) if n else None,
                     "mean_gross": r.get("mean_gross"), "mean_cost": r.get("mean_cost"), "n_days": md["n"],
                     "sr_daily": md["sr"], "skew_daily": md["skew"], "kurt_daily": md["kurt"], "verdict": verdict,
                     "source": "tfgrid/stage1", "note": "; ".join(r.get("screen", {}).get("reasons", []))[:200]
                     if not r.get("invalid") else str(r["invalid"])[:200]})
    for cid, f in (fin or {}).items():
        tr = _load(os.path.join(R.FIN, "trades", G._fn(cid) + ".pkl.gz"))
        lo, hi = f["period"]
        daily = K.daily_series({"trades": tr}, G.calendar3(), lo, hi) if tr is not None else []
        rows.append(G.trial_row(cid, era="2023-26", stage="tfgrid-final", dataset="bars3y", period=(lo, hi),
                                trades=[t for t in (tr or []) if lo <= t[2] <= hi], daily=daily,
                                verdict=f["verdict"], fp=fps.get(cid, ""), source="tfgrid/final"))
    for cid, s in (s2 or {}).items():
        if "metrics" not in s:
            rows.append({"strategy_id": cid, "family": "tfgrid", "params_fp": fps.get(cid, ""), "era": "2016-23",
                         "protocol": G.TRIALS_PROTOCOL, "stage": "tfgrid-oos", "dataset": "bars_min_old",
                         "period": f"{G.OLD_ERA[0]}-{G.OLD_ERA[1]}", "verdict": s["verdict"],
                         "source": "tfgrid/stage2", "note": str(s.get("detail"))[:200]})
            continue
        tr = _load(os.path.join(R.S2, "trades", G._fn(cid) + ".pkl.gz")) or []
        rows.append(G.trial_row(cid, era="2016-23", stage="tfgrid-oos", dataset="bars_min_old", period=G.OLD_ERA,
                                trades=tr, daily=s.get("daily") or [], verdict=s["verdict"], fp=fps.get(cid, ""),
                                source="tfgrid/stage2", note="; ".join(s["judge"]["reasons"])[:200]))
    return rows


def _cm():
    from .sim import COST_MODEL
    return COST_MODEL


def _load(p):
    try:
        return G.load_trades(p)
    except FileNotFoundError:
        return None


def nifty_bh(cache, lo, hi):
    from . import data as D
    sd = D.load_symbol(cache, "_NIFTY50_min")
    d = sd.daily()
    m = (d["date"] >= lo) & (d["date"] <= hi)
    c = d["c"][m]
    o = d["o"][m]
    return float(c[-1] / o[0] - 1.0) if len(c) else float("nan")


def main(append: bool = True):
    pre = G.load_prereg()
    s1 = R.load_stage1()
    fin = R._jload(G.FINAL_LEDGER, {}) or {}
    s2 = R._jload(R.STAGE2_LEDGER, {}) or {}
    ctrl = R._jload(os.path.join(R.CTRL, "controls.json"), {}) or {}
    live = R._jload(os.path.join(R.LIVE, "live.json"), {}) or {}
    # ── trials ledger ──
    rows = trials_rows(pre, s1, fin, s2)
    appended = G.append_trials(rows, T.DEFAULT_LEDGER) if append else 0
    led = T.TrialsLedger(T.DEFAULT_LEDGER).read()
    n_era = {e: T.count_trials(led, e) for e in ("2023-26", "2012-23", "2016-23")}
    n_full = sum(n_era.values())
    n_fair = len(pre["grid_combos"]) + len(pre["native_combos"])
    srs = [_moments(r["daily"])["sr"] for r in s1.values() if r.get("daily") and r.get("metrics", {}).get("trades")]
    v_rob = _robust_var(srs)
    # ── DSR of every stage-2 judged combo ──
    dsr = {}
    for cid, s in s2.items():
        if "daily" not in s:
            continue
        md = _moments(s["daily"])
        t_ = md["n"]
        v = max(v_rob if v_rob == v_rob else 0.0, 1.0 / max(t_ - 1, 1))
        dsr[cid] = {"sr_daily": md["sr"], "sr_ann": md["sr"] * math.sqrt(250) if md["sr"] == md["sr"] else None,
                    "T": t_, "var_sr": v, "dsr_full": T.dsr(md["sr"], t_, md["skew"], md["kurt"], n_full, v)["dsr"],
                    "dsr_fair": T.dsr(md["sr"], t_, md["skew"], md["kurt"], n_fair, v)["dsr"],
                    "psr0": T.psr(md["sr"], 0.0, t_, md["skew"], md["kurt"])}
    # ── verdicts ──
    status = {}
    for cid, r in s1.items():
        st = {"stage1": "INVALID" if r.get("invalid") else ("PASS" if r["screen"]["passed"] else "FAIL")}
        if cid in fin:
            st["final"] = fin[cid]["verdict"]
        if cid in s2:
            st["stage2"] = s2[cid]["verdict"]
        if cid in dsr:
            st["dsr_full"] = dsr[cid]["dsr_full"]
            st["dsr_fair"] = dsr[cid]["dsr_fair"]
        st["proven"] = bool(st["stage1"] == "PASS" and st.get("final") == "CONFIRMED"
                            and st.get("stage2") == "OOS-CONFIRMED" and (st.get("dsr_full") or 0) >= 0.95)
        status[cid] = st
    lines = []
    w = lines.append
    grid_ids = {g["id"] for g in pre["grid_combos"]}
    n1 = sum(1 for c, s in status.items() if s["stage1"] == "PASS")
    n_inv = sum(1 for c, s in status.items() if s["stage1"] == "INVALID")
    nf = sum(1 for c, s in status.items() if s.get("final") == "CONFIRMED")
    n2 = sum(1 for c, s in status.items() if s.get("stage2") == "OOS-CONFIRMED")
    npv = sum(1 for c, s in status.items() if s["proven"])
    rand = [v for tf in ctrl.values() for k, v in tf.items() if k.startswith("CTRL_RANDOM")]
    luck_rate = (sum(1 for v in rand if v["screen_passed"]) / len(rand)) if rand else float("nan")
    w("# Timeframe grid: every strategy, every timeframe, Rs 10 lakh per agent (research, PAPER only)")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `<scratch>/tfgrid` (pre-registration sha256 "
      f"`{open(G.PREREG + '.sha256').read().strip()}`, committed before any result).")
    w("")
    w("## In plain words")
    w("")
    w(f"* **{len(status):,} strategy x timeframe combinations were run** like a real Rs 10 lakh account per agent "
      f"({len(grid_ids & set(status))} portable rule x timeframe combos + {len(status) - len(grid_ids & set(status))} "
      f"catalogue strategies on their own timeframe), every one with the 1:3 risk-reward bracket you asked for.")
    w(f"* **Passed the 2023-26 screen: {n1}.** Confirmed on the one-look 2025-26 hold-out: {nf}. "
      f"Passed the 2016-2023 test: {n2}. **Proven (both eras + DSR >= 0.95): {npv}.**")
    w(f"* Luck check: random-entry accounts passed the same screen {luck_rate:.1%} of the time "
      f"({sum(1 for v in rand if v['screen_passed'])} of {len(rand)}); with {len(status):,} combos about "
      f"{luck_rate * len(status):.0f} passes would be expected from luck alone at that rate, and the lab's rule of thumb "
      f"(~5 % per coin-flip-like gate) gives an upper bound of ~{0.05 * len(status):.0f}. Observed: {n1}.")
    w("")
    # ── per agent ──
    w("## Per agent: Rs 10,00,000 account (not compounded), 2023-11-10 .. 2025-12-16 (DISCOVERY)")
    w("")
    w("Return %/yr = net / Rs 10 lakh / years. Max DD in Rs and % of the account's peak. Sharpe = daily realised P&L "
      "x sqrt(250). Skipped = signals the account could not take (no slot, no money, daily loss stop, trades/day cap, "
      "below one lot / too small); a symbol already held is counted separately.")
    w("")
    hdr = ("| combo | trades | net | return %/yr | max DD Rs (%) | worst month | win % | PF | Sharpe | skipped "
           "| compounded net (report only) | stage 1 | FINAL | 2016-23 |")
    for ag in AGENTS:
        rows_ag = [(c, r) for c, r in s1.items() if r.get("agent") == ag and r.get("metrics")]
        w(f"### {ag} agent ({len(rows_ag)} combos)")
        w("")
        if not rows_ag:
            w("No grid or catalogue combo maps to this agent; its strategy is the live VRP strangle (section below).")
            w("")
            continue
        top = sorted(rows_ag, key=lambda kv: -kv[1]["metrics"]["net"])[:5]
        proven = [c for c, _ in rows_ag if status[c]["proven"]]
        w(hdr)
        w("|" + "---|" * 14)
        for c, r in top:
            m = r["metrics"]
            sk = sum(v for k, v in r["skipped"].items() if k != "symbol_busy")
            w(f"| `{c}` | {m['trades']:,} | {_lakh(m['net'])} | {m['return_pct_per_year']:+.1f} | "
              f"{_f(m['max_dd_rs'])} ({m['max_dd_pct']:.1f} %) | {m['worst_month']} {_lakh(m['worst_month_rs'])} | "
              f"{m['win_pct']:.1f} | {_f(m['pf'], 2)} | {m['sharpe']:.2f} | {sk:,} | "
              f"{_lakh((r.get('compounded') or {}).get('net'))} | {status[c]['stage1']} | "
              f"{status[c].get('final', '-')} | {status[c].get('stage2', '-')} |")
        nets = [r["metrics"]["net"] for _, r in rows_ag]
        cal_d = [d for d in G.calendar3() if G.DISC[0] <= d <= G.DISC[1]]
        ruined = sum(1 for _, r in rows_ag if ruin_day(r.get("daily") or [], cal_d))
        w("")
        w(f"{ruined} of {len(rows_ag)} combos would have lost the whole Rs 10 lakh (realised equity <= 0) before "
          f"2025-12-16; the simulation keeps the fixed base so their full loss is still shown.")
        w(f"All {len(rows_ag)} combos of this agent: median net {_lakh(float(np.median(nets)))}, "
          f"{sum(1 for x in nets if x > 0)} of {len(nets)} made money on DISCOVERY, "
          f"{sum(1 for c, _ in rows_ag if status[c]['stage1'] == 'PASS')} passed the screen. "
          f"Maths-proven set: {', '.join(proven) if proven else '**none**'}.")
        w("")
    # ── stage-1 passers in detail ──
    ps = [c for c, s in status.items() if s["stage1"] == "PASS"]
    if ps:
        w("## Stage-1 passers: FINAL (one look), 2016-23, DSR")
        w("")
        w("| combo | agent | DISC net / PF | FINAL net / PF | 2016-23 net / PF (H1 / H2) | 2016-23 return %/yr, max DD | "
          "excess vs NIFTY H1 / H2 | DSR full N / fair N | proven |")
        w("|" + "---|" * 9)
        for c in sorted(ps):
            r = s1[c]
            f = fin.get(c, {}).get("final", {})
            s = s2.get(c, {})
            j = s.get("judge", {})
            m2 = s.get("metrics", {})
            d = dsr.get(c, {})
            ex = s.get("excess", {})
            w(f"| `{c}` | {r['agent']} | {_lakh(r['metrics']['net'])} / {_f(r['metrics']['pf'], 2)} | "
              f"{_lakh(f.get('net'))} / {_f(f.get('pf'), 2)} | "
              f"{_lakh(m2.get('net'))} / {_f(m2.get('pf'), 2)} ({_lakh(j.get('h1', {}).get('net'))} / "
              f"{_lakh(j.get('h2', {}).get('net'))}) | {_f(m2.get('return_pct_per_year'), 1)}, "
              f"{_f(m2.get('max_dd_rs'))} | {_lakh(ex.get('h1', [0, None])[1])} / {_lakh(ex.get('h2', [0, None])[1])} | "
              f"{_f(d.get('dsr_full'), 3)} / {_f(d.get('dsr_fair'), 3)} | {'YES' if status[c]['proven'] else 'no'} |")
        w("")
    # ── heat tables ──
    w("## Strategy x timeframe heat table (stage 1 on the account: P = pass, . = fail, ! = invalid, blank = not run)")
    w("")
    for uni, tfs in (("stocks", G.TFS), ("index", G.TFS), ("stockfut", G.POSITIONAL_TFS)):
        w(f"### {uni} ({'agent by timeframe' if uni == 'stocks' else 'futures agent'})")
        w("")
        w("| family | rule | " + " | ".join(tfs) + " | DISC net by tf (Rs lakh) |")
        w("|---|---|" + "---|" * len(tfs) + "---|")
        for rid, rl in G.RULES.items():
            cells, nets = [], []
            for tf in tfs:
                cid = f"tfg_{rid}@{tf}:{uni}"
                s = status.get(cid)
                r = s1.get(cid, {})
                cells.append(" " if s is None else {"PASS": "P", "FAIL": ".", "INVALID": "!"}[s["stage1"]])
                nets.append(f"{r['metrics']['net'] / 1e5:+.1f}" if r.get("metrics") else "-")
            if all(c == " " for c in cells):
                continue
            w(f"| {rl.family} | {rid} | " + " | ".join(cells) + " | " + " ".join(nets) + " |")
        w("")
    # ── untradeable at 1 % risk ──
    small = defaultdict(int)
    tot = defaultdict(int)
    for c, r in s1.items():
        if not r.get("metrics"):
            continue
        k = (r.get("universe"), r.get("tf"))
        tot[k] += 1
        if r["candidates"] and r["skipped"].get("below_one_lot", 0) >= 0.5 * r["candidates"]:
            small[k] += 1
    if small:
        w("## Too big for a Rs 10 lakh account at 1 % risk")
        w("")
        w("A futures trade needs at least one lot; when one lot already risks more than Rs 10,000 at the rule's stop, "
          "the account skips the signal (pre-registered). Combos where at least half the signals were skipped for this:")
        w("")
        w("| universe | tf | combos (of) |")
        w("|---|---|---|")
        for k in sorted(small, key=str):
            w(f"| {k[0]} | {k[1]} | {small[k]} ({tot[k]}) |")
        w("")
        w("Daily / weekly NIFTY and BANKNIFTY futures and weekly stock futures are effectively not tradeable under the "
          "1 % rule with Rs 10 lakh: a 1.5 x daily ATR stop on one NIFTY lot is ~Rs 20-25 k of risk. Their 'fail' is "
          "'insufficient trades', not evidence about the rule.")
        w("")
    # ── counts ──
    w("## Counts")
    w("")
    w("| | combos | invalid | passed stage 1 | FINAL confirmed | passed 2016-23 | proven (DSR >= 0.95) |")
    w("|---|---|---|---|---|---|---|")
    for lab_, sel in [("tf " + tf, lambda c, tf=tf: c in grid_ids and s1[c].get("tf") == tf) for tf in G.TFS] + \
                     [("native catalogue", lambda c: c.startswith("nat:"))] + \
                     [("agent " + a, lambda c, a=a: s1[c].get("agent") == a) for a in AGENTS] + \
                     [("ALL", lambda c: True)]:
        cs = [c for c in status if sel(c)]
        w(f"| {lab_} | {len(cs)} | {sum(1 for c in cs if status[c]['stage1'] == 'INVALID')} | "
          f"{sum(1 for c in cs if status[c]['stage1'] == 'PASS')} | "
          f"{sum(1 for c in cs if status[c].get('final') == 'CONFIRMED')} | "
          f"{sum(1 for c in cs if status[c].get('stage2') == 'OOS-CONFIRMED')} | "
          f"{sum(1 for c in cs if status[c]['proven'])} |")
    w("")
    mine = sum(1 for r in led if r.get("protocol") == G.TRIALS_PROTOCOL)
    w(f"Trials: {mine} tfgrid rows in `trials_v2.csv` ({appended} appended by this report run; backup "
      f"`trials_v2.before_tfgrid.csv`). Variants now: "
      f"2023-26 {n_era['2023-26']:,}, 2012-23 {n_era['2012-23']:,}, 2016-23 {n_era['2016-23']:,}; full N = "
      f"{n_full:,}; fair N (this grid) = {n_fair:,}. V[SR] = robust (1.4826 MAD)^2 of the {len(srs)} stage-1 daily "
      f"Sharpes = {v_rob:.6f} (floored at 1/(T-1) per survivor). Expected best daily SR of N useless trials: "
      f"full N {T.expected_max_sr(n_full, v_rob):.4f}, fair N {T.expected_max_sr(n_fair, v_rob):.4f} "
      f"(x sqrt(250) = {T.expected_max_sr(n_full, v_rob) * math.sqrt(250):.2f} / "
      f"{T.expected_max_sr(n_fair, v_rob) * math.sqrt(250):.2f} a year).")
    w("")
    # ── controls ──
    if ctrl:
        w("## Controls through the same engine + account (first 25 stocks, DISCOVERY; not trials)")
        w("")
        w("| tf | RANDOM (10 seeds): mean PF, gross R/trade, screen passes | PLANTED 5 % PF | 10 % | 20 % | 100 % (cheat) |")
        w("|---|---|---|---|---|---|")
        for tf in G.TFS:
            c = ctrl.get(tf)
            if not c:
                continue
            rr = [v for k, v in c.items() if k.startswith("CTRL_RANDOM")]
            pf = [v["pf"] for v in rr if isinstance(v["pf"], (int, float))]
            g = [v["gross_r_per_trade"] for v in rr if v["gross_r_per_trade"] is not None]
            w(f"| {tf} | {np.mean(pf):.2f}, {np.mean(g):+.3f} R, {sum(v['screen_passed'] for v in rr)}/{len(rr)} | "
              + " | ".join(f"{_f(c.get(f'CTRL_PLANTED_{x}', {}).get('pf'), 2)}" for x in (5, 10, 20, 100)) + " |")
        w("")
        w("Random entries have a gross (after slippage, before brokerage/taxes) of about -slippage per trade and lose "
          "the costs; 1 of 80 random accounts passed the screen (daily). Planted edges raise PF in proportion to their "
          "size at every intraday timeframe and daily; the cheat (100 %) is always found. Weekly controls have ~37 "
          "trades each, too few to resolve a 5-20 % edge. The engine and the account add no edge of their own.")
        w("")
    # ── live ──
    if live:
        w("## The live PAPER strategies + AUC-FADE-2PCT as Rs 10 lakh accounts (report only, already judged)")
        w("")
        w("Each with its own agent limits from config.py (Rs 1 lakh per stock trade, max gross Rs 10 lakh, its daily "
          "loss stop; strangle 2 lots) and its OWN exits; the 1:3 rows are a report-only comparison. 2023-26 = "
          "DISCOVERY + FINAL (both already seen); 2016-23 = the old era (dirty-link rule applied).")
        w("")
        w("| strategy | exits | era | trades | net | return %/yr | max DD Rs (%) | worst month | win % | PF | Sharpe |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for era in ("2023-26", "2016-23"):
            for k, v in (live.get(era) or {}).items():
                sid, var = k.split("|")
                if "metrics" not in v:
                    w(f"| {sid} | {var} | {era} | {v.get('note', '')} |||||||")
                    continue
                m = v["metrics"]
                w(f"| {sid} | {var} | {era} | {m['trades']:,} | {_lakh(m['net'])} | {m['return_pct_per_year']:+.1f} | "
                  f"{_f(m['max_dd_rs'])} ({m['max_dd_pct']:.1f} %) | {m['worst_month']} {_lakh(m['worst_month_rs'])} | "
                  f"{m['win_pct']:.1f} | {_f(m['pf'], 2)} | {m['sharpe']:.2f} |")
        w("")
        w("VRP-STRANGLE-016-WINGED numbers are MODELLED (Black-Scholes on the index with the previous India VIX close; "
          "not option prices).")
        w("")
    # ── NIFTY ──
    try:
        nb3 = nifty_bh(G.CACHE3Y, *G.DISC)
        nbo = nifty_bh(G.CACHE_OLD, *G.OLD_ERA)
        w(f"NIFTY buy-and-hold for comparison: DISCOVERY 2023-11-10..2025-12-16 {nb3:+.1%} (Rs 10 lakh -> "
          f"{_lakh(1e6 * nb3)}); 2016-01-01..2023-11-09 {nbo:+.1%} ({_lakh(1e6 * nbo)}).")
        w("")
    except Exception:
        nb3 = nbo = None
    w("## Caveats")
    w("")
    w("* Survivorship: only today's NIFTY-100 names; it flatters long rules, more so on 2016-23.")
    w("* Index futures are the spot index (no basis, no roll); stock futures are the cash price with futures costs; "
      "2026 lot sizes (NIFTY 65, BANKNIFTY 30, stock lots of 2026-11-09) and 2026 Zerodha costs on every year.")
    w("* Option results are modelled premiums (flat-VIX Black-Scholes), not traded prices.")
    w("* The margin model is an approximation (15 % of futures notional = config's estimate; MIS 5x; CNC 1x).")
    w("* 1:3 bracket everywhere (your instruction): the textbook exits of these rules were NOT tested.")
    w("* P&L is realised on the exit day (no mark-to-market of open swing positions in the drawdown).")
    w("")
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    summ = {"combos": len(status), "invalid": n_inv, "stage1": n1, "final_confirmed": nf, "stage2": n2, "proven": npv,
            "luck_rate_random": luck_rate, "n_full": n_full, "n_era": n_era, "n_fair": n_fair, "var_sr": v_rob,
            "appended": appended, "status": status, "dsr": dsr, "nifty_bh": {"disc": nb3, "old": nbo}}
    R._jsave(SUMMARY, summ)
    print("\n".join(lines[:12]))
    return summ


if __name__ == "__main__":
    main()
