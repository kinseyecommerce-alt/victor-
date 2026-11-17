"""stratlab/size10l_report.py — writes stratlab/SIZE10L.md from <scratchpad>/size10l (size10l_run.py and
size10l_options.py outputs) and the stored results of the strategy types whose size does not change.

    cd algotrader_v4 && python -m stratlab.size10l_run --stage report
"""
from __future__ import annotations

import os
import statistics
import sys

from . import size10l_options as O
from . import size10l_run as Z

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "SIZE10L.md")
YEARS = 3.0                                  # 2023-11-10 .. 2026-11-09


def _r(x, nd=0):
    if x is None or x != x:
        return "-"
    if x == float("inf"):
        return "inf"
    return f"{x:,.{nd}f}"


def _both(r) -> bool:
    return r["h1"]["net"] > 0 and r["h2"]["net"] > 0


def _old_verdicts() -> dict:
    """{id: one-lot / Rs-1-lakh old-era verdict} from the lab's ledgers (corrected first)."""
    out = {}
    for rel in ("audit/ledgers/oldera_min_corrected.json", "oldera_min/ledger.json",
                "audit/ledgers/oldera_daily_corrected.json", "stratlab_cache/oldera_oos_ledger.json"):
        for k, v in (Z._jload(os.path.join(Z.SCRATCH, rel), {}) or {}).items():
            res = v.get("result", v) if isinstance(v, dict) else {}
            if isinstance(res, dict) and res.get("verdict"):
                out.setdefault(k.split("|")[0], f"{res['verdict']} ({rel.split('/')[-1].replace('.json', '')})")
    return out


def write() -> str:
    sys.path.insert(0, os.path.dirname(HERE))
    from costs import round_trip_cost
    from . import audit_run as AR
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    J = lambda f, d=None: Z._jload(os.path.join(Z.OUT, f), d)        # noqa: E731
    pre, pre2 = Z.load_prereg(), O.load_prereg()
    sha, sha2 = open(Z.PREREG + ".sha256").read().strip(), open(O.PREREG2 + ".sha256").read().strip()
    ver, ctl, scr = J("verify_1lakh.json"), J("controls.json"), J("screen.json")
    osc_all = J("options_screen.json", {}) or {}
    osc = {i: r for i, r in osc_all.items() if not r["report_only"]}
    fr = (J("final.json", {}) or {}).get("results", {})
    ofin = J("options_final.json", {}) or {}
    ofr = ofin.get("results", {})
    om, od, dsr, live = J("oldera_min.json", {}) or {}, J("oldera_daily.json", {}) or {}, J("dsr.json", {}) or {}, \
        J("live.json", {}) or {}
    led, oled = J("ledger.json", {}) or {}, J("options_ledger.json", {}) or {}
    stored = Z._jload(os.path.join(Z.SCRATCH, "audit", "rerun", "screen_corrected.json"), {})
    rc_final = (Z._jload(os.path.join(Z.SCRATCH, "audit", "rerun", "final_corrected.json"), {}) or {}).get("results", {})
    rc_oldmin = Z._jload(os.path.join(Z.SCRATCH, "audit", "rerun", "oldera_min.json"), {}) or {}
    oldv = _old_verdicts()
    ids = pre["scope"]["retested_ids"]
    drow = dsr.get("rows", {})
    ctl_ok = bool(ctl and ctl.get("ok"))
    twin_bad = sorted(i for i in ids if not scr[i].get("l1_matches_stored", True)) + \
        sorted(i for i in osc if not osc[i].get("l1_matches_stored", True))

    def funnel(sel, sc, fin):
        p5 = sorted(i for i in sel if sc[i]["passed"])
        p1 = sorted(i for i in sel if sc[i]["stored_1L"]["passed"])
        c5 = sorted(i for i in p5 if (fin.get(i, {}).get("final") or {}).get("confirmed"))
        return p1, p5, c5
    mis = [i for i in ids if REGISTRY[i].cost_kind == "equity_mis"]
    sfut = [i for i in ids if REGISTRY[i].cost_kind == "future"]
    f_mis, f_fut, f_opt = funnel(mis, scr, fr), funnel(sfut, scr, fr), funnel(sorted(osc), osc, ofr)
    conf5 = f_mis[2] + f_fut[2]
    oos5 = sorted(i for i in conf5 if om.get(i, {}).get("size10l", {}).get("verdict") == "OOS-CONFIRMED")
    proven = sorted(i for i in oos5 if (drow.get(i, {}).get("dsr") or 0) >= Z.DSR_MIN and ctl_ok and i not in twin_bad)
    cnc = sorted(i for i, sp in REGISTRY.items() if sp.cost_kind == "equity_cnc")
    idx = sorted(i for i, sp in REGISTRY.items() if sp.underlying)

    def existing(sel):
        p = sorted(i for i in sel if stored[i]["passed"])
        c = sorted(i for i in p if (rc_final.get(i, {}).get("final") or {}).get("confirmed"))
        o = sorted(i for i in c if (rc_oldmin.get(i, {}).get("corrected") or {}).get("verdict") == "OOS-CONFIRMED")
        return p, c, o
    e_cnc, e_idx = existing(cnc), existing(idx)
    tf = AR._tf_rows(os.path.join(AR.AUDIT, "tfgrid_corrected", "stage1"))
    tf_grid = [k for k in tf if k.startswith("tfg_")]
    tf_pass = sorted(k for k in tf if (tf[k].get("screen") or {}).get("passed"))
    tf_oos = Z._jload(os.path.join(Z.SCRATCH, "audit", "rerun", "tfgrid_oos.json"), {}) or {}
    tf_ok = sorted(k for k, v in tf_oos.items() if v.get("corrected") == "OOS-CONFIRMED")
    L = []
    A = L.append
    A("# Every lab strategy at Rs 10 lakh of capital per agent")
    A("")
    A(f"Pre-registered before any result at the new size: `stratlab/PREREGISTERED_SIZE10L.json` (sha256 `{sha}`) and, "
      f"before any option result, `stratlab/PREREGISTERED_SIZE10L_AMENDMENT1.json` (sha256 `{sha2}`), which supersedes "
      "it and adds the options strategies. Code: `stratlab/size10l_run.py`, `stratlab/size10l_options.py`. Research only; "
      "no live rule or verdict is changed.")
    A("")
    A("## Result")
    A("")
    A(f"- **Proven at the Rs 10 lakh size: {len(proven)}**" + (f" ({', '.join(proven)})" if proven else "") + ".")
    A(f"- Re-tested as new trials: **K = {pre2['N_K_V']['K']}** strategies ({len(mis)} intraday / scalping, {len(sfut)} "
      f"stock futures, {len(osc)} options). The luck correction counts all of them: N = {dsr.get('N', '-')}.")
    A(f"- {len(conf5) + len(f_opt[2])} strategies pass DISCOVERY and are confirmed on the FINAL year at the new size "
      f"({len(conf5)} stock futures, {len(f_opt[2])} options). None of them clears the remaining gates (below).")
    A("- No intraday or scalping strategy passes DISCOVERY at Rs 5 lakh per trade either.")
    if live:
        a, b = live["AUC-FADE-2PCT"]["at_1lakh"], live["AUC-FADE-2PCT"]["at_5lakh"]
        A(f"- The live rules keep their verdicts and do better per rupee at the new size (report only): AUC-FADE-2PCT's "
          f"FINAL year goes from PF {_r(a['final']['pf'], 3)} (net Rs {_r(a['final']['net'])} at Rs 1 lakh) to PF "
          f"{_r(b['final']['pf'], 3)} (net Rs {_r(b['final']['net'])} at Rs 5 lakh).")
    A("")
    A("## All types")
    A("")
    A("| Type | Strategies | At Rs 10 lakh per agent | DISCOVERY pass: before -> now | FINAL confirmed | Old-era OOS confirmed | Proven (all gates incl. DSR >= 0.95) |")
    A("|---|---|---|---|---|---|---|")
    A(f"| Intraday + scalping (MIS) | {len(mis)} | re-tested at Rs 5 lakh per trade | {len(f_mis[0])} -> {len(f_mis[1])} | "
      f"{len(f_mis[2])} | 0 | 0 |")
    A(f"| Stocks at futures costs | {len(sfut)} | re-tested at Rs 5 lakh per trade | {len(f_fut[0])} -> {len(f_fut[1])} | "
      f"{len(f_fut[2])} | {len(oos5)} | {len(proven)} |")
    A(f"| Options | {len(osc)} | re-tested, lots from a Rs 1 lakh premium / risk or Rs 5 lakh margin budget | "
      f"{len(f_opt[0])} -> {len(f_opt[1])} | {len(f_opt[2])} | not available at the new size | 0 |")
    A(f"| Delivery swing (CNC) | {len(cnc)} | identical at Rs 10 lakh - existing result stands | {len(e_cnc[0])} | "
      f"{len(e_cnc[1])} | {len(e_cnc[2])} | 0 |")
    A(f"| Index futures | {len(idx)} | identical (one real lot) - existing result stands | {len(e_idx[0])} | "
      f"{len(e_idx[1])} | {len(e_idx[2])} | 0 |")
    A(f"| Timeframe grid (tfgrid) | {len(tf_grid)} (+ {len(tf) - len(tf_grid)} catalogue rules with the 1:3 bracket) | "
      f"already traded from Rs 10 lakh accounts - existing result stands | {len(tf_pass)} | see tfgrid report | "
      f"{len(tf_ok)} | 0 |")
    A("")
    A("Why three types are not re-run: delivery fees are a fixed percentage of the trade (no flat brokerage), so the "
      "profit factor is the same at any size and a Rs 10 lakh account simply holds 10 positions of Rs 1 lakh; an index "
      "future is already one real lot, where the flat brokerage is under 0.3 bps; tfgrid sized every trade from a Rs 10 "
      "lakh account (median fee per intraday trade Rs 465 = trades of Rs 10-25 lakh).")
    A("")
    A("## Fees: what the size changes")
    A("")
    A("| Product | Round trip at Rs 1 lakh | per Rs 1 lakh at Rs 5 lakh | saving per Rs 1 lakh traded |")
    A("|---|---|---|---|")
    for kind, name in (("equity_mis", "Intraday (MIS)"), ("future", "Stock at futures costs")):
        c1, c5 = round_trip_cost(1e5, 1e5, kind), round_trip_cost(5e5, 5e5, kind) / 5
        A(f"| {name} | Rs {c1:.2f} ({c1 / 10:.2f} bps) | Rs {c5:.2f} ({c5 / 10:.2f} bps) | Rs {c1 - c5:.2f} |")
    A("")
    med = lambda xs: statistics.median(xs) if xs else float("nan")       # noqa: E731
    for sel, name in ((mis, "intraday / scalping"), (sfut, "stock-futures")):
        s2 = [i for i in sel if scr[i]["disc"]["trades"] > 0]
        g1 = med([scr[i]["stored_1L"]["disc"]["gross"] / scr[i]["stored_1L"]["disc"]["trades"] for i in s2])
        c1 = med([scr[i]["stored_1L"]["disc"]["cost"] / scr[i]["stored_1L"]["disc"]["trades"] for i in s2])
        c5 = med([scr[i]["disc"]["cost"] / scr[i]["disc"]["trades"] / 5 for i in s2])
        b1, b5 = sum(1 for i in sel if _both(scr[i]["stored_1L"])), sum(1 for i in sel if _both(scr[i]))
        A(f"- Median {name} strategy ({len(s2)} with trades), per Rs 1 lakh traded: gross after slippage Rs {g1:.0f}; fees "
          f"Rs {c1:.0f} at Rs 1 lakh and Rs {c5:.0f} at Rs 5 lakh. Net positive in both halves of DISCOVERY: {b1} -> {b5} "
          f"of {len(sel)}.")
    oc = [i for i in osc if osc[i]["disc"]["trades"] > 0]
    A(f"- Options: the median strategy's fee per lot falls from Rs "
      f"{med([osc[i]['stored_1L']['disc']['cost'] / osc[i]['stored_1L']['disc']['trades'] for i in oc]):.0f} to Rs "
      f"{med([osc[i]['disc']['cost'] / osc[i]['disc']['trades'] / max(1, osc[i]['lots_median'] or 1) for i in oc]):.0f} "
      f"(approximate: fees per trade divided by the median lots). Net positive in both halves: "
      f"{sum(1 for i in osc if _both(osc[i]['stored_1L']))} -> {sum(1 for i in osc if _both(osc[i]))} of {len(osc)}.")
    A("- Slippage (2 bps a side on stocks, the option fill haircut per unit) does not shrink with size and is unchanged.")
    A("")
    A("## Stock strategies that pass DISCOVERY at Rs 5 lakh per trade")
    A("")
    A("| Strategy | DISCOVERY trades / PF / net | at Rs 1 lakh: passed, PF | FINAL trades / PF / net | FINAL | "
      "2016-23 OOS at the new size | DSR | Net per year |")
    A("|---|---|---|---|---|---|---|---|")
    pass5 = f_mis[1] + f_fut[1]
    for i in pass5:
        r, s1 = scr[i], scr[i]["stored_1L"]
        f = (fr.get(i) or {}).get("final") or {}
        o = (om.get(i) or {}).get("size10l") or {}
        d = drow.get(i, {})
        A(f"| `{i}` | {r['disc']['trades']} / {_r(r['disc']['pf'], 3)} / Rs {_r(r['disc']['net'])} | "
          f"{'yes' if s1['passed'] else 'no'}, {_r(s1['disc']['pf'], 3)} | {f.get('trades', '-')} / {_r(f.get('pf'), 3)} / "
          f"Rs {_r(f.get('net'))} | {'CONFIRMED' if f.get('confirmed') else 'rejected'} | "
          f"{o.get('verdict', 'not run (FINAL rejected)')} | {_r(d.get('dsr'), 4)} | "
          f"Rs {_r((r['disc']['net'] + (f.get('net') or 0.0)) / YEARS)} |")
    A("")
    A("All of them are stock rules booked at futures costs and held for days. Net per year = (DISCOVERY + FINAL net) / 3 "
      "at Rs 5 lakh per trade; the number of positions open together is not capped by the account in this test, so it "
      "is not a return on Rs 10 lakh (tfgrid is the account-level test).")
    A("")
    for i in pass5:
        A(f"- `{i}`: {REGISTRY[i].desc}")
    A("")
    if conf5:
        A("### The FINAL-confirmed stock strategies")
        A("")
        for i in conf5:
            f, o, d = fr[i]["final"], (om.get(i) or {}).get("size10l") or {}, drow.get(i, {})
            e = o.get("era") or {}
            dd = od.get(i) or {}
            a1 = (om.get(i) or {}).get("at_1lakh") or {}
            A(f"- `{i}`. FINAL: {f['trades']} trades, PF {_r(f['pf'], 3)}, net Rs {_r(f['net'])} (the same trades at "
              f"Rs 1 lakh: PF {_r(fr[i]['l1_final']['pf'], 3)}, net Rs {_r(fr[i]['l1_final']['net'])}). 2016-23 minute at "
              f"the new size: **{o.get('verdict')}** - {o.get('why')} (era: {e.get('trades')} trades, PF {_r(e.get('pf'), 3)}, "
              f"net Rs {_r(e.get('net'))}; at Rs 1 lakh it was {a1.get('verdict', '-')})."
              + (f" 2012-23 daily (report): {dd.get('verdict')} - {dd.get('why')}." if dd else "")
              + f" Sharpe {_r(d.get('sr_annual'), 2)} a year over 2023-26, DSR {_r(d.get('dsr'), 4)} (needs >= {Z.DSR_MIN}).")
        A("")
    A("## Options strategies at the new lots")
    A("")
    A("| Strategy | Lot rule (median lots) | DISCOVERY trades / PF / net | one lot: passed, PF, net | FINAL PF / net | "
      "FINAL | DSR | One-lot old-era verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for i in sorted(osc, key=lambda k: (not osc[k]["passed"], k)):
        r, s1 = osc[i], osc[i]["stored_1L"]
        if not (r["passed"] or s1["passed"]):
            continue
        f = (ofr.get(i) or {}).get("final") or {}
        d = drow.get(i, {})
        A(f"| `{i}` | {', '.join(r['rules'])} ({r['lots_median']}) | {r['disc']['trades']} / {_r(r['disc']['pf'], 3)} / "
          f"Rs {_r(r['disc']['net'])} | {'yes' if s1['passed'] else 'no'}, {_r(s1['disc']['pf'], 3)}, "
          f"Rs {_r(s1['disc']['net'])} | {_r(f.get('pf'), 3)} / Rs {_r(f.get('net'))} | "
          f"{('CONFIRMED' if f.get('confirmed') else 'rejected') if r['passed'] else 'not opened (failed DISCOVERY)'} | "
          f"{_r(d.get('dsr'), 4)} | {oldv.get(i, 'none on record')} |")
    A("")
    lost = sorted(i for i in osc if osc[i]["stored_1L"]["passed"] and not osc[i]["passed"])
    if lost:
        A(f"Passed DISCOVERY at one lot but not at the new lots: {', '.join('`' + i + '`' for i in lost)} "
          f"({'; '.join(', '.join(osc[i]['reasons']) for i in lost)}): the lot count now varies from trade to trade, "
          "which re-weights the weeks.")
        A("")
    A("An option strategy cannot be proven by this re-test: no old-era option record carries per-leg prices, so its "
      "out-of-sample years cannot be re-sized (pre-registered). The four FINAL-confirmed option strategies were already "
      "confirmed at one lot; all four sell option premium, and their DSR is far below 0.95.")
    ro = ofin.get("report_only", {})
    A("")
    A("## The live rules at the new size (report only; no verdict is re-opened)")
    A("")
    if live:
        A("| Rule | Size | DISCOVERY net / PF | FINAL net / PF | FINAL gross and fee per trade | 3-year net | Sharpe a year |")
        A("|---|---|---|---|---|---|---|")
        for i in ("AUC-FADE-2PCT", "AUC-FADE-3PCT", "SWING-ATTN-ON-3X"):
            for tag, name in (("at_1lakh", "Rs 1 lakh"), ("at_5lakh", "Rs 5 lakh")):
                v = live[i][tag]
                A(f"| {i} | {name} | Rs {_r(v['disc']['net'])} / {_r(v['disc']['pf'], 3)} | Rs {_r(v['final']['net'])} / "
                  f"{_r(v['final']['pf'], 3)} | Rs {_r(v['final_mean_gross'])} and Rs {_r(v['final_mean_cost'])} | "
                  f"Rs {_r(v['full']['net'])} | {_r(v['full']['sr_daily'] * 252 ** 0.5, 2)} |")
        A("")
        rep = {i: live[i]["reproduces_stored_1lakh"] for i in ("AUC-FADE-2PCT", "AUC-FADE-3PCT", "SWING-ATTN-ON-3X")}
        A(f"The Rs 1 lakh rows reproduce the stored corrected-data results: {rep}.")
    w = osc_all.get("VRP-STRANGLE-016-WINGED")
    if w and ro.get("VRP-STRANGLE-016-WINGED"):
        x = ro["VRP-STRANGLE-016-WINGED"]
        A(f"- VRP-STRANGLE-016-WINGED at the max-risk lot rule (median {w['lots_median']} lots; the lab used 2): DISCOVERY "
          f"PF {_r(w['disc']['pf'], 3)}, net Rs {_r(w['disc']['net'])} (2 lots: PF {_r(w['stored_1L']['disc']['pf'], 3)}, "
          f"Rs {_r(w['stored_1L']['disc']['net'])}); FINAL PF {_r(x['final']['pf'], 3)}, net Rs {_r(x['final']['net'])} "
          f"(2 lots: PF {_r(x['l1_final']['pf'], 3)}, Rs {_r(x['l1_final']['net'])}).")
    A("- EV-PEAD-LONG-H60 is delivery: identical at any size.")
    A("")
    A("## Types whose result stands unchanged")
    A("")
    for name, sel, e in (("Delivery swing", cnc, e_cnc), ("Index futures", idx, e_idx)):
        A(f"**{name}** ({len(sel)} strategies): {len(e[0])} pass DISCOVERY, {len(e[1])} confirmed on FINAL, {len(e[2])} "
          "confirmed on 2016-23.")
        for i in e[0]:
            s = stored[i]
            f = (rc_final.get(i, {}).get("final") or {})
            o = (rc_oldmin.get(i, {}).get("corrected") or {}).get("verdict", "not judged")
            per = "per Rs 10 lakh (10 positions of Rs 1 lakh)" if name.startswith("Delivery") else "per lot"
            A(f"- `{i}`: DISCOVERY {s['disc']['trades']} trades, PF {_r(s['disc']['pf'], 3)}, net Rs {_r(s['disc']['net'])} "
              f"at the lab size ({per}: the same PF); FINAL "
              f"{('PF ' + _r(f.get('pf'), 3) + ', ' + ('confirmed' if f.get('confirmed') else 'rejected')) if f else 'see ledger'}; "
              f"2016-23: {o}.")
        A("")
    dfc = Z._jload(os.path.join(Z.SCRATCH, "audit", "rerun", "dsr_final_catalogue.json"), {}) or {}
    for i in e_cnc[2]:
        if i in e_cnc[1] and i in dfc:
            A(f"`{i}` is the one catalogue rule that is confirmed on both the FINAL year and 2016-23 on corrected data; "
              f"its Sharpe on the FINAL year is {dfc[i]['sr_annual']} and its DSR {dfc[i]['dsr_final']} (needs 0.95), so it "
              "is not proven. Its size does not change, so nothing here changes that.")
            A("")
    A(f"**Timeframe grid**: {len(tf_pass)} of {len(tf)} combos passed stage 1 in Rs 10 lakh accounts; 2016-23 confirmed: "
      f"{', '.join('`' + k + '`' for k in tf_ok) or 'none'}; proven (with DSR): 0. See the tfgrid report.")
    A("")
    A("## Near misses (failures)")
    A("")
    near = sorted((i for i in ids if not scr[i]["passed"] and _both(scr[i]) and scr[i]["disc"]["pf"] >= 1.05),
                  key=lambda i: -scr[i]["disc"]["pf"])[:10]
    if near:
        A("| Strategy | Kind | Trades | PF | Net | Why it failed DISCOVERY at Rs 5 lakh |")
        A("|---|---|---|---|---|---|")
        for i in near:
            r = scr[i]
            A(f"| `{i}` | {REGISTRY[i].kind} | {r['disc']['trades']} | {_r(r['disc']['pf'], 3)} | Rs {_r(r['disc']['net'])} | "
              f"{'; '.join(r['reasons'])} |")
        A("")
    rej = [i for i in pass5 if i not in conf5]
    A(f"Passed DISCOVERY and rejected on the FINAL year: {', '.join('`' + i + '`' for i in rej) or 'none'}.")
    best_mis = sorted((i for i in mis if scr[i]["disc"]["trades"] >= 150), key=lambda i: -scr[i]["disc"]["pf"])[:3]
    A("Best intraday / scalping strategies at Rs 5 lakh (all fail): " + "; ".join(
        f"`{i}` PF {_r(scr[i]['disc']['pf'], 3)} ({', '.join(scr[i]['reasons'])})" for i in best_mis) + ".")
    A("")
    A("## Controls and checks")
    A("")
    if ver:
        A(f"- Before the plan was frozen, the size wrapper forced to Rs 1 lakh reproduced the stored DISCOVERY results of "
          f"{', '.join('`' + i + '`' for i in ver['ids'])} to the paisa: **{'yes' if ver['ok'] else 'NO'}**.")
    nt = len(ids) + len(osc)
    A(f"- In the main pass every strategy's original-size twin was recomputed next to its new-size result: "
      f"{nt - len(twin_bad)} of {nt} equal their stored results" + (f"; differing: {', '.join(twin_bad)}." if twin_bad else "."))
    if ctl:
        A(f"- Random-entry placebos at Rs 5 lakh: {len(ctl['random_passed'])} of {ctl['random_n']} pass DISCOVERY "
          f"(expected 0). Planted look-ahead edge at Rs 5 lakh passes: **{'yes' if ctl['planted_passed'] else 'NO'}**. "
          f"Controls gate: **{'met' if ctl_ok else 'NOT met'}**.")
    if dsr:
        A(f"- DSR: N = {dsr['N']} (= {dsr['N_before']} earlier trials + K {dsr['K']} + {dsr['final_looks']} FINAL looks), "
          f"V = {dsr['V']:.6f}; the luck bar is a Sharpe of {dsr['sr0_annual']:.2f} a year.")
        from . import trials as T
        vb = 0.0013                                   # the builder batches' V (their own DISCOVERY Sharpes), report only
        fc = [i for i in conf5 + f_opt[2] if i in drow]
        alt = {i: T.dsr(drow[i]["sr_daily"], drow[i]["n_days"], drow[i]["skew_daily"], drow[i]["kurt_daily"],
                        dsr["N"], vb)["dsr"] for i in fc}
        A(f"- The pre-registered V is large because most catalogue strategies trade often and lose their fees every day, "
          f"which spreads their Sharpes widely; with it no strategy here can reach DSR 0.95 (all are 0.00). Report only, "
          f"not a verdict: with the builder's much smaller V ({vb}) the FINAL-confirmed strategies' DSR would be "
          + ", ".join(f"`{i}` {_r(alt[i], 3)} (Sharpe {_r(drow[i]['sr_annual'], 2)})" for i in fc) + ".")
    if led or oled:
        A(f"- Trials ledger: {led.get('appended', 0) + oled.get('appended', 0)} rows appended as new trials (ids "
          f"`sz10l:<id>`); ledger N for 2023-26 {led.get('ledger_n_2023_26_before', '-')} -> "
          f"{oled.get('ledger_n_2023_26_after', led.get('ledger_n_2023_26_after', '-'))}. Backup: "
          f"`{os.path.basename(led.get('backup', ''))}`.")
    A("- Two code fixes after the plans were hashed, neither touching a rule or a gate: a missing catalogue import in "
      "the FINAL stage and a wrong read of the option simulator's detail records (both crashed before any result).")
    A("")
    A("## What this does and does not show")
    A("")
    A("- Same strategies, entries, exits, data and gates; only the quantity per trade changed, so only the fee per rupee "
      "(and, for options, the weight of each trade) changed.")
    A("- The FINAL year and the 2016-23 years were already looked at once at the old size. This is a second, strongly "
      "correlated look, counted in N; it is not an independent confirmation.")
    A("- The minute universe is today's 97 NIFTY-100 names (survivorship-biased), as in every earlier minute test.")
    A("- At Rs 5 lakh per trade the 2 bps-a-side slippage assumption is untested; option freeze quantities are not "
      "modelled; the option model prices of options_vrp / options_r3 are Black-Scholes values, not quotes.")
    txt = "\n".join(L) + "\n"
    with open(REPORT, "w") as fh:
        fh.write(txt)
    Z.log(f"report written: {REPORT}; proven {proven}; stock FINAL-confirmed {conf5}; OOS-confirmed {oos5}")
    return txt


if __name__ == "__main__":
    write()
