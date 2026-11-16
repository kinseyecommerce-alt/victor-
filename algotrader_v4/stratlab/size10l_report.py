"""stratlab/size10l_report.py — writes stratlab/SIZE10L.md from <scratchpad>/size10l (size10l_run.py's outputs).

    cd algotrader_v4 && python -m stratlab.size10l_run --stage report
"""
from __future__ import annotations

import os
import statistics
import sys

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


def write() -> str:
    sys.path.insert(0, os.path.dirname(HERE))
    from costs import round_trip_cost
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    J = lambda f, d=None: Z._jload(os.path.join(Z.OUT, f), d)        # noqa: E731
    pre = Z.load_prereg()
    sha = open(Z.PREREG + ".sha256").read().strip()
    ver, ctl, scr = J("verify_1lakh.json"), J("controls.json"), J("screen.json")
    fin = J("final.json", {}) or {}
    om, od, dsr, live, led = J("oldera_min.json", {}) or {}, J("oldera_daily.json", {}) or {}, J("dsr.json", {}) or {}, \
        J("live.json", {}) or {}, J("ledger.json", {}) or {}
    rc_final = (Z._jload(os.path.join(Z.SCRATCH, "audit", "rerun", "final_corrected.json"), {}) or {}).get("results", {})
    ids = pre["scope"]["retested_ids"]
    K = len(ids)
    un = pre["scope"]["unchanged_catalogue"]
    fr = fin.get("results", {})
    pass5 = sorted(i for i in ids if scr[i]["passed"])
    pass1 = sorted(i for i in ids if scr[i]["stored_1L"]["passed"])
    conf5 = sorted(i for i in pass5 if fr.get(i, {}).get("final", {}).get("confirmed"))
    conf1 = sorted(i for i in pass1 if rc_final.get(i, {}).get("final", {}).get("confirmed"))
    oos5 = sorted(i for i in conf5 if om.get(i, {}).get("size10l", {}).get("verdict") == "OOS-CONFIRMED")
    drow = dsr.get("rows", {})
    ctl_ok = bool(ctl and ctl.get("ok"))
    twin_bad = sorted(i for i in ids if not scr[i].get("l1_matches_stored", True))
    proven = sorted(i for i in oos5 if (drow.get(i, {}).get("dsr") or 0) >= Z.DSR_MIN and ctl_ok and i not in twin_bad)
    L = []
    A = L.append
    A("# Old lab catalogue re-tested at the Rs 10 lakh-per-agent trade size")
    A("")
    A(f"Pre-registered before any Rs-5-lakh result: `stratlab/PREREGISTERED_SIZE10L.json` (sha256 `{sha}`, written "
      f"{pre['written']}). Code: `stratlab/size10l_run.py`. Research only; nothing here changes a live rule.")
    A("")
    A("## Result")
    A("")
    A(f"- **Proven at the new size: {len(proven)}**" + (f" ({', '.join(proven)})" if proven else "") + ".")
    A(f"- Re-tested: **K = {K}** catalogue strategies whose trades are booked at intraday (MIS) or stock-futures costs "
      f"({', '.join(f'{v} {k}' for k, v in sorted(pre['scope']['retested_by_kind'].items()))}), at Rs 5 lakh per trade "
      "instead of Rs 1 lakh.")
    A(f"- Not re-run, not counted again (size unchanged, earlier verdicts stand): {un['count']} catalogue strategies "
      f"({', '.join(f'{v} {k}' for k, v in sorted(un['by_kind'].items()))}) and all 668 tfgrid combos. tfgrid already "
      "traded every combo (and all 608 catalogue strategies with the 1:3 bracket) from a Rs 10 lakh account per agent: "
      "its median fee per intraday stock trade was Rs 465 and per scalping trade Rs 896, i.e. trades of Rs 10-25 lakh. "
      "Its result (0 proven of 1,276) stands.")
    A("")
    A("| Stage | at Rs 1 lakh (stored, same K ids) | at Rs 5 lakh (this re-test) |")
    A("|---|---|---|")
    A(f"| Net > 0 in both halves of DISCOVERY | {sum(1 for i in ids if _both(scr[i]['stored_1L']))} | "
      f"{sum(1 for i in ids if _both(scr[i]))} |")
    A(f"| DISCOVERY gates passed (2023-11-10 .. 2025-12-16) | {len(pass1)} | {len(pass5)} |")
    A(f"| FINAL confirmed (2025-12-17 .. 2026-11-09, one look) | {len(conf1)} | {len(conf5)} |")
    A(f"| 2016-23 minute OOS confirmed | see the corrected re-run | {len(oos5)} |")
    A(f"| DSR >= 0.95 (N = {dsr.get('N', '-')}) = proven | - | {len(proven)} |")
    A("")
    lab_bar = oos5
    A(f"The lab's earlier bar for a catalogue strategy (FINAL confirmed + 2016-23 OOS confirmed, no DSR gate): "
      f"**{len(lab_bar)}** at the new size" + (f" ({', '.join(lab_bar)})" if lab_bar else "") + ".")
    A("")
    A("## Fees: what the size changes")
    A("")
    A("| Product | Round trip at Rs 1 lakh | per Rs 1 lakh at Rs 5 lakh | saving per Rs 1 lakh traded |")
    A("|---|---|---|---|")
    for kind, name in (("equity_mis", "Intraday (MIS)"), ("future", "Stock at futures costs")):
        c1, c5 = round_trip_cost(1e5, 1e5, kind), round_trip_cost(5e5, 5e5, kind) / 5
        A(f"| {name} | Rs {c1:.2f} ({c1 / 10:.2f} bps) | Rs {c5:.2f} ({c5 / 10:.2f} bps) | Rs {c1 - c5:.2f} |")
    A("")
    A("Slippage (2 bps a side, about Rs 40 per Rs 1 lakh round trip) does not shrink with size and is unchanged.")
    med = lambda xs: statistics.median(xs) if xs else float("nan")       # noqa: E731
    for kinds, name in ((("intraday", "scalping"), "intraday + scalping"), (("futures",), "stock futures")):
        sel = [i for i in ids if REGISTRY[i].kind in kinds and scr[i]["disc"]["trades"] > 0]
        g1 = med([scr[i]["stored_1L"]["disc"]["gross"] / scr[i]["stored_1L"]["disc"]["trades"] for i in sel])
        c1 = med([scr[i]["stored_1L"]["disc"]["cost"] / scr[i]["stored_1L"]["disc"]["trades"] for i in sel])
        g5 = med([scr[i]["disc"]["gross"] / scr[i]["disc"]["trades"] / 5 for i in sel])
        c5 = med([scr[i]["disc"]["cost"] / scr[i]["disc"]["trades"] / 5 for i in sel])
        A(f"- Median {name} strategy ({len(sel)} with trades), per Rs 1 lakh traded: gross after slippage Rs {g1:.0f} "
          f"and fees Rs {c1:.0f} at Rs 1 lakh; gross Rs {g5:.0f} and fees Rs {c5:.0f} at Rs 5 lakh.")
    A("")
    A("## Every DISCOVERY passer at the new size")
    A("")
    if not pass5:
        A("None.")
    else:
        A("| Strategy | Kind | DISCOVERY trades / PF / net | at Rs 1 lakh: passed, PF, net | FINAL trades / PF / net | "
          "FINAL | 2016-23 OOS | DSR | Net per year (2023-26) |")
        A("|---|---|---|---|---|---|---|---|---|")
        for i in pass5:
            r, s1 = scr[i], scr[i]["stored_1L"]
            f = (fr.get(i) or {}).get("final") or {}
            o = (om.get(i) or {}).get("size10l") or {}
            d = drow.get(i, {})
            tot = r["disc"]["net"] + (f.get("net") or 0.0)
            A(f"| `{i}` | {REGISTRY[i].kind} | {r['disc']['trades']} / {_r(r['disc']['pf'], 3)} / Rs {_r(r['disc']['net'])} | "
              f"{'yes' if s1['passed'] else 'no'}, {_r(s1['disc']['pf'], 3)}, Rs {_r(s1['disc']['net'])} | "
              f"{f.get('trades', '-')} / {_r(f.get('pf'), 3)} / Rs {_r(f.get('net'))} | "
              f"{'CONFIRMED' if f.get('confirmed') else 'rejected'} | {o.get('verdict', 'not run (FINAL rejected)')} | "
              f"{_r(d.get('dsr'), 4)} | Rs {_r(tot / YEARS)} |")
        A("")
        A("Net per year = (DISCOVERY + FINAL net) / 3 years at Rs 5 lakh per trade; the number of concurrent positions "
          "is not capped by the Rs 10 lakh account in this test (tfgrid is the account-level test).")
        A("")
        for i in pass5:
            A(f"- `{i}`: {REGISTRY[i].desc}")
    A("")
    if conf5:
        A("### FINAL-confirmed strategies in detail")
        A("")
        for i in conf5:
            f, o, d = fr[i]["final"], (om.get(i) or {}).get("size10l") or {}, drow.get(i, {})
            e = o.get("era") or {}
            dd = od.get(i) or {}
            A(f"- `{i}` ({REGISTRY[i].desc}). FINAL: {f['trades']} trades, PF {_r(f['pf'], 3)}, net Rs {_r(f['net'])} "
              f"(at Rs 1 lakh the same trades: PF {_r(fr[i]['l1_final']['pf'], 3)}, net Rs {_r(fr[i]['l1_final']['net'])}). "
              f"2016-23 minute: {o.get('verdict')} - {o.get('why')} (era {e.get('trades')} trades, PF {_r(e.get('pf'), 3)}, "
              f"net Rs {_r(e.get('net'))})." + (f" 2012-23 daily (report): {dd.get('verdict')} - {dd.get('why')}." if dd else "")
              + f" Full-sample Sharpe {_r(d.get('sr_annual'), 2)} a year, DSR {_r(d.get('dsr'), 4)} "
              f"(needs >= {Z.DSR_MIN}); with the Rs 1 lakh ledger's V (report only) {_r(d.get('dsr_ledger_v_report_only'), 4)}.")
        A("")
    A("## Near misses (failures)")
    A("")
    near = sorted((i for i in ids if not scr[i]["passed"] and _both(scr[i]) and scr[i]["disc"]["pf"] >= 1.05),
                  key=lambda i: -scr[i]["disc"]["pf"])[:12]
    if near:
        A("Failed the DISCOVERY gates at Rs 5 lakh although net positive in both halves (largest PF first):")
        A("")
        A("| Strategy | Trades | PF | Net | Why it failed |")
        A("|---|---|---|---|---|")
        for i in near:
            r = scr[i]
            A(f"| `{i}` | {r['disc']['trades']} | {_r(r['disc']['pf'], 3)} | Rs {_r(r['disc']['net'])} | {'; '.join(r['reasons'])} |")
    else:
        A("No strategy was net positive in both halves with PF >= 1.05 and still failed the gates.")
    rej = [i for i in pass5 if i not in conf5]
    if rej:
        A("")
        A(f"Passed DISCOVERY and were rejected on the FINAL look: {', '.join('`' + i + '`' for i in rej)}.")
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
        A(f"The Rs 1 lakh rows reproduce the stored corrected-data results: {rep}. "
          "VRP-STRANGLE-016-WINGED (option lots) and EV-PEAD-LONG-H60 (delivery) do not change size.")
    else:
        A("Not run.")
    A("")
    A("## Controls and checks")
    A("")
    if ver:
        A(f"- Method check before the plan was frozen: the size wrapper forced to Rs 1 lakh reproduced the stored "
          f"DISCOVERY results of {', '.join('`' + i + '`' for i in ver['ids'])} exactly (trades, gross, fees, net in each "
          f"half): **{'yes' if ver['ok'] else 'NO'}**.")
    A(f"- In the main pass every strategy's Rs-1-lakh twin was recomputed next to its Rs-5-lakh result: "
      f"{K - len(twin_bad)} of {K} equal their stored rows to the paisa" +
      (f"; differing: {', '.join(twin_bad)} (cannot be proven)." if twin_bad else "."))
    if ctl:
        A(f"- Random-entry placebos at Rs 5 lakh: {len(ctl['random_passed'])} of {ctl['random_n']} pass the DISCOVERY "
          f"gates (expected 0). Planted look-ahead edge at Rs 5 lakh passes: **{'yes' if ctl['planted_passed'] else 'NO'}**. "
          f"Controls gate: **{'met' if ctl_ok else 'NOT met'}**.")
    if dsr:
        A(f"- DSR: N = {dsr['N']} (= {dsr['N_before']} before + K {dsr['K']} + {dsr['final_looks']} FINAL looks), "
          f"V = {dsr['V']:.6f}; the luck bar is a Sharpe of {dsr['sr0_annual']:.2f} a year.")
    if led:
        A(f"- Trials ledger: {led['appended']} rows appended as new trials (ids `sz10l:<id>`); ledger N for 2023-26 "
          f"{led['ledger_n_2023_26_before']} -> {led['ledger_n_2023_26_after']}. Backup: `{os.path.basename(led['backup'])}`.")
    A("")
    A("## What this does and does not show")
    A("")
    A("- Same strategies, same entries and exits, same data and gates; only the quantity per trade changed, so only "
      "the fee per rupee changed.")
    A("- The FINAL and 2016-23 windows were already looked at once at Rs 1 lakh. This is a second, strongly correlated "
      "look, counted in N; it is not an independent confirmation.")
    A("- The minute universe is today's 97 NIFTY-100 names (survivorship-biased), as in every earlier minute test.")
    A("- At Rs 5 lakh per trade the 2 bps-a-side slippage assumption is untested; paper trading measures it.")
    txt = "\n".join(L) + "\n"
    with open(REPORT, "w") as fh:
        fh.write(txt)
    Z.log(f"report written: {REPORT}; proven {proven}; lab bar {lab_bar}")
    return txt


if __name__ == "__main__":
    write()
