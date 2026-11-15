"""Cross-check the lab's bars against the official corporate actions -> corpactions/bar_adjustment_issues.csv
(the bar files are NOT modified).

Bars checked
  min_2016_2023   <scratch>/oldera_min/cache        (Kite 1-minute bars, old era)
  min_2023_2026   <scratch>/stratlab_cache/bars3y_* (Kite 1-minute bars, 3-year era)
  daily_2012_2023 nse_old/adj_factors.csv            (the factors applied to eq_daily_adj.csv.gz), all symbols

Method (minute bars).  Per symbol and session, s(d) = median of bar / official for the day's open, high and low
(official = NSE bhavcopy, unadjusted, the symbol in force that day; the three are the same prints in both sources,
the close is not: NSE's close is a 30-minute VWAP).  Bars back-adjusted by a factor f at an ex-date show s stepping
there:  applied(d) = median s over the 3 sessions before d / median s over d..d+2.
An official action whose ex-date is not a bar session is checked on the first bar session after it.
  split / bonus / consolidation   official factor known        -> applied must match within 3 %
  rights                          theoretical ex-rights factor  -> "unadjusted_rights" when not applied and factor < 0.98
  demerger / scheme               no official ratio: reference = the official overnight gap open(d) / close(d-1)
                                  "unadjusted_demerger": bars show the drop as a return; "wrong_factor_demerger": bars
                                  are adjusted by something else than the gap
  dividend                        "adjusted_for_dividend": s steps by 1 - dividend / price (bars back-adjusted for the
                                  dividend); "unadjusted_large_dividend": a dividend > 5 % of price left in the bars
  no official action              "adjusted_without_official_action": s jumps > 3 % in one day and the following
                                  3 sessions stay > 3 % away from the local fit bar = a x official + b of the previous
                                  20 sessions (an affine fit, so that additive offsets do not raise false alarms)
"additive_offset" (one row per symbol and era): on a 250-session window without any step the bars fit
a x official + b with |b| > 1 % of price, far better than a pure ratio: dividends were subtracted in rupees instead of
scaled, so every percentage return computed from those bars is off by about b / price.
Volume: vs(d) = bar volume / official volume; with a price factor f applied, vs before the ex-date should be 1 / f
times vs after it (5-session medians).  "unadjusted_volume" = price adjusted, volume not.
"era_scale_mismatch": the two minute sets are on different adjustment bases at their junction (2023-11-09 / 10).
Usage: python3 check_bar_adjust.py CORPACTIONS_DIR REPO_ALGOTRADER_DIR"""
import bisect, collections, csv, glob, gzip, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal, symmap

CA_DIR, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
from stratlab import data as D  # noqa: E402

S = cal.S
ERAS = [("min_2016_2023", os.path.join(S, "oldera_min", "cache")),
        ("min_2023_2026", glob.glob(os.path.join(S, "stratlab_cache", "bars3y_*"))[0])]
TOL_ONE = 0.015          # |applied - 1| below this = "not adjusted" for structural actions
TOL_MATCH = 0.03
COLS = ["source", "symbol", "date", "issue", "official_kinds", "official_factor", "dividend", "official_gap", "bars_gap",
        "applied_factor", "vol_scale_before", "vol_scale_after", "subject"]

ca = {}
for r in csv.DictReader(gzip.open(os.path.join(CA_DIR, "corpactions.csv.gz"), "rt")):
    ca.setdefault(r["lab_symbol"], []).append(r)


def official(sym, days):
    out, names = {}, {}
    for d in days:
        ym = d // 100
        if ym not in names:
            names[ym] = {symmap.symbol_on(sym, ym * 100 + 1), symmap.symbol_on(sym, ym * 100 + 28)}
        for nm in names[ym]:
            p = cal.lookup(nm, d)
            if p and p["close"] > 0 and p["open"] > 0:
                out[d] = p
                if nm == symmap.symbol_on(sym, d):
                    break
    return out


def row(**k):
    return {c: k.get(c, "") for c in COLS}


issues, stats, junction = [], {}, {}
for era, cdir in ERAS:
    syms = sorted(f[:-4] for f in os.listdir(cdir) if f.endswith(".npz") and not f.startswith("_"))
    n_events = n_ok = 0
    for sym in syms:
        sd = D.load_symbol(cdir, sym)
        if sd is None or sd.n_sessions < 30:
            continue
        dl = sd.daily()
        days = [int(x) for x in dl["date"]]
        off = official(sym, days)
        idx = [i for i, d in enumerate(days) if d in off]
        if len(idx) < 30:
            continue
        dd = [days[i] for i in idx]
        bo, bh, bl, bc, bv = (dl[k][idx] for k in ("o", "h", "l", "c", "v"))
        oo, oh, ol, oc, ov = (np.array([off[d][k] for d in dd]) for k in ("open", "high", "low", "close", "volume"))
        s = np.median(np.vstack([bo / oo, bh / oh, bl / ol]), axis=0)
        ls = np.log(s)
        vs = np.where(ov > 0, bv / np.maximum(ov, 1), np.nan)
        n = len(dd)
        junction.setdefault(sym, {})[era] = (dd[0], float(np.median(s[:3])), dd[-1], float(np.median(s[-3:])))

        def step(j, w=3):
            return float(np.exp(np.median(ls[max(0, j - w):j]) - np.median(ls[j:j + w])))

        # official events -> first bar session on/after the ex-date
        ev = collections.defaultdict(list)
        for e in ca.get(sym, []):
            ex = int(e["ex_date"])
            if ex < dd[0] or ex > dd[-1]:
                continue
            j = bisect.bisect_left(dd, ex)
            if 1 <= j < n:
                ev[j].append(e)
        # steps without an official action (affine-aware)
        spur = {}
        for g in range(21, n - 2):
            if abs(ls[g - 1] - ls[g]) <= 0.03:
                continue
            x = np.concatenate([oo[g - 20:g], oh[g - 20:g], ol[g - 20:g]])
            y = np.concatenate([bo[g - 20:g], bh[g - 20:g], bl[g - 20:g]])
            A = np.vstack([x, np.ones(len(x))]).T
            (fa, fb), *_ = np.linalg.lstsq(A, y, rcond=None)
            if float(np.ptp(x)) / float(np.median(x)) < 0.03:        # no price variation: use the plain ratio
                fa, fb = float(np.median(y / x)), 0.0
            xa = np.concatenate([oo[g:g + 3], oh[g:g + 3], ol[g:g + 3]])
            ya = np.concatenate([bo[g:g + 3], bh[g:g + 3], bl[g:g + 3]])
            ratio = float(np.median(ya / (fa * xa + fb)))
            if abs(ratio - 1) > 0.03:
                spur[g] = 1.0 / ratio
        struct_days = set()
        for j in sorted(set(ev) | set(spur)):
            d = dd[j]
            evs = ev.get(j, [])
            applied = step(j)
            r_off = float(oo[j] / oc[j - 1])
            r_bar = float(bo[j] / bc[j - 1])
            f_known, kinds, div = 1.0, [], 0.0
            for e in evs:
                kinds.append(e["kind"])
                if e["kind"] in ("bonus", "split", "consolidation", "rights") and e["factor"]:
                    f_known *= float(e["factor"])
                if e["kind"] == "dividend" and e["dividend"]:
                    div += float(e["dividend"])
            f_div = 1.0 - div / oc[j - 1] if div else 1.0
            structural = [k for k in kinds if k in ("bonus", "split", "consolidation")]
            has_dem = "demerger" in kinds or "capital_reduction" in kinds
            rights = "rights" in kinds
            adjusted = abs(applied - 1.0) >= TOL_ONE
            issue = None
            if structural or has_dem or rights:
                n_events += 1
                struct_days.add(j)
            if structural and abs(f_known - 1) > 0.02:
                if not adjusted:
                    issue = "unadjusted_price"
                elif abs(applied / f_known - 1) < TOL_MATCH or (has_dem and abs(applied / r_off - 1) < 0.05):
                    n_ok += 1
                else:
                    issue = "wrong_factor"
            elif has_dem:
                if not adjusted:
                    if r_off < 0.95:
                        issue = "unadjusted_demerger"
                    else:
                        n_ok += 1
                elif abs(applied / r_off - 1) < 0.05:
                    n_ok += 1
                else:
                    issue = "wrong_factor_demerger"
            elif rights:
                if not adjusted and f_known < 0.98:
                    issue = "unadjusted_rights"
                elif adjusted and abs(applied / f_known - 1) >= TOL_MATCH:
                    issue = "wrong_factor"
                else:
                    n_ok += 1
            elif div:
                one = 1 - float(s[j - 1] / s[j])
                if (1 - f_div) > 0.004 and abs(one / (1 - f_div) - 1) < 0.25:
                    issue = "adjusted_for_dividend"
                elif f_div < 0.95 and abs(one) < 0.004:
                    issue = "unadjusted_large_dividend"
                elif j in spur:
                    issue = "adjusted_without_official_action"
            elif j in spur:
                issue = "adjusted_without_official_action"
                applied = spur[j]
            vb = np.nanmedian(vs[max(0, j - 5):j])
            va = np.nanmedian(vs[j:j + 5])
            vstep = float(vb / va) if va == va and vb == vb and va > 0 else float("nan")
            vissue = None
            if adjusted and issue is None and vstep == vstep and abs(applied - 1) > 0.05:
                if abs(vstep * applied - 1) > 0.25 and abs(vstep - 1) < 0.25:
                    vissue = "unadjusted_volume"
            for iss in (issue, vissue):
                if iss:
                    issues.append(row(source=era, symbol=sym, date=d, issue=iss, official_kinds="+".join(kinds),
                                      official_factor=round(f_known, 6) if (structural or rights) else "",
                                      dividend=div or "", official_gap=round(r_off, 5), bars_gap=round(r_bar, 5),
                                      applied_factor=round(applied, 5),
                                      vol_scale_before=round(float(vb), 4) if vb == vb else "",
                                      vol_scale_after=round(float(va), 4) if va == va else "",
                                      subject=" || ".join(e["subject"] for e in evs)[:200]))
        # additive offset on a clean 250-session window
        dirty = sorted(struct_days | set(spur) | {g for g in range(1, n) if abs(ls[g - 1] - ls[g]) > 0.02})
        k = 250
        for a0 in range(0, max(1, n - k + 1), 60):
            if any(a0 < g < a0 + k for g in dirty) or a0 + k > n:
                continue
            x, y = oo[a0:a0 + k], bo[a0:a0 + k]
            if float(np.ptp(np.log(x))) < 0.15:
                continue
            A = np.vstack([x, np.ones(k)]).T
            (fa, fb), *_ = np.linalg.lstsq(A, y, rcond=None)
            res_aff = float(np.std(y - (fa * x + fb)))
            fp = float(np.sum(y * x) / np.sum(x * x))
            res_prop = float(np.std(y - fp * x))
            medp = float(np.median(y))
            if abs(fb) / medp > 0.01 and res_aff < 0.5 * res_prop:
                issues.append(row(source=era, symbol=sym, date=dd[a0], issue="additive_offset",
                                  applied_factor=round(float(fa), 5),
                                  subject=f"{dd[a0]}..{dd[a0 + k - 1]}: bar = {fa:.4f} x official {fb:+.2f} (offset "
                                          f"{100 * fb / medp:+.1f} % of the median bar price; residual sd {res_aff:.3f} "
                                          f"vs {res_prop:.3f} for a pure ratio)"))
            break
    stats[era] = {"symbols": len(syms), "structural_rights_demerger_events_checked": n_events, "consistent": n_ok}

for sym, j in junction.items():
    if len(j) == 2:
        a, b = j["min_2016_2023"], j["min_2023_2026"]
        if abs(a[3] / b[1] - 1) > TOL_ONE:
            issues.append(row(source="junction_2023-11", symbol=sym, date=b[0], issue="era_scale_mismatch",
                              applied_factor=round(a[3] / b[1], 5),
                              subject=f"bar/official {a[3]:.4f} at {a[2]} (old era) vs {b[1]:.4f} at {b[0]} (3-year era)"))

# daily_2012_2023: factors applied to eq_daily_adj vs the official list (all symbols)
adj = list(csv.DictReader(open(os.path.join(cal.OLD, "adj_factors.csv"))))
adj_by = collections.defaultdict(list)
for r in adj:
    adj_by[(r["symbol"], int(r["ex_date"]))].append(r)
f_off, kinds_off = {}, collections.defaultdict(list)
for lab, evs in ca.items():
    for e in evs:
        ex = int(e["ex_date"])
        if 20120101 <= ex <= 20231109 and e["kind"] in ("bonus", "split", "consolidation") and e["factor"]:
            k = (e["symbol_then"], ex)
            f_off[k] = f_off.get(k, 1.0) * float(e["factor"])
            kinds_off[k].append(e["kind"])
n_daily = n_daily_ok = n_not_trading = 0
for k in sorted(f_off):
    n_daily += 1
    rows_a = [r for r in adj_by.get(k, []) if r["status"].startswith("applied") and r["factor"]]
    px = cal.lookup(k[0], k[1])
    if not rows_a:
        if px is None:
            n_not_trading += 1
            continue
        gap = px["open"] / px["prev_close"] if px["prev_close"] else float("nan")
        note = "; ".join(r["status"] + ": " + r["purpose"] for r in adj_by.get(k, [])) or "no row in adj_factors.csv"
        issues.append(row(source="daily_2012_2023", symbol=k[0], date=k[1], issue="official_action_not_applied",
                          official_kinds="+".join(kinds_off[k]), official_factor=round(f_off[k], 6),
                          official_gap=round(gap, 5), applied_factor=1.0, subject=note[:200]))
        continue
    fa = 1.0
    for r in rows_a:
        fa *= float(r["factor"])
    if abs(fa / f_off[k] - 1) > TOL_MATCH:
        issues.append(row(source="daily_2012_2023", symbol=k[0], date=k[1], issue="wrong_factor",
                          official_kinds="+".join(kinds_off[k]), official_factor=round(f_off[k], 6),
                          official_gap=rows_a[0].get("gap_open_ratio", ""), applied_factor=round(fa, 6),
                          subject="; ".join(r["status"] + ": " + r["purpose"] for r in rows_a)[:200]))
    else:
        n_daily_ok += 1
stats["daily_2012_2023"] = {"official_split_bonus_consolidation_events": n_daily,
                            "applied_with_matching_factor": n_daily_ok, "symbol_not_trading_on_ex_date": n_not_trading}

issues.sort(key=lambda r: (r["source"], r["symbol"], r["date"], r["issue"]))
with open(os.path.join(CA_DIR, "bar_adjustment_issues.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader(); w.writerows(issues)
by = collections.Counter((r["source"], r["issue"]) for r in issues)
stats["issues"] = {f"{a}:{b}": c for (a, b), c in sorted(by.items())}
json.dump(stats, open(os.path.join(CA_DIR, "_bar_adjust_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
