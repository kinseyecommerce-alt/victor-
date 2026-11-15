"""Data check of corpactions/corpactions.csv.gz -> corpactions/data_check.json
Usage: python3 check_corpactions.py CORPACTIONS_DIR"""
import bisect, collections, csv, gzip, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal

D = sys.argv[1]
rows = list(csv.DictReader(gzip.open(os.path.join(D, "corpactions.csv.gz"), "rt")))
for r in rows:
    r["ex_date"] = int(r["ex_date"]); r["known_date"] = int(r["known_date"])
ses = cal.sessions()
sset = set(ses)
build = json.load(open(os.path.join(D, "_build.json")))
out = {"file": "corpactions.csv.gz", "rows": len(rows), "first_ex_date": rows[0]["ex_date"], "last_ex_date": rows[-1]["ex_date"],
       "source": "NSE corporate-actions feed (www.nseindia.com/api/corporates-corporateActions, equities, by ex-date "
                 "quarter) + NSE daily PR zip 'Bc' files for the first-listed date",
       "availability": "known_date = first session whose Bc file lists the (symbol, ex-date), else the session before "
                       "the ex-date; a row is visible when known_date < before",
       "build": build}
per = collections.defaultdict(collections.Counter)
for r in rows:
    per[r["kind"]][r["ex_date"] // 10000] += 1
out["rows_per_year"] = {k: dict(sorted(v.items())) for k, v in sorted(per.items())}
out["lab_rows"] = dict(collections.Counter(r["kind"] for r in rows if r["in_lab"] == "1"))
out["ex_dates_not_in_session_calendar"] = sorted({r["ex_date"] for r in rows if r["ex_date"] not in sset and r["ex_date"] <= ses[-1]})[:60]
out["ex_dates_after_last_session"] = sum(1 for r in rows if r["ex_date"] > ses[-1])
key = collections.Counter((r["symbol"], r["ex_date"], r["kind"]) for r in rows)
out["duplicate_keys"] = sum(1 for v in key.values() if v > 1)
out["rows_merging_several_feed_lines"] = sum(1 for r in rows if "||" in r["subject"])
# first-listed coverage and lead time
fl = collections.defaultdict(lambda: [0, 0])
lead = []
for r in rows:
    y = r["ex_date"] // 10000
    fl[y][1] += 1
    if r["first_listed"]:
        fl[y][0] += 1
        a = bisect.bisect_left(ses, int(r["first_listed"])); b = bisect.bisect_left(ses, r["ex_date"])
        if r["ex_date"] <= ses[-1]:
            lead.append(b - a)
out["first_listed_coverage_per_year"] = {str(y): f"{a}/{b}" for y, (a, b) in sorted(fl.items())}
lead.sort()
out["sessions_between_first_listing_and_ex_date"] = {"n": len(lead), "median": lead[len(lead) // 2], "p10": lead[len(lead) // 10],
                                                     "p90": lead[9 * len(lead) // 10], "listed_on_or_after_ex_date": sum(1 for x in lead if x <= 0)}
out["missing_factor"] = {k: sum(1 for r in rows if r["kind"] == k and not r["factor"]) for k in ("bonus", "split", "consolidation", "rights")}
out["dividend_rows_without_amount"] = sum(1 for r in rows if r["kind"] == "dividend" and not r["dividend"])
# second source 1: Bc purposes vs feed kinds, both directions (split / bonus)
bc = list(csv.DictReader(gzip.open(os.path.join(D, "_bc_keys.csv.gz"), "rt")))
bc_struct = {(r["symbol"], int(r["ex_date"])) for r in bc if re.search(r"bon|spl|sub-?div|split", r["purposes"], re.I)
             and 20120110 <= int(r["ex_date"]) <= ses[-1]}
api_struct = {(r["symbol_then"], r["ex_date"]) for r in rows if r["kind"] in ("bonus", "split") and 20120110 <= r["ex_date"] <= ses[-1]}
api_any = {(r["symbol_then"], r["ex_date"]) for r in rows} | {(r["symbol"], r["ex_date"]) for r in rows}
only_bc = sorted(bc_struct - api_struct - {k for k in bc_struct if k in api_any})
bcp = {(r["symbol"], int(r["ex_date"])): r["purposes"] for r in bc}
out["bonus_split_vs_bc_files"] = {
    "feed_bonus_split_keys": len(api_struct), "also_in_bc": len(api_struct & bc_struct) + sum(1 for k in api_struct - bc_struct if k in bcp),
    "feed_only": len([k for k in api_struct if k not in bcp]),
    "bc_bonus_split_keys": len(bc_struct), "bc_only_not_in_feed": len(only_bc),
    "bc_only_examples": [[k[0], k[1], bcp[k][:80]] for k in only_bc[:15]],
    "note": "bc_only rows are mostly non-EQ series (the feed is filtered to EQ/BE/BZ) or actions later withdrawn"}
# second source 2: official prices - the overnight gap on the ex-date must equal the factor
ok = tot = 0
bad = []
np = __import__("numpy")
prod = collections.OrderedDict()
for r in rows:
    if r["kind"] in ("bonus", "split", "consolidation") and r["factor"] and r["ex_date"] <= ses[-1]:
        k = (r["symbol_then"], r["ex_date"])
        e = prod.setdefault(k, [1.0, []])
        e[0] *= float(r["factor"]); e[1].append(r["subject"][:70])
others = {(r["symbol_then"], r["ex_date"]) for r in rows if r["kind"] in ("demerger", "rights", "capital_reduction")}
for (sym, ex), (f, subj) in prod.items():
    s = cal.series(sym)
    if s is None:
        continue
    j = int(np.searchsorted(s["date"], ex))
    if j <= 0 or j >= len(s["date"]) or int(s["date"][j]) != ex:
        continue
    gap = float(s["open"][j] / s["close"][j - 1])
    tot += 1
    if abs(gap / f - 1) < 0.2:
        ok += 1
    elif len(bad) < 25:
        bad.append({"symbol": sym, "ex_date": ex, "factor_product": round(f, 5), "official_gap": round(gap, 4),
                    "other_action_same_day": (sym, ex) in others, "subject": " || ".join(sorted(set(subj)))[:120]})
out["factor_vs_official_overnight_gap"] = {"split_bonus_ex_dates_with_prices": tot, "gap_within_20pct_of_factor": ok,
                                           "outside": tot - ok, "outside_examples": bad,
                                           "note": "outside rows carry another action on the same ex-date (bonus + split, demerger), "
                                                   "an ex-date the stock did not trade, or a feed ratio that the price contradicts"}
# hand spot checks: 10 lab-symbol actions vs the old-era agent's independently derived factors and the official gap
adj = {(r["symbol"], int(r["ex_date"])): r for r in csv.DictReader(open(os.path.join(cal.OLD, "adj_factors.csv")))}
random.seed(3)
cand = [r for r in rows if r["in_lab"] == "1" and r["kind"] in ("bonus", "split") and r["factor"] and r["ex_date"] <= ses[-1]]
spots = []
for r in random.sample(cand, 10):
    s = cal.series(r["symbol_then"])
    gap = None
    if s is not None:
        j = int(np.searchsorted(s["date"], r["ex_date"]))
        if 0 < j < len(s["date"]) and int(s["date"][j]) == r["ex_date"]:
            gap = round(float(s["open"][j] / s["close"][j - 1]), 4)
    a = adj.get((r["symbol_then"], r["ex_date"]))
    spots.append({"symbol": r["symbol"], "symbol_then": r["symbol_then"], "ex_date": r["ex_date"], "kind": r["kind"],
                  "factor": float(r["factor"]), "official_open_over_prev_close": gap,
                  "old_era_adj_factors_csv": (a["factor"] + " (" + a["status"] + ")") if a else "n/a (after 2023-11-09 or not in file)",
                  "bc_file_purpose": bcp.get((r["symbol_then"], r["ex_date"]), "")[:80], "first_listed": r["first_listed"],
                  "subject": r["subject"][:80]})
out["spot_checks"] = spots
p = os.path.join(D, "_bar_adjust_stats.json")
if os.path.exists(p):
    out["bar_adjustment_cross_check"] = json.load(open(p))
    out["bar_adjustment_cross_check"]["file"] = "bar_adjustment_issues.csv"
json.dump(out, open(os.path.join(D, "data_check.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("spot_checks", "bar_adjustment_cross_check", "rows_per_year")}, indent=1)[:4500])
for s in spots:
    print(s)
