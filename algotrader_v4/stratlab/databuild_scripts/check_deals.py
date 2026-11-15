"""Data check of deals/deals.csv.gz -> deals/data_check.json
Usage: python3 -I check_deals.py DEALS_DIR"""
import csv, gzip, json, os, random, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal

D = sys.argv[1]
rows = list(csv.DictReader(gzip.open(os.path.join(D, "deals.csv.gz"), "rt")))
for r in rows:
    r["date"] = int(r["date"]); r["qty"] = int(r["qty"]); r["price"] = float(r["price"])
ses = cal.sessions()
sset = set(ses)
out = {"file": "deals.csv.gz", "rows": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
       "source": "NSE historical report 'Bulk / Block deals' (www.nseindia.com/api/historicalOR/bulk-block-short-deals, "
                 "csv=true), one request per calendar quarter and kind",
       "availability": "published after the close of the trade date -> usable from the next session's open "
                       "(accessors return rows dated strictly before `before`)"}
per = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    per[r["kind"]][r["date"] // 10000] += 1
out["rows_per_year"] = {k: dict(sorted(v.items())) for k, v in per.items()}
out["lab_rows"] = {k: sum(1 for r in rows if r["kind"] == k and r["in_lab"] == "1") for k in ("bulk", "block")}
out["lab_symbols_with_rows"] = len({r["lab_symbol"] for r in rows if r["in_lab"] == "1"})
# calendar
dd = {k: {r["date"] for r in rows if r["kind"] == k} for k in ("bulk", "block")}
last = rows[-1]["date"]
in_range = [s for s in ses if rows[0]["date"] <= s <= last]
out["dates_not_in_session_calendar"] = {k: sorted(d for d in v if d not in sset and d <= ses[-1]) for k, v in dd.items()}
miss_bulk = [s for s in in_range if s not in dd["bulk"]]
out["sessions_in_range"] = len(in_range)
out["sessions_without_any_bulk_deal"] = {"count": len(miss_bulk), "per_year": dict(sorted(collections.Counter(
    s // 10000 for s in miss_bulk).items())), "dates": miss_bulk}
out["sessions_with_block_deals"] = len(dd["block"] & sset)
out["note_missing"] = ("a session without a bulk deal is possible (deals are event-driven); in 2012-2016 a few such "
                       "sessions per year are expected. Block deals occur on a minority of sessions by nature.")
# duplicates
key = collections.Counter((r["date"], r["kind"], r["symbol"], r["client"], r["side"], r["qty"], r["price"]) for r in rows)
dups = {k: v for k, v in key.items() if v > 1}
out["exact_duplicate_rows"] = {"groups": len(dups), "extra_rows": sum(v - 1 for v in dups.values()),
                               "examples": [list(k) + [v] for k, v in list(dups.items())[:5]],
                               "note": "kept as published: NSE lists one row per reported deal, and two identical "
                                       "block trades (same client, qty, price) on a day are legitimate"}
# price sanity vs the official unadjusted daily range of the same symbol and date
n = ok = nopx = 0
viol = []
for r in rows:
    p = cal.lookup(r["symbol"], r["date"])
    if p is None:
        nopx += 1; continue
    n += 1
    if p["low"] * 0.999 - 0.05 <= r["price"] <= p["high"] * 1.001 + 0.05:
        ok += 1
    else:
        viol.append(r)
out["price_vs_daily_range"] = {
    "rows_with_official_daily_bar": n, "inside_low_high": ok, "outside": n - ok, "no_daily_bar": nopx,
    "outside_by_kind": dict(collections.Counter(r["kind"] for r in viol)),
    "outside_examples": [{k: r[k] for k in ("date", "kind", "symbol", "client", "side", "qty", "price")} | {
        "day": cal.lookup(r["symbol"], r["date"])} for r in viol[:8]],
    "note": "bulk rows are weighted average prices and must lie inside the day's range; block-window trades are "
            "priced within +-1 % of the reference price and can fall just outside the normal-market range; rows "
            "without a daily bar are SME / other-series symbols (the price file holds EQ / BE only) and all rows "
            "after the last date of the lab's daily file"}
# hand spot checks: 10 random lab-symbol rows, compared with the day's official OHLC
random.seed(7)
lab = [r for r in rows if r["in_lab"] == "1" and cal.lookup(r["symbol"], r["date"])]
spot = []
for r in random.sample(lab, 10):
    p = cal.lookup(r["symbol"], r["date"])
    spot.append({"date": r["date"], "kind": r["kind"], "symbol": r["symbol"], "lab_symbol": r["lab_symbol"],
                 "client": r["client"], "side": r["side"], "qty": r["qty"], "price": r["price"],
                 "day_low": p["low"], "day_high": p["high"], "day_volume": p["volume"],
                 "qty_pct_of_day_volume": round(100 * r["qty"] / p["volume"], 2) if p["volume"] else None,
                 "price_inside_range": bool(p["low"] * 0.999 <= r["price"] <= p["high"] * 1.001)})
out["spot_checks_vs_official_daily_bars"] = spot
ren = collections.Counter((r["symbol"], r["lab_symbol"]) for r in rows if r["symbol"] != r["lab_symbol"])
out["renamed_symbols"] = {"pairs": len(ren), "rows": sum(ren.values()),
                          "lab_examples": [[a, b, c] for (a, b), c in ren.items() if cal and b in {x["lab_symbol"] for x in lab}][:15]}
json.dump(out, open(os.path.join(D, "data_check.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("sessions_without_any_bulk_deal",)}, indent=1)[:5000])
print("sessions without bulk:", out["sessions_without_any_bulk_deal"]["count"], out["sessions_without_any_bulk_deal"]["per_year"])
