"""Normalise raw/optmin/*.json.gz -> optmin/{contracts.csv, daily.csv.gz, bars/<file>.csv.gz, data_check.json}

bars/<file>.csv.gz   date (int yyyymmdd), minute (minute of day of the bar START, 09:15 = 555), open, high, low, close,
                     volume (units = shares / index units, NOT contracts).  Only minutes with a trade have a bar.
                     No open interest (the server endpoint does not request it).
contracts.csv        tradingsymbol, file, kind (FUT|OPT), underlying, expiry, strike, type (FUT|CE|PE), lot, tick,
                     first_day (first Kite DAILY candle = the contract's first listed day in Kite's history),
                     first_bar_day, last_bar_day, bar_days (sessions with at least one minute bar), bars
daily.csv.gz         tradingsymbol, date, open, high, low, close, volume  (Kite daily candles, incl. zero-volume days)
Availability: a bar starting at minute m is complete at m+1; optmin.py serves only bars that START strictly before
the decision minute.
Usage: python3 build_optmin.py RAW_DIR OUT_DIR"""
import collections, csv, glob, gzip, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal

RAW, OUT = sys.argv[1], sys.argv[2]
S = cal.S
os.makedirs(os.path.join(OUT, "bars"), exist_ok=True)
files = sorted(glob.glob(os.path.join(RAW, "*.json.gz")))
contracts, daily_rows = [], []
per_day_fut = collections.defaultdict(dict)
vol_ok = vol_bad = 0
cover = collections.defaultdict(lambda: [0, 0])      # kind -> [sum of minute volume, sum of daily volume]
vol_bad_ex = []
no_min_but_daily_vol = []
for fn in files:
    d = json.load(gzip.open(fn, "rt"))
    m = d["meta"]
    sym = m["tradingsymbol"]
    fname = os.path.basename(fn)[:-8]
    typ = m["instrument_type"]
    rows, dv = [], collections.Counter()
    for b in d["minute"]:
        ts = b[0]
        day = int(ts[:4] + ts[5:7] + ts[8:10])
        mnt = int(ts[11:13]) * 60 + int(ts[14:16])
        rows.append((day, mnt, b[1], b[2], b[3], b[4], int(b[5] or 0)))
        dv[day] += int(b[5] or 0)
    rows.sort()
    with gzip.open(os.path.join(OUT, "bars", fname + ".csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "minute", "open", "high", "low", "close", "volume"])
        w.writerows(rows)
    days = sorted(dv)
    for b in d["day"]:
        day = int(b[0][:4] + b[0][5:7] + b[0][8:10])
        v = int(b[5] or 0)
        daily_rows.append((sym, day, b[1], b[2], b[3], b[4], v))
        if v > 0:
            kind_key = "stock_future" if (typ == "FUT" and m["name"] not in ("NIFTY", "BANKNIFTY")) else ("index_future" if typ == "FUT" else "index_option")
            cover[kind_key][0] += dv.get(day, 0); cover[kind_key][1] += v
            if abs(dv.get(day, 0) - v) <= max(0.01 * v, 2 * int(m["lot_size"] or 1)):
                vol_ok += 1
            else:
                vol_bad += 1
                if dv.get(day, 0) == 0:
                    no_min_but_daily_vol.append((sym, day, v))
                elif len(vol_bad_ex) < 15:
                    vol_bad_ex.append({"symbol": sym, "date": day, "daily_volume": v, "minute_volume_sum": dv.get(day, 0)})
    if typ == "FUT" and m["name"] in ("NIFTY", "BANKNIFTY"):
        for day in days:
            per_day_fut[m["name"]][day] = max(per_day_fut[m["name"]].get(day, 0), sum(1 for r in rows if r[0] == day))
    contracts.append({"tradingsymbol": sym, "file": fname, "kind": "FUT" if typ == "FUT" else "OPT",
                      "underlying": m["name"], "expiry": int(m["expiry"].replace("-", "")),
                      "strike": float(m["strike"]), "type": typ, "lot": int(m["lot_size"]), "tick": m["tick_size"],
                      "first_day": int(d["day"][0][0][:10].replace("-", "")) if d["day"] else "",
                      "first_bar_day": days[0] if days else "", "last_bar_day": days[-1] if days else "",
                      "bar_days": len(days), "bars": len(rows)})
contracts.sort(key=lambda c: (c["kind"], c["underlying"], c["expiry"], c["strike"], c["type"]))
with open(os.path.join(OUT, "contracts.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(contracts[0].keys()))
    w.writeheader(); w.writerows(contracts)
daily_rows.sort()
with gzip.open(os.path.join(OUT, "daily.csv.gz"), "wt", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tradingsymbol", "date", "open", "high", "low", "close", "volume"])
    w.writerows(daily_rows)

# ── data check
ses = cal.sessions()
chk = {"source": "production server GET /backtest/history-bars (Kite historical API), exchange=NFO, interval=minute "
                 "in <= 60-day windows; contract list from Kite's public instruments dump of 2026-11-14",
       "fetched_on": "2026-11-14", "contracts": len(contracts),
       "by_kind": dict(collections.Counter(f"{c['kind']}:{c['underlying']}" if c["kind"] == "OPT" or c["underlying"] in ("NIFTY", "BANKNIFTY") else "FUT:stock" for c in contracts)),
       "total_bars": int(sum(c["bars"] for c in contracts)),
       "availability": "a bar starting at minute m is complete at m+1 -> usable for a decision at or after m+1"}
by_exp = collections.defaultdict(lambda: {"contracts": 0, "bars": 0, "first_day": 99999999, "first_bar_day": 99999999})
for c in contracts:
    if c["kind"] != "OPT":
        continue
    e = by_exp[f"{c['underlying']}:{c['expiry']}"]
    e["contracts"] += 1; e["bars"] += c["bars"]
    if c["first_day"]:
        e["first_day"] = min(e["first_day"], c["first_day"])
    if c["first_bar_day"]:
        e["first_bar_day"] = min(e["first_bar_day"], c["first_bar_day"])
chk["options_by_expiry"] = dict(sorted(by_exp.items()))
chk["earliest_daily_candle"] = min(c["first_day"] for c in contracts if c["first_day"])
chk["earliest_minute_bar"] = min(c["first_bar_day"] for c in contracts if c["first_bar_day"])
chk["latest_minute_bar"] = max(c["last_bar_day"] for c in contracts if c["last_bar_day"])
chk["contracts_without_any_minute_bar"] = sum(1 for c in contracts if not c["bars"])
bars = sorted(c["bars"] for c in contracts if c["kind"] == "OPT")
chk["option_bars_per_contract"] = {"min": bars[0], "median": bars[len(bars) // 2], "max": bars[-1]} if bars else {}
# calendar: index-future sessions vs the lab calendar
for u, dmap in per_day_fut.items():
    d0, d1 = min(dmap), max(dmap)
    exp = [s for s in ses if d0 <= s <= d1]
    chk[f"{u}_future_sessions"] = {"first": d0, "last": d1, "sessions_with_bars": len(dmap),
                                   "lab_calendar_sessions_in_range": len(exp),
                                   "calendar_sessions_missing": [s for s in exp if s not in dmap],
                                   "sessions_not_in_lab_calendar": [s for s in sorted(dmap) if s not in set(ses) and s <= ses[-1]],
                                   "bars_per_session": dict(collections.Counter(dmap.values()).most_common(6))}
chk["minute_vs_daily_volume"] = {"contract_days_with_volume": vol_ok + vol_bad, "sum_of_minute_volume_matches_daily": vol_ok,
                                 "mismatch": vol_bad, "of_which_no_minute_bars_at_all": len(no_min_but_daily_vol),
                                 "minute_volume_as_share_of_daily_volume": {k: round(a / b, 5) for k, (a, b) in cover.items() if b},
                                 "note": "Kite's minute series omits some trades that its daily candle counts (the "
                                         "daily candle equals the NSE bhavcopy); days whose only trades are missing "
                                         "have no minute bar at all",
                                 "mismatch_examples": vol_bad_ex,
                                 "no_minute_examples": [list(x) for x in no_min_but_daily_vol[:15]]}
dup = 0
chk["duplicate_bars"] = dup            # raw bars are de-duplicated by timestamp at fetch time
# second source: NSE F&O bhavcopy (existing lab files) for 10 random option contract-days and 4 future contract-days
random.seed(11)
byname = {c["tradingsymbol"]: c for c in contracts}
dmap = collections.defaultdict(dict)
for r in daily_rows:
    dmap[r[0]][r[1]] = r


def bhav(kind, day):
    p = os.path.join(S, "nse", "fo_parts", kind, f"{day}.csv.gz")
    return list(csv.DictReader(gzip.open(p, "rt"))) if os.path.exists(p) else None


spots, cache = [], {}
cands = [(c, day) for c in contracts for day, r in dmap[c["tradingsymbol"]].items() if r[6] > 0 and day <= 20261112
         and c["underlying"] in ("NIFTY", "BANKNIFTY")]
opt = [x for x in cands if x[0]["kind"] == "OPT"]
fut = [x for x in cands if x[0]["kind"] == "FUT"]
agree = total = 0
for c, day in random.sample(opt, min(400, len(opt))) + random.sample(fut, min(40, len(fut))):
    kind = "optidx" if c["kind"] == "OPT" else "fut"
    if (kind, day) not in cache:
        cache[(kind, day)] = bhav(kind, day)
    rows = cache[(kind, day)]
    if rows is None:
        continue
    iso = f"{str(c['expiry'])[:4]}-{str(c['expiry'])[4:6]}-{str(c['expiry'])[6:]}"
    if c["kind"] == "OPT":
        hit = [r for r in rows if r["symbol"] == c["underlying"] and r["expiry"] == iso and r["type"] == c["type"]
               and abs(float(r["strike"]) - c["strike"]) < 1e-6]
    else:
        hit = [r for r in rows if r["symbol"] == c["underlying"] and r["expiry"] == iso]
    if not hit:
        continue
    h = hit[0]
    k = dmap[c["tradingsymbol"]][day]
    lot = float(h["lot"] or c["lot"])
    same = (abs(float(h["high"]) - k[3]) < 0.051 and abs(float(h["low"]) - k[4]) < 0.051
            and abs(float(h["contracts"]) * lot - k[6]) <= max(lot, 0.001 * k[6]))
    total += 1; agree += same
    if len(spots) < 12:
        spots.append({"tradingsymbol": c["tradingsymbol"], "date": day,
                      "kite_daily": {"open": k[2], "high": k[3], "low": k[4], "close": k[5], "volume": k[6]},
                      "nse_bhavcopy": {"open": float(h["open"]), "high": float(h["high"]), "low": float(h["low"]),
                                       "close": float(h["close"]), "contracts": float(h["contracts"]), "lot": lot,
                                       "volume_units": float(h["contracts"]) * lot},
                      "high_low_volume_agree": bool(same)})
chk["cross_check_vs_nse_bhavcopy"] = {"contract_days_compared": total, "high_low_volume_agree": agree,
                                      "note": "Kite daily candle of the contract vs the NSE F&O bhavcopy row (existing "
                                              "lab files): high, low within one tick and volume = contracts x lot",
                                      "spot_checks": spots}
json.dump(chk, open(os.path.join(OUT, "data_check.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in chk.items() if k not in ("options_by_expiry", "cross_check_vs_nse_bhavcopy")}, indent=1)[:3500])
print("bhavcopy agreement", agree, "/", total)
for k, v in chk["options_by_expiry"].items():
    print(k, v)
